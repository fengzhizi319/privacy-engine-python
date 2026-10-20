"""多进程启动器（launcher.py）单元测试。

覆盖要点：
- create_reuse_port_socket 的 SO_REUSEPORT 绑定能力（两个 socket 绑定同一端口）；
- _monitor_workers 的 worker 意外退出自动拉起逻辑（回归：曾因缺失 `import threading`
  导致 launch() 一调用即 NameError 崩溃，见审计报告 §8.2）；
- _terminate_workers 的先 terminate 后 kill 两段式终止；
- main()/launch() 的参数解析与默认 worker 数计算。
"""

from __future__ import annotations

import signal
import socket
import sys

import pytest

from engine import launcher


class _FakeProcess:
    """模拟 multiprocessing.Process 的最小接口。"""

    def __init__(self, alive: bool = True, target=None, args=(), name: str = "", daemon: bool = False):
        self._alive = alive
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.pid = 99999
        self.exitcode = 1 if not alive else None
        self.terminated = False
        self.killed = False
        self.started = False
        self.join_calls: list[float | None] = []

    def start(self) -> None:
        self.started = True
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestCreateReusePortSocket:
    def test_binds_requested_address(self):
        port = _free_port()
        sock = launcher.create_reuse_port_socket("127.0.0.1", port)
        try:
            assert sock.getsockname()[1] == port
        finally:
            sock.close()

    @pytest.mark.skipif(not hasattr(socket, "SO_REUSEPORT"), reason="平台不支持 SO_REUSEPORT")
    def test_two_sockets_share_same_port(self):
        """SO_REUSEPORT 的核心语义：两个独立 socket 可绑定同一 IP:Port。"""
        port = _free_port()
        s1 = launcher.create_reuse_port_socket("127.0.0.1", port)
        s2 = launcher.create_reuse_port_socket("127.0.0.1", port)
        try:
            assert s1.getsockname()[1] == s2.getsockname()[1] == port
            assert s1.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 1
        finally:
            s1.close()
            s2.close()


class TestMonitorWorkers:
    def test_respawns_dead_worker(self, monkeypatch):
        """worker 意外退出时用同一 worker_id 自动拉起新进程。"""
        dead = _FakeProcess(alive=False)
        workers = [dead]
        spawn_args = (0, "127.0.0.1", 8079, "127.0.0.1", 50051, 64)

        created: list[_FakeProcess] = []

        class _FakeCtx:
            def Process(self, target=None, args=(), name="", daemon=False):
                p = _FakeProcess(target=target, args=args, name=name, daemon=daemon)
                created.append(p)
                return p

        monkeypatch.setattr(launcher.mp, "get_context", lambda method: _FakeCtx())

        # 第一次循环结束后用 KeyboardInterrupt 跳出监控循环
        monkeypatch.setattr(
            launcher.time,
            "sleep",
            lambda _s: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

        # signal 处理器替换需恢复原样，避免污染 pytest 进程
        old_term, old_int = signal.signal(signal.SIGTERM, signal.SIG_DFL), None
        try:
            event = launcher._monitor_workers(workers, respawn=True, spawn_args=spawn_args)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        assert len(created) == 1, "应对死掉的 worker 拉起恰好一个新进程"
        assert created[0].started, "新 worker 应被 start()"
        assert created[0].args[0] == 0, "新 worker 应沿用原 worker_id"
        assert workers[0] is created[0], "workers 列表应原地替换"
        assert isinstance(event, launcher.threading.Event)

    def test_no_respawn_when_disabled(self, monkeypatch):
        workers = [_FakeProcess(alive=False)]
        monkeypatch.setattr(
            launcher.time,
            "sleep",
            lambda _s: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        old_term = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            launcher._monitor_workers(workers, respawn=False, spawn_args=None)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        assert workers[0].started is False


class TestTerminateWorkers:
    def test_terminate_then_kill_when_stubborn(self):
        stubborn = _FakeProcess(alive=True)  # terminate 后仍存活 → 需要 kill
        stubborn.terminate = lambda: None  # terminate 不改变存活状态

        def _terminate_side_effect():
            stubborn.terminated = True

        stubborn.terminate = _terminate_side_effect
        launcher._terminate_workers([stubborn])
        assert stubborn.terminated, "应先尝试 terminate"
        assert stubborn.killed, "join 超时后仍存活应 kill"

    def test_graceful_worker_not_killed(self):
        graceful = _FakeProcess(alive=True)

        def _terminate_then_exit():
            graceful.terminated = True
            graceful._alive = False

        graceful.terminate = _terminate_then_exit
        launcher._terminate_workers([graceful])
        assert graceful.terminated
        assert not graceful.killed, "能优雅退出的 worker 不应被 kill"


class TestMainArgParsing:
    def test_main_calls_launch_with_parsed_args(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(launcher, "launch", lambda **kw: captured.update(kw))
        monkeypatch.setattr(
            sys,
            "argv",
            ["launcher", "--workers", "3", "--rest-port", "18079", "--grpc-port", "15051"],
        )
        launcher.main()
        assert captured["num_workers"] == 3
        assert captured["port_rest"] == 18079
        assert captured["port_grpc"] == 15051

    def test_main_warmup_routes_to_warmup_launcher(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(launcher, "launch_with_warmup", lambda **kw: captured.update(kw))
        monkeypatch.setattr(sys, "argv", ["launcher", "--workers", "2", "--warmup"])
        launcher.main()
        assert captured["num_workers"] == 2


class TestLaunchDefaults:
    def test_default_worker_count_and_system_exit(self, monkeypatch):
        """launch() 默认 worker 数为 min(cpu_count, 8)，结束时 SystemExit(0)。"""
        created: list[_FakeProcess] = []

        class _FakeCtx:
            def Process(self, target=None, args=(), name="", daemon=False):
                p = _FakeProcess(target=target, args=args, name=name, daemon=daemon)
                created.append(p)
                return p

        monkeypatch.setattr(launcher.mp, "get_context", lambda method: _FakeCtx())
        monkeypatch.setattr(launcher, "_monitor_workers", lambda *a, **kw: None)
        monkeypatch.setattr(launcher, "_terminate_workers", lambda *a, **kw: None)
        monkeypatch.delenv("PRIVACY_WORKERS", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            launcher.launch(num_workers=None)
        assert exc_info.value.code == 0

        import os

        expected = min(os.cpu_count() or 4, 8)
        assert len(created) == expected
        # spawn_args 元组：(worker_id, host_rest, port_rest, host_grpc, port_grpc, grpc_max_workers)
        assert created[0].args[2] == 8079
        assert created[0].args[4] == 50051
