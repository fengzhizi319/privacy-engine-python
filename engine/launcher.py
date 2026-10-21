"""多进程启动器：fork N 个 worker 共享同一端口（SO_REUSEPORT）。

中文说明：
通过 ``SO_REUSEPORT`` 让多个 worker 进程绑定同一 IP:Port，由操作系统内核
在 TCP 层均匀分发连接，无需外部负载均衡器。每个 worker 独立运行一套
REST + gRPC 双协议服务，充分利用多核 CPU 绕过 GIL 限制。

English Description:
Multi-process launcher using ``SO_REUSEPORT`` to share a single port across
N worker processes. The OS kernel distributes TCP connections evenly at the
kernel level, eliminating the need for an external load balancer. Each worker
runs an independent REST + gRPC server pair, fully utilizing multi-core CPUs
to bypass the Python GIL limitation.

环境变量 / Environment Variables:
    PRIVACY_WORKERS: worker 进程数，默认 min(cpu_count, 8)。
    PRIVACY_GRPC_MAX_WORKERS: 每个 worker 的 gRPC 线程池大小，默认 64。
    PRIVACY_REST_HOST / PRIVACY_REST_PORT: REST 监听地址与端口。
    PRIVACY_GRPC_HOST / PRIVACY_GRPC_PORT: gRPC 监听地址与端口。

使用示例 / Usage Example::

    # 命令行启动（推荐）
    python -m engine.launcher --workers 4

    # 代码调用
    from engine.launcher import launch
    launch(num_workers=4, host_rest="0.0.0.0", port_rest=8079,
           host_grpc="0.0.0.0", port_grpc=50051)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import socket
import sys
import threading
import time

from .observability.logging_config import get_logger

logger = get_logger(__name__)


def create_reuse_port_socket(host: str, port: int) -> socket.socket:
    """创建支持 SO_REUSEPORT 的 TCP socket。

    多个进程可以各自创建独立的 socket 并绑定到同一端口，由内核在
    TCP 连接建立时均匀分发。

    Args:
        host: 监听地址（如 "0.0.0.0"）。
        port: 监听端口号。

    Returns:
        已绑定但未 listen 的 socket 对象。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # SO_REUSEPORT 内核级负载均衡仅 Linux 原生支持；
    # macOS 虽定义该常量但语义不同（无内核连接分发），Windows 未定义。
    if sys.platform == "linux" and hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    return sock


def _worker_entry(
    worker_id: int,
    host_rest: str,
    port_rest: int,
    host_grpc: str,
    port_grpc: int,
    grpc_max_workers: int,
    host_ops: str,
    port_ops: int,
) -> None:
    """每个 worker 进程的入口函数。

    独立创建 SO_REUSEPORT socket 并启动 REST + gRPC 双协议服务与运维口。

    Args:
        worker_id: 当前 worker 编号（用于进程标题区分）。
        host_rest: REST 监听地址。
        port_rest: REST 监听端口。
        host_grpc: gRPC 监听地址。
        port_grpc: gRPC 监听端口。
        grpc_max_workers: gRPC 线程池大小。
        host_ops: 运维口监听地址。
        port_ops: 运维口监听端口。
    """
    # 设置进程标题便于运维识别（可选依赖 setproctitle）
    try:
        import setproctitle
        setproctitle.setproctitle(f"privacy-engine-worker-{worker_id}")
    except ImportError:
        # setproctitle 不可用时通过 os.setproctitle 或跳过
        pass

    logger.info(
        "worker_started",
        extra={
            "worker_id": worker_id,
            "pid": os.getpid(),
            "host_rest": host_rest,
            "port_rest": port_rest,
            "host_grpc": host_grpc,
            "port_grpc": port_grpc,
            "host_ops": host_ops,
            "port_ops": port_ops,
        },
    )

    # 延迟导入避免 fork 后重复初始化问题
    from .grpc_server import serve as grpc_serve
    from .main import app
    from .ops_port import (
        create_ops_asgi_app,
        load_ops_settings,
        ops_uvicorn_ssl_kwargs,
    )
    from .security.config import get_security_settings
    from .security.tls import uvicorn_ssl_kwargs

    import uvicorn

    # 复用 server.py 的 uvloop/httptools 自动检测结果（多进程模式同样生效）
    from .server import _UVICORN_LOOP_KWARG

    # 1. 创建 SO_REUSEPORT socket 给 REST
    rest_sock = create_reuse_port_socket(host_rest, port_rest)
    rest_sock.listen(1024)  # backlog 设为 1024 以应对突发连接

    # 2. 配置 uvicorn 使用预创建的 socket（含高并发优化：
    #    uvloop/httptools、并发连接限制、keep-alive 超时）
    security = get_security_settings()
    ssl_kwargs = uvicorn_ssl_kwargs(security)
    config = uvicorn.Config(
        app,
        sock=rest_sock,
        log_level="info",
        limit_concurrency=int(os.environ.get("PRIVACY_LIMIT_CONCURRENCY", "10000")),
        limit_max_requests=int(os.environ.get("PRIVACY_LIMIT_MAX_REQUESTS", "100000")),
        timeout_keep_alive=int(os.environ.get("PRIVACY_TIMEOUT_KEEP_ALIVE", "30")),
        timeout_graceful_shutdown=int(os.environ.get("PRIVACY_TIMEOUT_GRACEFUL_SHUTDOWN", "10")),
        **_UVICORN_LOOP_KWARG,
        **ssl_kwargs,
    )
    rest_server = uvicorn.Server(config)

    # 3. 在非守护线程中启动 REST 服务
    rest_thread = threading.Thread(
        target=rest_server.run,
        name=f"uvicorn-worker-{worker_id}",
        daemon=False,
    )
    rest_thread.start()

    # 3.5 独立运维口（SO_REUSEPORT 共享端口，仅放行探针/指标/诊断路径）。
    # lifespan="off"：主应用 lifespan 只由业务口执行一次。
    ops_settings = load_ops_settings(
        host_rest, port_rest, security, ops_host=host_ops, ops_port=port_ops
    )
    ops_sock = create_reuse_port_socket(ops_settings.host, ops_settings.port)
    ops_sock.listen(1024)
    ops_config = uvicorn.Config(
        create_ops_asgi_app(app),
        sock=ops_sock,
        log_level="info",
        lifespan="off",
        timeout_keep_alive=int(os.environ.get("PRIVACY_TIMEOUT_KEEP_ALIVE", "30")),
        timeout_graceful_shutdown=int(os.environ.get("PRIVACY_TIMEOUT_GRACEFUL_SHUTDOWN", "10")),
        **_UVICORN_LOOP_KWARG,
        **ops_uvicorn_ssl_kwargs(ops_settings, security),
    )
    ops_server = uvicorn.Server(ops_config)
    ops_thread = threading.Thread(
        target=ops_server.run,
        name=f"uvicorn-ops-worker-{worker_id}",
        daemon=False,
    )
    ops_thread.start()

    # 4. 启动 gRPC 服务（每个 worker 独立线程池）
    grpc_server = grpc_serve(
        host=host_grpc,
        port=port_grpc,
        max_workers=grpc_max_workers,
        wait_for_termination=False,
    )

    # 5. 注册信号处理：优雅退出
    shutdown_event = threading.Event()

    def handle_shutdown(signum: int, frame) -> None:
        logger.info("worker_shutdown_signal", extra={"worker_id": worker_id, "signal": signum})
        shutdown_event.set()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # 6. 主线程等待终止信号
    try:
        while not shutdown_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    # 7. 优雅退出
    logger.info("worker_stopping", extra={"worker_id": worker_id})
    grpc_server.stop(grace=5)
    rest_server.should_exit = True
    ops_server.should_exit = True
    rest_thread.join(timeout=10)
    ops_thread.join(timeout=10)
    logger.info("worker_stopped", extra={"worker_id": worker_id})


def _monitor_workers(
    workers: list[mp.Process],
    respawn: bool = True,
    spawn_args: tuple | None = None,
) -> threading.Event:
    """监控 worker 进程：意外退出时记录日志并可选自动拉起新 worker。

    在主进程中运行，注册 SIGTERM/SIGINT 信号处理，返回 shutdown_event。
    收到信号后停止监控并退出循环。

    Args:
        workers: 待监控的 worker 进程列表（会原地替换意外退出的进程）。
        respawn: 意外退出时是否自动拉起新 worker（默认 True）。
        spawn_args: 拉起新 worker 的 ``_worker_entry`` 参数元组，
            ``(worker_id, host_rest, port_rest, host_grpc, port_grpc, grpc_max_workers)``。
            为 None 时不支持 respawn（仅记录日志）。

    Returns:
        threading.Event：被设置表示收到终止信号。
    """
    shutdown_event = threading.Event()

    def handle_shutdown(signum: int, frame) -> None:
        logger.info("launcher_shutdown_signal", extra={"signal": signum})
        shutdown_event.set()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # fork 仅 Linux 原生支持（COW 共享内存）；
    # macOS 自 Python 3.8+ 默认 spawn（fork 在 macOS 已弃用，可能崩溃）；
    # Windows 仅支持 spawn。
    ctx = mp.get_context("fork" if sys.platform == "linux" else "spawn")
    try:
        while not shutdown_event.is_set():
            for i, p in enumerate(workers):
                if not p.is_alive() and not shutdown_event.is_set():
                    logger.warning(
                        "worker_unexpected_exit",
                        extra={"worker_id": i, "pid": p.pid, "exitcode": p.exitcode},
                    )
                    if respawn and spawn_args is not None:
                        # 用同一 worker_id 拉起新 worker 顶替
                        args = list(spawn_args)
                        args[0] = i
                        new_p = ctx.Process(
                            target=_worker_entry,
                            args=tuple(args),
                            name=f"privacy-engine-worker-{i}",
                            daemon=False,
                        )
                        new_p.start()
                        workers[i] = new_p
                        logger.info("worker_respawned", extra={"worker_id": i, "pid": new_p.pid})
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    return shutdown_event


def _terminate_workers(workers: list[mp.Process]) -> None:
    """优雅关闭所有 worker：先 terminate，超时后 kill。"""
    logger.info("launcher_terminating_workers", extra={"count": len(workers)})
    for p in workers:
        if p.is_alive():
            p.terminate()

    for p in workers:
        p.join(timeout=15)
        if p.is_alive():
            logger.warning("worker_force_kill", extra={"pid": p.pid})
            p.kill()


def launch(
    num_workers: int | None = None,
    host_rest: str = "0.0.0.0",
    port_rest: int = 8079,
    host_grpc: str = "0.0.0.0",
    port_grpc: int = 50051,
    grpc_max_workers: int | None = None,
    host_ops: str | None = None,
    port_ops: int | None = None,
) -> None:
    """启动多进程 worker 池，所有 worker 共享同一端口。

    通过 ``SO_REUSEPORT`` 实现内核级连接分发，无需外部负载均衡器。
    主进程负责监控 worker 并在收到 SIGTERM/SIGINT 时优雅关闭所有子进程。

    Args:
        num_workers: worker 进程数，默认 ``min(os.cpu_count(), 8)``。
        host_rest: REST 监听地址，默认 0.0.0.0。
        port_rest: REST 监听端口，默认 8079。
        host_grpc: gRPC 监听地址，默认 0.0.0.0。
        port_grpc: gRPC 监听端口，默认 50051。
        grpc_max_workers: 每个 worker 的 gRPC 线程池大小，
            默认 ``min(64, max(16, os.cpu_count() * 4))``。
        host_ops: 运维口监听地址，默认继承 REST 地址或 ``PRIVACY_OPS_HOST``。
        port_ops: 运维口监听端口，默认 REST 端口 +1000 或 ``PRIVACY_OPS_PORT``。
    """
    # 参数默认值
    if num_workers is None:
        num_workers = int(os.environ.get("PRIVACY_WORKERS", min(os.cpu_count() or 4, 8)))
    if grpc_max_workers is None:
        grpc_max_workers = int(
            os.environ.get("PRIVACY_GRPC_MAX_WORKERS", min(64, max(16, (os.cpu_count() or 4) * 4)))
        )

    # 在主进程解析并校验运维口配置：证书缺失/协议栈非法在 fork 之前拒绝启动。
    from .ops_port import load_ops_settings
    from .security.config import get_security_settings

    ops_settings = load_ops_settings(
        host_rest, port_rest, get_security_settings(), ops_host=host_ops, ops_port=port_ops
    )
    host_ops, port_ops = ops_settings.host, ops_settings.port

    logger.info(
        "launcher_starting",
        extra={
            "num_workers": num_workers,
            "host_rest": host_rest,
            "port_rest": port_rest,
            "host_grpc": host_grpc,
            "port_grpc": port_grpc,
            "host_ops": host_ops,
            "port_ops": port_ops,
            "grpc_max_workers": grpc_max_workers,
        },
    )

    # 多进程启动方式按平台分支：
    # Linux → fork（利用 COW 共享内存，性能最优）
    # macOS → spawn（fork 在 macOS 已弃用，Python 3.8+ 默认 spawn）
    # Windows → spawn（仅支持 spawn）
    ctx = mp.get_context("fork" if sys.platform == "linux" else "spawn")
    workers: list[mp.Process] = []
    spawn_args = (
        0, host_rest, port_rest, host_grpc, port_grpc, grpc_max_workers, host_ops, port_ops
    )

    for i in range(num_workers):
        p = ctx.Process(
            target=_worker_entry,
            args=(
                i, host_rest, port_rest, host_grpc, port_grpc, grpc_max_workers, host_ops, port_ops
            ),
            name=f"privacy-engine-worker-{i}",
            daemon=False,
        )
        p.start()
        workers.append(p)
        logger.info("worker_launched", extra={"worker_id": i, "pid": p.pid})

    # 监控 worker 进程：意外退出时记录日志并自动拉起新 worker
    _monitor_workers(workers, respawn=True, spawn_args=spawn_args)

    # 优雅关闭所有 worker
    _terminate_workers(workers)

    logger.info("launcher_all_workers_stopped")
    sys.exit(0)


def main() -> None:
    """命令行入口：解析参数并启动多进程 worker 池。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="engine.launcher",
        description="PrivShield multi-process launcher with SO_REUSEPORT.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: min(cpu_count, 8) or PRIVACY_WORKERS).",
    )
    parser.add_argument(
        "--rest-host",
        default=os.environ.get("PRIVACY_REST_HOST", "0.0.0.0"),
        help="REST server host (default: 0.0.0.0 or PRIVACY_REST_HOST).",
    )
    parser.add_argument(
        "--rest-port",
        type=int,
        default=int(os.environ.get("PRIVACY_REST_PORT", "8079")),
        help="REST server port (default: 8079 or PRIVACY_REST_PORT).",
    )
    parser.add_argument(
        "--grpc-host",
        default=os.environ.get("PRIVACY_GRPC_HOST", "0.0.0.0"),
        help="gRPC server host (default: 0.0.0.0 or PRIVACY_GRPC_HOST).",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=int(os.environ.get("PRIVACY_GRPC_PORT", "50051")),
        help="gRPC server port (default: 50051 or PRIVACY_GRPC_PORT).",
    )
    parser.add_argument(
        "--grpc-max-workers",
        type=int,
        default=None,
        help="gRPC thread pool size per worker (default: min(64, cpu_count*4) or PRIVACY_GRPC_MAX_WORKERS).",
    )
    parser.add_argument(
        "--ops-host",
        default=os.environ.get("PRIVACY_OPS_HOST", os.environ.get("PRIVACY_REST_HOST", "0.0.0.0")),
        help="Ops listener host (default: REST host or PRIVACY_OPS_HOST).",
    )
    parser.add_argument(
        "--ops-port",
        type=int,
        default=None,
        help="Ops listener port (default: REST port + 1000 or PRIVACY_OPS_PORT).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help="Enable fork-after-warmup mode: preload ML models before forking workers to share model weights via Copy-on-Write.",
    )

    args = parser.parse_args()

    if args.warmup:
        launch_with_warmup(
            num_workers=args.workers,
            host_rest=args.rest_host,
            port_rest=args.rest_port,
            host_grpc=args.grpc_host,
            port_grpc=args.grpc_port,
            grpc_max_workers=args.grpc_max_workers,
            host_ops=args.ops_host,
            port_ops=args.ops_port,
        )
    else:
        launch(
            num_workers=args.workers,
            host_rest=args.rest_host,
            port_rest=args.rest_port,
            host_grpc=args.grpc_host,
            port_grpc=args.grpc_port,
            grpc_max_workers=args.grpc_max_workers,
            host_ops=args.ops_host,
            port_ops=args.ops_port,
        )


def launch_with_warmup(
    num_workers: int | None = None,
    host_rest: str = "0.0.0.0",
    port_rest: int = 8079,
    host_grpc: str = "0.0.0.0",
    port_grpc: int = 50051,
    grpc_max_workers: int | None = None,
    host_ops: str | None = None,
    port_ops: int | None = None,
) -> None:
    """fork-after-warmup 模式：预热 ML 模型后再 fork worker。

    中文说明：
    在主进程中预先加载 ML 模型（NER、LLM），然后 fork 出 N 个 worker。
    利用 Linux Copy-on-Write (COW) 机制，子进程继承主进程的只读内存页，
    从而共享模型权重，避免 N 份模型内存翻倍。

    注意：仅适用于 fork 启动方式（Linux 默认）。macOS 的 spawn 模式不支持 COW。

    English Description:
    Fork-after-warmup mode: preload ML models in the main process before forking
    N workers. Leverages Linux Copy-on-Write (COW) to share read-only memory
    pages (model weights) across workers, avoiding N-fold memory duplication.

    Note: Only works with fork start method (Linux default). macOS spawn mode
    does not support COW.

    Args:
        num_workers: worker 进程数，默认 ``min(os.cpu_count(), 8)``。
        host_rest: REST 监听地址。
        port_rest: REST 监听端口。
        host_grpc: gRPC 监听地址。
        port_grpc: gRPC 监听端口。
        grpc_max_workers: gRPC 线程池大小。
        host_ops: 运维口监听地址。
        port_ops: 运维口监听端口。
    """
    if num_workers is None:
        num_workers = int(os.environ.get("PRIVACY_WORKERS", min(os.cpu_count() or 4, 8)))
    if grpc_max_workers is None:
        grpc_max_workers = int(
            os.environ.get("PRIVACY_GRPC_MAX_WORKERS", min(64, max(16, (os.cpu_count() or 4) * 4)))
        )

    # 与 launch() 相同：在 fork 之前解析并校验运维口配置，坏配置不进入 fork。
    from .ops_port import load_ops_settings
    from .security.config import get_security_settings

    ops_settings = load_ops_settings(
        host_rest, port_rest, get_security_settings(), ops_host=host_ops, ops_port=port_ops
    )
    host_ops, port_ops = ops_settings.host, ops_settings.port

    logger.info(
        "warmup_launcher_starting",
        extra={
            "num_workers": num_workers,
            "mode": "fork_after_warmup",
        },
    )

    # 1. 主进程预热 ML 模型（可选依赖，失败不影响启动）
    #    加载完成后注册到 dynclassification.service 的预加载注册表，
    #    fork 后 worker 首次使用 NER/LLM 时直接复用（COW 共享模型只读页，
    #    避免 N 份模型内存翻倍）。
    try:
        from .dynclassification.ner_adapter import NerAdapter
        from .dynclassification.service import register_preloaded_adapter

        ner = NerAdapter()
        ner._lazy_init()  # 触发模型加载
        register_preloaded_adapter("ner", ner)
        logger.info("warmup_ner_model_loaded")
    except Exception as e:
        logger.warning("warmup_ner_skipped", extra={"reason": str(e)})

    try:
        from .dynclassification.llm_adapter import LlmAdapter
        from .dynclassification.service import register_preloaded_adapter

        llm = LlmAdapter()
        llm._lazy_init()  # 触发模型加载
        register_preloaded_adapter("llm", llm)
        logger.info("warmup_llm_model_loaded")
    except Exception as e:
        logger.warning("warmup_llm_skipped", extra={"reason": str(e)})

    # 2. fork/spawn worker（利用 COW 共享模型只读页）
    # 多进程启动方式按平台分支：Linux → fork；macOS/Windows → spawn
    ctx = mp.get_context("fork" if sys.platform == "linux" else "spawn")
    workers: list[mp.Process] = []
    spawn_args = (
        0, host_rest, port_rest, host_grpc, port_grpc, grpc_max_workers, host_ops, port_ops
    )

    for i in range(num_workers):
        p = ctx.Process(
            target=_worker_entry,
            args=(
                i, host_rest, port_rest, host_grpc, port_grpc, grpc_max_workers, host_ops, port_ops
            ),
            name=f"privacy-engine-worker-{i}",
            daemon=False,
        )
        p.start()
        workers.append(p)
        logger.info("warmup_worker_forked", extra={"worker_id": i, "pid": p.pid})

    # 3. 主进程监控 worker（与 launch() 相同逻辑：意外退出自动拉起）
    _monitor_workers(workers, respawn=True, spawn_args=spawn_args)

    # 4. 优雅关闭
    _terminate_workers(workers)

    logger.info("warmup_launcher_stopped")
    sys.exit(0)


if __name__ == "__main__":
    main()
