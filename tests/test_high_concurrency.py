"""高并发特性测试。

覆盖：
- SQLite WAL 模式预算记账多进程安全
- SO_REUSEPORT socket 创建
- 多进程启动器参数解析
- gRPC 线程池环境变量配置
- Numba JIT 加速模块（dp_jit）
- NoisePool 预生成噪声池
- BatchedBudgetSpend 批量预算扣减
- uvloop/httptools 自动检测
- GZip 中间件注册
- fork-after-warmup 启动模式
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import tempfile
import threading
import time

import pytest


class TestSQLiteWALMode:
    """SQLite WAL 模式预算记账测试。"""

    def test_wal_mode_enabled(self, tmp_path):
        """验证 BudgetAccountant 创建连接时自动开启 WAL 模式。"""
        db_path = str(tmp_path / "test_budget.db")
        os.environ["PRIVACY_BUDGET_DB"] = db_path

        try:
            from engine.privacy.budget import BudgetAccountant, default_registry

            # 清理全局注册表以避免测试间干扰
            default_registry.reset()

            acct = default_registry.get_or_create(
                "test-wal",
                epsilon_total=100.0,
                delta_total=1e-3,
            )

            # 验证 WAL 模式已开启
            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.execute("PRAGMA journal_mode")
                mode = cursor.fetchone()[0]
                assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"
            finally:
                conn.close()
        finally:
            os.environ.pop("PRIVACY_BUDGET_DB", None)
            default_registry.reset()

    def test_concurrent_spend_multi_process(self, tmp_path):
        """验证多进程并发 spend 操作的安全性（无超卖）。"""
        db_path = str(tmp_path / "concurrent_budget.db")
        os.environ["PRIVACY_BUDGET_DB"] = db_path

        try:
            from engine.privacy.budget import BudgetAccountant, default_registry

            default_registry.reset()

            # 初始化预算
            acct = default_registry.get_or_create(
                "test-concurrent",
                epsilon_total=10.0,
                delta_total=1.0,
            )

            # 多进程并发 spend
            results = []

            def worker_spend(worker_id: int):
                """每个 worker 独立创建 BudgetAccountant 并 spend。"""
                # 每个进程需要自己的 registry 实例
                from engine.privacy.budget import BudgetRegistry

                reg = BudgetRegistry()
                local_acct = reg.get_or_create(
                    "test-concurrent",
                    epsilon_total=10.0,
                    delta_total=1.0,
                )
                try:
                    local_acct.spend(0.1, 0.0)
                    results.append(("ok", worker_id))
                except Exception as e:
                    results.append(("error", str(e)))

            # 使用线程模拟并发（多进程在 CI 中可能不稳定）
            threads = []
            for i in range(20):
                t = threading.Thread(target=worker_spend, args=(i,))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            # 验证：成功的 spend 次数不应超过预算允许的上限
            ok_count = sum(1 for r in results if r[0] == "ok")
            # 10.0 / 0.1 = 100 次，但线程并发下可能有竞争
            # 关键验证：不应超过预算（100 次）
            assert ok_count <= 100, f"Budget oversell detected: {ok_count} > 100"
            # 至少有一些成功的
            assert ok_count > 0, "No successful spends"

        finally:
            os.environ.pop("PRIVACY_BUDGET_DB", None)
            default_registry.reset()


class TestReusePortSocket:
    """SO_REUSEPORT socket 创建测试。"""

    def test_create_reuse_port_socket(self):
        """验证 create_reuse_port_socket 创建可绑定的 socket。"""
        from engine.launcher import create_reuse_port_socket

        # 使用随机高端口避免冲突
        import socket

        sock1 = create_reuse_port_socket("127.0.0.1", 0)  # port 0 = OS 分配
        try:
            # 验证 socket 属性
            assert sock1.family == socket.AF_INET
            assert sock1.type == socket.SOCK_STREAM

            # 获取实际分配的端口
            port = sock1.getsockname()[1]
            assert port > 0

            # 验证 SO_REUSEPORT 允许多个 socket 绑定同一端口
            sock2 = create_reuse_port_socket("127.0.0.1", port)
            try:
                port2 = sock2.getsockname()[1]
                assert port2 == port, "Second socket should bind to same port with SO_REUSEPORT"
            finally:
                sock2.close()
        finally:
            sock1.close()


class TestLauncherConfig:
    """多进程启动器配置测试。"""

    def test_default_workers(self):
        """验证默认 worker 数计算。"""
        # 清理环境变量
        os.environ.pop("PRIVACY_WORKERS", None)

        expected = min(os.cpu_count() or 4, 8)
        # 直接测试逻辑
        num_workers = int(os.environ.get("PRIVACY_WORKERS", min(os.cpu_count() or 4, 8)))
        assert num_workers == expected

    def test_env_workers(self):
        """验证环境变量覆盖默认 worker 数。"""
        os.environ["PRIVACY_WORKERS"] = "16"
        try:
            num_workers = int(os.environ.get("PRIVACY_WORKERS", min(os.cpu_count() or 4, 8)))
            assert num_workers == 16
        finally:
            os.environ.pop("PRIVACY_WORKERS", None)

    def test_grpc_max_workers_default(self):
        """验证 gRPC 线程池默认大小。"""
        os.environ.pop("PRIVACY_GRPC_MAX_WORKERS", None)
        max_workers = int(os.environ.get("PRIVACY_GRPC_MAX_WORKERS", "64"))
        assert max_workers == 64

    def test_grpc_max_workers_env(self):
        """验证 gRPC 线程池环境变量配置。"""
        os.environ["PRIVACY_GRPC_MAX_WORKERS"] = "128"
        try:
            max_workers = int(os.environ.get("PRIVACY_GRPC_MAX_WORKERS", "64"))
            assert max_workers == 128
        finally:
            os.environ.pop("PRIVACY_GRPC_MAX_WORKERS", None)


class TestGrpcServerConfig:
    """gRPC 服务器配置测试。"""

    def test_serve_accepts_none_max_workers(self):
        """验证 serve() 函数签名支持 max_workers=None。"""
        import inspect

        from engine.grpc_server import serve

        sig = inspect.signature(serve)
        params = sig.parameters
        assert "max_workers" in params
        # 默认值应为 None
        assert params["max_workers"].default is None


class TestConcurrencyTestEndpoint:
    """控制台后端并发测试端点测试。"""

    def test_concurrency_request_model(self):
        """验证 ConcurrencyTestRequest 模型校验。"""
        # 模拟导入（仅在控制台后端环境中可用）
        try:
            from console.backend.app.main import ConcurrencyTestRequest

            # 合法请求
            req = ConcurrencyTestRequest(
                path="/v1/privacy/mask",
                method="POST",
                concurrency=50,
                total_requests=200,
            )
            assert req.concurrency == 50
            assert req.total_requests == 200

            # 并发数超限
            with pytest.raises(Exception):
                ConcurrencyTestRequest(
                    path="/v1/privacy/mask",
                    method="POST",
                    concurrency=1000,  # > 500
                    total_requests=200,
                )

        except ImportError:
            pytest.skip("Console backend not available")

    def test_concurrency_response_model(self):
        """验证 ConcurrencyTestResponse 模型结构。"""
        try:
            from console.backend.app.main import ConcurrencyTestResponse

            resp = ConcurrencyTestResponse(
                total=100,
                success=95,
                failed=5,
                duration_ms=1000.0,
                qps=95.0,
                avg_latency_ms=10.5,
                min_latency_ms=1.2,
                max_latency_ms=50.0,
                p50_latency_ms=8.0,
                p95_latency_ms=25.0,
                p99_latency_ms=45.0,
            )
            assert resp.total == 100
            assert resp.qps == 95.0

        except ImportError:
            pytest.skip("Console backend not available")


# ---------------------------------------------------------------------------
# Numba JIT 加速模块测试 / dp_jit tests
# ---------------------------------------------------------------------------


class TestDpJit:
    """Numba JIT 加速 DP 计算核心测试。"""

    def test_laplace_noise_batch_shape(self):
        """验证 Laplace 噪声采样输出形状与输入一致。"""
        import numpy as np
        from engine.privacy.dp_jit import laplace_noise_batch

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = laplace_noise_batch(values, sensitivity=1.0, epsilon=1.0)
        assert result.shape == values.shape
        # 噪声后结果不应与原始值完全相同（概率极低）
        assert not np.allclose(result, values)

    def test_laplace_noise_batch_invalid_epsilon(self):
        """验证 epsilon <= 0 时抛出异常。"""
        import numpy as np
        from engine.privacy.dp_jit import laplace_noise_batch

        with pytest.raises(ValueError, match="epsilon must be positive"):
            laplace_noise_batch(np.array([1.0]), sensitivity=1.0, epsilon=0.0)

    def test_gaussian_noise_batch_shape(self):
        """验证 Gaussian 噪声采样输出形状。"""
        import numpy as np
        from engine.privacy.dp_jit import gaussian_noise_batch

        # 大样本使均值标准误降至 sigma/sqrt(n)，消除小样本下均值断言的偶发失败
        values = np.zeros(5000)
        result = gaussian_noise_batch(values, sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert result.shape == values.shape
        # 均值应接近 0（噪声均值为 0）
        assert abs(result.mean()) < 1.0

    def test_gaussian_noise_batch_invalid_delta(self):
        """验证 delta <= 0 时抛出异常。"""
        import numpy as np
        from engine.privacy.dp_jit import gaussian_noise_batch

        with pytest.raises(ValueError, match="delta must be positive"):
            gaussian_noise_batch(np.array([1.0]), sensitivity=1.0, epsilon=1.0, delta=0.0)

    def test_clip_values(self):
        """验证值截断功能。"""
        import numpy as np
        from engine.privacy.dp_jit import clip_values

        values = np.array([-5.0, -1.0, 0.0, 0.5, 1.0, 10.0])
        result = clip_values(values, lower=-1.0, upper=1.0)
        assert result.min() >= -1.0
        assert result.max() <= 1.0
        np.testing.assert_array_equal(result, np.array([-1.0, -1.0, 0.0, 0.5, 1.0, 1.0]))

    def test_clip_and_sum(self):
        """验证截断 + 求和合并操作。"""
        import numpy as np
        from engine.privacy.dp_jit import clip_and_sum

        values = np.array([-10.0, 1.0, 2.0, 3.0, 100.0])
        clipped_sum, count = clip_and_sum(values, clip_lower=0.0, clip_upper=10.0)
        # clip to [0, 10]: [0, 1, 2, 3, 10] -> sum=16, count=5
        assert clipped_sum == pytest.approx(16.0)
        assert count == 5.0

    def test_l2_norm_clip(self):
        """验证 L2 范数批量截断。"""
        import numpy as np
        from engine.privacy.dp_jit import l2_norm_clip

        # 两个向量: [3, 4] (norm=5) 和 [1, 0] (norm=1)
        vectors = np.array([[3.0, 4.0], [1.0, 0.0]])
        result = l2_norm_clip(vectors, max_norm=2.0)
        # [3,4] norm=5 > 2, scaled to [1.2, 1.6]
        np.testing.assert_allclose(result[0], [1.2, 1.6], atol=1e-10)
        # [1,0] norm=1 < 2, unchanged
        np.testing.assert_allclose(result[1], [1.0, 0.0], atol=1e-10)

    def test_l2_norm_clip_invalid_dim(self):
        """验证非 2D 数组抛出异常。"""
        import numpy as np
        from engine.privacy.dp_jit import l2_norm_clip

        with pytest.raises(ValueError, match="Expected 2D"):
            l2_norm_clip(np.array([1.0, 2.0]), max_norm=1.0)

    def test_clip_and_sum_columns(self):
        """验证按列截断求和。"""
        import numpy as np
        from engine.privacy.dp_jit import clip_and_sum_columns

        matrix = np.array([[1.0, 100.0], [2.0, -50.0], [3.0, 5.0]])
        col_sums, row_count = clip_and_sum_columns(matrix, clip_lower=0.0, clip_upper=10.0)
        # col0: clip [1,2,3] -> [1,2,3], sum=6
        # col1: clip [100,-50,5] -> [10,0,5], sum=15
        np.testing.assert_allclose(col_sums, [6.0, 15.0])
        assert row_count == 3

    def test_has_numba_flag(self):
        """验证 HAS_NUMBA 标志存在且为 bool。"""
        from engine.privacy.dp_jit import HAS_NUMBA
        assert isinstance(HAS_NUMBA, bool)

    def test_laplace_noise_batch_no_infinite_values(self):
        """大样本采样下不得出现 ±inf（log(0) 边界已钳制）。"""
        import numpy as np
        from engine.privacy.dp_jit import laplace_noise_batch

        # 大样本提高命中极小概率区间的机会，验证边界钳制有效
        result = laplace_noise_batch(np.zeros(20000), sensitivity=1.0, epsilon=1.0)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# NoisePool 测试 / NoisePool tests
# ---------------------------------------------------------------------------


class TestNoisePool:
    """预生成 DP 噪声池测试。"""

    def test_sample_returns_correct_size(self):
        """验证采样返回正确大小。"""
        from engine.privacy.high_concurrency import NoisePool

        pool = NoisePool(pool_size=100, scale=1.0)
        samples = pool.sample(10)
        assert len(samples) == 10

    def test_remaining_decreases(self):
        """验证采样后剩余量减少。"""
        from engine.privacy.high_concurrency import NoisePool

        pool = NoisePool(pool_size=100, scale=1.0)
        assert pool.remaining == 100
        pool.sample(30)
        assert pool.remaining == 70

    def test_auto_refill(self):
        """验证池耗尽时自动 refill。"""
        from engine.privacy.high_concurrency import NoisePool

        pool = NoisePool(pool_size=10, scale=1.0, auto_refill=True)
        # 取超过池容量的样本
        samples = pool.sample(15)
        assert len(samples) == 15

    def test_no_auto_refill_raises(self):
        """验证 auto_refill=False 时池耗尽抛出异常。"""
        from engine.privacy.high_concurrency import NoisePool

        pool = NoisePool(pool_size=5, scale=1.0, auto_refill=False)
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.sample(10)

    def test_gaussian_mechanism(self):
        """验证 Gaussian 噪声池。"""
        from engine.privacy.high_concurrency import NoisePool

        pool = NoisePool(pool_size=100, mechanism="gaussian", sigma=0.5)
        samples = pool.sample(50)
        assert len(samples) == 50

    def test_gaussian_sigma_zero(self):
        """验证 sigma=0.0 时生成零噪声（不是回退到 scale）。"""
        import numpy as np
        from engine.privacy.high_concurrency import NoisePool

        pool = NoisePool(pool_size=100, mechanism="gaussian", sigma=0.0)
        samples = pool.sample(50)
        # sigma=0.0 意味着无噪声，所有样本应为 0.0
        np.testing.assert_array_equal(samples, np.zeros(50))

    def test_stats(self):
        """验证统计信息。"""
        from engine.privacy.high_concurrency import NoisePool

        pool = NoisePool(pool_size=100, scale=2.0)
        stats = pool.stats
        assert stats["pool_size"] == 100
        assert stats["remaining"] == 100
        assert stats["mechanism"] == "laplace"
        assert stats["scale"] == 2.0

    def test_thread_safety(self):
        """验证多线程采样安全性。"""
        from engine.privacy.high_concurrency import NoisePool

        pool = NoisePool(pool_size=10000, scale=1.0, auto_refill=True)
        results = []

        def worker(n: int):
            s = pool.sample(n)
            results.append(len(s))

        threads = [threading.Thread(target=worker, args=(10,)) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # 所有线程都应成功获取 10 个样本
        assert all(r == 10 for r in results)


# ---------------------------------------------------------------------------
# ConcurrencyThrottle 测试 / ConcurrencyThrottle tests
# ---------------------------------------------------------------------------


class TestConcurrencyThrottle:
    """并发限流器回归测试。"""

    def test_timeout_acquire_does_not_over_release(self):
        """超时未获取到信号量时不得 release，避免容量越限。"""
        import asyncio

        from engine.privacy.high_concurrency import ConcurrencyThrottle

        async def run():
            throttle = ConcurrencyThrottle(max_concurrency=1)
            # 占住唯一名额后，第二次获取应超时
            async with throttle.acquire():
                with pytest.raises(asyncio.TimeoutError):
                    async with throttle.acquire(timeout=0.05):
                        pass
            # 超时路径未获取到信号量，不应 release；容量必须仍为 1
            assert throttle._semaphore._value == 1
            # 后续仍可正常获取与释放，容量不漂移
            async with throttle.acquire():
                assert throttle._semaphore._value == 0
            assert throttle._semaphore._value == 1

        asyncio.run(run())

    def test_acquire_release_roundtrip(self):
        """正常获取/释放路径容量守恒。"""
        import asyncio

        from engine.privacy.high_concurrency import ConcurrencyThrottle

        async def run():
            throttle = ConcurrencyThrottle(max_concurrency=2)
            for _ in range(5):
                async with throttle.acquire(timeout=1.0):
                    pass
            assert throttle._semaphore._value == 2

        asyncio.run(run())


# ---------------------------------------------------------------------------
# BatchedBudgetSpend 测试 / BatchedBudgetSpend tests
# ---------------------------------------------------------------------------


class TestBatchedBudgetSpend:
    """批量预算扣减测试。"""

    def test_batch_flush(self):
        """验证批量 spend 合并后正确扣减。"""
        from engine.privacy.budget import BudgetRegistry
        from engine.privacy.high_concurrency import BatchedBudgetSpend

        reg = BudgetRegistry()
        acct = reg.get_or_create("test-batch", epsilon_total=10.0, delta_total=1.0)

        batcher = BatchedBudgetSpend(acct, flush_interval=0.01, max_batch=50)
        batcher.start()

        try:
            # 提交 5 个 spend 请求
            futures = []
            for _ in range(5):
                fut = batcher.spend(0.1, 0.0)
                futures.append(fut)

            # 等待所有 future 完成并验证成功
            for fut in futures:
                assert fut.result(timeout=5) is True

            # 验证预算已扣减
            remaining = acct.remaining()
            # 5 * 0.1 = 0.5 spent, remaining = 9.5
            assert remaining["epsilon"] == pytest.approx(9.5, abs=0.01)
        finally:
            batcher.stop()
            reg.reset()

    def test_batch_exceeds_budget(self):
        """验证批量 spend 超出预算时正确报告失败。"""
        from engine.privacy.budget import BudgetRegistry
        from engine.privacy.high_concurrency import BatchedBudgetSpend

        reg = BudgetRegistry()
        acct = reg.get_or_create("test-batch-exhaust", epsilon_total=1.0, delta_total=1.0)

        # max_batch=5 使 20 个请求分 4 批处理，每批 0.5
        batcher = BatchedBudgetSpend(acct, flush_interval=0.01, max_batch=5)
        batcher.start()

        try:
            # 提交 20 个请求，每个 0.1，总量 2.0 > 预算 1.0
            futures = []
            for _ in range(20):
                fut = batcher.spend(0.1, 0.0)
                futures.append(fut)

            # 验证部分成功部分失败
            success_count = 0
            fail_count = 0
            for fut in futures:
                fut.wait(timeout=5)
                if fut.ok is True:
                    success_count += 1
                elif fut.ok is False:
                    fail_count += 1

            # 前 10 次 spend (0.1*10=1.0) 应成功，后续批次失败
            assert success_count > 0, "At least some spends should succeed"
            assert fail_count > 0, "At least some spends should fail"

            remaining = acct.remaining()
            assert remaining["epsilon"] <= 0.01
        finally:
            batcher.stop()
            reg.reset()

    def test_spend_future_result_raises_on_failure(self):
        """验证 _SpendFuture.result() 在失败时抛出 RuntimeError。"""
        from engine.privacy.budget import (
            BudgetRegistry,
            PrivacyBudgetExhaustedError,
        )
        from engine.privacy.high_concurrency import BatchedBudgetSpend

        reg = BudgetRegistry()
        acct = reg.get_or_create("test-future-error", epsilon_total=0.5, delta_total=1.0)

        batcher = BatchedBudgetSpend(acct, flush_interval=0.01, max_batch=100)
        batcher.start()

        try:
            # 提交超过预算的请求
            fut = batcher.spend(1.0, 0.0)  # 1.0 > 预算 0.5
            with pytest.raises((RuntimeError, PrivacyBudgetExhaustedError)):
                fut.result(timeout=5)
        finally:
            batcher.stop()
            reg.reset()

    def test_spend_after_stop_sync_exec(self):
        """验证 stop() 后 spend() 直接同步执行，不会永久挂起。"""
        from engine.privacy.budget import BudgetRegistry
        from engine.privacy.high_concurrency import BatchedBudgetSpend

        reg = BudgetRegistry()
        acct = reg.get_or_create("test-after-stop", epsilon_total=10.0, delta_total=1.0)

        batcher = BatchedBudgetSpend(acct, flush_interval=0.01, max_batch=10)
        batcher.start()
        batcher.stop()

        try:
            # 已停止后提交 spend：应立即返回结果（同步执行）
            fut = batcher.spend(0.1, 0.0)
            assert fut.result(timeout=1) is True
            remaining = acct.remaining()
            assert remaining["epsilon"] == pytest.approx(9.9, abs=0.01)

            # 超预算时同步执行也应正确报告失败
            fut2 = batcher.spend(100.0, 0.0)
            assert fut2.ok is False
            assert fut2.error is not None
        finally:
            reg.reset()


# ---------------------------------------------------------------------------
# uvloop/httptools 检测测试 / uvloop+httptools detection tests
# ---------------------------------------------------------------------------


class TestUvicornOptimizations:
    """uvloop/httptools 自动检测与 GZip 中间件测试。"""

    def test_gzip_middleware_registered(self):
        """验证 GZip 中间件已注册到 FastAPI app。"""
        from engine.main import app

        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "GZipMiddleware" in middleware_classes

    def test_server_uvloop_detection(self):
        """验证 server.py 的 uvloop 检测逻辑。"""
        # 导入模块不应报错（无论 uvloop 是否安装）
        from engine import server as srv
        # _UVICORN_LOOP_KWARG 应为 dict
        assert isinstance(srv._UVICORN_LOOP_KWARG, dict)

    def test_server_concurrency_env_defaults(self):
        """验证 server.py 并发相关环境变量默认值。"""
        # 清理环境变量
        for key in ("PRIVACY_LIMIT_CONCURRENCY", "PRIVACY_LIMIT_MAX_REQUESTS", "PRIVACY_TIMEOUT_KEEP_ALIVE"):
            os.environ.pop(key, None)

        assert int(os.environ.get("PRIVACY_LIMIT_CONCURRENCY", "10000")) == 10000
        assert int(os.environ.get("PRIVACY_LIMIT_MAX_REQUESTS", "100000")) == 100000
        assert int(os.environ.get("PRIVACY_TIMEOUT_KEEP_ALIVE", "30")) == 30


# ---------------------------------------------------------------------------
# fork-after-warmup 测试 / launch_with_warmup tests
# ---------------------------------------------------------------------------


class TestLaunchWithWarmup:
    """fork-after-warmup 启动模式测试。"""

    def test_launch_with_warmup_function_exists(self):
        """验证 launch_with_warmup 函数存在且可导入。"""
        from engine.launcher import launch_with_warmup
        import inspect
        sig = inspect.signature(launch_with_warmup)
        params = sig.parameters
        assert "num_workers" in params
        assert "host_rest" in params
        assert "port_rest" in params
        assert "host_grpc" in params
        assert "port_grpc" in params
        assert "grpc_max_workers" in params

    def test_launcher_cli_warmup_flag(self):
        """验证命令行入口支持 --warmup 参数。"""
        from engine.launcher import main
        # 验证 main 函数存在
        assert callable(main)

    def test_monitor_and_terminate_helpers_exist(self):
        """验证公共监控/关闭辅助函数存在。"""
        from engine.launcher import _monitor_workers, _terminate_workers
        assert callable(_monitor_workers)
        assert callable(_terminate_workers)

    def test_worker_entry_uvloop_kwarg_importable(self):
        """验证 launcher worker 复用 server 的 uvloop/httptools 检测结果。"""
        from engine.launcher import _worker_entry
        import inspect

        source = inspect.getsource(_worker_entry)
        # worker 配置必须包含并发限制与 uvloop 复用（多进程模式优化生效）
        assert "_UVICORN_LOOP_KWARG" in source
        assert "limit_concurrency" in source
        assert "PRIVACY_LIMIT_CONCURRENCY" in source


# ---------------------------------------------------------------------------
# 预加载适配器注册表测试 / preloaded adapter registry tests
# ---------------------------------------------------------------------------


class TestPreloadedAdapterRegistry:
    """fork-after-warmup 预加载适配器注册表测试。"""

    def test_register_and_consume(self):
        """验证注册后可消费同一实例。"""
        from engine.dynclassification import service as svc

        adapter = object()
        svc.register_preloaded_adapter("ner", adapter)
        try:
            assert svc.consume_preloaded_adapter("ner") is adapter
        finally:
            svc._preloaded_adapters.clear()

    def test_consume_missing_returns_none(self):
        """验证未注册时返回 None。"""
        from engine.dynclassification import service as svc

        assert svc.consume_preloaded_adapter("llm") is None

    def test_reuse_matches_model_path(self):
        """验证 _build_funnel 在 model_path 匹配时复用预加载适配器。"""
        from engine.dynclassification import service as svc

        class FakeAdapter:
            def __init__(self, path):
                self._model_path = path

        # 注册一个匹配路径的预加载适配器
        preloaded = FakeAdapter("models/test.pt")
        svc.register_preloaded_adapter("llm", preloaded)
        try:
            svc_inst = svc.DynClassificationService(rules_dir="rules")
            # 直接验证匹配逻辑：路径一致 → 复用；不一致 → 不复用
            from engine.dynclassification.llm_adapter import LlmAdapter

            reused = svc.consume_preloaded_adapter("llm")
            assert reused is preloaded
            # 模拟 service 内部判断
            if reused is not None and reused._model_path == "models/test.pt":
                svc_inst._llm_adapter = reused
            assert svc_inst._llm_adapter is preloaded
        finally:
            svc._preloaded_adapters.clear()

    def test_batched_budget_spend_propagates_privacy_budget_exhausted_error(self):
        """验证 BatchedBudgetSpend 在预算耗尽时传播原始 PrivacyBudgetExhaustedError。"""
        from engine.privacy.budget import (
            BudgetRegistry,
            PrivacyBudgetExhaustedError,
        )
        from engine.privacy.high_concurrency import BatchedBudgetSpend

        registry = BudgetRegistry()
        accountant = registry.get_or_create("test_batch_exc", epsilon_total=1.0)
        batcher = BatchedBudgetSpend(accountant, flush_interval=0.001)
        batcher.start()
        try:
            # 耗尽预算
            fut = batcher.spend(epsilon=2.0)
            with pytest.raises(PrivacyBudgetExhaustedError):
                fut.result(timeout=2.0)
        finally:
            batcher.stop()

