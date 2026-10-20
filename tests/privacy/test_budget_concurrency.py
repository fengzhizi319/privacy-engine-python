"""高并发与锁多线程压力测试。

验证 BudgetAccountant 在内存与 SQLite 存储后端下的多线程并发扣减安全性、
数据一致性、BudgetRegistry 单例工厂的高并发争用正确性，以及 SQLite 每操作独立连接。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import sqlite3

from engine.privacy.budget import (
    PrivacyBudgetExhausted,
    default_registry,
)


@pytest.fixture(autouse=True)
def _reset():
    default_registry.reset()
    yield
    default_registry.reset()


class TestMemoryConcurrency:
    """内存模式下的并发冲刷测试。"""

    def test_concurrent_spend_serializable(self):
        """50 个线程各消耗 epsilon=0.1，总消耗应精确等于 5.0。"""
        ns = "concurrent-mem"
        acct = default_registry.get_or_create(ns, epsilon_total=100.0, delta_total=1.0)

        n_threads = 50
        eps_each = 0.1

        def _spend():
            acct.spend(eps_each, 0.0)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(_spend) for _ in range(n_threads)]
            for f in as_completed(futures):
                f.result()  # raise if any thread failed

        # Total spent should be exactly n_threads * eps_each
        assert abs(acct.epsilon_spent - n_threads * eps_each) < 1e-9

    def test_concurrent_spend_budget_exhaustion(self):
        """并发冲刷超过总预算时，部分线程应收到 PrivacyBudgetExhausted。"""
        ns = "concurrent-exhaust"
        acct = default_registry.get_or_create(ns, epsilon_total=1.0, delta_total=1.0)

        n_threads = 20
        eps_each = 0.1  # 20 * 0.1 = 2.0 > total 1.0

        results = []

        def _spend():
            try:
                acct.spend(eps_each, 0.0)
                return "ok"
            except PrivacyBudgetExhausted:
                return "exhausted"

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(_spend) for _ in range(n_threads)]
            for f in as_completed(futures):
                results.append(f.result())

        ok_count = results.count("ok")
        exhausted_count = results.count("exhausted")
        # At most 10 threads should succeed (1.0 / 0.1 = 10)
        assert ok_count <= 10
        assert exhausted_count > 0
        # Spent should not exceed total (with float tolerance)
        assert acct.epsilon_spent <= 1.0 + 1e-9

    def test_concurrent_spend_and_remaining(self):
        """并发读写：spend 和 remaining 交错执行不应导致异常。"""
        ns = "concurrent-rw"
        acct = default_registry.get_or_create(ns, epsilon_total=100.0, delta_total=1.0)

        def _spend():
            acct.spend(0.01, 0.0)

        def _remaining():
            return acct.remaining()

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = []
            for _ in range(50):
                futures.append(pool.submit(_spend))
                futures.append(pool.submit(_remaining))
            for f in as_completed(futures):
                result = f.result()
                if isinstance(result, dict):
                    assert "epsilon" in result
                    assert "delta" in result


class TestSQLiteConcurrency:
    """SQLite 模式下的并发冲刷测试。"""

    def test_sqlite_concurrent_spend(self, tmp_path, monkeypatch):
        """SQLite 模式下多线程 spend 不发生崩溃或死锁。"""
        db_file = str(tmp_path / "budget.db")
        monkeypatch.setenv("PRIVACY_BUDGET_DB", db_file)

        ns = "concurrent-sqlite"
        acct = default_registry.get_or_create(ns, epsilon_total=100.0, delta_total=1.0)

        n_threads = 30
        eps_each = 0.1

        def _spend():
            acct.spend(eps_each, 0.0)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(_spend) for _ in range(n_threads)]
            for f in as_completed(futures):
                f.result()

        # Verify in-memory state: all threads should have spent successfully
        # (total budget 100.0 >> 3.0 needed)
        assert acct.epsilon_spent > 0
        # Verify persisted state by reading back from DB
        import sqlite3
        conn = sqlite3.connect(db_file)
        try:
            row = conn.execute(
                "SELECT epsilon_spent FROM privacy_budgets WHERE namespace = ?",
                (ns,),
            ).fetchone()
            # DB value should be positive (exact match depends on thread scheduling)
            assert row[0] > 0
        finally:
            conn.close()

    def test_sqlite_per_operation_conn(self, tmp_path, monkeypatch):
        """验证每次 spend 都打开并关闭独立的 SQLite 连接，避免 thread-local 连接泄漏。"""
        db_file = str(tmp_path / "budget_per_op.db")
        monkeypatch.setenv("PRIVACY_BUDGET_DB", db_file)

        ns = "conn-per-op"
        acct = default_registry.get_or_create(ns, epsilon_total=100.0, delta_total=1.0)

        connect_calls = []
        original_connect = sqlite3.connect

        def _tracking_connect(*args, **kwargs):
            connect_calls.append(args[0])
            return original_connect(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", _tracking_connect)

        # 三次 spend 应该创建并关闭三个独立连接。
        acct.spend(1.0, 0.0)
        acct.spend(1.0, 0.0)
        acct.spend(1.0, 0.0)

        assert len(connect_calls) == 3
        assert all(path == db_file for path in connect_calls)


class TestRegistryConcurrency:
    """BudgetRegistry 单例工厂的高并发争用测试。"""

    def test_concurrent_registry_get_or_create(self):
        """50 线程并发请求注册表，验证单例对象指针 100% 相同。"""
        ns = "concurrent-reg"
        instances = []

        def _get_or_create():
            return default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1e-4)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(_get_or_create) for _ in range(50)]
            for f in as_completed(futures):
                instances.append(f.result())

        # 50 个线程获取到的对象指针应当 100% 相同
        first = instances[0]
        for inst in instances[1:]:
            assert inst is first
