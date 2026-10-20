"""隐私预算记账模块测试。

验证单机内存预算模式、多进程 SQLite 持久化预算模式，以及时间窗口重置机制的正确性。
"""

import os
import time

import pytest

from engine.privacy.budget import (
    BudgetAccountant,
    BudgetRegistry,
    PrivacyBudgetExhausted,
    default_registry,
    get_budget,
)


def test_memory_budget_accountant():
    """测试常规内存单例记账模式。"""
    default_registry.reset()

    accountant = default_registry.get_or_create("test-ns-memory", epsilon_total=5.0, delta_total=1e-5)
    assert accountant.remaining()["epsilon"] == 5.0

    accountant.spend(2.0, 0.0)
    assert accountant.remaining()["epsilon"] == 3.0

    # 尝试超出预算限制，应该抛出异常
    with pytest.raises(PrivacyBudgetExhausted):
        accountant.spend(4.0, 0.0)


def test_sqlite_budget_accountant(tmp_path):
    """测试基于 SQLite 的持久化记账模式，模拟多进程/实例数据共享。"""
    default_registry.reset()

    db_file = os.path.join(tmp_path, "budget_test.db")
    os.environ["PRIVACY_BUDGET_DB"] = db_file

    try:
        acc1 = default_registry.get_or_create("test-ns-sqlite", epsilon_total=8.0, delta_total=1e-5)
        assert os.path.exists(db_file)

        acc1.spend(3.0, 0.0)
        assert acc1.remaining()["epsilon"] == 5.0

        default_registry.reset()

        acc2 = default_registry.get_or_create("test-ns-sqlite", epsilon_total=8.0, delta_total=1e-5)
        assert acc2.remaining()["epsilon"] == 5.0

        acc2.spend(2.0, 0.0)
        assert acc2.remaining()["epsilon"] == 3.0

        with pytest.raises(PrivacyBudgetExhausted):
            acc2.spend(4.0, 0.0)

        assert acc2.remaining()["epsilon"] == 3.0

    finally:
        os.environ.pop("PRIVACY_BUDGET_DB", None)


def test_memory_budget_window_reset():
    """测试内存模式下预算按时间窗口自动重置。"""
    default_registry.reset()

    accountant = default_registry.get_or_create(
        "test-ns-window-memory",
        epsilon_total=2.0,
        delta_total=1e-5,
        window_seconds=0.1,
    )
    accountant.spend(1.5, 0.0)
    assert accountant.remaining()["epsilon"] == pytest.approx(0.5, abs=1e-9)

    time.sleep(0.15)

    accountant.spend(1.5, 0.0)
    assert accountant.remaining()["epsilon"] == pytest.approx(0.5, abs=1e-9)


def test_sqlite_budget_window_reset(tmp_path):
    """测试 SQLite 模式下预算按时间窗口自动重置，并能在新实例间共享。"""
    default_registry.reset()

    db_file = os.path.join(tmp_path, "budget_window_test.db")
    os.environ["PRIVACY_BUDGET_DB"] = db_file

    try:
        acc1 = default_registry.get_or_create(
            "test-ns-window-sqlite",
            epsilon_total=2.0,
            delta_total=1e-5,
            window_seconds=0.1,
        )
        acc1.spend(1.5, 0.0)
        assert acc1.remaining()["epsilon"] == pytest.approx(0.5, abs=1e-9)

        time.sleep(0.15)

        default_registry.reset()
        acc2 = default_registry.get_or_create(
            "test-ns-window-sqlite",
            epsilon_total=2.0,
            delta_total=1e-5,
            window_seconds=0.1,
        )
        assert acc2.remaining()["epsilon"] == pytest.approx(2.0, abs=1e-9)

        acc2.spend(1.0, 0.0)
        assert acc2.remaining()["epsilon"] == pytest.approx(1.0, abs=1e-9)

    finally:
        os.environ.pop("PRIVACY_BUDGET_DB", None)


def test_env_budget_window_seconds():
    """测试通过环境变量 PRIVACY_BUDGET_WINDOW_SECONDS 配置时间窗口。"""
    default_registry.reset()
    os.environ["PRIVACY_BUDGET_WINDOW_SECONDS"] = "0.05"

    try:
        accountant = default_registry.get_or_create(
            "test-ns-env-window",
            epsilon_total=2.0,
            delta_total=1e-5,
        )
        assert accountant.window_seconds == 0.05

        accountant.spend(1.5, 0.0)
        assert accountant.remaining()["epsilon"] == pytest.approx(0.5, abs=1e-9)

        time.sleep(0.08)
        accountant.spend(1.0, 0.0)
        assert accountant.remaining()["epsilon"] == pytest.approx(1.0, abs=1e-9)

    finally:
        os.environ.pop("PRIVACY_BUDGET_WINDOW_SECONDS", None)


def test_budget_registry_crud_and_warning(caplog):
    """测试 BudgetRegistry 的完整生命周期方法及参数冲突警告。"""
    import logging

    registry = BudgetRegistry()
    assert registry.get("hr") is None

    acct1 = registry.get_or_create("hr", epsilon_total=10.0, delta_total=1e-4)
    assert acct1.epsilon_total == 10.0
    assert registry.get("hr") is acct1

    with caplog.at_level(logging.WARNING, logger="engine.privacy.budget"):
        acct2 = registry.get_or_create("hr", epsilon_total=999.0, delta_total=1e-4)
    assert any("budget_registry_params_ignored" in r.message for r in caplog.records)

    assert acct2 is acct1
    assert acct2.epsilon_total == 10.0

    caplog.clear()
    acct3 = registry.get_or_create("window-ns", epsilon_total=5.0, window_seconds=60.0)
    with caplog.at_level(logging.WARNING, logger="engine.privacy.budget"):
        acct4 = registry.get_or_create("window-ns", window_seconds=120.0)
    assert any("budget_registry_params_ignored" in r.message for r in caplog.records)
    assert acct4 is acct3
    assert acct3.window_seconds == 60.0

    registry.get_or_create("hr")

    removed = registry.remove("hr")
    assert removed is acct1
    assert registry.get("hr") is None

    registry.get_or_create("sales", epsilon_total=5.0)
    registry.get_or_create("marketing", epsilon_total=5.0)
    assert registry.get("sales") is not None
    assert registry.get("marketing") is not None

    registry.reset()
    assert registry.get("sales") is None
    assert registry.get("marketing") is None


def test_get_or_create_no_warning_when_params_omitted(caplog):
    """未显式传入预算参数时，即使现有实例配置不同也不应告警。"""
    import logging

    registry = BudgetRegistry()
    registry.get_or_create("ns-no-warn", epsilon_total=100.0, delta_total=1e-3)
    with caplog.at_level(logging.WARNING, logger="engine.privacy.budget"):
        registry.get_or_create("ns-no-warn")
    assert not any("budget_registry_params_ignored" in r.message for r in caplog.records)


def test_direct_construction_raises_typeerror():
    """测试直接调用 BudgetAccountant() 抛出 TypeError。"""
    default_registry.reset()
    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        BudgetAccountant("legacy-ns", epsilon_total=10.0, delta_total=1e-4)


def test_subclass_construction_raises_typeerror():
    """测试继承 BudgetAccountant 时直接构造同样抛出 TypeError。"""
    class CustomAccountant(BudgetAccountant):
        pass

    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        CustomAccountant("subclass-ns")


def test_budget_accountant_repr():
    """测试 BudgetAccountant 的 __repr__ 输出。"""
    default_registry.reset()
    acct = default_registry.get_or_create("repr-ns", epsilon_total=10.0, delta_total=1e-4)
    acct.spend(2.5, 1e-5)

    repr_str = repr(acct)
    assert "namespace='repr-ns'" in repr_str
    assert "epsilon=7.5000/10.0" in repr_str
    assert "delta=9.00e-05/0.0001" in repr_str


def test_get_budget_convenience_function():
    """测试模块级便捷函数 get_budget()。"""
    default_registry.reset()
    acct1 = get_budget("conv-ns", epsilon_total=5.0, delta_total=1e-4)
    assert acct1.namespace == "conv-ns"
    assert acct1.epsilon_total == 5.0

    acct2 = get_budget("conv-ns")
    assert acct2 is acct1
