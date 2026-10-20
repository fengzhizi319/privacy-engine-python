"""安全审计残留问题修复的回归测试（dp.py / budget.py）。

覆盖：
- vector_mean 高维 Laplace 噪声尺度的 √d L1 敏感度校准；
- clip 边界从数据推断时 DPResult.noise_scale / confidence_interval 置 None（防值域泄露），
  显式 clip 时正常返回；
- SQLite 预算路径下 spend 后 remaining 正确、窗口重置在排他事务内完成；
- 审计日志默认密钥改为进程级随机生成；
- noisy_mean / vector_mean / chunked_mean 低计数守卫分支与 mean 空数据集分支
  如实上报实际消耗的预算。
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import math
import sqlite3
import time

import numpy as np
import pytest

from engine.privacy.budget import (
    BudgetAuditLogger,
    BudgetRegistry,
    default_registry,
)
from engine.privacy.dp import DPApi, DPResult, LocalDPApi


def _make_api(ns: str, epsilon_total: float = 100.0, delta_total: float = 10.0) -> DPApi:
    """创建隔离命名空间的 DPApi（充足预算，避免跨用例干扰）。"""
    default_registry.get_or_create(ns, epsilon_total=epsilon_total, delta_total=delta_total)
    return DPApi(namespace=ns)


def test_corrupt_budget_database_fails_closed(tmp_path, monkeypatch):
    db_path = tmp_path / "corrupt-budget.db"
    db_path.write_bytes(b"not a sqlite database")
    monkeypatch.setenv("PRIVACY_BUDGET_DB", str(db_path))

    registry = BudgetRegistry()
    with pytest.raises(RuntimeError, match="integrity check failed"):
        registry.get_or_create("corrupt-budget-db")


class TestVectorMeanSqrtD:
    """#1 vector_mean 高维 Laplace 必须按 sqrt(d) * max_norm 的 L1 敏感度校准。"""

    def test_vector_mean_laplace_high_dim_uses_sqrt_d_sensitivity(self, monkeypatch):
        ns = "test-vm-sqrtd"
        api = _make_api(ns)
        # 固定 Laplace 噪声为 0（u = 0 → log(1) = 0），使 noisy_count 精确等于 n
        monkeypatch.setattr(api.rng, "random", lambda: 0.5)

        captured: dict[str, float] = {}

        def spy(self, d, mechanism, scale):
            captured["d"] = d
            captured["scale"] = scale
            return np.zeros(d)

        monkeypatch.setattr(DPApi, "_sample_isotropic_noise", spy)

        vectors = np.full((10, 4), 0.1)  # n=10, d=4
        res = api.vector_mean(
            vectors,
            max_norm=1.0,
            epsilon=2.0,  # eps_sub = 1.0
            mechanism="laplace",
            min_count=1.0,
            return_details=True,
        )
        assert isinstance(res, DPResult)
        # L1 敏感度上界 = sqrt(d) * max_norm = 2.0；sum 子查询 scale = 2.0 / eps_sub = 2.0
        assert captured["d"] == 4
        assert captured["scale"] == pytest.approx(2.0)
        # DPResult 上报的是 mean 等效 scale = sum_scale / noisy_count = 2.0 / 10
        assert res.noise_scale == pytest.approx(0.2)

    def test_vector_mean_laplace_one_dim_no_sqrt_factor(self, monkeypatch):
        ns = "test-vm-sqrtd-1d"
        api = _make_api(ns)
        monkeypatch.setattr(api.rng, "random", lambda: 0.5)

        captured: dict[str, float] = {}

        def spy(self, d, mechanism, scale):
            captured["scale"] = scale
            return np.zeros(d)

        monkeypatch.setattr(DPApi, "_sample_isotropic_noise", spy)

        vectors = np.full((10, 1), 0.1)  # d=1：L1 == L2，不应有 √d 放大
        api.vector_mean(
            vectors, max_norm=1.0, epsilon=2.0, mechanism="laplace", min_count=1.0
        )
        assert captured["scale"] == pytest.approx(1.0 / 1.0)  # max_norm / eps_sub


class TestInferredBoundsHiding:
    """#2 clip 边界由数据推断时，DPResult 不得携带数据推导的 noise_scale 与 CI。"""

    def test_sum_inferred_bounds_hides_noise_scale_and_ci(self):
        ns = "test-sum-inferred-hide"
        api = _make_api(ns)
        api.rng.seed(42)
        res = api.sum(
            [1.0, 2.0, 3.0], epsilon=1.0, mechanism="laplace", return_details=True
        )
        assert isinstance(res, DPResult)
        assert res.noise_scale is None
        assert res.confidence_interval is None

    def test_sum_inferred_bounds_to_arrow_exports_none(self):
        pa = pytest.importorskip("pyarrow")  # noqa: F841
        ns = "test-sum-inferred-arrow"
        api = _make_api(ns)
        api.rng.seed(42)
        res = api.sum(
            [1.0, 2.0, 3.0], epsilon=1.0, mechanism="laplace", return_details=True
        )
        table = res.to_arrow()
        meta = json.loads(table.schema.metadata[b"dp_metadata"])
        assert meta["noise_scale"] is None
        assert meta["confidence_interval"] is None

    def test_sum_explicit_clip_returns_noise_scale_and_ci(self):
        ns = "test-sum-explicit-scale"
        api = _make_api(ns)
        api.rng.seed(42)
        res = api.sum(
            [1.0, 2.0, 3.0],
            epsilon=1.0,
            mechanism="laplace",
            clip_lower=0.0,
            clip_upper=10.0,
            return_details=True,
        )
        # sensitivity = 10 - 0 = 10，Laplace b = 10 / 1.0
        assert res.noise_scale == pytest.approx(10.0)
        lo, hi = res.confidence_interval
        assert lo < res.value < hi

    def test_mean_inferred_bounds_hides_noise_scale_and_ci(self):
        ns = "test-mean-inferred-hide"
        api = _make_api(ns)
        api.rng.seed(42)
        res = api.mean(
            [1.0] * 10, epsilon=2.0, mechanism="laplace", min_count=1.0,
            return_details=True,
        )
        assert isinstance(res, DPResult)
        assert res.noise_scale is None
        assert res.confidence_interval is None

    def test_mean_explicit_clip_returns_noise_scale_and_ci(self):
        ns = "test-mean-explicit-scale"
        api = _make_api(ns)
        api.rng.seed(42)
        res = api.mean(
            [1.0] * 10,
            epsilon=2.0,
            mechanism="laplace",
            clip_lower=0.0,
            clip_upper=10.0,
            min_count=1.0,
            return_details=True,
        )
        assert res.noise_scale is not None
        assert res.noise_scale > 0.0
        assert res.confidence_interval is not None
        lo, hi = res.confidence_interval
        assert lo <= res.value <= hi

    def test_sum_sparse_inferred_bounds_hides_noise_scale(self):
        sp = pytest.importorskip("scipy.sparse")
        ns = "test-sum-sparse-inferred-hide"
        api = _make_api(ns)
        api.rng.seed(42)
        arr = sp.csr_matrix([1.0, 0.0, 2.0, 0.0, 3.0])
        res = api.sum(arr, epsilon=1.0, mechanism="laplace", return_details=True)
        assert res.noise_scale is None
        assert res.confidence_interval is None


class TestBudgetRemainingSQLite:
    """#3 SQLite 路径下 spend 后 remaining 正确，窗口重置在排他事务内完成。"""

    def test_sqlite_spend_then_remaining(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRIVACY_BUDGET_DB", str(tmp_path / "budget.db"))
        registry = BudgetRegistry()
        acc = registry.get_or_create("ns-rem", epsilon_total=10.0, delta_total=1e-4)
        acc.spend(3.0, 0.0)
        rem = acc.remaining()
        assert rem["epsilon"] == pytest.approx(7.0)
        assert rem["delta"] == pytest.approx(1e-4)

        # 模拟另一进程（新注册表/新实例）读到一致状态
        registry_b = BudgetRegistry()
        acc_b = registry_b.get_or_create("ns-rem", epsilon_total=10.0, delta_total=1e-4)
        assert acc_b.remaining()["epsilon"] == pytest.approx(7.0)
        acc_b.spend(2.0, 0.0)

        registry_c = BudgetRegistry()
        acc_c = registry_c.get_or_create("ns-rem", epsilon_total=10.0, delta_total=1e-4)
        assert acc_c.remaining()["epsilon"] == pytest.approx(5.0)

    def test_sqlite_remaining_uses_immediate_transaction(self, tmp_path, monkeypatch):
        """remaining() 的窗口检查 + 重置必须包裹在 BEGIN IMMEDIATE 排他事务中。"""
        db_file = str(tmp_path / "budget_tx.db")
        monkeypatch.setenv("PRIVACY_BUDGET_DB", db_file)
        registry = BudgetRegistry()
        acc = registry.get_or_create("ns-rem-tx", epsilon_total=10.0, delta_total=1e-4)

        class _ConnProxy:
            """包装真实连接以记录 SQL 语句（sqlite3.Connection 为 C 类型无法打桩）。"""

            def __init__(self, real):
                self._real = real
                self.statements: list[str] = []

            def execute(self, sql, *args, **kwargs):
                if isinstance(sql, str):
                    self.statements.append(sql.strip().upper())
                return self._real.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._real, name)

        proxy = _ConnProxy(sqlite3.connect(db_file))

        @contextlib.contextmanager
        def _patched_db_conn(path):
            yield proxy

        monkeypatch.setattr(acc, "_db_conn", _patched_db_conn)
        acc.remaining()
        assert "BEGIN IMMEDIATE" in proxy.statements

    def test_sqlite_remaining_window_reset_then_spend(self, tmp_path, monkeypatch):
        """窗口过期后 remaining 在事务内重置；重置后可立即花费满额预算。"""
        monkeypatch.setenv("PRIVACY_BUDGET_DB", str(tmp_path / "budget_win.db"))
        registry = BudgetRegistry()
        acc = registry.get_or_create(
            "ns-rem-win", epsilon_total=10.0, delta_total=1e-4, window_seconds=0.1
        )
        acc.spend(4.0, 0.0)
        assert acc.remaining()["epsilon"] == pytest.approx(6.0)

        time.sleep(0.15)
        # 窗口过期：remaining 重置并返回全额
        assert acc.remaining()["epsilon"] == pytest.approx(10.0)
        # 另一实例看到的也是重置后的全额，且另一实例花费后本实例不得抹掉
        registry_b = BudgetRegistry()
        acc_b = registry_b.get_or_create(
            "ns-rem-win", epsilon_total=10.0, delta_total=1e-4, window_seconds=0.1
        )
        assert acc_b.remaining()["epsilon"] == pytest.approx(10.0)
        acc_b.spend(3.0, 0.0)
        assert acc.remaining()["epsilon"] == pytest.approx(7.0)

    def test_sqlite_remaining_does_not_wipe_concurrent_spend(self, tmp_path, monkeypatch):
        """B 在窗口过期后 spend（事务内重置+扣减），A 随后的 remaining 不得抹掉 B 的花费。"""
        monkeypatch.setenv("PRIVACY_BUDGET_DB", str(tmp_path / "budget_race.db"))
        reg_a, reg_b = BudgetRegistry(), BudgetRegistry()
        acc_a = reg_a.get_or_create(
            "ns-race", epsilon_total=10.0, delta_total=1e-4, window_seconds=0.1
        )
        acc_b = reg_b.get_or_create(
            "ns-race", epsilon_total=10.0, delta_total=1e-4, window_seconds=0.1
        )
        acc_a.spend(5.0, 0.0)
        time.sleep(0.15)
        # B 先 spend：窗口重置并扣 3.0，窗口起点刷新为当前时间
        acc_b.spend(3.0, 0.0)
        # A 随后 remaining：窗口未过期，不得触发重置 UPDATE
        assert acc_a.remaining()["epsilon"] == pytest.approx(7.0)
        # 数据库中的累计花费仍是 B 扣减后的 3.0
        conn = sqlite3.connect(str(tmp_path / "budget_race.db"))
        try:
            row = conn.execute(
                "SELECT epsilon_spent FROM privacy_budgets WHERE namespace = ?",
                ("ns-race",),
            ).fetchone()
            assert row[0] == pytest.approx(3.0)
        finally:
            conn.close()


class TestAuditLoggerKey:
    """#4 未显式提供密钥时使用进程级随机密钥，且功能正常。"""

    def test_default_key_is_random_not_hardcoded(self, tmp_path, monkeypatch, caplog):
        monkeypatch.delenv("PRIVACY_AUDIT_KEY", raising=False)
        with caplog.at_level(logging.WARNING, logger="engine.privacy.budget"):
            logger1 = BudgetAuditLogger(log_file=str(tmp_path / "a1.log"))
            logger2 = BudgetAuditLogger(log_file=str(tmp_path / "a2.log"))
        # 不再是随源码公开的硬编码密钥
        assert logger1.secret_key != b"PrivShield-default-audit-key"
        # 每次生成均不同（进程级随机）
        assert logger1.secret_key != logger2.secret_key
        assert len(logger1.secret_key) == 64  # token_hex(32) 的 hex 长度
        # 强化后的 warning：说明进程随机、重启后不可校验、生产应显式配置
        assert any("audit_logger_process_random_key" in r.message for r in caplog.records)

    def test_default_key_signing_functional(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRIVACY_AUDIT_KEY", raising=False)
        log_file = str(tmp_path / "audit.log")
        audit = BudgetAuditLogger(log_file=log_file)
        sig = audit.log_spend("ns", 10.0, 1e-4, 1.0, 0.0)
        assert len(sig) == 64
        # 用实例内的随机密钥可复验日志行签名（本进程生命周期内可校验）
        with open(log_file, encoding="utf-8") as f:
            line = f.readline().strip()
        msg, recorded_sig = line.rsplit("|", 1)
        expected = hmac.new(audit.secret_key, msg.encode("utf-8"), hashlib.sha256).hexdigest()
        assert recorded_sig == sig == expected

    def test_explicit_key_still_supported(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PRIVACY_AUDIT_KEY", raising=False)
        log_file = str(tmp_path / "audit_explicit.log")
        audit = BudgetAuditLogger(secret_key=b"my-key", log_file=log_file)
        sig = audit.log_spend("ns", 10.0, 1e-4, 1.0, 0.0)
        with open(log_file, encoding="utf-8") as f:
            line = f.readline().strip()
        msg, recorded_sig = line.rsplit("|", 1)
        expected = hmac.new(b"my-key", msg.encode("utf-8"), hashlib.sha256).hexdigest()
        assert recorded_sig == sig == expected


class TestGuardBranchHonestSpend:
    """#5 低计数守卫/空数据集分支如实上报实际消耗的预算。"""

    def test_noisy_mean_guard_reports_half_budget(self, monkeypatch):
        ns = "test-nm-guard-honest"
        api = _make_api(ns)
        monkeypatch.setattr(api.rng, "random", lambda: 0.5)  # Laplace 噪声恒 0
        res = api.noisy_mean(
            true_sum=100.0,
            true_count=2.0,  # noisy_count = 2.0 < min_count = 5 → 守卫触发
            sensitivity=10.0,
            epsilon=1.0,
            mechanism="laplace",
            min_count=5.0,
            return_details=True,
        )
        assert res.value == 0.0
        # 仅 count 子查询花费 eps/2，sum 子查询未执行
        assert res.epsilon_spent == pytest.approx(0.5)
        assert res.delta_spent == pytest.approx(0.0)

    def test_noisy_mean_guard_reports_half_budget_gaussian(self, monkeypatch):
        ns = "test-nm-guard-honest-gauss"
        api = _make_api(ns)
        monkeypatch.setattr(api.rng, "gauss", lambda mu, sigma: 0.0)
        res = api.noisy_mean(
            true_sum=100.0,
            true_count=2.0,
            sensitivity=10.0,
            epsilon=1.0,
            delta=1e-4,
            mechanism="gaussian",
            min_count=5.0,
            return_details=True,
        )
        assert res.value == 0.0
        assert res.epsilon_spent == pytest.approx(0.5)
        assert res.delta_spent == pytest.approx(5e-5)

    def test_vector_mean_guard_reports_half_budget(self, monkeypatch):
        ns = "test-vm-guard-honest"
        api = _make_api(ns)
        monkeypatch.setattr(api.rng, "random", lambda: 0.5)
        vectors = np.array([[0.1, 0.2], [0.3, 0.4]])  # n=2 < min_count=5
        res = api.vector_mean(
            vectors,
            max_norm=1.0,
            epsilon=2.0,
            mechanism="laplace",
            min_count=5.0,
            return_details=True,
        )
        assert res.epsilon_spent == pytest.approx(1.0)  # eps_sub = 2.0 / 2
        assert res.delta_spent == pytest.approx(0.0)

    def test_chunked_mean_guard_reports_half_budget(self, monkeypatch):
        ns = "test-cm-guard-honest"
        api = _make_api(ns)
        monkeypatch.setattr(api.rng, "random", lambda: 0.5)
        res = api.chunked_mean(
            [[1.0, 2.0]],  # true_count = 2 < min_count = 5 → 守卫触发
            epsilon=2.0,
            mechanism="laplace",
            clip_lower=0.0,
            clip_upper=10.0,
            min_count=5.0,
            return_details=True,
        )
        assert res.value == 0.0
        assert res.epsilon_spent == pytest.approx(1.0)  # eps_sub
        assert res.delta_spent == pytest.approx(0.0)

    def test_mean_empty_dataset_reports_zero_spend(self):
        ns = "test-mean-empty-honest"
        api = _make_api(ns)
        res = api.mean([], epsilon=1.0, return_details=True)
        assert res.value == 0.0
        # 空数据集未执行任何子查询，一分未花
        assert res.epsilon_spent == 0.0
        assert res.delta_spent == 0.0


class TestTauThresholdFormula:
    """#6 dp_groupby 采用标准 Tau-Thresholding：tau = 1 + ln(1/(2δ_q))/ε_q。"""

    def test_group_between_old_and_new_tau_is_retained(self, monkeypatch):
        """计数落在旧公式与新公式 tau 之间的分组应被保留（验证公式已修正）。

        2 个分组、epsilon=1.0、delta=0.25：
        ε_q = 1/(2*2) = 0.25，δ_q = 0.25/4 = 0.0625
        新 tau = 1 + ln(1/(2*0.0625))/0.25 = 1 + ln(8)/0.25  ≈ 9.32
        旧 tau = 1 + ln(1/0.0625)/0.25      = 1 + ln(16)/0.25 ≈ 12.09
        计数 11 的分组：新公式保留，旧公式丢弃。
        """
        pd = pytest.importorskip("pandas")
        ns = "test-groupby-tau-formula"
        api = _make_api(ns)
        monkeypatch.setattr(api.rng, "random", lambda: 0.5)  # 噪声恒 0，计数精确

        df = pd.DataFrame({
            "city": ["BigCity"] * 100 + ["MidTown"] * 11,
            "income": [100.0] * 111,
        })
        res = api.dp_groupby(
            df, group_col="city", target_col="income", agg="count",
            epsilon=1.0, delta=0.25,
        )
        assert "BigCity" in res
        # 新公式 tau ≈ 9.32 < 11 → MidTown 保留（旧公式 tau ≈ 12.09 会丢弃）
        assert "MidTown" in res
        assert res["MidTown"] == pytest.approx(11.0)


class TestDiscreteLaplaceCI:
    """#8 离散 Laplace 的 CI 使用 Two-sided Geometric 精确尾部公式。"""

    def test_discrete_laplace_ci_uses_discrete_tail(self):
        ns = "test-discrete-ci"
        api = _make_api(ns)
        api.rng.seed(42)
        epsilon = 1.0
        res = api.count(
            [1.0, 2.0, 3.0], epsilon=epsilon, discrete=True, return_details=True
        )
        b = 1.0 / epsilon
        alpha = 0.05  # 默认 confidence_level = 0.95
        lam = math.exp(-1.0 / b)
        expected_margin = b * math.log(2.0 / (alpha * (1.0 + lam)))
        lo, hi = res.confidence_interval
        assert hi - res.value == pytest.approx(expected_margin)
        assert res.value - lo == pytest.approx(expected_margin)
        # 离散精确公式的 margin 大于连续近似 -b*ln(alpha)
        assert expected_margin > -b * math.log(alpha)


class TestAccumulatorClipWarning:
    """#7 create_accumulator 无 clip 时发出 DP 保证失效告警。"""

    def test_create_accumulator_without_clip_warns(self, caplog):
        ns = "test-acc-warn"
        api = _make_api(ns)
        with caplog.at_level(logging.WARNING, logger="engine.privacy.dp"):
            acc = api.create_accumulator([1.0, 2.0, 3.0])
        assert acc.sensitivity == 1.0
        assert any(
            "accumulator_sensitivity_unbounded" in r.message for r in caplog.records
        )

    def test_create_accumulator_with_clip_no_warning(self, caplog):
        ns = "test-acc-no-warn"
        api = _make_api(ns)
        with caplog.at_level(logging.WARNING, logger="engine.privacy.dp"):
            acc = api.create_accumulator([1.0, 2.0], clip_lower=0.0, clip_upper=10.0)
        assert acc.sensitivity == 10.0
        assert not any(
            "accumulator_sensitivity_unbounded" in r.message for r in caplog.records
        )


class TestLocalDPRandomSource:
    """#10 LDP 默认使用密码学安全 RNG；显式 seed 才用确定性 PRNG。"""

    def test_default_rng_is_system_random(self):
        import secrets as secrets_mod

        api = LocalDPApi()
        assert isinstance(api.rng, secrets_mod.SystemRandom)
        # 功能正常：扰动输出仍在合法值域内
        assert api.perturb_binary(1, epsilon=1.0) in (0, 1)
        assert api.perturb_categorical("A", ["A", "B", "C"], epsilon=1.0) in ["A", "B", "C"]

    def test_seeded_rng_is_deterministic(self):
        import random as random_mod

        api1 = LocalDPApi(seed=7)
        api2 = LocalDPApi(seed=7)
        assert isinstance(api1.rng, random_mod.Random)
        assert not isinstance(api1.rng, random_mod.SystemRandom)
        seq1 = [api1.perturb_binary(1, epsilon=1.0) for _ in range(50)]
        seq2 = [api2.perturb_binary(1, epsilon=1.0) for _ in range(50)]
        assert seq1 == seq2
