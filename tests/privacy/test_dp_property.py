"""Property-based tests for DP module using hypothesis.

验证差分隐私模块的不变量和统计性质：
- 预算单调递减（只扣不减）
- 噪声对称性（Laplace 噪声均值为 0）
- 后处理封闭性（clip_non_negative 保证非负）
- 种子确定性（同 seed 同结果）
- 组合定理（多次查询预算叠加）
- 直方图单次预算消耗
"""
from __future__ import annotations

import math
from statistics import mean

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from engine.privacy.budget import default_registry
from engine.privacy.dp import DPApi

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registry():
    """每个测试前清空预算注册表。"""
    default_registry.reset()
    yield
    default_registry.reset()


def _make_api(epsilon: float = 100.0, seed: int = 42) -> DPApi:
    """创建大预算 DPApi（减少预算耗尽干扰）。"""
    return DPApi(namespace="prop_test", epsilon_total=epsilon, random_state=seed)


# ---------------------------------------------------------------------------
# 1. 预算单调性：每次查询后 epsilon_spent 只增不减
# ---------------------------------------------------------------------------

class TestBudgetMonotonicity:
    """预算单调递减性质。"""

    @given(
        epsilon=st.floats(min_value=0.01, max_value=1.0),
        data_vals=st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=50),
    )
    @settings(max_examples=30, deadline=None)
    def test_count_budget_decreases_monotonically(self, epsilon, data_vals):
        """每次 count 查询后，剩余预算严格递减。"""
        api = _make_api(epsilon=100.0)
        budget = api.budget
        eps_before = budget.epsilon_total - budget.epsilon_spent

        api.count(data_vals, epsilon=epsilon)

        eps_after = budget.epsilon_total - budget.epsilon_spent
        assert eps_after < eps_before

    @given(
        epsilon=st.floats(min_value=0.01, max_value=0.5),
        data_vals=st.lists(st.floats(min_value=-10, max_value=10), min_size=1, max_size=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_multiple_queries_budget_monotone(self, epsilon, data_vals):
        """多次查询后预算持续递减。"""
        api = _make_api(epsilon=100.0)
        budget = api.budget
        remaining = [budget.epsilon_total - budget.epsilon_spent]

        for _ in range(3):
            api.count(data_vals, epsilon=epsilon)
            remaining.append(budget.epsilon_total - budget.epsilon_spent)

        # Each step must be strictly less than the previous
        for i in range(1, len(remaining)):
            assert remaining[i] < remaining[i - 1]


# ---------------------------------------------------------------------------
# 2. 后处理封闭性：clip_non_negative 保证输出 >= 0
# ---------------------------------------------------------------------------

class TestPostProcessing:
    """后处理不变量。"""

    @given(
        data_vals=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=100),
        epsilon=st.floats(min_value=0.001, max_value=0.1),
    )
    @settings(max_examples=50, deadline=None)
    def test_count_clip_non_negative(self, data_vals, epsilon):
        """clip_non_negative=True 时，count 输出永远 >= 0。"""
        api = _make_api(epsilon=1000.0)  # large budget → small noise
        result = api.count(
            data_vals, epsilon=epsilon,
            clip_non_negative=True, round_int=True,
        )
        assert result >= 0.0

    @given(
        data_vals=st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=50),
        clip_lower=st.floats(min_value=-50, max_value=0),
        clip_upper=st.floats(min_value=0.01, max_value=50),
        epsilon=st.floats(min_value=0.001, max_value=0.1),
    )
    @settings(max_examples=40, deadline=None)
    def test_sum_clip_non_negative(self, data_vals, clip_lower, clip_upper, epsilon):
        """clip_non_negative=True 时，sum 输出永远 >= 0。"""
        assume(clip_lower < clip_upper)
        api = _make_api(epsilon=1000.0)
        result = api.sum(
            data_vals, epsilon=epsilon,
            clip_lower=clip_lower, clip_upper=clip_upper,
            clip_non_negative=True,
        )
        assert result >= 0.0

    @given(
        true_count=st.floats(min_value=0, max_value=1000),
        epsilon=st.floats(min_value=0.001, max_value=0.1),
    )
    @settings(max_examples=40, deadline=None)
    def test_noisy_count_clip_non_negative(self, true_count, epsilon):
        """noisy_count clip_non_negative=True 保证输出 >= 0。"""
        api = _make_api(epsilon=1000.0)
        result = api.noisy_count(true_count, epsilon=epsilon, clip_non_negative=True)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# 3. 种子确定性：同 seed 同查询 → 同结果
# ---------------------------------------------------------------------------

class TestSeedDeterminism:
    """种子可复现性。"""

    @given(
        data_vals=st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=50),
        epsilon=st.floats(min_value=0.1, max_value=5.0),
        seed=st.integers(min_value=0, max_value=99999),
    )
    @settings(max_examples=20, deadline=None)
    def test_same_seed_same_count_result(self, data_vals, epsilon, seed):
        """相同 seed + 相同数据 → 完全相同的 count 结果。"""
        api1 = DPApi(namespace="det1", epsilon_total=100.0, random_state=seed)
        r1 = api1.count(data_vals, epsilon=epsilon)

        default_registry.reset()

        api2 = DPApi(namespace="det1", epsilon_total=100.0, random_state=seed)
        r2 = api2.count(data_vals, epsilon=epsilon)

        assert r1 == r2

    @given(
        epsilon=st.floats(min_value=0.1, max_value=5.0),
        seed=st.integers(min_value=0, max_value=99999),
    )
    @settings(max_examples=20, deadline=None)
    def test_same_seed_same_noisy_sum(self, epsilon, seed):
        """相同 seed → noisy_sum 结果完全一致。"""
        api1 = DPApi(namespace="det2", epsilon_total=100.0, random_state=seed)
        r1 = api1.noisy_sum(42.0, sensitivity=1.0, epsilon=epsilon)

        default_registry.reset()

        api2 = DPApi(namespace="det2", epsilon_total=100.0, random_state=seed)
        r2 = api2.noisy_sum(42.0, sensitivity=1.0, epsilon=epsilon)

        assert r1 == r2


# ---------------------------------------------------------------------------
# 4. 组合定理：多次查询预算叠加
# ---------------------------------------------------------------------------

class TestComposition:
    """预算组合性质。"""

    @given(
        eps1=st.floats(min_value=0.1, max_value=2.0),
        eps2=st.floats(min_value=0.1, max_value=2.0),
        data_vals=st.lists(st.floats(min_value=-10, max_value=10), min_size=1, max_size=20),
    )
    @settings(max_examples=30, deadline=None)
    def test_two_queries_consume_additive_budget(self, eps1, eps2, data_vals):
        """两次查询消耗的总预算 = eps1 + eps2。"""
        api = _make_api(epsilon=100.0)
        budget = api.budget
        eps_before = budget.epsilon_spent

        api.count(data_vals, epsilon=eps1)
        api.count(data_vals, epsilon=eps2)

        assert math.isclose(budget.epsilon_spent - eps_before, eps1 + eps2, rel_tol=1e-9)

    @given(
        eps=st.floats(min_value=0.1, max_value=1.0),
        data_vals=st.lists(st.floats(min_value=-10, max_value=10), min_size=1, max_size=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_n_queries_consume_n_times_budget(self, eps, data_vals):
        """n 次相同查询消耗 n * eps 预算。"""
        n = 5
        api = _make_api(epsilon=100.0)
        budget = api.budget
        eps_before = budget.epsilon_spent

        for _ in range(n):
            api.count(data_vals, epsilon=eps)

        assert math.isclose(budget.epsilon_spent - eps_before, n * eps, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 5. 噪声统计性质：Laplace 噪声对称性（均值 ≈ 0）
# ---------------------------------------------------------------------------

class TestNoiseSymmetry:
    """噪声统计性质。"""

    def test_laplace_noise_zero_mean(self):
        """Laplace 噪声的样本均值应接近 0（大数定律）。"""
        true_value = 100.0
        epsilon = 1.0
        n_trials = 500

        noisy_results = []
        for i in range(n_trials):
            default_registry.reset()
            api = DPApi(namespace=f"sym_{i}", epsilon_total=1000.0, random_state=i)
            r = api.noisy_count(true_value, epsilon=epsilon)
            noisy_results.append(r)

        sample_mean = mean(noisy_results)
        # Laplace(0, b=1/eps) has mean 0; std = sqrt(2)*b = sqrt(2)/eps
        # With n=500, SE = std/sqrt(n) ≈ 0.045; use 5-sigma tolerance
        std_error = math.sqrt(2.0) / epsilon / math.sqrt(n_trials)
        assert abs(sample_mean - true_value) < 5 * std_error, (
            f"Sample mean {sample_mean} deviates from true value {true_value} "
            f"by more than 5 SE ({5 * std_error})"
        )

    def test_gaussian_noise_zero_mean(self):
        """Gaussian 噪声的样本均值应接近 0。"""
        true_value = 50.0
        epsilon = 1.0
        delta = 1e-5
        n_trials = 500

        noisy_results = []
        for i in range(n_trials):
            default_registry.reset()
            api = DPApi(namespace=f"gsym_{i}", epsilon_total=1000.0, random_state=i)
            r = api.noisy_count(true_value, epsilon=epsilon, delta=delta, mechanism="gaussian")
            noisy_results.append(r)

        sample_mean = mean(noisy_results)
        # For Gaussian, sigma is calibrated; use empirical std estimate
        se = 3.0 / math.sqrt(n_trials)  # conservative upper bound on SE
        assert abs(sample_mean - true_value) < 5 * se


# ---------------------------------------------------------------------------
# 6. 直方图单次预算消耗
# ---------------------------------------------------------------------------

class TestHistogramBudget:
    """直方图预算性质。"""

    @given(
        epsilon=st.floats(min_value=0.1, max_value=2.0),
        categories=st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=10, unique=True),
    )
    @settings(max_examples=20, deadline=None)
    def test_histogram_consumes_budget_once(self, epsilon, categories):
        """直方图无论多少桶，只消耗一次 epsilon 预算。"""
        api = _make_api(epsilon=100.0)
        budget = api.budget
        eps_before = budget.epsilon_spent

        true_counts = {c: float(c % 5 + 1) for c in categories}
        api.noisy_histogram(true_counts, epsilon=epsilon)

        assert math.isclose(budget.epsilon_spent - eps_before, epsilon, rel_tol=1e-9)

    @given(
        epsilon=st.floats(min_value=0.1, max_value=2.0),
        categories=st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=10, unique=True),
    )
    @settings(max_examples=20, deadline=None)
    def test_histogram_clip_non_negative(self, epsilon, categories):
        """直方图 clip_non_negative=True 保证所有桶 >= 0。"""
        api = _make_api(epsilon=1000.0)
        true_counts = {c: float(c % 5 + 1) for c in categories}
        result = api.noisy_histogram(
            true_counts, epsilon=epsilon,
            clip_non_negative=True, round_int=True,
        )
        for v in result.values():
            assert v >= 0.0


# ---------------------------------------------------------------------------
# 7. 输入验证 property：非法输入必定报错
# ---------------------------------------------------------------------------

class TestInputValidation:
    """输入校验不变量。"""

    @given(epsilon=st.floats(max_value=0.0))
    @settings(max_examples=10, deadline=None)
    def test_non_positive_epsilon_rejected(self, epsilon):
        """epsilon <= 0 必须抛出 ValueError。"""
        api = _make_api()
        with pytest.raises(ValueError, match="epsilon must be positive"):
            api.noisy_count(10.0, epsilon=epsilon)

    @given(delta=st.floats(max_value=-0.001))
    @settings(max_examples=10, deadline=None)
    def test_negative_delta_rejected(self, delta):
        """delta < 0 必须抛出 ValueError。"""
        api = _make_api()
        with pytest.raises(ValueError, match="delta must be non-negative"):
            api.noisy_count(10.0, epsilon=1.0, delta=delta)

    @given(confidence=st.floats(min_value=1.0, max_value=100.0) | st.floats(max_value=0.0, min_value=-100.0))
    @settings(max_examples=10, deadline=None)
    def test_invalid_confidence_level_rejected(self, confidence):
        """confidence_level 不在 (0,1) 必须抛出 ValueError。"""
        assume(confidence <= 0.0 or confidence >= 1.0)
        api = _make_api()
        with pytest.raises(ValueError, match="confidence_level must be in"):
            api.noisy_count(10.0, epsilon=1.0, confidence_level=confidence)
