"""Differential Privacy algorithm correctness tests.

Covers Laplace/Gaussian mechanisms, numeric clipping, privacy budget accounting,
composition theorem, and Local DP randomized response with frequency estimation.

Test scope:
- Central DP: count, sum, mean, histogram with calibrated noise injection.
- Local DP: binary/categorical randomized response and unbiased frequency estimation.
- Budget accounting: epsilon/delta consumption tracking and exhaustion detection.
- Advanced features: sparse matrices, batch operations, chunked streaming,
  vector sum (DP-SGD), adaptive clipping, distributed accumulators, RDP accountant,
  GroupBy with tau-thresholding, User-Level DP, and Discrete Laplace mechanism.
"""

from __future__ import annotations

import statistics

import pytest

# Import central DP API (DPApi) for aggregate queries and local DP API (LocalDPApi)
# for per-record randomized response mechanisms.
from engine.privacy.budget import (
    PrivacyBudgetExhausted,
    default_registry,
)
from engine.privacy.dp import DPApi, LocalDPApi


class TestDPCount:
    """Tests for DP count queries under Laplace and Gaussian mechanisms.
    api.count 主要是用来做差分隐私计数的。
    简单说，它的作用是：
    ·统计数量：对输入数据里的有效记录/非零项做计数；
    ·保护隐私：在结果里加入受控噪声，避免泄露单个样本信息；
    ·消耗隐私预算：按 epsilon / delta 扣减预算，防止被重复查询“拼图式”推断；
    ·做结果后处理：比如可选地四舍五入、截断为非负数；
    ·支持多种数据格式：像数组、DataFrame、稀疏矩阵等都能处理。

    The count query has sensitivity = 1 (adding/removing one record changes
    the count by at most 1). Noise scale b = sensitivity / epsilon = 1/epsilon
    for Laplace mechanism.
    """

    def test_count_laplace_basic(self) -> None:
        # Step 1: Create DPApi with isolated namespace to avoid budget cross-contamination
        api = DPApi(namespace="test-count-laplace")
        # Step 2: Execute count query with large epsilon (10.0) => small noise scale b=0.1
        # True count of non-zero elements in [1.0, 0.0, 1.0, 1.0] = 3
        result = api.count([1.0, 0.0, 1.0, 1.0], epsilon=10.0, mechanism="laplace")
        #打印 result 以便调试
        print(result)
        # Step 3: With b=0.1, noise is negligible; result should be close to true count (3)
        assert 0 <= result <= 5

    def test_count_gaussian_basic(self) -> None:
        # Gaussian mechanism requires (epsilon, delta) with delta > 0
        # Analytic Gaussian calibrates sigma to satisfy (eps, delta)-DP
        api = DPApi(namespace="test-count-gaussian")
        result = api.count([1.0, 0.0, 1.0, 1.0], epsilon=10.0, delta=1e-5, mechanism="gaussian")
        # 打印 result 以便调试
        print(result)
        # Large epsilon => small sigma => result close to true count
        assert 0 <= result <= 5

    def test_count_gaussian_requires_delta(self) -> None:
        # Gaussian mechanism is undefined when delta=0; must raise ValueError
        api = DPApi(namespace="test-count-gaussian-delta")
        with pytest.raises(ValueError, match="delta must be positive"):
            api.count([1.0, 1.0], epsilon=1.0, delta=0.0, mechanism="gaussian")


class TestDPSum:
    """Tests for DP sum queries with explicit numeric clipping.

    Sensitivity for sum = clip_upper - clip_lower (bounded by clipping).
    Laplace noise scale b = sensitivity / epsilon.
    Without clipping, sensitivity is unbounded => Gaussian mechanism cannot be calibrated.
    """

    def test_sum_laplace_with_clipping(self) -> None:
        api = DPApi(namespace="test-sum-laplace-clip")
        api.rng.seed(42)  # Fix seed for reproducibility
        values = [1.0, 2.0, 3.0, 100.0]
        # Clipping to [0, 10]: value 100.0 gets clamped to 10.0
        result = api.sum(
            values,
            epsilon=10.0,
            mechanism="laplace",
            clip_lower=0.0,
            clip_upper=10.0,
        )
        # clipped sum = 1+2+3+10 = 16, noise scale b = (10-0)/10 = 1
        # With b=1, noise is small relative to sum; expect result in [10, 25]
        assert 10 <= result <= 25

    def test_sum_gaussian_requires_clip(self) -> None:
        # Gaussian mechanism requires explicit clip bounds to compute sensitivity
        # Without clip bounds, sensitivity = infinity => cannot calibrate noise
        api = DPApi(namespace="test-sum-gaussian-clip")
        with pytest.raises(ValueError, match="clip_lower and clip_upper"):
            api.sum([1.0, 2.0, 3.0], epsilon=1.0, delta=1e-6, mechanism="gaussian")

    def test_sum_gaussian_with_clip(self) -> None:
        api = DPApi(namespace="test-sum-gaussian-clip-ok")
        api.rng.seed(42)
        # sensitivity = 10-0 = 10; sigma = calibrate_analytic_gaussian(10.0, 1e-5, 10)
        result = api.sum(
            [1.0, 2.0, 3.0],
            epsilon=10.0,
            delta=1e-5,
            mechanism="gaussian",
            clip_lower=0.0,
            clip_upper=10.0,
        )
        # True clipped sum = 6; with large epsilon, result should be nearby
        assert 0 <= result <= 20

    def test_sum_backward_compat_without_clip(self, caplog) -> None:
        # Laplace mechanism can infer clip bounds from data (with warning)
        # for backward compatibility, but this is NOT recommended for production
        import logging

        api = DPApi(namespace="test-sum-compat")
        api.rng.seed(42)
        with caplog.at_level(logging.WARNING, logger="engine.privacy.dp"):
            result = api.sum([1.0, 2.0, 3.0], epsilon=10.0, mechanism="laplace")
        assert any("clip_bounds_inferred_from_data" in r.message for r in caplog.records)
        # Data-dependent clipping: sensitivity inferred from data range
        assert 0 <= result <= 10


class TestDPMean:
    """Tests for DP mean queries using the composition theorem.

    Mean is computed as noisy_sum / noisy_count. By the composition theorem,
    the total budget (epsilon, delta) is split equally: eps/2 for count, eps/2 for sum.
    The Delta method (first-order Taylor expansion) estimates the ratio variance:
    Var(S/C) ~ Var(S)/C^2 + (S^2 * Var(C))/C^4
    A min_count threshold prevents divergence when noisy_count is too small.
    """

    def test_mean_laplace_basic(self) -> None:
        api = DPApi(namespace="test-mean-laplace")
        api.rng.seed(42)
        # True mean = 3.0; budget split: eps/2=5.0 for count, eps/2=5.0 for sum
        result = api.mean(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            epsilon=10.0,
            mechanism="laplace",
            clip_lower=0.0,
            clip_upper=10.0,
        )
        # With large epsilon and 5 samples, result should be near true mean (3.0)
        assert 0 <= result <= 10

    def test_mean_gaussian_basic(self) -> None:
        api = DPApi(namespace="test-mean-gaussian")
        api.rng.seed(42)
        result = api.mean(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            epsilon=10.0,
            delta=1e-5,
            mechanism="gaussian",
            clip_lower=0.0,
            clip_upper=10.0,
        )
        assert 0 <= result <= 10

    def test_mean_empty(self) -> None:
        # Empty input: mean is undefined; implementation returns 0.0 without consuming budget
        api = DPApi(namespace="test-mean-empty")
        assert api.mean([], epsilon=1.0) == 0.0


class TestDPBudget:
    """Tests for privacy budget accounting and exhaustion detection.

    BudgetAccountant tracks cumulative (epsilon, delta) spending per namespace.
    Laplace consumes only epsilon; Gaussian consumes both epsilon and delta.
    Mean uses composition: splits budget into two sub-queries (count + sum).
    """

    def test_laplace_consumes_epsilon_only(self) -> None:
        ns = "test-budget-laplace"
        api = DPApi(namespace=ns)
        accountant = default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1e-4)
        # Laplace sum consumes epsilon=1.0, delta=0.0
        api.sum([1.0, 2.0], epsilon=1.0, mechanism="laplace", clip_lower=0.0, clip_upper=10.0)
        remaining = accountant.remaining()
        # epsilon: 10.0 - 1.0 = 9.0; delta unchanged at 1e-4
        assert remaining["epsilon"] == pytest.approx(9.0, abs=1e-9)
        assert remaining["delta"] == pytest.approx(1e-4, abs=1e-9)

    def test_gaussian_consumes_delta(self) -> None:
        ns = "test-budget-gaussian"
        api = DPApi(namespace=ns)
        accountant = default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1e-4)
        # Gaussian sum consumes both epsilon=1.0 and delta=1e-5
        api.sum(
            [1.0, 2.0],
            epsilon=1.0,
            delta=1e-5,
            mechanism="gaussian",
            clip_lower=0.0,
            clip_upper=10.0,
        )
        remaining = accountant.remaining()
        # epsilon: 10.0 - 1.0 = 9.0; delta: 1e-4 - 1e-5 = 9e-5
        assert remaining["epsilon"] == pytest.approx(9.0, abs=1e-9)
        assert remaining["delta"] == pytest.approx(9e-5, abs=1e-9)

    def test_mean_composition_consumes_full_budget(self, monkeypatch) -> None:
        ns = "test-budget-mean"
        api = DPApi(namespace=ns)
        accountant = default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1e-4)
        # Fix Gaussian noise to a constant (0.1) to prevent noisy_count from
        # being clipped to 0, which would trigger the min_count early-return
        # and skip the sum sub-query (thus not consuming full budget).
        monkeypatch.setattr(api.rng, "gauss", lambda mu, sigma: 0.1)
        api.mean(
            [1.0, 2.0, 3.0],
            epsilon=2.0,
            delta=1e-5,
            mechanism="gaussian",
            clip_lower=0.0,
            clip_upper=10.0,
            min_count=0.0,  # Disable low-count shield to ensure full budget consumption
        )
        remaining = accountant.remaining()
        # Composition: mean splits (eps=2.0, delta=1e-5) into two sub-queries
        # each consuming (eps=1.0, delta=5e-6); total consumed = (2.0, 1e-5)
        # epsilon: 10.0 - 2.0 = 8.0; delta: 1e-4 - 1e-5 = 9e-5
        assert remaining["epsilon"] == pytest.approx(8.0, abs=1e-9)
        assert remaining["delta"] == pytest.approx(9e-5, abs=1e-9)

    def test_exhausted_budget_raises(self) -> None:
        ns = "test-budget-exhaust"
        # Create BudgetAccountant with tight total budget (0.5);
        # DPApi reuses the singleton, so subsequent queries share this limit.
        default_registry.get_or_create(ns, epsilon_total=0.5, delta_total=1e-4)
        api = DPApi(namespace=ns)
        # Attempting to spend epsilon=1.0 > remaining 0.5 => PrivacyBudgetExhausted
        with pytest.raises(PrivacyBudgetExhausted):
            api.count([1.0, 1.0], epsilon=1.0)


class TestBinaryRandomizedResponse:
    """Tests for Local DP binary randomized response.

    The randomized response mechanism flips each binary value with probability
    p = 1/(1+exp(epsilon)). The debiased frequency estimator is:
    f_hat = (f_reported - p) / (1 - 2p)
    This satisfies epsilon-DP in the local model.
    """

    def test_perturb_binary_returns_zero_or_one(self):
        # Output domain must remain {0, 1} regardless of input
        api = LocalDPApi(seed=42)
        for _ in range(100):
            assert api.perturb_binary(1, epsilon=1.0) in (0, 1)
            assert api.perturb_binary(0, epsilon=1.0) in (0, 1)

    def test_perturb_binary_rejects_invalid_value(self):
        # Input must be binary (0 or 1); value=2 is out of domain
        api = LocalDPApi()
        with pytest.raises(ValueError, match="binary value must be 0 or 1"):
            api.perturb_binary(2, epsilon=1.0)

    def test_perturb_binary_rejects_invalid_epsilon(self):
        # Epsilon must be strictly positive for a valid privacy guarantee
        api = LocalDPApi()
        with pytest.raises(ValueError, match="epsilon must be positive"):
            api.perturb_binary(1, epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            api.perturb_binary(1, epsilon=-1.0)

    def test_higher_epsilon_preserves_more_truth(self):
        """Larger epsilon => higher probability of keeping original value."""
        api_low = LocalDPApi(seed=42)
        api_high = LocalDPApi(seed=42)

        n = 2000
        # Low epsilon (0.1): flip probability ~ 0.475 => many 1s become 0
        low_eps = [api_low.perturb_binary(1, epsilon=0.1) for _ in range(n)]
        # High epsilon (5.0): flip probability ~ 0.0067 => most 1s stay 1
        high_eps = [api_high.perturb_binary(1, epsilon=5.0) for _ in range(n)]

        # Higher epsilon preserves more truth => more 1s in output
        assert sum(low_eps) / n < sum(high_eps) / n

    def test_frequency_estimation_unbiased(self):
        """Debiased frequency estimator converges to true frequency (law of large numbers)."""
        api = LocalDPApi(seed=42)
        # True frequency = 1000/4000 = 0.25
        true_values = [1 if i % 4 == 0 else 0 for i in range(4000)]
        true_freq = sum(true_values) / len(true_values)

        # Apply randomized response with epsilon=1.0
        reported = api.perturb_binary_batch(true_values, epsilon=1.0)
        # Debias using the known flip probability
        est_freq = api.estimate_binary_frequency(reported, epsilon=1.0)

        # With n=4000, estimation error should be < 0.05 with high probability
        assert abs(est_freq - true_freq) < 0.05

    def test_frequency_estimation_empty(self):
        # Empty input => frequency is 0.0 (no data to estimate from)
        api = LocalDPApi()
        assert api.estimate_binary_frequency([], epsilon=1.0) == 0.0


class TestCategoricalRandomizedResponse:
    """Tests for Local DP categorical randomized response (RAPPOR-like).

    For k categories, the mechanism keeps the true value with probability
    p = exp(epsilon)/(exp(epsilon) + k - 1) and uniformly randomizes otherwise.
    The debiased histogram estimator inverts this mixing matrix.
    """

    def test_perturb_categorical_returns_valid_category(self):
        # Output must always be one of the provided categories
        api = LocalDPApi(seed=42)
        categories = ["A", "B", "C"]
        for _ in range(100):
            assert api.perturb_categorical("A", categories, epsilon=1.0) in categories

    def test_perturb_categorical_rejects_unknown_value(self):
        # Input value must belong to the category domain
        api = LocalDPApi()
        with pytest.raises(ValueError, match="value must be one of the provided categories"):
            api.perturb_categorical("D", ["A", "B", "C"], epsilon=1.0)

    def test_perturb_categorical_rejects_single_category(self):
        # With only 1 category, randomization is meaningless (no privacy possible)
        api = LocalDPApi()
        with pytest.raises(ValueError, match="categories must contain at least 2 items"):
            api.perturb_categorical("A", ["A"], epsilon=1.0)

    def test_histogram_estimation_unbiased(self):
        """Debiased histogram estimator converges to true distribution (n=10000)."""
        api = LocalDPApi(seed=42)
        categories = ["A", "B", "C"]
        # True distribution: A=50%, B=30%, C=20%
        true_values = ["A"] * 5000 + ["B"] * 3000 + ["C"] * 2000
        api.rng.shuffle(true_values)

        # Apply categorical randomized response
        reported = api.perturb_categorical_batch(true_values, categories, epsilon=1.0)
        # Invert the mixing matrix to estimate true proportions
        hist = api.estimate_categorical_histogram(reported, categories, epsilon=1.0)

        # With n=10000 and epsilon=1.0, estimation error < 5% per category
        assert abs(hist["A"] - 0.50) < 0.05
        assert abs(hist["B"] - 0.30) < 0.05
        assert abs(hist["C"] - 0.20) < 0.05
        # Probabilities must sum to exactly 1.0 (normalization invariant)
        assert abs(sum(hist.values()) - 1.0) < 1e-9

    def test_histogram_normalizes_when_all_negative_estimates(self):
        """When debiasing produces negative estimates, truncate to 0 and renormalize."""
        api = LocalDPApi()
        # With epsilon=0.1 and only 2 samples, debiasing likely yields negative estimates
        hist = api.estimate_categorical_histogram(
            ["A", "A"], categories=["A", "B"], epsilon=0.1
        )
        # Normalization must hold even after truncation
        assert abs(sum(hist.values()) - 1.0) < 1e-9
        assert all(v >= 0 for v in hist.values())


class TestBatchUtilities:
    """Tests for batch perturbation utilities (binary and categorical)."""

    def test_perturb_binary_batch_length(self):
        # Batch output length must match input length; all values remain binary
        api = LocalDPApi(seed=42)
        values = [1, 0, 1, 1, 0]
        reported = api.perturb_binary_batch(values, epsilon=1.0)
        assert len(reported) == len(values)
        assert all(v in (0, 1) for v in reported)

    def test_perturb_categorical_batch_length(self):
        # Batch output length must match input; all values remain valid categories
        api = LocalDPApi(seed=42)
        values = ["A", "B", "A", "C"]
        reported = api.perturb_categorical_batch(values, ["A", "B", "C"], epsilon=1.0)
        assert len(reported) == len(values)
        assert all(v in ["A", "B", "C"] for v in reported)


class TestStatisticalProperties:
    """Tests for statistical properties of randomized response estimators.

    The debiased estimator is unbiased: E[f_hat] = f_true.
    Across multiple independent trials, the mean estimation error should converge to 0.
    """

    def test_binary_noise_mean_tends_to_zero(self):
        """Mean estimation error across 20 independent trials should be near 0."""
        trials = []
        for seed in range(20):
            api = LocalDPApi(seed=seed)
            # True frequency = 0.5 (500 ones out of 1000)
            true_values = [1 if i % 2 == 0 else 0 for i in range(1000)]
            reported = api.perturb_binary_batch(true_values, epsilon=1.0)
            est = api.estimate_binary_frequency(reported, epsilon=1.0)
            # Record estimation error (bias)
            trials.append(est - 0.5)

        # Unbiased estimator: mean of errors across trials ~ 0
        assert abs(statistics.mean(trials)) < 0.02


class TestAnalyticGaussianAndMeanThreshold:
    """Tests for Analytic Gaussian calibration and mean low-count thresholding.

    The Analytic Gaussian mechanism provides tighter noise calibration than the
    classic formula sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon.
    Mean thresholding: if noisy_count < min_count, return 0.0 to prevent
    divergence of the ratio estimator noisy_sum / noisy_count.
    """

    def test_calibrate_analytic_gaussian_basic(self) -> None:
        from engine.privacy.dp import calibrate_analytic_gaussian
        # Analytic Gaussian sigma for eps=1.0, delta=1e-5, sensitivity=1.0
        sigma = calibrate_analytic_gaussian(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        assert sigma > 0
        # Classic formula gives ~4.84; analytic Gaussian is generally tighter
        assert sigma < 5.0

    def test_mean_thresholding_protection(self) -> None:
        ns = "test-mean-thresh"
        default_registry.get_or_create(ns, epsilon_total=100.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        # Normal case: 10 samples with min_count=2 => noisy_count >> 2 => valid result
        res = api.mean([10.0] * 10, epsilon=10.0, min_count=2.0)
        assert res > 0.0
        # Low-count case: 3 samples with min_count=5 => noisy_count likely < 5
        # => ratio estimator diverges => shield returns 0.0
        res_shielded = api.mean([10.0] * 3, epsilon=10.0, min_count=5.0)
        assert res_shielded == 0.0

    def test_dp_histogram_joint_sensitivity(self) -> None:
        """Histogram uses joint sensitivity = 1 (adding/removing one record affects one bin)."""
        ns = "test-hist-joint"
        default_registry.get_or_create(ns, epsilon_total=100.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        values = ["A"] * 100 + ["B"] * 200 + ["C"] * 50
        categories = ["A", "B", "C", "D"]
        # Laplace mechanism: noise scale b = 1/epsilon = 0.1 per bin
        res = api.histogram(values, categories, epsilon=10.0, mechanism="laplace")
        assert len(res) == 4
        assert res["A"] > 50   # true=100
        assert res["B"] > 100  # true=200
        assert res["C"] > 20   # true=50
        assert res["D"] >= 0.0  # true=0, but noise can be positive

        # Gaussian mechanism also supported for histograms with (eps, delta)-DP
        res_g = api.histogram(values, categories, epsilon=10.0, delta=1e-5, mechanism="gaussian")
        assert len(res_g) == 4
        assert res_g["A"] > 50




class TestVectorizedClip:
    """Tests for vectorized numeric clipping (_clip_values).

    Clipping bounds each value to [clip_lower, clip_upper] to control sensitivity.
    For sum: sensitivity = clip_upper - clip_lower.
    NaN values are preserved (not clipped) to maintain data integrity signals.
    """

    def test_clip_values_basic(self):
        import numpy as np
        api = DPApi(namespace="test-clip-basic")
        # Clip to [0, 10]: 100.0 -> 10.0, -5.0 -> 0.0
        result = api._clip_values([1.0, 2.0, 100.0, -5.0], 0.0, 10.0)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 10.0, 0.0])

    def test_clip_values_with_nan(self):
        import math

        api = DPApi(namespace="test-clip-nan")
        # NaN values pass through clipping unchanged (preserved as missing-data signals)
        result = api._clip_values([1.0, float("nan"), 100.0], 0.0, 10.0)
        assert result[0] == 1.0    # within bounds, unchanged
        assert result[2] == 10.0   # clipped to upper bound
        assert math.isnan(result[1])  # NaN preserved


class TestNoisyAggregation:
    """Tests for Noisify-style interfaces: inject DP noise into pre-aggregated results.

    These interfaces are used when the true aggregate (count, sum, histogram)
    has already been computed by an external engine. The DPApi only adds
    calibrated noise and tracks budget consumption.
    """

    def test_noisy_count_laplace(self):
        ns = "test-noisy-count"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        # Inject Laplace noise with b = sensitivity/epsilon = 1/1.0 = 1.0
        result = api.noisy_count(100.0, epsilon=1.0, mechanism="laplace")
        assert result >= 0.0
        accountant = default_registry.get_or_create(ns)
        # Budget consumed: epsilon=1.0 from total 10.0
        assert accountant.remaining()["epsilon"] == pytest.approx(9.0, abs=1e-9)

    def test_noisy_sum_requires_non_negative_sensitivity(self):
        # Negative sensitivity is mathematically invalid (|f(D)-f(D')| >= 0)
        ns = "test-noisy-sum-validation"
        api = DPApi(namespace=ns)
        with pytest.raises(ValueError, match="sensitivity"):
            api.noisy_sum(100.0, sensitivity=-1.0, epsilon=1.0)

    def test_noisy_sum_laplace(self):
        ns = "test-noisy-sum"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        # Noise scale b = sensitivity/epsilon = 10/1 = 10
        result = api.noisy_sum(
            100.0, sensitivity=10.0, epsilon=1.0, mechanism="laplace"
        )
        # With b=10, result is within ~30 of true sum with high probability
        assert result >= 80.0
        accountant = default_registry.get_or_create(ns)
        assert accountant.remaining()["epsilon"] == pytest.approx(9.0, abs=1e-9)

    def test_noisy_mean_low_count_shield(self):
        ns = "test-noisy-mean-shield"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        # true_count=2 < min_count=5 => ratio estimator unstable => return 0.0
        result = api.noisy_mean(
            true_sum=100.0,
            true_count=2.0,
            sensitivity=10.0,
            epsilon=1.0,
            mechanism="laplace",
            min_count=5.0,
        )
        assert result == 0.0

    def test_noisy_histogram(self):
        ns = "test-noisy-hist"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        # Joint sensitivity = 1 for histogram; noise b = 1/10 = 0.1 per bin
        result = api.noisy_histogram(
            {"A": 100.0, "B": 200.0, "C": 50.0},
            epsilon=10.0,
            mechanism="laplace",
        )
        assert len(result) == 3
        assert result["A"] > 50   # true=100
        assert result["B"] > 150  # true=200
        assert result["C"] > 20   # true=50


class TestChunkedAggregation:
    """Tests for chunked / streaming DP aggregation interface.

    Chunked APIs allow data to arrive in multiple batches (chunks) rather than
    all at once. The DP mechanism processes each chunk and accumulates noise
    only once at the final aggregation step, preserving the same privacy cost
    as a single-shot query over the concatenated data.
    """

    def test_chunked_count(self):
        ns = "test-chunked-count"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        # 3 chunks with total non-zero count = 5 (three 1.0s in chunk 0, two in chunk 1)
        chunks = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0]]
        result = api.chunked_count(chunks, epsilon=10.0, mechanism="laplace")
        # True count = 5; with large epsilon, result should be nearby
        assert 4 <= result <= 7
        accountant = default_registry.get_or_create(ns)
        # Entire budget consumed by the single chunked query
        assert accountant.remaining()["epsilon"] == pytest.approx(0.0, abs=1e-9)

    def test_chunked_sum_requires_clip(self):
        # Chunked sum requires explicit clip bounds, same as regular sum
        ns = "test-chunked-sum-clip"
        api = DPApi(namespace=ns)
        with pytest.raises(ValueError, match="clip_lower and clip_upper"):
            api.chunked_sum([[1.0, 2.0]], epsilon=1.0)

    def test_chunked_sum(self):
        ns = "test-chunked-sum"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        # Chunk 0: [1,2,3], Chunk 1: [100->10(clipped),5], Chunk 2: [8]
        chunks = [[1.0, 2.0, 3.0], [100.0, 5.0], [8.0]]
        result = api.chunked_sum(
            chunks,
            epsilon=10.0,
            mechanism="laplace",
            clip_lower=0.0,
            clip_upper=10.0,
        )
        # Clipped sum = 1+2+3+10+5+8 = 29; noise scale b = 10/10 = 1
        assert 20 <= result <= 40

    def test_chunked_mean(self):
        ns = "test-chunked-mean"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        # Concatenated data: [1,2,3,4,5]; true mean = 3.0
        chunks = [[1.0, 2.0, 3.0], [4.0, 5.0]]
        result = api.chunked_mean(
            chunks,
            epsilon=10.0,
            mechanism="laplace",
            clip_lower=0.0,
            clip_upper=10.0,
        )
        # Result should be within clipping bounds
        assert 0 <= result <= 10

    def test_chunked_histogram(self):
        ns = "test-chunked-hist"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        # Concatenated: A=3, B=3, C=1; noise scale b = 1/10 = 0.1 per bin
        chunks = [["A", "B", "A"], ["B", "C", "A"], ["B"]]
        result = api.chunked_histogram(
            chunks, categories=["A", "B", "C"], epsilon=10.0, mechanism="laplace"
        )
        assert result["A"] > 1   # true=3
        assert result["B"] > 2   # true=3
        assert result["C"] > 0   # true=1


class TestDataAdapters:
    """Tests for DP input data format adapters.

    DPApi methods accept multiple input types: Python list, numpy ndarray,
    pandas Series, and pandas DataFrame (with explicit column selection).
    These tests verify that all adapters produce correct results and that
    DataFrame inputs raise ValueError when no column is specified.
    """

    def test_count_with_pandas_series(self):
        # Count query accepts a pandas Series directly
        pd = pytest.importorskip("pandas")
        ns = "test-count-pd-series"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        series = pd.Series([1.0, 0.0, 1.0, 1.0])
        result = api.count(series, epsilon=10.0, mechanism="laplace")
        assert 0 <= result <= 5

    def test_sum_with_pandas_dataframe_column(self):
        # DataFrame input requires explicit column selection via `column` parameter
        pd = pytest.importorskip("pandas")
        ns = "test-sum-pd-df"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        df = pd.DataFrame({"salary": [1.0, 2.0, 3.0, 100.0], "age": [20, 30, 40, 50]})
        # Select "salary" column; clip 100.0 -> 10.0
        result = api.sum(
            df,
            epsilon=10.0,
            mechanism="laplace",
            clip_lower=0.0,
            clip_upper=10.0,
            column="salary",
        )
        # Clipped sum = 1+2+3+10 = 16
        assert 10 <= result <= 25

    def test_sum_with_numpy_array(self):
        # numpy ndarray input is handled via the same adapter path
        np = pytest.importorskip("numpy")
        ns = "test-sum-np"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        arr = np.array([1.0, 2.0, 3.0, 100.0])
        result = api.sum(
            arr, epsilon=10.0, mechanism="laplace", clip_lower=0.0, clip_upper=10.0
        )
        # Clipped sum = 1+2+3+10 = 16
        assert 10 <= result <= 25

    def test_histogram_with_pandas_series(self):
        # Categorical histogram accepts pandas Series of strings
        pd = pytest.importorskip("pandas")
        ns = "test-hist-pd"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        series = pd.Series(["A", "B", "A", "C"])
        result = api.histogram(
            series, categories=["A", "B", "C", "D"], epsilon=10.0, mechanism="laplace"
        )
        assert len(result) == 4
        assert result["A"] > 0.5  # true count = 2

    def test_dataframe_requires_column(self):
        # DataFrame without column parameter must raise ValueError
        pd = pytest.importorskip("pandas")
        ns = "test-df-requires-column"
        api = DPApi(namespace=ns)
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="column"):
            api.count(df, epsilon=1.0)


class TestDPResultAndPostProcessing:
    """Tests for DPResult structured output and post-processing options.

    When return_details=True, DPApi returns a DPResult dataclass containing:
    - value: the noisy aggregate
    - noise_mechanism: 'laplace' or 'gaussian'
    - noise_scale: calibrated noise parameter (b for Laplace, sigma for Gaussian)
    - epsilon_spent / delta_spent: actual budget consumed
    - confidence_interval: (lower, upper) bounds at a default confidence level

    Post-processing options:
    - round_int: round the result to the nearest integer
    - clip_non_negative: clamp negative results to 0.0
    """

    def test_dp_result_metadata(self):
        # Verify DPResult fields are populated correctly when return_details=True
        from engine.privacy.dp import DPResult

        ns = "test-dp-result"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        res = api.sum(
            [1.0, 2.0, 3.0],
            epsilon=1.0,
            mechanism="laplace",
            clip_lower=0.0,
            clip_upper=10.0,
            return_details=True,
        )
        assert isinstance(res, DPResult)
        assert res.noise_mechanism == "laplace"
        assert res.epsilon_spent == 1.0
        # Confidence interval must be a 2-tuple and contain the noisy value
        assert len(res.confidence_interval) == 2
        assert res.confidence_interval[0] < res.value < res.confidence_interval[1]

    def test_post_processing_options(self):
        # round_int => result is an integer-valued float
        # clip_non_negative => negative noise results are clamped to 0.0
        ns = "test-dp-postprocess"
        default_registry.get_or_create(ns, epsilon_total=10.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)
        res = api.count(
            [1.0, 2.0, 3.0],
            epsilon=1.0,
            round_int=True,
            clip_non_negative=True,
        )
        assert isinstance(res, float)
        assert res.is_integer()
        assert res >= 0.0


class TestSparseMatrixDP:
    """Tests for DP queries on scipy sparse matrices.

    Sparse matrices arise in high-dimensional settings (e.g., one-hot encodings,
    TF-IDF features). The DPApi extracts non-zero values for counting and applies
    clipping + noise for sum queries, preserving sparsity-aware efficiency.
    """

    def test_sparse_matrix_count_and_sum(self):
        sp = pytest.importorskip("scipy.sparse")
        ns = "test-sparse-dp"
        default_registry.get_or_create(ns, epsilon_total=20.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)

        # CSR matrix with 3 non-zero values: [1.0, 2.0, 3.0]
        sparse_arr = sp.csr_matrix([1.0, 0.0, 2.0, 0.0, 3.0])
        # Count of non-zero elements = 3 (before noise)
        count_res = api.count(sparse_arr, epsilon=5.0)
        assert count_res >= 0.0

        # Sum with clipping: clipped values = [1, 0, 2, 0, 3] => sum=6
        sum_res = api.sum(
            sparse_arr, epsilon=5.0, clip_lower=0.0, clip_upper=10.0
        )
        assert sum_res > 0.0


class TestBatchDP:
    """Tests for batch (multi-column) DP queries.

    batch_count / batch_sum apply DP noise independently to each column of a
    2D numpy array. Each column consumes its own portion of the privacy budget.
    Per-column clip bounds can be specified as lists.
    """

    def test_batch_count_and_sum(self):
        np = pytest.importorskip("numpy")
        from engine.privacy.dp import DPResult

        ns = "test-batch-dp"
        default_registry.get_or_create(ns, epsilon_total=100.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)

        # 3x2 matrix: column 0 = [1,3,5], column 1 = [2,4,6]
        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])

        # Batch count: one noisy count per column => 2 results
        batch_counts = api.batch_count(data, epsilon=2.0)
        assert len(batch_counts) == 2

        # Batch sum with per-column clip bounds; return_details => DPResult
        batch_sums = api.batch_sum(
            data,
            epsilon=2.0,
            clip_lower=[0.0, 0.0],
            clip_upper=[10.0, 10.0],
            return_details=True,
        )
        assert isinstance(batch_sums, DPResult)
        # value is a list of per-column noisy sums
        assert len(batch_sums.value) == 2
        # confidence_interval is a list of per-column (low, high) tuples
        assert len(batch_sums.confidence_interval) == 2


class TestRefinedDPFixes:
    """Tests for edge-case fixes in DP mean and confidence interval computation.

    - Sparse matrix mean: verifies noise_scale > 0 and CI is non-degenerate
      even when most values are zero (high sparsity).
    - Gaussian CI monotonicity: higher confidence level => wider interval.
    """

    def test_mean_sparse_matrix_shape_and_delta_ci(self):
        sp = pytest.importorskip("scipy.sparse")
        from engine.privacy.dp import DPResult

        ns = "test-sparse-mean-delta"
        default_registry.get_or_create(ns, epsilon_total=50.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)

        # 1000 rows but only 3 non-zero values (997 zeros + 10, 20, 30)
        dense_data = [0.0] * 997 + [10.0, 20.0, 30.0]
        sparse_arr = sp.csr_matrix(dense_data).T  # 1000x1 column matrix

        res_details = api.mean(
            sparse_arr,
            epsilon=2.0,
            clip_lower=0.0,
            clip_upper=100.0,
            min_count=1.0,
            return_details=True,
        )
        assert isinstance(res_details, DPResult)
        # Verify noise_scale > 0 and CI is non-degenerate (lower < value < upper)
        assert res_details.noise_scale > 0.0
        assert res_details.confidence_interval[0] < res_details.value < res_details.confidence_interval[1]

    def test_gaussian_confidence_interval_arbitrary_level(self):
        from engine.privacy.dp import compute_confidence_interval

        # Higher confidence level (0.99 vs 0.90) must produce a strictly wider CI
        low90, high90 = compute_confidence_interval(
            10.0, noise_scale=2.0, mechanism="gaussian", confidence_level=0.90
        )
        low99, high99 = compute_confidence_interval(
            10.0, noise_scale=2.0, mechanism="gaussian", confidence_level=0.99
        )
        assert (high99 - low99) > (high90 - low90)


class TestAdvancedDPFeatures:
    """Tests for advanced DP features: aggregate DSL, adaptive clipping,
    distributed accumulators, DP-SGD vector sum, RDP accountant,
    GroupBy with tau-thresholding, User-Level DP, Discrete Laplace,
    Arrow IPC metadata, and HMAC budget audit trail.
    """

    def test_dp_aggregate_dataframe(self):
        # dp_aggregate applies heterogeneous DP queries to different DataFrame columns
        pd = pytest.importorskip("pandas")
        ns = "test-dp-aggregate"
        default_registry.get_or_create(ns, epsilon_total=20.0, delta_total=1.0)
        api = DPApi(namespace=ns)

        df = pd.DataFrame({
            "age": [20, 30, 40, 50],
            "salary": [1000.0, 2000.0, 3000.0, 4000.0],
            "dept": ["eng", "hr", "eng", "sales"]
        })

        # Each column gets a different aggregation type with its own parameters
        specs = {
            "age": ("mean", {"clip_lower": 0, "clip_upper": 100}),
            "salary": ("sum", {"clip_lower": 0, "clip_upper": 10000}),
            "dept": ("histogram", {"categories": ["eng", "hr", "sales"]}),
        }

        res = api.dp_aggregate(df, specs, epsilon=3.0)
        assert "age" in res and "salary" in res and "dept" in res

    def test_adaptive_clip(self):
        # Adaptive clipping uses a portion of epsilon to find data-driven bounds
        # Lower bound is always 0 for non-negative data; upper bound is estimated
        ns = "test-adaptive-clip"
        api = DPApi(namespace=ns)
        lower, upper = api.adaptive_clip([1.0, 5.0, 10.0, 15.0, 20.0], epsilon=1.0, initial_clip=10.0)
        assert lower == 0.0
        assert upper > 0.0

    def test_accumulator_serialize_merge_finalize(self):
        """Test distributed DP accumulator: create -> serialize -> merge -> finalize."""
        from engine.privacy.dp import Accumulator

        ns = "test-accumulator"
        default_registry.get_or_create(ns, epsilon_total=20.0, delta_total=1.0)
        api = DPApi(namespace=ns)
        api.rng.seed(42)

        # Create two partial accumulators from disjoint data partitions
        acc1 = api.create_accumulator([1.0, 2.0], clip_lower=0.0, clip_upper=10.0)
        acc2 = api.create_accumulator([3.0, 4.0], clip_lower=0.0, clip_upper=10.0)

        # Serialize acc2, deserialize it back (simulates network transfer)
        serialized = acc2.serialize()
        acc2_deser = Accumulator.deserialize(serialized)

        # Merge accumulators: count=4, sum=10 (1+2+3+4)
        merged = acc1 + acc2_deser
        assert merged.count == 4.0
        assert merged.sum == 10.0

        # Finalize: add calibrated DP noise to the merged accumulator
        final_sum = api.finalize_dp(merged, aggregation="sum", epsilon=1.0)
        assert isinstance(final_sum, float)

    def test_vector_sum_dpsgd(self):
        # DP-SGD: clip each gradient vector by max_norm, then add Gaussian noise
        np = pytest.importorskip("numpy")
        ns = "test-vector-sum"
        default_registry.get_or_create(ns, epsilon_total=20.0, delta_total=1.0)
        api = DPApi(namespace=ns)

        # 2 gradient vectors of dimension 3
        grads = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        # Aggregate with L2 clipping (max_norm=5.0) and Gaussian noise
        noisy_grads = api.vector_sum(grads, max_norm=5.0, epsilon=1.0, delta=1e-4)
        assert noisy_grads.shape == (3,)

    def test_rdp_accountant(self):
        # Renyi DP accountant: tracks composition of Gaussian mechanisms
        # using Renyi divergence for tighter epsilon bounds
        from engine.privacy.budget import RDPAccountant

        rdp = RDPAccountant(target_delta=1e-5)
        # Record 10 Gaussian queries with sigma=2.0, sensitivity=1.0
        for _ in range(10):
            rdp.record_gaussian(sigma=2.0, sensitivity=1.0)

        eps = rdp.get_epsilon()
        assert 0.0 < eps < 50.0

    def test_dp_groupby(self):
        # DP GroupBy: adds noise per group and applies tau-thresholding
        # to suppress groups with too few members (prevents privacy leakage)
        pd = pytest.importorskip("pandas")
        ns = "test-dp-groupby"
        default_registry.get_or_create(ns, epsilon_total=100.0, delta_total=1.0)
        api = DPApi(namespace=ns)

        df = pd.DataFrame({
            "city": ["Beijing"] * 50 + ["Shanghai"] * 40 + ["TinyVillage"] * 1,
            "income": [100.0] * 91,
        })

        # TinyVillage has only 1 member => suppressed by tau-thresholding
        res = api.dp_groupby(df, group_col="city", target_col="income", agg="count", epsilon=2.0, delta=1e-3)
        assert "Beijing" in res
        assert "TinyVillage" not in res

    def test_user_level_dp_bounding(self):
        # User-level DP: bound each user's contribution to at most max_contributions
        # via downsampling, then apply standard DP mechanism
        np = pytest.importorskip("numpy")
        ns = "test-user-level-dp"
        default_registry.get_or_create(ns, epsilon_total=20.0, delta_total=1.0)
        api = DPApi(namespace=ns)

        # user_A has 100 records in the log; downsample to at most 2 contributions
        vals = np.array([10.0] * 100)
        uids = ["user_A"] * 100
        res = api.count(vals, epsilon=1.0, user_ids=uids, max_contributions=2)
        assert res >= 0.0

    def test_discrete_laplace(self):
        # Discrete Laplace mechanism: adds integer-valued noise from
        # the discrete Laplace distribution (no rounding needed)
        ns = "test-discrete-laplace"
        api = DPApi(namespace=ns)
        res = api.count([1.0, 2.0, 3.0], epsilon=1.0, discrete=True)
        assert isinstance(res, (int, float))

    def test_dp_result_to_arrow(self):
        # DPResult can serialize to an Arrow Table with DP metadata in schema
        pa = pytest.importorskip("pyarrow")
        ns = "test-arrow-metadata"
        api = DPApi(namespace=ns)
        res_details = api.count([1.0, 2.0, 3.0], epsilon=1.0, return_details=True)

        arrow_table = res_details.to_arrow()
        assert isinstance(arrow_table, pa.Table)
        assert b"dp_metadata" in arrow_table.schema.metadata

    def test_hmac_budget_audit(self):
        # HMAC-based audit logger: creates a tamper-evident log of budget spends
        import os
        import tempfile

        from engine.privacy.budget import BudgetAuditLogger

        # 使用 tempfile 保证跨平台兼容（Windows 无 /tmp）
        audit_path = os.path.join(tempfile.gettempdir(), "test_budget_audit.log")
        if os.path.exists(audit_path):
            os.remove(audit_path)

        logger = BudgetAuditLogger(log_file=audit_path)
        # Log a budget spend; returns a 64-char HMAC-SHA256 signature
        sig = logger.log_spend("test_ns", 1.0, 0.0, 1.0, 0.0)
        assert len(sig) == 64
        assert os.path.exists(audit_path)
        if os.path.exists(audit_path):
            os.remove(audit_path)





def test_dpapi_passes_budget_params():
    """DPApi 应能将预算参数透传给注册表。"""
    ns = "test-dpapi-budget-params"
    api = DPApi(
        namespace=ns,
        epsilon_total=3.0,
        delta_total=1e-3,
        window_seconds=120.0,
    )
    assert api.budget.epsilon_total == 3.0
    assert api.budget.delta_total == 1e-3
    assert api.budget.window_seconds == 120.0

    # 当 DPApi 未指定参数时，应复用已有实例且不触发告警
    existing = api.budget
    api2 = DPApi(namespace=ns)
    assert api2.budget is existing
