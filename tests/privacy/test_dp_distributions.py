"""KS 统计分布检验：验证 DP 噪声采样的统计学正确性 / KS Distribution Tests for DP Noise Sampling.

中文说明：
本模块使用 Kolmogorov-Smirnov 检验验证差分隐私噪声生成的统计学正确性：

1. KS 检验原理 / KS Test Principle:
   - 比较经验分布函数 (ECDF) 与理论 CDF 的最大偏差 D
   - p-value > alpha (0.01) 则不能拒绝“样本来自目标分布”的原假设
   - 采样 50,000 个点以确保统计功效 (statistical power)

2. 测试覆盖 / Test Coverage:
   - Laplace(0, scale) 分布：用于满足 (ε,0)-DP 的噪声机制
   - Gaussian N(0, σ²) 分布：用于满足 (ε,δ)-DP 的噪声机制
   - Discrete Laplace：用于整数计数的离散化噪声
   - 端到端噪声校准：验证 count 查询的经验方差符合理论值

3. 数学基础 / Mathematical Foundation:
   - Laplace(0,b): Var = 2b², b = sensitivity/ε
   - N(0,σ²): Var = σ², σ = calibrate_analytic_gaussian(ε, δ, sensitivity)
   - 大数定律：样本均值 → 0（对称分布的期望）

English Description:
Kolmogorov-Smirnov tests verifying statistical correctness of DP noise sampling,
covering Laplace, Gaussian, and Discrete Laplace distributions.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine.privacy.budget import default_registry
from engine.privacy.dp import DPApi


@pytest.fixture(autouse=True)
def _reset():
    default_registry.reset()
    yield
    default_registry.reset()


# scipy 为可选依赖，未安装时自动跳过 KS 检验
scipy_stats = pytest.importorskip("scipy.stats", reason="scipy required for KS tests")


N_SAMPLES = 50_000
KS_ALPHA = 0.01  # Significance level for KS test


class TestLaplaceDistribution:
    """验证 Laplace 噪声采样的分布正确性 / Laplace Noise Distribution Validation.

    Laplace 机制是满足 (ε,0)-DP 的标准机制，
    噪声尺度 b = sensitivity / epsilon。
    概率密度: f(x) = (1/2b) * exp(-|x|/b)
    """

    def test_sample_laplace_ks_test(self):
        """KS 检验：_sample_laplace(scale=1.0) 采样符合 Laplace(0, 1) 分布。"""
        api = DPApi(namespace="ks-laplace-1", random_state=12345)
        samples = np.array([api._sample_laplace(1.0) for _ in range(N_SAMPLES)])

        # Theoretical CDF: Laplace(0, 1)
        # CDF(x) = 0.5 * exp(x) for x < 0, 1 - 0.5 * exp(-x) for x >= 0
        # Use frozen distribution object for cleaner API
        ks_stat, p_value = scipy_stats.kstest(samples, scipy_stats.laplace(loc=0, scale=1.0).cdf)
        assert p_value > KS_ALPHA, (
            f"Laplace(0,1) KS test failed: stat={ks_stat:.4f}, p={p_value:.6f}"
        )

    def test_sample_laplace_different_scale(self):
        """KS 检验：_sample_laplace(scale=2.5) 符合 Laplace(0, 2.5) 分布。"""
        api = DPApi(namespace="ks-laplace-2", random_state=54321)
        scale = 2.5
        samples = np.array([api._sample_laplace(scale) for _ in range(N_SAMPLES)])

        ks_stat, p_value = scipy_stats.kstest(samples, scipy_stats.laplace(loc=0, scale=scale).cdf)
        assert p_value > KS_ALPHA, (
            f"Laplace(0,{scale}) KS test failed: stat={ks_stat:.4f}, p={p_value:.6f}"
        )

    def test_laplace_zero_mean(self):
        """Laplace 噪声均值应接近 0（大数定律）。"""
        api = DPApi(namespace="ks-laplace-mean", random_state=42)
        samples = [api._sample_laplace(1.0) for _ in range(N_SAMPLES)]
        mean = sum(samples) / len(samples)
        # Standard error of mean for Laplace(0,1) is sqrt(2/n) ≈ 0.0063
        assert abs(mean) < 5 * math.sqrt(2.0 / N_SAMPLES), (
            f"Laplace mean too far from 0: {mean:.6f}"
        )


class TestGaussianDistribution:
    """验证 Gaussian 噪声采样的分布正确性 / Gaussian Noise Distribution Validation.

    Gaussian 机制满足 (ε,δ)-DP，需要 delta > 0。
    噪声标准差 σ 通过 Analytic Gaussian 校准得到，
    比简单的 σ = sensitivity * sqrt(2*ln(1.25/δ)) / ε 更紧。
    """

    def test_sample_gaussian_ks_test(self):
        """KS 检验：_sample_gaussian(sigma=1.0) 采样符合 N(0, 1) 分布。"""
        api = DPApi(namespace="ks-gauss-1", random_state=12345)
        samples = np.array([api._sample_gaussian(1.0) for _ in range(N_SAMPLES)])

        # Use frozen distribution object for cleaner API
        ks_stat, p_value = scipy_stats.kstest(samples, scipy_stats.norm(loc=0, scale=1.0).cdf)
        assert p_value > KS_ALPHA, (
            f"N(0,1) KS test failed: stat={ks_stat:.4f}, p={p_value:.6f}"
        )

    def test_sample_gaussian_different_sigma(self):
        """KS 检验：_sample_gaussian(sigma=3.0) 符合 N(0, 9) 分布。"""
        api = DPApi(namespace="ks-gauss-3", random_state=54321)
        sigma = 3.0
        samples = np.array([api._sample_gaussian(sigma) for _ in range(N_SAMPLES)])

        ks_stat, p_value = scipy_stats.kstest(samples, scipy_stats.norm(loc=0, scale=sigma).cdf)
        assert p_value > KS_ALPHA, (
            f"N(0,{sigma}^2) KS test failed: stat={ks_stat:.4f}, p={p_value:.6f}"
        )

    def test_gaussian_zero_mean(self):
        """Gaussian 噪声均值应接近 0。"""
        api = DPApi(namespace="ks-gauss-mean", random_state=42)
        samples = [api._sample_gaussian(1.0) for _ in range(N_SAMPLES)]
        mean = sum(samples) / len(samples)
        # Standard error of mean for N(0,1) is 1/sqrt(n) ≈ 0.0045
        assert abs(mean) < 5 * (1.0 / math.sqrt(N_SAMPLES)), (
            f"Gaussian mean too far from 0: {mean:.6f}"
        )


class TestDiscreteLaplaceDistribution:
    """验证 Discrete Laplace 噪声采样的基本统计特性 / Discrete Laplace Properties.

    Discrete Laplace 是连续 Laplace 的整数版本，
    适用于计数查询（结果必须为整数）。
    P(X=k) ∝ exp(-|k|/scale), k ∈ ℤ
    """

    def test_discrete_laplace_integer_output(self):
        """Discrete Laplace 输出必须为整数。"""
        api = DPApi(namespace="ks-dlaplace-int", random_state=42)
        for _ in range(1000):
            val = api._sample_discrete_laplace(1.0)
            assert isinstance(val, int)

    def test_discrete_laplace_zero_mean(self):
        """Discrete Laplace 均值应接近 0（对称分布）。"""
        api = DPApi(namespace="ks-dlaplace-mean", random_state=42)
        samples = [api._sample_discrete_laplace(2.0) for _ in range(N_SAMPLES)]
        mean = sum(samples) / len(samples)
        # Discrete Laplace variance = 2*scale*(scale+1) for large scale
        var = 2.0 * 2.0 * 3.0  # scale=2: var ≈ 12
        se = math.sqrt(var / N_SAMPLES)
        assert abs(mean) < 5 * se, (
            f"Discrete Laplace mean too far from 0: {mean:.6f}"
        )

    def test_discrete_laplace_symmetry(self):
        """Discrete Laplace 分布关于 0 对称：P(X=k) ≈ P(X=-k)。"""
        api = DPApi(namespace="ks-dlaplace-sym", random_state=42)
        samples = [api._sample_discrete_laplace(1.5) for _ in range(N_SAMPLES)]
        # Count positive and negative values
        pos = sum(1 for s in samples if s > 0)
        neg = sum(1 for s in samples if s < 0)
        # Should be roughly equal (within 5% tolerance)
        total = pos + neg
        if total > 0:
            ratio = pos / total
            assert 0.45 < ratio < 0.55, (
                f"Discrete Laplace asymmetry: pos_ratio={ratio:.3f}"
            )


class TestCalibratedNoise:
    """验证端到端查询的噪声校准正确性 / End-to-End Noise Calibration Validation.

    通过大量重复查询的经验方差与理论方差对比，
    验证整个 DP 管线（参数解析 → 敏感度计算 → 噪声注入）的正确性。
    """

    def test_count_noise_scale_laplace(self):
        """count 查询在 Laplace 机制下的噪声 scale 应为 1/epsilon。"""
        api = DPApi(namespace="ks-count-scale", random_state=42, epsilon_total=200000.0)
        epsilon = 1.0
        expected_scale = 1.0 / epsilon  # = 1.0

        # Run many count queries and check empirical variance
        n_queries = 10_000
        results = []
        for _ in range(n_queries):
            r = api.count([1, 1, 1, 1, 1], epsilon=epsilon, mechanism="laplace",
                          return_details=True)
            results.append(r.value)

        # True count = 5; noise ~ Laplace(0, 1.0)
        # Var(Laplace(0,b)) = 2*b^2 = 2*1.0 = 2.0
        empirical_var = np.var(results)
        expected_var = 2.0 * expected_scale ** 2
        # Allow 10% tolerance
        assert abs(empirical_var - expected_var) / expected_var < 0.10, (
            f"Count noise variance mismatch: empirical={empirical_var:.4f}, "
            f"expected={expected_var:.4f}"
        )
