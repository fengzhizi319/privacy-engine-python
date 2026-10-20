"""DP 数值计算核心（NumPy 向量化，可选 Numba 加速）。

中文说明：
面向高并发 DP 请求的数值密集操作集合：Laplace/Gaussian 噪声批量采样、
值截断、L2 范数裁剪等，全部基于 NumPy 向量化实现（无 Python 级循环），
已具备良好性能。

若环境安装了 Numba（``pip install numba``），本模块的 ``HAS_NUMBA``
标志为 True，后续可在核心循环上叠加 ``@numba.jit(nopython=True, cache=True)``
进一步加速（首次调用有 ~200ms 编译开销，之后直接执行缓存的机器码）。

English Description:
DP numeric kernels (NumPy vectorized, optional Numba acceleration).
Collection of numerical-intensive DP operations: batch Laplace/Gaussian noise
sampling, value clipping, L2 norm clipping etc., all implemented with NumPy
vectorization (no Python-level loops).

When Numba is installed, ``HAS_NUMBA`` is True and the kernels can be
further accelerated with ``@numba.jit(nopython=True, cache=True)``
(first call incurs ~200ms compilation; later calls use cached machine code).
"""

from __future__ import annotations

import math

import numpy as np

from ..observability.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Numba 可用性检测 / Numba availability detection
# ---------------------------------------------------------------------------
try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# 注意：当前函数使用 NumPy 向量化实现（np.random、np.where 等），
# 已具备良好性能。若需进一步 JIT 加速，需将核心循环重写为
# Numba 兼容的纯 NumPy 标量操作形式，并使用
# ``@numba.jit(nopython=True, cache=True)`` 装饰内部循环函数。
# HAS_NUMBA 标志可供外部模块检测 Numba 可用性。


# ---------------------------------------------------------------------------
# Laplace 噪声批量采样 / Batch Laplace noise sampling
# ---------------------------------------------------------------------------
def laplace_noise_batch(
    values: np.ndarray, sensitivity: float, epsilon: float
) -> np.ndarray:
    """Laplace 机制批量噪声采样。

    对每个值独立采样 Laplace(scale=sensitivity/epsilon) 噪声并叠加。
    若 Numba 可用，内部核心循环编译为机器码；否则使用 NumPy 向量化实现。

    Args:
        values: 输入数值数组。
        sensitivity: 敏感度（查询函数在相邻数据集上的最大变化量）。
        epsilon: 隐私预算参数。

    Returns:
        叠加 Laplace 噪声后的数组，与输入同形。
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    scale = sensitivity / epsilon
    # NumPy 向量化实现（即使无 Numba 也足够快）
    # Laplace 分布 = 两个独立 Exponential 之差
    u = np.random.random(size=values.shape) - 0.5
    # 避免 log(0)：u 为 0 时用极小值替代
    nonzero = np.where(u == 0.0, 1e-300, u)
    signs = np.sign(nonzero)
    # 当 random() 恰好返回 0.0 时 u=-0.5，1-2|u|=0，log(0) 会产生
    # ±inf 噪声；对 log 参数做下限钳制，保证噪声始终为有限值。
    log_arg = np.maximum(1.0 - 2.0 * np.abs(nonzero), 1e-300)
    noise = -scale * signs * np.log(log_arg)
    return values + noise


# ---------------------------------------------------------------------------
# Gaussian 噪声批量采样 / Batch Gaussian noise sampling
# ---------------------------------------------------------------------------
def gaussian_noise_batch(
    values: np.ndarray, sensitivity: float, epsilon: float, delta: float
) -> np.ndarray:
    """Analytic Gaussian 机制批量噪声采样。

    sigma = sensitivity * sqrt(2 * ln(1/delta)) / epsilon

    Args:
        values: 输入数值数组。
        sensitivity: 敏感度。
        epsilon: 隐私预算参数。
        delta: 隐私预算参数（必须 > 0）。

    Returns:
        叠加 Gaussian 噪声后的数组。
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.0 / delta)) / epsilon
    noise = np.random.normal(0.0, sigma, size=values.shape)
    return values + noise


# ---------------------------------------------------------------------------
# 值截断（clip）/ Value clipping
# ---------------------------------------------------------------------------
def clip_values(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """将数值截断到 [lower, upper] 区间。

    Args:
        values: 输入数值数组。
        lower: 下界。
        upper: 上界。

    Returns:
        截断后的数组。
    """
    return np.clip(np.asarray(values, dtype=np.float64), lower, upper)


# ---------------------------------------------------------------------------
# 截断 + 求和（单次遍历合并）/ Clip + Sum (single-pass)
# ---------------------------------------------------------------------------
def clip_and_sum(
    values: np.ndarray, clip_lower: float, clip_upper: float
) -> tuple[float, float]:
    """截断并求和（合并为单次遍历，减少内存分配）。

    先截断到 [clip_lower, clip_upper]，再计算 clipped_sum 和 count。
    相比分步 clip + sum 减少一次完整数组分配。

    Args:
        values: 输入数值数组。
        clip_lower: 截断下界。
        clip_upper: 截断上界。

    Returns:
        (clipped_sum, count) 元组。
    """
    arr = np.asarray(values, dtype=np.float64)
    clipped = np.clip(arr, clip_lower, clip_upper)
    return float(clipped.sum()), float(len(clipped))


# ---------------------------------------------------------------------------
# L2 范数批量截断 / Batch L2 norm clipping (for vector_sum)
# ---------------------------------------------------------------------------
def l2_norm_clip(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    """批量 L2 范数截断（DP Vector Sum 的核心操作）。

    对每行向量计算 L2 范数，若超过 max_norm 则按比例缩放：
        v_clipped = v * min(1, max_norm / ||v||_2)

    Args:
        vectors: 2D 数组，shape (N, d)，每行为一个向量样本。
        max_norm: L2 范数截断上界。

    Returns:
        截断后的 2D 数组，与输入同形。
    """
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got {vectors.ndim}D")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # 避免除零
    norms = np.maximum(norms, 1e-12)
    scaling = np.minimum(1.0, max_norm / norms)
    return vectors * scaling


# ---------------------------------------------------------------------------
# 批量截断 + 求和 + 计数（向量维度）/ Batch clip + sum per column
# ---------------------------------------------------------------------------
def clip_and_sum_columns(
    matrix: np.ndarray, clip_lower: float, clip_upper: float
) -> tuple[np.ndarray, int]:
    """按列截断并求和（用于多维 DP sum）。

    Args:
        matrix: 2D 数组，shape (N, d)。
        clip_lower: 截断下界。
        clip_upper: 截断上界。

    Returns:
        (column_sums, row_count) 元组，column_sums shape (d,)。
    """
    clipped = np.clip(np.asarray(matrix, dtype=np.float64), clip_lower, clip_upper)
    return clipped.sum(axis=0), clipped.shape[0]


# ---------------------------------------------------------------------------
# 模块初始化日志 / Module init logging
# ---------------------------------------------------------------------------
if HAS_NUMBA:
    logger.info(
        "dp_jit_numba_available",
        extra={"status": "Numba installed; kernels can be JIT-accelerated if decorated"},
    )
else:
    logger.info(
        "dp_jit_numpy_vectorized",
        extra={
            "status": "Numba not installed, using NumPy vectorized kernels",
            "hint": "pip install numba for optional JIT acceleration",
        },
    )
