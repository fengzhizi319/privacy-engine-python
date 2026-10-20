"""差分隐私（Differential Privacy, DP）API 实现 / Differential Privacy Primitive API Implementation.

中文说明：
支持拉普拉斯（Laplace）机制与高斯（Gaussian）机制下的 count、sum、mean 聚合。
每次调用都会先向 BudgetAccountant 申请消耗 (epsilon, delta) 预算，再对真实结果
注入符合差分隐私要求的 calibrated 噪声。
支持显式 clip_lower / clip_upper，确保 sum/mean 的敏感度在 seeing data 之前即已确定。

English Description:
Provides calibrated noise injection mechanisms (Laplace, Analytic Gaussian, Discrete Laplace)
for count, sum, mean, histogram, vector sum (DP-SGD), and private SQL groupby.
Privacy budgets (epsilon, delta) are accounted and consumed via BudgetAccountant before operations.
Supports explicit numeric clipping to bound sensitivity in advance of seeing actual data.

扩展能力 / Key Features:
- 向量化/零拷贝：PyArrow Table Metadata (to_arrow), NumPy C-contiguous acceleration, scipy.sparse optimization.
- 会计与审计：Rényi DP accountant (RDP), HMAC-SHA256 tamper-proof audit logger.
- 级联与侧车协议：User-Level DP contribution bounding, Tau-Thresholding GroupBy, gRPC (packed=true) & REST endpoints.
"""

from __future__ import annotations

import math
import random
import secrets
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from ..observability.logging_config import get_logger
from ..observability.metrics import DP_DURATION, DP_QUERIES_TOTAL, observe_duration
from .budget import BudgetRegistry, PrivacyBudgetExhausted, RDPAccountant, default_registry
from .data_adapters import _is_sparse_matrix, _to_2d_numpy_array, extract_values

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# Module-level structured logger for DP query and budget events
logger = get_logger(__name__)


class Mechanism(str, Enum):
    """差分隐私噪声机制枚举 / DP Noise Mechanism Enum.

    继承 str 保证与字符串的向后兼容性：Mechanism.LAPLACE == "laplace" 为 True。
    IDE 自动补全 + 静态类型检查，避免裸字符串拼写错误。
    """

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"


class AggregationType(str, Enum):
    """差分隐私聚合类型枚举 / DP Aggregation Type Enum.

    继承 str 保证与字符串的向后兼容性：AggregationType.COUNT == "count" 为 True。
    IDE 自动补全 + 静态类型检查，避免裸字符串拼写错误。
    """

    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    HISTOGRAM = "histogram"
    VECTOR_SUM = "vector_sum"
    VECTOR_MEAN = "vector_mean"


@dataclass
# 分布式无噪流式累加器，用于 MapReduce 场景下各 Worker 节点局部聚合后在 Master 节点合并并统一注入 DP 噪声
class Accumulator:
    """分布式无噪流式累加器 / Distributed Noise-Free Streaming Accumulator.

    中文说明：
    各个分布式 Worker 节点在本地数据块（Chunk）上执行 Map 阶段局部累加；
    导出的 Accumulator 不含噪声，可在 Master/Combine 节点通过 + 运算合并，
    最终由 master 统一调用 finalize_dp 只注入一次差分隐私噪声。

    English Description:
    Used in distributed/federated MapReduce setups. Workers compute unnoisy local aggregates
    on chunk subsets and serialize them. Master nodes combine accumulators via '+' operator
    and call finalize_dp once to inject privacy noise.

    执行步骤 / Execution Steps:
    1. Worker 节点在本地调用 `create_accumulator` 对局部 Chunk 累加 sum/count/histogram.
       (Worker computes unnoisy local count/sum/histogram via create_accumulator)
    2. 调用 `serialize()` 将无噪中间状态打包序列化为二进制 bytes.
       (Serialize local accumulator state to UTF-8 bytes for transmission)
    3. Master 节点接收 bytes 调用 `deserialize()` 并通过 `+` 运算（`__add__`）合并累加器.
       (Master deserializes and merges accumulators via add operator)
    4. 统一调用 `finalize_dp()` 对合并后的无噪总计注入一次 DP 噪声.
       (Master calls finalize_dp to inject noise once into global total)
    """

    count: float = 0.0
    sum: float = 0.0
    histogram: dict[Any, float] = field(default_factory=dict)
    sensitivity: float = 1.0

    def __add__(self, other: Accumulator) -> Accumulator:
        # 合并两个 Accumulator 的局部统计量（count/sum 相加，histogram 合并，sensitivity 取最大值）
        """合并两个 Accumulator 对象的局部统计量（加法结合律）/ Combine two Accumulators (Additive Law)."""
        # Reject non-Accumulator operands to preserve type safety
        if not isinstance(other, Accumulator):
            return NotImplemented
        # Shallow-copy left histogram, then merge right histogram bins (sum counts for shared keys)
        new_hist = dict(self.histogram)
        for k, v in other.histogram.items():
            new_hist[k] = new_hist.get(k, 0.0) + v
        # Return merged Accumulator: counts/sums add up, sensitivity takes the worst-case (max)
        return Accumulator(
            count=self.count + other.count,
            sum=self.sum + other.sum,
            histogram=new_hist,
            sensitivity=max(self.sensitivity, other.sensitivity),
        )

    def serialize(self) -> bytes:
        # 将无噪累加器状态序列化为 JSON 编码的 UTF-8 字节串，用于跨网络传输
        """将无噪累加器序列化为 JSON 编码的 UTF-8 字节串 / Serialize noise-free accumulator to JSON bytes."""
        import json

        # Pack all accumulator fields into a plain dict for JSON serialization
        data = {
            "count": self.count,
            "sum": self.sum,
            "histogram": self.histogram,
            "sensitivity": self.sensitivity,
        }
        # Encode to JSON string then to UTF-8 bytes for network transmission
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, b: bytes) -> Accumulator:
        # 从 UTF-8 字节串反序列化重建 Accumulator 实例，用于 Master 节点接收 Worker 数据
        """从 UTF-8 字节串反序列化重建 Accumulator 实例 / Deserialize bytes back to Accumulator."""
        import json

        # Decode UTF-8 bytes and parse JSON back into a plain dict
        data = json.loads(b.decode("utf-8"))
        # Reconstruct Accumulator with explicit float casts to ensure type consistency
        return cls(
            count=float(data["count"]),
            sum=float(data["sum"]),
            # Histogram keys may have been stringified by JSON; cast values back to float
            histogram={k: float(v) for k, v in data.get("histogram", {}).items()},
            # Fall back to default sensitivity=1.0 if not present (backward compatibility)
            sensitivity=float(data.get("sensitivity", 1.0)),
        )


@dataclass
# 差分隐私计算结果的结构化包装，包含带噪值、噪声机制、预算消耗和置信区间等元数据
class DPResult:
    """差分隐私计算结果及结构化元数据包装 / Differential Privacy Result Dataclass.

    Attributes:
        value: 带噪声的聚合结果（标量 float 或多维数组）/ Noisy query value (scalar or array).
        noise_mechanism: 使用的 DP 噪声机制（"laplace" 或 "gaussian"）/ Noise mechanism ("laplace" or "gaussian").
        noise_scale: 噪声分布的核心参数（Laplace b 或 Gaussian sigma）/ Noise distribution scale parameter.
            当 clip 边界是从数据中推断出来时，该值由数据推导（= (max-min)/ε），
            直接返回会把数据范围泄露给调用方，此时置 None。
            (None when clip bounds were data-inferred, to avoid leaking the data range.)
        epsilon_spent: 本次运算所消耗的 epsilon 预算 / Epsilon budget consumed.
        delta_spent: 本次运算所消耗的 delta 预算 / Delta budget consumed.
        confidence_interval: 置信区间 (lower, upper)，根据 confidence_level 计算 / Two-sided confidence interval.
            同样地，clip 边界为数据推断时置信区间亦由数据推导，置 None。
            (None when clip bounds were data-inferred.)
    """

    value: Any
    noise_mechanism: str
    noise_scale: float | list[float] | np.ndarray | None
    epsilon_spent: float
    delta_spent: float
    confidence_interval: tuple[float, float] | list[tuple[float, float]] | dict[Any, tuple[float, float]] | None

    def to_arrow(self) -> Any:
        # 将 DPResult 转换为附带 DP 隐私 Metadata 的 PyArrow Table，支持零拷贝列式传输
        """将 DPResult 包装转换为附带 DP 隐私 Metadata 的 PyArrow Table /
        Export to PyArrow Table with Embedded DP Metadata.

        执行步骤 / Execution Steps:
        1. 提取 DPResult 的隐私元数据，构造 JSON 可序列化字典。
           (Extract DP metadata fields into JSON-serializable dictionary)
        2. 将元数据编码为字节串保存至 Schema 字典 Key `b"dp_metadata"` 中。
           (Encode metadata into Schema bytes under key b"dp_metadata")
        3. 根据 `value` 的数据类型（标量/数组/字典）构造对应的 PyArrow Array 列。
           (Construct PyArrow Arrays according to value data type)
        4. 组合创建 `pyarrow.Table` 并将 Metadata 嵌入 Schema 导出。
           (Construct pyarrow.Table and attach schema metadata)
        """
        import json

        import pyarrow as pa

        # Ensure all fields are JSON-serializable native Python types (no numpy objects)
        def _to_jsonable(obj: Any) -> Any:
            """递归将 DPResult 字段转换为 JSON 可序列化类型（消除 numpy 对象）。"""
            # Convert numpy arrays to plain Python lists
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Convert numpy scalar types to Python float
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            # Recursively convert dict keys to str and values to JSON-safe types
            if isinstance(obj, dict):
                return {str(k): _to_jsonable(v) for k, v in obj.items()}
            # Recursively convert list/tuple elements
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(x) for x in obj]
            return obj

        # Build the DP metadata dictionary with mechanism, scale, budget, and CI info
        meta = {
            "noise_mechanism": self.noise_mechanism,
            "noise_scale": _to_jsonable(self.noise_scale),
            "epsilon_spent": self.epsilon_spent,
            "delta_spent": self.delta_spent,
            "confidence_interval": _to_jsonable(self.confidence_interval),
        }
        # Encode metadata as JSON bytes under the schema key b"dp_metadata"
        custom_metadata = {b"dp_metadata": json.dumps(meta).encode("utf-8")}

        # Construct PyArrow Table based on the runtime type of self.value
        if isinstance(self.value, (int, float)):
            # Scalar: wrap in a single-element array
            arr = pa.array([self.value])
            table = pa.Table.from_arrays([arr], names=["dp_value"])
        elif isinstance(self.value, np.ndarray):
            # N-D array: directly convert to PyArrow array
            arr = pa.array(self.value)
            table = pa.Table.from_arrays([arr], names=["dp_value"])
        elif isinstance(self.value, dict):
            # Dict (e.g. histogram): split into category keys and noisy value columns
            keys = pa.array(list(self.value.keys()))
            vals = pa.array(list(self.value.values()))
            table = pa.Table.from_arrays([keys, vals], names=["category", "dp_value"])
        else:
            # Fallback: stringify the value
            arr = pa.array([str(self.value)])
            table = pa.Table.from_arrays([arr], names=["dp_value"])

        # Merge with any existing schema metadata (preserve prior metadata if present)
        existing_meta = table.schema.metadata or {}
        merged_meta = {**existing_meta, **custom_metadata}
        return table.replace_schema_metadata(merged_meta)


# User-Level DP 贡献限定：按 user_id 下采样，每个用户最多保留 max_contributions 条记录以控制敏感度
def _bound_contributions(arr: np.ndarray, user_ids: Sequence[Any], max_contributions: int) -> np.ndarray:
    """User-Level DP 贡献限定：按 user_id 对数据进行下采样限制 / User-Level DP Contribution Bounding.

    中文说明：每个 user_id 最多保留 max_contributions 项记录，控制个体最大敏感度放缩。
    English Description: Subsamples records per user_id up to max_contributions, preventing sensitivity amplification.

    执行步骤 / Execution Steps:
    1. 校验输入数据与 user_ids 长度一致性，以及 max_contributions 必须为正整数。
       (Validate array and user_ids length alignment and positive max_contributions limit)
    2. 使用字典记录每个 user_id 已遍历和保留的记录条数。
       (Track per-user occurrence count using a hashmap)
    3. 顺序扫描数据，仅保留每个 user_id 前 max_contributions 次出现的样本索引。
       (Scan elements sequentially, keeping indices up to max_contributions)
    4. 返回切片截断后的 ndarray 子集。
       (Return downsampled array slice)
    """
    # Validate that the data array and user_ids have the same number of records
    if len(arr) != len(user_ids):
        raise ValueError("user_ids length must match values length")
    # Each user must contribute at least 1 record; 0 or negative is meaningless
    if max_contributions <= 0:
        raise ValueError("max_contributions must be positive")
    from collections import defaultdict

    # Track how many records have been retained for each user_id so far
    user_counts = defaultdict(int)
    indices = []
    # Sequential scan: retain the first max_contributions occurrences per user_id
    for idx, uid in enumerate(user_ids):
        if user_counts[uid] < max_contributions:
            user_counts[uid] += 1
            indices.append(idx)
    # Fancy-index the original array to produce the downsampled subset
    return arr[indices]


# ──────────────────────────────────────────────
# 截断算子 / Clipping operators
# ──────────────────────────────────────────────

def clip_value(value: float, bound: float) -> float:
    """将数值截断至 [-bound, bound]（与 Go ``dp.ClipValue`` 对齐）。"""
    if value > bound:
        return bound
    if value < -bound:
        return -bound
    return value


def clip_value_range(value: float, lower: float, upper: float) -> float:
    """将数值截断至 [lower, upper]（与 Go ``dp.ClipValueRange`` 对齐）。"""
    if value > upper:
        return upper
    if value < lower:
        return lower
    return value


def clip_l2_norm(vector: Sequence[float], max_norm: float) -> list[float]:
    """将向量缩放到 L2 范数不超过 max_norm（与 Go ``dp.ClipL2Norm`` 对齐）。

    返回新的 list，不修改入参；max_norm <= 0 或空向量时原样复制。
    非有限 max_norm / 分量返回同维 NaN 向量，与 Go 的 invalidVector 口径一致。
    """
    values = [float(v) for v in vector]
    if max_norm <= 0 or not values:
        return values
    if not math.isfinite(max_norm) or not all(math.isfinite(v) for v in values):
        return [math.nan] * len(values)
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0.0 or norm <= max_norm:
        return values
    scale = max_norm / norm
    return [value * scale for value in values]


# 根据噪声机制（Laplace/Gaussian）和置信水平计算双边置信区间
def compute_confidence_interval(
    val: float, noise_scale: float, mechanism: str, confidence_level: float = 0.95
) -> tuple[float, float]:
    """计算单值 DP 结果的双边置信区间 / Compute Two-Sided Confidence Interval for DP Estimate.

    执行步骤 / Execution Steps:
    1. 校验 noise_scale 或输入值有效性，若为 0 或 NaN 则返回点估计本身。
       (Validate noise_scale; if 0 or NaN, return the point estimate as interval)
    2. 计算显著性水平 alpha = 1.0 - confidence_level。
       (Compute significance level alpha = 1 - confidence_level)
    3. 根据噪声机制计算边际误差 margin：
       - Laplace 机制：Margin = -b * ln(alpha)
       - Gaussian 机制：Margin = NormalDist().inv_cdf(1 - alpha/2) * sigma
       (Calculate margin error based on analytical noise distribution tails)
    4. 返回区间 (val - margin, val + margin)。
       (Return boundary tuple (val - margin, val + margin))
    """
    # Degenerate case: zero noise scale or NaN value → return point estimate as a zero-width interval
    if noise_scale <= 0.0 or math.isnan(val):
        return (val, val)
    # Clamp alpha to avoid log(0); alpha = 1 - confidence_level (e.g. 0.05 for 95% CI)
    alpha = max(1e-12, 1.0 - confidence_level)
    if mechanism == Mechanism.LAPLACE:
        # Laplace tail bound: Margin = -b * ln(alpha), where b is the noise scale parameter
        margin = -noise_scale * math.log(alpha)
    else:
        # Gaussian tail bound: Margin = z_{1-alpha/2} * sigma (two-sided)
        p = 1.0 - alpha / 2.0
        # Compute the inverse CDF (quantile function) of the standard normal distribution
        z = statistics.NormalDist().inv_cdf(p)
        margin = z * noise_scale
    return (val - margin, val + margin)


# 对带噪 DP 结果应用后处理（非负截断和/或整数取整），不消耗额外隐私预算
def _apply_post_processing(
    val: Any, round_int: bool = False, clip_non_negative: bool = False
) -> Any:
    """对带噪聚合结果应用 DP 后处理 / Apply Differential Privacy Post-Processing.

    中文说明：后处理封闭性定理（Post-processing Theorem）保证对 DP 输出做确定性变换不增加隐私开销。
    English Description: By Post-processing Immunity, deterministic post-processing incurs no extra privacy budget cost.

    执行步骤 / Execution Steps:
    1. 判定输入为 ndarray 或标量数值。
       (Check input data type, ndarray or scalar)
    2. 若 clip_non_negative 为 True，则将负数值截断为 0.0。
       (If clip_non_negative, truncate negative noisy values to 0.0)
    3. 若 round_int 为 True，则对数值应用四舍五入取整。
    4. 返回后处理转换后的数值。
    """
    # Handle numpy ndarray: copy to avoid mutating the original array
    if isinstance(val, np.ndarray):
        res = val.copy()
        # Clip negative values to 0.0 (useful for count/sum that should be non-negative)
        if clip_non_negative:
            res = np.clip(res, 0.0, None)
        # Round to nearest integer (for count queries where fractional results are meaningless)
        if round_int:
            res = np.round(res)
        return res
    # Handle scalar numeric values (int or float)
    if isinstance(val, (int, float)):
        res = float(val)
        # Enforce non-negativity by clamping to 0.0
        if clip_non_negative:
            res = max(0.0, res)
        # Round to nearest integer and cast back to float for type consistency
        if round_int:
            res = float(round(res))
        return res
    # Unsupported type: return as-is (e.g. dict for histogram results)
    return val


# 使用 Balle & Wang (ICML'18) 解析高斯机制，通过二分查找校准满足 (eps,delta)-DP 的高斯噪声标准差 sigma
def calibrate_analytic_gaussian(epsilon: float, delta: float, sensitivity: float, tol: float = 1e-12) -> float:
    """使用 Balle & Wang (ICML'18) 提出的解析高斯机制计算噪声的标准差 sigma。

    执行步骤：
    1. 校验敏感度，若敏感度为 0 则直接返回噪声标准差 0.0。
    2. 定义标准正态累积分布函数 phi(t) 与辅助方程 delta(v)。
    3. 利用二分查找在 [0, 1] 找到临界点 v_star，使得 Delta(v_star) == 0。
    4. 根据敏感度与求解得到的比例值计算并返回校准后的高斯噪声标准差 sigma。
    """
    # Zero sensitivity means the query output is unchanged by any single record → no noise needed
    if sensitivity == 0.0:
        return 0.0
    if sensitivity < 0.0:
        raise ValueError(f"sensitivity must be non-negative, got {sensitivity}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be strictly positive, got {epsilon}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be strictly between 0 and 1, got {delta}")

    # Standard normal CDF: phi(t) = 0.5 * (1 + erf(t / sqrt(2)))
    def phi(t: float) -> float:
        """标准正态分布累积分布函数 / Standard normal CDF."""
        return 0.5 * (1.0 + math.erf(float(t) / math.sqrt(2.0)))

    # Case A helper: delta as a function of the ratio s when delta >= delta_threshold
    def case_a(eps: float, s: float) -> float:
        """解析高斯 Case A 辅助函数（delta >= delta_threshold 时使用）。"""
        return phi(math.sqrt(eps * s)) - math.exp(eps) * phi(-math.sqrt(eps * (s + 2.0)))

    # Case B helper: delta as a function of the ratio s when delta < delta_threshold
    def case_b(eps: float, s: float) -> float:
        """解析高斯 Case B 辅助函数（delta < delta_threshold 时使用）。"""
        return phi(-math.sqrt(eps * s)) - math.exp(eps) * phi(-math.sqrt(eps * (s + 2.0)))

    # Doubling trick: exponentially expand the search interval [s_inf, s_sup] until predicate is met
    def doubling_trick(predicate_stop, s_inf: float, s_sup: float) -> tuple[float, float]:
        """倍增法搜索满足 predicate_stop 的 s 区间 / Doubling-trick interval search."""
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    # Binary search: narrow down the interval to find s_final within tolerance
    def binary_search(predicate_stop, predicate_left, s_inf: float, s_sup: float) -> float:
        """二分查找在 [s_inf, s_sup] 内定位满足 predicate_stop 的 s 值。"""
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid  # s_final is in the left half
            else:
                s_inf = s_mid  # s_final is in the right half
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    # Compute the threshold delta that separates Case A and Case B
    delta_thr = case_a(epsilon, 0.0)
    if delta == delta_thr:
        # Exact threshold: alpha = 1 (boundary case)
        alpha = 1.0
    else:
        if delta > delta_thr:
            # Case A: use case_a function for doubling + binary search
            def predicate_stop_dt(s):
                return case_a(epsilon, s) >= delta
            def function_s_to_delta(s):
                return case_a(epsilon, s)
            def predicate_left_bs(s):
                return function_s_to_delta(s) > delta
        else:
            # Case B: use case_b function for doubling + binary search
            def predicate_stop_dt(s):
                return case_b(epsilon, s) <= delta
            def function_s_to_delta(s):
                return case_b(epsilon, s)
            def predicate_left_bs(s):
                return function_s_to_delta(s) < delta

        # Stop binary search when the delta gap is within tolerance
        def predicate_stop_bs(s):
            return abs(function_s_to_delta(s) - delta) <= tol

        # Phase 1: find an interval containing s_final via doubling
        s_inf, s_sup = doubling_trick(predicate_stop_dt, 0.0, 1.0)
        # Phase 2: binary search within [s_inf, s_sup] to pinpoint s_final
        s_final = binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)

        # Convert the optimal ratio s_final to the noise multiplier alpha
        if delta > delta_thr:
            alpha = math.sqrt(1.0 + s_final / 2.0) - math.sqrt(s_final / 2.0)
        else:
            alpha = math.sqrt(1.0 + s_final / 2.0) + math.sqrt(s_final / 2.0)

    # Final sigma = alpha * sensitivity / sqrt(2 * epsilon)
    return alpha * sensitivity / math.sqrt(2.0 * epsilon)


# 安全随机数生成器：生产环境使用密码学安全 RNG（不可预测），测试时支持确定性 PRNG 模式
class SecureRandom(random.Random):
    """一个安全的随机数生成器。

    在生产环境中默认使用 secrets.SystemRandom() 以确保密码学安全（不可预测）；
    在测试或显式调用 seed(x) 时，切换到基类 random.Random 的确定性伪随机数生成，
    保证单元测试的可复现性与确定性边界校验。
    """

    def __init__(self):
        """初始化 SecureRandom：默认使用 OS 熵源的密码学安全 RNG。"""
        super().__init__()
        self._system_rng = secrets.SystemRandom()
        # Flag: False → use system RNG (production); True → use deterministic PRNG (testing)
        self._seeded = False

    # 设置随机种子：提供种子时切换到确定性 PRNG 模式，否则恢复密码学安全模式
    def seed(self, a=None, version=2) -> None:
        """设置随机种子或恢复密码学安全模式。

        Args:
            a: 随机种子；为 None 时恢复 OS 熵源安全 RNG。
            version: 种子格式版本（兼容 random.Random 接口）。
        """
        # If a seed is provided, switch to deterministic mode for reproducible tests
        if a is not None:
            super().seed(a, version=version)
            self._seeded = True
        else:
            # Reset to cryptographically secure mode
            self._seeded = False

    # 生成 [0,1) 随机浮点：确定性模式用 Mersenne Twister，生产模式用 OS 安全 RNG
    def random(self) -> float:
        """生成 [0, 1) 均匀分布随机浮点数。

        确定性模式使用 Mersenne Twister PRNG；生产模式使用 OS 安全 RNG。
        """
        # In deterministic mode, use the base class Mersenne Twister PRNG
        if self._seeded:
            return super().random()
        # In production mode, delegate to OS-backed secure RNG
        return self._system_rng.random()

    # 生成高斯分布随机值：确定性模式用基类实现，生产模式用密码学安全实现
    def gauss(self, mu: float, sigma: float) -> float:
        """生成高斯分布 N(mu, sigma^2) 随机浮点数。

        确定性模式使用基类 Box-Muller 实现；生产模式使用 OS 安全 RNG。
        """
        # Deterministic Gaussian sampling for reproducible unit tests
        if self._seeded:
            return super().gauss(mu, sigma)
        # Cryptographically secure Gaussian sampling for production
        return self._system_rng.gauss(mu, sigma)


# 差分隐私计算核心接口：封装 Laplace/Gaussian 噪声采样与预算扣减，提供 count/sum/mean/histogram 等聚合方法
class DPApi:
    """差分隐私计算接口。

    封装了 Laplace/Gaussian 采样与预算扣减逻辑，提供 count/sum/mean/histogram 等聚合方法。
    支持可选的 RDPAccountant 集成，在使用 Gaussian 机制时自动记录 Rényi DP 消耗，
    通过 ``rdp_accountant.get_epsilon()`` 可获得比 Basic Composition 更紧致的预算上界。

    Attributes:
        budget: 当前命名空间对应的 BudgetAccountant 实例。
        rng: 私有随机数生成器，用于噪声采样。
        rdp_accountant: 可选的 RDPAccountant 实例，Gaussian 机制下自动跟踪 RDP 消耗。
    """

    def __init__(
        self,
        namespace: str = "default",
        random_state: int | None = None,
        registry: BudgetRegistry | None = None,
        epsilon_total: float | None = None,
        delta_total: float | None = None,
        window_seconds: float | None = None,
        rdp_accountant: RDPAccountant | None = None,
    ):
        # 初始化 DPApi：创建关联 namespace 的预算账户和安全随机数生成器
        """初始化 DPApi。

        Args:
            namespace: 命名空间，用于关联隐私预算账户。
            random_state: 可选随机种子，用于可复现测试。与旧参数 ``seed`` 等价。
            registry: 可选的 BudgetRegistry 注册表，未提供时使用全局 default_registry。
            epsilon_total: 可选 epsilon 总预算；仅在对应 BudgetAccountant 尚未创建时生效。
            delta_total: 可选 delta 总预算；仅在对应 BudgetAccountant 尚未创建时生效。
            window_seconds: 可选预算重置窗口（秒）；仅在对应 BudgetAccountant 尚未创建时生效。
            rdp_accountant: 可选的 RDPAccountant 实例；使用 Gaussian 机制时自动记录
                Rényi DP 消耗，提供更紧致的组合预算上界。
        """
        self.registry = registry or default_registry
        # Create a BudgetAccountant tied to the given namespace for (epsilon, delta) tracking
        self.budget = self.registry.get_or_create(
            namespace,
            epsilon_total=epsilon_total,
            delta_total=delta_total,
            window_seconds=window_seconds,
        )
        # Optional RDPAccountant for tighter Rényi DP composition tracking (Gaussian only)
        self.rdp_accountant = rdp_accountant
        logger.info(
            "dp_api_initialized",
            extra={
                "namespace": self.budget.namespace,
                "epsilon_total": self.budget.epsilon_total,
                "delta_total": self.budget.delta_total,
                "seeded": random_state is not None,
                "rdp_enabled": rdp_accountant is not None,
            },
        )
        # Create a SecureRandom instance (cryptographic RNG by default)
        self.rng = SecureRandom()
        # If a seed is provided, switch to deterministic PRNG mode for reproducibility
        if random_state is not None:
            self.rng.seed(random_state)

    @classmethod
    def from_seed(cls, namespace: str = "default", seed: int | None = None) -> DPApi:
        # 兼容旧接口的工厂方法：通过 seed 创建 DPApi 实例
        """兼容旧构造方式：通过 seed 创建 DPApi 实例。"""
        # Delegate to __init__ with random_state parameter
        return cls(namespace=namespace, random_state=seed)

    def _sample_laplace(self, scale: float) -> float:
        # 使用逆变换采样从 Laplace(0, scale) 分布中采样一个随机值
        """从拉普拉斯分布 Laplace(0, scale) 中采样一个随机值。

        使用逆变换采样（inverse transform sampling）。
        """
        # Inverse transform sampling: U ~ Uniform(-0.5, 0.5)
        u = self.rng.random() - 0.5
        # Determine sign of the sample
        sign = -1.0 if u < 0 else 1.0
        # Apply inverse CDF: X = -b * sign * ln(1 - 2|u|) with domain safety check.
        # 注意：max(1e-12, ...) 将采样尾部截断在 |noise| <= ln(1e12) * scale ≈ 27.6 * scale，
        # 相对理想 Laplace 分布存在极轻微失真（尾部概率 ~1e-12 被截除）。该截断为防御
        # log(0) 的数值安全措施，对 (epsilon, 0)-DP 保证无实质影响，故保留此行为。
        return -scale * sign * math.log(max(1e-12, 1.0 - 2.0 * abs(u)))

    def _sample_gaussian(self, sigma: float) -> float:
        # 从 N(0, sigma^2) 高斯分布中采样一个随机值
        """从高斯分布 N(0, sigma^2) 中采样一个随机值。"""
        # Use the underlying RNG's Gaussian sampler (Box-Muller or system RNG)
        return self.rng.gauss(0.0, sigma)

    def _sample_discrete_laplace(self, scale: float) -> int:
        # 在整数格 Z 上采样 Discrete Laplace（Two-sided Geometric）随机数
        """采样 Discrete Laplace (Two-sided Geometric) 随机数在整数格 ℤ 上。"""
        # Zero or negative scale => degenerate distribution at 0
        if scale <= 0:
            return 0
        # Compute geometric distribution parameter p = 1 - exp(-1/scale)
        p = 1.0 - math.exp(-1.0 / scale)
        # Guard against floating-point underflow: for very large scale, exp(-1/scale)
        # rounds to 1.0 making p == 0.0, which would cause log(1-p) == 0 and a
        # ZeroDivisionError below. Fall back to the continuous Laplace sampler.
        if p <= 0.0:
            return int(round(self._sample_laplace(scale)))
        # Sample two independent geometric random variables via inverse CDF
        u1 = self.rng.random()
        u2 = self.rng.random()
        # g1, g2 ~ Geometric(p) using floor(log(1-U)/log(1-p))
        g1 = math.floor(math.log(1.0 - max(1e-12, u1)) / math.log(1.0 - p))
        g2 = math.floor(math.log(1.0 - max(1e-12, u2)) / math.log(1.0 - p))
        # Difference of two geometric RVs gives discrete Laplace on Z
        return int(g1 - g2)

    def _sample_isotropic_noise(
        self, d: int, mechanism: Mechanism, scale: float
    ) -> np.ndarray:
        """预分配数组并按维度采样同构噪声，避免列表推导式逐次分配。"""
        noises = np.empty(d, dtype=np.float64)
        if mechanism == Mechanism.LAPLACE:
            for i in range(d):
                noises[i] = self._sample_laplace(scale)
        else:
            for i in range(d):
                noises[i] = self._sample_gaussian(scale)
        return noises

    @staticmethod
    def _mean_delta_method_ci(
        final_val: float,
        noisy_sum: float,
        noisy_count: float,
        sum_scale: float,
        count_scale: float,
        mechanism: Mechanism,
        confidence_level: float,
    ) -> tuple[float, tuple[float, float]]:
        """使用 Delta 方法计算均值估计的等效噪声 scale 与置信区间。"""
        # Variance of sum noise: Var(Laplace) = 2b^2, Var(Gaussian) = sigma^2
        var_sum = (2.0 * sum_scale**2) if mechanism == Mechanism.LAPLACE else (sum_scale**2)
        # Variance of count noise
        var_count = (2.0 * count_scale**2) if mechanism == Mechanism.LAPLACE else (count_scale**2)
        # Delta method: Var(S/C) ~ Var(S)/C^2 + (S^2 * Var(C))/C^4
        var_mean = (var_sum / (noisy_count**2)) + ((noisy_sum**2 * var_count) / (noisy_count**4))
        # Standard deviation of the mean estimate
        std_mean = math.sqrt(max(0.0, var_mean))
        # Effective scale: divide by sqrt(2) for Laplace to match Gaussian equivalent
        eff_scale = std_mean / math.sqrt(2.0) if mechanism == Mechanism.LAPLACE else std_mean
        ci = compute_confidence_interval(final_val, eff_scale, mechanism, confidence_level)
        return eff_scale, ci

    @staticmethod
    def _compute_noise_scale(
        sensitivity: float, epsilon: float, delta: float, mechanism: Mechanism
    ) -> float:
        """根据机制统一计算噪声 scale（Laplace b 或 Gaussian sigma）。"""
        if mechanism == Mechanism.LAPLACE and epsilon > 0:
            return sensitivity / epsilon
        return calibrate_analytic_gaussian(epsilon, delta, sensitivity)

    @staticmethod
    def _clip_values(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        # 向量化截断输入值到 [lower, upper] 区间，使用 NumPy 加速大规模数据处理
        """对输入值做截断，将其限制在 [lower, upper] 区间内。

        使用 NumPy 向量化 ``np.clip`` 以提升大规模数据处理效率。
        """
        # Convert to float64 ndarray for consistent numeric clipping
        arr = np.asarray(values, dtype=np.float64)
        # Empty array: return as-is without clipping
        if arr.size == 0:
            return arr
        # Vectorized clip to [lower, upper] bounds
        return np.clip(arr, lower, upper)

    def _resolve_clip_bounds(
        self,
        values: np.ndarray,
        clip_lower: float | None,
        clip_upper: float | None,
        mechanism: str,
    ) -> tuple[float, float, bool]:
        # 解析 clip 上下界：Gaussian 必须显式提供，Laplace 允许数据推断（带警告）
        """解析并返回 clip 上下界及"是否从数据推断"标志。

        Gaussian 机制必须提供显式 clip 区间；Laplace 在未提供时允许使用数据推断
        的区间作为向后兼容，但会发出警告。

        Returns:
            (lower, upper, inferred)：inferred 为 True 表示边界由数据 min/max 推导，
            调用方不得把由该边界计算的 noise_scale / confidence_interval 返回给调用者
            （否则通过 sensitivity = (max - min) 反推泄露数据范围）。
        """
        # Both bounds explicitly provided: validate ordering and return as floats
        if clip_lower is not None and clip_upper is not None:
            if clip_lower > clip_upper:
                raise ValueError("clip_lower must be <= clip_upper")
            return float(clip_lower), float(clip_upper), False

        # Gaussian mechanism requires explicit clip bounds (no data-dependent inference)
        if mechanism == Mechanism.GAUSSIAN:
            raise ValueError(
                "clip_lower and clip_upper are required for Gaussian mechanism"
            )

        # Laplace backward compatibility: infer [min, max] from data when clip not provided.
        # WARNING: strict DP requires clip to be data-independent; production should set bounds explicitly.
        if values.size > 0:
            lower = float(values.min())
            upper = float(values.max())
        else:
            lower, upper = 0.0, 0.0
        logger.warning(
            "clip_bounds_inferred_from_data",
            extra={
                "recommendation": "Set clip_lower/clip_upper explicitly for production DP guarantees.",
            },
        )
        return lower, upper, True

    def _validate_inputs(
        self,
        epsilon: float,
        delta: float = 0.0,
        confidence_level: float | None = None,
    ) -> None:
        """统一校验公共输入参数，确保失败时快速抛出清晰错误。

        Args:
            epsilon: 隐私预算 epsilon，必须 > 0。
            delta: 隐私预算 delta，必须 >= 0。
            confidence_level: 置信区间水平，若有值则必须在 (0, 1) 区间。
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if delta < 0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        if confidence_level is not None and not (0.0 < confidence_level < 1.0):
            raise ValueError(
                f"confidence_level must be in (0, 1), got {confidence_level}"
            )


    def _validate_mechanism(self, mechanism: str, delta: float) -> str:
        """校验 mechanism 与 delta 参数，返回规范化后的 mechanism 字符串。

        内部使用 Mechanism 枚举做校验，返回 .value 保证下游兼容性（Prometheus 标签、序列化等）。
        """
        # Normalize mechanism string to lowercase for case-insensitive comparison
        mechanism = mechanism.lower()
        # Map to Mechanism enum for validation; return .value (plain str) for downstream compatibility
        if mechanism == Mechanism.LAPLACE:
            return Mechanism.LAPLACE.value
        if mechanism == Mechanism.GAUSSIAN:
            # Gaussian mechanism requires strictly positive delta for finite sigma calibration
            if delta <= 0:
                raise ValueError("delta must be positive for Gaussian mechanism")
            return Mechanism.GAUSSIAN.value
        raise ValueError(f"mechanism must be 'laplace' or 'gaussian', got '{mechanism}'")

    def _notify_rdp(self, mechanism: str, sigma: float, sensitivity: float) -> None:
        """Gaussian 机制下自动通知 RDPAccountant 记录 Rényi DP 消耗。

        仅在 rdp_accountant 已注入且机制为 Gaussian 时生效；Laplace 机制无 RDP 语义，静默跳过。
        通过回调钩子模式解耦 DPApi 与 RDPAccountant，避免在每个查询方法中硬编码 if 分支。

        Args:
            mechanism: 当前查询使用的噪声机制（已规范化为小写字符串）。
            sigma: 高斯噪声标准差（Analytic Gaussian calibration 输出）。
            sensitivity: 查询的全局敏感度。
        """
        if self.rdp_accountant is not None and mechanism == Mechanism.GAUSSIAN and sigma > 0:
            self.rdp_accountant.record_gaussian(sigma=sigma, sensitivity=sensitivity)

    def _sample_count_noise(self, epsilon: float, delta: float, mechanism: str) -> float:
        # 采样 count/histogram 所需的 DP 噪声（L1/L2 敏感度均为 1）
        """采样 count / histogram bin 所需的 DP 噪声。

        count 与直方图分桶的 L1 敏感度为 1（Laplace）或 L2 敏感度为 1（Gaussian）。
        """
        if mechanism == Mechanism.LAPLACE:
            # Laplace scale b = sensitivity/epsilon = 1/epsilon for count (L1 sens=1)
            return self._sample_laplace(1.0 / epsilon)
        # Analytic Gaussian calibration: sigma = alpha(eps,delta) * sens / sqrt(2*eps), sens=1
        sigma = calibrate_analytic_gaussian(epsilon, delta, 1.0)
        return self._sample_gaussian(sigma)

    def _sample_sum_noise(
        self, sensitivity: float, epsilon: float, delta: float, mechanism: str
    ) -> float:
        # 采样 sum 所需的 DP 噪声，敏感度由调用方提供的 sensitivity 参数决定
        """采样 sum 所需的 DP 噪声。

        sum 的敏感度由调用方提供的 sensitivity 决定。
        """
        if mechanism == Mechanism.LAPLACE:
            # Laplace scale b = sensitivity / epsilon for bounded sum
            scale = sensitivity / epsilon if epsilon > 0 else 0.0
            return self._sample_laplace(scale)
        # Zero sensitivity means no noise needed regardless of mechanism
        if sensitivity == 0:
            return 0.0
        # Analytic Gaussian: calibrate sigma to the given sensitivity
        sigma = calibrate_analytic_gaussian(epsilon, delta, sensitivity)
        return self._sample_gaussian(sigma)

    def _execute_scalar_query(
        self,
        aggregation: str,
        true_value: float,
        epsilon: float,
        delta: float,
        mechanism: str,
        noise_sampler: Any,
        noise_scale: float,
        return_details: bool,
        confidence_level: float,
        round_int: bool = False,
        clip_non_negative: bool = False,
        sensitivity: float = 1.0,
    ) -> float | DPResult:
        """执行标量 DP 查询的公共模板：预算扣减、噪声采样、后处理、置信区间。

        集中处理单次 (epsilon, delta) 标量查询的公共流程，减少 noisy_*/chunked_*
        方法中的重复代码。
        """
        try:
            self.budget.spend(epsilon, delta)
        except PrivacyBudgetExhausted:
            logger.error(
                "privacy_budget_exhausted",
                extra={
                    "aggregation": aggregation,
                    "namespace": self.budget.namespace,
                    "epsilon": epsilon,
                    "delta": delta,
                },
            )
            raise

        DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation=aggregation).inc()
        noise = noise_sampler()
        raw_val = float(true_value) + noise
        final_val = _apply_post_processing(
            raw_val, round_int=round_int, clip_non_negative=clip_non_negative
        )

        logger.info(
            "dp_scalar_query_completed",
            extra={
                "aggregation": aggregation,
                "mechanism": mechanism,
                "epsilon": epsilon,
                "delta": delta,
                "noise_scale": noise_scale,
            },
        )

        # Auto-track RDP consumption for Gaussian mechanism via callback hook
        self._notify_rdp(mechanism, noise_scale, sensitivity)

        if return_details:
            ci = compute_confidence_interval(
                final_val, noise_scale, mechanism, confidence_level
            )
            return DPResult(
                value=final_val,
                noise_mechanism=mechanism,
                noise_scale=noise_scale,
                epsilon_spent=epsilon,
                delta_spent=delta,
                confidence_interval=ci,
            )
        return final_val

    def _execute_histogram_query(
        self,
        true_counts: dict[Any, float],
        epsilon: float,
        delta: float,
        mechanism: str,
        noise_scale: float,
        return_details: bool,
        confidence_level: float,
        round_int: bool = False,
        clip_non_negative: bool = True,
    ) -> dict[Any, float] | DPResult:
        """执行直方图 DP 查询的公共模板：单次预算为所有分桶加噪。

        直方图各分桶互斥，联合敏感度为 1，因此只扣减一次预算。
        """
        try:
            self.budget.spend(epsilon, delta)
        except PrivacyBudgetExhausted:
            logger.error(
                "privacy_budget_exhausted",
                extra={
                    "aggregation": "histogram",
                    "namespace": self.budget.namespace,
                    "epsilon": epsilon,
                    "delta": delta,
                },
            )
            raise

        DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="histogram").inc()
        res_dict: dict[Any, float] = {}
        ci_dict: dict[Any, tuple[float, float]] = {}
        for c in true_counts:
            noise = self._sample_count_noise(epsilon, delta, mechanism)
            raw_val = float(true_counts[c]) + noise
            final_val = _apply_post_processing(
                raw_val, round_int=round_int, clip_non_negative=clip_non_negative
            )
            res_dict[c] = final_val
            if return_details:
                ci_dict[c] = compute_confidence_interval(
                    final_val, noise_scale, mechanism, confidence_level
                )

        logger.info(
            "dp_histogram_query_completed",
            extra={
                "aggregation": "histogram",
                "mechanism": mechanism,
                "epsilon": epsilon,
                "delta": delta,
                "num_bins": len(true_counts),
            },
        )

        # Auto-track RDP consumption for Gaussian mechanism (histogram joint sensitivity = 1)
        self._notify_rdp(mechanism, noise_scale, sensitivity=1.0)

        if return_details:
            return DPResult(
                value=res_dict,
                noise_mechanism=mechanism,
                noise_scale=noise_scale,
                epsilon_spent=epsilon,
                delta_spent=delta,
                confidence_interval=ci_dict,
            )
        return res_dict

    def count(
        self,
        values: Any,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        column: str | None = None,
        party: str | None = None,
        round_int: bool = False,
        clip_non_negative: bool = True,
        return_details: bool = False,
        confidence_level: float = 0.95,
        user_ids: Sequence[Any] | None = None,
        max_contributions: int = 1,
        discrete: bool = False,
    ) -> float | DPResult:
        # 执行差分隐私计数查询：提取数据 → 扣减预算 → 计算真实计数 → 注入 DP 噪声 → 后处理
        """差分隐私计数查询 / Differentially Private Count Query.

        执行步骤 / Execution Steps:
        1. 校验 mechanism 与 delta 参数有效性。
           (Validate mechanism parameter and ensure positive delta if Gaussian)
        2. 使用 `extract_values` 提取数据为连续 ndarray 或 sparse 矩阵。
           (Extract input data into contiguous C-ndarray or scipy.sparse matrix)
        3. 若传入 user_ids，通过 `_bound_contributions` 做 User-Level DP 贡献限定并调整敏感度。
           (Bound user contributions per user_id if user_ids is provided)
        4. 向 BudgetAccountant 申请扣减 (epsilon, delta) 隐私预算。
           (Consume privacy budget (epsilon, delta) via BudgetAccountant)
        5. 计算数据集无噪真实计数（count_nonzero 或 nnz）。
           (Compute true raw count non-zero elements or nnz)
        6. 根据概率分布采样差分隐私噪声（支持 Discrete Laplace 整数采样）。
           (Sample noise from Laplace, Gaussian or Discrete Laplace distribution)
        7. 聚合计算带噪数值并应用后处理选项（非负截断、四舍五入）。
           (Add noise to true count and apply post-processing like non-negative clip or rounding)
        8. 若 return_details 为 True，计算置信区间并封装导出 DPResult。
           (If return_details, compute confidence intervals and package into DPResult)

        Args:
            values: 输入数据 (Input data, supports List, Array, Series, DataFrame, Arrow Table, Sparse matrix).
            epsilon: 隐私预算 epsilon (Epsilon privacy budget > 0).
            delta: 隐私预算 delta (Delta budget, must be > 0 if Gaussian).
            mechanism: 噪声机制 (Noise mechanism, "laplace" or "gaussian").
            column: 目标列名 (Target column name for tabular inputs).
            party: SecretFlow 参与方标识 (Party identifier for SecretFlow HDataFrame).
            round_int: 是否四舍五入取整 (Whether to round noisy result to integer).
            clip_non_negative: 是否截断保证非负 (Whether to truncate negative noisy result to 0).
            return_details: 是否返回 DPResult 结构 (Whether to return structured DPResult).
            confidence_level: 置信区间水平 (Confidence interval level, default 0.95).
            user_ids: 用户标识符序列 (User IDs for User-Level DP).
            max_contributions: 每个用户最多贡献条数 (Max records retained per user ID).
            discrete: 是否使用离散拉普拉斯机制 (Whether to use Discrete Laplace on integer lattice Z).

        Returns:
            Union[float, DPResult]: 带噪计数值或 DPResult 包装结构 / Noisy count float or DPResult dataclass.
        """
        # Step 1: Validate mechanism name and delta compatibility
        mechanism = self._validate_mechanism(mechanism, delta)
        # Step 1.5: Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="count", mechanism=mechanism):
            # Step 2: Extract input data into contiguous ndarray or scipy.sparse matrix
            arr = extract_values(values, column=column, party=party)

            # Step 3: If user_ids provided, bound per-user contributions (User-Level DP)
            if user_ids is not None and isinstance(arr, np.ndarray) and arr.size > 0:
                arr = _bound_contributions(arr, user_ids, max_contributions)
                # Each user contributes at most max_contributions records => sensitivity scales accordingly
                sensitivity = float(max_contributions)
            else:
                # Default: adding/removing one record changes count by at most 1
                sensitivity = 1.0

            # Step 4: Compute true (noise-free) count of non-zero / non-empty elements
            if _is_sparse_matrix(arr):
                # Sparse matrix: nnz gives the number of stored (non-zero) entries in O(1)
                true_count = float(arr.nnz)
            elif not isinstance(arr, np.ndarray) or arr.size == 0:
                # Empty or non-array input => count is zero
                true_count = 0.0
            elif arr.dtype.kind in ("i", "f", "u", "b"):
                # Numeric/boolean dtype: use vectorized count_nonzero for speed
                true_count = float(np.count_nonzero(arr))
            else:
                # Object/string dtype: fall back to Python-level truthiness check
                true_count = float(sum(1 for v in arr if v))

            # Step 5: Sample DP noise calibrated to the sensitivity
            if discrete and mechanism == Mechanism.LAPLACE:
                # Discrete Laplace on integer lattice Z for exact integer output
                scale = sensitivity / epsilon if epsilon > 0 else 0.0
                noise = float(self._sample_discrete_laplace(scale))
            else:
                # Continuous Laplace or Analytic Gaussian noise
                noise = self._sample_sum_noise(sensitivity, epsilon, delta, mechanism)

            # Consume privacy budget only after noise sampling succeeds, so failures do not exhaust budget
            self.budget.spend(epsilon, delta)

            # Increment Prometheus metrics counter for observability
            DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="count").inc()

            # Step 6: Add noise to true count and apply post-processing (clip/round)
            raw_val = true_count + noise
            final_val = _apply_post_processing(
                raw_val, round_int=round_int, clip_non_negative=clip_non_negative
            )

            # Compute the noise distribution scale parameter for confidence interval reporting
            noise_scale = (
                (sensitivity / epsilon)
                if mechanism == Mechanism.LAPLACE
                else calibrate_analytic_gaussian(epsilon, delta, sensitivity)
            )
            # Auto-track RDP consumption for Gaussian mechanism via callback hook
            self._notify_rdp(mechanism, noise_scale, sensitivity)
            # Step 8: If return_details, compute confidence interval and wrap in DPResult
            if return_details:
                if discrete and mechanism == Mechanism.LAPLACE:
                    # Discrete Laplace (Two-sided Geometric) 的精确尾部公式：
                    # P(|X| >= t) = 2 λ^t / (1 + λ)，其中 λ = e^{-1/b}；
                    # 反解得 margin = b * ln(2 / (alpha * (1 + λ)))。
                    # 不能套用连续 Laplace 的 -b*ln(alpha)，否则 CI 偏窄。
                    if noise_scale > 0.0 and not math.isnan(final_val):
                        alpha = max(1e-12, 1.0 - confidence_level)
                        lam = math.exp(-1.0 / noise_scale)
                        margin = noise_scale * math.log(2.0 / (alpha * (1.0 + lam)))
                        ci = (final_val - margin, final_val + margin)
                    else:
                        ci = (final_val, final_val)
                else:
                    ci = compute_confidence_interval(
                        final_val, noise_scale, mechanism, confidence_level
                    )
                return DPResult(
                    value=final_val,
                    noise_mechanism=mechanism,
                    noise_scale=noise_scale,
                    epsilon_spent=epsilon,
                    delta_spent=delta,
                    confidence_interval=ci,
                )
            # Otherwise return the scalar noisy value directly
            return final_val

    def sum(
        self,
        values: Any,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        clip_lower: float | None = None,
        clip_upper: float | None = None,
        column: str | None = None,
        party: str | None = None,
        round_int: bool = False,
        clip_non_negative: bool = False,
        return_details: bool = False,
        confidence_level: float = 0.95,
        user_ids: Sequence[Any] | None = None,
        max_contributions: int = 1,
    ) -> float | DPResult:
        # 执行差分隐私求和查询：提取数据 → clip 截断 → 扣减预算 → 注入 calibrated 噪声
        """差分隐私求和查询 / Differentially Private Sum Query.

        执行步骤 / Execution Steps:
        1. 校验 mechanism 与 delta 参数有效性。
           (Validate mechanism and delta parameters)
        2. 提取数据并视需要按 user_ids 执行 User-Level DP 贡献限定。
           (Extract inputs and bound user contributions if user_ids provided)
        3. 解析 clip 截断上下界 [clip_lower, clip_upper]，计算敏感度 sensitivity = (upper - lower) * user_scale。
           (Resolve clipping bounds [clip_lower, clip_upper] and calculate sensitivity)
        4. 对原始数值使用 NumPy 向量化做 `_clip_values` 裁剪（稀疏矩阵保持疏性累加）。
           (Perform vector numeric clipping on values to bound maximum per-element change)
        5. 扣减 (epsilon, delta) 隐私预算并采样对应的 Laplace / Gaussian 噪声注入。
           (Deduct privacy budget and draw noise calibrated to sensitivity / epsilon)
        6. 应用后处理选项，根据需要封装为 DPResult 并计算置信区间。
           (Apply post-processing, compute confidence interval and package result)

        Args:
            values: 输入数据 (Input values).
            epsilon: 隐私预算 epsilon (Epsilon privacy budget > 0).
            delta: 隐私预算 delta (Delta budget, must be > 0 if Gaussian).
            mechanism: 噪声机制 (Noise mechanism, "laplace" or "gaussian").
            clip_lower: 截断下界 (Lower numeric clipping bound).
            clip_upper: 截断上界 (Upper numeric clipping bound).
            column: 目标列名 (Target column name).
            party: SecretFlow 参与方标识 (Party identifier).
            round_int: 是否四舍五入取整 (Whether to round result).
            clip_non_negative: 是否截断保证非负 (Whether to truncate negative result).
            return_details: 是否返回 DPResult 结构 (Whether to return DPResult dataclass).
            confidence_level: 置信区间水平 (Confidence level, default 0.95).
            user_ids: 用户标识符序列 (User IDs for User-Level DP).
            max_contributions: 每个用户最多贡献条数 (Max records retained per user).

        Returns:
            Union[float, DPResult]: 带噪求和值或 DPResult 包装结构 / Noisy sum float or DPResult dataclass.
        """
        # Step 1: Validate mechanism name and delta compatibility
        mechanism = self._validate_mechanism(mechanism, delta)
        # Step 1.5: Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="sum", mechanism=mechanism):
        
                # Step 2: Extract input data into contiguous ndarray or scipy.sparse matrix
                arr = extract_values(values, column=column, party=party)
                # If user_ids provided, apply User-Level DP contribution bounding
                if user_ids is not None and isinstance(arr, np.ndarray) and arr.size > 0:
                    arr = _bound_contributions(arr, user_ids, max_contributions)
                    # Each user contributes at most max_contributions => sensitivity multiplier
                    user_scale = float(max_contributions)
                else:
                    # Default: single-record contribution
                    user_scale = 1.0
        
                # Step 3: Resolve clip bounds and compute true sum on clipped data
                if _is_sparse_matrix(arr):
                    # Sparse path: resolve clip bounds from min/max of stored values
                    lower, upper, bounds_inferred = self._resolve_clip_bounds(
                        np.array([arr.min(), arr.max()]) if arr.nnz > 0 else np.array([]),
                        clip_lower,
                        clip_upper,
                        mechanism,
                    )
                    # Clip stored (explicit) values to [lower, upper] to bound per-element
                    # sensitivity while preserving sparsity. Implicit zeros are structural
                    # and contribute 0 to the sum. Without this clip an out-of-range stored
                    # value would make the true sensitivity unbounded while the noise is
                    # calibrated only for (upper - lower), breaking the DP guarantee.
                    clipped_data = np.clip(arr.data, lower, upper)
                    true_sum = float(clipped_data.sum()) if arr.nnz > 0 else 0.0
                else:
                    # Dense path: resolve clip bounds (may infer from data for Laplace)
                    lower, upper, bounds_inferred = self._resolve_clip_bounds(
                        arr, clip_lower, clip_upper, mechanism
                    )
                    # Clip values to [lower, upper] to bound per-element sensitivity
                    clipped = self._clip_values(arr, lower, upper)
                    true_sum = float(clipped.sum()) if clipped.size > 0 else 0.0
        
                # Step 4: Compute global sensitivity = (upper - lower) * user_scale
                sensitivity = max(0.0, upper - lower) * user_scale
        
                # Step 5: Consume privacy budget and draw calibrated noise
                self.budget.spend(epsilon, delta)
                # Increment Prometheus metrics counter
                DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="sum").inc()
                # Sample noise proportional to sensitivity / epsilon
                noise = self._sample_sum_noise(sensitivity, epsilon, delta, mechanism)
                # Add noise to the true (clipped) sum
                raw_val = true_sum + noise
                # Apply post-processing: non-negative clip and/or integer rounding
                final_val = _apply_post_processing(
                    raw_val, round_int=round_int, clip_non_negative=clip_non_negative
                )
        
                # Compute noise scale parameter for confidence interval reporting
                noise_scale = (
                    (sensitivity / epsilon)
                    if (mechanism == Mechanism.LAPLACE and epsilon > 0)
                    else calibrate_analytic_gaussian(epsilon, delta, sensitivity)
                )
                # Auto-track RDP consumption for Gaussian mechanism via callback hook
                self._notify_rdp(mechanism, noise_scale, sensitivity)
                # Step 6: If return_details, compute CI and wrap in DPResult
                if return_details:
                    if bounds_inferred:
                        # 边界由数据 [min, max] 推断：noise_scale = (max-min)/ε 与 CI 均由数据推导，
                        # 返回给调用方会泄露数据范围（值域泄露），因此置 None 不上报。
                        return DPResult(
                            value=final_val,
                            noise_mechanism=mechanism,
                            noise_scale=None,
                            epsilon_spent=epsilon,
                            delta_spent=delta,
                            confidence_interval=None,
                        )
                    ci = compute_confidence_interval(
                        final_val, noise_scale, mechanism, confidence_level
                    )
                    return DPResult(
                        value=final_val,
                        noise_mechanism=mechanism,
                        noise_scale=noise_scale,
                        epsilon_spent=epsilon,
                        delta_spent=delta,
                        confidence_interval=ci,
                    )
                return final_val
        
    def mean(
        self,
        values: Any,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        clip_lower: float | None = None,
        clip_upper: float | None = None,
        min_count: float = 5.0,
        column: str | None = None,
        party: str | None = None,
        round_int: bool = False,
        clip_non_negative: bool = False,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> float | DPResult:
        # 执行差分隐私均值查询：组合定理拆分预算，分别计算 noisy_count 和 noisy_sum 后求比值
        """差分隐私均值查询。

        执行步骤：
        1. 提取样本数据并精准判定样本总行数 N（稀疏矩阵使用 shape[0] 避免 nnz 语义错误）。
        2. 若样本为空，直接返回 0.0 或零元 DPResult 结构。
        3. 应用预算组合定理，将总预算 (epsilon, delta) 均匀拆分为两半 (eps/2, delta/2)。
        4. 独立计算带噪样本计数 `noisy_count`；若 `noisy_count < min_count`，触发保护防发散并返回 0.0。
        5. 独立计算带噪样本求和 `noisy_sum`。
        6. 计算比率估计值 `raw_val = noisy_sum / noisy_count` 并应用后处理。
        7. 使用 Delta 方法一阶泰勒展开估计比率估计量的组合方差：
           Var(mean) ≈ Var(sum)/(count^2) + (sum^2 * Var(count))/(count^4)
        8. 根据组合方差导出等效噪声 scale，计算双边置信区间并封装 DPResult。
        """
        # Step 1: Extract data and determine total sample count N
        arr = extract_values(values, column=column, party=party)
        if _is_sparse_matrix(arr):
            # Sparse matrix: use shape[0] (number of rows), not nnz which counts stored entries
            n_samples = arr.shape[0]
        elif isinstance(arr, np.ndarray):
            # Dense ndarray: use number of rows (samples), not total element count.
            # For a 2-D array each row is one sample; arr.size would over-count
            # columns and produce a wildly wrong mean denominator.
            n_samples = arr.shape[0]
        else:
            # Fallback for list/iterable inputs
            n_samples = len(arr) if hasattr(arr, "__len__") else 0

        # Step 2: Empty dataset => return 0.0 immediately (no budget consumed)
        if n_samples == 0:
            if return_details:
                # 未执行任何子查询，预算分文未花，如实上报 0.0
                return DPResult(
                    value=0.0,
                    noise_mechanism=mechanism,
                    noise_scale=0.0,
                    epsilon_spent=0.0,
                    delta_spent=0.0,
                    confidence_interval=(0.0, 0.0),
                )
            return 0.0

        # Validate mechanism after empty check (avoid unnecessary validation)
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="mean", mechanism=mechanism):
        
                # Step 3: Apply composition theorem: split budget equally for count and sum
                eps_sub = epsilon / 2.0
                delta_sub = delta / 2.0
        
                # Step 4: Compute noisy count using ones-vector (each element counts as 1)
                count_res = self.count(
                    np.ones(n_samples, dtype=np.float64),
                    eps_sub,
                    delta_sub,
                    mechanism,
                    return_details=True,
                    confidence_level=confidence_level,
                )
                noisy_count = count_res.value
                count_scale = count_res.noise_scale
        
                # Guard against divergence: if noisy_count too small, ratio estimate is unstable
                if noisy_count < min_count or noisy_count <= 0.0:
                    if return_details:
                        # Only the count sub-query (eps_sub) has been spent at this point;
                        # the sum sub-query was never executed, so report the actual spend.
                        return DPResult(
                            value=0.0,
                            noise_mechanism=mechanism,
                            noise_scale=0.0,
                            epsilon_spent=eps_sub,
                            delta_spent=delta_sub,
                            confidence_interval=(0.0, 0.0),
                        )
                    return 0.0
        
                # Step 5: Compute noisy sum on the clipped data
                sum_res = self.sum(
                    arr,
                    eps_sub,
                    delta_sub,
                    mechanism,
                    clip_lower,
                    clip_upper,
                    return_details=True,
                    confidence_level=confidence_level,
                )
                noisy_sum = sum_res.value
                sum_scale = sum_res.noise_scale
        
                # Step 6: Compute ratio estimate and apply post-processing
                DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="mean").inc()
                raw_val = noisy_sum / noisy_count
                final_val = _apply_post_processing(
                    raw_val, round_int=round_int, clip_non_negative=clip_non_negative
                )
        
                # Step 7: Delta method variance estimation for ratio estimator
                if return_details:
                    if sum_scale is None or count_scale is None:
                        # sum 子查询的 clip 边界由数据推断（noise_scale 已被置 None）：
                        # 等效 scale 与 CI 均由数据推导，返回会泄露数据范围，故置 None。
                        return DPResult(
                            value=final_val,
                            noise_mechanism=mechanism,
                            noise_scale=None,
                            epsilon_spent=epsilon,
                            delta_spent=delta,
                            confidence_interval=None,
                        )
                    eff_scale, ci = self._mean_delta_method_ci(
                        final_val, noisy_sum, noisy_count, sum_scale, count_scale, mechanism, confidence_level
                    )
                    return DPResult(
                        value=final_val,
                        noise_mechanism=mechanism,
                        noise_scale=eff_scale,
                        epsilon_spent=epsilon,
                        delta_spent=delta,
                        confidence_interval=ci,
                    )
                return final_val
        
    def batch_count(
        self,
        data: Any,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        round_int: bool = False,
        clip_non_negative: bool = True,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> np.ndarray | DPResult:
        # 多列/2D 矩阵批量 DP 计数：按列独立加噪，预算按组合定理乘以列数
        """多列 / 2D 矩阵批量差分隐私计数 (按列 axis=0 计算)。

        执行步骤：
        1. 提取 2D 矩阵与确定列数 `num_cols`（针对 scipy.sparse 走 CSC 指针 $O(1)$ 高效统计）。
        2. 按列数累计扣减预算：`epsilon_total = num_cols * epsilon`, `delta_total = num_cols * delta`。
        3. 对每列非零元素独立采样注入 DP 噪声（支持向量化处理）。
        4. 应用后处理选项，根据需要按列封装各列置信区间并返回 DPResult。
        """
        # Step 1: Validate mechanism and convert input to 2D matrix
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="count", mechanism=mechanism):
                matrix = _to_2d_numpy_array(data)
        
                # Determine true per-column non-zero counts
                if _is_sparse_matrix(matrix):
                    # Sparse: convert to CSC format for O(1) per-column nnz via index pointer diff
                    csc = matrix.tocsc()
                    num_cols = csc.shape[1]
                    # np.diff(indptr) gives the number of stored entries per column
                    true_counts = np.diff(csc.indptr).astype(np.float64)
                else:
                    num_cols = matrix.shape[1] if matrix.ndim == 2 else 1
                    if matrix.dtype.kind in ("i", "f", "u", "b"):
                        # Numeric/boolean: vectorized count_nonzero along axis=0
                        true_counts = np.count_nonzero(matrix, axis=0).astype(np.float64)
                    else:
                        # Object dtype: per-column Python-level counting
                        true_counts = np.array(
                            [np.count_nonzero(matrix[:, i]) for i in range(num_cols)],
                            dtype=np.float64,
                        )
        
                # Step 2: Deduct budget for all columns at once (composition theorem)
                self.budget.spend(epsilon * num_cols, delta * num_cols)
                # Increment metrics counter by num_cols (one query per column)
                DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="count").inc(num_cols)
        
                # Compute noise scale for confidence interval reporting
                noise_scale = (
                    1.0 / epsilon
                    if mechanism == Mechanism.LAPLACE
                    else calibrate_analytic_gaussian(epsilon, delta, 1.0)
                )
                # Step 3: Sample independent noise for each column
                noises = np.array(
                    [self._sample_count_noise(epsilon, delta, mechanism) for _ in range(num_cols)]
                )
                # Add noise to true counts
                raw_vals = true_counts + noises
                # Step 4: Apply post-processing (non-negative clip / integer rounding)
                final_vals = _apply_post_processing(
                    raw_vals, round_int=round_int, clip_non_negative=clip_non_negative
                )
        
                # If return_details, compute per-column confidence intervals
                if return_details:
                    ci = [
                        compute_confidence_interval(
                            float(final_vals[i]), noise_scale, mechanism, confidence_level
                        )
                        for i in range(num_cols)
                    ]
                    return DPResult(
                        value=final_vals,
                        noise_mechanism=mechanism,
                        noise_scale=noise_scale,
                        epsilon_spent=epsilon * num_cols,
                        delta_spent=delta * num_cols,
                        confidence_interval=ci,
                    )
                return final_vals
        
    def batch_sum(
        self,
        data: Any,
        epsilon: float,
        clip_lower: float | Sequence[float],
        clip_upper: float | Sequence[float],
        delta: float = 0.0,
        mechanism: str = "laplace",
        round_int: bool = False,
        clip_non_negative: bool = False,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> np.ndarray | DPResult:
        # 多列/2D 矩阵批量 DP 求和：每列独立 clip 并以各自敏感度加噪
        """多列 / 2D 矩阵批量差分隐私求和 (按列 axis=0 计算)。

        执行步骤：
        1. 转换数据为 2D 矩阵并解析每列的 `clip_lower` 与 `clip_upper` 边界向量。
        2. 计算每列独立的敏感度 `sensitivities = uppers - lowers`。
        3. 对矩阵实施广播或逐列裁剪累加（针对 sparse 避免全量 Dense 展平）。
        4. 按 `num_cols * (epsilon, delta)` 扣减预算并逐列生成对应敏感度的噪声。
        5. 返回每列带噪求和值（支持返回多维 noise_scales 列表和置信区间组）。
        """
        # Step 1: Validate mechanism and convert input to 2D matrix
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="sum", mechanism=mechanism):
                matrix = _to_2d_numpy_array(data)
        
                # Determine number of columns
                if _is_sparse_matrix(matrix):
                    csc = matrix.tocsc()
                    num_cols = csc.shape[1]
                else:
                    num_cols = matrix.shape[1] if matrix.ndim == 2 else 1
        
                # Step 2: Parse per-column clip bounds (scalar or per-column vector)
                if isinstance(clip_lower, (list, tuple, np.ndarray)):
                    # Per-column lower bounds provided as sequence
                    lowers = np.asarray(clip_lower, dtype=np.float64)
                else:
                    # Scalar lower bound broadcast to all columns
                    lowers = np.full(num_cols, float(clip_lower), dtype=np.float64)
        
                if isinstance(clip_upper, (list, tuple, np.ndarray)):
                    # Per-column upper bounds provided as sequence
                    uppers = np.asarray(clip_upper, dtype=np.float64)
                else:
                    # Scalar upper bound broadcast to all columns
                    uppers = np.full(num_cols, float(clip_upper), dtype=np.float64)
        
                # Validate that clip bound vectors match the number of columns
                if len(lowers) != num_cols or len(uppers) != num_cols:
                    raise ValueError(f"clip_lower/upper length must match num_cols ({num_cols})")
        
                # Step 3: Clip values and compute true per-column sums
                if _is_sparse_matrix(matrix):
                    # Sparse path: extract each column, clip, and accumulate
                    csc = matrix.tocsc()
                    num_cols = csc.shape[1]
                    true_sums = np.zeros(num_cols, dtype=np.float64)
                    for i in range(num_cols):
                        col = np.asarray(csc[:, i].todense()).ravel().astype(np.float64)
                        clipped_col = np.clip(col, lowers[i], uppers[i])
                        true_sums[i] = clipped_col.sum()
                else:
                    # Dense path: vectorized broadcast clipping and column-wise sum
                    clipped = np.clip(matrix, lowers, uppers)
                    true_sums = clipped.sum(axis=0).astype(np.float64)
        
                # Step 4: Compute per-column sensitivity = upper - lower for each column
                sensitivities = np.maximum(0.0, uppers - lowers)
        
                # Deduct budget for all columns (composition theorem)
                self.budget.spend(epsilon * num_cols, delta * num_cols)
                DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="sum").inc(num_cols)
        
                # Step 5: Sample independent noise for each column with its own sensitivity
                noises = np.array(
                    [
                        self._sample_sum_noise(float(sensitivities[i]), epsilon, delta, mechanism)
                        for i in range(num_cols)
                    ]
                )
                # Add noise to true per-column sums
                raw_vals = true_sums + noises
                # Apply post-processing (non-negative clip / integer rounding)
                final_vals = _apply_post_processing(
                    raw_vals, round_int=round_int, clip_non_negative=clip_non_negative
                )
        
                # Compute per-column noise scale for confidence interval reporting
                noise_scales = [
                    (sensitivities[i] / epsilon)
                    if (mechanism == Mechanism.LAPLACE and epsilon > 0)
                    else calibrate_analytic_gaussian(epsilon, delta, sensitivities[i])
                    for i in range(num_cols)
                ]
        
                # If return_details, compute per-column confidence intervals
                if return_details:
                    ci = [
                        compute_confidence_interval(
                            float(final_vals[i]), noise_scales[i], mechanism, confidence_level
                        )
                        for i in range(num_cols)
                    ]
                    return DPResult(
                        value=final_vals,
                        noise_mechanism=mechanism,
                        noise_scale=noise_scales,
                        epsilon_spent=epsilon * num_cols,
                        delta_spent=delta * num_cols,
                        confidence_interval=ci,
                    )
                return final_vals
        
    def histogram(
        self,
        values: Any,
        categories: Sequence[Any],
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        column: str | None = None,
        party: str | None = None,
        round_int: bool = False,
        clip_non_negative: bool = True,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> dict[Any, float] | DPResult:
        # DP 直方图计数：利用互斥分类的联合敏感度为 1，所有 Bin 共享一次预算
        """差分隐私直方图计数（基于联合敏感度为 1）。

        执行步骤：
        1. 提取样本特征，根据互斥分类性质确定直方图全量 Bin 的联合 L1/L2 敏感度均为 1。
        2. 扣减单次 (epsilon, delta) 预算（所有 Bin 共享一次预算，提升效用）。
        3. 聚合各 Bin 的无噪真实 Counter 计数。
        4. 对每个 Bin 独立注入 DP 噪声并后处理。
        5. 返回类别到带噪计数的映射字典或包含各 Bin 置信区间的 DPResult。
        """
        # Step 1: Validate mechanism and extract data
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="histogram", mechanism=mechanism):
                arr = extract_values(values, column=column, party=party)
        
                # Step 2: Consume budget once for all bins (joint sensitivity = 1 for histogram)
                self.budget.spend(epsilon, delta)
                # Increment Prometheus metrics counter
                DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="histogram").inc()
        
                # Step 3: Compute true (noise-free) count for each category bin
                counts = dict.fromkeys(categories, 0.0)
                if _is_sparse_matrix(arr):
                    # Sparse matrix: densify to 1D array and count occurrences
                    arr_arr = arr.toarray().ravel()
                    from collections import Counter
        
                    item_counts = dict(Counter(arr_arr))
                    for c in counts:
                        if c in item_counts:
                            counts[c] = float(item_counts[c])
                elif arr.size > 0:
                    if arr.dtype.kind in ("i", "f", "u", "b"):
                        # Numeric dtype: use vectorized np.unique for fast counting
                        unique_vals, unique_counts = np.unique(arr, return_counts=True)
                        item_counts = dict(zip(unique_vals.tolist(), unique_counts.tolist()))
                    else:
                        # Object/string dtype: use Python Counter
                        from collections import Counter
        
                        item_counts = dict(Counter(arr))
                    # Map counted items to the requested category bins
                    for c in counts:
                        if c in item_counts:
                            counts[c] = float(item_counts[c])
        
                # Compute noise scale for confidence interval reporting
                noise_scale = (
                    1.0 / epsilon
                    if mechanism == Mechanism.LAPLACE
                    else calibrate_analytic_gaussian(epsilon, delta, 1.0)
                )
                # Step 4: Inject independent DP noise into each bin
                res_dict = {}
                ci_dict = {}
                for c in counts:
                    # Sample fresh noise for each bin (L1 joint sensitivity = 1)
                    noise = self._sample_count_noise(epsilon, delta, mechanism)
                    raw_val = counts[c] + noise
                    # Apply post-processing: non-negative clip and/or integer rounding
                    final_val = _apply_post_processing(
                        raw_val, round_int=round_int, clip_non_negative=clip_non_negative
                    )
                    res_dict[c] = final_val
                    if return_details:
                        # Compute per-bin confidence interval
                        ci_dict[c] = compute_confidence_interval(
                            final_val, noise_scale, mechanism, confidence_level
                        )
        
                # Step 5: Return DPResult with per-bin CIs or plain dict
                if return_details:
                    return DPResult(
                        value=res_dict,
                        noise_mechanism=mechanism,
                        noise_scale=noise_scale,
                        epsilon_spent=epsilon,
                        delta_spent=delta,
                        confidence_interval=ci_dict,
                    )
                return res_dict
        
    def noisy_count(
        self,
        true_count: float,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        round_int: bool = False,
        clip_non_negative: bool = True,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> float | DPResult:
        # 对已聚合好的预计算计数直接注入 DP 噪声（用于分布式/流式场景）
        """对已经聚合好的计数结果直接注入 DP 噪声。"""
        # Validate mechanism and delta compatibility
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="count", mechanism=mechanism):
                # Compute noise scale for confidence interval reporting
                noise_scale = (
                    1.0 / epsilon
                    if mechanism == Mechanism.LAPLACE
                    else calibrate_analytic_gaussian(epsilon, delta, 1.0)
                )
                # Delegate common scalar DP flow to the shared template
                return self._execute_scalar_query(
                    aggregation="count",
                    true_value=true_count,
                    epsilon=epsilon,
                    delta=delta,
                    mechanism=mechanism,
                    noise_sampler=lambda: self._sample_count_noise(epsilon, delta, mechanism),
                    noise_scale=noise_scale,
                    return_details=return_details,
                    confidence_level=confidence_level,
                    round_int=round_int,
                    clip_non_negative=clip_non_negative,
                )
        
    def noisy_sum(
        self,
        true_sum: float,
        sensitivity: float,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        round_int: bool = False,
        clip_non_negative: bool = False,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> float | DPResult:
        # 对已聚合好的预计算求和直接注入 DP 噪声（用于分布式/流式场景）
        """对已经聚合好的求和结果直接注入 DP 噪声。"""
        # Validate mechanism and delta compatibility
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="sum", mechanism=mechanism):
                # Sensitivity must be non-negative (bounded range width)
                if sensitivity < 0:
                    raise ValueError("sensitivity must be non-negative")
                # Compute noise scale for confidence interval reporting
                noise_scale = (
                    (sensitivity / epsilon)
                    if (mechanism == Mechanism.LAPLACE and epsilon > 0)
                    else calibrate_analytic_gaussian(epsilon, delta, sensitivity)
                )
                # Delegate common scalar DP flow to the shared template
                return self._execute_scalar_query(
                    aggregation="sum",
                    true_value=true_sum,
                    epsilon=epsilon,
                    delta=delta,
                    mechanism=mechanism,
                    noise_sampler=lambda: self._sample_sum_noise(
                        sensitivity, epsilon, delta, mechanism
                    ),
                    noise_scale=noise_scale,
                    return_details=return_details,
                    confidence_level=confidence_level,
                    round_int=round_int,
                    clip_non_negative=clip_non_negative,
                )
        
    def noisy_mean(
        self,
        true_sum: float,
        true_count: float,
        sensitivity: float,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        min_count: float = 5.0,
        round_int: bool = False,
        clip_non_negative: bool = False,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> float | DPResult:
        # 对已聚合好的 sum/count 分别加噪后计算比值均值（Delta 方法估计方差）
        """对已经聚合好的 sum/count 分别注入 DP 噪声后得到均值。"""
        # Validate mechanism and delta compatibility
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="mean", mechanism=mechanism):
                # Sensitivity must be non-negative (bounded range width)
                if sensitivity < 0:
                    raise ValueError("sensitivity must be non-negative")
        
                # Apply composition theorem: split budget equally for count and sum
                eps_sub = epsilon / 2.0
                delta_sub = delta / 2.0
        
                # Compute noisy count with half the budget
                count_res = self.noisy_count(
                    true_count,
                    eps_sub,
                    delta_sub,
                    mechanism,
                    return_details=True,
                    confidence_level=confidence_level,
                )
                noisy_count = count_res.value
                count_scale = count_res.noise_scale
        
                # Guard against divergence: if noisy_count too small, ratio is unstable
                if noisy_count < min_count or noisy_count <= 0.0:
                    if return_details:
                        # 此处仅执行过 count 子查询（花费 eps_sub），sum 子查询未执行，
                        # 如实上报实际花费而非整个 epsilon。
                        return DPResult(
                            value=0.0,
                            noise_mechanism=mechanism,
                            noise_scale=0.0,
                            epsilon_spent=eps_sub,
                            delta_spent=delta_sub,
                            confidence_interval=(0.0, 0.0),
                        )
                    return 0.0
        
                # Compute noisy sum with the other half of the budget
                sum_res = self.noisy_sum(
                    true_sum,
                    sensitivity,
                    eps_sub,
                    delta_sub,
                    mechanism,
                    return_details=True,
                    confidence_level=confidence_level,
                )
                noisy_sum = sum_res.value
                sum_scale = sum_res.noise_scale
        
                # Compute ratio estimate and apply post-processing
                DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="mean").inc()
                raw_val = noisy_sum / noisy_count
                final_val = _apply_post_processing(
                    raw_val, round_int=round_int, clip_non_negative=clip_non_negative
                )
        
                # Delta method variance estimation for ratio estimator
                if return_details:
                    eff_scale, ci = self._mean_delta_method_ci(
                        final_val, noisy_sum, noisy_count, sum_scale, count_scale, mechanism, confidence_level
                    )
                    return DPResult(
                        value=final_val,
                        noise_mechanism=mechanism,
                        noise_scale=eff_scale,
                        epsilon_spent=epsilon,
                        delta_spent=delta,
                        confidence_interval=ci,
                    )
                return final_val
        
    def noisy_histogram(
        self,
        true_counts: dict[Any, float],
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        round_int: bool = False,
        clip_non_negative: bool = True,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> dict[Any, float] | DPResult:
        # 对已聚合好的直方图各 Bin 计数直接注入 DP 噪声
        """对已经聚合好的直方图计数直接注入 DP 噪声。"""
        # Validate mechanism and delta compatibility
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="histogram", mechanism=mechanism):
                # Compute noise scale for confidence interval reporting
                noise_scale = (
                    1.0 / epsilon
                    if mechanism == Mechanism.LAPLACE
                    else calibrate_analytic_gaussian(epsilon, delta, 1.0)
                )
                # Delegate common histogram DP flow to the shared template
                return self._execute_histogram_query(
                    true_counts=true_counts,
                    epsilon=epsilon,
                    delta=delta,
                    mechanism=mechanism,
                    noise_scale=noise_scale,
                    return_details=return_details,
                    confidence_level=confidence_level,
                    round_int=round_int,
                    clip_non_negative=clip_non_negative,
                )
        
    def chunked_count(
        self,
        chunks: Iterable[Any],
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        column: str | None = None,
        party: str | None = None,
        round_int: bool = False,
        clip_non_negative: bool = True,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> float | DPResult:
        # 分块流式 DP 计数：增量聚合多个 chunk 的真实计数，最后只加一次噪声
        """分块流式差分隐私计数。

        允许调用方以多个 chunk（生成器/迭代器）分批传入数据，
        sidecar 增量聚合真实计数后只加一次噪声、只消耗一次 (epsilon, delta) 预算。
        适用于上亿级数据无法一次性加载内存的场景。

        Args:
            chunks: 数据块的可迭代对象，每块支持 list/tuple/ndarray/Series/DataFrame/SecretFlow 格式。
            epsilon: 隐私预算参数。
            delta: 隐私预算 delta 参数。
            mechanism: "laplace" 或 "gaussian"。
            column: 当 chunk 为表格类型时，指定目标列名。
            party: 当 chunk 为 SecretFlow HDataFrame 时，指定参与方。
            round_int: 是否后处理进行整数取整。
            clip_non_negative: 是否后处理截断保证非负。
            return_details: 是否返回 DPResult。
            confidence_level: 置信区间水平，默认 0.95。

        Returns:
            带噪声的计数值或 DPResult 结构。
        """
        # Step 1: Validate mechanism and delta compatibility
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="count", mechanism=mechanism):
                # Initialize running total for true count across all chunks
                true_count = 0.0
                # Step 2: Iterate over chunks, accumulating true count incrementally
                for chunk in chunks:
                    # Extract each chunk into ndarray or sparse matrix
                    chunk_arr = extract_values(chunk, column=column, party=party)
                    if _is_sparse_matrix(chunk_arr):
                        # Sparse: nnz gives stored non-zero entry count
                        true_count += float(chunk_arr.nnz)
                    elif chunk_arr.size > 0:
                        if chunk_arr.dtype.kind in ("i", "f", "u", "b"):
                            # Numeric/boolean: vectorized count_nonzero
                            true_count += float(np.count_nonzero(chunk_arr))
                        else:
                            # Object dtype: Python-level truthiness counting
                            true_count += float(sum(1 for v in chunk_arr if v))
                # Step 3: Compute noise scale for confidence interval reporting
                noise_scale = (
                    1.0 / epsilon
                    if mechanism == Mechanism.LAPLACE
                    else calibrate_analytic_gaussian(epsilon, delta, 1.0)
                )
                # Delegate common scalar DP flow to the shared template
                return self._execute_scalar_query(
                    aggregation="count",
                    true_value=true_count,
                    epsilon=epsilon,
                    delta=delta,
                    mechanism=mechanism,
                    noise_sampler=lambda: self._sample_count_noise(epsilon, delta, mechanism),
                    noise_scale=noise_scale,
                    return_details=return_details,
                    confidence_level=confidence_level,
                    round_int=round_int,
                    clip_non_negative=clip_non_negative,
                )
        
    def chunked_sum(
        self,
        chunks: Iterable[Any],
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        clip_lower: float | None = None,
        clip_upper: float | None = None,
        column: str | None = None,
        party: str | None = None,
        round_int: bool = False,
        clip_non_negative: bool = False,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> float | DPResult:
        # 分块流式 DP 求和：按块 clip 并累加局部和，最终只消耗一次预算
        """分块流式差分隐私求和。

        调用方必须显式提供 clip 边界；本方法按块 clip 并累加局部和，
        最终只消耗一次 (epsilon, delta) 预算并注入噪声。

        Args:
            chunks: 数据块的可迭代对象，每块支持多种数据格式。
            epsilon: 隐私预算参数。
            delta: 隐私预算 delta 参数。
            mechanism: "laplace" 或 "gaussian"。
            clip_lower: 截断下界（必须显式提供）。
            clip_upper: 截断上界（必须显式提供）。
            column: 当 chunk 为表格类型时，指定目标列名。
            party: 当 chunk 为 SecretFlow HDataFrame 时，指定参与方。
            round_int: 是否后处理进行取整。
            clip_non_negative: 是否后处理截断保证非负。
            return_details: 是否返回 DPResult。
            confidence_level: 置信区间水平，默认 0.95。

        Returns:
            带噪声的求和结果或 DPResult 结构。
        """
        # Step 1: Validate mechanism and delta compatibility
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="sum", mechanism=mechanism):
                # chunked_sum requires explicit clip bounds (no data-dependent inference allowed)
                if clip_lower is None or clip_upper is None:
                    raise ValueError(
                        "chunked_sum requires explicit clip_lower and clip_upper"
                    )
                if clip_lower > clip_upper:
                    raise ValueError("clip_lower must be <= clip_upper")
                # Convert bounds to float and compute sensitivity = range width
                lower, upper = float(clip_lower), float(clip_upper)
                sensitivity = max(0.0, upper - lower)
        
                # Step 2: Iterate over chunks, clipping and accumulating partial sums
                true_sum = 0.0
                for chunk in chunks:
                    # Extract each chunk into ndarray or sparse matrix
                    chunk_arr = extract_values(chunk, column=column, party=party)
                    if _is_sparse_matrix(chunk_arr):
                        # Sparse: densify to 1D, clip, and accumulate
                        col = np.asarray(chunk_arr.todense()).ravel().astype(np.float64)
                        clipped = np.clip(col, lower, upper)
                        true_sum += float(clipped.sum())
                    elif chunk_arr.size > 0:
                        # Dense: vectorized clip and accumulate
                        clipped = self._clip_values(chunk_arr, lower, upper)
                        true_sum += float(clipped.sum())
        
                # Step 3: Compute noise scale for confidence interval reporting
                noise_scale = (
                    (sensitivity / epsilon)
                    if (mechanism == Mechanism.LAPLACE and epsilon > 0)
                    else calibrate_analytic_gaussian(epsilon, delta, sensitivity)
                )
                # Delegate common scalar DP flow to the shared template
                return self._execute_scalar_query(
                    aggregation="sum",
                    true_value=true_sum,
                    epsilon=epsilon,
                    delta=delta,
                    mechanism=mechanism,
                    noise_sampler=lambda: self._sample_sum_noise(
                        sensitivity, epsilon, delta, mechanism
                    ),
                    noise_scale=noise_scale,
                    return_details=return_details,
                    confidence_level=confidence_level,
                    round_int=round_int,
                    clip_non_negative=clip_non_negative,
                )
        
    def chunked_mean(
        self,
        chunks: Iterable[Any],
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        clip_lower: float | None = None,
        clip_upper: float | None = None,
        min_count: float = 5.0,
        column: str | None = None,
        party: str | None = None,
        round_int: bool = False,
        clip_non_negative: bool = False,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> float | DPResult:
        # 分块流式 DP 均值：组合定理拆分预算，内存占用与总数据量解耦
        """分块流式差分隐私均值。

        使用组合定理：将 (epsilon, delta) 拆分为两份，分别用于 count 与 sum。
        每个 chunk 只被遍历一次，内存占用与总数据量解耦。
        """
        # Step 1: Validate mechanism and require explicit clip bounds
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="mean", mechanism=mechanism):
                if clip_lower is None or clip_upper is None:
                    raise ValueError(
                        "chunked_mean requires explicit clip_lower and clip_upper"
                    )
                if clip_lower > clip_upper:
                    raise ValueError("clip_lower must be <= clip_upper")
                # Convert bounds to float and compute sensitivity = range width
                lower, upper = float(clip_lower), float(clip_upper)
                sensitivity = max(0.0, upper - lower)
        
                # Step 2: Iterate over chunks, accumulating true_count and true_sum
                true_count = 0.0
                true_sum = 0.0
                for chunk in chunks:
                    # Extract each chunk into ndarray or sparse matrix
                    chunk_arr = extract_values(chunk, column=column, party=party)
                    if _is_sparse_matrix(chunk_arr):
                        # Sparse: row count = shape[0], then densify and clip for sum
                        true_count += float(chunk_arr.shape[0])
                        col = np.asarray(chunk_arr.todense()).ravel().astype(np.float64)
                        clipped = np.clip(col, lower, upper)
                        true_sum += float(clipped.sum())
                    else:
                        # Dense: size gives row count, then clip and accumulate sum
                        true_count += float(chunk_arr.size)
                        if chunk_arr.size > 0:
                            clipped = self._clip_values(chunk_arr, lower, upper)
                            true_sum += float(clipped.sum())
        
                # Step 3: Apply composition theorem: split budget for count and sum
                eps_sub = epsilon / 2.0
                delta_sub = delta / 2.0
        
                # Compute noisy count with half the budget
                count_res = self.noisy_count(
                    true_count, eps_sub, delta_sub, mechanism,
                    return_details=True, confidence_level=confidence_level,
                )
                noisy_count = count_res.value
                count_scale = count_res.noise_scale
        
                # Guard against divergence: if noisy_count too small, ratio is unstable
                if noisy_count < min_count or noisy_count <= 0.0:
                    if return_details:
                        # 仅 count 子查询已花费 eps_sub，sum 子查询未执行，如实上报。
                        return DPResult(
                            value=0.0, noise_mechanism=mechanism, noise_scale=0.0,
                            epsilon_spent=eps_sub, delta_spent=delta_sub,
                            confidence_interval=(0.0, 0.0),
                        )
                    return 0.0
        
                # Compute noisy sum with the other half of the budget
                sum_res = self.noisy_sum(
                    true_sum, sensitivity, eps_sub, delta_sub, mechanism,
                    return_details=True, confidence_level=confidence_level,
                )
                noisy_sum = sum_res.value
                sum_scale = sum_res.noise_scale
        
                # Compute ratio estimate and apply post-processing
                DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="mean").inc()
                raw_val = noisy_sum / noisy_count
                final_val = _apply_post_processing(
                    raw_val, round_int=round_int, clip_non_negative=clip_non_negative
                )
        
                # Delta method variance estimation for ratio estimator
                if return_details:
                    eff_scale, ci = self._mean_delta_method_ci(
                        final_val, noisy_sum, noisy_count, sum_scale, count_scale, mechanism, confidence_level
                    )
                    return DPResult(
                        value=final_val, noise_mechanism=mechanism, noise_scale=eff_scale,
                        epsilon_spent=epsilon, delta_spent=delta, confidence_interval=ci,
                    )
                return final_val
        
    def chunked_histogram(
        self,
        chunks: Iterable[Any],
        categories: Sequence[Any],
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        column: str | None = None,
        party: str | None = None,
        round_int: bool = False,
        clip_non_negative: bool = True,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> dict[Any, float] | DPResult:
        # 分块流式 DP 直方图：按块统计各分类计数并合并，只消耗一次预算
        """分块流式差分隐私直方图计数。

        按块统计各分类计数并合并，最终对所有分桶加噪，只消耗一次预算。
        """
        # Step 1: Validate mechanism and delta compatibility
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="histogram", mechanism=mechanism):
                # Initialize per-category bin counts to zero
                counts = dict.fromkeys(categories, 0.0)
        
                # Step 2: Iterate over chunks, merging per-chunk counts into global bins
                for chunk in chunks:
                    # Extract each chunk into ndarray or sparse matrix
                    chunk_arr = extract_values(chunk, column=column, party=party)
                    if _is_sparse_matrix(chunk_arr):
                        # Sparse: densify to 1D array for counting
                        chunk_arr = chunk_arr.toarray().ravel()
                    if chunk_arr.size > 0:
                        if chunk_arr.dtype.kind in ("i", "f", "u", "b"):
                            # Numeric dtype: use vectorized np.unique for fast counting
                            unique_vals, unique_counts = np.unique(chunk_arr, return_counts=True)
                            item_counts = dict(zip(unique_vals.tolist(), unique_counts.tolist()))
                        else:
                            # Object/string dtype: use Python Counter
                            from collections import Counter
                            item_counts = dict(Counter(chunk_arr))
                        # Accumulate matching items into the category bins
                        for c in counts:
                            if c in item_counts:
                                counts[c] += float(item_counts[c])
        
                # Step 3: Compute noise scale for confidence interval reporting
                noise_scale = (
                    1.0 / epsilon
                    if mechanism == Mechanism.LAPLACE
                    else calibrate_analytic_gaussian(epsilon, delta, 1.0)
                )
                # Delegate common histogram DP flow to the shared template
                return self._execute_histogram_query(
                    true_counts=counts,
                    epsilon=epsilon,
                    delta=delta,
                    mechanism=mechanism,
                    noise_scale=noise_scale,
                    return_details=return_details,
                    confidence_level=confidence_level,
                    round_int=round_int,
                    clip_non_negative=clip_non_negative,
                )
        
    def dp_aggregate(
        self,
        df: Any,
        specs: dict[str, Any],
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        return_details: bool = False,
    ) -> dict[str, Any]:
        # Table-Level 原位表格 DP 聚合：按组合定理均分预算到各列，分发调用对应聚合方法
        """Table-Level 原位表格差分隐私聚合。

        执行步骤：
        1. 获取待聚合规格 `num_specs = len(specs)`。
           (Determine number of aggregation specs for budget splitting)
        2. 根据预算组合定理，将总预算均匀切分为 `eps_per_col = epsilon / num_specs`
           与 `delta_per_col = delta / num_specs`。
           (Split total budget equally across columns)
        3. 遍历 `specs` 配置字典（列名 -> 聚合类型及特定参数 tuple/str）。
           (Iterate over specs dict)
        4. 链式分发调用 `self.count` / `self.sum` / `self.mean` / `self.histogram` 完成原位表格多列 DP 聚合。
           (Dispatch to aggregation methods)
        5. 返回包含各列带噪聚合结果或 DPResult 对象的字典。
           (Return dict of results)
        """
        # Validate common numeric inputs (epsilon, delta)
        self._validate_inputs(epsilon, delta)
        with observe_duration(DP_DURATION, aggregation="aggregate", mechanism=mechanism):
                # Step 1: Determine number of aggregation specs for budget splitting
                num_specs = len(specs)
                if num_specs == 0:
                    return {}
                # Step 2: Apply composition theorem: split budget equally across columns
                eps_per_col = epsilon / num_specs
                delta_per_col = delta / num_specs
        
                # Step 3: Iterate over each column and its aggregation specification
                results = {}
                for col_name, spec in specs.items():
                    # Parse spec: either a simple string ("count") or (type, kwargs) tuple
                    if isinstance(spec, str):
                        agg_type = spec
                        kwargs = {}
                    elif isinstance(spec, (tuple, list)):
                        agg_type = spec[0]
                        kwargs = dict(spec[1]) if len(spec) > 1 else {}
                    else:
                        raise TypeError(f"Invalid spec for column {col_name}: {spec}")
        
                    # Inject common parameters into the kwargs dict
                    kwargs["column"] = col_name
                    kwargs["epsilon"] = eps_per_col
                    kwargs["delta"] = delta_per_col
                    kwargs["mechanism"] = mechanism
                    kwargs["return_details"] = return_details
        
                    # Step 4: Dispatch to the appropriate aggregation method
                    if agg_type == AggregationType.COUNT:
                        results[col_name] = self.count(df, **kwargs)
                    elif agg_type == AggregationType.SUM:
                        results[col_name] = self.sum(df, **kwargs)
                    elif agg_type == AggregationType.MEAN:
                        results[col_name] = self.mean(df, **kwargs)
                    elif agg_type == AggregationType.HISTOGRAM:
                        results[col_name] = self.histogram(df, **kwargs)
                    else:
                        raise ValueError(f"Unsupported aggregation type: {agg_type}")
                # Step 5: Return dict mapping column names to noisy results
                return results
        
    def adaptive_clip(
        self,
        values: Any,
        epsilon: float,
        target_quantile: float = 0.95,
        num_iterations: int = 15,
        initial_clip: float = 10.0,
        column: str | None = None,
        party: str | None = None,
    ) -> tuple[float, float]:
        # DP 自适应二分搜索估计 clip 上界：通过 DP 分位数估计确定安全截断范围
        """差分隐私自适应二分搜索估计 [0.0, clip_upper] 上下界。

        执行步骤：
        1. 提取样本数值并确定总样本数 total_count。
        2. 将总隐私预算拆分为 num_iterations 份：`eps_per_iter = epsilon / num_iterations`。
        3. 进行 num_iterations 次 DP 循环估计：
           - 统计小于等于当前估计界 `cur_clip` 的无噪样本数。
           - 调用 `noisy_count` 使用 `eps_per_iter` 注入 DP 噪声得到 `noisy_below`。
           - 比较 `frac = noisy_below / total_count` 与目标分位数 `target_quantile`：
             若小于分位数则扩大 clip 界（`cur_clip *= 1.5`），反之缩减（`cur_clip *= 0.85`）。
        4. 返回迭代收敛后的安全上下界 `(0.0, float(cur_clip))`。
        """
        # Validate common numeric inputs (epsilon)
        self._validate_inputs(epsilon)
        with observe_duration(DP_DURATION, aggregation="adaptive_clip", mechanism="n/a"):
                # Step 1: Extract data and handle empty input edge case
                arr = extract_values(values, column=column, party=party)
                if (isinstance(arr, np.ndarray) and arr.size == 0) or (
                    _is_sparse_matrix(arr) and arr.nnz == 0
                ):
                    # Empty data: return initial clip bounds unchanged
                    return (0.0, initial_clip)
        
                # Step 2: Split budget equally across all binary search iterations
                eps_per_iter = epsilon / num_iterations
                cur_clip = initial_clip
                # Total number of data points (for fraction computation)
                total_count = arr.nnz if _is_sparse_matrix(arr) else arr.size
        
                # Step 3: Iterative binary search with DP noisy comparisons
                for _ in range(num_iterations):
                    # Count how many values fall below the current clip estimate
                    if _is_sparse_matrix(arr):
                        sub_arr = arr.data
                        below_count = float(np.count_nonzero(sub_arr <= cur_clip))
                    else:
                        below_count = float(np.count_nonzero(arr <= cur_clip))
        
                    # Inject DP noise into the below-count for privacy
                    noisy_below = self.noisy_count(below_count, eps_per_iter)
                    # Compute the noisy fraction of data below cur_clip
                    frac = noisy_below / max(1.0, float(total_count))
        
                    # Adjust clip bound based on comparison with target quantile
                    if frac < target_quantile:
                        # Too few below => clip bound is too small, expand it
                        cur_clip *= 1.5
                    else:
                        # Enough below => clip bound is too large, shrink it
                        cur_clip *= 0.85
        
                # Step 4: Return converged clip bounds (0.0, cur_clip) with minimum floor
                return (0.0, max(0.01, float(cur_clip)))
        
    def create_accumulator(
        self,
        values: Any,
        column: str | None = None,
        party: str | None = None,
        clip_lower: float | None = None,
        clip_upper: float | None = None,
        categories: Sequence[Any] | None = None,
    ) -> Accumulator:
        # 构建分布式 Worker 节点的无噪 Accumulator，供 Master 合并后统一注入 DP 噪声
        """构建分布式 Worker 节点的无噪流式 Accumulator。

        执行步骤：
        1. Worker 节点从输入提取局部样本 Chunk 数据。
        2. 若配置了 clip_lower 与 clip_upper，对局部样本先执行截断裁剪，并累加无噪 sum。
        3. 累加局部样本的无噪 count 与 histogram 频次。
        4. 包装并返回无噪的 `Accumulator` 对象，供跨网络传输序列化。
        """
        with observe_duration(DP_DURATION, aggregation="create_accumulator", mechanism="n/a"):
                # Step 1: Extract data and determine size (nnz for sparse, size for dense)
                arr = extract_values(values, column=column, party=party)
                size = arr.nnz if _is_sparse_matrix(arr) else arr.size
        
                # Step 2: If clip bounds provided, clip values and compute clipped sum + sensitivity
                if clip_lower is not None and clip_upper is not None:
                    lower, upper = float(clip_lower), float(clip_upper)
                    if _is_sparse_matrix(arr):
                        # Sparse: densify to 1D, clip, and sum
                        col = np.asarray(arr.todense()).ravel().astype(np.float64)
                        clipped = np.clip(col, lower, upper)
                        s = float(clipped.sum())
                    else:
                        # Dense: vectorized clip and sum
                        clipped = self._clip_values(arr, lower, upper)
                        s = float(clipped.sum()) if clipped.size > 0 else 0.0
                    # Sensitivity = range width (max per-element change after clipping)
                    sens = max(0.0, upper - lower)
                else:
                    # No clip bounds: use raw sum with default sensitivity=1.
                    # WARNING: 无界数据下 sensitivity=1.0 是任意假定的值，DP 保证随之失效；
                    # 与 _resolve_clip_bounds 的告警保持一致，生产环境必须显式提供 clip 边界。
                    logger.warning(
                        "accumulator_sensitivity_unbounded",
                        extra={
                            "recommendation": "Set clip_lower/clip_upper explicitly for production DP guarantees; "
                            "without clipping, sensitivity=1.0 is an arbitrary assumption and the DP guarantee is void for unbounded data.",
                        },
                    )
                    s = float(arr.sum()) if size > 0 else 0.0
                    sens = 1.0
        
                # Step 3: Compute histogram if categories are provided
                hist = {}
                if categories:
                    # Initialize all category bins to zero
                    hist = dict.fromkeys(categories, 0.0)
                    if size > 0:
                        from collections import Counter
        
                        # Flatten sparse or dense array for counting
                        arr_data = arr.toarray().ravel() if _is_sparse_matrix(arr) else arr
                        item_counts = Counter(arr_data)
                        # Map counted items to the requested category bins
                        for c in hist:
                            if c in item_counts:
                                hist[c] = float(item_counts[c])
        
                # Step 4: Return noise-free Accumulator for distributed aggregation
                return Accumulator(count=float(size), sum=s, histogram=hist, sensitivity=sens)
        
    def finalize_dp(
        self,
        accumulator: Accumulator,
        aggregation: str,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        return_details: bool = False,
    ) -> float | dict[Any, float] | DPResult:
        # Master 节点对合并后的 Accumulator 统一注入一次 DP 噪声并导出结果
        """Master/Combine 节点对合并后的 Accumulator 统一注入一次 DP 噪声。

        执行步骤：
        1. Master 节点汇总来自于各个 Worker 的 `Accumulator` 对象并执行加法合并。
           (Master merges Accumulator objects from workers)
        2. 根据 `aggregation` 目标算子（"count"/"sum"/"mean"/"histogram"）转发调用对应的 `noisy_*` 函数。
           (Dispatch to noisy_* based on aggregation type)
        3. 仅在此处消耗一次 (epsilon, delta) 隐私预算，为总聚合统计量注入 DP 噪声并导出。
           (Consume budget once and inject noise)
        """
        # Validate common numeric inputs (epsilon, delta)
        self._validate_inputs(epsilon, delta)
        with observe_duration(DP_DURATION, aggregation="finalize_dp", mechanism=mechanism):
                # Step 1: Dispatch to the appropriate noisy_* method based on aggregation type
                if aggregation == AggregationType.COUNT:
                    # Inject noise into the accumulated count
                    return self.noisy_count(
                        accumulator.count, epsilon, delta, mechanism, return_details=return_details
                    )
                elif aggregation == AggregationType.SUM:
                    # Inject noise into the accumulated sum with its sensitivity
                    return self.noisy_sum(
                        accumulator.sum, accumulator.sensitivity, epsilon, delta, mechanism, return_details=return_details
                    )
                elif aggregation == AggregationType.MEAN:
                    # Compute noisy mean from accumulated sum/count with budget splitting
                    return self.noisy_mean(
                        accumulator.sum,
                        accumulator.count,
                        accumulator.sensitivity,
                        epsilon,
                        delta,
                        mechanism,
                        return_details=return_details,
                    )
                elif aggregation == AggregationType.HISTOGRAM:
                    # Inject noise into each histogram bin
                    return self.noisy_histogram(
                        accumulator.histogram, epsilon, delta, mechanism, return_details=return_details
                    )
                else:
                    raise ValueError(f"Unsupported aggregation for accumulator: {aggregation}")
        
    def vector_sum(
        self,
        vectors: Any,
        max_norm: float,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "gaussian",
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> np.ndarray | DPResult:
        # 高维向量 L2 范数截断与各向同性加噪（DP-SGD 基础逻辑），敏感度为 max_norm
        """高维向量 / 梯度 L2 范数截断与各向同性加噪 (DP-SGD 基础逻辑)。

        执行步骤：
        1. 转换输入向量样本为 2D NumPy 矩阵 (shape: N x d)。
        2. 计算每个向量的 L2 范数 ||v||_2。
        3. 应用 L2 范数截断放缩：v_clipped = v * min(1, max_norm / ||v||_2)。
        4. 累加求和得无噪向量 true_sum（敏感度上限锁死为 max_norm）。
        5. 向 BudgetAccountant 申请扣减 (epsilon, delta) 预算。
        6. 生成由 max_norm 标定的 d 维各向同性 (Isotropic) Gaussian / Laplace 噪声向量叠加。
        7. 返回带噪向量或包含各维度置信区间的 DPResult。
        """
        # Step 1: Validate mechanism and convert input to 2D matrix (N x d)
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="vector_sum", mechanism=mechanism):
                matrix = _to_2d_numpy_array(vectors)
        
                # Step 2: Compute L2 norm of each row vector
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                # Clamp norms to avoid division by zero for zero vectors
                norms = np.maximum(norms, 1e-12)
                # Step 3: L2 clip: scale each row so ||v_clipped||_2 <= max_norm
                scaling = np.minimum(1.0, max_norm / norms)
                clipped = matrix * scaling
                # Step 4: Sum all clipped vectors to get the true aggregate
                true_sum = clipped.sum(axis=0).astype(np.float64)
        
                # Step 5: Consume privacy budget
                self.budget.spend(epsilon, delta)
                # Increment metrics counter by dimensionality d
                DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="sum").inc(matrix.shape[1])
        
                # Step 6: Generate isotropic noise vector
                # Gaussian 机制的 L2 敏感度为 max_norm
                # Laplace 机制对 d 维向量在 L2 clip 约束下的 L1 敏感度上界为 sqrt(d) * max_norm
                d = matrix.shape[1]
                import math
                sensitivity = (max_norm * math.sqrt(d)) if (mechanism == "laplace" and d > 1) else max_norm
                noise_scale = self._compute_noise_scale(sensitivity, epsilon, delta, mechanism)
                noises = self._sample_isotropic_noise(d, mechanism, noise_scale)
        
                # Add noise to the true vector sum
                noisy_vec = true_sum + noises
                # If return_details, compute per-dimension confidence intervals
                if return_details:
                    ci = [
                        compute_confidence_interval(
                            float(noisy_vec[i]), noise_scale, mechanism, confidence_level
                        )
                        for i in range(matrix.shape[1])
                    ]
                    return DPResult(
                        value=noisy_vec,
                        noise_mechanism=mechanism,
                        noise_scale=noise_scale,
                        epsilon_spent=epsilon,
                        delta_spent=delta,
                        confidence_interval=ci,
                    )
                # Return noisy vector directly
                return noisy_vec
        
    def vector_mean(
        self,
        vectors: Any,
        max_norm: float,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "gaussian",
        min_count: float = 5.0,
        return_details: bool = False,
        confidence_level: float = 0.95,
    ) -> np.ndarray | DPResult:
        # DP 向量均值：L2 clip + 各向同性加噪 + noisy_count 归一化，用于 DP-SGD 平均梯度
        """高维向量 / 梯度 DP 均值：L2 范数截断 + 各向同性加噪 + noisy_count 归一化。

        用于 DP-SGD 平均梯度计算：先对每行做 L2 clip，再对 sum 和 count 分别加噪，
        最终 noisy_sum / noisy_count 得到带噪均值向量。
        """
        # Step 1: Validate mechanism and convert input to 2D matrix (N x d)
        mechanism = self._validate_mechanism(mechanism, delta)
        # Validate common numeric inputs (epsilon, delta, confidence_level)
        self._validate_inputs(epsilon, delta, confidence_level)
        with observe_duration(DP_DURATION, aggregation="vector_mean", mechanism=mechanism):
                matrix = _to_2d_numpy_array(vectors)
                n_rows = matrix.shape[0]
        
                # Step 2: L2 clip each row vector to max_norm
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                # Clamp norms to avoid division by zero
                norms = np.maximum(norms, 1e-12)
                # Scaling factor: min(1, max_norm / ||v||) ensures ||v_clipped|| <= max_norm
                scaling = np.minimum(1.0, max_norm / norms)
                clipped = matrix * scaling
                # Sum all clipped vectors to get the true aggregate
                true_sum = clipped.sum(axis=0).astype(np.float64)
        
                # Step 3: Apply composition theorem: split budget for count and sum
                eps_sub = epsilon / 2.0
                delta_sub = delta / 2.0
        
                # Compute noisy count with half the budget
                count_res = self.noisy_count(
                    float(n_rows), eps_sub, delta_sub, mechanism,
                    return_details=True, confidence_level=confidence_level,
                )
                noisy_count = count_res.value
        
                # Guard against divergence: if noisy_count too small, return zero vector
                if noisy_count < min_count or noisy_count <= 0.0:
                    zero_vec = np.zeros(matrix.shape[1], dtype=np.float64)
                    if return_details:
                        # 仅 count 子查询已花费 eps_sub，sum 子查询未执行，如实上报。
                        return DPResult(
                            value=zero_vec, noise_mechanism=mechanism, noise_scale=0.0,
                            epsilon_spent=eps_sub, delta_spent=delta_sub,
                            confidence_interval=[(0.0, 0.0)] * matrix.shape[1],
                        )
                    return zero_vec
        
                # Step 4: Compute noisy sum with the other half of the budget
                # Gaussian 机制的 L2 敏感度为 max_norm
                # Laplace 机制对 d 维向量在 L2 clip 约束下的 L1 敏感度上界为 sqrt(d) * max_norm
                # （与 vector_sum 保持一致的 √d 校准，否则高维 Laplace 欠噪）
                d = matrix.shape[1]
                import math
                sensitivity = (max_norm * math.sqrt(d)) if ((mechanism == Mechanism.LAPLACE or mechanism == "laplace") and d > 1) else max_norm
                noise_scale = self._compute_noise_scale(sensitivity, eps_sub, delta_sub, mechanism)
                noises = self._sample_isotropic_noise(d, mechanism, noise_scale)
        
                # Consume budget for the sum query
                self.budget.spend(eps_sub, delta_sub)
                # Increment metrics counter by dimensionality d
                DP_QUERIES_TOTAL.labels(mechanism=mechanism, aggregation="sum").inc(matrix.shape[1])
        
                # Step 5: Compute noisy mean vector = noisy_sum / noisy_count
                noisy_sum = true_sum + noises
                noisy_mean_vec = noisy_sum / noisy_count
        
                # If return_details, compute per-dimension confidence intervals
                if return_details:
                    ci = [
                        compute_confidence_interval(
                            float(noisy_mean_vec[i]), noise_scale / noisy_count, mechanism, confidence_level
                        )
                        for i in range(matrix.shape[1])
                    ]
                    return DPResult(
                        value=noisy_mean_vec,
                        noise_mechanism=mechanism,
                        noise_scale=noise_scale / noisy_count,
                        epsilon_spent=epsilon,
                        delta_spent=delta,
                        confidence_interval=ci,
                    )
                return noisy_mean_vec
        
    def dp_groupby(
        self,
        df: Any,
        group_col: str,
        target_col: str,
        agg: str,
        epsilon: float,
        delta: float = 1e-5,
        clip_lower: float | None = None,
        clip_upper: float | None = None,
        mechanism: str = "laplace",
        return_details: bool = False,
    ) -> dict[Any, Any]:
        # Tau-Thresholding DP GroupBy：按噪声计数阈值过滤罕见分组，防范存在性泄露
        """Tau-Thresholding 差分隐私 SQL Group-By 过滤。

        执行步骤：
        1. 按 group_col 字段对输入的 pandas DataFrame 做的分组 num_groups。
        2. 依据密码学 Tau-Thresholding 理论求解临界过滤阈值 tau = 1.0 + ln(1/(2*delta_query)) / eps_query。
        3. 对每个 Group 先调用 self.count 获得带噪频次 cnt。
        4. 判定 cnt >= tau：若小于 tau 则判定该 Group 为罕见/孤立敏感数据，予以强行丢弃（Drop）；
        5. 对保留下来的 Group 进一步计算目标 target_col 的 count/sum/mean 带噪聚合。
        6. 返回 Group 映射字典，有效防范数据记录存在性泄露（Group Leakage）。
        """
        # Validate common numeric inputs (epsilon, delta)
        self._validate_inputs(epsilon, delta)
        with observe_duration(DP_DURATION, aggregation="groupby", mechanism=mechanism):
                import pandas as pd
        
                # Step 1: Validate input is a pandas DataFrame
                if not isinstance(df, pd.DataFrame):
                    raise TypeError("dp_groupby currently requires pandas DataFrame input")
        
                # Step 2: Group by the specified column
                groups = df.groupby(group_col)
                num_groups = groups.ngroups
                if num_groups == 0:
                    return {}
        
                # Step 3: Compute per-query budget using composition theorem
                # Each group needs 2 queries (count + agg), total = num_groups * 2
                eps_per_query = epsilon / (num_groups * 2)
                del_per_query = delta / (num_groups * 2)
                # Step 4: Compute tau threshold for privacy-safe group filtering
                # Standard Tau-Thresholding: tau = 1 + ln(1/(2*delta_query)) / eps_query
                tau = 1.0 + math.log(1.0 / (2.0 * max(1e-12, del_per_query))) / max(1e-12, eps_per_query)
        
                # Step 5: Iterate over groups, filter by noisy count >= tau, then aggregate
                result = {}
        
                for key, group in groups:
                    # Compute noisy count for this group
                    cnt = self.count(
                        group, column=target_col, epsilon=eps_per_query, delta=del_per_query, mechanism=mechanism
                    )
                    # Tau-thresholding: drop groups with noisy count below tau (prevents group leakage)
                    if cnt >= tau:
                        if agg == AggregationType.COUNT:
                            # Reuse the noisy count as the result
                            result[key] = cnt
                        elif agg == AggregationType.SUM:
                            # Compute noisy sum for this group
                            result[key] = self.sum(
                                group,
                                column=target_col,
                                epsilon=eps_per_query,
                                delta=del_per_query,
                                mechanism=mechanism,
                                clip_lower=clip_lower,
                                clip_upper=clip_upper,
                                return_details=return_details,
                            )
                        elif agg == AggregationType.MEAN:
                            # Compute noisy mean for this group
                            result[key] = self.mean(
                                group,
                                column=target_col,
                                epsilon=eps_per_query,
                                delta=del_per_query,
                                mechanism=mechanism,
                                clip_lower=clip_lower,
                                clip_upper=clip_upper,
                                return_details=return_details,
                            )
        
                # Step 6: Return dict mapping group keys to noisy aggregation results
                return result
        
        
        # 本地差分隐私（Local DP）接口：基于随机响应在客户端采集端植入扰动，服务端无偏估计还原分布
class LocalDPApi:
    """本地差分隐私（Local DP）计算接口。

    执行步骤：
    1. 接收客户端单条或批量数据（二值 0/1 或多分类 category）。
    2. 基于 Warner / Randomized Response 随机响应算法在客户端采集端植入随机扰动。
    3. 服务端/收集端接收扰动后的统计结果，使用无偏估计（Debias）公式还原真实分布。
    """

    def __init__(self, seed: int | None = None):
        # 初始化 LocalDPApi：显式 seed 用确定性 PRNG（测试复现），默认用密码学安全 RNG
        """初始化 LocalDPApi，绑定随机数源。

        Args:
            seed: 显式传入时使用 ``random.Random(seed)``（Mersenne Twister，
                仅供测试复现，非加密安全）；默认 None 时使用
                ``secrets.SystemRandom()``（OS 熵源，密码学安全），
                与中央 DP 路径的 SecureRandom 默认姿态一致。
        """
        if seed is not None:
            # Deterministic PRNG for reproducible tests only
            self.rng: random.Random = random.Random(seed)
        else:
            # Cryptographically secure RNG for production (unpredictable)
            self.rng = secrets.SystemRandom()

    @staticmethod
    def _validate_epsilon(epsilon: float) -> None:
        # 校验 epsilon 预算必须为正值，否则抛出 ValueError
        """校验 epsilon 预算正值。"""
        # Epsilon must be strictly positive for valid DP guarantee
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @staticmethod
    def _normalize_ldp_epsilon(epsilon: float) -> float:
        """对齐 Go ldp 语义：NaN / 负值按 0 处理（此时隐私保证最弱、扰动最均匀）。"""
        if not math.isfinite(epsilon) or epsilon < 0:
            return 0.0
        return float(epsilon)

    def randomized_response(self, value: bool, epsilon: float) -> bool:
        """对布尔值执行 Randomized Response（与 Go ``ldp.RandomizedResponse`` 对齐）。

        以概率 ``p = e^ε / (1 + e^ε)`` 返回真实值，否则翻转；epsilon 允许为 0。
        """
        normalized = self._normalize_ldp_epsilon(epsilon)
        p = 1.0 / (1.0 + math.exp(-normalized))
        return bool(value) if self.rng.random() < p else not bool(value)

    def orr_response(self, value: int, epsilon: float, domain_size: int) -> int:
        """对离散类别执行 Optimized Randomized Response（与 Go ``ldp.ORRResponse`` 对齐）。

        以概率 ``p = e^ε / (e^ε + k - 1)`` 返回真实值，否则从其余 k-1 个类别中
        均匀抽取替代值。非法域或越界值返回 -1。
        """
        if domain_size < 1 or value < 0 or value >= domain_size:
            return -1
        if domain_size <= 1:
            return value
        normalized = self._normalize_ldp_epsilon(epsilon)
        p = 1.0 / (1.0 + (domain_size - 1) * math.exp(-normalized))
        if self.rng.random() < p:
            return value
        other = self.rng.randrange(domain_size - 1)
        if other >= value:
            other += 1
        return other

    def perturb_binary(self, value: int, epsilon: float) -> int:
        # 对单个二值数据进行 epsilon-LDP 随机响应扰动（Warner 模型）
        """对单个二值数据进行 ε-本地差分隐私扰动。

        执行步骤：
        1. 校验 value 属于 {0, 1}，校验 epsilon > 0。
        2. 计算保持真值的概率 $p = \frac{e^\varepsilon}{1 + e^\varepsilon}$。
        3. 以概率 p 保持原值，以概率 1-p 返回翻转后的相反值（1-value）。
        """
        # Validate epsilon is positive
        self._validate_epsilon(epsilon)
        # Binary value must be exactly 0 or 1
        if value not in (0, 1):
            raise ValueError("binary value must be 0 or 1")

        # Compute probability of keeping the true value using the logistically stable form
        p = 1.0 / (1.0 + math.exp(-epsilon))
        # With probability p return original value, otherwise flip it
        return value if self.rng.random() < p else 1 - value

    def perturb_binary_batch(self, values: Sequence[int], epsilon: float) -> np.ndarray:
        # 批量对二值序列进行 LDP 扰动，返回 int64 数组
        """批量对二值数据进行本地 DP 扰动。"""
        # Apply perturb_binary to each element and return as int64 array
        return np.array([self.perturb_binary(int(v), epsilon) for v in values], dtype=np.int64)

    def perturb_categorical(
        self, value: Any, categories: Sequence[Any], epsilon: float
    ) -> Any:
        # 对单个类别型数据进行 k-ary Randomized Response 本地 DP 扰动
        """对单个类别型数据进行 ε-本地差分隐私扰动（k-ary Randomized Response）。

        执行步骤：
        1. 校验输入类别合法性与 categories 候选长度 k >= 2。
        2. 计算保留原值的概率 $p = \frac{e^\varepsilon}{k - 1 + e^\varepsilon}$。
        3. 以概率 p 返回原类别，以概率 1-p 从其余 $k-1$ 个类别中等概率随机抽取替代值。
        """
        # Validate epsilon is positive
        self._validate_epsilon(epsilon)
        # Input value must be one of the provided categories
        if value not in categories:
            raise ValueError("value must be one of the provided categories")

        # k = number of categories (must be >= 2 for meaningful randomization)
        k = len(categories)
        if k < 2:
            raise ValueError("categories must contain at least 2 items")

        # Probability of keeping the true value: p = exp(eps) / (k-1 + exp(eps))
        p = 1.0 / (1.0 + (k - 1) * math.exp(-epsilon))
        # With probability p, return the original value
        if self.rng.random() < p:
            return value

        # With probability 1-p, uniformly sample from the other k-1 categories
        others = [c for c in categories if c != value]
        return self.rng.choice(others)

    def perturb_categorical_batch(
        self, values: Sequence[Any], categories: Sequence[Any], epsilon: float
    ) -> np.ndarray:
        # 批量对类别型序列进行 k-ary Randomized Response LDP 扰动
        """批量对类别型数据进行本地 DP 扰动。"""
        # Apply perturb_categorical to each element and return as object array
        return np.array([self.perturb_categorical(v, categories, epsilon) for v in values], dtype=object)

    def estimate_binary_frequency(
        self, reported_values: Sequence[int], epsilon: float
    ) -> float:
        # 根据扰动后的二值样本，使用无偏纠偏公式估计真实比例为 1 的频率
        """根据扰动后的二值样本估计真实比例为 1 的无偏频率。

        执行步骤：
        1. 统计客户端上报的带噪样本中 1 的比例 f_reported。
        2. 计算算法采样真值概率 p = exp(eps) / (1 + exp(eps))。
        3. 应用概率纠偏公式：hat_f = (f_reported - (1 - p)) / (2p - 1)。
        4. 截断区间到 [0.0, 1.0] 导出无偏频率。
        """
        # Validate epsilon is positive
        self._validate_epsilon(epsilon)
        n = len(reported_values)
        # Empty input => return 0.0 frequency
        if n == 0:
            return 0.0

        # Step 1: Compute the probability of keeping true value in randomized response
        p = 1.0 / (1.0 + math.exp(-epsilon))
        # Step 2: Compute the observed fraction of 1s in reported (noisy) data
        f_reported = sum(1 for v in reported_values if v == 1) / n
        # Step 3: Apply unbiased debiasing formula: hat_f = (f_reported - (1-p)) / (2p-1)
        est = (f_reported - (1.0 - p)) / (2.0 * p - 1.0)
        # Step 4: Clamp estimate to valid probability range [0, 1]
        return float(max(0.0, min(1.0, est)))

    def estimate_categorical_histogram(
        self,
        reported_values: Sequence[Any],
        categories: Sequence[Any],
        epsilon: float,
    ) -> dict[Any, float]:
        # 根据扰动后的类别样本，使用无偏纠偏公式估计各类别的真实直方图分布
        """根据扰动后的类别样本估计各类别的真实直方图分布。

        执行步骤：
        1. 计算随机响应概率：p = e^ε/(k-1+e^ε)（报出真实类别的概率），
           q = (1-p)/(k-1)（报出任一指定错误类别的概率），纠偏分母 D = p - q。
        2. 统计上报样本中各类别的**频率** f_reported_j = count_j / n，
           应用纠偏估计量：hat_f_j = (f_reported_j - q) / (p - q)。
        3. 将负的估计值非负截断为 0，再归一化使所有类别频率之和为 1.0
           （若截断后总和为 0，则回退为均匀分布 1/k）。

        注意：第 3 步的非负截断与归一化是对无偏估计量的正则化后处理，
        因此最终输出是"归一化后的频率估计"，严格意义上不再是无偏估计；
        随样本量增大，截断/归一化的影响趋近于 0，估计渐近无偏。
        """
        # Validate epsilon is positive
        self._validate_epsilon(epsilon)
        n = len(reported_values)
        k = len(categories)
        # Must have at least 2 categories for meaningful estimation
        if k < 2:
            raise ValueError("categories must contain at least 2 items")

        # Step 1: Compute randomized response probabilities
        # p = probability of reporting true category
        p = 1.0 / (1.0 + (k - 1) * math.exp(-epsilon))
        # q = probability of reporting any specific wrong category
        q = (1.0 - p) / (k - 1)
        # Denominator for debiasing: D = p - q
        denominator = p - q

        # Step 2: Count occurrences of each category in reported data
        counts: dict[Any, int] = dict.fromkeys(categories, 0)
        for v in reported_values:
            if v in counts:
                counts[v] += 1

        # Step 3: Apply unbiased debiasing estimator for each category
        # hat_f_j = (f_reported_j - q) / (p - q)
        estimates: dict[Any, float] = {}
        for c in categories:
            f_reported = counts[c] / n if n > 0 else 0.0
            est = (f_reported - q) / denominator if denominator != 0 else 1.0 / k
            # Clamp negative estimates to 0 (non-negative truncation)
            estimates[c] = max(0.0, est)

        # Step 4: Normalize estimates so they sum to 1.0 (valid probability distribution)
        total = sum(estimates.values())
        estimates = (
            {c: v / total for c, v in estimates.items()} if total > 0
            else dict.fromkeys(categories, 1.0 / k)
        )

        return estimates
