"""REST 请求模型集合（Pydantic）。

按域分组集中定义各端点的请求体模型，供 ``routers/*`` 子路由导入。
这些模型与 ``main.py`` 拆分前的定义保持完全一致，确保接口契约不变。
"""

from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# 脱敏 / 哈希
# --------------------------------------------------------------------------- #


class MaskRequest(BaseModel):
    """单字段脱敏请求模型。

    同时接受 Go 契约的 ``field``/``type``/``salt`` 与 Python 历史的
    ``field_name``/``context``；``field`` 优先。
    """

    field_name: str = Field(default="", max_length=200)
    field: str = Field(default="", max_length=200)
    value: str = Field(max_length=100_000)
    type: str = Field(default="", max_length=50)
    salt: str = Field(default="", max_length=1_000)
    context: str = Field(default="", max_length=10_000)


class MaskRecordRequest(BaseModel):
    """整记录脱敏请求模型。"""

    record: dict[str, str] = Field(max_length=200)
    context: str = Field(default="", max_length=10_000)


class MaskBatchRequest(BaseModel):
    """批量脱敏请求模型。

    支持两种 Go 契约形态：
    - ``records``：整记录批量自动脱敏；
    - ``field_names`` + ``values``：按字段名推断类型逐值脱敏。
    两者同时提供时 ``values`` 优先（与 Go 一致）。
    """

    records: list[dict[str, str]] | None = Field(default=None, max_length=1_000)
    field_names: list[str] = Field(default_factory=list, max_length=200)
    values: list[str] = Field(default_factory=list, max_length=10_000)
    context: str = Field(default="", max_length=10_000)


class MaskDataFrameRequest(BaseModel):
    """DataFrame 脱敏请求模型。

    data 为 records 列表（可来自 pandas/SecretFlow DataFrame 的转换）。
    columns 指定需要脱敏的列；未指定则对所有字符串列脱敏。
    """

    data: list[dict[str, Any]] = Field(max_length=1_000)
    columns: list[str] | None = Field(default=None, max_length=200)
    context: str = Field(default="", max_length=10_000)


class HashRequest(BaseModel):
    """HMAC 哈希请求模型。"""

    value: str = Field(max_length=100_000)
    salt: str = Field(max_length=1_000)


# --------------------------------------------------------------------------- #
# 差分隐私（DP）
# --------------------------------------------------------------------------- #


class DPRequest(BaseModel):
    """差分隐私聚合请求模型。

    values 为输入数据列表；params 为可选参数，用于覆盖默认或 profile 中的配置。
    """

    values: list[float] = Field(max_length=10_000)
    params: dict[str, object] = Field(default={}, max_length=100)


class DPHistogramRequest(BaseModel):
    """差分隐私直方图请求模型。"""

    values: list[str] = Field(max_length=10_000)
    categories: list[str] = Field(max_length=1_000)
    params: dict[str, object] = Field(default={}, max_length=100)


class DPNoisyCountRequest(BaseModel):
    """对已聚合计数进行 DP 加噪的请求模型。"""

    true_count: float = Field(description="真实聚合计数值")
    params: dict[str, object] = Field(default={}, max_length=100)


class DPNoisySumRequest(BaseModel):
    """对已聚合求和进行 DP 加噪的请求模型。

    params 中需提供 sensitivity，或同时提供 clip_lower 与 clip_upper。
    """

    true_sum: float = Field(description="真实聚合求和值")
    params: dict[str, object] = Field(default={}, max_length=100)


class DPNoisyMeanRequest(BaseModel):
    """对已聚合 sum/count 进行 DP 加噪得到均值的请求模型。"""

    true_sum: float = Field(description="真实聚合求和值")
    true_count: float = Field(description="真实聚合计数值")
    params: dict[str, object] = Field(default={}, max_length=100)


class DPNoisyHistogramRequest(BaseModel):
    """对已聚合直方图计数进行 DP 加噪的请求模型。"""

    true_counts: dict[str, float] = Field(max_length=1_000, description="各桶真实计数")
    params: dict[str, object] = Field(default={}, max_length=100)


class DPChunkedCountRequest(BaseModel):
    """分块流式 DP 计数请求模型。"""

    chunks: list[list[float]] = Field(max_length=1_000)
    params: dict[str, object] = Field(default={}, max_length=100)


class DPChunkedSumRequest(BaseModel):
    """分块流式 DP 求和请求模型。"""

    chunks: list[list[float]] = Field(max_length=1_000)
    params: dict[str, object] = Field(default={}, max_length=100)


class DPChunkedMeanRequest(BaseModel):
    """分块流式 DP 均值请求模型。"""

    chunks: list[list[float]] = Field(max_length=1_000)
    params: dict[str, object] = Field(default={}, max_length=100)


class DPAggregateRequest(BaseModel):
    """表格级原位 DP 聚合请求模型。"""

    rows: list[dict[str, Any]] = Field(max_length=1_000)
    specs: dict[str, Any] = Field(default={}, max_length=100)
    params: dict[str, object] = Field(default={}, max_length=100)


class DPVectorSumRequest(BaseModel):
    """高维向量 / 梯度 $L_2$ 范数截断加噪请求模型。"""

    vectors: list[list[float]] = Field(max_length=1_000)
    params: dict[str, object] = Field(default={}, max_length=100)


class DPAdaptiveClipRequest(BaseModel):
    """差分隐私自适应二分搜索估计上下界请求模型。"""

    values: list[float] = Field(max_length=10_000)
    params: dict[str, object] = Field(default={}, max_length=100)


class DPGroupByRequest(BaseModel):
    """Tau-Thresholding 差分隐私 SQL Group-By 请求模型。"""

    rows: list[dict[str, Any]] = Field(max_length=1_000)
    group_col: str = Field(max_length=200)
    target_col: str = Field(max_length=200)
    agg: str = Field(max_length=50)
    params: dict[str, object] = Field(default={}, max_length=100)


class DPChunkedHistogramRequest(BaseModel):
    """分块流式 DP 直方图请求模型。"""

    chunks: list[list[str]] = Field(max_length=1_000)
    categories: list[str] = Field(max_length=1_000)
    params: dict[str, object] = Field(default={}, max_length=100)


# --------------------------------------------------------------------------- #
# K-匿名
# --------------------------------------------------------------------------- #


class KAnonRequest(BaseModel):
    """K-匿名单条/数据集请求模型（兼容 Go ``records``/``qi_fields`` 别名）。"""

    record: dict[str, object] | None = Field(default=None, max_length=200)
    records: list[dict[str, object]] | None = Field(default=None, max_length=1_000)
    qi_cols: list[str] = Field(default_factory=list, max_length=50)
    qi_fields: list[str] = Field(default_factory=list, max_length=50)
    # Go 对 k < 1 的取值统一收口为 2（minK），这里保留原始值由路由层收口
    k: int = Field(default=0, le=1000)


class KAnonTableRequest(BaseModel):
    """K-匿名整张表请求模型（兼容 Go ``records``/``qi_fields`` 别名）。"""

    rows: list[dict[str, object]] = Field(default_factory=list, max_length=1_000)
    records: list[dict[str, object]] | None = Field(default=None, max_length=1_000)
    qi_cols: list[str] = Field(default_factory=list, max_length=50)
    qi_fields: list[str] = Field(default_factory=list, max_length=50)
    k: int = Field(default=0, le=1000)
    max_depth: int = Field(default=10, ge=1, le=50)


class KAnonDataFrameRequest(BaseModel):
    """K-匿名 DataFrame 请求模型。

    data 为 records 列表（可来自 pandas/SecretFlow DataFrame）；同时接受
    Go 契约的 ``records`` 与 ``qi_fields`` 别名。
    """

    data: list[dict[str, Any]] = Field(default_factory=list, max_length=1_000)
    records: list[dict[str, Any]] | None = Field(default=None, max_length=1_000)
    qi_cols: list[str] = Field(default_factory=list, max_length=50)
    qi_fields: list[str] = Field(default_factory=list, max_length=50)
    k: int = Field(default=0, le=1000)
    max_depth: int = Field(default=10, ge=1, le=50)


class LDiversityCheckRequest(BaseModel):
    """L-多样性合规检查请求模型。"""

    records: list[dict[str, Any]] = Field(default_factory=list, max_length=5_000)
    qi_fields: list[str] = Field(default_factory=list, max_length=50)
    qi_cols: list[str] = Field(default_factory=list, max_length=50)
    sensitive_field: str = Field(..., max_length=128)
    l: int = Field(default=2, ge=1, le=100)


class KAnonLDiversityRequest(BaseModel):
    """带 L-多样性约束的 K-匿名请求模型。"""

    records: list[dict[str, Any]] | None = Field(default=None, max_length=5_000)
    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=5_000)
    qi_fields: list[str] = Field(default_factory=list, max_length=50)
    qi_cols: list[str] = Field(default_factory=list, max_length=50)
    sensitive_field: str = Field(..., max_length=128)
    k: int = Field(default=2, le=1000)
    l: int = Field(default=2, ge=1, le=100)
    max_depth: int = Field(default=10, ge=1, le=50)


# --------------------------------------------------------------------------- #
# 查询混淆（QoL）
# --------------------------------------------------------------------------- #


class QolRequest(BaseModel):
    """查询混淆请求模型（Go 契约：num_decoys 别名，非正数由服务层收口）。"""

    query: str = Field(max_length=10_000)
    num_dummies: int = Field(default=3)
    num_decoys: int = Field(default=0)
    domain: str = Field(default="medical", max_length=100)
    medical_pool: list[str] | None = Field(default=None, max_length=1_000)
    generic_pool: list[str] | None = Field(default=None, max_length=1_000)
    seed: int | None = None


class QolBatchRequest(BaseModel):
    """批量查询混淆请求模型（Go 契约：num_decoys 别名，非正数由服务层收口）。"""

    queries: list[str] = Field(max_length=1_000)
    num_dummies: int = Field(default=3)
    num_decoys: int = Field(default=0)
    domain: str = Field(default="medical", max_length=100)
    medical_pool: list[str] | None = Field(default=None, max_length=1_000)
    generic_pool: list[str] | None = Field(default=None, max_length=1_000)
    seed: int | None = None


# --------------------------------------------------------------------------- #
# 本地差分隐私（LDP）
# --------------------------------------------------------------------------- #


class LdpPerturbBinaryRequest(BaseModel):
    """二值本地 DP 扰动请求模型。"""

    values: list[int] = Field(max_length=10_000)
    epsilon: float = Field(gt=0)


class LdpPerturbCategoricalRequest(BaseModel):
    """类别型本地 DP 扰动请求模型。"""

    values: list[str] = Field(max_length=10_000)
    categories: list[str] = Field(max_length=1_000)
    epsilon: float = Field(gt=0)


class LdpEstimateBinaryRequest(BaseModel):
    """二值本地 DP 估计请求模型。"""

    reported_values: list[int] = Field(max_length=10_000)
    epsilon: float = Field(gt=0)


class LdpEstimateCategoricalRequest(BaseModel):
    """类别型本地 DP 估计请求模型。"""

    reported_values: list[str] = Field(max_length=10_000)
    categories: list[str] = Field(max_length=1_000)
    epsilon: float = Field(gt=0)


class LdpRandomizedResponseRequest(BaseModel):
    """单值 Randomized Response 请求模型（与 Go ``ldp/randomized_response`` 对齐）。"""

    value: bool
    epsilon: float = Field(ge=0)


class LdpOrrRequest(BaseModel):
    """单值 O-RR 请求模型（与 Go ``ldp/orr`` 对齐）。"""

    value: int = Field(ge=0)
    epsilon: float = Field(ge=0)
    domain_size: int = Field(ge=1)


# --------------------------------------------------------------------------- #
# 隐私参数推荐
# --------------------------------------------------------------------------- #


class RecommendRequest(BaseModel):
    """隐私参数推荐请求模型。"""

    namespace: str = Field(max_length=200)
    values: list[float] | None = Field(default=None, max_length=10_000)
    rows: list[dict[str, object]] | None = Field(default=None, max_length=1_000)
    qi_cols: list[str] | None = Field(default=None, max_length=50)


# --------------------------------------------------------------------------- #
# 缩写规范与语义增强别名（保证 PEP 8 专有名词全大写规范与向后兼容）
# --------------------------------------------------------------------------- #

LDPPerturbBinaryRequest = LdpPerturbBinaryRequest
LDPPerturbCategoricalRequest = LdpPerturbCategoricalRequest
LDPEstimateBinaryRequest = LdpEstimateBinaryRequest
LDPEstimateCategoricalRequest = LdpEstimateCategoricalRequest

QOLRequest = QolRequest
QOLBatchRequest = QolBatchRequest
QueryObfuscationRequest = QolRequest
QueryObfuscationBatchRequest = QolBatchRequest

KAnonymityRequest = KAnonRequest
KAnonymityTableRequest = KAnonTableRequest
KAnonymityDataFrameRequest = KAnonDataFrameRequest

