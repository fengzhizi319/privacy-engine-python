"""隐私计算服务编排层。

PrivacyService 作为 REST/gRPC 两种协议的统一业务入口，将上层请求路由到
具体的隐私原语实现（脱敏、差分隐私、K-匿名、查询混淆等），并负责参数解析。

Service orchestration layer. PrivacyService is the shared business entry point
for both REST and gRPC interfaces, routing requests to underlying privacy
primitives (masking, differential privacy, K-anonymity, query obfuscation) and
resolving parameters from profile/config.
"""

from typing import Any, cast

from .privacy.budget import BudgetRegistry, default_registry
from .privacy.dp import DPApi, DPResult, LocalDPApi
from .privacy.kano import (
    BUILTIN_HIERARCHIES,
    GroupDiversity,
    KAnonymityRecordResult,
    LDiversityResult,
    anonymize_record,
    check_distinct_l_diversity,
)
from .privacy.kano_table import (
    KAnonymityResult,
    anonymize_with_l_diversity,
    k_anonymize_dataframe,
    k_anonymize_table,
    mondrian_with_l_diversity,
)
from .privacy.masking import (
    MaskingResult,
    hash_value,
    mask_dataframe,
    mask_record,
    mask_value,
    mask_value_batch,
    truncate,
)
from .privacy.profile import get_resolver
from .privacy.qol import obfuscate_query, obfuscate_query_batch


class PrivacyService:
    """隐私计算统一服务类。

    持有 ParameterResolver（用于解析各原语参数）、DPApi（差分隐私预算管理）。
    所有 REST/gRPC handler 均委托给本类的实例方法，便于复用与单元测试。

    Attributes:
        resolver: 参数解析器，从 profile 与请求中解析并校验参数。
        namespace: 当前隐私命名空间，用于隔离预算。
        dp_api: 差分隐私 API 实例。
    """

    def __init__(
        self,
        profile_path: str | None = None,
        namespace: str = "default",
        registry: BudgetRegistry | None = None,
        epsilon_total: float | None = None,
        delta_total: float | None = None,
        window_seconds: float | None = None,
    ):
        """初始化 PrivacyService。

        Args:
            profile_path: YAML 配置文件路径，可覆盖默认参数。
            namespace: 隐私命名空间，用于隔离隐私预算。
            registry: 可选的 BudgetRegistry 注册表，未提供时使用全局 default_registry。
            epsilon_total: 可选 epsilon 总预算；仅在对应 BudgetAccountant 尚未创建时生效。
            delta_total: 可选 delta 总预算；仅在对应 BudgetAccountant 尚未创建时生效。
            window_seconds: 可选预算重置窗口（秒）；仅在对应 BudgetAccountant 尚未创建时生效。
        """
        self.resolver = get_resolver(profile_path)
        self.namespace = namespace
        self.registry = registry or default_registry
        self.dp_api = DPApi(
            namespace,
            registry=self.registry,
            epsilon_total=epsilon_total,
            delta_total=delta_total,
            window_seconds=window_seconds,
        )
        self.local_dp_api = LocalDPApi()

    def mask(self, field_name: str, value: str, context: str = "") -> str | MaskingResult:
        """对单个字段值进行脱敏。

        Args:
            field_name: 字段名，用于推断敏感类型。
            value: 原始字段值。
            context: 上下文信息，透传给底层 masking 模块。

        Returns:
            脱敏后的字符串。
        """
        return mask_value(field_name, value, context)

    def mask_by_type(self, field_type: str, value: str, salt: str = "") -> str:
        """按显式类型执行脱敏（与 Go ``service.MaskField`` / handler 口径对齐）。

        支持 Go 的 9 类算子（id_card/phone/bank_card/name/email/address/
        officer_id/sm3/hmac，含 hash_* 别名），另保留 Python 的 mobile/medical
        等扩展类型。未知类型抛出 ``ValueError``，由入口映射为 400。
        """
        from .privacy.masking import (
            mask_address,
            mask_bank_card,
            mask_email,
            mask_id_card,
            mask_medical,
            mask_mobile,
            mask_name,
            mask_officer_id,
        )
        from .privacy.masking import hash_value as _hash_hmac
        from .security.sm3 import new_sm3

        ftype = (field_type or "").strip().lower()
        if ftype in ("sm3", "hash_sm3"):
            if value == "":
                return ""
            digest = new_sm3()
            if salt:
                digest.update(salt.encode("utf-8"))
            digest.update(value.encode("utf-8"))
            return digest.hexdigest()[:16]
        if ftype in ("hmac", "hash_hmac"):
            return str(_hash_hmac(value, salt))
        handlers = {
            "id_card": mask_id_card,
            "phone": mask_mobile,
            "mobile": mask_mobile,
            "bank_card": mask_bank_card,
            "name": mask_name,
            "email": mask_email,
            "address": mask_address,
            "officer_id": mask_officer_id,
            "medical": mask_medical,
        }
        handler = handlers.get(ftype)
        if handler is None:
            raise ValueError(f"unknown mask type: {field_type}")
        return handler(value)

    def mask_record(self, record: dict[str, str], context: str = "") -> dict[str, str] | MaskingResult:
        """对整条记录进行脱敏。

        Args:
            record: 原始记录字典。
            context: 上下文信息，透传给底层 masking 模块。

        Returns:
            脱敏后的记录字典。
        """
        return mask_record(record, context)

    def mask_batch(
        self, field_names: list[str], values: list[str], context: str = ""
    ) -> list[str] | MaskingResult:
        """批量对字段值进行脱敏。"""
        return mask_value_batch(field_names, values, context)

    def mask_dataframe(
        self, df: Any, columns: list[str] | None = None, context: str = ""
    ) -> Any:
        """对 DataFrame 进行脱敏。

        Args:
            df: 输入 DataFrame（pandas 或 SecretFlow）。
            columns: 需要脱敏的列名列表；None 则对所有字符串列脱敏。
            context: 上下文信息。

        Returns:
            脱敏后的 DataFrame。
        """
        return mask_dataframe(df, columns=columns, context=context)

    def hash(self, value: str, salt: str) -> str | MaskingResult:
        """对字符串进行 HMAC 哈希。

        Args:
            value: 待哈希原始值。
            salt: 哈希盐值。

        Returns:
            16 字符长度的 base64 摘要。
        """
        return hash_value(value, salt)

    def hash_sm3(self, value: str, salt: str) -> str:
        """国密 SM3 确定性哈希（十六进制前 16 位），对齐 Go ``HashSM3``。"""
        return str(self.mask_by_type("sm3", value, salt))

    def truncate(self, value: str, keep_prefix: int) -> str | MaskingResult:
        """截断字符串，保留指定前缀。

        Args:
            value: 原始字符串。
            keep_prefix: 保留的前缀长度。

        Returns:
            截断后的字符串。
        """
        return truncate(value, keep_prefix)

    @staticmethod
    def _dp_format_kwargs(params: dict[str, Any] | None) -> dict[str, Any]:
        """从 params 中提取数据格式相关的 column / party 参数。"""
        kwargs: dict[str, Any] = {}
        if params:
            if "column" in params:
                kwargs["column"] = params["column"]
            if "party" in params:
                kwargs["party"] = params["party"]
        return kwargs

    def dp_count(self, values: Any, params: dict[str, Any] | None = None) -> float | DPResult:
        """差分隐私计数。

        Args:
            values: 输入数值，支持 list/tuple/ndarray/Series/DataFrame/SecretFlow 格式。
            params: 请求级 DP 参数，例如 {"epsilon": 1.0, "mechanism": "laplace"}。

        Returns:
            带噪声的计数值。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.count(
            values,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            **self._dp_format_kwargs(params),
        )

    def dp_sum(self, values: Any, params: dict[str, Any] | None = None) -> float | DPResult:
        """差分隐私求和。

        Args:
            values: 输入数值，支持多种数据格式。
            params: 请求级 DP 参数，可包含 clip_lower/clip_upper/column/party。

        Returns:
            带噪声的求和结果。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.sum(
            values,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            clip_lower=p.get("clip_lower"),
            clip_upper=p.get("clip_upper"),
            **self._dp_format_kwargs(params),
        )

    def dp_mean(self, values: Any, params: dict[str, Any] | None = None) -> float | DPResult:
        """差分隐私均值。

        Args:
            values: 输入数值，支持多种数据格式。
            params: 请求级 DP 参数，可包含 clip_lower/clip_upper/min_count/column/party。

        Returns:
            带噪声的均值。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.mean(
            values,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            clip_lower=p.get("clip_lower"),
            clip_upper=p.get("clip_upper"),
            min_count=float(p.get("min_count", 5.0)),
            **self._dp_format_kwargs(params),
        )

    def dp_histogram(
        self, values: Any, categories: list[Any], params: dict[str, Any] | None = None
    ) -> dict[Any, float] | DPResult:
        """差分隐私直方图计数（使用联合敏感度为 1）。

        Args:
            values: 输入类别，支持多种数据格式。
            categories: 目标类别集合。
            params: 请求级 DP 参数，可包含 column/party。

        Returns:
            分桶名到带噪计数的字典。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.histogram(
            values,
            categories,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            **self._dp_format_kwargs(params),
        )

    def dp_noisy_count(self, true_count: float, params: dict[str, Any] | None = None) -> float | DPResult:
        """对已经聚合好的计数结果注入 DP 噪声。

        Args:
            true_count: 真实计数值。
            params: 请求级 DP 参数。

        Returns:
            带噪声的计数值。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.noisy_count(
            true_count,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
        )

    def dp_noisy_sum(self, true_sum: float, params: dict[str, Any] | None = None) -> float | DPResult:
        """对已经聚合好的求和结果注入 DP 噪声。

        Args:
            true_sum: 真实求和值。
            params: 请求级 DP 参数，需包含 sensitivity（或 clip_lower/clip_upper）。

        Returns:
            带噪声的求和结果。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        lower = p.get("clip_lower")
        upper = p.get("clip_upper")
        sensitivity = p.get("sensitivity")
        if sensitivity is None:
            if lower is None or upper is None:
                raise ValueError(
                    "dp_noisy_sum requires 'sensitivity' or both 'clip_lower' and 'clip_upper'"
                )
            sensitivity = float(upper) - float(lower)
        else:
            sensitivity = float(sensitivity)
        return self.dp_api.noisy_sum(
            true_sum,
            sensitivity,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
        )

    def dp_aggregate(self, df: Any, specs: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Table-Level 原位表格 DP 聚合。"""
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.dp_aggregate(
            df,
            specs,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            return_details=bool(p.get("return_details", False)),
        )

    def dp_vector_sum(self, vectors: Any, params: dict[str, Any] | None = None) -> Any:
        """高维向量 / 梯度 $L_2$ 范数截断与加噪。"""
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        max_norm = float(p.get("max_norm", 1.0))
        return self.dp_api.vector_sum(
            vectors,
            max_norm,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "gaussian")),
            return_details=bool(p.get("return_details", False)),
        )

    def dp_vector_mean(self, vectors: Any, params: dict[str, Any] | None = None) -> Any:
        """高维向量 DP 均值：L2 范数截断 + 各向同性加噪 + noisy_count 归一化。"""
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        max_norm = float(p.get("max_norm", 1.0))
        return self.dp_api.vector_mean(
            vectors,
            max_norm,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "gaussian")),
            min_count=float(p.get("min_count", 5.0)),
            return_details=bool(p.get("return_details", False)),
        )

    def dp_adaptive_clip(self, values: Any, params: dict[str, Any] | None = None) -> tuple[float, float]:
        """差分隐私自适应二分搜索估计 [0, clip_upper] 上下界。"""
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.adaptive_clip(
            values,
            float(p["epsilon"]),
            target_quantile=float(p.get("target_quantile", 0.95)),
            num_iterations=int(p.get("num_iterations", 15)),
            initial_clip=float(p.get("initial_clip", 10.0)),
        )

    def dp_groupby(
        self,
        df: Any,
        group_col: str,
        target_col: str,
        agg: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Tau-Thresholding 差分隐私 SQL Group-By 过滤。"""
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.dp_groupby(
            df,
            group_col=group_col,
            target_col=target_col,
            agg=agg,
            epsilon=float(p["epsilon"]),
            delta=float(p.get("delta", 1e-5)),
            clip_lower=p.get("clip_lower"),
            clip_upper=p.get("clip_upper"),
            mechanism=str(p.get("mechanism", "laplace")),
            return_details=bool(p.get("return_details", False)),
        )

    def dp_noisy_mean(
        self,
        true_sum: float,
        true_count: float,
        params: dict[str, Any] | None = None,
    ) -> float | DPResult:
        """对已经聚合好的 sum/count 注入 DP 噪声后得到均值。

        Args:
            true_sum: 真实求和值。
            true_count: 真实计数值。
            params: 请求级 DP 参数，需包含 sensitivity（或 clip_lower/clip_upper）。

        Returns:
            带噪声的均值。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        lower = p.get("clip_lower")
        upper = p.get("clip_upper")
        sensitivity = p.get("sensitivity")
        if sensitivity is None:
            if lower is None or upper is None:
                raise ValueError(
                    "dp_noisy_mean requires 'sensitivity' or both 'clip_lower' and 'clip_upper'"
                )
            sensitivity = float(upper) - float(lower)
        else:
            sensitivity = float(sensitivity)
        return self.dp_api.noisy_mean(
            true_sum,
            true_count,
            sensitivity,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            min_count=float(p.get("min_count", 5.0)),
        )

    def dp_noisy_histogram(
        self, true_counts: dict[Any, Any], params: dict[str, Any] | None = None
    ) -> dict[Any, float] | DPResult:
        """对已经聚合好的直方图计数注入 DP 噪声。

        Args:
            true_counts: 分桶名到真实计数的字典。
            params: 请求级 DP 参数。

        Returns:
            分桶名到带噪计数的字典。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.noisy_histogram(
            true_counts,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
        )

    def dp_chunked_count(
        self, chunks: list[Any], params: dict[str, Any] | None = None
    ) -> float | DPResult:
        """分块流式差分隐私计数。

        Args:
            chunks: 数据块列表，每块支持多种数据格式。
            params: 请求级 DP 参数，可包含 column/party。

        Returns:
            带噪声的计数值。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.chunked_count(
            chunks,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            **self._dp_format_kwargs(params),
        )

    def dp_chunked_sum(
        self, chunks: list[Any], params: dict[str, Any] | None = None
    ) -> float | DPResult:
        """分块流式差分隐私求和。

        Args:
            chunks: 数据块列表，每块支持多种数据格式。
            params: 请求级 DP 参数，必须包含 clip_lower/clip_upper，可包含 column/party。

        Returns:
            带噪声的求和结果。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.chunked_sum(
            chunks,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            clip_lower=p.get("clip_lower"),
            clip_upper=p.get("clip_upper"),
            **self._dp_format_kwargs(params),
        )

    def dp_chunked_mean(
        self, chunks: list[Any], params: dict[str, Any] | None = None
    ) -> float | DPResult:
        """分块流式差分隐私均值。

        Args:
            chunks: 数据块列表，每块支持多种数据格式。
            params: 请求级 DP 参数，必须包含 clip_lower/clip_upper，可包含 column/party。

        Returns:
            带噪声的均值。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.chunked_mean(
            chunks,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            clip_lower=p.get("clip_lower"),
            clip_upper=p.get("clip_upper"),
            min_count=float(p.get("min_count", 5.0)),
            **self._dp_format_kwargs(params),
        )

    def dp_chunked_histogram(
        self,
        chunks: list[Any],
        categories: list[Any],
        params: dict[str, Any] | None = None,
    ) -> dict[Any, float] | DPResult:
        """分块流式差分隐私直方图计数。

        Args:
            chunks: 数据块列表，每块支持多种数据格式。
            categories: 目标类别集合。
            params: 请求级 DP 参数，可包含 column/party。

        Returns:
            分桶名到带噪计数的字典。
        """
        p = self.resolver.resolve("dp", params, namespace=self.namespace)
        return self.dp_api.chunked_histogram(
            chunks,
            categories,
            float(p["epsilon"]),
            float(p.get("delta", 0.0)),
            str(p.get("mechanism", "laplace")),
            **self._dp_format_kwargs(params),
        )


    def randomized_response(self, value: bool, epsilon: float) -> bool:
        """单值 Randomized Response（与 Go ``RandomizedResponse`` 对齐）。"""
        return self.local_dp_api.randomized_response(value, epsilon)

    def orr_response(self, value: int, epsilon: float, domain_size: int) -> int:
        """单值 O-RR 响应（与 Go ``ORRResponse`` 对齐）。"""
        return self.local_dp_api.orr_response(value, epsilon, domain_size)

    def perturb_binary_batch(self, values: list[int], epsilon: float) -> list[int]:
        """批量对二值数据进行本地 DP 扰动。"""
        return cast("list[int]", self.local_dp_api.perturb_binary_batch(values, epsilon).tolist())

    def perturb_categorical_batch(
        self, values: list[Any], categories: list[Any], epsilon: float
    ) -> list[Any]:
        """批量对类别型数据进行本地 DP 扰动。"""
        return cast("list[Any]", self.local_dp_api.perturb_categorical_batch(values, categories, epsilon).tolist())

    def estimate_binary_frequency(self, reported_values: list[int], epsilon: float) -> float:
        """根据扰动后的二值样本估计真实比例为 1 的频率。"""
        return self.local_dp_api.estimate_binary_frequency(reported_values, epsilon)

    def estimate_categorical_histogram(
        self, reported_values: list[Any], categories: list[Any], epsilon: float
    ) -> dict[Any, float]:
        """根据扰动后的类别样本估计各类别的真实频率。"""
        return self.local_dp_api.estimate_categorical_histogram(
            reported_values, categories, epsilon
        )

    def k_anonymize_record(
        self,
        record: dict[str, Any],
        qi_cols: list[str],
        k: int = 5,
        hierarchies: dict[str, Any] | None = None,
    ) -> dict[str, Any] | KAnonymityRecordResult:
        """对单条记录执行 K-匿名泛化。

        Args:
            record: 原始记录字典。
            qi_cols: 准标识符列名列表。
            k: K-匿名参数，决定泛化层级。
            hierarchies: 自定义泛化层次结构，当前 MVP 仅使用内置层次。

        Returns:
            泛化后的记录字典。
        """
        params = self.resolver.resolve("k_anonymity", {"k": k}, namespace=self.namespace)
        resolved_k = params.get("k", k)
        hiers = dict(BUILTIN_HIERARCHIES)
        if hierarchies:
            hiers.update(hierarchies)
        return anonymize_record(record, qi_cols, hiers, resolved_k)

    def k_anonymize_table(
        self,
        rows: list[dict[str, Any]],
        qi_cols: list[str],
        k: int = 5,
        max_depth: int = 10,
        return_details: bool = False,
    ) -> list[dict[str, Any]] | KAnonymityResult:
        """对整张表执行 K-匿名泛化。

        Args:
            rows: 原始记录列表。
            qi_cols: 准标识符列名列表.
            k: K-匿名阈值。
            max_depth: Mondrian 最大递归深度。
            return_details: 是否返回带等价类计数的 KAnonymityResult。

        Returns:
            泛化后的记录列表；return_details=True 时返回 KAnonymityResult。
        """
        params = self.resolver.resolve(
            "k_anonymity",
            {"k": k, "max_depth": max_depth},
            namespace=self.namespace,
        )
        resolved_k = params.get("k", k)
        resolved_max_depth = params.get("max_depth", max_depth)
        return k_anonymize_table(
            rows, qi_cols, resolved_k, resolved_max_depth, return_details=return_details
        )

    def k_anonymize_dataframe(
        self,
        df: Any,
        qi_cols: list[str],
        k: int = 5,
        max_depth: int = 10,
        return_details: bool = False,
    ) -> Any:
        """对 DataFrame 执行 K-匿名泛化。

        Args:
            df: 输入 DataFrame（pandas 或 SecretFlow）。
            qi_cols: 准标识符列名列表。
            k: K-匿名阈值。
            max_depth: Mondrian 最大递归深度。
            return_details: 是否返回带等价类计数的 KAnonymityResult。

        Returns:
            泛化后的 DataFrame；return_details=True 时返回 KAnonymityResult。
        """
        params = self.resolver.resolve(
            "k_anonymity",
            {"k": k, "max_depth": max_depth},
            namespace=self.namespace,
        )
        resolved_k = params.get("k", k)
        resolved_max_depth = params.get("max_depth", max_depth)
        return k_anonymize_dataframe(
            df, qi_cols, resolved_k, resolved_max_depth, return_details=return_details
        )

    def check_distinct_l_diversity(
        self,
        records: list[dict[str, Any]],
        qi_fields: list[str],
        sensitive_field: str,
        l: int = 2,
    ) -> LDiversityResult:
        """检查数据集在给定准标识符与敏感属性下是否满足 Distinct L-多样性。"""
        return check_distinct_l_diversity(records, qi_fields, sensitive_field, l)

    def k_anonymize_ldiversity(
        self,
        rows: list[dict[str, Any]],
        qi_cols: list[str],
        sensitive_field: str,
        k: int = 5,
        l: int = 2,
        max_depth: int = 10,
        return_details: bool = False,
    ) -> list[dict[str, Any]] | KAnonymityResult:
        """对整张表执行带 L-多样性约束的 K-匿名泛化。"""
        params = self.resolver.resolve(
            "k_anonymity",
            {"k": k, "max_depth": max_depth},
            namespace=self.namespace,
        )
        resolved_k = params.get("k", k)
        resolved_max_depth = params.get("max_depth", max_depth)
        return mondrian_with_l_diversity(
            rows,
            qi_cols,
            sensitive_field,
            k=resolved_k,
            l=l,
            max_depth=resolved_max_depth,
            return_details=return_details,
        )

    # Go service 层 clampDecoys 同口径：<=0 取 3，>64 收敛为 64。
    _DEFAULT_NUM_DECOYS = 3
    _MAX_NUM_DECOYS = 64

    @classmethod
    def _clamp_num_decoys(cls, n: int) -> int:
        if n <= 0:
            return cls._DEFAULT_NUM_DECOYS
        if n > cls._MAX_NUM_DECOYS:
            return cls._MAX_NUM_DECOYS
        return n

    def obfuscate_query(
        self,
        query: str,
        num_dummies: int = 3,
        domain: str = "medical",
        medical_pool: list[str] | None = None,
        generic_pool: list[str] | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """对查询进行混淆。

        Args:
            query: 真实查询字符串。
            num_dummies: 虚假查询数量。
            domain: 查询领域，影响 dummy 查询池选择。
            medical_pool: 自定义医疗 dummy 池。
            generic_pool: 自定义通用 dummy 池。
            seed: 可选随机种子。

        Returns:
            混淆后的查询列表。
        """
        request_params: dict[str, Any] = {"num_dummies": num_dummies}
        if medical_pool is not None:
            request_params["medical_pool"] = medical_pool
        if generic_pool is not None:
            request_params["generic_pool"] = generic_pool

        params = self.resolver.resolve("qol", request_params)
        resolved_num_dummies = self._clamp_num_decoys(
            int(params.get("num_dummies", num_dummies))
        )
        resolved_medical_pool = params.get("medical_pool")
        resolved_generic_pool = params.get("generic_pool")
        result = obfuscate_query(
            query,
            resolved_num_dummies,
            domain,
            medical_pool=resolved_medical_pool,
            generic_pool=resolved_generic_pool,
            seed=seed,
        )
        return result if isinstance(result, list) else result.queries

    def obfuscate_query_batch(
        self,
        queries: list[str],
        num_dummies: int = 3,
        domain: str = "medical",
        medical_pool: list[str] | None = None,
        generic_pool: list[str] | None = None,
        seed: int | None = None,
    ) -> list[list[str]]:
        """批量对查询进行混淆。"""
        request_params: dict[str, Any] = {"num_dummies": num_dummies}
        if medical_pool is not None:
            request_params["medical_pool"] = medical_pool
        if generic_pool is not None:
            request_params["generic_pool"] = generic_pool

        params = self.resolver.resolve("qol", request_params)
        resolved_num_dummies = self._clamp_num_decoys(
            int(params.get("num_dummies", num_dummies))
        )
        resolved_medical_pool = params.get("medical_pool")
        resolved_generic_pool = params.get("generic_pool")
        raw_results = obfuscate_query_batch(
            queries,
            resolved_num_dummies,
            domain,
            medical_pool=resolved_medical_pool,
            generic_pool=resolved_generic_pool,
            seed=seed,
        )
        return [r if isinstance(r, list) else r.queries for r in raw_results]

    def recommend_and_save_params(
        self,
        values: list[float] | None = None,
        rows: list[dict[str, Any]] | None = None,
        qi_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """根据数据特点，自动推荐 DP 和 K-Anonymity 隐私参数并保存起来。"""
        from .privacy.profile import save_personalized_params
        recommendations: dict[str, Any] = {}

        # 1. 推荐差分隐私（DP）参数
        if values:
            n = len(values)
            # 计算分位数推荐截断上下界
            sorted_vals = sorted(float(v) for v in values)
            p5_idx = int(n * 0.05)
            p95_idx = int(n * 0.95)
            clip_lower = sorted_vals[p5_idx] if n > 0 else 0.0
            clip_upper = sorted_vals[min(p95_idx, n - 1)] if n > 0 else 10.0
            if clip_lower == clip_upper:
                clip_lower -= 1.0
                clip_upper += 1.0

            # 推荐 delta = 1 / (10 * n^2) 并限制在 1e-5 以内
            recommended_delta = min(1e-5, 1.0 / (10.0 * (n ** 2))) if n > 0 else 1e-5

            dp_params = {
                "epsilon": 1.0,
                "delta": recommended_delta,
                "mechanism": "laplace",
                "clip_lower": clip_lower,
                "clip_upper": clip_upper,
            }
            save_personalized_params(self.namespace, "dp", dp_params)
            recommendations["dp"] = dp_params

        # 2. 推荐 K-Anonymity 参数
        if rows:
            n = len(rows)
            # 根据数据行数推荐 k 值为行数的 10%，在 [2, 10] 之间
            recommended_k = max(2, min(10, n // 10))
            kano_params = {
                "k": recommended_k,
                "max_depth": 10,
            }
            save_personalized_params(self.namespace, "k_anonymity", kano_params)
            recommendations["k_anonymity"] = kano_params

        return recommendations

    def process_medical_data(self, records: list[dict[str, str]]) -> dict[str, Any]:
        """对医疗数据集执行分类分级与 L4/L5 敏感数据强抹平脱敏。
        
        Args:
            records: 包含 27 个医疗/身份字段的数据记录列表。

        Returns:
            包含 classification_report、sanitized_data 与 summary 的双输出结构。
        """
        from .medical_pipeline.pipeline import process_medical_dataset, MedicalPipelineResult
        res: MedicalPipelineResult = process_medical_dataset(records)
        return {
            "classification_report": res.classification_report,
            "sanitized_data": res.sanitized_data,
            "summary": res.summary,
        }

    @staticmethod
    def _resolve_medical_domain(domain: str) -> str:
        """归一化入站数据源标识并执行 Fail-Closed 校验（与 Go SanitizeMedical* 对齐）。"""
        from .naming import DS_KANGYANG, DS_YIBAO, UnknownDataSourceError, resolve_inbound

        try:
            ds_id = resolve_inbound(domain)
        except (UnknownDataSourceError, PermissionError) as exc:
            raise ValueError(f"unsupported datasource: {domain}") from exc
        if ds_id not in (DS_YIBAO, DS_KANGYANG):
            raise ValueError(f"unsupported datasource: {ds_id}")
        return ds_id

    def sanitize_medical_record(self, record: dict[str, str], domain: str) -> dict[str, str]:
        """单条医疗记录脱敏（``/v1/medical/sanitize``，与 Go 对齐）。"""
        from .medical_pipeline.pipeline import get_default_pipeline

        self._resolve_medical_domain(domain)
        pipeline = get_default_pipeline()
        return {key: pipeline.sanitize_field(key, "" if val is None else str(val)) for key, val in record.items()}

    def sanitize_medical_batch(
        self, records: list[dict[str, str]], domain: str
    ) -> list[dict[str, str]]:
        """批量医疗记录脱敏（``/v1/medical/sanitize/batch``，与 Go 对齐）。"""
        self._resolve_medical_domain(domain)
        return [self.sanitize_medical_record(record, domain) for record in records]

    def budget_remaining(self) -> dict[str, float]:
        """查询当前命名空间下剩余隐私预算。

        Returns:
            当前命名空间下 epsilon 与 delta 的剩余量字典。
        """
        return self.registry.get_or_create(self.namespace).remaining()

    def budget_reset(self, namespace: str | None = None) -> dict[str, float]:
        """重置指定（或当前）命名空间的隐私预算。

        Args:
            namespace: 可选命名空间，默认为当前服务配置的命名空间。

        Returns:
            重置后当前命名空间的预算剩余字典。
        """
        ns = namespace or self.namespace
        acct = self.registry.get_or_create(ns)
        acct.reset()
        return acct.remaining()
