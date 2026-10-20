"""隐私参数解析与校验模块。

中文说明：
从 YAML 配置文件加载各隐私原语的默认参数，并支持请求级参数覆盖。
提供参数校验能力，确保 DP epsilon 为正、K-Anonymity k 不小于 2 等。
支持个性化推荐参数的持久化与命名空间隔离。

English Description:
Privacy parameter resolver and validator. Loads default parameters from a YAML
profile and allows request-level overrides with built-in validation rules.
Supports personalized parameter persistence with namespace isolation.
"""

from __future__ import annotations

import os
import threading
from typing import Any, cast

import yaml

from ..observability.logging_config import get_logger
from ..observability.metrics import PROFILE_RESOLVE_TOTAL

# Module-level structured logger for profile resolution events
logger = get_logger(__name__)


def default_params(primitive: str) -> dict[str, Any]:
    """获取指定隐私原语的默认参数 / Get Default Parameters for a Privacy Primitive.

    中文说明：
    返回内置的各原语默认参数配置，作为参数解析的最底层回退值。

    English Description:
    Returns built-in default parameter configuration for the specified primitive,
    serving as the lowest-priority fallback in parameter resolution.

    Args:
        primitive: 隐私原语名称 / Privacy primitive name,
            e.g. "dp", "k_anonymity", "sanitization", "qol", "classification".

    Returns:
        对应原语的默认参数字典；若不存在则返回空字典。
        Default parameter dict for the primitive; empty dict if unknown.
    """
    return cast("dict[str, Any]", {
        "dp": {"epsilon": 1.0, "delta": 0.0, "mechanism": "laplace"},
        "k_anonymity": {"k": 5, "l": 2, "t": 0.2, "max_depth": 10},
        "sanitization": {"engine": "mask"},
        "qol": {
            "num_dummies": 3,
            "medical_pool": [
                "高血压患者的日常饮食建议",
                "糖尿病患者运动注意事项",
                "冠心病的早期症状有哪些",
                "流感疫苗接种人群建议",
                "儿童常见过敏反应处理",
            ],
            "generic_pool": [
                "天气预报查询",
                "附近医院挂号流程",
                "健康档案如何查询",
                "医保报销比例说明",
                "体检报告解读指南",
            ],
        },
        "classification": {
            "version": "1.0.0",
            "default_level": "L3",
            "enable_rule_engine": True,
            "enable_small_ner": False,
            "enable_llm": False,
            "icd10_l4_intervals": [
                {"start": "B20", "end": "B24"},   # HIV 相关诊断编码区间
                {"start": "A50", "end": "A53"},   # 梅毒（先天性/早期/晚期）
                {"start": "A54", "end": "A64"},   # 其他性传播疾病（淋病、衣原体等）
                {"start": "F20", "end": "F29"},   # 精神分裂症等重型精神疾病区间
                {"start": "C00", "end": "C97"},   # 恶性肿瘤编码区间
            ],
            "genomic_keywords": [
                "brca1",
                "brca2",
                "tp53",
                "rs",
                "snp",
                "cnv",
                "genome",
                "genomic",
                "gene",
                "mutation",
                "variant",
            ],
            "public_field_whitelist": ["public_report", "annual_summary", "科普"],
            "operational_field_patterns": ["turnover_rate", "device_usage", "inventory"],
            "manual_override": {},
        },
    }.get(primitive, {}))


def validate(primitive: str, params: dict[str, Any]) -> None:
    """校验隐私原语参数是否合法 / Validate Privacy Primitive Parameters.

    中文说明：
    对合并后的参数进行业务规则校验，不合法时快速失败并抛出清晰错误信息。

    English Description:
    Validates merged parameters against business rules. Fails fast with a clear
    error message when parameters are invalid.

    执行步骤 / Execution Steps:
    1. 根据 primitive 类型分派到对应校验逻辑。
       (Dispatch to validation logic based on primitive type)
    2. 校验不通过时抛出 ValueError 并记录结构化日志。
       (Raise ValueError and log structured event on validation failure)

    Args:
        primitive: 隐私原语名称 / Privacy primitive name.
        params: 合并后的参数 / Merged parameters dict.

    Raises:
        ValueError: 当参数不满足业务规则时抛出 / When parameters violate business rules.
            - DP epsilon 必须为正数 / DP epsilon must be positive.
            - K-Anonymity k 必须大于等于 2 / K-Anonymity k must be >= 2.
    """
    if primitive == "dp":
        eps = float(params.get("epsilon", 0))
        if eps <= 0:
            logger.warning(
                "profile_validation_failed",
                extra={"primitive": primitive, "field": "epsilon", "value": eps},
            )
            raise ValueError("DP epsilon must be positive")
    elif primitive == "k_anonymity":
        k = int(params.get("k", 0))
        if k < 2:
            logger.warning(
                "profile_validation_failed",
                extra={"primitive": primitive, "field": "k", "value": k},
            )
            raise ValueError("K-Anonymity k must be >= 2")


class ParameterResolver:
    """参数解析器 / Parameter Resolver.

    中文说明：
    负责将配置文件中的默认参数、YAML profile 中的原语参数以及请求参数
    按优先级合并，并做合法性校验。

    English Description:
    Merges built-in defaults, YAML profile primitive parameters, personalized
    profile parameters, and request-level overrides by priority, then validates
    the final result.

    Attributes:
        profile: 解析后的 YAML profile 字典 / Parsed YAML profile dict.

    参数优先级（从高到低） / Parameter Priority (high to low):
        1. 请求级参数 request_params
        2. 个性化推荐参数 config/personalized-profiles.yaml
        3. YAML profile 中 primitives.<primitive> 的配置
        4. 内置 default_params 默认值
    """

    def __init__(self, profile_path: str | None = None):
        """初始化参数解析器 / Initialize Parameter Resolver.

        执行步骤 / Execution Steps:
        1. 检查 profile_path 是否存在。
           (Check if profile_path exists)
        2. 存在则加载 YAML 内容到 self.profile。
           (Load YAML content into self.profile if file exists)
        3. 记录结构化加载日志。
           (Emit structured load log event)

        Args:
            profile_path: YAML 配置文件路径 / Path to YAML profile file.
                若提供且文件存在，则加载其内容；否则 profile 为空字典。
                (If provided and file exists, loads content; otherwise profile is empty dict)
        """
        self.profile: dict[str, Any] = {}
        if profile_path and os.path.exists(profile_path):
            with open(profile_path, encoding="utf-8") as f:
                self.profile = yaml.safe_load(f) or {}
            logger.info(
                "profile_loaded",
                extra={"profile_path": profile_path, "keys": list(self.profile.keys())},
            )

    def resolve(
        self,
        primitive: str,
        request_params: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """解析并合并指定隐私原语的参数 / Resolve and Merge Parameters for a Primitive.

        执行步骤 / Execution Steps:
        1. 以内置 default_params 为基础层。
           (Start with built-in default_params as base layer)
        2. 使用 YAML profile 中的原语配置进行覆盖。
           (Override with YAML profile primitive config)
        3. 若指定 namespace，合并个性化推荐参数。
           (Merge personalized profile params if namespace specified)
        4. 使用请求参数做最终覆盖。
           (Apply request params as final override)
        5. 执行参数校验并记录指标。
           (Validate parameters and record metrics)

        Args:
            primitive: 隐私原语名称 / Privacy primitive name.
            request_params: 请求级参数，优先级最高 / Request-level params (highest priority).
            namespace: 命名空间 / Namespace for personalized profile lookup.

        Returns:
            合并并校验后的完整参数字典 / Merged and validated parameter dict.

        Raises:
            ValueError: 当参数校验失败时抛出 / When parameter validation fails.
        """
        # 以内置默认值为基础
        params = dict(default_params(primitive))
        # 使用 YAML profile 中的原语配置进行覆盖
        primitives = self.profile.get("primitives") or {}
        if primitive in primitives:
            params.update(primitives[primitive])

        # 使用个性化推荐保存的参数进行覆盖（若指定了 namespace）
        if namespace:
            personalized_path = os.environ.get("PRIVACY_PERSONALIZED_PROFILE", "config/personalized-profiles.yaml")
            if os.path.exists(personalized_path):
                with _profile_lock:
                    try:
                        with open(personalized_path, encoding="utf-8") as f:
                            personalized_data = yaml.safe_load(f) or {}
                        namespace_config = personalized_data.get(namespace, {})
                        primitive_config = namespace_config.get(primitive, {})
                        if primitive_config:
                            params.update(primitive_config)
                    except Exception as e:
                        logger.warning(
                            "personalized_profile_load_failed",
                            extra={"namespace": namespace, "primitive": primitive, "error": str(e)},
                        )

        # 使用请求参数做最终覆盖
        if request_params:
            params.update(request_params)

        try:
            validate(primitive, params)
            PROFILE_RESOLVE_TOTAL.labels(primitive=primitive, status="success").inc()
        except ValueError:
            PROFILE_RESOLVE_TOTAL.labels(primitive=primitive, status="validation_failed").inc()
            raise

        return params


_resolver_cache: dict[str, ParameterResolver] = {}
_profile_lock = threading.Lock()


def get_resolver(profile_path: str | None = None) -> ParameterResolver:
    """获取参数解析器单例/缓存实例 / Get Cached ParameterResolver Instance.

    中文说明：
    避免重复解析 YAML 文件，相同路径复用已创建的解析器实例。

    English Description:
    Returns a cached ParameterResolver instance for the given profile path,
    avoiding redundant YAML parsing.

    Args:
        profile_path: YAML 配置文件路径 / Path to YAML profile file.
            若未指定，从环境变量 PRIVACY_PROFILE 读取。
            (Falls back to PRIVACY_PROFILE env var if not specified)

    Returns:
        ParameterResolver 实例 / ParameterResolver instance.
    """
    path = profile_path or os.environ.get("PRIVACY_PROFILE", "privacy-profile.yaml")

    abs_path = os.path.abspath(path)
    # 仅在解析器缓存未命中（新建 resolver）时记录，避免每次请求重复输出
    if abs_path not in _resolver_cache:
        logger.debug("profile_resolver_created", extra={"abs_path": abs_path})
        _resolver_cache[abs_path] = ParameterResolver(abs_path)
    return _resolver_cache[abs_path]


def save_personalized_params(namespace: str, primitive: str, params: dict[str, Any]) -> None:
    """持久化保存推荐的个性化参数 / Persist Personalized Parameters.

    中文说明：
    将推荐参数写入 config/personalized-profiles.yaml，按 namespace + primitive 分层存储。

    English Description:
    Persists recommended parameters to config/personalized-profiles.yaml, organized
    by namespace and primitive layers.

    执行步骤 / Execution Steps:
    1. 加锁并读取现有个性化配置文件。
       (Acquire lock and read existing personalized profile)
    2. 合并新参数到对应 namespace/primitive 路径。
       (Merge new params into the namespace/primitive path)
    3. 写回 YAML 文件并记录结构化日志。
       (Write back YAML file and emit structured log)

    Args:
        namespace: 命名空间 / Namespace identifier.
        primitive: 隐私原语名称 / Privacy primitive name.
        params: 待保存的参数字典 / Parameters dict to persist.
    """
    if not namespace:
        raise ValueError("namespace must not be empty when saving personalized params")
    if not primitive:
        raise ValueError("primitive must not be empty when saving personalized params")

    personalized_path = os.environ.get("PRIVACY_PERSONALIZED_PROFILE", "config/personalized-profiles.yaml")
    with _profile_lock:
        personalized_data: dict[str, Any] = {}
        if os.path.exists(personalized_path):
            try:
                with open(personalized_path, encoding="utf-8") as f:
                    personalized_data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(
                    "personalized_profile_read_failed",
                    extra={"path": personalized_path, "error": str(e)},
                )

        if namespace not in personalized_data:
            personalized_data[namespace] = {}
        if primitive not in personalized_data[namespace]:
            personalized_data[namespace][primitive] = {}

        personalized_data[namespace][primitive].update(params)

        try:
            dir_name = os.path.dirname(personalized_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(personalized_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(personalized_data, f, allow_unicode=True)
            logger.info(
                "personalized_params_saved",
                extra={"namespace": namespace, "primitive": primitive, "path": personalized_path},
            )
        except Exception as e:
            logger.error(
                "personalized_profile_save_failed",
                extra={"namespace": namespace, "primitive": primitive, "error": str(e)},
            )
            raise

