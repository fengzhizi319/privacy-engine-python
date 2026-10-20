"""K-匿名（K-Anonymity）记录级处理模块 / K-Anonymity Record-Level Primitive API Implementation.

中文说明：
提供针对准标识符（Quasi-Identifiers, QI）的泛化函数与内置泛化层次结构，
支持对单条记录按指定 k 值进行泛化，降低重识别风险。
内置输入校验、结构化日志与 Prometheus 指标埋点。

English Description:
K-anonymity primitives for record-level generalization. Provides hierarchy
generalization functions for common QIs such as age, zipcode, gender, and salary.
Supports single-record and batch generalization with configurable hierarchies.
Built-in input validation, structured logging, and Prometheus metrics instrumentation.

扩展能力 / Key Features:
- 枚举类型安全：QIType / GeneralizationStrategy 枚举避免裸字符串拼写错误。
- 结构化日志：每次操作记录 k 值、准标识符列、泛化层级等上下文信息。
- 输入校验：统一的参数合法性检查，快速失败并给出清晰错误信息。
- 可扩展泛化层次：支持自定义泛化函数与内置层次结构映射表。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..observability.logging_config import get_logger
from ..observability.metrics import KANO_DURATION, KANO_OPERATIONS_TOTAL, observe_duration

# Module-level structured logger for K-anonymity operations
logger = get_logger(__name__)

# 大数据集超时保护阈值（秒）/ Large dataset timeout threshold (seconds)
DEFAULT_BATCH_TIMEOUT_SECONDS = 60.0
# 大数据集记录数阈值 / Large dataset record count threshold
LARGE_DATASET_THRESHOLD = 10000


class QIType(str, Enum):
    """准标识符类型枚举 / Quasi-Identifier Type Enum.

    继承 str 保证与字符串的向后兼容性：QIType.AGE == "age" 为 True。
    IDE 自动补全 + 静态类型检查，避免裸字符串拼写错误。
    """

    AGE = "age"
    ZIPCODE = "zipcode"
    GENDER = "gender"
    SALARY = "salary"
    EDUCATION = "education"


class GeneralizationStrategy(str, Enum):
    """泛化策略枚举 / Generalization Strategy Enum.

    继承 str 保证与字符串的向后兼容性。
    用于标识不同的泛化方法（区间泛化、集合泛化、抑制等）。
    """

    INTERVAL = "interval"      # 数值区间泛化 / Numeric interval generalization
    SET = "set"                # 分类集合泛化 / Categorical set generalization
    SUPPRESSION = "suppression"  # 完全抑制 / Full suppression (*)
    PREFIX = "prefix"          # 前缀保留泛化 / Prefix-preserving generalization


def _validate_k(k: int) -> None:
    """校验 K-匿名参数有效性 / Validate K-anonymity parameter.

    Args:
        k: K-匿名参数。

    Raises:
        ValueError: 当 k 不是正整数或小于 2 时抛出。
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(f"k must be an integer, got {type(k).__name__}")
    if k < 2:
        raise ValueError(f"k must be at least 2 for meaningful anonymity, got {k}")


def _validate_qi_cols(qi_cols: list[str]) -> None:
    """校验准标识符列名列表有效性 / Validate QI column list.

    Args:
        qi_cols: 准标识符列名列表。

    Raises:
        ValueError: 当列表为空或包含非字符串元素时抛出。
    """
    if not qi_cols:
        raise ValueError("qi_cols must not be empty")
    for col in qi_cols:
        if not isinstance(col, str) or not col.strip():
            raise ValueError(f"each qi_col must be a non-empty string, got {col!r}")


@dataclass
class KAnonymityRecordResult:
    """K-匿名单条记录泛化结果及结构化元数据包装。

    Attributes:
        value: 泛化后的记录字典。
        k: K-匿名阈值参数。
        qi_cols: 准标识符列名列表。
        applied_level: 实际应用的泛化层级（由 k 值启发式决定）。
        hierarchies_used: 本次使用的泛化层次函数名称映射。
    """

    value: dict[str, Any]
    k: int
    qi_cols: list[str]
    applied_level: int = 1
    hierarchies_used: dict[str, str] = field(default_factory=dict)

    def to_arrow(self):
        """将 KAnonymityRecordResult 包装转换为附带 K-匿名 Metadata 的 PyArrow Table。

        执行步骤：
        1. 提取 K-匿名记录级元数据（k 值、准标识符列表、应用层级）构造 JSON。
        2. 将元数据编码存储为 Schema Metadata Key `b"k_anonymity_record_metadata"`。
        3. 根据 `value` 字典构造 PyArrow Table。
        4. 替换 Table Schema Metadata 后导出。
        """
        import json

        import pyarrow as pa

        meta = {
            "k": str(self.k),
            "qi_cols": str(self.qi_cols),
            "applied_level": str(self.applied_level),
            "hierarchies_used": str(self.hierarchies_used),
        }
        custom_metadata = {b"k_anonymity_record_metadata": json.dumps(meta).encode("utf-8")}

        keys = pa.array(list(self.value.keys()))
        vals = pa.array([str(v) for v in self.value.values()])
        table = pa.Table.from_arrays([keys, vals], names=["field", "generalized_value"])

        existing_meta = table.schema.metadata or {}
        merged_meta = {**existing_meta, **custom_metadata}
        return table.replace_schema_metadata(merged_meta)


@dataclass
class GroupDiversity:
    """单个等价类的多样性统计 / Single equivalence class diversity statistics."""

    group_index: int
    record_count: int
    distinct_count: int
    sensitive_values: dict[str, int]
    is_compliant: bool


@dataclass
class LDiversityResult:
    """L-多样性合规检查报告 / L-Diversity Compliance Verification Report."""

    is_compliant: bool
    l: int
    min_diversity: int
    violations: int
    group_count: int
    group_stats: list[GroupDiversity] = field(default_factory=list)


def check_distinct_l_diversity(
    records: list[dict[str, Any]],
    qi_fields: list[str],
    sensitive_field: str,
    l: int = 2,
) -> LDiversityResult:
    """校验数据集在给定准标识符与敏感属性下是否满足 Distinct L-Diversity。

    中文说明：
    按准标识符 QI 字段将记录集合划分为各个等价类，并对每个等价类中敏感属性的不同非空取值数进行统计。
    若每个等价类中敏感属性的不同取值数均 >= l，则满足 L-多样性，有效防御同质性攻击（Homogeneity Attack）。

    Args:
        records: 输入记录列表。
        qi_fields: 准标识符（QI）字段列表。
        sensitive_field: 敏感属性字段名。
        l: 要求的不同敏感属性取值数下限（Distinct L-Diversity），l <= 1 时按 1 处理。

    Returns:
        LDiversityResult: 包含是否合规、最小多样性、违规等价类数、统计详情的结构体。
    """
    if l <= 1:
        l = 1
    if not records:
        return LDiversityResult(
            is_compliant=True,
            l=l,
            min_diversity=0,
            violations=0,
            group_count=0,
            group_stats=[],
        )

    # 按准标识符 QI 组合键将记录归入等价类
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        key = "|".join(str(r.get(q, "")) for q in qi_fields) + "|"
        groups.setdefault(key, []).append(r)

    # 等价类按 QI 组合键排序后再编号，保证结果确定性可复现（与 Go 对齐）
    qi_keys = sorted(groups.keys())
    group_stats: list[GroupDiversity] = []
    violations = 0
    min_div = float("inf")

    for idx, key in enumerate(qi_keys, start=1):
        grp = groups[key]
        val_counts: dict[str, int] = {}
        for r in grp:
            val = r.get(sensitive_field)
            if val is not None and str(val) != "":
                sval = str(val)
                val_counts[sval] = val_counts.get(sval, 0) + 1
        distinct_count = len(val_counts)
        if distinct_count < min_div:
            min_div = distinct_count

        compliant = distinct_count >= l
        if not compliant:
            violations += 1

        group_stats.append(
            GroupDiversity(
                group_index=idx,
                record_count=len(grp),
                distinct_count=distinct_count,
                sensitive_values=val_counts,
                is_compliant=compliant,
            )
        )

    final_min_div = int(min_div) if min_div != float("inf") else 0
    is_compliant = violations == 0

    return LDiversityResult(
        is_compliant=is_compliant,
        l=l,
        min_diversity=final_min_div,
        violations=violations,
        group_count=len(groups),
        group_stats=group_stats,
    )


# 泛化层次函数类型：输入原始值与泛化层级，输出泛化后的字符串
GeneralizationHierarchy = Callable[[str, int], str]


def adaptive_age_hierarchy(
    value: str | int | float,
    under_60_interval: int = 3,
    senior_interval: int = 2,
    output_format: str = "floor",
) -> str:
    """单条记录自适应分段年龄泛化函数 (Adaptive Single-Record Age Generalization).

    针对单条/流式数据中的准标识符年龄 (Age) 执行自适应分段 K-匿名泛化：
    1. < 60 岁 (青年/中年)：年龄精细度对基础健康建议影响较小，采用较粗区间（默认 3 岁或 5 岁区间），
       计算公式为：generalized_age = age - (age % under_60_interval)。
       例如：30, 31, 32 均取余 3，泛化向下取整为 30。
    2. >= 60 岁 (老年/康养高敏)：考虑年龄对老年康养照护与慢性病建议的高敏感度影响因子，采用较窄精细区间（2 岁区间），
       计算公式为：generalized_age = age - (age % senior_interval)。
       例如：60, 61 泛化为 60；62, 63 泛化为 62。

    Args:
        value: 原始年龄（整数、浮点数或数字字符串）。
        under_60_interval: 60 岁以下泛化区间大小（默认 3 岁，也可配置为 5 岁）。
        senior_interval: 60 岁及以上康养高敏泛化区间大小（默认 2 岁）。
        output_format: "floor" (输出向下对齐数字如 "30") 或 "range" (输出区间如 "[30-32]")。

    Returns:
        泛化后的年龄字符串。若输入无法解析为数字（脏数据），原样返回。
    """
    try:
        clean_val = str(value).rstrip("岁").strip()
        age = int(float(clean_val))
    except (ValueError, TypeError):
        return str(value)

    if age < 0:
        return str(value)

    if age < 60:
        interval = max(1, under_60_interval)
        start = age - (age % interval)
    else:
        interval = max(1, senior_interval)
        start = age - (age % interval)

    if output_format == "range":
        end = start + interval - 1
        return f"[{start}-{end}]" if interval > 1 else str(start)
    else:
        return str(start)


def age_hierarchy(value: str, level: int) -> str:
    """年龄泛化层次函数。

    根据 level 将具体年龄泛化为区间：
    - level 0: 原始值
    - level 1: 5 岁区间，如 "[25-30]"
    - level 2: 10 岁区间
    - level 3: 20 岁区间
    - level >= 4: "*"

    Args:
        value: 原始年龄字符串，需可转换为整数。
        level: 泛化层级，数值越大粒度越粗。

    Returns:
        泛化后的年龄表示。
    """
    # 与 salary_hierarchy 保持一致的容错：非数字输入（脏数据）不应使
    # 整个 K-匿名管线崩溃，level >= 1 时抑制为 "*"，level 0 时原样返回。
    try:
        age = int(float(value))
    except (ValueError, TypeError):
        return "*" if level >= 1 else value
    if level == 0:
        return value
    if level == 1:
        start = (age // 5) * 5
        return f"[{start}-{start + 5}]"
    if level == 2:
        start = (age // 10) * 10
        return f"[{start}-{start + 10}]"
    if level == 3:
        start = (age // 20) * 20
        return f"[{start}-{start + 20}]"
    return "*"


def zipcode_hierarchy(value: str, level: int) -> str:
    """邮编泛化层次函数。

    根据 level 逐步隐藏邮编后几位：
    - level 0: 原始值
    - level 1: 保留前 3 位
    - level 2: 保留前 2 位
    - level 3: 保留前 1 位
    - level >= 4 或长度不足: "*"

    Args:
        value: 原始邮编字符串。
        level: 泛化层级。

    Returns:
        泛化后的邮编表示。
    """
    if level == 0:
        return value
    if level == 1 and len(value) >= 3:
        return value[:3] + "***"
    if level == 2 and len(value) >= 2:
        return value[:2] + "****"
    if level == 3 and len(value) >= 1:
        return value[0] + "*****"
    return "*"


def gender_hierarchy(value: str, level: int) -> str:
    """性别泛化层次函数 / Gender Generalization Hierarchy.

    - level 0: 原始值 / Original value
    - level >= 1: "*" (完全抑制 / Full suppression)

    Args:
        value: 原始性别字符串 / Original gender string.
        level: 泛化层级 / Generalization level.

    Returns:
        泛化后的性别表示 / Generalized gender representation.
    """
    return "*" if level >= 1 else value


def salary_hierarchy(value: str, level: int) -> str:
    """薪资泛化层次函数 / Salary Generalization Hierarchy.

    根据 level 将具体薪资泛化为区间：
    - level 0: 原始值 / Original value
    - level 1: 5K 区间 / 5K interval, e.g. "[10K-15K]"
    - level 2: 10K 区间 / 10K interval
    - level 3: 50K 区间 / 50K interval
    - level >= 4: "*" (完全抑制 / Full suppression)

    Args:
        value: 原始薪资字符串（单位：K）/ Original salary string (unit: K).
        level: 泛化层级 / Generalization level.

    Returns:
        泛化后的薪资表示 / Generalized salary representation.
    """
    try:
        salary = int(float(value))
    except (ValueError, TypeError):
        return "*" if level >= 1 else value
    if level == 0:
        return value
    if level == 1:
        start = (salary // 5) * 5
        return f"[{start}K-{start + 5}K]"
    if level == 2:
        start = (salary // 10) * 10
        return f"[{start}K-{start + 10}K]"
    if level == 3:
        start = (salary // 50) * 50
        return f"[{start}K-{start + 50}K]"
    return "*"


def education_hierarchy(value: str, level: int) -> str:
    """学历泛化层次函数 / Education Generalization Hierarchy.

    根据 level 将具体学历泛化为更宽泛的类别：
    - level 0: 原始值 / Original value
    - level 1: 合并为 "高等教育" / "基础教育" / Merge into "Higher" / "Basic"
    - level >= 2: "*" (完全抑制 / Full suppression)

    Args:
        value: 原始学历字符串 / Original education string.
        level: 泛化层级 / Generalization level.

    Returns:
        泛化后的学历表示 / Generalized education representation.
    """
    if level == 0:
        return value
    higher_edu = {"本科", "硕士", "博士", "博士后", "MBA", "EMBA", "bachelor", "master", "phd", "doctorate"}
    if level == 1:
        if value.lower() in higher_edu or value in higher_edu:
            return "高等教育"
        return "基础教育"
    return "*"


# 内置准标识符泛化层次结构映射表 / Built-in QI Generalization Hierarchy Registry
BUILTIN_HIERARCHIES: dict[str, GeneralizationHierarchy] = {
    QIType.AGE.value: age_hierarchy,
    QIType.ZIPCODE.value: zipcode_hierarchy,
    QIType.GENDER.value: gender_hierarchy,
    QIType.SALARY.value: salary_hierarchy,
    QIType.EDUCATION.value: education_hierarchy,
}


def choose_level(k: int, max_level: int) -> int:
    """根据 k 值选择泛化层级 / Select Generalization Level by K Value.

    采用启发式策略：level 与 k // 5 成正比，但不超过 max_level 且至少为 1。

    执行步骤 / Execution Steps:
    1. 校验 k 和 max_level 参数有效性。
       (Validate k and max_level parameters)
    2. 计算启发式层级 level = k // 5。
       (Compute heuristic level = k // 5)
    3. 将层级限制在 [1, max_level] 区间内。
       (Clamp level to [1, max_level] range)

    Args:
        k: K-匿名参数，目标匿名集大小 / K-anonymity parameter.
        max_level: 该字段支持的最大泛化层级 / Maximum supported generalization level.

    Returns:
        选定的泛化层级（正整数）/ Selected generalization level (positive integer).

    Raises:
        ValueError: 当 k < 2 或 max_level < 1 时 / When k < 2 or max_level < 1.
    """
    _validate_k(k)
    if max_level < 1:
        raise ValueError(f"max_level must be at least 1, got {max_level}")
    level = k // 5
    return max(1, min(level, max_level))


def anonymize_record(
    record: dict[str, Any],
    qi_cols: list[str],
    hierarchies: dict[str, GeneralizationHierarchy],
    k: int,
    return_details: bool = False,
) -> dict[str, Any] | KAnonymityRecordResult:
    """对单条记录按 K-匿名要求进行泛化 / Generalize Single Record for K-Anonymity.

    执行步骤 / Execution Steps:
    1. 校验 k 值和 qi_cols 参数有效性。
       (Validate k value and qi_cols parameters)
    2. 增加 `KANO_OPERATIONS_TOTAL` 操作指标计数。
       (Increment KANO_OPERATIONS_TOTAL metrics counter)
    3. 拷贝输入记录字典，仅处理 `qi_cols` 中指定的准标识符列。
       (Copy input record, process only specified QI columns)
    4. 寻找匹配的泛化层次函数（若未显式传参则查询内置 `BUILTIN_HIERARCHIES`）。
       (Resolve hierarchy function from explicit param or built-in registry)
    5. 根据 `k` 智选粒度层级 `choose_level`，应用泛化替换准标识符属性。
       (Select level via choose_level and apply generalization)
    6. 记录结构化日志并返回泛化后的记录或封装结果。
       (Emit structured log and return generalized record or result)

    Args:
        record: 待泛化的记录字典 / Record dict to generalize.
        qi_cols: 准标识符列名列表 / Quasi-Identifier column names.
        hierarchies: 自定义泛化层次函数映射 / Custom hierarchy function mapping.
        k: K-匿名阈值参数 / K-anonymity threshold parameter.
        return_details: 是否返回 KAnonymityRecordResult / Whether to return result struct.

    Returns:
        泛化后的记录字典或 KAnonymityRecordResult / Generalized record or result struct.

    Raises:
        ValueError: 当 k < 2 或 qi_cols 为空时 / When k < 2 or qi_cols is empty.
    """
    _validate_k(k)
    _validate_qi_cols(qi_cols)
    if not isinstance(record, dict):
        raise ValueError(f"record must be a dict, got {type(record).__name__}")

    KANO_OPERATIONS_TOTAL.labels(operation="record").inc()
    with observe_duration(KANO_DURATION, operation="record"):
        result = dict(record)
        effective_hierarchies = {**BUILTIN_HIERARCHIES, **(hierarchies or {})}
        applied_levels: dict[str, int] = {}
        hierarchies_used: dict[str, str] = {}
        for col in qi_cols:
            h = effective_hierarchies.get(col)
            val = result.get(col)
            if h is not None and isinstance(val, str):
                max_level = 4 if col not in (QIType.GENDER.value, QIType.EDUCATION.value) else (
                    1 if col == QIType.GENDER.value else 2
                )
                level = choose_level(k, max_level)
                result[col] = h(val, level)
                applied_levels[col] = level
                hierarchies_used[col] = h.__name__

        logger.info(
            "kano_anonymize_record_completed",
            extra={
                "k": k,
                "qi_cols": qi_cols,
                "applied_levels": applied_levels,
                "num_qi_processed": len(applied_levels),
            },
        )

        if return_details:
            avg_level = (
                sum(applied_levels.values()) // len(applied_levels)
                if applied_levels
                else 1
            )
            return KAnonymityRecordResult(
                value=result,
                k=k,
                qi_cols=qi_cols,
                applied_level=avg_level,
                hierarchies_used=hierarchies_used,
            )
        return result


def anonymize_records_batch(
    records: list[dict[str, Any]],
    qi_cols: list[str],
    hierarchies: dict[str, GeneralizationHierarchy] | None = None,
    k: int = 5,
    return_details: bool = False,
    timeout_seconds: float | None = None,
) -> list[dict[str, Any]] | KAnonymityRecordResult:
    """批量对多条记录按 K-匿名要求进行泛化 / Batch Generalize Records for K-Anonymity.

    执行步骤 / Execution Steps:
    1. 校验 k 值、qi_cols 和 records 参数有效性。
       (Validate k, qi_cols, and records parameters)
    2. 增加 `KANO_OPERATIONS_TOTAL` 操作指标计数。
       (Increment KANO_OPERATIONS_TOTAL metrics counter)
    3. 遍历每条记录调用 `anonymize_record` 执行泛化，支持超时保护。
       (Iterate records and apply anonymize_record per record with timeout protection)
    4. 记录结构化日志并返回泛化后的记录列表或封装结果。
       (Emit structured log and return generalized records or result)

    Args:
        records: 待泛化的记录字典列表 / List of record dicts to generalize.
        qi_cols: 准标识符列名列表 / Quasi-Identifier column names.
        hierarchies: 可选自定义泛化层次函数映射 / Optional custom hierarchy mapping.
        k: K-匿名阈值参数 / K-anonymity threshold parameter.
        return_details: 是否返回 KAnonymityRecordResult / Whether to return result struct.
        timeout_seconds: 超时保护阈值（秒），None 表示使用默认值 / Timeout threshold (seconds), None for default.

    Returns:
        泛化后的记录列表或 KAnonymityRecordResult / Generalized records or result struct.

    Raises:
        ValueError: 当 records 为空或 k < 2 时 / When records is empty or k < 2.
        TimeoutError: 当处理时间超过超时阈值时 / When processing exceeds timeout threshold.
    """
    _validate_k(k)
    _validate_qi_cols(qi_cols)
    if not records:
        raise ValueError("records must not be empty")
    if not isinstance(records, list):
        raise ValueError(f"records must be a list, got {type(records).__name__}")

    # 超时保护：大数据集使用更严格的超时 / Timeout protection for large datasets
    effective_timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_BATCH_TIMEOUT_SECONDS
    if len(records) > LARGE_DATASET_THRESHOLD:
        logger.warning(
            "Processing large dataset, timeout protection enabled",
            extra={"num_records": len(records), "timeout_seconds": effective_timeout},
        )

    start_time = time.perf_counter()
    KANO_OPERATIONS_TOTAL.labels(operation="record_batch").inc()
    with observe_duration(KANO_DURATION, operation="record_batch"):
        effective_hierarchies = hierarchies or {}
        generalized: list[dict[str, Any]] = []
        total_levels: list[int] = []
        all_hierarchies_used: dict[str, str] = {}

        for i, record in enumerate(records):
            # 超时检查 / Timeout check
            elapsed = time.perf_counter() - start_time
            if elapsed > effective_timeout:
                KANO_DURATION.labels(operation="record_batch_timeout").observe(elapsed)
                logger.error(
                    "Batch anonymization timeout",
                    extra={
                        "processed_records": i,
                        "total_records": len(records),
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": effective_timeout,
                    },
                )
                raise TimeoutError(
                    f"Batch anonymization timed out after {elapsed:.2f}s "
                    f"(processed {i}/{len(records)} records, timeout={effective_timeout}s)"
                )

            result = anonymize_record(
                record, qi_cols, effective_hierarchies, k, return_details=True
            )
            if isinstance(result, KAnonymityRecordResult):
                generalized.append(result.value)
                total_levels.append(result.applied_level)
                all_hierarchies_used.update(result.hierarchies_used)
            else:
                generalized.append(result)

        duration = time.perf_counter() - start_time

        logger.info(
            "kano_anonymize_records_batch_completed",
            extra={
                "k": k,
                "qi_cols": qi_cols,
                "num_records": len(records),
                "avg_level": sum(total_levels) // max(1, len(total_levels)),
                "duration_seconds": round(duration, 4),
            },
        )

        if return_details:
            avg_level = sum(total_levels) // max(1, len(total_levels)) if total_levels else 1
            return KAnonymityRecordResult(
                value={"records": generalized, "count": len(generalized)},
                k=k,
                qi_cols=qi_cols,
                applied_level=avg_level,
                hierarchies_used=all_hierarchies_used,
            )
        return generalized
