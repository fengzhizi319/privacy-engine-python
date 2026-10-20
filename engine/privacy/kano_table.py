"""数据集级 K-匿名（K-Anonymity）处理模块 / Dataset-Level K-Anonymity Primitive API Implementation.

中文说明：
使用 Mondrian 多维分区算法对整张表进行 K-匿名泛化，确保每个等价组的大小
至少为 k。对数值型准标识符输出区间泛化，对分类型准标识符输出取值集合。
内置输入校验、结构化日志与 Prometheus 指标埋点。

English Description:
Dataset-level K-anonymity using the Mondrian multidimensional partitioning
algorithm. Generalizes numeric QIs to intervals and categorical QIs to value sets.
Ensures each equivalence class has at least k records.
Built-in input validation, structured logging, and Prometheus metrics instrumentation.

扩展能力 / Key Features:
- Mondrian 多维分区：递归选择跨度最大维度进行中位数切分。
- Pandas 向量化加速：优先使用 pandas 向量化实现，避免纯 Python 循环开销。
- 结构化日志：记录 k 值、等价组数、处理行数等上下文信息。
- 输入校验：统一的参数合法性检查，快速失败并给出清晰错误信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ..observability.logging_config import get_logger
from ..observability.metrics import KANO_DURATION, KANO_OPERATIONS_TOTAL, observe_duration
from .kano import BUILTIN_HIERARCHIES

# Module-level structured logger for dataset-level K-anonymity operations
logger = get_logger(__name__)


@dataclass
class KAnonymityResult:
    """K-匿名处理结果及结构化元数据包装。

    Attributes:
        value: 泛化后的记录列表 (List[Dict]) 或 DataFrame。
        k: K-匿名阈值参数。
        qi_cols: 准标识符列名列表。
        equivalence_classes_count: 形成的等价类（Equivalence Class）分区总数。
    """

    value: Any
    k: int
    qi_cols: list[str]
    equivalence_classes_count: int = 1

    def to_arrow(self):
        """将 KAnonymityResult 包装转换为附带 K-匿名 Metadata 的 PyArrow Table。

        执行步骤：
        1. 提取 K-匿名的隐私元数据（k 值、准标识符列表、等价组数量）构造 JSON。
        2. 将元数据编码存储为 Schema Metadata Key `b"k_anonymity_metadata"`。
        3. 根据 `value`（Dict 列表或 DataFrame）转换为 `pyarrow.Table`。
        4. 替换 Table Schema Metadata 后导出。
        """
        import json

        import pyarrow as pa

        meta = {
            "k": str(self.k),
            "qi_cols": str(self.qi_cols),
            "equivalence_classes_count": str(self.equivalence_classes_count),
        }
        custom_metadata = {b"k_anonymity_metadata": json.dumps(meta).encode("utf-8")}

        if isinstance(self.value, list):
            table = pa.Table.from_pylist(self.value)
        else:
            try:
                import pandas as pd
                if isinstance(self.value, pd.DataFrame):
                    table = pa.Table.from_pandas(self.value)
                else:
                    arr = pa.array([str(self.value)])
                    table = pa.Table.from_arrays([arr], names=["kanonymity_value"])
            except ImportError:
                arr = pa.array([str(self.value)])
                table = pa.Table.from_arrays([arr], names=["kanonymity_value"])

        existing_meta = table.schema.metadata or {}
        merged_meta = {**existing_meta, **custom_metadata}
        return table.replace_schema_metadata(merged_meta)


def _is_numeric(value: Any) -> bool:
    """判断一个值是否为数值类型（int/float），排除布尔值。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _span(records: list[dict[str, Any]], col: str) -> float:
    """计算某列的跨度，用于选择分割维度。"""
    values = [r.get(col) for r in records if r.get(col) is not None]
    if not values:
        return 0.0
    if all(_is_numeric(v) for v in values):
        numeric_values = cast("list[float]", values)
        return float(max(numeric_values) - min(numeric_values))
    return float(len({str(v) for v in values}) - 1)


def _choose_dimension(records: list[dict[str, Any]], qi_cols: list[str]) -> str:
    """选择跨度最大的准标识符列作为当前分割维度。"""
    spans = {col: _span(records, col) for col in qi_cols}
    return max(spans, key=spans.get)  # type: ignore[arg-type]


def _median_split(
    records: list[dict[str, Any]], dim: str, k: int
) -> int | None:
    """按指定维度的中位数进行分割，并确保左右两部分均不少于 k 条记录。"""
    if len(records) < 2 * k:
        return None

    def _sort_key(record: dict[str, Any]) -> Any:
        value = record.get(dim)
        if _is_numeric(value):
            return value
        return str(value)

    sorted_records = sorted(records, key=_sort_key)
    mid = len(sorted_records) // 2
    split_idx = max(k, min(mid, len(sorted_records) - k))
    if split_idx < k or len(sorted_records) - split_idx < k:
        return None
    return split_idx


def _format_num(value: float) -> str:
    """按 Go ``formatNum`` 口径输出数值（整数去掉小数尾）。"""
    return str(int(value)) if float(value).is_integer() else str(value)


def _hierarchy_generalize(col: str, values: list[str]) -> str | None:
    """优先用内置固定步长层次泛化，未注册该列时返回 None。"""
    func = BUILTIN_HIERARCHIES.get(col)
    if func is None or not values:
        return None
    for level in range(1, 5):
        first = func(values[0], level)
        if all(func(value, level) == first for value in values[1:]):
            return first
    return None


def _generalized_value(col: str, raw_values: list[Any]) -> str:
    """计算单个准标识符列在等价组内的泛化值（与 Go ``generalize`` 对齐）。"""
    numeric_values: list[float] = []
    all_numeric = True
    for value in raw_values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            all_numeric = False
            break
    if all_numeric and numeric_values:
        low, high = min(numeric_values), max(numeric_values)
        if low == high:
            return _format_num(low)
        hierarchy = _hierarchy_generalize(col, [str(value) for value in raw_values])
        if hierarchy is not None:
            return hierarchy
        return f"[{_format_num(low)}-{_format_num(high)}]"

    unique = sorted({str(value) for value in raw_values})
    if len(unique) == 1:
        return unique[0]
    hierarchy = _hierarchy_generalize(col, [str(value) for value in raw_values])
    if hierarchy is not None:
        return hierarchy
    return "{" + ",".join(unique) + "}"


def _generalize(
    records: list[dict[str, Any]], qi_cols: list[str]
) -> list[dict[str, Any]]:
    """对等价组内的记录进行泛化。"""
    if not records:
        return []

    generalized: dict[str, str] = {}
    for col in qi_cols:
        values = [r.get(col) for r in records if r.get(col) is not None]
        if values:
            generalized[col] = _generalized_value(col, values)

    result: list[dict[str, Any]] = []
    for record in records:
        new_record = dict(record)
        new_record.update(generalized)
        result.append(new_record)
    return result


def k_anonymize_table(
    rows: list[dict[str, Any]],
    qi_cols: list[str],
    k: int = 5,
    max_depth: int = 10,
    return_details: bool = False,
) -> list[dict[str, Any]] | KAnonymityResult:
    """对整张表执行 Mondrian K-匿名泛化 / Mondrian K-Anonymity on Full Table.

    执行步骤 / Execution Steps:
    1. 校验准标识符列 qi_cols 非空；行数不足 2k 时整表作为单一等价组泛化
       （与 Go kano.Anonymize 一致，不对 len(rows) < k 报错）。
       (Validate qi_cols non-empty; tables smaller than k stay as one group,
       matching Go kano.Anonymize which does not reject len(rows) < k)
    2. 计算多维分割：使用 Pandas 向量化或递归选定跨度最大的准标识符维度。
       (Multidimensional partitioning: select max-span QI dimension via pandas or recursion)
    3. 寻找合法中位数切分点，使得切分后的左右两个等价组记录数均 >= k。
       (Find valid median split ensuring both partitions have >= k records)
    4. 对终止切分的叶子节点（等价组）实施区间/组合集合泛化。
       (Apply interval/set generalization on leaf equivalence classes)
    5. 记录结构化日志并返回泛化后的记录。
       (Emit structured log and return generalized records)

    Args:
        rows: 输入记录列表 / Input record list.
        qi_cols: 准标识符列名列表 / Quasi-Identifier column names.
        k: K-匿名阈值 / K-anonymity threshold.
        max_depth: 最大递归深度 / Maximum recursion depth.
        return_details: 是否返回 KAnonymityResult / Whether to return result struct.

    Returns:
        泛化后的记录列表或 KAnonymityResult / Generalized records or result struct.

    Raises:
        ValueError: 当 k < 2 或 qi_cols 为空时 / When k < 2 or qi_cols is empty.
    """
    if not rows:
        return []
    if k < 2:
        raise ValueError(f"k must be at least 2 for meaningful anonymity, got {k}")
    if not qi_cols:
        raise ValueError("qi_cols must not be empty")
    # 缺失的 QI 列按空串参与泛化（Go 以 map 取值，缺失键等价于 ""），不报错。
    rows = [
        {**row, **{col: "" for col in qi_cols if col not in row}} for row in rows
    ]

    KANO_OPERATIONS_TOTAL.labels(operation="table").inc()
    with observe_duration(KANO_DURATION, operation="table"):
        try:
            import pandas as pd

            # 使用 Pandas 向量化优化版 Mondrian 算法，避免递归的 Python list sorting 开销
            df = pd.DataFrame(rows)

            def _mondrian_pd(sub_df: pd.DataFrame, depth: int) -> pd.DataFrame:
                if len(sub_df) < 2 * k or depth <= 0:
                    # 泛化当前等价组
                    gen_df = sub_df.copy()
                    for col in qi_cols:
                        col_vals = gen_df[col]
                        raw_values = [
                            value
                            for value in col_vals.tolist()
                            if value is not None and not (isinstance(value, float) and pd.isna(value))
                        ]
                        if raw_values:
                            gen_df[col] = _generalized_value(col, raw_values)
                    return gen_df

                # 选择跨度最大的分割列
                max_span = -1.0
                best_dim = None
                for col in qi_cols:
                    col_vals = sub_df[col].dropna()
                    if not col_vals.empty:
                        is_num = pd.api.types.is_numeric_dtype(col_vals) and not pd.api.types.is_bool_dtype(col_vals)
                        span = float(col_vals.max() - col_vals.min()) if is_num else float(col_vals.nunique() - 1)
                        if span > max_span:
                            max_span = span
                            best_dim = col

                if best_dim is None:
                    return sub_df

                # 按选定维度进行中位数划分，并满足左右两部分均不少于 k 条记录
                sorted_sub = sub_df.sort_values(by=best_dim)
                mid = len(sorted_sub) // 2
                split_idx = max(k, min(mid, len(sorted_sub) - k))
                if split_idx < k or len(sorted_sub) - split_idx < k:
                    # 无法满足边界要求，直接泛化
                    return _mondrian_pd(sub_df, 0)

                left = _mondrian_pd(sorted_sub.iloc[:split_idx], depth - 1)
                right = _mondrian_pd(sorted_sub.iloc[split_idx:], depth - 1)
                return pd.concat([left, right])

            result_df = _mondrian_pd(df, max_depth)
            res_list = cast("list[dict[str, Any]]", result_df.to_dict(orient="records"))
            # 真实等价类计数：按泛化后的准标识符组合分组统计，
            # 而非 len(rows)//k 的估算（单叶子或尾组合并时估算值会虚高）。
            eq_count = len({tuple(str(r.get(c)) for c in qi_cols) for r in res_list}) if res_list else 0
            logger.info(
                "kano_table_completed",
                extra={
                    "k": k,
                    "qi_cols": qi_cols,
                    "num_rows": len(rows),
                    "equivalence_classes": eq_count,
                    "max_depth": max_depth,
                },
            )
            if return_details:
                return KAnonymityResult(value=res_list, k=k, qi_cols=qi_cols, equivalence_classes_count=eq_count)
            return res_list
        except ImportError:
            pass

        def _mondrian(
            records: list[dict[str, Any]], depth: int
        ) -> list[dict[str, Any]]:
            if len(records) < 2 * k or depth <= 0:
                return _generalize(records, qi_cols)

            dim = _choose_dimension(records, qi_cols)
            split_idx = _median_split(records, dim, k)
            if split_idx is None:
                return _generalize(records, qi_cols)

            def _sort_key(record: dict[str, Any]) -> Any:
                value = record.get(dim)
                if _is_numeric(value):
                    return value
                return str(value)

            sorted_records = sorted(records, key=_sort_key)
            left = _mondrian(sorted_records[:split_idx], depth - 1)
            right = _mondrian(sorted_records[split_idx:], depth - 1)
            return left + right

        final_res = _mondrian(rows, max_depth)
        # 真实等价类计数：按泛化后的准标识符组合分组统计（与 pandas 路径口径一致）
        eq_count = len({tuple(str(r.get(c)) for c in qi_cols) for r in final_res}) if final_res else 0
        logger.info(
            "kano_table_completed",
            extra={
                "k": k,
                "qi_cols": qi_cols,
                "num_rows": len(rows),
                "equivalence_classes": eq_count,
                "max_depth": max_depth,
            },
        )
        if return_details:
            return KAnonymityResult(value=final_res, k=k, qi_cols=qi_cols, equivalence_classes_count=eq_count)
        return final_res


def k_anonymize_dataframe(
    df: Any,
    qi_cols: list[str],
    k: int = 5,
    max_depth: int = 10,
    return_details: bool = False,
) -> Any:
    """对 DataFrame 执行 Mondrian K-匿名泛化。

    支持 pandas DataFrame 与 SecretFlow DataFrame（H/V）。
    内部转换为 records 后调用 k_anonymize_table，再按原类型返回。

    Args:
        df: 输入 DataFrame。
        qi_cols: 准标识符列名列表。
        k: K-匿名阈值。
        max_depth: 最大递归深度。
        return_details: 是否返回 KAnonymityResult 结构体。

    Returns:
        泛化后的 DataFrame（pandas DataFrame），或当 return_details=True 时返回 KAnonymityResult。
    """
    from .data_adapters import from_records, to_records

    KANO_OPERATIONS_TOTAL.labels(operation="dataframe").inc()
    with observe_duration(KANO_DURATION, operation="dataframe"):
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                if not qi_cols:
                    raise ValueError("qi_cols must not be empty")
                # 与 Go KAnonymizeDataFrame 一致：DataFrame 转 records 后走同一套
                # kano.Anonymize 语义，保证与表级接口输出完全一致。
                anonymized = k_anonymize_table(
                    cast("list[dict[str, Any]]", df.to_dict(orient="records")),
                    qi_cols,
                    k=k,
                    max_depth=max_depth,
                    return_details=return_details,
                )
                if return_details:
                    return anonymized
                assert isinstance(anonymized, list)
                return pd.DataFrame(anonymized)
        except ImportError:
            pass

        records = to_records(df)
        anonymized = k_anonymize_table(
            records, qi_cols, k=k, max_depth=max_depth, return_details=return_details
        )
        if return_details:
            return anonymized
        assert isinstance(anonymized, list)
        return from_records(anonymized, df)


def distinct_field_count(records: list[dict[str, Any]], field: str) -> int:
    """统计 records 中 field 的去重非空取值数（空值不计入）。"""
    seen: set[str] = set()
    for r in records:
        val = r.get(field)
        if val is not None and str(val) != "":
            seen.add(str(val))
    return len(seen)


def mondrian_with_l_diversity(
    rows: list[dict[str, Any]],
    qi_cols: list[str],
    sensitive_field: str,
    k: int = 5,
    l: int = 2,
    max_depth: int = 10,
    return_details: bool = False,
) -> list[dict[str, Any]] | KAnonymityResult:
    """带 L-多样性约束的表级 Mondrian K-匿名泛化算法（与 Go sdk/kano 对齐）。

    在切分过程中，只有当切分后左右两部分的敏感属性去重数均 >= l 时才允许该次切分；
    否则放弃该切分，使当前等价类保持为一个更大但满足 L-多样性的叶子节点，从而在生成阶段杜绝同质性攻击。

    Args:
        rows: 输入行记录列表。
        qi_cols: 准标识符（QI）列名列表。
        sensitive_field: 敏感属性列名。
        k: K-匿名阈值（>=2）。
        l: L-多样性阈值（>=1，若 <1 自动收敛为 1）。
        max_depth: 最大递归深度。
        return_details: 是否返回 KAnonymityResult 结构体。

    Returns:
        泛化后的记录列表，或附带元数据的 KAnonymityResult。
    """
    if not rows:
        return [] if not return_details else KAnonymityResult(value=[], k=k, qi_cols=qi_cols, equivalence_classes_count=0)
    if k < 2:
        k = 2
    if l < 1:
        l = 1
    if not qi_cols:
        raise ValueError("qi_cols must not be empty")

    rows = [
        {**row, **{col: "" for col in qi_cols if col not in row}} for row in rows
    ]

    KANO_OPERATIONS_TOTAL.labels(operation="table_ldiversity").inc()
    with observe_duration(KANO_DURATION, operation="table_ldiversity"):
        def _can_split_ldiv(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
            return distinct_field_count(left, sensitive_field) >= l and distinct_field_count(right, sensitive_field) >= l

        def _mondrian_ldiv(
            records: list[dict[str, Any]], depth: int
        ) -> list[dict[str, Any]]:
            if len(records) < 2 * k or depth <= 0:
                return _generalize(records, qi_cols)

            dim = _choose_dimension(records, qi_cols)
            split_idx = _median_split(records, dim, k)
            if split_idx is None:
                return _generalize(records, qi_cols)

            def _sort_key(record: dict[str, Any]) -> Any:
                value = record.get(dim)
                if _is_numeric(value):
                    return value
                return str(value)

            sorted_records = sorted(records, key=_sort_key)
            left_part = sorted_records[:split_idx]
            right_part = sorted_records[split_idx:]

            # 校验 L-多样性约束：若切分后任意一侧敏感属性多样性跌破 l，则放弃此次切分
            if not _can_split_ldiv(left_part, right_part):
                return _generalize(records, qi_cols)

            left = _mondrian_ldiv(left_part, depth - 1)
            right = _mondrian_ldiv(right_part, depth - 1)
            return left + right

        final_res = _mondrian_ldiv(rows, max_depth)
        eq_count = len({tuple(str(r.get(c)) for c in qi_cols) for r in final_res}) if final_res else 0
        logger.info(
            "kano_table_ldiversity_completed",
            extra={
                "k": k,
                "l": l,
                "sensitive_field": sensitive_field,
                "qi_cols": qi_cols,
                "num_rows": len(rows),
                "equivalence_classes": eq_count,
            },
        )
        if return_details:
            return KAnonymityResult(value=final_res, k=k, qi_cols=qi_cols, equivalence_classes_count=eq_count)
        return final_res


def anonymize_with_l_diversity(
    records: list[dict[str, Any]],
    qi_fields: list[str],
    sensitive_field: str,
    k: int = 2,
    l: int = 2,
    return_details: bool = False,
) -> list[dict[str, Any]] | KAnonymityResult:
    """记录集级 L-多样性 K-匿名泛化入口（与 Go sdk/kano AnonymizeWithLDiversity 对齐）。"""
    return mondrian_with_l_diversity(
        records,
        qi_cols=qi_fields,
        sensitive_field=sensitive_field,
        k=k,
        l=l,
        max_depth=32,
        return_details=return_details,
    )
