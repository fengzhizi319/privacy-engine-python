"""数据集级 K-匿名（Mondrian）算法测试 / Dataset-Level K-Anonymity (Mondrian) Algorithm Tests.

中文说明：
本模块全面验证数据集级 K-匿名实现，涵盖以下核心场景：

1. Mondrian 多维递归划分算法 / Mondrian Multi-Dimensional Recursive Partitioning:
   - 数值型准标识符（QI）泛化为区间 [min-max] / Numeric QI generalization to intervals
   - 分类型 QI 泛化为集合 {v1,v2} / Categorical QI generalization to sets
   - 等价组大小保证 >= k / Equivalence class size guarantee >= k

2. 记录级泛化 / Record-Level Generalization:
   - anonymize_record 单条记录泛化 / Single record generalization
   - anonymize_records_batch 批量泛化 / Batch record generalization
   - 自定义泛化层次结构（salary/education）/ Custom hierarchies

3. 输入校验与异常处理 / Input Validation & Error Handling:
   - k < 2 拒绝 / k < 2 rejection
   - 空 QI 列拒绝 / Empty QI columns rejection
   - 数据量不足 k 时拒绝 / Insufficient data rejection

4. 扩展特性 / Extended Features:
   - Pandas DataFrame 接口 / Pandas DataFrame interface
   - KAnonymityResult 详情返回 / Detailed result object
   - PyArrow IPC 元数据导出 / PyArrow metadata export
   - Prometheus 指标计数 / Prometheus metrics counting

English Description:
Comprehensive tests for dataset-level K-anonymity using the Mondrian algorithm,
covering numeric/categorical generalization, equivalence class validation,
record-level operations, input validation, and extended features (DataFrame,
PyArrow, Prometheus metrics).
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from engine.privacy.kano import (
    GeneralizationStrategy,
    KAnonymityRecordResult,
    QIType,
    anonymize_record,
    anonymize_records_batch,
    choose_level,
    education_hierarchy,
    salary_hierarchy,
)
from engine.privacy.kano_table import (
    k_anonymize_dataframe,
    k_anonymize_table,
)


class TestKAnonymizeTable:
    """Mondrian 算法单元测试 / Mondrian Algorithm Unit Tests.

    Mondrian 算法核心思想：
    1. 选择跨度最大的 QI 维度 / Select dimension with largest span
    2. 按中位数将数据划分为两半 / Split data at median
    3. 递归直到每个分区 >= k 条记录 / Recurse until each partition has >= k records
    4. 对每个等价组内的 QI 值进行泛化 / Generalize QI values within each equivalence class
    """

    def test_numeric_qi_generalizes_to_intervals(self) -> None:
        """验证数值型 QI 被泛化为区间表示 [min-max]。

        测试数据包含两个自然聚类（25-27 和 55-57），k=3 时
        Mondrian 应将它们划分为至少 2 个等价组，每组的 age
        被泛化为区间形式（如 [25-27]）。
        """
        rows = [
            {"age": 25, "zipcode": "100001", "disease": "A"},
            {"age": 26, "zipcode": "100002", "disease": "B"},
            {"age": 27, "zipcode": "100003", "disease": "C"},
            {"age": 55, "zipcode": "200001", "disease": "D"},
            {"age": 56, "zipcode": "200002", "disease": "E"},
            {"age": 57, "zipcode": "200003", "disease": "F"},
        ]
        result = k_anonymize_table(rows, ["age", "zipcode"], k=3)
        assert len(result) == len(rows)
        # 敏感字段应保持不变
        assert {r["disease"] for r in result} == {"A", "B", "C", "D", "E", "F"}
        # age 应被泛化为区间
        for r in result:
            assert "[" in str(r["age"])

    def test_categorical_qi_generalizes_to_set(self) -> None:
        """验证分类型 QI 被泛化为集合表示 {v1,v2}。

        gender 是典型的分类型 QI，当等价组内包含 M 和 F 时，
        泛化结果应为 {F,M}（集合表示），而非保留单一值。
        """
        rows = [
            {"gender": "M", "age": 25, "salary": 5000},
            {"gender": "M", "age": 26, "salary": 6000},
            {"gender": "F", "age": 35, "salary": 7000},
            {"gender": "F", "age": 36, "salary": 8000},
        ]
        result = k_anonymize_table(rows, ["gender", "age"], k=2)
        assert len(result) == 4
        # 分类型 gender 可能被泛化为 {F,M} 或保持原值
        gender_values = {str(r["gender"]) for r in result}
        assert all(v in {"M", "F", "{F,M}", "{M,F}"} for v in gender_values)

    def test_each_equivalence_group_size_at_least_k(self) -> None:
        """验证 K-匿名核心保证：每个等价组至少包含 k 条记录。

        这是 K-匿名的数学定义：对于任意等价类 EC_i，
        |EC_i| >= k。使用 20 条记录、k=5，至少应形成 4 个等价组。
        """
        rows = [
            {"age": i, "gender": "M" if i % 2 == 0 else "F"}
            for i in range(20)
        ]
        result = k_anonymize_table(rows, ["age", "gender"], k=5)
        # 按 age/gender 泛化结果统计等价组大小
        from collections import Counter

        group_counts = Counter(
            (str(r["age"]), str(r["gender"])) for r in result
        )
        assert all(c >= 5 for c in group_counts.values())

    def test_empty_input(self) -> None:
        """边界条件：空输入应返回空列表，不触发 Mondrian 递归。"""
        assert k_anonymize_table([], ["age"], k=2) == []

    def test_input_smaller_than_k_returns_single_group(self) -> None:
        """与 Go kano.Anonymize 对齐：行数 < k 不报错，整表作为单一等价组。"""
        result = k_anonymize_table([{"age": 1}], ["age"], k=2)
        assert len(result) == 1

    def test_missing_qi_cols_treated_as_empty(self) -> None:
        """与 Go map 取值一致：缺失的 QI 列按空串参与泛化，不报错。"""
        result = k_anonymize_table([{"age": 1}, {"age": 2}], ["gender"], k=2)
        assert len(result) == 2

    def test_k_anonymize_dataframe(self) -> None:
        """验证 Pandas DataFrame 接口的 K-匿名处理。

        k_anonymize_dataframe 内部将 DataFrame 转为 records 列表，
        调用 Mondrian 算法后再转回 DataFrame，保持索引和列类型。
        """
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "age": [25, 26, 27, 55, 56, 57],
                "zipcode": ["100001", "100002", "100003", "200001", "200002", "200003"],
                "disease": ["A", "B", "C", "D", "E", "F"],
            }
        )
        result = k_anonymize_dataframe(df, ["age", "zipcode"], k=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6
        assert set(result["disease"]) == {"A", "B", "C", "D", "E", "F"}

    def test_k_anonymize_table_records_metric(self) -> None:
        """验证每次 table 级 K-匿名操作都会递增 Prometheus 计数器。

        指标名：privacy_kano_operations_total{operation="table"}
        采用 Before/After 差值断言，隔离其他测试的影响。
        """
        before = REGISTRY.get_sample_value(
            "privacy_kano_operations_total", {"operation": "table"}
        ) or 0.0
        k_anonymize_table(
            [{"age": 25, "zipcode": "100001"}, {"age": 26, "zipcode": "100002"}],
            ["age", "zipcode"],
            k=2,
        )
        after = REGISTRY.get_sample_value(
            "privacy_kano_operations_total", {"operation": "table"}
        )
        assert after == before + 1

    def test_k_anonymize_table_return_details(self) -> None:
        from engine.privacy.kano_table import KAnonymityResult

        rows = [
            {"age": 25, "zipcode": "100001", "disease": "A"},
            {"age": 26, "zipcode": "100002", "disease": "B"},
            {"age": 27, "zipcode": "100003", "disease": "C"},
            {"age": 55, "zipcode": "200001", "disease": "D"},
            {"age": 56, "zipcode": "200002", "disease": "E"},
            {"age": 57, "zipcode": "200003", "disease": "F"},
        ]
        result = k_anonymize_table(rows, ["age", "zipcode"], k=3, return_details=True)
        assert isinstance(result, KAnonymityResult)
        assert result.k == 3
        assert result.qi_cols == ["age", "zipcode"]
        assert result.equivalence_classes_count >= 1
        assert isinstance(result.value, list)
        assert len(result.value) == 6

    def test_k_anonymize_dataframe_return_details(self) -> None:
        pd = pytest.importorskip("pandas")
        from engine.privacy.kano_table import KAnonymityResult

        df = pd.DataFrame(
            {
                "age": [25, 26, 27, 55, 56, 57],
                "zipcode": ["100001", "100002", "100003", "200001", "200002", "200003"],
                "disease": ["A", "B", "C", "D", "E", "F"],
            }
        )
        result = k_anonymize_dataframe(df, ["age", "zipcode"], k=3, return_details=True)
        assert isinstance(result, KAnonymityResult)
        assert result.k == 3
        assert result.qi_cols == ["age", "zipcode"]
        assert isinstance(result.value, list)
        assert len(result.value) == 6

    def test_k_anonymity_result_to_arrow(self) -> None:
        pa = pytest.importorskip("pyarrow")
        from engine.privacy.kano_table import KAnonymityResult

        result = KAnonymityResult(
            value=[{"age": "[25-30]", "zipcode": "100***"}],
            k=3,
            qi_cols=["age", "zipcode"],
            equivalence_classes_count=2,
        )
        table = result.to_arrow()
        assert isinstance(table, pa.Table)
        assert b"k_anonymity_metadata" in table.schema.metadata


class TestAnonymizeRecord:
    """记录级 K-匿名泛化测试 / Record-Level K-Anonymity Generalization Tests.

    与数据集级 Mondrian 不同，记录级泛化使用预定义的层次结构
    （如 age → [25-30] → [20-40] → *），根据 k 值自动选择泛化层级。
    """

    def test_anonymize_record_basic(self) -> None:
        """基本记录泛化：QI 列被泛化，非 QI 列保持不变。"""
        record = {"age": "30", "zipcode": "100001", "name": "Alice"}
        result = anonymize_record(record, ["age", "zipcode"], {}, k=10)
        assert isinstance(result, dict)
        # age 应被泛化
        assert result["age"] != "30" or "[" in str(result["age"])
        # name 不在 qi_cols 中，保持不变
        assert result["name"] == "Alice"

    def test_anonymize_record_return_details(self) -> None:
        record = {"age": "30", "zipcode": "100001", "name": "Alice"}
        result = anonymize_record(
            record, ["age", "zipcode"], {}, k=10, return_details=True
        )
        assert isinstance(result, KAnonymityRecordResult)
        assert result.k == 10
        assert result.qi_cols == ["age", "zipcode"]
        assert result.applied_level >= 1
        assert "age" in result.hierarchies_used
        assert isinstance(result.value, dict)

    def test_k_anonymity_record_result_to_arrow(self) -> None:
        pa = pytest.importorskip("pyarrow")
        result = KAnonymityRecordResult(
            value={"age": "[30-35]", "zipcode": "100***"},
            k=10,
            qi_cols=["age", "zipcode"],
            applied_level=2,
            hierarchies_used={"age": "age_hierarchy", "zipcode": "zipcode_hierarchy"},
        )
        table = result.to_arrow()
        assert isinstance(table, pa.Table)
        assert b"k_anonymity_record_metadata" in table.schema.metadata


class TestQITypeEnum:
    """准标识符类型枚举测试 / Quasi-Identifier Type Enum Tests.

    QIType 枚举定义了系统内置支持的准标识符类型，
    每种类型对应不同的泛化策略（数值区间/前缀截断/集合泛化）。
    """

    def test_qi_type_enum_values(self) -> None:
        """验证所有内置 QI 类型的枚举值正确。"""
        assert QIType.AGE == "age"
        assert QIType.ZIPCODE == "zipcode"
        assert QIType.GENDER == "gender"
        assert QIType.SALARY == "salary"
        assert QIType.EDUCATION == "education"


class TestGeneralizationStrategyEnum:
    """泛化策略枚举测试 / Generalization Strategy Enum Tests.

    泛化策略决定了 QI 值如何被抽象化：
    - INTERVAL: 数值 → 区间 [min-max] / Numeric to interval
    - SET: 分类 → 集合 {v1,v2} / Categorical to set
    - SUPPRESSION: 完全抑制为 * / Full suppression
    - PREFIX: 保留前缀，后缀替换为 * / Prefix retention
    """

    def test_generalization_strategy_enum_values(self) -> None:
        """验证所有泛化策略枚举值正确。"""
        assert GeneralizationStrategy.INTERVAL == "interval"
        assert GeneralizationStrategy.SET == "set"
        assert GeneralizationStrategy.SUPPRESSION == "suppression"
        assert GeneralizationStrategy.PREFIX == "prefix"


class TestInputValidationKano:
    """输入校验测试 / Input Validation Tests.

    验证 K-匿名模块的防御性编程：对非法参数（k<2、空 QI 列、
    非字典记录、无效 max_level）及时抛出 ValueError。
    """

    def test_anonymize_record_k_less_than_2_raises(self) -> None:
        """k < 2 违反 K-匿名定义（至少需要 2 条记录才能形成等价组）。"""
        with pytest.raises(ValueError, match="k must be at least 2"):
            anonymize_record({"age": "25"}, ["age"], {}, k=1)

    def test_anonymize_record_empty_qi_cols_raises(self) -> None:
        with pytest.raises(ValueError, match="qi_cols must not be empty"):
            anonymize_record({"age": "25"}, [], {}, k=5)

    def test_anonymize_record_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="record must be a dict"):
            anonymize_record(["not", "a", "dict"], ["age"], {}, k=5)  # type: ignore

    def test_choose_level_invalid_max_level_raises(self) -> None:
        with pytest.raises(ValueError, match="max_level must be at least 1"):
            choose_level(5, 0)

    def test_k_anonymize_table_k_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be at least 2"):
            k_anonymize_table([{"age": 25}], ["age"], k=1)


class TestNewHierarchies:
    """新增泛化层次函数测试 / Custom Hierarchy Function Tests.

    泛化层次定义了从精确值到完全抑制的渐进路径：
    - salary: 精确值 → 5K区间 → 10K区间 → 50K区间 → *
    - education: 精确值 → 教育阶段 → *
    """

    def test_salary_hierarchy(self) -> None:
        """验证薪资泛化层次：level 0=原值, 1=5K区间, 2=10K区间, 3=50K区间, 4=*。"""
        assert salary_hierarchy("15", 0) == "15"
        assert salary_hierarchy("15", 1) == "[15K-20K]"
        assert salary_hierarchy("15", 2) == "[10K-20K]"
        assert salary_hierarchy("15", 3) == "[0K-50K]"
        assert salary_hierarchy("15", 4) == "*"

    def test_education_hierarchy(self) -> None:
        assert education_hierarchy("本科", 0) == "本科"
        assert education_hierarchy("本科", 1) == "高等教育"
        assert education_hierarchy("高中", 1) == "基础教育"
        assert education_hierarchy("本科", 2) == "*"


class TestAnonymizeRecordsBatch:
    """批量记录泛化测试 / Batch Record Generalization Tests.

    anonymize_records_batch 对多条记录应用相同的泛化策略，
    适用于需要对数据集逐行泛化但又不需要 Mondrian 全局优化的场景。
    """

    def test_anonymize_records_batch_basic(self) -> None:
        """基本批量泛化：输出长度与输入一致。"""
        records = [
            {"age": "25", "zipcode": "100001"},
            {"age": "30", "zipcode": "100002"},
        ]
        result = anonymize_records_batch(records, ["age", "zipcode"], k=2)
        assert len(result) == 2

    def test_anonymize_records_batch_return_details(self) -> None:
        records = [
            {"age": "25", "zipcode": "100001"},
            {"age": "30", "zipcode": "100002"},
        ]
        result = anonymize_records_batch(records, ["age", "zipcode"], k=2, return_details=True)
        assert isinstance(result, KAnonymityRecordResult)
        assert result.k == 2

    def test_anonymize_records_batch_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="records must not be empty"):
            anonymize_records_batch([], ["age"], k=2)

    def test_adaptive_age_hierarchy(self) -> None:
        """测试单条记录自适应分段年龄泛化：<60岁按3岁区间(减余数)，>=60岁按2岁精细康养区间(减余数)。"""
        from engine.privacy.kano import adaptive_age_hierarchy

        # < 60 岁：3 岁区间测试 (30, 31, 32 -> 30)
        assert adaptive_age_hierarchy(30, under_60_interval=3, senior_interval=2, output_format="floor") == "30"
        assert adaptive_age_hierarchy(31, under_60_interval=3, senior_interval=2, output_format="floor") == "30"
        assert adaptive_age_hierarchy(32, under_60_interval=3, senior_interval=2, output_format="floor") == "30"
        assert adaptive_age_hierarchy(28, under_60_interval=3, senior_interval=2, output_format="floor") == "27"
        assert adaptive_age_hierarchy(59, under_60_interval=3, senior_interval=2, output_format="floor") == "57"

        # >= 60 岁：2 岁精细区间测试 (60, 61 -> 60; 62, 63 -> 62)
        assert adaptive_age_hierarchy(60, under_60_interval=3, senior_interval=2, output_format="floor") == "60"
        assert adaptive_age_hierarchy(61, under_60_interval=3, senior_interval=2, output_format="floor") == "60"
        assert adaptive_age_hierarchy(62, under_60_interval=3, senior_interval=2, output_format="floor") == "62"
        assert adaptive_age_hierarchy(63, under_60_interval=3, senior_interval=2, output_format="floor") == "62"

        # 范围格式测试 output_format="range"
        assert adaptive_age_hierarchy(31, under_60_interval=3, senior_interval=2, output_format="range") == "[30-32]"
        assert adaptive_age_hierarchy(61, under_60_interval=3, senior_interval=2, output_format="range") == "[60-61]"
