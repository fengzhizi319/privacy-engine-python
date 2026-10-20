"""数据集级 K-匿名内部辅助函数与回退路径测试 / kano_table Internal Unit Tests.

中文说明：
补充 test_kano_table.py 未覆盖的内部纯函数与 pandas 缺失时的纯 Python 回退路径：
- _is_numeric / _span / _choose_dimension / _median_split / _generalize
- KAnonymityResult.to_arrow 的 DataFrame / 标量分支
- k_anonymize_table / k_anonymize_dataframe 在 pandas 不可用时的回退实现

English Description:
Covers kano_table internal helpers and the pure-Python fallback used when pandas
is unavailable, plus extra to_arrow branches.
"""

from __future__ import annotations

import sys

import pytest

from engine.privacy import kano_table as kt


class TestIsNumeric:
    """测试 _is_numeric 内部函数：判断值是否为数值类型。

    注意：bool 虽然继承自 int，但在 K-匿名中不应视为数值型
    （因为 True/False 本质上是分类值）。
    """

    def test_int_float(self):
        """整数和浮点数应被识别为数值型。"""
        assert kt._is_numeric(1)
        assert kt._is_numeric(1.5)

    def test_bool_excluded(self):
        """布尔值不应被视为数值型（分类值）。"""
        assert not kt._is_numeric(True)

    def test_non_numeric(self):
        """字符串和 None 不应被识别为数值型。"""
        assert not kt._is_numeric("x")
        assert not kt._is_numeric(None)


class TestSpan:
    """测试 _span 内部函数：计算指定维度的跨度。

    跨度是 Mondrian 选择划分维度的依据：
    - 数值型: span = max - min
    - 分类型: span = unique_count - 1
    跨度越大表示该维度信息量越大，优先划分。
    """

    def test_empty(self):
        """空数据或全 None 时跨度为 0。"""
        assert kt._span([], "age") == 0.0
        assert kt._span([{"age": None}], "age") == 0.0

    def test_numeric(self):
        """数值型跨度 = max - min = 30 - 10 = 20。"""
        records = [{"age": 10}, {"age": 30}, {"age": 20}]
        assert kt._span(records, "age") == 20.0

    def test_categorical(self):
        """分类型跨度 = unique_count - 1 = 2 - 1 = 1。"""
        records = [{"c": "a"}, {"c": "b"}, {"c": "a"}]
        assert kt._span(records, "c") == 1.0  # 2 unique - 1


class TestChooseDimension:
    """测试 _choose_dimension：选择跨度最大的 QI 维度进行划分。

    这是 Mondrian 算法的核心步骤之一：
    在所有 QI 维度中选择跨度最大的，然后沿该维度的中位数划分。
    """

    def test_picks_max_span(self):
        """应选择跨度最大的维度 (age span=40 > zip span=1)。"""
        records = [
            {"age": 10, "zip": "a"},
            {"age": 50, "zip": "b"},
        ]
        # age span=40, zip span=1 → age
        assert kt._choose_dimension(records, ["age", "zip"]) == "age"


class TestMedianSplit:
    """测试 _median_split：沿指定维度的中位数划分数据。

    返回划分索引 idx，使得 records[:idx] 和 records[idx:] 各至少 k 条。
    如果数据量不足以划分（划分后某半 < k），返回 None。
    """

    def test_too_small_returns_none(self):
        """数据量不足以划分时返回 None。"""
        records = [{"age": i} for i in range(3)]
        assert kt._median_split(records, "age", k=2) is None

    def test_valid_split(self):
        """10 条记录 k=2：划分索引应在 [2,8] 范围内。"""
        records = [{"age": i} for i in range(10)]
        idx = kt._median_split(records, "age", k=2)
        assert idx is not None
        assert 2 <= idx <= 8

    def test_categorical_split(self):
        """分类型维度也可划分（按字典序排序后取中位）。"""
        records = [{"c": f"v{i}"} for i in range(8)]
        idx = kt._median_split(records, "c", k=2)
        assert idx is not None


class TestGeneralize:
    """测试 _generalize：对等价组内的 QI 值执行泛化。

    泛化规则：
    - 数值型: 不同值 → "[min-max]"，相同值 → 保持原值
    - 分类型: 不同值 → "{v1,v2,...}"，相同值 → 保持原值
    """

    def test_empty(self):
        """空输入返回空列表。"""
        assert kt._generalize([], ["age"]) == []

    def test_numeric_interval(self):
        """未注册层次函数的数值列 → 泛化为区间 [10-20]。"""
        records = [{"score": 10}, {"score": 20}]
        out = kt._generalize(records, ["score"])
        assert all(r["score"] == "[10-20]" for r in out)

    def test_numeric_equal(self):
        """数值相同 → 保持原值（与 Go formatNum 一致，输出字符串）。"""
        records = [{"age": 5}, {"age": 5}]
        out = kt._generalize(records, ["age"])
        assert all(r["age"] == "5" for r in out)

    def test_registered_numeric_hierarchy(self):
        """注册了层次函数的列优先使用固定步长泛化（与 Go hierarchyGeneralize 对齐）。"""
        records = [{"age": 25}, {"age": 26}]
        out = kt._generalize(records, ["age"])
        assert all(r["age"] == "[25-30]" for r in out)

    def test_categorical_set(self):
        """分类不同 → 泛化为集合 {a,b}。"""
        records = [{"c": "a"}, {"c": "b"}]
        out = kt._generalize(records, ["c"])
        assert all(r["c"] == "{a,b}" for r in out)

    def test_categorical_single(self):
        """分类相同 → 保持原值。"""
        records = [{"c": "a"}, {"c": "a"}]
        out = kt._generalize(records, ["c"])
        assert all(r["c"] == "a" for r in out)


class TestToArrowBranches:
    """测试 KAnonymityResult.to_arrow 的不同输入分支。

    to_arrow 支持多种 value 类型：
    - list of dict → 转为 PyArrow Table
    - pandas DataFrame → 直接转换
    - 标量 → 包装为单列 Table
    """

    def test_dataframe_value(self):
        """分支：value 为 DataFrame 时直接转 PyArrow Table。"""
        pa = pytest.importorskip("pyarrow")
        pd = pytest.importorskip("pandas")
        result = kt.KAnonymityResult(
            value=pd.DataFrame({"age": ["[25-30]"]}),
            k=3,
            qi_cols=["age"],
            equivalence_classes_count=1,
        )
        table = result.to_arrow()
        assert isinstance(table, pa.Table)
        assert b"k_anonymity_metadata" in table.schema.metadata

    def test_scalar_value(self):
        """分支：value 为标量时包装为单列 Table。"""
        pa = pytest.importorskip("pyarrow")
        result = kt.KAnonymityResult(
            value="single", k=2, qi_cols=["age"], equivalence_classes_count=1
        )
        table = result.to_arrow()
        assert isinstance(table, pa.Table)
        assert table.column_names == ["kanonymity_value"]


class TestPurePythonFallback:
    """通过屏蔽 pandas 触发纯 Python 回退实现 / Pure-Python Fallback When Pandas Unavailable.

    当 pandas 未安装时，k_anonymize_table 和 k_anonymize_dataframe
    应回退到纯 Python 实现（使用 dict 列表而非 DataFrame）。
    通过 monkeypatch 将 sys.modules["pandas"] 设为 None 模拟缺失。
    """

    @pytest.fixture
    def no_pandas(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pandas", None)
        yield

    def test_table_fallback(self, no_pandas):
        rows = [
            {"age": 25, "zip": "100001", "disease": "A"},
            {"age": 26, "zip": "100002", "disease": "B"},
            {"age": 55, "zip": "200001", "disease": "C"},
            {"age": 56, "zip": "200002", "disease": "D"},
        ]
        result = kt.k_anonymize_table(rows, ["age", "zip"], k=2)
        assert len(result) == 4
        # age 被泛化为区间
        assert all("[" in str(r["age"]) for r in result)

    def test_table_fallback_return_details(self, no_pandas):
        rows = [{"age": i} for i in range(6)]
        result = kt.k_anonymize_table(rows, ["age"], k=3, return_details=True)
        assert isinstance(result, kt.KAnonymityResult)
        assert len(result.value) == 6

    def test_dataframe_fallback_with_records(self, no_pandas):
        records = [
            {"age": 25, "zip": "100001"},
            {"age": 26, "zip": "100002"},
            {"age": 55, "zip": "200001"},
            {"age": 56, "zip": "200002"},
        ]
        result = kt.k_anonymize_dataframe(records, ["age", "zip"], k=2)
        assert isinstance(result, list)
        assert len(result) == 4

    def test_dataframe_fallback_return_details(self, no_pandas):
        records = [{"age": i} for i in range(6)]
        result = kt.k_anonymize_dataframe(
            records, ["age"], k=3, return_details=True
        )
        assert isinstance(result, kt.KAnonymityResult)


class TestDataFrameValidation:
    """测试 k_anonymize_dataframe 的输入校验分支。"""

    def test_len_less_than_k_returns_single_group(self):
        """与 Go 对齐：数据行数 < k 不报错，整表作为单一等价组。"""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"age": [1]})
        result = kt.k_anonymize_dataframe(df, ["age"], k=2)
        assert len(result) == 1

    def test_empty_qi_cols_raises(self):
        """空 QI 列抛出 ValueError。"""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"age": [1, 2, 3]})
        with pytest.raises(ValueError, match="qi_cols must not be empty"):
            kt.k_anonymize_dataframe(df, [], k=2)

    def test_missing_cols_treated_as_empty(self):
        """与 Go map 取值一致：缺失的 QI 列按空串参与泛化，不报错。"""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"age": [1, 2, 3]})
        result = kt.k_anonymize_dataframe(df, ["gender"], k=2)
        assert len(result) == 3
