"""隐私原语统一数据适配器与 PyArrow Metadata 导出测试 / Privacy Primitive Unified Adapters
and PyArrow Metadata Export Tests.

中文说明：
验证 privacy 模块下各隐私原语（masking, kano_table, qol）的：
- 统一数据输入输出处理（return_details 模式）。
- PyArrow Table 导出与 Schema Metadata 嵌入。
- 结构化结果包装类（MaskingResult, KAnonymityResult, QoLResult）的正确性。

English Description:
Tests for unified data I/O adapters and PyArrow metadata export across privacy primitives
(masking, kano_table, qol):
- Unified input/output handling (return_details mode).
- PyArrow Table export with embedded Schema Metadata.
- Structured result wrapper classes (MaskingResult, KAnonymityResult, QoLResult) correctness.
"""

import pandas as pd

from engine.privacy.kano_table import (
    KAnonymityResult,
    k_anonymize_table,
)
from engine.privacy.masking import (
    MaskingResult,
    mask_dataframe,
    mask_value,
)
from engine.privacy.qol import QoLResult, obfuscate_query


class TestMaskingUnifiedAdapters:
    """数据脱敏统一适配器测试 / Masking Unified Adapter Tests.

    中文说明：验证 mask_value 和 mask_dataframe 的 return_details 模式
    及 PyArrow Metadata 导出能力。

    English Description: Tests return_details mode and PyArrow metadata export
    for mask_value and mask_dataframe.
    """

    def test_mask_value_return_details_and_to_arrow(self):
        """测试单字段脱敏返回结构化结果并导出 PyArrow Table。"""
        res = mask_value("mobile", "13812345678", return_details=True)
        assert isinstance(res, MaskingResult)
        assert res.value == "138****5678"
        assert res.operation == "mask_value:mobile"

        # 验证 PyArrow Table 导出包含 masking_metadata
        table = res.to_arrow()
        assert b"masking_metadata" in table.schema.metadata
        assert table.column("masked_value")[0].as_py() == "138****5678"

    def test_mask_dataframe_return_details_and_to_arrow(self):
        """测试 DataFrame 脱敏返回结构化结果并导出 PyArrow Table。"""
        df = pd.DataFrame({"user_name": ["张三", "李四"], "mobile": ["13812345678", "13987654321"]})
        res = mask_dataframe(df, return_details=True)
        assert isinstance(res, MaskingResult)
        assert isinstance(res.value, pd.DataFrame)
        assert res.value.iloc[0]["user_name"] == "张*"

        # 验证 PyArrow Table 导出包含 masking_metadata 和原始列名
        table = res.to_arrow()
        assert b"masking_metadata" in table.schema.metadata
        assert "user_name" in table.column_names


class TestKAnonymityUnifiedAdapters:
    """K-匿名统一适配器测试 / K-Anonymity Unified Adapter Tests.

    中文说明：验证 k_anonymize_table 的 return_details 模式
    及 PyArrow Metadata 导出能力。

    English Description: Tests return_details mode and PyArrow metadata export
    for k_anonymize_table.
    """

    def test_k_anonymize_table_return_details_and_to_arrow(self):
        """测试整表 K-匿名返回结构化结果并导出 PyArrow Table。"""
        rows = [
            {"age": "25", "zipcode": "310000", "gender": "M"},
            {"age": "26", "zipcode": "310001", "gender": "M"},
        ]
        res = k_anonymize_table(rows, qi_cols=["age", "zipcode"], k=2, return_details=True)
        assert isinstance(res, KAnonymityResult)
        assert res.k == 2

        # 验证 PyArrow Table 导出包含 k_anonymity_metadata
        table = res.to_arrow()
        assert b"k_anonymity_metadata" in table.schema.metadata
        assert "age" in table.column_names


class TestQoLUnifiedAdapters:
    """查询混淆统一适配器测试 / Query Obfuscation Unified Adapter Tests.

    中文说明：验证 obfuscate_query 的 return_details 模式
    及 PyArrow Metadata 导出能力。

    English Description: Tests return_details mode and PyArrow metadata export
    for obfuscate_query.
    """

    def test_obfuscate_query_return_details_and_to_arrow(self):
        """测试查询混淆返回结构化结果并导出 PyArrow Table。"""
        res = obfuscate_query("高血压饮食", num_dummies=3, domain="medical", return_details=True)
        assert isinstance(res, QoLResult)
        assert len(res.queries) == 4
        assert res.queries[res.real_query_index] == "高血压饮食"

        # 验证 PyArrow Table 导出包含 qol_metadata
        table = res.to_arrow()
        assert b"qol_metadata" in table.schema.metadata
        assert "obfuscated_query" in table.column_names
