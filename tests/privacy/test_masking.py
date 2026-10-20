"""数据脱敏模块单元测试 / Data Masking Module Unit Tests.

中文说明：
本模块全面验证 engine.privacy.masking 模块的所有功能：

1. 单字段脱敏 / Single Field Masking:
   - 手机号: 保留前3后4，中间 **** / Mobile: keep first 3 & last 4
   - 身份证: 保留前6后4，中间 ******** / ID card: keep first 6 & last 4
   - 姓名: 保留首尾，中间 ** / Name: keep first & last char
   - 邮箱: 用户名首尾 + *** + @域名 / Email: first & last + *** + @domain
   - 地址: 保留前6字符 + **** / Address: keep first 6 chars
   - 默认: 保留前3后3，中间 * / Default: keep first 3 & last 3

2. 批量与记录级脱敏 / Batch & Record Masking:
   - mask_value_batch: 多字段并行脱敏
   - mask_record: 整条记录自动路由
   - mask_dataframe: DataFrame/PyArrow/Polars/numpy 多格式支持
   - chunked_mask_records: 流式分块脱敏

3. 工具函数 / Utility Functions:
   - hash_value: HMAC-SHA256 确定性哈希
   - truncate: 字符串截断

4. 输入校验与异常 / Input Validation & Errors:
   - 空字段名/非字符串值/空盐值/负截断位数 等异常场景

5. 可观测性 / Observability:
   - Prometheus 计数器递增验证

English Description:
Comprehensive unit tests for the data masking module covering all field types,
batch operations, multi-format DataFrame support, utilities, and validation.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from engine.privacy.masking import (
    FieldType,
    MaskingOperation,
    chunked_mask_records,
    guess_field_type,
    hash_value,
    mask_address,
    mask_dataframe,
    mask_default,
    mask_email,
    mask_id_card,
    mask_mobile,
    mask_name,
    mask_record,
    mask_value,
    mask_value_batch,
    truncate,
)


class TestMaskValue:
    """单字段脱敏测试。

    验证各字段类型（手机号、身份证、姓名、邮箱、地址、默认）的脱敏规则正确性，
    以及 mask_value 根据字段名自动路由到对应脱敏函数的行为。
    """

    def test_mask_mobile(self) -> None:
        """验证手机号脱敏规则：保留前 3 位与后 4 位，中间替换为 ****。"""
        # 标准 11 位手机号：前3 + **** + 后4
        assert mask_mobile("13812345678") == "138****5678"
        # 带 +86 前缀：前缀保留并补空格（与 Go MaskPhone 对齐）
        assert mask_mobile("+8613812345678") == "+86 138****5678"
        # 非标准长度按码点保留前3后4，短值整体置星（与 Go MaskPhone 对齐）
        assert mask_mobile("123") == "***"
        assert mask_mobile("123456789") == "123**6789"

    def test_mask_id_card(self) -> None:
        """验证 18 位身份证号脱敏规则：保留前 6 位与后 4 位，中间替换为 ********。"""
        # 标准 18 位身份证：前6 + ******** + 后4
        assert mask_id_card("110101199001011234") == "110101********1234"

    def test_mask_name(self) -> None:
        """验证中文姓名脱敏规则：自动剥离末尾数字序号与后缀后进行规范脱敏。"""
        # 3 字姓名：保留首尾字，中间替换为 **
        assert mask_name("张三丰") == "张**丰"
        # 带数字序号的 3 字姓名（如 "韩雨泽_3" -> 剥离为 "韩雨泽" -> "韩**泽"）
        assert mask_name("韩雨泽_3") == "韩**泽"
        assert mask_name("韩雨泽3") == "韩**泽"
        # 2 字姓名：保留首字，后接 *
        assert mask_name("李四") == "李*"
        assert mask_name("李四-12") == "李*"
        assert mask_name("王五 (3)") == "王*"

    def test_mask_email(self) -> None:
        """验证邮箱地址脱敏规则：用户名保留首尾字符，中间替换为 ***，域名完整保留。"""
        # 常规用户名（>2 字符）：首字 + *** + 尾字 + @ + 域名
        assert mask_email("zhangsan@example.com") == "z***n@example.com"
        # 短用户名（<=2 字符）：首字 + *** + @ + 域名
        assert mask_email("ab@test.com") == "a***@test.com"
        # 无 @ 符号时回退到 mask_default 策略（保留前后 3 位）
        assert mask_email("noemail") == "noe*ail"

    def test_mask_address(self) -> None:
        """验证地址脱敏规则：保留前 6 个字符（省/市/区），剩余替换为 ****。"""
        # 长度 > 6 的地址：保留前 6 字符 + ****
        assert mask_address("北京市朝阳区某某街道123号") == "北京市朝阳区****"
        # 长度 <= 6 的地址信息量不足，原样返回
        assert mask_address("短地址") == "短地址"

    def test_mask_default(self) -> None:
        """验证默认脱敏策略：保留前 3 位与后 3 位，中间用 * 填充。"""
        # 8 字符：前3 + ** + 后3 = abc**fgh
        assert mask_default("abcdefgh") == "abc**fgh"

    def test_mask_value_routes_by_field_name(self) -> None:
        """验证 mask_value 根据字段名关键字自动路由到对应的脱敏函数。

        guess_field_type 通过子串匹配（大小写不敏感）识别字段类型，
        然后路由到对应的 mask_mobile / mask_id_card / mask_name 等函数。
        """
        # 字段名包含 "mobile" → 路由到 mask_mobile
        assert mask_value("mobile", "13812345678") == "138****5678"
        # 字段名包含 "id_card" → 路由到 mask_id_card
        assert mask_value("id_card", "110101199001011234") == "110101********1234"
        # 字段名包含 "name" → 路由到 mask_name
        assert mask_value("name", "张三丰") == "张**丰"
        # 字段名包含 "email" → 路由到 mask_email
        assert mask_value("email", "test@example.com") == "t***t@example.com"
        # 字段名包含 "address" → 路由到 mask_address
        assert mask_value("address", "北京市朝阳区某某街道") == "北京市朝阳区****"
        # 字段名无匹配关键字 → 路由到 mask_default（保留前后各 3 位）
        assert mask_value("unknown", "abcdefgh") == "abc**fgh"

    def test_mask_value_records_metric(self) -> None:
        """Verify that mask_value increments the Prometheus masking operations counter.
        验证 mask_value 调用后 Prometheus 脱敏操作计数器递增。

        Execution Steps:
        1. Read the current counter value for the 'mask_value' operation label.
        2. Execute one mask_value call to trigger the metrics increment.
        3. Re-read the counter and assert it increased by exactly 1.
        执行步骤：
        1. 读取 'mask_value' 操作标签对应的当前计数器值。
        2. 执行一次 mask_value 调用，触发内部指标递增。
        3. 重新读取计数器并断言恰好增加了 1。

        Prometheus 核心概念：
        - Counter：只增不减的单调递增计数器，适合统计“总操作数”。
        - REGISTRY：prometheus_client 全局默认注册表，所有 Counter/Gauge/Histogram 自动注册于此。
        - labels：指标维度，不同 operation 标签各自独立计数。
        - 指标定义位于 observability/metrics.py：
            MASKING_OPERATIONS_TOTAL = Counter(
                "privacy_masking_operations_total",
                "Total number of masking operations.",
                ["operation"],  # 标签维度
            )
        - 业务代码埋点：MASKING_OPERATIONS_TOTAL.labels(operation="mask_value").inc()
        - /metrics 端点输出示例：
            privacy_masking_operations_total{operation="mask_value"} 128.0
        """

        # Step 1: 快照当前 Prometheus 计数器值
        # REGISTRY.get_sample_value(name, labels_dict) 从注册表中查找指标名为
        # "privacy_masking_operations_total"、标签为 {operation: mask_value} 的当前累计值。
        # 返回 Optional[float]：若该标签组合从未被 .inc() 过，返回 None（子序列尚未创建）。
        counter = REGISTRY.get_sample_value(
            "privacy_masking_operations_total", {"operation": "mask_value"}
        )
        # `counter or 0.0`：Python 短路求值，若 counter 为 None（falsy）则取 0.0 作为基线。
        # 这保证测试在隔离环境首次运行时不会因 None 而报错。
        before = counter or 0.0

        # Step 2: 触发一次真实脱敏调用，内部执行：
        #   1) 校验参数 → 2) guess_field_type("mobile") → 3) mask_mobile("13812345678")
        #   4) MASKING_OPERATIONS_TOTAL.labels(operation="mask_value").inc()  ← 计数器 +1
        #   5) 记录结构化日志
        mask_value("mobile", "13812345678")

        # Step 3: 再次读取同一标签组合的值，断言恰好增加 1。
        # 使用 == 而非 >=，能捕获重复 .inc() 的 bug。
        # Before/After 差值断言模式的优势：
        #   - 隔离性：不依赖绝对值，其他测试的调用不会干扰本测试
        #   - 幂等性：无论测试执行顺序如何，差值始终为 1
        #   - 精确性：用 == 而非 >=，能检测意外的多次递增
        after = REGISTRY.get_sample_value(
            "privacy_masking_operations_total", {"operation": "mask_value"}
        )
        assert after == before + 1


class TestFieldTypeEnum:
    """字段类型枚举测试。

    验证 FieldType 枚举的值正确性，以及 guess_field_type 根据字段名关键字
    正确识别敏感字段类型（支持英文关键字和中文关键字）。
    """

    def test_field_type_enum_values(self) -> None:
        """验证 FieldType 枚举继承 str，枚举值与字符串比较返回 True。"""
        # FieldType 继承 str，保证向后兼容性：FieldType.MOBILE == "mobile" 为 True
        assert FieldType.MOBILE == "mobile"
        assert FieldType.ID_CARD == "id_card"
        assert FieldType.EMAIL == "email"
        assert FieldType.ADDRESS == "address"

    def test_guess_field_type_email(self) -> None:
        """验证包含 email/mail/邮箱 关键字的字段名均被识别为 email 类型。"""
        # 精确匹配 "email"
        assert guess_field_type("email") == FieldType.EMAIL.value
        # 子串匹配 "mail"（如 user_mail）
        assert guess_field_type("user_mail") == FieldType.EMAIL.value
        # 中文关键字 "邮箱"
        assert guess_field_type("邮箱") == FieldType.EMAIL.value

    def test_guess_field_type_address(self) -> None:
        """验证包含 addr/address/地址 关键字的字段名均被识别为 address 类型。"""
        # 精确匹配 "address"
        assert guess_field_type("address") == FieldType.ADDRESS.value
        # 子串匹配 "addr"（如 home_addr）
        assert guess_field_type("home_addr") == FieldType.ADDRESS.value
        # 中文关键字 "地址"
        assert guess_field_type("地址") == FieldType.ADDRESS.value


class TestMaskingOperationEnum:
    """脱敏操作枚举测试。

    验证 MaskingOperation 枚举值与字符串的向后兼容性，
    确保枚举值可正确用于 Prometheus 指标标签和 MaskingResult.operation 字段。
    """

    def test_masking_operation_enum_values(self) -> None:
        """验证 MaskingOperation 枚举继承 str，枚举值与字符串比较返回 True。"""
        # 枚举继承 str，保证与 Prometheus 指标标签和日志字段的字符串兼容性
        assert MaskingOperation.MASK_VALUE == "mask_value"
        assert MaskingOperation.HASH_VALUE == "hash_value"
        assert MaskingOperation.TRUNCATE == "truncate"


class TestInputValidation:
    """输入校验测试。

    验证各公开接口的参数合法性检查，确保非法输入时快速失败并抛出清晰的 ValueError。
    """

    def test_mask_value_empty_field_name_raises(self) -> None:
        """验证空字符串 field_name 触发 ValueError。"""
        # pytest.raises(ExpectedException, match="regex") 是 pytest 的异常断言上下文管理器：
        #   - 用法：with 块内的代码必须抛出指定异常，否则测试失败
        #   - match 参数：对异常的 str(exc) 做 re.search 正则匹配，确保错误信息符合预期
        #   - 若抛出了其他类型的异常（如 TypeError），测试也会失败
        # 此处验证：传入空字符串 field_name 时，_validate_field_name 抛出 ValueError，且消息匹配
        with pytest.raises(ValueError, match="field_name must not be empty"):
            mask_value("", "test")

    def test_mask_value_non_string_value_raises(self) -> None:
        """验证非字符串类型 value 触发 ValueError。"""
        # 传入整数 12345 而非字符串，_validate_value 检测到类型不匹配并抛出 ValueError
        with pytest.raises(ValueError, match="value must be a string"):
            mask_value("mobile", 12345)  # type: ignore

    def test_hash_value_empty_salt_raises(self) -> None:
        """验证空字符串 salt 触发 ValueError。

        HMAC 哈希必须提供非空盐值，否则无法保证哈希的唯一性和安全性。
        """
        with pytest.raises(ValueError, match="salt must not be empty"):
            hash_value("test", "")

    def test_truncate_negative_prefix_raises(self) -> None:
        """验证负数 keep_prefix 触发 ValueError。

        截断位数必须为非负整数，负数无意义且可能导致切片异常。
        """
        with pytest.raises(ValueError, match="keep_prefix must be non-negative"):
            truncate("test", -1)

    def test_mask_record_non_dict_raises(self) -> None:
        """验证非字典类型 record 触发 ValueError。

        mask_record 要求输入为字典，键为字段名，值为待脱敏值。
        """
        # 传入列表而非字典，触发类型校验失败
        with pytest.raises(ValueError, match="record must be a dict"):
            mask_record(["not", "a", "dict"])  # type: ignore

    def test_mask_record_empty_raises(self) -> None:
        """验证空字典 record 触发 ValueError。

        空字典无字段可脱敏，属于无意义调用，快速失败。
        """
        with pytest.raises(ValueError, match="record must not be empty"):
            mask_record({})

    def test_mask_value_batch_empty_raises(self) -> None:
        """验证空列表触发 ValueError。

        批量脱敏要求 field_names 和 values 非空，否则无实际处理对象。
        """
        with pytest.raises(ValueError, match="must not be empty"):
            mask_value_batch([], [])


class TestMaskRecord:
    """整记录脱敏测试。

    验证 mask_record 对字典类型记录的脱敏行为：
    - 根据字段名自动路由到对应脱敏函数
    - 非字符串类型字段（如 age）保持不变
    - 不修改原始输入字典（纯函数特性）
    """

    def test_mask_record(self) -> None:
        """验证整记录脱敏：多字段同时脱敏，非字符串字段保持不变，原记录不被修改。"""
        # 构造包含手机号、姓名、年龄的测试记录
        record = {"mobile": "13812345678", "name": "张三丰", "age": 30}
        # 执行整记录脱敏，内部对每个字段调用 mask_value
        result = mask_record(record)
        # 验证手机号脱敏：前3 + **** + 后4
        assert result["mobile"] == "138****5678"
        # 验证姓名脱敏：保留首尾，中间 **
        assert result["name"] == "张**丰"
        # 验证非字符串字段（age）保持不变
        assert result["age"] == 30
        # 验证原记录未被修改（纯函数特性）
        assert record["mobile"] == "13812345678"


class TestMaskBatch:
    """批量字段脱敏测试。

    验证 mask_value_batch 对多个字段同时脱敏的行为：
    - 字段名列表与值列表长度必须一致
    - 返回脱敏后的值列表，顺序与输入一致
    """

    def test_mask_value_batch(self) -> None:
        """验证批量脱敏：多个字段同时处理，返回结果列表。"""
        # 批量脱敏：手机号、姓名、身份证号
        results = mask_value_batch(
            ["mobile", "name", "id_card"],
            ["13812345678", "张三丰", "110101199001011234"],
        )
        # 验证返回结果与预期脱敏值一致
        assert results == ["138****5678", "张**丰", "110101********1234"]

    def test_mask_value_batch_length_mismatch(self) -> None:
        """验证字段名列表与值列表长度不一致时触发 ValueError。"""
        # 字段名 1 个，值 2 个，长度不匹配
        with pytest.raises(ValueError, match="same length"):
            mask_value_batch(["mobile"], ["13812345678", "13812345679"])


class TestMaskDataFrame:
    """DataFrame 脱敏测试。

    验证 mask_dataframe 对 pandas DataFrame 的脱敏行为：
    - 自动检测字符串类型列并脱敏
    - 数值类型列保持不变
    - 支持 columns 参数指定脱敏列
    - 空 DataFrame 正确处理
    """

    def test_mask_dataframe(self) -> None:
        """验证 DataFrame 脱敏：自动检测字符串列，数值列保持不变。"""
        # 跳过测试如果 pandas 未安装（pytest.importorskip 自动处理）
        pd = pytest.importorskip("pandas")
        # 构造包含手机号、姓名、年龄的测试 DataFrame
        df = pd.DataFrame(
            {
                "mobile": ["13812345678", "13912345678"],
                "name": ["张三", "李四"],
                "age": [25, 34],
            }
        )
        # 执行 DataFrame 脱敏，内部使用 pandas 向量化操作
        result = mask_dataframe(df)
        # 验证手机号列脱敏结果
        assert result["mobile"].tolist() == ["138****5678", "139****5678"]
        # 验证姓名列脱敏结果（2 字姓名：首字 + *）
        assert result["name"].tolist() == ["张*", "李*"]
        # 验证年龄列（数值类型）保持不变
        assert result["age"].tolist() == [25, 34]

    def test_mask_dataframe_with_columns(self) -> None:
        """验证 columns 参数：仅对指定列脱敏，其他列保持不变。"""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "mobile": ["13812345678"],
                "name": ["张三"],
            }
        )
        # 仅对 mobile 列脱敏，name 列保持不变
        result = mask_dataframe(df, columns=["mobile"])
        assert result["mobile"].tolist() == ["138****5678"]
        # name 列未被指定，保持原值
        assert result["name"].tolist() == ["张三"]

    def test_mask_dataframe_empty(self) -> None:
        """验证空 DataFrame 正确处理，返回空 DataFrame。"""
        pd = pytest.importorskip("pandas")
        # 构造空 DataFrame（无数据行）
        df = pd.DataFrame({"mobile": []})
        result = mask_dataframe(df)
        # 验证返回结果仍为空 DataFrame
        assert result.empty

    def test_mask_dataframe_list_of_dict(self) -> None:
        """验证 list of dict 记录列表输入：走通用 _convert_to_records 路径。"""
        # 构造记录列表（最常见的通用格式）
        records = [
            {"mobile": "13812345678", "name": "张三丰", "age": 30},
            {"mobile": "13912345678", "name": "李四", "age": 25},
        ]
        result = mask_dataframe(records)
        # 返回类型为 list of dict
        assert isinstance(result, list)
        assert len(result) == 2
        # 验证手机号脱敏
        assert result[0]["mobile"] == "138****5678"
        assert result[1]["mobile"] == "139****5678"
        # 验证姓名脱敏
        assert result[0]["name"] == "张**丰"
        assert result[1]["name"] == "李*"
        # 验证非字符串字段保持不变
        assert result[0]["age"] == 30
        assert result[1]["age"] == 25

    def test_mask_dataframe_numpy_1d(self) -> None:
        """验证 numpy 1-D ndarray 输入：每个元素生成 {"value": str(v)} 记录。"""
        np = pytest.importorskip("numpy")
        # 1-D 数组：每个元素作为一条单字段记录
        arr = np.array(["13812345678", "13912345678"])
        result = mask_dataframe(arr)
        # 返回记录列表，字段名为 "value"（无敏感关键字，走 mask_default）
        assert isinstance(result, list)
        assert len(result) == 2
        # "value" 不含 mobile/name 等关键字 → mask_default（保留前3后3，中间 * 填充）
        # "13812345678" 长度 11 > 6 → 前3 "138" + 5个* + 后3 "678"
        assert result[0]["value"] == "138*****678"
        assert result[1]["value"] == "139*****678"

    def test_mask_dataframe_numpy_2d(self) -> None:
        """验证 numpy 2-D ndarray 输入：每行为一条记录，列名自动生成 col_0, col_1, ...。"""
        np = pytest.importorskip("numpy")
        # 2-D 数组：2行2列
        arr = np.array([["13812345678", "张三丰"], ["13912345678", "李四"]])
        result = mask_dataframe(arr)
        assert isinstance(result, list)
        assert len(result) == 2
        # col_0 不含敏感关键字 → mask_default（保留前3后3，中间 * 填充）
        # "13812345678" 长度 11 > 6 → "138" + "*****" + "678"
        assert result[0]["col_0"] == "138*****678"
        # col_1 不含敏感关键字 → mask_default
        # 短值保护（与 Go MaskDefault 对齐）：保留首字符，其余星化
        assert result[0]["col_1"] == "张**"
        assert result[1]["col_1"] == "李*"

    def test_mask_dataframe_pyarrow_table(self) -> None:
        """验证 PyArrow Table 输入：走列级向量化 pc.utf8_* 快速路径。"""
        pa = pytest.importorskip("pyarrow")
        # 构造 PyArrow Table（包含手机号和姓名列）
        table = pa.table({
            "mobile": ["13812345678", "13912345678"],
            "name": ["张三", "李四"],
        })
        result = mask_dataframe(table)
        # 返回类型仍为 PyArrow Table（PyArrow fast path 保持输入输出类型一致）
        assert isinstance(result, pa.Table)
        # 验证手机号列脱敏（向量化：前3 + **** + 后4）
        assert result.column("mobile").to_pylist() == ["138****5678", "139****5678"]
        # 验证姓名列脱敏（2字：首字 + *）
        assert result.column("name").to_pylist() == ["张*", "李*"]

    def test_mask_dataframe_pyarrow_record_batch(self) -> None:
        """验证 PyArrow RecordBatch 输入：自动转为 Table 后走列级向量化路径。"""
        pa = pytest.importorskip("pyarrow")
        # 构造 RecordBatch（单批次数据）
        batch = pa.record_batch({
            "mobile": ["13812345678"],
            "email": ["zhangsan@example.com"],
        })
        result = mask_dataframe(batch)
        # RecordBatch 被转为 Table 处理，返回 PyArrow Table
        assert isinstance(result, pa.Table)
        # 验证手机号脱敏
        assert result.column("mobile").to_pylist() == ["138****5678"]
        # 验证邮箱脱敏（长用户名：首字 + *** + 尾字 + @域名）
        assert result.column("email").to_pylist() == ["z***n@example.com"]

    def test_mask_dataframe_arrow_ipc_bytes(self) -> None:
        """验证 Arrow IPC Stream 二进制字节流输入：解析为 Table 后转记录列表脱敏。"""
        pa = pytest.importorskip("pyarrow")
        import pyarrow.ipc as ipc
        # 构造原始 Table 并序列化为 IPC Stream 字节流
        table = pa.table({"mobile": ["13812345678", "13912345678"]})
        sink = pa.BufferOutputStream()
        writer = ipc.RecordBatchStreamWriter(sink, table.schema)
        writer.write_table(table)
        writer.close()
        ipc_bytes = sink.getvalue().to_pybytes()  # 获取 bytes 对象
        # 传入 bytes 类型数据
        result = mask_dataframe(ipc_bytes)
        # 走 _convert_to_records 路径，返回记录列表
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["mobile"] == "138****5678"
        assert result[1]["mobile"] == "139****5678"

    def test_mask_dataframe_arrow_ipc_bytearray(self) -> None:
        """验证 Arrow IPC Stream bytearray 输入：与 bytes 行为一致。"""
        pa = pytest.importorskip("pyarrow")
        import pyarrow.ipc as ipc
        # 构造 IPC 字节流并转为 bytearray
        table = pa.table({"name": ["张三丰"]})
        sink = pa.BufferOutputStream()
        writer = ipc.RecordBatchStreamWriter(sink, table.schema)
        writer.write_table(table)
        writer.close()
        ipc_bytearray = bytearray(sink.getvalue().to_pybytes())
        # 传入 bytearray 类型数据
        result = mask_dataframe(ipc_bytearray)
        assert isinstance(result, list)
        assert result[0]["name"] == "张**丰"

    def test_mask_dataframe_polars(self) -> None:
        """验证 Polars DataFrame 输入：通过 to_dicts() 转换后走通用路径。"""
        pl = pytest.importorskip("polars")
        # 构造 Polars DataFrame
        df = pl.DataFrame({
            "mobile": ["13812345678", "13912345678"],
            "name": ["张三", "李四"],
        })
        result = mask_dataframe(df)
        # 走 _convert_to_records → to_dicts() 路径，返回记录列表
        assert isinstance(result, list)
        assert len(result) == 2
        # 验证手机号脱敏
        assert result[0]["mobile"] == "138****5678"
        assert result[1]["mobile"] == "139****5678"
        # 验证姓名脱敏（2字：首字 + *）
        assert result[0]["name"] == "张*"
        assert result[1]["name"] == "李*"

    def test_mask_dataframe_secretflow_mock(self, monkeypatch) -> None:
        """验证 SecretFlow DataFrame 输入（Mock）：通过 data_adapters.to_records 适配。

        SecretFlow 为可选重量级依赖，测试中使用 Mock 对象模拟其行为：
        1. 构造 FakeSecretFlowDataFrame（内含 pandas DataFrame + partitions 属性）
        2. monkeypatch _is_secretflow_available 返回 True
        3. monkeypatch _extract_dataframe_partition 返回内部 pandas DataFrame
        """
        pd = pytest.importorskip("pandas")
        import engine.privacy.data_adapters as da

        # 构造模拟的 SecretFlow DataFrame（鸭子类型：具有 partitions 属性）
        class FakeSecretFlowDataFrame:
            def __init__(self, pdf):
                self._pdf = pdf
                self.partitions = {"alice": pdf}  # 模拟单参与方分区

        pdf = pd.DataFrame({"mobile": ["13812345678"], "name": ["张三丰"]})
        sf_data = FakeSecretFlowDataFrame(pdf)

        # monkeypatch：让 _is_secretflow_available 返回 True
        monkeypatch.setattr(da, "_is_secretflow_available", lambda: True)
        # monkeypatch：让 _extract_dataframe_partition 返回内部 pandas DataFrame
        monkeypatch.setattr(da, "_extract_dataframe_partition", lambda data: data._pdf)

        result = mask_dataframe(sf_data)
        # 走 data_adapters.to_records → pandas to_dict 路径，返回记录列表
        assert isinstance(result, list)
        assert len(result) == 1
        # 验证手机号脱敏
        assert result[0]["mobile"] == "138****5678"
        # 验证姓名脱敏
        assert result[0]["name"] == "张**丰"

    def test_mask_dataframe_return_details_pyarrow(self) -> None:
        """验证 PyArrow Table + return_details=True 返回 MaskingResult 元数据。"""
        from engine.privacy.masking import MaskingResult

        pa = pytest.importorskip("pyarrow")
        table = pa.table({"mobile": ["13812345678", "13912345678"]})
        result = mask_dataframe(table, return_details=True)
        # 返回 MaskingResult 包装对象
        assert isinstance(result, MaskingResult)
        assert result.operation == "mask_dataframe"
        assert "mobile" in result.masked_fields
        assert result.total_masked == 2
        # value 为脱敏后的 PyArrow Table
        assert isinstance(result.value, pa.Table)
        assert result.value.column("mobile").to_pylist() == ["138****5678", "139****5678"]

    def test_mask_dataframe_columns_filter_pyarrow(self) -> None:
        """验证 PyArrow Table + columns 参数：仅脱敏指定列，其他列保持不变。"""
        pa = pytest.importorskip("pyarrow")
        table = pa.table({
            "mobile": ["13812345678"],
            "name": ["张三"],
        })
        # 仅对 mobile 列脱敏
        result = mask_dataframe(table, columns=["mobile"])
        assert result.column("mobile").to_pylist() == ["138****5678"]
        # name 列未被指定，保持原值
        assert result.column("name").to_pylist() == ["张三"]


class TestHashAndTruncate:
    """哈希与截断测试。

    验证 hash_value 和 truncate 两个工具函数的行为：
    - hash_value：HMAC-SHA256 哈希，固定长度输出，确定性
    - truncate：截断字符串，不足长度原样返回
    """

    def test_hash_value(self) -> None:
        """验证 HMAC-SHA256 哈希：固定 16 字符输出，确定性，盐值敏感性。"""
        # 执行哈希计算，默认返回 16 字符的十六进制字符串
        result = hash_value("hello", "salt")
        assert len(result) == 16
        # 验证确定性：相同输入产生相同输出
        assert hash_value("hello", "salt") == result
        # 验证盐值敏感性：不同盐值产生不同输出
        assert hash_value("hello", "other") != result

    def test_truncate(self) -> None:
        """验证字符串截断：保留前 N 个字符，剩余用 * 填充；不足 N 字符原样返回。"""
        # 长度 6 截断为 3：保留前 3 字符 + *** 填充
        assert truncate("abcdef", 3) == "abc***"
        # 长度 2 不足 3：原样返回（不填充）
        assert truncate("ab", 3) == "ab"


class TestChunkedMaskRecords:
    """流式分块脱敏测试。

    验证 chunked_mask_records 对大数据集的流式处理能力：
    - 支持迭代器/生成器输入，逐块处理避免内存溢出
    - 支持 columns 参数过滤脱敏列
    - 支持 return_details 返回 MaskingResult 元数据
    - 空输入正确处理
    """

    def test_basic_chunked_masking(self) -> None:
        """验证基本流式脱敏：多个分块逐块处理，返回结果列表。"""
        # 构造两个分块，每个分块包含一条记录
        chunks = [
            [{"mobile": "13812345678", "name": "张三丰", "age": 30}],
            [{"mobile": "13912345678", "name": "李四", "age": 25}],
        ]
        # 执行流式脱敏，list() 将生成器转换为列表
        results = list(chunked_mask_records(chunks))
        # 验证返回 2 个分块的结果
        assert len(results) == 2
        # 验证第一个分块的脱敏结果
        assert results[0][0]["mobile"] == "138****5678"
        assert results[0][0]["name"] == "张**丰"
        assert results[0][0]["age"] == 30
        # 验证第二个分块的脱敏结果
        assert results[1][0]["mobile"] == "139****5678"
        assert results[1][0]["name"] == "李*"

    def test_chunked_with_columns_filter(self) -> None:
        """验证 columns 参数：仅对指定列脱敏，其他列保持不变。"""
        chunks = [
            [{"mobile": "13812345678", "name": "张三丰"}],
        ]
        # 仅对 mobile 列脱敏
        results = list(chunked_mask_records(chunks, columns=["mobile"]))
        assert results[0][0]["mobile"] == "138****5678"
        # name 列未被指定，保持原值
        assert results[0][0]["name"] == "张三丰"

    def test_chunked_return_details(self) -> None:
        """验证 return_details=True 时返回 MaskingResult 元数据对象。"""
        from engine.privacy.masking import MaskingResult

        chunks = [
            [{"mobile": "13812345678", "name": "张三丰"}],
            [{"mobile": "13912345678"}],
        ]
        # 启用 return_details，返回 MaskingResult 而非原始字典
        results = list(chunked_mask_records(chunks, return_details=True))
        assert len(results) == 2
        # 验证返回类型为 MaskingResult
        assert isinstance(results[0], MaskingResult)
        # 验证操作名称
        assert results[0].operation == "chunked_mask_records"
        # 验证脱敏字段列表包含 mobile
        assert "mobile" in results[0].masked_fields
        # 验证脱敏总数 >= 1
        assert results[0].total_masked >= 1

    def test_chunked_empty_chunks(self) -> None:
        """验证空输入（空列表）正确处理，返回空列表。"""
        results = list(chunked_mask_records([]))
        assert results == []

    def test_chunked_generator_input(self) -> None:
        """验证生成器输入：支持惰性迭代，逐块处理。"""
        # 定义生成器函数，逐块 yield 记录
        def gen():
            yield [{"mobile": "13812345678"}]
            yield [{"mobile": "13912345678"}]

        # 传入生成器对象，验证流式处理能力
        results = list(chunked_mask_records(gen()))
        assert len(results) == 2
        assert results[0][0]["mobile"] == "138****5678"


class TestAdvancedOperators:
    """高级脱敏算子测试 (FPE, Date Offset, Column Shuffle)。"""

    def test_fpe_encrypt_numeric(self) -> None:
        """验证保留格式加密 (FPE)：保持长度、格式形态一致且可重复相等匹配。"""
        from engine.privacy.masking import fpe_encrypt_numeric

        id_card = "110101199001011234"
        encrypted = fpe_encrypt_numeric(id_card, "conformance-secret")
        # 1. 验证长度一致
        assert len(encrypted) == len(id_card)
        # 2. 验证加密后依然全为数字形态
        assert encrypted.isdigit()
        # 3. 验证值与原文不同（被置换加密）
        assert encrypted != id_card
        # 4. 验证确定性与相等比较（同 Key 下两次加密结果一致）
        assert fpe_encrypt_numeric(id_card, "conformance-secret") == encrypted
        # 5. 空密钥不回退到进程随机密钥，直接返回空串（与 Go 对齐）
        assert fpe_encrypt_numeric(id_card, "") == ""
        assert fpe_encrypt_numeric(id_card) == ""

    def test_random_date_offset(self) -> None:
        """验证日期统一随机偏移 (Random Date Offset)：格式保持且时间序列差不变。"""
        from engine.privacy.masking import random_date_offset

        date_1 = "2025-03-15"
        date_2 = "2025-03-20"
        
        # 统一偏移 10 天
        offset_1 = random_date_offset(date_1, offset_days=10)
        offset_2 = random_date_offset(date_2, offset_days=10)
        
        assert offset_1 == "2025-03-25"
        assert offset_2 == "2025-03-30"

    def test_shuffle_column(self) -> None:
        """验证列洗牌 (Column Shuffle)：打乱映射关系但保持集合全集一致。"""
        from engine.privacy.masking import shuffle_column

        dept_list = ["肿瘤科", "心内科", "精神科", "皮肤科", "消化科"]
        shuffled = shuffle_column(dept_list, seed=123)
        
        # 1. 验证元素全集一致
        assert set(shuffled) == set(dept_list)
        # 2. 验证排列顺序被打乱
        assert shuffled != dept_list

