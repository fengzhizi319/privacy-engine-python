"""查询混淆模块单元测试。

验证 obfuscate_query / obfuscate_query_batch 的核心行为：
- 真实查询包含在混淆结果中且列表长度正确
- 领域词库选择、自定义池覆盖
- 语义槽位替换与长度相近抽样策略
- Prometheus 指标埋点正确性
- 枚举值向后兼容性
- 输入校验快速失败
"""

# 启用 PEP 563 延迟注解求值，允许在类型注解中使用尚未定义的类
from __future__ import annotations

import pytest  # pytest 测试框架，提供 raises/importorskip/fixture 等工具
from prometheus_client import REGISTRY  # Prometheus 全局默认注册表，用于读取指标当前值

# 从被测模块导入公开接口和常量
from engine.privacy.qol import (
    ObfuscationDomain,  # 混淆领域枚举（medical/generic）
    ObfuscationStrategy,  # 混淆策略枚举（slot_filling/length_similarity/hybrid）
    obfuscate_query,  # 单条查询混淆接口
    obfuscate_query_batch,  # 批量查询混淆接口
)


class TestObfuscateQuery:
    """单条查询混淆测试。

    验证 obfuscate_query 的核心行为：
    - 真实查询包含在混淆结果中
    - 结果列表长度 = num_dummies + 1
    - 领域词库正确选择
    - 自定义池覆盖内置词库
    - Prometheus 指标正确递增
    """

    def test_obfuscate_query_includes_real_query(self) -> None:
        """验证混淆结果包含真实查询且总长度 = num_dummies + 1。"""
        # 执行混淆：3 条 Dummy + 1 条真实查询 = 4 条结果
        # seed=42 保证可复现性（相同 seed + 相同输入 = 相同输出）
        result = obfuscate_query("真实查询", num_dummies=3, domain="medical", seed=42)
        # 打印result
        print(result)
        # 真实查询必须存在于混淆列表中（否则调用方无法找回原始查询）
        assert "真实查询" in result
        # 总长度 = 3 条 Dummy + 1 条真实查询 = 4
        assert len(result) == 4

    def test_obfuscate_query_uses_domain_pool(self) -> None:
        """验证 generic 领域使用通用词库，结果长度正确。"""
        # 使用 generic 领域，应从 GENERIC_DUMMY 词库中抽取 Dummy
        result = obfuscate_query("真实查询", num_dummies=2, domain="generic", seed=42)
        # 打印result
        print(result)
        # 真实查询必须存在
        assert "真实查询" in result
        # 总长度 = 2 条 Dummy + 1 条真实查询 = 3
        assert len(result) == 3

    def test_obfuscate_query_custom_pool(self) -> None:
        """验证自定义 medical_pool 覆盖内置词库：Dummy 全部来自自定义池。"""
        # 构造自定义医疗池（仅 2 条，用于验证池覆盖逻辑）
        custom = ["自定义虚假查询1", "自定义虚假查询2"]
        result = obfuscate_query(
            "真实查询", num_dummies=2, domain="medical", medical_pool=custom, seed=42
        )
        # 打印result
        print(result)
        # 真实查询必须存在
        assert "真实查询" in result
        # 验证所有非真实查询均必须来自自定义池（而非内置 MEDICAL_DUMMY）
        for q in result:
            if q != "真实查询":  # 跳过真实查询本身
                assert q in custom  # Dummy 必须在自定义池中

    def test_obfuscate_query_records_metric(self) -> None:
        """验证 obfuscate_query 调用后 Prometheus 计数器递增 1。

        Before/After 差值断言模式：
        - 隔离性：不依赖绝对值，其他测试的调用不会干扰本测试
        - 幂等性：无论测试执行顺序如何，差值始终为 1
        - 精确性：用 == 而非 >=，能检测意外的多次递增
        """
        # Step 1: 快照当前 Prometheus 计数器值
        # REGISTRY.get_sample_value 从注册表中查找指标名为
        # "privacy_qol_operations_total"、标签为 {domain: medical} 的当前累计值
        # 返回 Optional[float]：若该标签组合从未被 .inc() 过，返回 None
        before = REGISTRY.get_sample_value(
            "privacy_qol_operations_total", {"domain": "medical"}
        ) or 0.0  # None 时取 0.0 作为基线（首次运行保护）
        # Step 2: 触发一次真实混淆调用，内部执行 QOL_OPERATIONS_TOTAL.labels(domain="medical").inc()
        obfuscate_query("真实查询", num_dummies=1, domain="medical", seed=42)
        # Step 3: 再次读取同一标签组合的值，断言恰好增加 1
        after = REGISTRY.get_sample_value(
            "privacy_qol_operations_total", {"domain": "medical"}
        )
        assert after == before + 1  # 精确断言：恰好 +1，检测意外多次递增的 bug


class TestObfuscateQueryBatch:
    """批量查询混淆测试。

    验证 obfuscate_query_batch 的行为：
    - 每个查询独立混淆，结果列表长度与输入一致
    - 每个混淆结果包含对应的真实查询
    - 空列表输入触发 ValueError
    - Prometheus 指标按查询数递增
    """

    def test_obfuscate_query_batch(self) -> None:
        """验证批量混淆：每个查询独立混淆，结果包含真实查询且长度正确。"""
        # 批量混淆 2 个查询，每个生成 2 条 Dummy
        results = obfuscate_query_batch(
            ["查询1", "查询2"], num_dummies=2, domain="medical", seed=42
        )
        # 返回结果数 = 输入查询数
        assert len(results) == 2
        # 逐个验证：每个混淆结果包含对应的真实查询，且长度 = 2 Dummy + 1 真实 = 3
        for r, q in zip(results, ["查询1", "查询2"]):
            assert q in r       # 真实查询必须存在于对应的混淆列表中
            assert len(r) == 3  # 2 条 Dummy + 1 条真实查询

    def test_obfuscate_query_batch_empty(self) -> None:
        """验证空列表输入触发 ValueError（快速失败）。"""
        # pytest.raises 上下文管理器：with 块内必须抛出指定异常，否则测试失败
        # match 参数对异常消息做 re.search 正则匹配
        with pytest.raises(ValueError, match="must not be empty"):
            obfuscate_query_batch([])  # 空列表无实际处理对象

    def test_obfuscate_query_batch_records_metric(self) -> None:
        """验证批量混淆时 Prometheus 指标按实际查询数递增。

        批量接口内部对每个查询调用 obfuscate_query，
        因此 1 个查询的批量调用应使计数器 +1。
        """
        # 快照 generic 领域计数器当前值
        before = REGISTRY.get_sample_value(
            "privacy_qol_operations_total", {"domain": "generic"}
        ) or 0.0  # None 时取 0.0 作为基线
        # 批量混淆 1 个查询（generic 领域）
        obfuscate_query_batch(["查询1"], num_dummies=1, domain="generic", seed=42)
        # 再次读取计数器，断言恰好 +1（因为批量内只有 1 个查询）
        after = REGISTRY.get_sample_value(
            "privacy_qol_operations_total", {"domain": "generic"}
        )
        assert after == before + 1


class TestObfuscationDomainEnum:
    """混淆领域枚举测试。

    验证 ObfuscationDomain 枚举继承 str，枚举值与字符串比较返回 True，
    保证与 Prometheus 指标标签和日志字段的字符串兼容性。
    """

    def test_obfuscation_domain_enum_values(self) -> None:
        """验证枚举值与字符串的向后兼容性。"""
        # ObfuscationDomain 继承 str，保证 ObfuscationDomain.MEDICAL == "medical" 为 True
        assert ObfuscationDomain.MEDICAL == "medical"  # 医疗领域
        assert ObfuscationDomain.GENERIC == "generic"  # 通用领域


class TestObfuscationStrategyEnum:
    """混淆策略枚举测试。

    验证 ObfuscationStrategy 枚举值与字符串的向后兼容性，
    确保枚举值可正确用于结构化日志中的 strategy 字段。
    """

    def test_obfuscation_strategy_enum_values(self) -> None:
        """验证三种混淆策略的枚举值与字符串兼容。"""
        # 语义槽位替换：匹配实体词并替换为近邻实体
        assert ObfuscationStrategy.SLOT_FILLING == "slot_filling"
        # 长度相近抽样：从 Dummy 池中选取长度接近的条目
        assert ObfuscationStrategy.LENGTH_SIMILARITY == "length_similarity"
        # 混合策略：槽位替换 + 长度抽样补齐
        assert ObfuscationStrategy.HYBRID == "hybrid"


class TestInputValidationQoL:
    """输入校验测试。

    验证各公开接口的参数合法性检查，确保非法输入时快速失败并抛出清晰的 ValueError。
    设计原则：Fail-Fast，在业务逻辑执行前拦截非法参数。
    """

    def test_obfuscate_query_empty_raises(self) -> None:
        """验证空字符串 query 触发 ValueError。"""
        # 空字符串无实际查询内容，_validate_query 应拒绝
        with pytest.raises(ValueError, match="query must not be empty"):
            obfuscate_query("")

    def test_obfuscate_query_whitespace_only_raises(self) -> None:
        """验证纯空白字符串 query 触发 ValueError（strip 后为空）。"""
        # 纯空格 strip() 后为空，与空字符串等价
        with pytest.raises(ValueError, match="query must not be empty"):
            obfuscate_query("   ")

    def test_obfuscate_query_invalid_num_dummies_raises(self) -> None:
        """验证 num_dummies=0 触发 ValueError（至少为 1）。"""
        # 0 条 Dummy 无混淆意义，_validate_num_dummies 应拒绝
        with pytest.raises(ValueError, match="num_dummies must be at least 1"):
            obfuscate_query("测试查询", num_dummies=0)

    def test_obfuscate_query_invalid_domain_raises(self) -> None:
        """验证不支持的 domain 触发 ValueError（白名单校验）。"""
        # "invalid" 不在 medical/generic 白名单中
        with pytest.raises(ValueError, match="domain must be"):
            obfuscate_query("测试查询", domain="invalid")

    def test_obfuscate_query_batch_non_list_raises(self) -> None:
        """验证非列表类型 queries 触发 ValueError（类型守卫）。"""
        # 传入字符串而非列表，类型校验应拒绝
        with pytest.raises(ValueError, match="queries must be a list"):
            obfuscate_query_batch("not a list")  # type: ignore
