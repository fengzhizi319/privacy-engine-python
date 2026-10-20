"""动态分类分级 - 高级边缘场景、并发与覆盖测试 / Edge Cases & Concurrency Tests."""

import concurrent.futures
import threading
from pathlib import Path
import pytest

from engine.dynclassification import (
    DynClassificationService,
    OperatorRegistry,
    ProfileLoader,
    DomainTaxonomy,
    SensitivityLevelDef,
    CategoryDef,
    RuleProfile,
    RuleDef,
    MatcherDef,
    DowngradeRuleDef,
    StandardDef,
)


class TestConcurrencySafety:
    """多线程高并发下的争抢防护测试"""

    def test_concurrent_operator_registration_and_lookup(self):
        """测试多线程并发注册与获取算子不冲突"""
        def worker(idx: int):
            op_name = f"concurrent_op_{idx}"
            OperatorRegistry.register_func(op_name, lambda val, params: True)
            assert OperatorRegistry.has(op_name) is True
            op = OperatorRegistry.get(op_name)
            assert op("test", {}) is True

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(50)]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 确保无抛出 Exception

    def test_concurrent_service_classification_and_reload(self, tmp_path):
        """测试多线程并发执行分类评估与重载刷新不冲突"""
        # 建立规则文件
        tax_dir = tmp_path / "taxonomies"
        dom_dir = tmp_path / "domains"
        std_dir = tmp_path / "standards"
        tax_dir.mkdir()
        dom_dir.mkdir()
        std_dir.mkdir()

        (tax_dir / "default.yaml").write_text(
            """
domain: default
standard_id: INTERNAL
levels:
  L1: {id: L1, name: 公开, rank: 1}
  L3: {id: L3, name: 敏感, rank: 3}
categories:
  PII: {id: PII, name: 个人基本信息}
default_level: L1
""",
            encoding="utf-8",
        )

        (dom_dir / "general-pii.yaml").write_text(
            """
domain: general-pii
rules:
  - id: RULE_PHONE
    category: PII
    level: L3
    matchers:
      - target: field_value
        operator: regex
        params: {pattern: "^1[3-9]\\\\d{9}$"}
""",
            encoding="utf-8",
        )

        service = DynClassificationService(rules_dir=tmp_path)

        def reader():
            for _ in range(20):
                resp = service.classify_field("phone", "13800138000", domain="general-pii")
                assert resp.field_result is not None

        def reloader():
            for _ in range(5):
                service.reload()

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            f1 = executor.submit(reader)
            f2 = executor.submit(reader)
            f3 = executor.submit(reloader)
            f1.result()
            f2.result()
            f3.result()


class TestRuleOverridesAndExtraRules:
    """标准规则级覆盖与追加规则测试"""

    def test_standard_rule_overrides(self, tmp_path):
        """测试 StandardDef 中的 rule_overrides 覆盖底层 Domain 规则敏感等级"""
        loader = ProfileLoader(rules_dir=tmp_path)

        # 构造规则: 默认领域将 BANK_CARD 定为 L4
        profile = RuleProfile(
            domain="finance",
            rules=[
                RuleDef(
                    id="RULE_BANK_CARD",
                    name="银行卡检测",
                    category="FINANCIAL",
                    level="L4",
                    matchers=[
                        MatcherDef(target="field_name", operator="keyword_contains", params={"keywords": ["bank_card"]})
                    ],
                )
            ],
        )
        loader._profile_cache["finance"] = profile

        taxonomy = DomainTaxonomy(
            domain="default",
            standard_id="INTERNAL",
            levels={"L1": SensitivityLevelDef(id="L1", name="L1", rank=1),
                    "L3": SensitivityLevelDef(id="L3", name="L3", rank=3),
                    "L4": SensitivityLevelDef(id="L4", name="L4", rank=4)},
            categories={"FINANCIAL": CategoryDef(id="FINANCIAL", name="金融")},
            default_level="L1",
        )
        loader._taxonomy_cache["default"] = taxonomy

        # 构造标准定义: 将 RULE_BANK_CARD 的 level 覆盖降为 L3
        std_def = StandardDef(
            standard_id="custom_std",
            taxonomy="default",
            domains=["finance"],
            rule_overrides={
                "RULE_BANK_CARD": {"level": "L3"}
            },
        )
        loader._standard_cache["custom_std"] = std_def

        engine = loader.get_engine(standard="custom_std")
        tags, _suppressed = engine.evaluate("bank_card_no", "6222021001123456789")
        assert len(tags) == 1
        assert tags[0].level == "L3"  # 成功从 L4 覆盖升级为 L3


class TestDowngradeRules:
    """降级规则测试"""

    def test_downgrade_rule_execution(self):
        taxonomy = DomainTaxonomy(
            domain="default",
            standard_id="INTERNAL",
            levels={"L1": SensitivityLevelDef(id="L1", name="L1", rank=1),
                    "L2": SensitivityLevelDef(id="L2", name="L2", rank=2),
                    "L3": SensitivityLevelDef(id="L3", name="L3", rank=3)},
            categories={"MGMT": CategoryDef(id="MGMT", name="管理信息")},
            default_level="L3",
        )

        profile = RuleProfile(
            domain="general",
            rules=[],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_OPS",
                    name="运营指标降级",
                    keywords=["turnover", "inventory"],
                    level="L2",
                    category="MGMT",
                )
            ],
        )

        from engine.dynclassification.engine import ConfigurableRuleEngine
        engine = ConfigurableRuleEngine(taxonomy, [profile])

        # 匹配降级关键词 "inventory"
        tags, _suppressed = engine.evaluate("device_inventory_count", None)
        assert len(tags) == 1
        assert tags[0].level == "L2"
        assert tags[0].category == "MGMT"
        assert tags[0].rule_id == "DOWN_OPS"
