"""针对最后剩余 3% 的死角分支补充全覆盖测试 / 100% Coverage Ultimate Boost Tests."""

from pathlib import Path
import pytest
import yaml

from engine.dynclassification.composite import CompositeRuleEngine
from engine.dynclassification.engine import ConfigurableRuleEngine
from engine.dynclassification.standard_profile_generator import StandardProfileGenerator
from engine.dynclassification.operator_registry import OperatorRegistry
from engine.dynclassification.operators import _validate_id_card, _validate_medical_card, _in_icd10_interval
from engine.dynclassification.profile_loader import ProfileLoader
from engine.dynclassification.models import (
    DomainTaxonomy, SecurityTag, SensitivityLevelDef, CategoryDef,
)
from engine.dynclassification.rule_schema import RuleProfile, RuleDef, MatcherDef
from engine.dynclassification.service import DynClassificationService


class TestUltimateCoverageDetails:

    def test_composite_rule_engine_apply_record_level(self):
        """测试 CompositeRuleEngine.apply_to_record_level 边界"""
        comp_engine = CompositeRuleEngine(rules=[])

        # 无 composite_tags 时保留原 level
        assert comp_engine.apply_to_record_level("L3", []) == "L3"

        # 无 taxonomy 时使用字符串 max
        tags = [SecurityTag(level="L5", category="PII"), SecurityTag(level="L2", category="PII")]
        assert comp_engine.apply_to_record_level("L3", tags, taxonomy=None) == "L5"

    def test_configurable_rule_engine_evaluate_batch(self):
        """测试 ConfigurableRuleEngine.evaluate_batch"""
        taxonomy = DomainTaxonomy(domain="default", standard_id="INTERNAL", default_level="L1")
        engine = ConfigurableRuleEngine(taxonomy, [])
        res = engine.evaluate_batch([("f1", "v1"), ("f2", "v2")])
        assert "f1" in res and "f2" in res

    def test_operators_helper_functions_branches(self):
        """测试 operators.py 中辅助校验函数的内部异常分支"""
        # _validate_id_card (Int 校验中 IndexError/ValueError 捕获)
        assert _validate_id_card("1101011990030723X") is False
        assert _validate_id_card("110101199003072375") is True

        # _validate_medical_card (匹配逻辑)
        assert _validate_medical_card("123456780") is False

        # _in_icd10_interval (包含非法 start/end 编码)
        assert _in_icd10_interval(("B", 20), "INVALID_START", "B24") is False
        assert _in_icd10_interval(("B", 20), "B20", "INVALID_END") is False

    def test_generator_standard_id_variants(self, tmp_path):
        """测试 generator.py 识别多种行标编码 (GB/T 35273, GB/T 43697)"""
        print("tmp_path=", tmp_path)
        gb35273_md = tmp_path / "gb35273.md"
        gb35273_md.write_text("# 规范\n标准编号：GB/T 35273-2020", encoding="utf-8")
        assert StandardProfileGenerator(gb35273_md)._extract_standard_id() == "gb35273"


        gb43697_md = tmp_path / "gb43697.md"
        gb43697_md.write_text("# 规范\n标准编号：GB/T 43697-2024", encoding="utf-8")
        assert StandardProfileGenerator(gb43697_md)._extract_standard_id() == "gb43697"

    def test_profile_loader_apply_rule_overrides(self, tmp_path):
        """测试 ProfileLoader._apply_rule_overrides 覆盖分支"""
        loader = ProfileLoader(rules_dir=tmp_path)
        (tmp_path / "taxonomies").mkdir(exist_ok=True)
        (tmp_path / "domains").mkdir(exist_ok=True)
        (tmp_path / "standards").mkdir(exist_ok=True)

        (tmp_path / "taxonomies" / "default.yaml").write_text("domain: default\nstandard_id: INTERNAL\ndefault_level: L1", encoding="utf-8")
        (tmp_path / "domains" / "finance.yaml").write_text("""
domain: finance
rules:
  - id: RULE_A
    category: PII
    level: L4
    matchers: []
""", encoding="utf-8")
        (tmp_path / "standards" / "std_override.yaml").write_text("""
standard_id: std_override
taxonomy: default
domains: [finance]
rule_overrides:
  RULE_A:
    level: L2
""", encoding="utf-8")

        engine = loader.get_engine(standard="std_override")
        assert engine.rules[0].level == "L2"


# ===========================================================================
# 金融标准 (C1~C4) 端到端测试
# ===========================================================================


class TestFinanceStandardE2E:
    """金融标准 C1~C4 体系的完整分类流程测试。"""

    @pytest.fixture()
    def finance_rules_dir(self, tmp_path):
        """构建金融标准规则目录。"""
        tax_dir = tmp_path / "taxonomies"
        dom_dir = tmp_path / "domains"
        std_dir = tmp_path / "standards"
        tax_dir.mkdir()
        dom_dir.mkdir()
        std_dir.mkdir()

        (tax_dir / "finance_jrt0197.yaml").write_text("""
domain: finance
standard_id: jrt0197
version: "1.0.0"
levels:
  C1: {id: C1, name: 第1级, rank: 1}
  C2: {id: C2, name: 第2级, rank: 2}
  C3: {id: C3, name: 第3级, rank: 3}
  C4: {id: C4, name: 第4级, rank: 4}
categories:
  FINANCIAL_ACCOUNT: {id: FINANCIAL_ACCOUNT, name: 金融账户数据}
  PERSONAL_BASIC: {id: PERSONAL_BASIC, name: 个人基本信息}
default_level: C2
""", encoding="utf-8")

        (dom_dir / "finance.yaml").write_text("""
domain: finance
rules:
  - id: RULE_BANK_CARD
    name: 银行卡号
    category: FINANCIAL_ACCOUNT
    level: C4
    priority: 200
    matchers:
      - target: field_name
        operator: keyword_contains
        params: {keywords: ["bank_card", "card_no"]}
  - id: RULE_PHONE
    name: 手机号
    category: PERSONAL_BASIC
    level: C3
    priority: 100
    matchers:
      - target: field_value
        operator: regex
        params: {pattern: "^1[3-9]\\\\d{9}$"}
""", encoding="utf-8")

        (std_dir / "jrt0197.yaml").write_text("""
standard_id: jrt0197
taxonomy: finance_jrt0197
domains: [finance]
""", encoding="utf-8")

        return tmp_path

    def test_finance_field_classification_c4(self, finance_rules_dir):
        """金融标准下银行卡字段应返回 C4 等级。"""
        svc = DynClassificationService(rules_dir=finance_rules_dir)
        resp = svc.classify_field("bank_card_no", "6222021234567890", standard="jrt0197")

        assert resp.field_result is not None
        assert resp.field_result.final_level == "C4"
        assert any(t.level == "C4" for t in resp.field_result.tags)
        assert resp.field_result.tags[0].category == "FINANCIAL_ACCOUNT"

    def test_finance_field_classification_c3(self, finance_rules_dir):
        """金融标准下手机号值应返回 C3 等级。"""
        svc = DynClassificationService(rules_dir=finance_rules_dir)
        resp = svc.classify_field("contact", "13800138000", standard="jrt0197")

        assert resp.field_result is not None
        assert resp.field_result.final_level == "C3"

    def test_finance_default_level_c2(self, finance_rules_dir):
        """金融标准下无规则命中时回退到 default_level=C2。"""
        svc = DynClassificationService(rules_dir=finance_rules_dir)
        resp = svc.classify_field("unknown_field", "some_value", standard="jrt0197")

        assert resp.field_result is not None
        assert resp.field_result.final_level == "C2"
        assert len(resp.field_result.tags) == 0


# ===========================================================================
# classify_table 边界测试
# ===========================================================================


class TestClassifyTableEdgeCases:
    """classify_table 边界场景测试。"""

    @pytest.fixture()
    def simple_service(self, tmp_path):
        """构建简单规则服务。"""
        tax_dir = tmp_path / "taxonomies"
        dom_dir = tmp_path / "domains"
        tax_dir.mkdir()
        dom_dir.mkdir()

        (tax_dir / "default.yaml").write_text("""
domain: default
standard_id: INTERNAL
levels:
  L1: {id: L1, name: 公开, rank: 1}
  L3: {id: L3, name: 敏感, rank: 3}
categories:
  PII: {id: PII, name: 个人信息}
default_level: L1
""", encoding="utf-8")

        (dom_dir / "general-pii.yaml").write_text("""
domain: general-pii
rules:
  - id: RULE_PHONE
    category: PII
    level: L3
    matchers:
      - target: field_value
        operator: regex
        params: {pattern: "^1[3-9]\\\\d{9}$"}
""", encoding="utf-8")

        return DynClassificationService(rules_dir=tmp_path)

    def test_classify_table_empty_rows(self, simple_service):
        """空 rows 应返回默认等级且无标签。"""
        resp = simple_service.classify_table(
            schema=["name", "phone"],
            rows=[],
            domain="general-pii",
        )
        assert resp.table_result is not None
        assert resp.table_result.final_level == "L1"  # default_level
        assert len(resp.table_result.aggregated_tags) == 0

    def test_classify_table_single_row(self, simple_service):
        """单行记录应正确分类。"""
        resp = simple_service.classify_table(
            schema=["name", "phone"],
            rows=[{"name": "张三", "phone": "13800138000"}],
            domain="general-pii",
        )
        assert resp.table_result is not None
        assert resp.table_result.final_level == "L3"
        assert len(resp.table_result.record_results) == 1

    def test_classify_table_schema_mismatch(self, simple_service):
        """当 rows 字段与 schema 不匹配时不应崩溃。"""
        resp = simple_service.classify_table(
            schema=["col_a", "col_b"],
            rows=[{"phone": "13800138000", "extra": "data"}],
            domain="general-pii",
        )
        # 不应抛异常，正常返回结果
        assert resp.table_result is not None
        assert resp.table_result.final_level == "L3"
