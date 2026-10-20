"""针对 dynclassification 模块未覆盖分支的深度提升测试 / Full Coverage Boost Tests."""

from pathlib import Path
import pytest
import yaml

from engine.dynclassification.standard_profile_generator import StandardProfileGenerator, main as generator_main
from engine.dynclassification.models import DomainTaxonomy, SensitivityLevelDef, CategoryDef, SecurityTag
from engine.dynclassification.operator_registry import OperatorRegistry
from engine.dynclassification.profile_loader import ProfileLoader
from engine.dynclassification.engine import ConfigurableRuleEngine
from engine.dynclassification.rule_schema import RuleProfile, RuleDef, MatcherDef
from engine.dynclassification.service import DynClassificationService



class TestModelsCoverage:
    """models.py 补充测试"""

    def test_category_path_traversal(self):
        """测试 CategoryDef.get_category_path() 祖先树追溯与循环引用防护"""
        taxonomy = DomainTaxonomy(
            domain="test",
            standard_id="test",
            categories={
                "ROOT": CategoryDef(id="ROOT", name="根分类"),
                "SUB_1": CategoryDef(id="SUB_1", name="一级分类", parent_id="ROOT"),
                "SUB_2": CategoryDef(id="SUB_2", name="二级分类", parent_id="SUB_1"),
                # 构造循环父节点防护测试
                "LOOP_A": CategoryDef(id="LOOP_A", name="A", parent_id="LOOP_B"),
                "LOOP_B": CategoryDef(id="LOOP_B", name="B", parent_id="LOOP_A"),
            },
        )

        path = taxonomy.get_category_path("SUB_2")
        assert path == ["ROOT", "SUB_1", "SUB_2"]

        # 防护无限循环
        loop_path = taxonomy.get_category_path("LOOP_A")
        assert "LOOP_A" in loop_path and "LOOP_B" in loop_path

    def test_security_tag_str(self):
        tag = SecurityTag(level="L3", category="PII")
        assert str(tag) == "L3_PII"


class TestOperatorRegistryCoverage:
    """operator_registry.py 补充测试"""

    def test_has_and_list(self):
        assert OperatorRegistry.has("regex") is True
        assert OperatorRegistry.has("non_existent_op") is False


class TestGeneratorCoverage:
    """standard_doc_generator.py 包含各类多行业标准识别与分支测试"""

    def test_parser_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            StandardProfileGenerator("non_existent_doc_12345.md")

    def test_finance_doc_parsing(self, tmp_path):
        """测试金融行业 (JR/T 0197) 文本的抽出"""
        finance_md = tmp_path / "finance_spec.md"
        finance_md.write_text(
            """# 金融数据安全分级指南
标准编号：JR/T 0197-2020
包含 C1, C2, C3, C4 分级和金融账户数据、支付卡号等举例。
""",
            encoding="utf-8",
        )

        parser = StandardProfileGenerator(finance_md)
        taxonomy, profile, std_def = parser.parse()

        assert taxonomy.standard_id == "jrt0197"
        assert "C1" in taxonomy.levels
        assert "C4" in taxonomy.levels
        assert "FINANCIAL_ACCOUNT" in taxonomy.categories

    def test_guangdong_doc_parsing(self, tmp_path):
        """测试广东省技术规范文本抽取"""
        gd_md = tmp_path / "广东省健康医疗数据规范.md"
        gd_md.write_text(
            """# 广东省健康医疗数据安全分类分级管理技术规范
标准编号：DB44/T 1234
包含个人基本信息数据和诊疗信息数据。
""",
            encoding="utf-8",
        )

        parser = StandardProfileGenerator(gd_md)
        taxonomy, _, _ = parser.parse()
        assert taxonomy.standard_id == "gd_health_db44"

    def test_fallback_slug_parsing(self, tmp_path):
        """测试通用无法识别标准的退役解析"""
        generic_md = tmp_path / "custom_industry.md"
        generic_md.write_text(
            """# 自定义行业规范说明
包含通用个人信息和业务数据。
""",
            encoding="utf-8",
        )

        parser = StandardProfileGenerator(generic_md)
        taxonomy, _, _ = parser.parse()
        assert taxonomy.standard_id == "custom_industry"

    def test_generator_main_cli(self, tmp_path, monkeypatch):
        """测试 generator.py 的 main() CLI 命令行"""
        doc_file = tmp_path / "test_doc.md"
        doc_file.write_text("# 测试规范文档\n标准编号：DB51/T 2989", encoding="utf-8")
        out_dir = tmp_path / "out_rules"

        monkeypatch.setattr("sys.argv", ["generator", "--doc", str(doc_file), "--output", str(out_dir)])
        generator_main()

        assert (out_dir / "taxonomies" / "sc_health_db51.yaml").exists()


class TestProfileLoaderCoverage:
    """profile_loader.py 缺失路径与发现方法测试"""

    def test_loader_discovery_and_defaults(self, tmp_path):
        loader = ProfileLoader(rules_dir=tmp_path)

        # 目录不存在时
        assert loader.list_taxonomies() == []
        assert loader.list_domains() == []
        assert loader.list_standards() == []

        # 建立目录结构
        (tmp_path / "taxonomies").mkdir()
        (tmp_path / "domains").mkdir()
        (tmp_path / "standards").mkdir()

        (tmp_path / "taxonomies" / "default.yaml").write_text("domain: default\nstandard_id: INTERNAL\ndefault_level: L1", encoding="utf-8")
        (tmp_path / "domains" / "general-pii.yaml").write_text("domain: general-pii\nrules: []", encoding="utf-8")
        (tmp_path / "standards" / "std_a.yaml").write_text("standard_id: std_a\ntaxonomy: default\ndomains: [general-pii]", encoding="utf-8")

        assert "default" in loader.list_taxonomies()
        assert "general-pii" in loader.list_domains()
        assert "std_a" in loader.list_standards()

        # 测试默认引擎构建 (无 domain, 无 standard)
        engine_def = loader.get_engine()
        assert engine_def.domain == "default"

        # 测试纯 domain 引擎构建
        engine_dom = loader.get_engine(domain="general-pii")
        assert engine_dom.domain == "general-pii"

        # 测试复合引擎获取
        comp_engine = loader.get_composite_engine(domain="general-pii")
        assert comp_engine.domain == "general-pii"

        comp_engine_std = loader.get_composite_engine(standard="std_a")
        assert comp_engine_std.standard_id == "std_a"

        comp_engine_default = loader.get_composite_engine()
        assert comp_engine_default.domain == "default"


class TestEngineAndServiceCoverage:
    """engine.py 与 service.py 性能与异常分支覆盖"""

    def test_engine_properties_and_exception_log(self, caplog):
        taxonomy = DomainTaxonomy(domain="default", standard_id="INTERNAL", default_level="L1")
        engine = ConfigurableRuleEngine(taxonomy, [])

        assert engine.rule_count == 0
        assert engine.downgrade_rule_count == 0

        # 测试算子运行时异常引发 logger 结构化记录
        @OperatorRegistry.register("op_throws_error")
        def buggy_op(val, params):
            raise RuntimeError("算子运行崩溃测试")

        from engine.dynclassification.rule_schema import RuleDef, MatcherDef
        profile = RuleProfile(
            domain="buggy",
            rules=[
                RuleDef(
                    id="BUGGY_RULE",
                    category="TEST",
                    level="L1",
                    matchers=[MatcherDef(target="field_value", operator="op_throws_error")],
                )
            ],
        )

        buggy_engine = ConfigurableRuleEngine(taxonomy, [profile])
        tags, _suppressed = buggy_engine.evaluate("field", "value")

        assert len(tags) == 0  # 异常安全捕获
        assert "operator_execution_failed" in caplog.text

    def test_service_generate_profile_from_doc(self, tmp_path):
        """测试 DynClassificationService 的 generate_profile_from_doc() API"""
        doc_file = tmp_path / "spec.md"
        doc_file.write_text("# 医疗标准文档\n包含身份证和手机号码等隐私字段", encoding="utf-8")

        service = DynClassificationService(rules_dir=tmp_path / "rules")
        res = service.generate_profile_from_doc(doc_file)

        assert "taxonomy" in res
        assert "domain" in res
        assert "standard" in res
