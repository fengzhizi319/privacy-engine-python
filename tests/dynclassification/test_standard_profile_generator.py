"""标准 Markdown 文档生成 YAML 规则测试 / Standard Markdown Document Generator Tests.

测试覆盖场景：
- StandardProfileGenerator 解析 Markdown 文档并生成完整的三套 YAML 模型 (Taxonomy, Profile, StandardDef)
- 验证自动生成的 Profile 包含 full schema 字段（包含 default match_logic, force_suppress, exempt_rules 等全量参考字段）
- 验证写入临时目录后的 YAML 文件语法合法且包含预期结构
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from engine.dynclassification.standard_profile_generator import StandardProfileGenerator


@pytest.fixture()
def dummy_markdown_doc(tmp_path: Path) -> Path:
    """构造一个模拟分类分级标准的 Markdown 文件。"""
    md_content = """# 四川省健康医疗大数据应用指南 (DB51/T 2851-2021)
标准编号：DB51/T 2851-2021

## 一、 数据分类体系
1. 个人基本信息数据 (PERSONAL_BASIC)
2. 诊疗信息数据 (MEDICAL_TREATMENT)
3. 管理信息数据 (MANAGEMENT)

## 二、 敏感度等级定义
- 第1级 (L1): 公开数据
- 第2级 (L2): 内部数据
- 第3级 (L3): 敏感数据
- 第4级 (L4): 高敏感数据
- 第5级 (L5): 极敏感数据 (基因)

## 三、 规则词条举例
- 身份证件号码检测 (身份证, idcard)
- 手机号码检测 (手机, 电话)
- 敏感病种检测 (艾滋病, 性病, 精神病)
- 个人遗传基因数据检测 (基因, 染色体, 地中海贫血)
"""
    doc_file = tmp_path / "test_standard_db51.md"
    doc_file.write_text(md_content, encoding="utf-8")
    return doc_file


def test_standard_profile_parser(dummy_markdown_doc: Path, tmp_path: Path):
    """测试 StandardProfileGenerator 能否成功解析文档并导出包含全量参考字段的 YAML。"""
    parser = StandardProfileGenerator(dummy_markdown_doc)
    taxonomy, profile, standard_def = parser.parse()

    assert taxonomy.domain == "sc_health_db51"
    assert "L1" in taxonomy.levels
    assert len(profile.rules) >= 4
    assert len(profile.downgrade_rules) >= 1

    ops_down = profile.downgrade_rules[0]
    assert hasattr(ops_down, "exempt_rules")
    assert hasattr(ops_down, "force_suppress")
    assert hasattr(ops_down, "max_force_suppress_level")

    output_dir = tmp_path / "rules_output"
    generated_files = parser.generate_files(output_dir)

    assert generated_files["domain"].exists()
    assert generated_files["taxonomy"].exists()
    assert generated_files["standard"].exists()

    domain_yaml_data = yaml.safe_load(generated_files["domain"].read_text(encoding="utf-8"))
    assert "downgrade_rules" in domain_yaml_data
    down_yaml = domain_yaml_data["downgrade_rules"][0]
    assert "override" in down_yaml or "force_suppress" in down_yaml
    assert "exempt_rules" in down_yaml or "exclude_rules" in down_yaml


def test_parser_resilience_negation_filter(tmp_path: Path):
    """测试否定与排除句式过滤，验证误报率控制与代码韧性。"""
    negation_md = """# 测试排除否定规范
标准编号：GB/T 99999-2026

## 一、 范围与例外
- 本规范不适用于个人基因数据及基因组检测信息。
- 本规范不包含艾滋病诊疗记录。

## 二、 参考文献
- 包含参考文献中提到的手机与身份证讨论。
"""
    doc_path = tmp_path / "negation_test.md"
    doc_path.write_text(negation_md, encoding="utf-8")

    parser = StandardProfileGenerator(doc_path)
    _, profile, _ = parser.parse()

    rule_ids = [r.id for r in profile.rules]
    # 验证: 带有"不适用于/不包含"的否定词不应产生误报规则 (GENOMIC 和 DISEASE)
    assert not any("GENOMIC" in rid for rid in rule_ids)
    assert not any("DISEASE" in rid for rid in rule_ids)


def test_parser_resilience_synonym_trigger(tmp_path: Path):
    """测试同义词与多维度词汇识别，验证召回率与代码韧性。"""
    synonym_md = """# 同义词规范
标准编号：GB/T 88888-2026

## 正文
- 包含公民身份号码识别。
- 包含移动电话与联系电话信息。
"""
    doc_path = tmp_path / "synonym_test.md"
    doc_path.write_text(synonym_md, encoding="utf-8")

    parser = StandardProfileGenerator(doc_path)
    _, profile, _ = parser.parse()

    rule_ids = [r.id for r in profile.rules]
    # 验证: 同义词 "公民身份号码" 和 "移动电话" 成功召回规则
    assert any("IDCARD" in rid for rid in rule_ids)
    assert any("PHONE" in rid for rid in rule_ids)


def _write_and_parse(md_content: str, tmp_path: Path, filename: str = "test.md") -> list[str]:
    """辅助函数：写入临时 Markdown 并返回生成的规则 ID 列表。"""
    doc_path = tmp_path / filename
    doc_path.write_text(md_content, encoding="utf-8")
    _, profile, _ = StandardProfileGenerator(doc_path).parse()
    return [r.id for r in profile.rules]


@pytest.fixture()
def common_markdown_prefix() -> str:
    """返回一份带基础等级与分类定义的 Markdown 前缀，供边界用例复用。"""
    return """# 测试规范
标准编号：GB/T 77777-2026

## 一、 数据分类
1. 个人基本信息数据 (PERSONAL_BASIC)
2. 诊疗信息数据 (MEDICAL_TREATMENT)

## 二、 敏感度等级
- 第1级 (L1): 公开数据
- 第2级 (L2): 内部数据
- 第3级 (L3): 敏感数据
- 第4级 (L4): 高敏感数据
- 第5级 (L5): 极敏感数据

## 三、 规则词条
"""


def test_parser_mixed_clause_does_not_over_suppress(common_markdown_prefix: str, tmp_path: Path):
    """转折句中前半句的关键词不应被后半句的否定误杀。"""
    md_content = common_markdown_prefix + """
- 本标准包含身份证数据，但不含基因数据。
- 本标准包含手机号码。
"""
    rule_ids = _write_and_parse(md_content, tmp_path, "mixed_clause.md")
    assert any("IDCARD" in rid for rid in rule_ids)
    assert any("PHONE" in rid for rid in rule_ids)
    assert not any("GENOMIC" in rid for rid in rule_ids)


def test_parser_prohibitive_is_positive_signal(common_markdown_prefix: str, tmp_path: Path):
    """'禁止/严禁'表示数据受监管，应生成规则而不是被排除。"""
    md_content = common_markdown_prefix + """
- 严格禁止非法收集身份证件号码和手机号码。
"""
    rule_ids = _write_and_parse(md_content, tmp_path, "prohibitive.md")
    assert any("IDCARD" in rid for rid in rule_ids)
    assert any("PHONE" in rid for rid in rule_ids)


def test_parser_exception_for_other_term_does_not_suppress(common_markdown_prefix: str, tmp_path: Path):
    """'例外'修饰的是另一个对象时，不应误杀当前关键词。"""
    md_content = common_markdown_prefix + """
- 身份证属于敏感信息，护照号码为例外情况。
"""
    rule_ids = _write_and_parse(md_content, tmp_path, "exception_other.md")
    assert any("IDCARD" in rid for rid in rule_ids)


def test_parser_comma_split_negation(common_markdown_prefix: str, tmp_path: Path):
    """逗号分隔的正负分句应独立处理，前半句命中、后半句否定互不影响。"""
    md_content = common_markdown_prefix + """
- 本标准包含身份证，但不含基因数据。
- 本标准包含手机，但不含艾滋病诊疗记录。
"""
    rule_ids = _write_and_parse(md_content, tmp_path, "comma_split.md")
    assert any("IDCARD" in rid for rid in rule_ids)
    assert any("PHONE" in rid for rid in rule_ids)
    assert not any("GENOMIC" in rid for rid in rule_ids)
    assert not any("DISEASE" in rid for rid in rule_ids)

