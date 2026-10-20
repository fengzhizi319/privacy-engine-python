"""从标准分级文档自动生成 YAML 配置文件功能测试。"""

from pathlib import Path
import pytest

from engine.dynclassification import DynClassificationService, StandardProfileGenerator


def test_generate_profiles_from_sichuan_doc(tmp_path):
    """测试解析《四川省健康医疗大数据应用指南.md》生成三套 YAML 配置文件。"""
    doc_path = Path("docs/standard/四川省健康医疗大数据应用指南.md")
    assert doc_path.exists(), "标准文档文件必须存在"

    # 1. 直接测试 StandardProfileGenerator
    parser = StandardProfileGenerator(doc_path)
    taxonomy, profile, standard_def = parser.parse()

    assert taxonomy.standard_id == "sc_health_db51"
    assert "L1" in taxonomy.levels
    assert "L5" in taxonomy.levels
    assert "PERSONAL_BASIC" in taxonomy.categories
    assert len(profile.rules) > 0

    # 2. 测试写入临时 output_dir
    generated = parser.generate_files(tmp_path)

    tax_file = generated["taxonomy"]
    dom_file = generated["domain"]
    std_file = generated["standard"]

    assert tax_file.exists()
    assert dom_file.exists()
    assert std_file.exists()
    assert tax_file.name == "sc_health_db51.yaml"

    # 3. 测试通过 DynClassificationService 加载生成出的配置文件并进行求值
    service = DynClassificationService(rules_dir=tmp_path)
    # 使用自动生成的标准 sc_health_db51 评估字段
    resp = service.classify_field("patient_brca1_gene", "rs80357906", standard="sc_health_db51")

    assert resp.field_result is not None
    assert len(resp.field_result.tags) > 0
    assert resp.field_result.final_level in ["L5", "L4", "L3"]
