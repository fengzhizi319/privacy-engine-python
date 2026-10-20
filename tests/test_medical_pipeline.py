"""医疗数据分类分级与脱敏 Pipeline 单元测试模块。
Unit Tests for Medical Privacy Pipeline.
"""

import csv
from pathlib import Path
import pytest

from engine.medical_pipeline.pipeline import (
    MedicalPrivacyPipeline,
    process_medical_dataset,
)
from engine.medical_pipeline.rules import (
    L4_PATTERNS,
    L5_PATTERNS,
    redact_medical_text,
    redact_medical_text_with_ner,
)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "data"))
from generate_medical_data import gen_id_card, generate_dataset


def test_generate_valid_id_card_checksum() -> None:
    """验证生成的身份证号码均符合 GB 11643-1999 校验码算法。"""
    id_card = gen_id_card()
    assert len(id_card) == 18
    
    # 校验码重新计算
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    checksum_map = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
    total = sum(int(id_card[i]) * weights[i] for i in range(17))
    expected_check = checksum_map[total % 11]
    assert id_card[-1].upper() == expected_check


def test_generated_dataset_fields_count() -> None:
    """验证生成的医疗数据集记录数量 (20) 与字段数量 (27 列)。"""
    records = generate_dataset(20)
    assert len(records) == 20
    assert len(records[0].keys()) == 27
    
    expected_fields = [
        "gender", "age", "diagnosis_name", "chief_complaint", "present_illness",
        "past_history", "personal_history", "is_smoking", "smoking_duration",
        "family_history", "allergic_history", "department", "height", "weight",
        "disability_category", "disability_level", "assess_type_name",
        "assess_result_name", "assess_score", "assess_time", "progress_note",
        "progress_note_time", "name", "id_card_no", "registered_address",
        "disability_cert_no", "medical_insurance_no",
    ]
    for field in expected_fields:
        assert field in records[0]


def test_medical_privacy_pipeline_dual_output() -> None:
    """验证 Pipeline 处理后生成符合结构的双重输出（分级报告 + 脱敏数据）。"""
    records = generate_dataset(5)
    pipeline = MedicalPrivacyPipeline()
    result = pipeline.process_records(records)
    
    # 验证分类分级报告
    assert len(result.classification_report) == 5
    first_report = result.classification_report[0]
    assert "record_index" in first_report
    assert "max_level" in first_report
    assert "field_details" in first_report
    
    # 验证脱敏数据
    assert len(result.sanitized_data) == 5
    first_sanitized = result.sanitized_data[0]
    assert len(first_sanitized.keys()) == 27
    
    # 验证 Summary 字段
    assert result.summary["total_records"] == 5
    assert result.summary["guarantee_no_l4_l5_raw_data"] is True


def test_medical_privacy_pipeline_no_raw_l4_l5_leak() -> None:
    """验证经 Pipeline 脱敏后的 sanitized_data 中绝对没有任何 L4/L5 级原始高危词汇。"""
    records = generate_dataset(20)
    res = process_medical_dataset(records)
    
    forbidden_terms = [
        "获得性免疫缺陷综合征", "HIV感染", "艾滋病", "重度精神分裂症",
        "幻听（命令性言语）", "被害妄想", "自伤倾向", "冲动砸物",
        "浸润性腺癌", "恶性肿瘤", "亨廷顿舞蹈病", "乙型肝炎", "丙型肝炎"
    ]
    
    for row in res.sanitized_data:
        # PII 脱敏校验
        assert "*" in row["id_card_no"]
        assert "*" in row["name"]
        
        # L4/L5 绝无泄漏校验
        for key, val in row.items():
            for term in forbidden_terms:
                assert term not in val, f"字段 {key} 泄漏了 L4/L5 敏感词汇 '{term}': {val}"


def test_kangyang_csv_file_pipeline_execution() -> None:
    """从本地读取真实生成的 kangyang.csv 并执行全流程测试。"""
    csv_path = Path(__file__).resolve().parent.parent / "engine" / "medical_pipeline" / "samples" / "kangyang.csv"
    assert csv_path.exists()
    
    records = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    assert len(records) >= 20
    res = process_medical_dataset(records)
    assert res.summary["total_records"] == len(records)
    assert res.summary["l5_records_count"] > 0
    assert res.summary["l4_records_count"] > 0


# === 增强测试：身份证批量校验、L4/L5 覆盖、图片病例、脱敏格式、替换标签泄露 ===


def test_batch_id_card_checksum_validation() -> None:
    """批量生成 50 个身份证号，全部必须通过 GB 11643-1999 MOD 11-2 校验。"""
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    checksum_map = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
    for _ in range(50):
        id_card = gen_id_card()
        assert len(id_card) == 18, f"身份证号长度不为 18 位: {id_card}"
        total = sum(int(id_card[i]) * weights[i] for i in range(17))
        expected = checksum_map[total % 11]
        assert id_card[-1].upper() == expected, f"校验码不匹配: {id_card}"


def test_generated_data_contains_l4_and_l5_content() -> None:
    """验证生成的数据中确实包含 L4 和 L5 级病史内容。
    
    使用固定种子保证可重复性，因为诊断与病史的组合是随机的。
    """
    import random
    random.seed(2026)  # 固定种子，确保可重复
    records = generate_dataset(20)
    pipeline = MedicalPrivacyPipeline()
    result = pipeline.process_records(records)
    # 分级报告应包含 L4 和 L5 级记录
    levels = {r["max_level"] for r in result.classification_report}
    assert "L5" in levels, "生成数据中缺少 L5 级记录"
    assert "L4" in levels, "生成数据中缺少 L4 级记录"


def test_generated_data_contains_image_cases() -> None:
    """验证生成数据中至少 3 条包含图片病例标记 (PACS / samples)。"""
    records = generate_dataset(20)
    image_count = sum(1 for r in records if "PACS" in r.get("present_illness", "") or "data/samples/" in r.get("present_illness", ""))
    assert image_count >= 3, f"图片病例数量不足: {image_count}"


def test_sanitize_text_strips_all_l4_l5_terms() -> None:
    """验证 sanitize_text 方法能剥离单条文本中的所有 L4/L5 术语。"""
    pipeline = MedicalPrivacyPipeline()
    # 混合 L5 (HIV) + L4 (恶性肿瘤) 的文本
    mixed_text = "患者HIV抗体阳性，同时确诊为恶性肿瘤，建议进一步检查。"
    sanitized = pipeline.sanitize_text(mixed_text)
    assert "HIV" not in sanitized
    assert "艾滋" not in sanitized
    assert "恶性肿瘤" not in sanitized
    assert "[L5-" in sanitized  # L5 替换标记存在
    assert "[L4-" in sanitized  # L4 替换标记存在


def test_replacement_tags_do_not_leak_sensitive_terms() -> None:
    """验证 L4/L5 替换标签中不包含原始敏感词汇（如 HIV、乙肝等）。"""
    leaked_terms = ["HIV", "艾滋", "精神分裂", "亨廷顿", "恶性肿瘤", "胃癌", "乙肝", "丙肝"]
    for _pat, replacement in L5_PATTERNS + L4_PATTERNS:
        for term in leaked_terms:
            assert term not in replacement, (
                f"替换标签 '{replacement}' 中泄露了敏感词 '{term}'"
            )


def test_pii_masking_format_id_card() -> None:
    """验证身份证号脱敏后保留前 6 后 4，中间为 8 个 *。"""
    records = [{"name": "张三", "id_card_no": "110101199003071234"}]
    res = process_medical_dataset(records)
    masked_id = res.sanitized_data[0]["id_card_no"]
    assert masked_id.startswith("110101"), f"身份证前 6 位应保留: {masked_id}"
    assert "********" in masked_id, f"身份证中间应为 8 个 *: {masked_id}"


def test_pii_masking_format_name() -> None:
    """验证姓名脱敏后首字保留、其余用 * 替代。"""
    records = [{"name": "张三丰", "id_card_no": "110101199003071234"}]
    res = process_medical_dataset(records)
    masked_name = res.sanitized_data[0]["name"]
    assert masked_name.startswith("张"), f"姓名首字应保留: {masked_name}"
    assert "*" in masked_name, f"姓名中间应含 *: {masked_name}"
    assert masked_name != "张三丰", "姓名不应为原始值"


def test_chinese_pii_field_names_are_classified_and_masked() -> None:
    """中文字段名应与对应英文规范字段使用相同的分类和脱敏策略。"""
    records = [{
        "姓名": "张三丰",
        "身份证号": "110101199003071234",
        "家庭住址": "北京市朝阳区幸福路100号",
        "残疾证号": "残疾证123456789",
        "医保卡号": "医保卡987654321",
    }]

    result = process_medical_dataset(records)
    masked = result.sanitized_data[0]
    report = result.classification_report[0]

    assert set(report["pii_fields_detected"]) == set(records[0])
    assert masked["姓名"] != records[0]["姓名"]
    assert masked["身份证号"].startswith("110101")
    assert "********" in masked["身份证号"]
    assert masked["家庭住址"] != records[0]["家庭住址"]
    assert masked["残疾证号"] != records[0]["残疾证号"]
    assert masked["医保卡号"] != records[0]["医保卡号"]


def test_sanitizer_rejects_unknown_mode() -> None:
    """脱敏模式拼写错误必须立即报错，不能静默进入默认分支。"""
    pipeline = MedicalPrivacyPipeline(redact_engine="rule")

    with pytest.raises(ValueError, match="Unsupported sanitization mode"):
        pipeline._medical_text_sanitizer("diagnosis_name", "乙肝", "L4", mode="redcat")


def test_pipeline_rejects_unknown_redact_engine() -> None:
    """脱敏引擎名称拼写错误必须在初始化时失败。"""
    with pytest.raises(ValueError, match="Unsupported redact_engine"):
        MedicalPrivacyPipeline(redact_engine="nerx")


def test_empty_records_handling() -> None:
    """验证空记录列表的处理不报错。"""
    res = process_medical_dataset([])
    assert res.summary["total_records"] == 0
    assert len(res.classification_report) == 0
    assert len(res.sanitized_data) == 0


def test_unified_patterns_importable_from_pipeline_masker() -> None:
    """验证 pipeline/masker.py 能正确导入统一的 L4/L5 词库。"""
    from engine.pipeline.masker import L4_PATTERNS as MP_L4, L5_PATTERNS as MP_L5
    assert len(MP_L4) == len(L4_PATTERNS)
    assert len(MP_L5) == len(L5_PATTERNS)


def test_unknown_field_l4_l5_text_is_removed() -> None:
    """未知字段名不能绕过医疗 L4/L5 文本门禁。"""
    raw = "扩展字段：患者 HIV 感染，同时存在恶性肿瘤。"
    result = process_medical_dataset([{"custom_note": raw}])
    sanitized = result.sanitized_data[0]["custom_note"]

    assert "HIV" not in sanitized
    assert "恶性肿瘤" not in sanitized
    assert sanitized != raw
    assert result.summary["guarantee_no_l4_l5_raw_data"] is True


def test_clean_text_fast_path_preserves_unmodified_text() -> None:
    """无敏感词文本应当通过 fast-path 原样保留，避免被语法自愈逻辑误篡改。"""
    cases = [
        "弟弟说'你好'，今天天气不错。",
        "母亲“高血压”控制良好。",
        "他长期。",
        "注意保暖。。。多休息",
        "第一段。\n\n第二段。",
    ]
    for text in cases:
        assert redact_medical_text(text) == text, f"干净文本不应被篡改: {text}"


def test_redact_paired_list_sensitive_suffix_cleanup() -> None:
    """顿号列表中敏感词在后场景不应遗留 '、。' 标点碎片。"""
    text = "一弟患'2型糖尿病'、'重度精神分裂症'。"
    assert redact_medical_text(text) == "一弟患'2型糖尿病'。"


def test_redact_medication_without_suffix() -> None:
    """无结尾治疗后缀的服药文本亦应被完整抹平。"""
    text = "服用'奥氮平片'20mg qd。"
    assert redact_medical_text(text) == ""


def test_redact_cause_of_death_with_complications() -> None:
    """带有并发症修饰的死因句法应重构为因病去世。"""
    text = "因'HIV'导致的并发症去世。"
    assert redact_medical_text(text) == "因病去世。"


def test_ner_only_redacts_major_sensitive_l4_l5_entities() -> None:
    """NER 引擎应仅对 L4/L5 重大高敏疾病/用药进行抹平，保留常规慢病与常用药。"""
    from engine.medical_pipeline.rules import redact_medical_text_with_ner

    class MockAdapter:
        def extract(self, text: str) -> list[dict[str, str]]:
            return [
                {"text": "高脂血症", "type": "DISEASE"},
                {"text": "阿托伐他汀", "type": "DRUG"},
                {"text": "高血压", "type": "DISEASE"},
                {"text": "重度精神分裂症", "type": "DISEASE"},
                {"text": "奥氮平片", "type": "DRUG"},
            ]

    common_text = "高脂血症病史5年，口服阿托伐他汀20mg qn。高血压病史3年，最高160/100mmHg。"
    res_common = redact_medical_text_with_ner(common_text, ner_adapter=MockAdapter())
    assert res_common == common_text, "常规慢病与常用药物不应被误脱敏"

    mixed_text = "高脂血症病史5年，口服阿托伐他汀20mg qn。长期服用奥氮平片20mg qd控制重度精神分裂症症状。"
    res_mixed = redact_medical_text_with_ner(mixed_text, ner_adapter=MockAdapter())
    assert "高脂血症" in res_mixed
    assert "阿托伐他汀" in res_mixed
def test_redact_pure_sensitive_symptom_clause_wiped_completely() -> None:
    """整句仅包含重大高敏症状与时间状语时，应直接抹平为空，不留标点或语病碎片。"""
    from engine.medical_pipeline.rules import redact_medical_text_with_ner

    class MockAdapter:
        def extract(self, text: str) -> list[dict[str, str]]:
            return [
                {"text": "幻听", "type": "SYMPTOM"},
                {"text": "被害妄想", "type": "SYMPTOM"},
            ]

    text = "幻听与被害妄想反复发作3年"
    assert redact_medical_text(text) == "", "规则引擎对纯高敏症状句应直接抹平"
    assert redact_medical_text_with_ner(text, ner_adapter=MockAdapter()) == "", "NER引擎对纯高敏症状句应直接抹平"


def test_redact_syphilis_case_complete_purge_and_path_sanitization() -> None:
    """梅毒与性传播疾病复杂病例：应完全擦除滴度、不洁接触史、硬下疳，去标识化图片文件名，且消除语病残渣。"""
    from engine.medical_pipeline.rules import redact_medical_text_with_ner

    class SyphilisMockAdapter:
        def extract(self, text: str) -> list[dict[str, str]]:
            return [
                {"text": "梅毒", "type": "DISEASE"},
                {"text": "TPPA阳性", "type": "DISEASE"},
                {"text": "RPR 1:16", "type": "DISEASE"},
                {"text": "不洁性接触史", "type": "DISEASE"},
                {"text": "无痛性溃疡", "type": "SYMPTOM"},
                {"text": "硬下疳", "type": "SYMPTOM"},
            ]

    text = (
        "患者1周前外院体检检查出'梅毒'，血清学检查示TPPA阳性，RPR 1:16。"
        "追问病史，1年前有不洁性接触史，半年前外阴曾出现无痛性溃疡(硬下疳)自愈。"
        "详见血清检验报告 data/samples/syphilis_case.png。"
    )

    res_rule = redact_medical_text(text)
    res_ner = redact_medical_text_with_ner(text, ner_adapter=SyphilisMockAdapter())

    assert "梅毒" not in res_rule and "梅毒" not in res_ner
    assert "TPPA" not in res_rule and "TPPA" not in res_ner
    assert "RPR" not in res_rule and "RPR" not in res_ner
    assert "不洁性接触史" not in res_rule and "不洁性接触史" not in res_ner
    assert "无痛性溃疡" not in res_rule and "无痛性溃疡" not in res_ner
    assert "硬下疳" not in res_rule and "硬下疳" not in res_ner
def test_redact_huntington_genetic_case_complete_purge() -> None:
    """遗传缺陷（亨廷顿舞蹈病与HTT基因CAG重复）病例：应完全擦除基因突变修饰、专用药四苯嗪、舞蹈样动作，且消除断句语病残渣。"""
    from engine.medical_pipeline.rules import redact_medical_text_with_ner

    class HuntingtonMockAdapter:
        def extract(self, text: str) -> list[dict[str, str]]:
            return [
                {"text": "遗传性亨廷顿舞蹈病", "type": "DISEASE"},
                {"text": "四肢舞蹈样动作", "type": "SYMPTOM"},
                {"text": "HTT基因CAG重复序列46次", "type": "DISEASE"},
                {"text": "四苯嗪", "type": "DRUG"},
                {"text": "舞蹈样症状", "type": "SYMPTOM"},
            ]

    text = (
        "患者2年前出现双手轻微不自主抖动与情绪易怒，近半年发展为四肢舞蹈样动作与步态不稳。"
        "基因检测提示'遗传性亨廷顿舞蹈病'(HTT基因CAG重复序列46次)。"
        "予'四苯嗪'12.5mg bid口服控制舞蹈样症状。"
    )

    res_rule = redact_medical_text(text)
    res_ner = redact_medical_text_with_ner(text, ner_adapter=HuntingtonMockAdapter())

    expected = "患者2年前双手轻微不自主抖动与情绪易怒，近半年发展为步态不稳。"
    assert res_rule == expected, f"Rule 脱敏不符合预期: {res_rule}"
    assert res_ner == expected, f"NER 脱敏不符合预期: {res_ner}"


def test_redact_psychiatric_hospital_case_complete_purge() -> None:
    """重度精神障碍病例：必须完全擦除专科就诊地点（精神卫生中心）及全部重症描述，不留'曾就诊于精神卫生中心'。"""
    from engine.medical_pipeline.rules import redact_medical_text_with_ner

    class PsychMockAdapter:
        def extract(self, text: str) -> list[dict[str, str]]:
            return [
                {"text": "言语关联妄想", "type": "SYMPTOM"},
                {"text": "命令性幻听", "type": "SYMPTOM"},
                {"text": "保护性约束倾向", "type": "SYMPTOM"},
                {"text": "重度精神分裂症", "type": "DISEASE"},
                {"text": "奥氮平片", "type": "DRUG"},
                {"text": "四苯嗪", "type": "DRUG"},
            ]

    text = (
        "患者3年前无明显诱因出现言语关联妄想、命令性幻听及保护性约束倾向。"
        "曾就诊于精神卫生中心，诊断为重度精神分裂症。"
        "长期服用'奥氮平片'20mg qd及'四苯嗪'控制症状。"
    )

    res_rule = redact_medical_text(text)
    res_ner = redact_medical_text_with_ner(text, ner_adapter=PsychMockAdapter())

    assert "精神卫生中心" not in res_rule and "精神卫生中心" not in res_ner, "专科就诊地点严禁泄露"
    assert "重度精神分裂症" not in res_rule and "重度精神分裂症" not in res_ner
    assert "奥氮平" not in res_rule and "奥氮平" not in res_ner
    assert res_rule == "", f"Rule 脱敏应抹平为空: {res_rule}"
    assert res_ner == "", f"NER 脱敏应抹平为空: {res_ner}"


def test_redact_family_history_death_and_paired_clause_syntax_fix() -> None:
    """家族史与死因病例：擦除恶性肿瘤与精神分裂症后，死因必须自然重构为'因病去世'，且不能残留'一弟患、'语法语病。"""
    from engine.medical_pipeline.rules import redact_medical_text_with_ner

    class FamilyMockAdapter:
        def extract(self, text: str) -> list[dict[str, str]]:
            return [
                {"text": "恶性肿瘤", "type": "DISEASE"},
                {"text": "重度精神分裂症", "type": "DISEASE"},
            ]

    text = "父亲因'恶性肿瘤'去世(65岁)，母亲健在。一弟患'重度精神分裂症'、'2型糖尿病'。否认其他家族遗传病史。"

    res_rule = redact_medical_text(text)
    res_ner = redact_medical_text_with_ner(text, ner_adapter=FamilyMockAdapter())

    expected = "父亲因病去世(64岁)，母亲健在。一弟患'2型糖尿病'。否认其他家族遗传病史。"

    assert "恶性肿瘤" not in res_rule and "恶性肿瘤" not in res_ner
    assert "精神分裂症" not in res_rule and "精神分裂症" not in res_ner
    assert "因去世" not in res_rule and "因去世" not in res_ner, "不应残留孤立介词'因去世'"
    assert "患、" not in res_rule and "患、" not in res_ner, "不应残留孤立顿号'患、'"
    assert res_rule == expected, f"Rule 脱敏语病修复不符合预期: {res_rule}"
    assert res_ner == expected, f"NER 脱敏语病修复不符合预期: {res_ner}"


def test_redact_family_death_causes_complex() -> None:
    """复杂家族死因句法（殁于...50岁、由...破裂出血导致去世）：死因必须自然重构为'因病去世'。"""
    from engine.medical_pipeline.rules import redact_medical_text_with_ner

    text = "外婆殁于'亨廷顿舞蹈病'(50岁)，伯父由'食管静脉曲张'破裂出血导致去世。母亲患'2型糖尿病'。"
    res_rule = redact_medical_text(text)
    res_ner = redact_medical_text_with_ner(text)

    expected = "外婆殁于48岁，伯父因病去世。母亲患'2型糖尿病'。"
    assert res_rule == expected, f"Rule 结果不符合预期: {res_rule!r}"
    assert res_ner == expected, f"NER 结果不符合预期: {res_ner!r}"


def test_redact_hepatitis_viral_load_and_biopsy_complete_purge() -> None:
    """病毒性肝炎病例：必须擦除 HBV-DNA 病毒载量、肝硬化、肝穿刺活检 G3S4 阶段及检测下限提示，完全抹平为记录。"""
    from engine.medical_pipeline.rules import redact_medical_text_with_ner

    class HepMockAdapter:
        def extract(self, text: str) -> list[dict[str, str]]:
            return [
                {"text": "慢性乙型病毒性肝炎", "type": "DISEASE"},
                {"text": "HBV-DNA 5.6×10^6 IU/mL", "type": "TEST"},
                {"text": "早期肝硬化", "type": "DISEASE"},
                {"text": "肝穿刺活检", "type": "TREATMENT"},
                {"text": "G3S4", "type": "STAGE"},
                {"text": "恩替卡韦", "type": "DRUG"},
                {"text": "HBV-DNA", "type": "TEST"},
            ]

    text = (
        "患者体检检查出'慢性乙型病毒性肝炎'(HBV-DNA 5.6×10^6 IU/mL)，"
        "腹部超声提示'早期肝硬化'改变。行肝穿刺活检提示G3S4。"
        "目前'恩替卡韦'0.5mg qd抗病毒治疗，HBV-DNA降至检测下限。"
    )

    res_rule = redact_medical_text(text)
    res_ner = redact_medical_text_with_ner(text, ner_adapter=HepMockAdapter())

    assert "HBV-DNA" not in res_rule and "HBV-DNA" not in res_ner, "HBV-DNA 病毒载量严禁泄露"
    assert "肝硬化" not in res_rule and "肝硬化" not in res_ner
    assert "G3S4" not in res_rule and "G3S4" not in res_ner
    assert "恩替卡韦" not in res_rule and "恩替卡韦" not in res_ner
    assert res_rule == "", f"Rule 脱敏应抹平为空: {res_rule}"
    assert res_ner == "", f"NER 脱敏应抹平为空: {res_ner}"



def test_failed_image_redaction_never_returns_original_value() -> None:
    """不存在或损坏图片必须 fail closed，不能把原路径当作脱敏结果。"""
    raw_path = "/tmp/medical-private-case.png"
    result = process_medical_dataset([{"case_image": raw_path}])

    assert result.sanitized_data[0]["case_image"] == "[IMAGE-REDACTION-FAILED]"
    assert result.sanitized_data[0]["case_image"] != raw_path
    assert result.summary["redaction_failures"] == 1
    assert result.summary["guarantee_no_l4_l5_raw_data"] is False


def test_sanitize_false_does_not_claim_a_safety_guarantee() -> None:
    """仅分类、不脱敏时不得报告已满足无高敏原文保证。"""
    result = process_medical_dataset([{"present_illness": "HIV 感染"}], sanitize=False)

    assert result.sanitized_data[0]["present_illness"] == "HIV 感染"
    assert result.summary["guarantee_no_l4_l5_raw_data"] is False


def test_implicit_std_description_redaction() -> None:
    """验证隐式性病描述（菜花状赘生物、醋酸白试验阳性、HPV 6/11、咪喹莫特、CO2激光）识别为 L4 级并彻底抹平。"""
    text = (
        "患者1月前发现外阴及会阴部多发菜花状赘生物，逐渐增多，伴局部轻度瘙痒与异物感。"
        "醋酸白试验阳性，HPV 6/11低危型阳性。行'CO2激光灼除术'及'咪喹莫特乳膏'局部涂抹。"
        "胸片详见 data/samples/xray_chest.png。"
    )
    res_text = redact_medical_text(text)
    assert "菜花状赘生物" not in res_text
    assert "醋酸白试验" not in res_text
    assert "HPV" not in res_text
    assert "咪喹莫特" not in res_text
    assert "CO2激光" not in res_text

    result = process_medical_dataset([
        {"present_illness": text, "chief_complaint": "外阴多发赘生物伴瘙痒1月"}
    ])
    assert result.classification_report[0]["max_level"] == "L4"
    assert result.sanitized_data[0]["present_illness"] == "胸片详见 data/samples/xray_chest.png。"
    assert result.sanitized_data[0]["chief_complaint"] == ""


def test_implicit_hepatitis_description_redaction() -> None:
    """验证乙肝隐式表征（HBsAg阳性、HBV-DNA定量、慢性乙型病毒性肝炎、恩替卡韦抗病毒治疗）识别为 L4 级且语法无残渣。"""
    text1 = (
        "患者2周前体检发现ALT 120U/L，AST 95U/L，HBsAg阳性。"
        "进一步检查发现'慢性乙型病毒性肝炎'，HBV-DNA 2.3×10^5 IU/mL。"
        "目前无发热、黄疸及腹水表现。"
    )
    text2 = "建议尽早启动恩替卡韦抗病毒治疗，注意休息，避免劳累及饮酒。"

    res1 = redact_medical_text(text1)
    assert "HBsAg阳性" not in res1
    assert "慢性乙型病毒性肝炎" not in res1
    assert "HBV-DNA" not in res1
    assert "前发" not in res1  # 严防单个字符 '现' 被误删导致 '发现' 变成 '发'
    assert not res1.endswith("腹水表。")  # 严防单个字符 '现' 被误删导致 '表现' 变成 '表'
    assert "腹水表现" in res1

    res2 = redact_medical_text(text2)
    assert "恩替卡韦" not in res2
    assert "启动，" not in res2  # 严防孤立动词残渣 '启动，'

    dataset_result = process_medical_dataset([
        {"present_illness": text1, "treatment_plan": text2}
    ])
    assert dataset_result.classification_report[0]["max_level"] == "L4"
    assert "HBsAg阳性" not in dataset_result.sanitized_data[0]["present_illness"]
    assert dataset_result.sanitized_data[0]["treatment_plan"] == "注意休息，避免劳累及饮酒。"


def test_family_history_cancer_redaction_and_age_cleanup() -> None:
    """验证家族肿瘤史中 '死于(62岁)' 的括号自愈为 '死于62岁'、'母亲患' 自愈为 '母亲患病' 且擦除消化道肿瘤。"""
    text = "父亲死于'胃癌'(62岁)，母亲患'乳腺癌'(55岁确诊)。家族中有明显消化道肿瘤聚集倾向。"

    rule_res = redact_medical_text(text)
    assert rule_res == "父亲死于62岁，母亲患病(54岁确诊)。家族中有明显消化道疾病聚集倾向。"

    ner_res = redact_medical_text_with_ner(text)
    assert ner_res == "父亲死于62岁，母亲患病(54岁确诊)。家族中有明显消化道疾病聚集倾向。"


def test_user_feedback_5_cases_fix() -> None:
    """验证用户反馈的 5 个典型脱敏案例全流程精准防护与分级。"""
    # Case 1: 梅毒 (早期隐性梅毒) -> 100% 抹平
    c1 = "梅毒 (早期隐性梅毒)"
    assert redact_medical_text(c1) == ""
    assert redact_medical_text_with_ner(c1) == ""

    # Case 2: HIV感染 (获得性免疫缺陷综合征) -> 100% 抹平 (防止将 HIV 的 IV 当成静脉注射而剩下 H)
    c2 = "HIV感染 (获得性免疫缺陷综合征)"
    assert redact_medical_text(c2) == ""
    assert redact_medical_text_with_ner(c2) == ""

    # Case 3: 长兄身亡于年龄去括号与 40岁 -> 39岁 泛化
    c3 = "长兄不幸身亡于'急性心肌梗死'(40岁)。否认其他家族病史。"
    assert redact_medical_text(c3) == "长兄身亡于39岁。否认其他家族病史。"
    assert redact_medical_text_with_ner(c3) == "长兄身亡于39岁。否认其他家族病史。"

    # Case 4: 乙肝与抗逆转录联合用药擦除
    c4 = "患者1年前查出'乙型肝炎'，HBV-DNA阳性。目前行'抗逆转录治疗'与'恩替卡韦'口服。"
    assert redact_medical_text(c4) == ""
    assert redact_medical_text_with_ner(c4) == ""

    # Case 5: 糖尿病(L2) + 奥氮平片(L5) 混合病例：定级升至 L5，仅擦除奥氮平片，保留糖尿病与二甲双胍
    c5 = "患者多饮多尿5年，检查出'2型糖尿病'，空腹血糖 11.2mmol/L。服'二甲双胍'及'奥氮平片'。"
    rule5 = redact_medical_text(c5)
    ner5 = redact_medical_text_with_ner(c5)
    expected5 = "患者多饮多尿5年，检查出'2型糖尿病'，空腹血糖 11.2mmol/L。服'二甲双胍'。"
    assert rule5 == expected5
    assert ner5 == expected5


def test_redos_catastrophic_backtracking_prevention() -> None:
    """验证 ReDoS 灾难性回溯防护：敏感词触发完整句法管线后，长空白/干扰串必须在线性时间内完成。"""
    import time
    # 干净文本：Fast-Path 原样返回（零篡改承诺）
    clean_input = "患者" + " " * 100 + "xyz"
    t0 = time.perf_counter()
    assert redact_medical_text(clean_input) == clean_input.strip()
    assert (time.perf_counter() - t0) * 1000 < 100.0

    # 真实攻击面：敏感词触发句法擦除管线后，长空白串在各可选组 \s* 槽位间引发组合回溯。
    # 修复前这些输入在用药/肝炎/病史句法正则上挂死（实测 >10s）；连续空白折叠后应秒级内完成。
    attacks = [
        "梅毒，患者" + " " * 100 + "xyz。",
        "梅毒，患者" + " " * 2000 + "xyz。",
        "梅毒，行" + " " * 500 + "noise",
        "梅毒，CD4" + " " * 300 + "xyz",
        "梅毒，" + " " * 2000 + "x",
        "梅毒，血清学" + " " * 400 + "y",
    ]
    for attack in attacks:
        t0 = time.perf_counter()
        res = redact_medical_text(attack)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 1000.0, f"ReDoS 匹配超时 ({elapsed_ms:.2f} ms)，输入长度 {len(attack)}"
        assert "梅毒" not in res


def test_all_33_sanitization_variant_bypasses() -> None:
    """验证全角/字符打散/英文病名/同义词变体全数精准捕获脱敏，无漏网之鱼。"""
    variants = [
        ("全角 ＨＩＶ 阳性", "全角阳性"),
        ("H I V 感染者", ""),  # flex 容忍字符间分隔符，"HIV感染" 复合词整词擦除
        ("艾 滋 病 患者", ""),
        ("AIDS 确诊病例", "确诊病例"),
        ("syphilis 检查报告", "检查报告"),
        ("奎硫平 20mg 服用", ""),
        ("蜘蛛痣 及 肝掌 体征", ""),
        ("CD4 计数 150/μL", ""),
        # 以下为对抗性变体回归（曾全部绕过词库端到端泄露）
        ("患者确诊为H I V感染", ""),
        ("患者确诊为H.I.V感染", ""),
        ("患者确诊为H-I-V感染", ""),
        ("患者确诊为H​IV感染", ""),  # 零宽字符插入
        ("患者确诊A I D S已一年", "患者患病已一年"),
        ("患者确诊艾-滋-病", "患者患病"),
        ("患者确诊艾·滋·病", "患者患病"),
        ("精 神 分 裂病史", ""),
        ("胃 癌术后", ""),
        ("患者感染人免疫缺陷病毒", "患者感染"),
        ("患者为获得性免疫缺陷者", ""),
        ("ＨＩＶ 阳性", "阳性"),
        ("患者确诊aizibing", "患者患病"),
        ("患者有meidu病史", "患者有"),
        ("患者患有feiai", "患者患病"),
        ("患者为yigan携带者", "患者为携带者"),
    ]
    for raw, expected in variants:
        res = redact_medical_text(raw)
        assert res == expected, f"变体 '{raw}' 脱敏输出 '{res}' 不匹配预期 '{expected}'"


def test_four_pillar_16_probes_coverage() -> None:
    """验证四柱强剥离探针（单药/抗精神病/肝硬化体征/CD4计数/肿瘤治疗手段等）全覆盖。"""
    probes = [
        "替诺福韦 300mg", "拉米夫定 150mg", "喹硫平 200mg", "阿立哌唑 10mg",
        "蜘蛛痣", "肝掌", "肝硬化腹水", "门静脉高压", "食管静脉曲张", "脾大", "脾功能亢进",
        "CD4 细胞", "CD4+ T细胞 180/μL", "硬下疳", "菜花状", "HTT基因",
        # 以下为审计补充的四柱探针（曾原样泄露）
        "氨磺必利", "利培酮", "氯氮平", "齐多夫定", "依非韦伦",
        "苄星青霉素", "干扰素", "病毒载量", "化疗", "放疗", "食管胃底静脉曲张", "门脉高压",
    ]
    for probe in probes:
        res = redact_medical_text(f"患者有 {probe} 表现。")
        assert probe not in res, f"四柱探针 '{probe}' 泄露在脱敏结果中: {res}"


def test_all_8_specification_cases_exact_match() -> None:
    """验证规范中的 8 个标准 Case 输出一致性。"""
    cases = [
        ("Case 1", "患者1月前发现外阴及会阴部多发菜花状赘生物，醋酸白试验阳性。行CO2激光灼除术及咪喹莫特乳膏外用。胸片详见 xray.png。", "胸片详见 xray.png。"),
        ("Case 2", "患者2周前体检发现ALT 120U/L，AST 95U/L，HBsAg阳性。进一步检查发现'慢性乙型病毒性肝炎'，HBV-DNA 2.3×10^5 IU/mL。建议尽早启动恩替卡韦抗病毒治疗。", "患者2周前体检发现ALT 120U/L，AST 95U/L。"),
        ("Case 3", "父亲死于'胃癌'(62岁)，母亲患'乳腺癌'(55岁确诊)。家族中有明显消化道肿瘤聚集倾向。", "父亲死于62岁，母亲患病(54岁确诊)。家族中有明显消化道疾病聚集倾向。"),
        ("Case 4", "患者查出'HIV抗体阳性'，CD4+ T细胞180/μL。行'替诺福韦+拉米夫定'抗逆转录治疗。同住者无感染。", "同住者无感染。"),
        ("Case 5", "曾就诊于精神卫生中心，诊断为'重度精神分裂症'。存在命令性幻听及保护性约束倾向。", ""),
        ("Case 6", "基因检测提示'遗传性亨廷顿舞蹈病'(HTT基因CAG重复46次)。四肢舞蹈样动作明显。", ""),
        ("Case 7", "原发性高血压病史10年，口服硝苯地平控释片30mg qd，血压控制良好。", "原发性高血压病史10年，口服硝苯地平控释片30mg qd，血压控制良好。"),
        ("Case 8", "一弟因'恶性肿瘤'去世，一妹患'重度精神分裂症'、'2型糖尿病'。", "一弟因病去世，一妹患'2型糖尿病'。"),
    ]
    for case_id, raw, expected in cases:
        res = redact_medical_text(raw)
        assert res == expected, f"{case_id} 脱敏结果 '{res}' 不符合预期 '{expected}'"


def test_ner_path_case4_cd4_and_art_residue_purged() -> None:
    """回归：NER 引擎路径下 CD4 计数与双联 ART 药物残渣必须被规则镜像句法彻底清除。

    修复前实测：Case 4 经 NER 引擎输出 '180/μL。行+。同住者无感染。'，
    HIV 可由 CD4 计数 + 双联药物反推（四柱覆盖缺口）。
    """
    case4 = "患者查出'HIV抗体阳性'，CD4+ T细胞180/μL。行'替诺福韦+拉米夫定'抗逆转录治疗。同住者无感染。"
    res = process_medical_dataset([{"present_illness": case4}])
    out = res.sanitized_data[0]["present_illness"]
    assert "180" not in out and "CD4" not in out
    assert "替诺福韦" not in out and "拉米夫定" not in out
    assert "同住者无感染。" in out
    assert res.summary["guarantee_no_l4_l5_raw_data"] is True


def test_summary_stats_are_measured_not_hardcoded() -> None:
    """summary 合规统计必须为实测值：实际 PII 字段计数 + fail-safe 门禁触发数 + 输出回扫验证。"""
    records = [{"name": "张伟", "id_card_no": "110101199003072381", "past_history": "确诊艾滋病"}]
    res = process_medical_dataset(records)
    assert res.summary["sanitized_pii_fields_total"] == 2
    assert res.summary["sanitized_pii_fields_per_record"] == 2
    assert "fail_safe_triggered_fields" in res.summary
    assert res.summary["guarantee_no_l4_l5_raw_data"] is True


def test_ascii_term_word_boundary_no_false_positive() -> None:
    """ASCII 词项必须带词边界：良性英文/编码文本不得因子串误命中而被擦除。

    回归：修复前 "archive" 被抠成 "arce"（含 hiv）、"http://" 被抠成 "seep://"（含 htt）、
    "ABCD4" 被抠成 "AB"（含 CD4），叠加最终门禁会导致良性字段被整值抹除。
    """
    benign_cases = [
        "archive the report please",       # arcHIVe
        "see http://example.com/result",   # HTTp
        "lab code ABCD4 pending",          # abCD4
        "HRPR positive control",           # hRPR
        "SHCV protocol v2",                # sHCV
        "chopper machine",
        "campus activity",
    ]
    for raw in benign_cases:
        assert redact_medical_text(raw) == raw, f"良性文本被误改: {raw!r} -> {redact_medical_text(raw)!r}"

    # 真敏感词在词边界位置仍必须命中
    assert "HIV" not in redact_medical_text("HIV test ordered")
    assert "tumor" not in redact_medical_text("the tumor board meets weekly")
    assert "CD4" not in redact_medical_text("CD4计数180个/μL")


def test_syntax_self_healing_rules() -> None:
    """测试规则引擎的全套语法自愈能力：
    1. 消除冒号与逗号碰撞 (初步诊断：， -> 初步诊断：)
    2. 消除多药联合处方擦除后的运算符与空括号残渣 (( + + ) -> "")
    3. 消除孤立分期/试验括号 ((HIV 感染期) -> "")
    4. 中文姓名掩码 (张三 -> 张*)
    """
    # Case 1: 标点碰撞与孤立分期括号自愈
    case1 = "初步诊断：艾滋病（HIV 感染期），伴卡氏肺孢子虫肺炎。"
    out1 = redact_medical_text(case1)
    assert "：，" not in out1, f"标点碰撞未自愈: {out1}"
    assert "（感染期）" not in out1, f"孤立分期括号未自愈: {out1}"
    assert "HIV" not in out1 and "艾滋病" not in out1
    assert "初步诊断：伴卡氏肺孢子虫肺炎。" in out1

    # Case 2: 用药处方、多重空格与运算符自愈
    case2 = "医嘱：立即转诊至定点传染病医院，开展 HAART 抗病毒治疗（替诺福韦 + 拉米夫定 + 依非韦伦）。"
    out2 = redact_medical_text(case2)
    assert "+" not in out2 and "(" not in out2 and "（" not in out2, f"运算符/空括号未清理: {out2}"
    assert "  " not in out2, f"多重空格未消除: {out2}"
    assert "替诺福韦" not in out2 and "拉米夫定" not in out2 and "依非韦伦" not in out2
    assert "医嘱：遵医嘱常规治疗与健康管理。" in out2

    # Case 3: 包含高敏词汇与 PII 时，触发全流程语法自愈与姓名遮蔽打码
    case3 = "患者姓名：张三，确诊为艾滋病，身份证号 510101199001011234。"
    out3 = redact_medical_text(case3)
    assert "张三" not in out3, f"姓名未掩码: {out3}"
    assert "张*" in out3, f"姓名未正确格式化遮蔽: {out3}"


def test_ner_fallback_preserves_clean_text() -> None:
    """NER 未检出高敏实体时的规则降级路径，对干净文本必须零篡改。

    回归：修复前 fallback 在 redact 结果上再套一层语法自愈清理，
    导致干净文本被误删词（"患者出现皮疹3天，伴瘙痒。" -> "患者皮疹3天。"）。
    """
    clean_texts = [
        "患者出现皮疹3天，伴瘙痒。",
        "患者2年前  曾行阑尾切除术。",
        "进一步检查发现右肺结节，建议随访。",
        "原发性高血压病史10年，口服硝苯地平控释片30mg qd，血压控制良好。",
    ]
    for text in clean_texts:
        assert redact_medical_text_with_ner(text) == text, f"干净文本被误改: {text!r}"


def test_pinyin_homophone_variants_caught() -> None:
    """拼音/同音/形近/字符替换变体必须命中词库（系统性覆盖，非仅审计样例）。"""
    cases = [
        ("患者得了aizibing", "aizibing"),
        ("肺ai晚期", "肺ai"),
        ("霉毒病史", "霉毒"),
        ("确诊meidu一年", "meidu"),
        ("jingshenfenlie病史", "jingshenfenlie"),
        ("精神分lie病史", "精神分lie"),
        ("乙gan病史10年", "乙gan"),
        ("丙gan史", "丙gan"),
        ("胃ai术后", "胃ai"),
        ("乳腺ai", "乳腺ai"),
        ("结直肠ai待查", "肠ai"),
        ("xingbing门诊就诊", "xingbing"),
        ("H1V携带者", "H1V"),
        ("HlV阳性", "HlV"),
    ]
    for raw, core in cases:
        res = redact_medical_text(raw)
        assert core not in res, f"拼音/形近变体 '{core}' 泄露在脱敏结果中: {res!r}"


def test_yibao_csv_pipeline_processing() -> None:
    """验证医保结算数据集 (yibao.csv 18 字段) 的分类分级与脱敏抹平测试。

    检查内容：
    1. 成功读取并解析 yibao.csv 50 条记录；
    2. MedicalPrivacyPipeline 动态适配 18 字段模式，双结构输出结构完整；
    3. 诊断名称与 ICD-10 中的 L4/L5 敏感病史 100% 抹平打码；
    4. PII 医疗标识（person_id, insurance_settlement_id）被强有效掩码/脱敏；
    5. 三级安全门禁回扫 100% 零高敏词泄漏。
    """
    import csv
    from pathlib import Path
    from engine.medical_pipeline.pipeline import MedicalPrivacyPipeline
    from engine.medical_pipeline.rules import classify_icd10_code

    yibao_csv_path = Path(__file__).resolve().parent.parent / "engine" / "medical_pipeline" / "samples" / "yibao.csv"
    assert yibao_csv_path.exists(), f"医保测试数据集 yibao.csv 不存在: {yibao_csv_path}"

    with open(yibao_csv_path, "r", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))

    assert len(records) == 50, f"预期 50 条医保记录，实际读取 {len(records)} 条"

    pipeline = MedicalPrivacyPipeline()
    res = pipeline.process_records(records, sanitize=True)

    reports = res.classification_report
    sanitized_records = res.sanitized_data

    assert len(reports) == 50
    assert len(sanitized_records) == 50

    # 高敏词词库（用于 Fail-Closed 端到端回扫校验）
    high_risk_terms = [
        "HIV", "艾滋病", "艾滋", "梅毒", "尖锐湿疣", "菜花状",
        "精神分裂症", "亨廷顿", "恶性肿瘤", "腺癌", "恩替卡韦"
    ]

    for idx, (rep, san) in enumerate(zip(reports, sanitized_records)):
        # 1. 验证 18 个字段齐全
        assert len(san) == 18, f"第 {idx+1} 条脱敏记录字段数量不符合 18 字段模式: {len(san)}"

        # 2. 检查 PII 脱敏 (person_id, insurance_settlement_id)
        assert "*" in san["person_id"], f"person_id 未脱敏掩码: {san['person_id']}"
        assert "*" in san["insurance_settlement_id"], f"insurance_settlement_id 未脱敏掩码: {san['insurance_settlement_id']}"

        # 3. 检查诊断名称 (diagnosis_name) 高敏词抹平
        diag_name = san["diagnosis_name"]
        for term in high_risk_terms:
            assert term not in diag_name, f"第 {idx+1} 条脱敏记录泄露高敏词 '{term}': {diag_name!r}"

        # 4. 门禁验证：_contains_high_risk_text 必须返回 False
        assert not pipeline._contains_high_risk_text(diag_name), f"门禁校验失败，第 {idx+1} 条诊断仍包含高风险特征: {diag_name!r}"

        # 5. ICD-10 高危编码不得原样泄露（§9 规约：诊断名抹平后编码仍可反推病种）
        raw_icd = rep["raw_record"]["icd10_code"]
        icd_result = classify_icd10_code(raw_icd)
        if icd_result is not None:
            assert san["icd10_code"] != raw_icd, f"第 {idx+1} 条高危 ICD-10 编码原样泄露: {raw_icd}"
            # L4 与 L5 均整值抹平（与 Go RedactICD10Code 的无痕口径对齐）
            assert san["icd10_code"] == "", (
                f"第 {idx+1} 条 {icd_result[0]} 编码未整值抹平: {san['icd10_code']!r}"
            )
        else:
            assert san["icd10_code"] == raw_icd, f"第 {idx+1} 条良性 ICD-10 编码被误改: {raw_icd} -> {san['icd10_code']}"

        # 6. 枚举型科室字段零篡改（"皮肤性病科" 含"性病"子串，不得误伤）
        assert san["admission_dept"] == rep["raw_record"]["admission_dept"]
        assert san["discharge_dept"] == rep["raw_record"]["discharge_dept"]

        # 7. 日期准标识符截断为年月（§9 规约 L2 泛化）
        for date_key in ("birth_date", "admission_date", "discharge_date"):
            assert len(san[date_key]) == 7 and san[date_key][4] == "-", (
                f"第 {idx+1} 条 {date_key} 未截断为年月: {san[date_key]!r}"
            )

        # 8. person_id 按 L3 人员标识定级（不得误映射为身份证 L4 抬高全量记录等级）
        pid_detail = next(f for f in rep["field_details"] if f["field_name"] == "person_id")
        assert pid_detail["level"] == "L3", f"person_id 定级错误: {pid_detail['level']}"
        assert san["person_id"].startswith("PID****"), f"person_id 掩码格式错误: {san['person_id']}"


def test_icd10_code_classification_and_redaction() -> None:
    """ICD-10 高危编码段定级与脱敏单元测试（rules.classify_icd10_code / redact_icd10_code）。"""
    from engine.medical_pipeline.rules import (
        classify_icd10_code,
        redact_icd10_code,
    )

    # L5 极高敏：HIV / 精神分裂症 / 亨廷顿舞蹈病 → 整值抹平
    l5_categories = {
        "B20.900": "MEDICAL_ICD10_HIV",
        "B24.x00": "MEDICAL_ICD10_HIV",
        "F20.900": "MEDICAL_ICD10_PSYCHIATRIC",
        "F25.100": "MEDICAL_ICD10_PSYCHIATRIC",
        "G10.x00": "MEDICAL_ICD10_GENERAL",
        "G10": "MEDICAL_ICD10_GENERAL",
    }
    for code, category in l5_categories.items():
        classified = classify_icd10_code(code)
        assert classified == ("L5", category), f"{code} 定级错误: {classified!r}"
        assert redact_icd10_code(code) == "", f"{code} 应整值抹平"

    # L4 高敏：性病 / 肿瘤 / 病毒性肝炎 / 急性心梗 / 肾衰 / 慢阻肺 → 整值抹平（无痕）
    l4_categories = {
        "A51.000": "MEDICAL_ICD10_STD",
        "A63.000": "MEDICAL_ICD10_STD",
        "C34.900": "MEDICAL_ICD10_CANCER",
        "D05.100": "MEDICAL_ICD10_CANCER",
        "B18.100": "MEDICAL_ICD10_GENERAL",
        "I21.900": "MEDICAL_ICD10_GENERAL",
        "N18.500": "MEDICAL_ICD10_GENERAL",
        "J44.100": "MEDICAL_ICD10_GENERAL",
    }
    for code, category in l4_categories.items():
        classified = classify_icd10_code(code)
        assert classified == ("L4", category), f"{code} 定级错误: {classified!r}"
        assert redact_icd10_code(code) == "", f"{code} 应整值抹平"

    # 良性编码与非法输入：原样保留
    for code in ("I10.x00", "E11.900", "J18.900", "K35.800", "", "abc", "12345"):
        assert classify_icd10_code(code) is None, f"{code} 不应判定为高危"
        assert redact_icd10_code(code) == code, f"{code} 应原样保留"

    # 抹平结果为空串，天然不触发高敏词门禁（防二次命中整值删除）
    from engine.medical_pipeline.rules import contains_high_risk_text
    for code in l4_categories:
        assert not contains_high_risk_text(redact_icd10_code(code))


def test_categorical_department_field_no_false_positive() -> None:
    """枚举型科室字段豁免自由文本高敏扫描：'皮肤性病科' 不得被定级 L4 或篡改。"""
    from engine.medical_pipeline.pipeline import MedicalPrivacyPipeline

    pipeline = MedicalPrivacyPipeline()
    res = pipeline.process_records(
        [{"admission_dept": "皮肤性病科", "discharge_dept": "皮肤性病科", "gender": "女"}],
        sanitize=True,
    )
    san = res.sanitized_data[0]
    assert san["admission_dept"] == "皮肤性病科"
    assert san["discharge_dept"] == "皮肤性病科"
    levels = {f["field_name"]: f["level"] for f in res.classification_report[0]["field_details"]}
    assert levels["admission_dept"] == "L1", f"科室字段被误定级: {levels['admission_dept']}"


def test_yibao_person_id_and_hospital_code_masking() -> None:
    """person_id / hospital_code 独立 PII 规则：定级 L3，格式掩码符合 §9 规约。"""
    from engine.medical_pipeline.pipeline import MedicalPrivacyPipeline

    pipeline = MedicalPrivacyPipeline()
    res = pipeline.process_records(
        [{"person_id": "PID66453983", "hospital_code": "H4201020015"}],
        sanitize=True,
    )
    san = res.sanitized_data[0]
    assert san["person_id"] == "PID****3983"
    assert san["hospital_code"] == "H4201****"
    levels = {f["field_name"]: f["level"] for f in res.classification_report[0]["field_details"]}
    assert levels["person_id"] == "L3"
    assert levels["hospital_code"] == "L3"


def test_chinese_and_combined_field_names_support() -> None:
    """测试 Pipeline 支持纯中文 Key（如 '身份证号'）、纯英文 Key 和中英组合 Key（如 'id_card_no (身份证号)'）无缝识别与治理。"""
    pipeline = MedicalPrivacyPipeline()

    chinese_record = {
        "姓名": "张三",
        "身份证号": "110101199003072345",
        "主诉": "确诊HIV感染3年，伴咳嗽",
        "诊断名称": "获得性免疫缺陷综合征(HIV)"
    }
    res = pipeline.process_records([chinese_record], sanitize=True)
    san = res.sanitized_data[0]
    assert "*" in san["身份证号"]
    assert "HIV" not in san["诊断名称"]
    assert "艾滋" not in san["诊断名称"]

    combined_record = {
        "id_card_no (身份证号)": "110101199003072345",
        "diagnosis_name (诊断名称)": "梅毒硬下疳",
    }
    res_comb = pipeline.process_records([combined_record], sanitize=True)
    san_comb = res_comb.sanitized_data[0]
    assert "*" in san_comb["id_card_no (身份证号)"]
    assert "梅毒" not in san_comb["diagnosis_name (诊断名称)"]


# === YAML 可配置脱敏策略测试 ===


def test_load_redaction_strategy_from_yaml() -> None:
    """验证从 YAML 加载脱敏策略配置正确解析 purge/generalization/replacement 三个节。"""
    from engine.medical_pipeline.rules import load_redaction_strategy

    config = load_redaction_strategy()
    # YAML 中应定义了 purge_categories
    assert "HIV_AIDS" in config.purge_categories
    assert "PSYCHIATRIC_DISORDER" in config.purge_categories
    assert "GENETIC_DEFECT" in config.purge_categories
    assert "STD_VENEREAL" in config.purge_categories
    # YAML 中应定义了 generalization_categories
    assert "MALIGNANT_NEOPLASM" in config.generalization_categories
    assert "HEPATITIS_VIRUS" in config.generalization_categories
    assert "SEVERE_ORGAN_DAMAGE" in config.generalization_categories
    # 替换标签映射应正确构建
    assert config.l5_replacement_map["HIV_AIDS"] == "IMMUNODEFICIENCY"
    assert config.l4_replacement_map["STD_VENEREAL"] == "INFECTIOUS_DISEASE"


def test_load_redaction_strategy_fallback_on_missing_yaml() -> None:
    """验证 YAML 不存在时回退到代码内置默认值。"""
    from engine.medical_pipeline.rules import load_redaction_strategy

    config = load_redaction_strategy(rules_dir="/nonexistent/path", domain="nonexistent")
    # 默认值应包含所有 L5 范畴 + STD_VENEREAL
    assert "HIV_AIDS" in config.purge_categories
    assert "STD_VENEREAL" in config.purge_categories
    # 默认值应包含 L4 中非 STD 的范畴
    assert "MALIGNANT_NEOPLASM" in config.generalization_categories
    assert "HEPATITIS_VIRUS" in config.generalization_categories


def test_pipeline_with_custom_redaction_strategy() -> None:
    """验证 Pipeline 接受自定义 RedactionStrategyConfig 并正确编译替换标签。"""
    from engine.medical_pipeline.rules import RedactionStrategyConfig

    custom_strategy = RedactionStrategyConfig(
        purge_categories=["HIV_AIDS"],
        generalization_categories=["MALIGNANT_NEOPLASM"],
        l5_replacement_map={"HIV_AIDS": "CUSTOM_L5_TAG"},
        l4_replacement_map={"MALIGNANT_NEOPLASM": "CUSTOM_L4_TAG"},
    )
    pipeline = MedicalPrivacyPipeline(redaction_strategy=custom_strategy)
    # 自定义替换标签应生效
    text = "患者HIV抗体阳性"
    sanitized = pipeline.sanitize_text(text)
    assert "CUSTOM_L5_TAG" in sanitized
    assert "HIV" not in sanitized


def test_custom_strategy_controls_redaction_generalization() -> None:
    """泛化/抹平决策必须由运行时策略控制，而非固定使用代码默认规则。"""
    from engine.medical_pipeline.rules import RedactionStrategyConfig, redact_medical_text

    default_text = redact_medical_text("母亲有恶性肿瘤家族史。")
    custom = RedactionStrategyConfig(
        purge_categories=["MALIGNANT_NEOPLASM"],
        generalization_categories=[],
    )
    custom_text = redact_medical_text("母亲有恶性肿瘤家族史。", strategy=custom)
    pipeline_text = MedicalPrivacyPipeline(
        redact_engine="rule", redaction_strategy=custom
    )._medical_text_sanitizer(
        "family_history", "母亲有恶性肿瘤家族史。", "L4"
    )

    assert "相关系统疾病" in default_text
    assert "恶性肿瘤" not in custom_text
    assert "相关系统疾病" not in custom_text
    assert "恶性肿瘤" not in pipeline_text
    assert "相关系统疾病" not in pipeline_text


def test_pipeline_default_strategy_matches_yaml() -> None:
    """验证默认 Pipeline（无显式策略）从 YAML 加载的策略与手动加载一致。"""
    from engine.medical_pipeline.rules import load_redaction_strategy

    pipeline = MedicalPrivacyPipeline()
    yaml_config = load_redaction_strategy()
    assert pipeline.redaction_strategy.purge_categories == yaml_config.purge_categories
    assert pipeline.redaction_strategy.generalization_categories == yaml_config.generalization_categories
    assert pipeline.redaction_strategy.l5_replacement_map == yaml_config.l5_replacement_map


def test_contains_high_risk_text_module_level_function() -> None:
    """验证模块级 contains_high_risk_text 函数与 Pipeline 实例方法行为一致。"""
    from engine.medical_pipeline.rules import contains_high_risk_text

    pipeline = MedicalPrivacyPipeline()
    test_cases = [
        "患者HIV抗体阳性",
        "血压正帘",
        "确诊恶性肿瘤",
        "皮肤性病科",
    ]
    for text in test_cases:
        assert contains_high_risk_text(text) == pipeline._contains_high_risk_text(text), (
            f"模块级函数与实例方法结果不一致: {text!r}"
        )


def test_contains_high_risk_text_accepts_custom_patterns() -> None:
    """验证 contains_high_risk_text 支持自定义 patterns 参数检测自定义替换标签。"""
    from engine.medical_pipeline.rules import (
        RedactionStrategyConfig,
        compile_l4_l5_patterns,
        contains_high_risk_text,
    )

    # 构建自定义策略，使用非默认替换标签
    custom = RedactionStrategyConfig(
        purge_categories=["HIV_AIDS"],
        l5_replacement_map={"HIV_AIDS": "CUSTOM_IMMUNO"},
    )
    custom_l5, custom_l4 = compile_l4_l5_patterns(
        l5_replacement_map=custom.l5_replacement_map,
    )
    custom_patterns = custom_l5 + custom_l4

    # 含自定义替换标签（标准格式）的文本应被检出
    # 注意：_MASKED_LABEL_RE 会匹配任何 [L4|L5-...-SENSITIVE-MASKED] 格式标签，
    # 因此无论 patterns 是否自定义，标准格式标签都会被检出（安全门禁的保守策略）。
    text_with_custom_label = "患者[L5-CUSTOM_IMMUNO-SENSITIVE-MASKED]抗体阳性"
    assert contains_high_risk_text(text_with_custom_label, patterns=custom_patterns)
    # 默认 patterns 也会检出标准格式标签（_MASKED_LABEL_RE 不区分标签内容）
    assert contains_high_risk_text(text_with_custom_label)

    # 非标准格式的裸标签不应被 _MASKED_LABEL_RE 匹配，
    # 默认 patterns 不含 CUSTOM_IMMUNO 相关模式，故不应检出
    text_with_bare_label = "患者[CUSTOM_IMMUNO]抗体阳性"
    assert not contains_high_risk_text(text_with_bare_label)

    # 原始敏感词仍应被检出（无论是否自定义 patterns）
    assert contains_high_risk_text("患者HIV抗体阳性", patterns=custom_patterns)
    assert contains_high_risk_text("患者HIV抗体阳性")


def test_pipeline_safety_check_detects_custom_labels() -> None:
    """验证 Pipeline 的 _contains_high_risk_text 能检测自定义替换标签。"""
    from engine.medical_pipeline.rules import RedactionStrategyConfig

    custom_strategy = RedactionStrategyConfig(
        purge_categories=["HIV_AIDS"],
        l5_replacement_map={"HIV_AIDS": "CUSTOM_TAG"},
    )
    pipeline = MedicalPrivacyPipeline(redaction_strategy=custom_strategy)

    # 含自定义替换标签的文本应被 Pipeline 实例方法检出
    assert pipeline._contains_high_risk_text("[L5-CUSTOM_TAG-SENSITIVE-MASKED]")
    # 原始敏感词也应被检出
    assert pipeline._contains_high_risk_text("HIV抗体阳性")
    # 干净文本不应误判
    assert not pipeline._contains_high_risk_text("血压控制良好")


def test_default_generalization_allowed_is_precomputed() -> None:
    """验证 _DEFAULT_GENERALIZATION_ALLOWED 预计算集合与规则定义一致。"""
    from engine.medical_pipeline.rules import (
        _CATEGORY_GENERALIZATION_RULES,
        _DEFAULT_GENERALIZATION_ALLOWED,
    )

    expected = {cat for cat, _pat, _repl in _CATEGORY_GENERALIZATION_RULES}
    assert _DEFAULT_GENERALIZATION_ALLOWED == expected
    assert "MALIGNANT_NEOPLASM" in _DEFAULT_GENERALIZATION_ALLOWED
    assert "HEPATITIS_VIRUS" in _DEFAULT_GENERALIZATION_ALLOWED
    assert "GENETIC_DEFECT" in _DEFAULT_GENERALIZATION_ALLOWED
    assert "SEVERE_ORGAN_DAMAGE" in _DEFAULT_GENERALIZATION_ALLOWED



# ── 双引擎规则快照：批次去重与可选开关（性能优化回归） ────────────────────


def _kangyang_records(n: int = 5) -> list[dict[str, str]]:
    """读取康养样例数据前 n 条，作为贴近真实字段分布的测试输入。"""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "kangyang.csv"
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [{k: (v or "") for k, v in row.items()} for row in rows[:n]]


def test_rule_snapshot_matches_independent_redaction() -> None:
    """批次去重后，规则快照必须与逐字段独立调用 redact_medical_text 的结果完全一致。"""
    from engine.medical_pipeline.rules import PII_FIELD_RULES, canonicalize_pii_field

    records = _kangyang_records()
    pipeline = MedicalPrivacyPipeline()
    result = pipeline.process_records(records)

    checked = 0
    for rep in result.classification_report:
        for fd in rep["field_details"]:
            if canonicalize_pii_field(fd["field_name"]) in PII_FIELD_RULES:
                continue  # PII 字段走强掩码分支，不经过文本抹平
            expected = redact_medical_text(fd["raw_value"], strategy=pipeline.redaction_strategy)
            assert fd["sanitized_value_rule"] == expected, fd["field_name"]
            if pipeline.redact_engine == "rule":
                assert fd["sanitized_value_ner"] == expected, fd["field_name"]
            checked += 1
    assert checked > 0


def test_batch_dedups_rule_redaction_calls(monkeypatch) -> None:
    """同一批次内每个唯一文本只应触发一次底层规则抹平（交付与快照共享结果）。"""
    import engine.medical_pipeline.pipeline as pipeline_module
    import engine.medical_pipeline.rules as rules_module

    real = rules_module.redact_medical_text
    calls: list[str] = []

    def counting(text, strategy=None):
        calls.append(text)
        return real(text, strategy=strategy)

    monkeypatch.setattr(rules_module, "redact_medical_text", counting)
    monkeypatch.setattr(pipeline_module, "redact_medical_text", counting)

    row = _kangyang_records(1)[0]
    pipeline = MedicalPrivacyPipeline()
    pipeline.process_records([row])
    assert calls, "样例数据未触发任何规则抹平，测试前提失效"
    assert len(calls) == len(set(calls)), "同一文本在同一批次内被重复抹平"

    calls.clear()
    pipeline.clear_cache()
    pipeline.process_records([dict(row) for _ in range(5)])
    assert len(calls) == len(set(calls)), "5 条重复记录触发了重复抹平"


