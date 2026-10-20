"""医疗分类分级规则与 L4/L5 级脱敏引擎模块 / Medical Privacy Rules & Redaction Engine.

采用 【🥇 动态字典 + 句法正则表达式 (Dynamic Dictionary & Regex Engine)】 核心架构：
1. **动态字典 (Dynamic Dictionaries)**：分层分类维护 PII 别名字典与 L4/L5 重大高敏词库（涵盖 HIV、精神障碍、遗传缺陷、性病、恶性肿瘤、病毒性肝炎、重度器官损害等）；
2. **Fast-Path 前置校验**：词库自动编译为长词优先正则，针对干净文本实现 <1ms 超低延迟原样放行，零篡改零误伤；
3. **句法正则表达式 (Clause Grammar Patterns)**：高精度匹配服药剂量频次、血清学滴度、基因检测突变、死因重构、就诊机构及列表顿号；
4. **语法自愈流水线 (_clean_orphan_syntax)**：自动消除断句残渣、悬空连词/介词/标点，对仅剩无主语状语从句执行 Purge to Empty 干净抹平。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..observability.logging_config import get_logger

logger = get_logger(__name__)

# PII 身份隐私字段及其默认脱敏规则定义
PII_FIELD_RULES: dict[str, str] = {
    "name": "CHINESE_NAME",
    "id_card_no": "ID_CARD",
    "registered_address": "ADDRESS",
    "disability_cert_no": "DISABILITY_CERT",
    "medical_insurance_no": "INSURANCE_NO",
    "person_id": "PERSON_ID",
    "hospital_code": "HOSPITAL_CODE",
}

# 中文数据源常用字段名到规范字段名的映射。保留规范英文键作为唯一规则来源，
# 避免把别名数量误计入 PII 类型统计，同时让分类和脱敏使用完全一致的语义。
PII_FIELD_ALIASES: dict[str, str] = {
    "姓名": "name",
    "真实姓名": "name",
    "用户姓名": "name",
    "patient_name": "name",
    "user_name": "name",
    "real_name": "name",
    "身份证": "id_card_no",
    "身份证号": "id_card_no",
    "居民身份证": "id_card_no",
    "公民身份号码": "id_card_no",
    "id_card": "id_card_no",
    "idcard": "id_card_no",
    "id_card_num": "id_card_no",
    "id_number": "id_card_no",
    "id_no": "id_card_no",
    "identity_card": "id_card_no",
    "identity_no": "id_card_no",
    "sfz": "id_card_no",
    "sfz_no": "id_card_no",
    "地址": "registered_address",
    "注册地址": "registered_address",
    "登记地址": "registered_address",
    "户籍地址": "registered_address",
    "居住地址": "registered_address",
    "居民住址": "registered_address",
    "家庭住址": "registered_address",
    "联系地址": "registered_address",
    "address": "registered_address",
    "home_address": "registered_address",
    "contact_address": "registered_address",
    "user_address": "registered_address",
    "resident_address": "registered_address",
    "location": "registered_address",
    "残疾证号": "disability_cert_no",
    "残疾人证号": "disability_cert_no",
    "disability_cert": "disability_cert_no",
    "disability_card": "disability_cert_no",
    "医保卡号": "medical_insurance_no",
    "医保号": "medical_insurance_no",
    "医疗保险号": "medical_insurance_no",
    "insurance_no": "medical_insurance_no",
    "med_insurance_no": "medical_insurance_no",
    "医保结算流水号": "medical_insurance_no",
    "insurance_settlement_id": "medical_insurance_no",
    "人员唯一标识": "person_id",
    "person_id": "person_id",
    "pid": "person_id",
    "定点医疗机构编码": "hospital_code",
    "hospital_code": "hospital_code",
    "明细结算流水号": "medical_insurance_no",
    "settlement_seq_no": "medical_insurance_no",
    # 临床与诊断字段别名映射
    "主诉": "chief_complaint",
    "现病史": "present_illness",
    "既往史": "past_history",
    "个人史": "personal_history",
    "家族史": "family_history",
    "过敏史": "allergic_history",
    "诊断名称": "diagnosis_name",
    "病程记录": "progress_note",
    "诊断编码": "icd10_code",
    "诊断编码(ICD-10)": "icd10_code",
    "诊断编码（ICD-10）": "icd10_code",
    "icd-10": "icd10_code",
    "icd10": "icd10_code",
    "入院病情": "admission_condition",
}


def canonicalize_pii_field(field_name: str) -> str:
    """将中文、英文或中英组合 (如 id_card_no (身份证号)) 字段名转换为规范字段名。"""
    if not field_name:
        return field_name
    cleaned = field_name.strip()

    # 直接匹配字典
    if cleaned in PII_FIELD_ALIASES:
        return PII_FIELD_ALIASES[cleaned]
    if cleaned.lower() in PII_FIELD_ALIASES:
        return PII_FIELD_ALIASES[cleaned.lower()]

    # 若包含括号如 "id_card_no (身份证号)" 或 "身份证号 (id_card_no)"，提取括号内外部分尝试匹配
    if "(" in cleaned or "（" in cleaned:
        import re
        parts = re.split(r"[（\(\）\)]+", cleaned)
        for part in parts:
            p = part.strip()
            if not p:
                continue
            if p in PII_FIELD_ALIASES:
                return PII_FIELD_ALIASES[p]
            if p.lower() in PII_FIELD_ALIASES:
                return PII_FIELD_ALIASES[p.lower()]

    return cleaned

# L5 极高风险病史与诊断词汇映射组（包含疾病名、缩写、临床特征及变体）
L5_TERMS_MAP: dict[str, list[str]] = {
    "HIV_AIDS": [
        "获得性免疫缺陷综合征", "获得性免疫缺陷", "人免疫缺陷病毒", "HIV感染", "HIV抗体阳性", "HIV抗体", "血清HIV-1", "HIV-1", "HIV-2", "HIV", "AIDS", "艾滋病", "艾滋",
        "ＨＩＶ", "ＡＩＤＳ", "aizibing", "aizi", "H1V", "HlV", "CD4+ T淋巴细胞", "CD4+ T细胞", "CD4+T细胞", "CD4细胞", "CD4+ T", "CD4+T", "CD4计数", "CD4/CD8", "CD4",
        "替诺福韦+拉米夫定+多替拉韦", "替诺福韦+拉米夫定", "替诺福韦", "拉米夫定", "多替拉韦", "依非韦伦", "阿巴卡韦", "恩曲他滨", "齐多夫定",
        "HAART抗逆转录治疗", "HAART抗病毒治疗", "HAART方案", "HAART", "抗逆转录治疗", "抗逆转录", "ART抗逆转录", "HIV病毒载量", "病毒载量"
    ],
    "PSYCHIATRIC_DISORDER": [
        "重度精神分裂症", "精神分裂症", "精神分裂", "jingshenfenlie", "精神分lie", "双相情感障碍", "言语关联妄想", "关联妄想", "命令性幻听", "保护性约束倾向", "幻听（命令性言语）",
        "命令性言语", "被害妄想", "幻听", "幻觉", "偏执", "自伤倾向", "冲动砸物", "保护性约束", "奥氮平片", "奥氮平", "富马酸喹硫平", "富马酸奎硫平",
        "喹硫平", "奎硫平", "阿立哌唑", "利培酮", "氯氮平", "氨磺必利", "舒必利", "奋乃静", "氟哌啶醇", "哈泊度醇", "丙戊酸钠", "碳酸锂",
        "精神卫生中心", "schizophrenia"
    ],
    "GENETIC_DEFECT": [
        "遗传性亨廷顿舞蹈病", "亨廷顿舞蹈病", "亨廷顿病", "Huntington Disease", "HTT基因CAG重复序列", "HTT基因", "HTT", "CAG重复序列",
        "CAG重复", "CAG扩增", "四苯嗪", "舞蹈样动作", "舞蹈样症状", "四肢舞蹈样动作", "舞蹈病", "Huntington"
    ],
}

# L4 高风险病史与诊断词汇映射组（肿瘤、性病/传染病、严重器官损害及变体）
L4_TERMS_MAP: dict[str, list[str]] = {
    "STD_VENEREAL": [
        "早期隐性梅毒", "隐性梅毒", "早期梅毒", "晚期梅毒", "神经梅毒", "心血管梅毒", "先天梅毒", "胎传梅毒", "梅毒", "霉毒", "meidu", "苍白密螺旋体",
        "TPPA阳性", "TPPA", "RPR阳性", "RPR 1:16", "RPR", "syphilis", "gonorrhea", "herpes", "chancroid",
        "淋病", "淋球菌", "尖锐湿疣", "生殖器疱疹", "软下疳", "性病", "性传播疾病", "xingbing", "linbing", "不洁性接触史", "不洁接触史", "无痛性溃疡", "硬下疳",
        "人乳头瘤病毒高危型", "外阴多发赘生物伴瘙痒", "外阴多发赘生物", "会阴部多发菜花状赘生物", "会阴部多发赘生物", "肛周多发菜花状赘生物",
        "肛周多发赘生物", "外阴菜花状赘生物", "外阴赘生物", "会阴部赘生物", "肛周赘生物", "多发赘生物", "菜花状赘生物", "鸡冠状赘生物",
        "乳头状赘生物", "生殖器赘生物", "赘生物伴瘙痒", "赘生物", "菜花状", "鸡冠状", "caihuazhuang", "jiguanzhuang", "醋酸白试验阳性", "醋酸白试验", "HPV 6/11低危型阳性", "HPV 6/11低危型",
        "HPV 6/11", "HPV 16/18", "HPV高危型", "HPV低危型", "HPV", "CO2激光灼除术", "CO2激光灼除", "激光灼除术", "咪喹莫特乳膏", "咪喹莫特", "二氧化碳激光",
        "苄星青霉素"
    ],
    "MALIGNANT_NEOPLASM": [
        "恶性肿瘤", "浸润性腺癌", "肺腺癌", "胃癌", "肝癌", "乳腺癌", "宫颈癌", "癌症", "腺癌", "导管癌", "鳞状细胞癌", "鳞癌", "肉瘤", "肺ai", "肝ai", "胃ai", "feiai", "ganai", "weiai", "乳腺ai", "肠ai", "直肠ai", "结肠ai", "食道ai", "食管ai", "胰ai", "胰腺ai", "宫颈ai", "卵巢ai", "前列腺ai", "鼻咽ai", "淋巴ai", "骨ai", "脑ai", "皮肤ai", "肾ai", "膀胱ai", "甲状腺ai", "消化道肿瘤", "消化道恶性肿瘤", "转移性肿瘤",
        "奥希替尼", "EGFR基因检测", "EGFR突变", "cancer", "tumor", "化疗", "放疗", "靶向治疗", "PD-1抑制剂", "PD-1"
    ],
    "HEPATITIS_VIRUS": [
        "慢性乙型病毒性肝炎", "乙型肝炎", "乙肝", "yigan", "乙gan", "丙型肝炎", "丙肝", "binggan", "丙gan", "肝硬化失代偿期", "早期肝硬化", "肝硬化代偿期", "肝硬化", "小肝癌",
        "蜘蛛痣", "肝掌", "肝硬化腹水", "门静脉高压", "门脉高压", "食管胃底静脉曲张破裂出血", "食管胃底静脉曲张", "静脉曲张破裂出血", "食管静脉曲张", "脾大", "脾肿大", "脾功能亢进",
        "HBV-DNA阳性", "HBV-DNA阴性", "HBV-DNA 5.6×10^6 IU/mL", "HBV-DNA定量", "HBV-DNA", "HBV", "HCV-RNA", "HCV",
        "恩贴卡韦", "恩替卡韦", "干扰素", "肝穿刺活检", "肝穿刺", "G3S4", "HBsAg阳性", "HBsAg", "HBeAg阳性", "HBeAg", "HBcAb阳性",
        "HBcAb", "HBsAb", "HBeAb", "乙肝表面抗原", "乙肝两对半", "hepatitis", "cirrhosis", "ＨＢＶ", "ＨＣＶ"
    ],
    "SEVERE_ORGAN_DAMAGE": [
        "慢性阻塞性肺疾病", "COPD", "急性心肌梗死", "心肌梗死", "心肌梗塞", "冠状动脉重度狭窄", "尿毒症", "肾功能衰竭"
    ],
}

# 替换标签映射：使用抽象类别代码，避免替换文本中泄露原始敏感词
# 例如不使用 [L5-HIV_AIDS-...] 而用 [L5-IMMUNODEFICIENCY-...]，防止替换后仍含 "HIV"
_L5_REPLACEMENT_MAP: dict[str, str] = {
    "HIV_AIDS": "IMMUNODEFICIENCY",
    "PSYCHIATRIC_DISORDER": "PSYCHIATRIC_DISORDER",
    "GENETIC_DEFECT": "GENETIC_DEFECT",
}

_L4_REPLACEMENT_MAP: dict[str, str] = {
    "STD_VENEREAL": "INFECTIOUS_DISEASE",
    "MALIGNANT_NEOPLASM": "MALIGNANT_NEOPLASM",
    "HEPATITIS_VIRUS": "HEPATITIS_VIRUS",
    "SEVERE_ORGAN_DAMAGE": "SEVERE_ORGAN_DAMAGE",
}


# ---------------------------------------------------------------------------
# YAML 可配置脱敏策略加载 (Redaction Strategy Loader)
# ---------------------------------------------------------------------------


@dataclass
class RedactionStrategyConfig:
    """运行时脱敏治理策略配置（从 YAML 加载或代码默认值构建）。

    Attributes:
        purge_categories: 彻底抹平范畴列表（严禁泛化）。
        generalization_categories: 范畴化泛化范畴列表。
        l5_replacement_map: L5 范畴 → 抽象替换标签映射。
        l4_replacement_map: L4 范畴 → 抽象替换标签映射。
    """

    purge_categories: list[str] = field(default_factory=list)
    generalization_categories: list[str] = field(default_factory=list)
    l5_replacement_map: dict[str, str] = field(default_factory=dict)
    l4_replacement_map: dict[str, str] = field(default_factory=dict)


def load_redaction_strategy(
    rules_dir: str | Path | None = None,
    domain: str = "medical",
) -> RedactionStrategyConfig:
    """从 YAML 领域规则包加载脱敏治理策略。

    读取 ``rules/domains/<domain>.yaml`` 中的 ``redaction_strategy`` 节，
    返回运行时策略配置。若 YAML 中未定义该节，则回退到代码内置默认值
    （与 ``_L5_REPLACEMENT_MAP`` / ``_L4_REPLACEMENT_MAP`` 保持一致）。

    Args:
        rules_dir: 规则配置根目录；为 None 时自动检测（环境变量 → 默认 ``rules``）。
        domain: 领域包名称（默认 ``"medical"``）。

    Returns:
        RedactionStrategyConfig: 运行时脱敏策略配置。
    """
    import os

    if rules_dir is None:
        rules_dir = os.environ.get("PRIVACY_DYNCLASSIFICATION_RULES_DIR", "rules")
    yaml_path = Path(rules_dir) / "domains" / f"{domain}.yaml"

    if not yaml_path.exists():
        logger.info(
            "redaction_strategy_yaml_not_found",
            extra={"path": str(yaml_path), "fallback": "hardcoded_defaults"},
        )
        return _build_default_strategy_config()

    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "redaction_strategy_yaml_parse_error",
            extra={"path": str(yaml_path), "error": str(exc), "fallback": "hardcoded_defaults"},
        )
        return _build_default_strategy_config()

    strategy_data = data.get("redaction_strategy") if isinstance(data, dict) else None
    if strategy_data is None:
        logger.info(
            "redaction_strategy_not_defined_in_yaml",
            extra={"domain": domain, "fallback": "hardcoded_defaults"},
        )
        return _build_default_strategy_config()

    # 校验：所有范畴必须存在于 L5_TERMS_MAP 或 L4_TERMS_MAP 中
    all_known_categories = set(L5_TERMS_MAP.keys()) | set(L4_TERMS_MAP.keys())
    purge_cats = strategy_data.get("purge_categories", [])
    gen_cats = strategy_data.get("generalization_categories", [])
    unknown = set(purge_cats + gen_cats) - all_known_categories
    if unknown:
        logger.warning(
            "redaction_strategy_unknown_categories",
            extra={"unknown": sorted(unknown), "known": sorted(all_known_categories)},
        )

    # 校验：purge 与 generalization 范畴不应重叠（重叠时 purge 静默胜出，易造成配置误解）
    overlap = set(purge_cats) & set(gen_cats)
    if overlap:
        logger.warning(
            "redaction_strategy_purge_generalization_overlap",
            extra={
                "overlapping_categories": sorted(overlap),
                "resolution": "purge_categories takes precedence; remove from generalization_categories if intentional",
            },
        )

    replacement_labels = strategy_data.get("replacement_labels", {})

    config = RedactionStrategyConfig(
        purge_categories=list(purge_cats),
        generalization_categories=list(gen_cats),
        l5_replacement_map={
            cat: replacement_labels.get(cat, _L5_REPLACEMENT_MAP.get(cat, cat))
            for cat in purge_cats
            if cat in L5_TERMS_MAP
        },
        l4_replacement_map={
            cat: replacement_labels.get(cat, _L4_REPLACEMENT_MAP.get(cat, cat))
            for cat in purge_cats + gen_cats
            if cat in L4_TERMS_MAP
        },
    )
    logger.info(
        "redaction_strategy_loaded_from_yaml",
        extra={
            "domain": domain,
            "purge_categories": config.purge_categories,
            "generalization_categories": config.generalization_categories,
        },
    )
    return config


def _build_default_strategy_config() -> RedactionStrategyConfig:
    """从代码内置默认值构建默认脱敏策略配置（YAML 缺失时的回退路径）。"""
    return RedactionStrategyConfig(
        purge_categories=list(L5_TERMS_MAP.keys()) + ["STD_VENEREAL"],
        generalization_categories=[
            cat for cat in L4_TERMS_MAP if cat != "STD_VENEREAL"
        ],
        l5_replacement_map=dict(_L5_REPLACEMENT_MAP),
        l4_replacement_map=dict(_L4_REPLACEMENT_MAP),
    )


def compile_l4_l5_patterns(
    l5_replacement_map: dict[str, str] | None = None,
    l4_replacement_map: dict[str, str] | None = None,
) -> tuple[list[tuple[re.Pattern, str]], list[tuple[re.Pattern, str]]]:
    """根据替换标签映射编译 L5/L4 术语正则模式列表。

    允许调用方传入自定义替换标签映射（来自 YAML 策略配置），
    若未传入则使用模块级默认映射。

    Args:
        l5_replacement_map: L5 范畴 → 抽象替换标签；为 None 时使用 ``_L5_REPLACEMENT_MAP``。
        l4_replacement_map: L4 范畴 → 抽象替换标签；为 None 时使用 ``_L4_REPLACEMENT_MAP``。

    Returns:
        (L5_PATTERNS, L4_PATTERNS) 元组，每个元素为 ``(compiled_regex, replacement_label)`` 列表。
    """
    l5_map = l5_replacement_map if l5_replacement_map is not None else _L5_REPLACEMENT_MAP
    l4_map = l4_replacement_map if l4_replacement_map is not None else _L4_REPLACEMENT_MAP

    l5_patterns = [
        (
            re.compile(
                "|".join([_flex_escape(t) for t in sorted(terms, key=len, reverse=True)]),
                re.IGNORECASE,
            ),
            f"[L5-{l5_map.get(cat, cat)}-SENSITIVE-MASKED]",
        )
        for cat, terms in L5_TERMS_MAP.items()
    ]
    l4_patterns = [
        (
            re.compile(
                "|".join([_flex_escape(t) for t in sorted(terms, key=len, reverse=True)]),
                re.IGNORECASE,
            ),
            f"[L4-{l4_map.get(cat, cat)}-SENSITIVE-MASKED]",
        )
        for cat, terms in L4_TERMS_MAP.items()
    ]
    return l5_patterns, l4_patterns

# 词项字符间容许的有界可选分隔符（空格/点/连字符/下划线/间隔号/零宽字符）。
# 用于容忍 "H I V"、"H.I.V"、"艾-滋-病" 这类在词项字符间插入噪声的绕过变体；
# {0,1} 有界量词保证每个字符间隙只有两种选择，匹配复杂度保持线性，杜绝引入 ReDoS。
_FLEX_SEP = r"[\s.\-_·•​‌‍﻿]"


def _flex_escape(term: str) -> str:
    """将词项编译为"字符间容许可选分隔符"的正则片段（抗插入变体绕过）。

    - 普通字符：re.escape 转义后用 ``_FLEX_SEP{0,1}`` 连接，容忍字符间插入至多一个分隔符；
    - 词项自带的空白字符：编译为 ``\\s*``（容忍有无空格两种书写，如 "CD4+ T细胞" 与 "CD4+T细胞"）；
    - **ASCII 词项词边界保护**：首/尾字符为 ASCII 字母数字时，分别附加
      ``(?<![A-Za-z0-9])`` / ``(?![A-Za-z0-9])`` 零宽断言，防止 "archive" 中的 "hiv"、
      "hearing aids" 中的 "aids"、"http" 中的 "htt"、"ABCD4" 中的 "CD4" 等
      子串误命中（良性英文/编码文本被整值门禁抹除的可用性事故）；
      CJK 词项无词边界概念，保持子串匹配（中文语境天然连用）。
    """
    tokens = [r"\s*" if ch.isspace() else re.escape(ch) for ch in term]
    if not tokens:
        return ""
    body = (_FLEX_SEP + "{0,1}").join(tokens) if len(tokens) > 1 else tokens[0]
    left = r"(?<![A-Za-z0-9])" if term[0].isascii() and term[0].isalnum() else ""
    right = r"(?![A-Za-z0-9])" if term[-1].isascii() and term[-1].isalnum() else ""
    return f"{left}{body}{right}"


# 文本脱敏正则表达式：按类别编译 L5/L4 术语为正则，长词优先 + 分隔符容忍匹配
L5_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile("|".join([_flex_escape(t) for t in sorted(terms, key=len, reverse=True)]), re.IGNORECASE),
     f"[L5-{_L5_REPLACEMENT_MAP.get(cat, cat)}-SENSITIVE-MASKED]")
    for cat, terms in L5_TERMS_MAP.items()
]

L4_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile("|".join([_flex_escape(t) for t in sorted(terms, key=len, reverse=True)]), re.IGNORECASE),
     f"[L4-{_L4_REPLACEMENT_MAP.get(cat, cat)}-SENSITIVE-MASKED]")
    for cat, terms in L4_TERMS_MAP.items()
]

# ---------------------------------------------------------------------------
# 无痕抹平模式 (Redaction / Purge Mode) 句法重构与全场景擦除正则引擎
# ---------------------------------------------------------------------------
_ALL_L4_L5_TERMS = sorted(
    [term for terms in list(L5_TERMS_MAP.values()) + list(L4_TERMS_MAP.values()) for term in terms],
    key=len,
    reverse=True,
)

# 超长文本保护阈值：超过此长度时降级为简单词库擦除，
# 防止复杂句法正则（含嵌套量词）在恶意/超长输入下触发灾难性回溯（ReDoS）
_REDACT_MAX_TEXT_LENGTH = 50_000

# 降级路径与 Fast-path 专用正则：词库级检测与擦除（长词优先 + 分隔符容忍）
_TERMS_ONLY_PATTERN = re.compile(
    "|".join([_flex_escape(t) for t in _ALL_L4_L5_TERMS]),
    re.IGNORECASE,
)

# 词库首字符快速预筛：任何词项匹配必以词项首字符开头（词边界为零宽断言，不改变首字符）。
# 干净文本（如 "男"、"34"、"血压控制良好"）极少包含词库首字符——
# 先用单字符类做一次 O(n) 预筛，未命中即跳过数百分支的全量词库匹配，
# 将 Fast-Path 从 ~240ms/49KB 降回毫秒级。
_TERMS_FIRST_CHARS_PATTERN = re.compile(
    "[" + re.escape("".join(sorted({t[0] for t in _ALL_L4_L5_TERMS if t}))) + "]",
    re.IGNORECASE,
)


# 替换标签匹配正则：匹配脱敏流程产生的标准格式标签 [L4|L5-...-SENSITIVE-MASKED]
# 定义于 contains_high_risk_text 之前，供其预筛逻辑引用
_MASKED_LABEL_PATTERN = r"\[(?:L4|L5)-[A-Z_]+-SENSITIVE-MASKED\]"
_MASKED_LABEL_RE = re.compile(_MASKED_LABEL_PATTERN)

# 字符间插值噪声剥离正则（预编译）
_STRIPPED_NOISE_RE = re.compile(
    r"(?<=[a-zA-Z0-9一-龥])[\s.\-_·•\u200b\u200c\ufeff]+(?=[a-zA-Z0-9一-龥])"
)


def contains_high_risk_text(
    text: str,
    patterns: list[tuple[re.Pattern, str]] | None = None,
) -> bool:
    """模块级高敏文本检测函数。

    供外部模块（如 dynclassification/service.py）在无 Pipeline 实例时调用。

    Args:
        text: 待检测文本。
        patterns: 自定义 ``(compiled_regex, replacement_label)`` 模式列表。
            为 None 时使用模块级默认模式（``L4_PATTERNS`` + ``L5_PATTERNS``）；
            Pipeline 实例应传入 ``self._l5_patterns + self._l4_patterns``
            以检测自定义替换标签。
    """
    if not text:
        return False
    # 替换标签（如 [L5-IMMUNODEFICIENCY-SENSITIVE-MASKED]）以 '[' 开头，
    # 不在词库首字符集中——必须先于 _TERMS_FIRST_CHARS_PATTERN 预筛检查，
    # 否则仅含替换标签的文本会被预筛误判为安全而提前返回 False。
    if _MASKED_LABEL_RE.search(text):
        return True
    if not _TERMS_FIRST_CHARS_PATTERN.search(text):
        return False

    if patterns is None:
        # 极速单正则路径：单次 DFA 扫描匹配所有词库
        if _TERMS_ONLY_PATTERN.search(text):
            return True
        norm = normalize_fullwidth_alphanumeric(text)
        if norm != text and _TERMS_ONLY_PATTERN.search(norm):
            return True
        stripped = _STRIPPED_NOISE_RE.sub("", norm)
        if stripped != norm and _TERMS_ONLY_PATTERN.search(stripped):
            return True
        return False

    effective_patterns = patterns
    if any(pattern.search(text) for pattern, _replacement in effective_patterns):
        return True
    norm = normalize_fullwidth_alphanumeric(text)
    if norm != text and any(pattern.search(norm) for pattern, _replacement in effective_patterns):
        return True
    stripped = _STRIPPED_NOISE_RE.sub("", norm)
    if stripped != norm and any(pattern.search(stripped) for pattern, _replacement in effective_patterns):
        return True
    return False

_TERMS_OR = "|".join([_MASKED_LABEL_PATTERN] + [_flex_escape(t) for t in _ALL_L4_L5_TERMS])
_Q = r"['\"“‘'”’]?"
_DOSE = r"(?:\s*\d+(?:\.\d+)?\s*(?:mg|g|ml|u|ug|片|粒|支|%))?"
_FREQ = r"(?:\s*(?:qd|bid|tid|qid|qn|qw|im|iv|po))?"

# 1. 死因相关句法短语重构：“因/由于/死于 'L4/L5词' (去世|死于|离世|逝世...)” -> “因病去世”
# 支持“因'HIV'导致的并发症去世”、“由于'恶性肿瘤'不幸身亡”、“身亡于'急性心肌梗死'(40岁)”等完整句法重构
_REDACT_DEATH_ACTION = r"(?:去世|死于|离世|殁于|身亡于|病逝于|不幸身亡|宣告不治|逝世)"
_REDACT_CAUSE_DEATH_PATTERN = re.compile(
    rf"(?:不幸)?\s*(?:因|由于|死于|殁于|身亡于|病逝于|离世于|因为|由)\s*{_Q}(?:{_TERMS_OR}){_Q}\s*(?:导致的并发症|引起的并发症|破裂出血导致|破裂出血引起|破裂出血|出血导致|并发症导致|并发症|抢救无效|导致|引起)?\s*(?:{_REDACT_DEATH_ACTION})?",
    re.IGNORECASE,
)
_REDACT_DEATH_WITH_AGE_PATTERN = re.compile(
    rf"(?:不幸)?\s*(身亡于|病逝于|死于|殁于|离世于|去世于|因|由于)\s*{_Q}(?:{_TERMS_OR}){_Q}\s*(?:{_REDACT_DEATH_ACTION})?\s*[\(（](\d+)\s*岁[\)）]",
    re.IGNORECASE,
)
_REDACT_SUFFER_DEATH_PATTERN = re.compile(
    rf"(?:患有?|确诊(?:为)?|诊断(?:为)?|患)\s*{_Q}(?:{_TERMS_OR}){_Q}\s*{_REDACT_DEATH_ACTION}",
    re.IGNORECASE,
)

# CD4 细胞计数句法擦除（"CD4+ T细胞180/μL"、"CD4计数180个/μL"）。
# 注意：字符间隙一律使用有界量词（\s{0,2}），杜绝长空白串上的组合回溯（ReDoS）。
_REDACT_CD4_PATTERN = re.compile(
    r"(?:CD4\+?\s{0,2}T?\s{0,1}(?:细胞|淋巴细胞)?\s{0,1}(?:计数)?\s{0,1}(?:为|约)?\s{0,1}\d+\s*(?:个|cells)?\s*(?:/μL|/µL|/ul|/mm3|/L)?\s*[，,。；;]?)",
    re.IGNORECASE,
)

# 1.2 病毒性肝炎载量/检查/活检特征句法整块擦除（"HBV-DNA 5.6×10^6 IU/mL"、"行肝穿刺活检提示G3S4"、"HBsAg阳性"）
# 数值字符类兼容上标数字（⁰¹²³⁴⁵⁶⁷⁸⁹，如 "4.8×10⁶ IU/mL"）
_REDACT_HEPATITIS_FEATURE_CLAUSE_PATTERN = re.compile(
    r"(?:\(?HBV-DNA\s*[\d.×^E+\-⁰¹²³⁴⁵⁶⁷⁸⁹]+\s*(?:IU/mL|copies/ml)?\)?\s*[，,。；;]?)|"
    r"(?:HBV-DNA(?:\s*阳性|\s*阴性|定量)?(?:降至|低于|为)?\s*(?:检测下限|阴性|阳性|\d+)?\s*[，,。；;]?)|"
    r"(?:(?:HBsAg|HBeAg|HBcAb|HBsAb|HBeAb)(?:阳性|阴性)?\s*[，,。；;]?)|"
    r"(?:(?:行)?肝(?:脏)?穿刺(?:活检)?(?:提示|示)?\s*[A-Z0-9]+\s*[，,。；;]?)|"
    r"(?:(?:腹部超声|超声|CT|MRI)?(?:提示|示)?\s*(?:'[^']*'|“[^”]*”)?\s*改变\s*[，,。；;]?)|"
    r"(?:(?:目前|近期|现)?\s*HBV-DNA降至检测下限[。；;]?)",
    re.IGNORECASE,
)

# 2. 完整服药/用药与处置句法擦除
# 要求：前缀(服用/口服/给予...)、剂量用法(20mg qd)或后缀(控制症状/方案)至少有其一存在，避免无修饰裸词抢先匹配
_MED_PREFIX = r"(?:建议)?\s*(?:尽早)?\s*(?:目前)?\s*(?:长期|定期|口服|服用|给予|使用|行|实施|接受|予|给予口服|开具|遵医嘱|启动|开始)"
_MED_SUFFIX = r"(?:控制舞蹈样症状|控制症状|抗逆转录治疗|抗逆转录|抗病毒治疗|抗病毒|对症治疗|治疗|对症处理|口服|方案)"
_MED_DOSE_FREQ_NONEMPTY = r"(?:(?:\s*\d+(?:\.\d+)?\s*(?:mg|g|ml|u|ug|片|粒|支|%))(?:\s*\b(?:qd|bid|tid|qid|qn|qw|im|iv|po)\b)?|\s*\b(?:qd|bid|tid|qid|qn|qw|im|iv|po)\b)"

_REDACT_MEDICATION_FULL_PATTERN = re.compile(
    rf"(?:{_MED_PREFIX}\s*(?:病理提示|提示|行)?\s*{_Q}(?:{_TERMS_OR}){_Q}{_DOSE}{_FREQ}\s*(?:口服|服用)?\s*(?:及|与|和|合并|\+)?\s*(?:{_Q}(?:{_TERMS_OR}){_Q}{_DOSE}{_FREQ})*\s*(?:{_MED_SUFFIX})?|"
    rf"{_Q}(?:{_TERMS_OR}){_Q}{_DOSE}{_FREQ}\s*(?:口服|服用)?\s*(?:及|与|和|合并|\+)?\s*(?:{_Q}(?:{_TERMS_OR}){_Q}{_DOSE}{_FREQ})*\s*{_MED_SUFFIX}|"
    rf"{_Q}(?:{_TERMS_OR}){_Q}{_MED_DOSE_FREQ_NONEMPTY})",
    re.IGNORECASE,
)

# 3. 就诊机构与独立诊断句法整句擦除（"曾就诊于精神卫生中心"、"就诊于某某医院"、"患者1年前查出'乙型肝炎'，"）
_REDACT_HOSPITAL_PATTERN = re.compile(
    rf"(?:曾?就诊于|就诊于|收治于|转诊至|住院于|门诊于)\s*{_Q}(?:{_TERMS_OR}|[\w\u4e00-\u9fa5]{{2,15}}(?:医院|中心|诊所|专科|卫生院|卫生所|外院)){_Q}\s*(?:，|,|。|；|;)?",
    re.IGNORECASE,
)
_REDACT_DIAGNOSIS_STANDALONE_PATTERN = re.compile(
    rf"(?:患者\s*\d+\s*(?:年|月|天)?前|既往)?\s*(?:诊断为|确诊为|检查出|查出|发现|提示为|考虑为)\s*{_Q}(?:{_TERMS_OR}){_Q}\s*(?:，|,|。|；|;)?",
    re.IGNORECASE,
)

# 4. 连词+敏感特征+倾向/表现整块擦除：“及保护性约束倾向” -> “”
# 要求：前缀(及|与|和|伴|伴有) 或 后缀(倾向|表现) 至少存在其一，避免裸词抢先匹配架空后续列表/亲属重构规则
_REDACT_FEATURE_TENDENCY_PATTERN = re.compile(
    rf"(?:(?:及|与|和|伴|伴有)\s*{_Q}(?:{_TERMS_OR}){_Q}\s*(?:倾向|表现|体征|症状)?|{_Q}(?:{_TERMS_OR}){_Q}\s*(?:倾向|表现|体征|症状))",
    re.IGNORECASE,
)

# 5. 顿号/逗号分隔的复合疾病列表中的敏感词擦除："患'重度精神分裂症'、'2型糖尿病'" -> "患'2型糖尿病'"
_REDACT_PAIRED_PATTERN = re.compile(
    rf"((?:因|患有?|确诊(?:为)?|诊断(?:为)?|患|有|合并|伴有?)\s*){_Q}(?:{_TERMS_OR}){_Q}\s*[、,，及与和]\s*",
    re.IGNORECASE,
)
# 5.1 补充：处理敏感词在列表非首位的场景（"患'2型糖尿病'、'重度精神分裂症'" -> "患'2型糖尿病'"）
_REDACT_PAIRED_SUFFIX_PATTERN = re.compile(
    rf"[、,，及与和]\s*{_Q}(?:{_TERMS_OR}){_Q}",
    re.IGNORECASE,
)

# 6. 亲属单疾病场景整句重构为"患病"："母亲患'乙型肝炎'。" -> "母亲患病。"
_REDACT_SINGLE_SUFFER_PATTERN = re.compile(
    rf"(?:患有?|确诊(?:为)?|诊断(?:为)?|患)\s*{_Q}(?:{_TERMS_OR}){_Q}\s*(?:病史|史)?",
    re.IGNORECASE,
)

# 7. 既往史/病史带前缀与后缀完全擦除：“慢性乙型肝炎病史4年” -> “病史4年” (消除残留的“慢性”/“慢史”)
_REDACT_HISTORY_PATTERN = re.compile(
    rf"(?:既往|慢性)?\s*{_Q}(?:{_TERMS_OR}){_Q}\s*(?:病史|史)?",
    re.IGNORECASE,
)

# 8. 亲属关系词定义与孤立主语/动词清理
_FAMILY_MEMBERS = (
    r"父亲|母亲|祖父|祖母|外公|外婆|爷爷|奶奶|伯父|叔叔|舅舅|姑姑|姨妈|大伯|大舅|大姨|二姨|小姨|"
    r"一弟|二弟|三弟|长子|次子|长女|次女|长兄|次兄|大哥|二哥|大姐|二姐|弟弟|妹妹|哥哥|姐姐|爱人|配偶|丈夫|妻子|儿子|女儿|家属|家族成员"
)
_CLEANUP_ORPHAN_SUBJECT_PATTERN = re.compile(rf"(?:^|[，,。；])\s*(?:{_FAMILY_MEMBERS})\s*([。；;])")
_CLEANUP_FAMILY_VERB_HEAL_PATTERN = re.compile(
    rf"({_FAMILY_MEMBERS})\s*(?:患有?|确诊(?:为)?|诊断(?:为)?|患|有)\s*([。；;，,])"
)

# 9. 模块级预编译规范清理正则
_CLEANUP_DEVELOP_AND_PATTERN = re.compile(r"发展为\s*与")
_CLEANUP_PATIENT_TIME_PREFIX_PATTERN = re.compile(r"(?:患者\s*\d+\s*(?:年|月|天|周)?前)\s*([，,])")
_CLEANUP_ORPHAN_PREP_PATTERN = re.compile(
    r"(?:目前行|目前|阳性|阴性|显示阳性|提示阳性|抗病毒治疗|抗病毒|抗逆转录治疗|抗逆转录|同时因|由于|同时|曾?就诊于|诊断为|确诊为|检查出|查出|提示为|及倾向|及控制症状|控制症状|控制|基因检测提示|基因检测示|基因检测|长期|定期|口服|服用|血清学|血清学检查示?|予|给予|及|与|和)\s*([。；;，,])"
)
_CLEANUP_ORPHAN_VERB_PATTERN = re.compile(
    r"(?:^|[，,。；])\s*(?:因|由于|患有?|确诊|患|有|行|进行|接受|服用|合并|伴有|予|控制)\s*([。；;，,])"
)
_CLEANUP_VERB_PUNCT_PATTERN = re.compile(
    r"((?:因|由于|患有?|确诊|患|有|行|进行|接受|服用|合并|伴有))\s*[、,，]"
)
_CLEANUP_NO_OBJ_VERB_PATTERN = re.compile(
    r"(?:急诊行|急诊就诊|就诊|行|实施|接受|予|给予)\s*(?:提示|检查提示|显示|示)?\s*(?:及|与|和)?\s*([。；;，,])"
)
_CLEANUP_NO_OBJ_HINT_PATTERN = re.compile(r"(?:基因检测提示|基因检测示|基因检测|检查提示|检查示|提示|显示|示|予|控制)\s*([。；;，,])")
_CLEANUP_EMPTY_QUOTES_PATTERN = re.compile(r"['\"“‘]['\"”’]")
_CLEANUP_PUNCTUATION_PATTERN = re.compile(r"([，。；：,;])\1+")
_CLEANUP_EMPTY_CLAUSE_PATTERN = re.compile(r"([，,、])\s*([。;；])")
_CLEANUP_LEADING_PUNCT_PATTERN = re.compile(r"^[，,；;。]\s*")
_CLEANUP_EMPTY_PAREN_PATTERN = re.compile(r"[\(（]\s*[\)）]")

# 语法自愈常用正则预编译池
_CLEANUP_EMPTY_OP_PAREN_RE = re.compile(r"[\(（][\s\+\-\*\/]*[\)）]")
_CLEANUP_HAART_LONG_RE = re.compile(r"开展\s*(?:HAART\s*)?抗病毒治疗")
_CLEANUP_HAART_SHORT_RE = re.compile(r"(?:HAART\s*)?抗病毒治疗")
_CLEANUP_HAART_WORD_RE = re.compile(r"\bHAART\b", re.IGNORECASE)
_CLEANUP_HIV_PAREN_RE = re.compile(r"[\(（]\s*(?:HIV\s*)?(?:[\u4e00-\u9fa5]{0,6}(?:期|型|阶段|试验)|期|型)?\s*[\)）]")
_CLEANUP_NAME_LABEL_RE = re.compile(r"(?<=姓名[：:])\s*([\u4e00-\u9fa5])[\u4e00-\u9fa5]{1,2}")
_CLEANUP_PATIENT_LABEL_RE = re.compile(r"(?<=患者[：:])\s*([\u4e00-\u9fa5])[\u4e00-\u9fa5]{1,2}")
_CLEANUP_COLON_COMMA_RE = re.compile(r"([：:])\s*[，,、]")
_CLEANUP_COLON_PERIOD_RE = re.compile(r"([：:])\s*[。；;]")
_CLEANUP_COMMA_PERIOD_RE = re.compile(r"([，,])\s*([。；;])")
_CLEANUP_APPEAR_PUNCT_RE = re.compile(r"(?:出现|发展为|表现为)\s*([。；;，,])")
_CLEANUP_REPEAT_QUOTES_RE = re.compile(r"(['\"“‘'”’])\1+")
_CLEANUP_REPEAT_PUNCT_RE = re.compile(r"([。；;,，])\1+")
_CLEANUP_TIME_PREFIX_PUNCT_RE = re.compile(r"(?:1年前有|半年前|1年前|既往有|曾有|自述有|外阴|曾出现|出现|自愈)\s*([。；;，,])")
_CLEANUP_TIME_PREFIX_RE = re.compile(r"(?:1年前有|半年前|1年前|既往有|曾有|自述有|外阴|曾出现|出现|自愈)")
_CLEANUP_HISTORY_START_PUNCT_RE = re.compile(r"(?:追问病史|诊断为|确诊为|建议尽早启动|尽早启动|启动|开展|进一步检查|进一步|发现)\s*([。；;，,])")
_CLEANUP_HISTORY_START_RE = re.compile(r"(?:追问病史|诊断为|确诊为|建议尽早启动|尽早启动|启动|开展|进一步检查|进一步)")
_CLEANUP_SEEK_CARE_PUNCT_RE = re.compile(r"(?:曾?就诊于|就诊于|收治于|转诊至|住院于)\s*([。；;，,])")
_CLEANUP_SEEK_CARE_RE = re.compile(r"(?:曾?就诊于|就诊于|收治于|转诊至|住院于)")
_CLEANUP_SYMPTOM_ITCH_RE = re.compile(r"(?:伴|与|和)?\s*(?:局部)?(?:轻度)?(?:瘙痒|异物感|接触性出血)\s*([。；;，,])?")
_CLEANUP_DOCTOR_ORDER_RE = re.compile(r"(?:医嘱[：:])\s*(?:立即|及时|定期)?\s*([。；;])")
_CLEANUP_DIE_PAREN_RE = re.compile(r"(死于|殁于)\s*[\(（]([^）\)]+)[\)）]")
_CLEANUP_ILL_PUNCT_RE = re.compile(r"(?<=[\u4e00-\u9fa5])患\s*([\(（。；;,，])")
_CLEANUP_BECAUSE_DIE_RE = re.compile(r"(?:因|死于|因于)\s*(去世|死于|离世|逝世)")
_CLEANUP_FAMILY_DIE_RE = re.compile(rf"({_FAMILY_MEMBERS})\s*(?:殁于|死于|身亡于|病逝于|离世于|由)\s*([。；;，,]|(?={_FAMILY_MEMBERS}))")
_CLEANUP_DIAG_PREFIX_COMMA_RE = re.compile(r"((?:因|患有?|确诊(?:为)?|诊断(?:为)?|患|有|合并|伴有?))\s*[、,，]\s*")
_CLEANUP_AGE_ANON_RE = re.compile(r"((?:死于|确诊|患病|发病|年龄|生于|现年|年满|[\(（])?\s*)(\d{1,3})\s*岁([^\)\n，,。；;]*)")
_CLEANUP_IMAGE_EXT_RE = re.compile(
    r"(\b[\w/\\.-]*?)(?:syphilis|hiv|aids|cancer|tumor|hepatitis)([\w/\\.-]*\.(?:png|jpg|jpeg|dcm|webp|gif)\b)",
    re.IGNORECASE,
)
_CLEANUP_LEADING_CONJ_RE = re.compile(r"^[与和及且并]+\s*")
_CLEANUP_LEADING_PATIENT_PUNCT_RE = re.compile(r"^患者[。；;，,]\s*")
_CLEANUP_LEADING_PATIENT_CHEST_RE = re.compile(r"^患者(?=[^，,。；;]{0,10}详见)")
_CLEANUP_HORIZ_SPACES_RE = re.compile(r"[ \t]{2,}")


# 10. 性传播疾病与极高敏特征综合句法擦除正则（涵盖血清学检查示TPPA/RPR滴度、不洁接触史、无痛性溃疡/硬下疳自愈等完整词句）
# ReDoS 防护说明：临床中文短语的组成字词之间天然无空白，因此各分支的可选修饰组之间
# 一律不使用 \s*（仅保留至多一处有界 \s{0,2}）。若可选组之间串联多个无界 \s*，
# 恶意构造的 "患者 + 长空白串 + 噪声" 输入会在各 \s* 槽位间产生组合级回溯分配（灾难性回溯）。
_REDACT_STD_FEATURE_CLAUSE_PATTERN = re.compile(
    r"(?:"
    # 分支 1：梅毒等性病病史句法（"患者1年前有梅毒病史，"、"梅毒（早期隐性梅毒）"）
    r"(?:患者)?(?:\d+[年月天]+前)?(?:既往有|曾有|自述有|有)?(?:早期|晚期|隐性|神经|心血管|胎传|先天)*梅毒(?:\s{0,2}[\(（][^)）]*[\)）])?(?:病史|史)?\s*[，,。；;]?"
    r"|(?:患者)?(?:\d+[年月天]+前)?(?:既往有|曾有|自述有|有)?(?:淋病|尖锐湿疣|生殖器疱疹|软下疳|性病)(?:\s{0,2}[\(（][^)）]*[\)）])?(?:病史|史)?\s*[，,。；;]?"
    # 分支 2：检查出/确诊为/诊断为 '梅毒' 类
    r"|(?:检查出|确诊为|诊断为)?['\"“]?(?:梅毒|TPPA阳性|RPR阳性|淋病|尖锐湿疣)['\"”]?\s*[，,。；;]?"
    # 分支 3：血清学 TPPA/RPR 滴度
    r"|(?:血清学检查示|血清学检查|血清学)?\s*(?:TPPA阳性|TPPA|RPR阳性|RPR\s*1:\d+|\d+:\d+)\s*[，,。；;]?"
    # 分支 4：不洁接触史（病因/诱因柱）
    r"|(?:追问病史[，,]?)?(?:1年前有|既往有|曾有)?(?:不洁性接触史|不洁接触史)\s*[，,。；;]?"
    # 分支 5：无痛性溃疡/硬下疳（体征柱）
    r"|(?:半年前|1年前)?(?:外阴)?(?:曾出现|出现)?无痛性溃疡(?:[\(（]硬下疳[\)）])?(?:自愈)?\s*[，,。；;]?"
    # 分支 6：菜花状/鸡冠状/乳头状赘生物体征群（体征柱）
    r"|(?:患者)?(?:\d+[年月天]+前)?(?:发现|出现)?(?:外阴及会阴部|外阴|会阴部|肛周)?(?:及(?:外阴|会阴部|肛周))?(?:多发)?(?:(?:菜花状|鸡冠状|乳头状)?赘生物|菜花状|鸡冠状)(?:[，,]逐渐增多)?(?:[，,]伴(?:局部)?(?:轻度)?(?:瘙痒|异物感|接触性出血))*\s*[，,。；;]?"
    # 分支 7：醋酸白试验/HPV 基因型/活检提示（检查柱）
    r"|(?:醋酸白试验(?:阳性)?|HPV\s*(?:6/11|16/18)?(?:低危型|高危型)?(?:阳性)?|(?:病理)?活检提示(?:尖锐湿疣)?)\s*[，,。；;]?"
    # 分支 8：CO2 激光/咪喹莫特等特异性处置（用药/处置柱）
    r"|(?:行|给予|实施)?['\"“]?(?:CO2激光灼除术|CO2激光灼除治疗|CO2激光灼除|CO2激光治疗|激光灼除术|二氧化碳激光|咪喹莫特乳膏(?:外用|局部涂抹)?|咪喹莫特)['\"”]?(?:及|与|和)?['\"“]?(?:CO2激光灼除术|CO2激光灼除治疗|CO2激光灼除|CO2激光治疗|激光灼除术|二氧化碳激光|咪喹莫特乳膏(?:外用|局部涂抹)?|咪喹莫特)?['\"”]?(?:外用|局部涂抹|治疗)?\s*[，,。；;]?"
    r")",
    re.IGNORECASE,
)

# 11. 遗传缺陷与基因检测综合句法擦除正则（涵盖基因检测提示'遗传性亨廷顿舞蹈病'(HTT基因CAG重复序列46次)、四肢舞蹈样动作等）
_REDACT_GENETIC_CLAUSE_PATTERN = re.compile(
    r"(?:"
    r"(?:基因检测提示|基因检测示|基因检测结果示?|基因检测)?\s*['\"“]?(?:遗传性亨廷顿舞蹈病|亨廷顿病?|舞蹈病|HTT基因|CAG重复序列|CAG重复|CAG扩增)['\"”]?\s*(?:\([^)]*\)|（[^）]*）)?\s*[，,。；;]?"
    r"|(?:四肢)?舞蹈样动作\s*(?:与|和|及)?\s*"
    r")",
    re.IGNORECASE,
)

# 12. 图片/附件路径名称清理（防止路径中泄露敏感词，如 /data/hiv_test_01.jpg -> /data/masked_01.jpg）
_IMAGE_PATH_PATTERN = re.compile(r"/(?:[^/]+\.png|[^/]+\.jpg|[^/]+\.jpeg|[^/]+\.bmp)")

# 13. 范畴化降级泛化规则映射 (Category Generalization Rules)
# 仅对适宜泛化的病种（肿瘤、肝炎、遗传缺陷、器官衰竭）自动重构降级为 L1/L2 通用系统/器官疾病表述；
# 性病 (STD)、艾滋病 (HIV)、重度精神障碍属于禁止泛化范畴，100% 自动直接抹平切除 (Purge Only)！
_CATEGORY_GENERALIZATION_RULES: list[tuple[str, re.Pattern, str]] = [
    # 1. 恶性肿瘤范畴 -> 通用系统/器官疾病
    # 注意：器官/系统前缀为必选匹配，且各系统专属规则先于裸"肿瘤"兜底规则，
    # 否则 "呼吸系统肿瘤" 会被首条规则误泛化为 "呼吸系统消化道疾病"（张冠李戴）。
    ("MALIGNANT_NEOPLASM", re.compile(r"消化道(?:恶性)?肿瘤(?=聚集倾向|家族史|史|风险)", re.IGNORECASE), "消化道疾病"),
    ("MALIGNANT_NEOPLASM", re.compile(r"(?:呼吸道|呼吸系统)(?:恶性)?肿瘤(?=聚集倾向|家族史|史|风险)", re.IGNORECASE), "呼吸系统疾病"),
    ("MALIGNANT_NEOPLASM", re.compile(r"生殖系统(?:恶性)?肿瘤(?=聚集倾向|家族史|史|风险)", re.IGNORECASE), "生殖系统疾病"),
    ("MALIGNANT_NEOPLASM", re.compile(r"神经系统(?:恶性)?肿瘤(?=聚集倾向|家族史|史|风险)", re.IGNORECASE), "神经系统疾病"),
    ("MALIGNANT_NEOPLASM", re.compile(r"(?:恶性)?肿瘤(?=聚集倾向|家族史|史|风险)", re.IGNORECASE), "相关系统疾病"),

    # 2. 病毒性肝炎范畴 -> 通用肝脏疾病
    ("HEPATITIS_VIRUS", re.compile(r"(?:慢性乙型病毒性肝炎|乙型肝炎|乙肝|丙型肝炎|丙肝|肝硬化代偿期|早期肝硬化|肝硬化)(?=家族史|史|聚集倾向)", re.IGNORECASE), "肝脏疾病"),

    # 3. 重大遗传缺陷范畴 -> 遗传性神经系统疾病
    ("GENETIC_DEFECT", re.compile(r"(?:遗传性亨廷顿舞蹈病|亨廷顿病|舞蹈病|罕见遗传病)(?=家族史|史|聚集倾向)", re.IGNORECASE), "遗传性神经系统疾病"),

    # 4. 严重器官衰竭范畴 -> 系统重大疾病
    ("SEVERE_ORGAN_DAMAGE", re.compile(r"(?:急性心肌梗死|冠状动脉重度狭窄)(?=家族史|史|聚集倾向)", re.IGNORECASE), "心血管系统疾病"),
    ("SEVERE_ORGAN_DAMAGE", re.compile(r"(?:慢性阻塞性肺疾病|COPD)(?=家族史|史|聚集倾向)", re.IGNORECASE), "慢性呼吸系统疾病"),
    ("SEVERE_ORGAN_DAMAGE", re.compile(r"(?:尿毒症|肾功能衰竭)(?=家族史|史|聚集倾向)", re.IGNORECASE), "肾脏系统疾病"),
]

# 预计算默认泛化允许集合（避免每次 _apply_category_generalizations 调用重复构建）
_DEFAULT_GENERALIZATION_ALLOWED: frozenset[str] = frozenset(
    {category for category, _pattern, _replacement in _CATEGORY_GENERALIZATION_RULES}
)


def _apply_category_generalizations(
    text: str, strategy: RedactionStrategyConfig | None = None
) -> str:
    """按运行时策略应用泛化；未列入泛化的类别继续走抹平流程。

    Args:
        text: 待泛化文本。
        strategy: 运行时策略配置。为 None 时使用代码内置默认泛化规则
        （所有 ``_CATEGORY_GENERALIZATION_RULES`` 中出现的类别均允许泛化，无 purge 排除）。
        注意：None 语义 ≠ YAML 默认策略——Pipeline 始终从 YAML 加载并传入显式策略。
    """
    s = text
    if strategy is not None:
        allowed = set(strategy.generalization_categories)
        purged = set(strategy.purge_categories)
    else:
        allowed = _DEFAULT_GENERALIZATION_ALLOWED
        purged = set()
    for category, pattern, replacement in _CATEGORY_GENERALIZATION_RULES:
        if category not in allowed or category in purged:
            continue
        s = pattern.sub(replacement, s)
    return s


def _redact_terms_only(text: str) -> str:
    """超长文本降级路径：仅做词库级擦除（无句法重构），保证性能与安全。

    当输入文本超过 _REDACT_MAX_TEXT_LENGTH 时调用，避免复杂句法正则的 ReDoS 风险。
    擦除后可能残留句法碎片（如"患有去世"），由上层 pipeline 的最终门禁兜底。
    """
    return _TERMS_ONLY_PATTERN.sub("", text)


def _clean_orphan_syntax(s: str) -> str:
    """清理擦除敏感实体后残存的孤立介词、连词、无宾语动词与多余标点。"""
    if not s:
        return s

    # ReDoS 全局防护（与 redact_medical_text 一致）：折叠连续水平空白串，
    # 防止下方清理正则在长空白 run 上的组合回溯（幂等操作，正常文本不受影响）
    s = _CLEANUP_HORIZ_SPACES_RE.sub(" ", s)

    # 0. 优先清理擦除产生的空括号，避免阻碍后续孤立动词与标点匹配；并自动擦除就诊医院/机构句法与肝炎体征载量短语
    s = _CLEANUP_EMPTY_PAREN_PATTERN.sub("", s)
    s = _REDACT_HOSPITAL_PATTERN.sub("", s)
    s = _REDACT_HEPATITIS_FEATURE_CLAUSE_PATTERN.sub("", s)

    # 0.1 语法自愈：清理多药联合处方擦除后残留的空运算符、空括号及残缺治疗短语（如 "（ +  + ）" -> ""；"开展 HAART 抗病毒治疗" -> "开展常规对症治疗"）
    s = _CLEANUP_EMPTY_OP_PAREN_RE.sub("", s)
    s = _CLEANUP_HAART_LONG_RE.sub("开展常规对症治疗", s)
    s = _CLEANUP_HAART_SHORT_RE.sub("常规对症治疗", s)
    s = _CLEANUP_HAART_WORD_RE.sub("", s)

    # 0.2 语法自愈：清理擦除主诊断后残存的孤立病期/分型修饰括号（如 "（感染期）"、"（期）"、"（确证试验）"）
    s = _CLEANUP_HIV_PAREN_RE.sub("", s)

    # 0.3 语法自愈：中文姓名自动掩码遮蔽（如 "姓名：张三" -> "姓名：张*"；"患者：张三" -> "患者：张*"）
    s = _CLEANUP_NAME_LABEL_RE.sub(r"\1*", s)
    s = _CLEANUP_PATIENT_LABEL_RE.sub(r"\1*", s)

    # 0.4 语法自愈：消除冒号后紧跟逗号/句号等非法中文标点碰撞（如 "初步诊断：，伴..." -> "初步诊断：伴..."）
    s = _CLEANUP_COLON_COMMA_RE.sub(r"\1", s)
    s = _CLEANUP_COLON_PERIOD_RE.sub("。", s)
    s = _CLEANUP_COMMA_PERIOD_RE.sub(r"\2", s)

    # 1. 清理孤立无宾语动词：如“示。”、“提示。”、“急诊行提示”、“予行”、“予行及”、“予。”
    s = _CLEANUP_NO_OBJ_VERB_PATTERN.sub(r"\1", s)
    s = _CLEANUP_NO_OBJ_HINT_PATTERN.sub(r"\1", s)

    # 2. 清理孤立连词与介词碎片：如“伴及。”、“及。”、“与。”、“伴。”、“长期。”、“发展为。”
    s = _CLEANUP_ORPHAN_PREP_PATTERN.sub(r"\1", s)
    s = _CLEANUP_APPEAR_PUNCT_RE.sub(r"\1", s)
    s = _CLEANUP_DEVELOP_AND_PATTERN.sub("发展为", s)

    # 3. 标点与空括号自愈
    s = _CLEANUP_REPEAT_QUOTES_RE.sub("", s)
    s = _CLEANUP_PUNCTUATION_PATTERN.sub(r"\1", s)
    s = _CLEANUP_EMPTY_CLAUSE_PATTERN.sub(r"\2", s)
    s = _CLEANUP_LEADING_PUNCT_PATTERN.sub("", s)
    s = _CLEANUP_EMPTY_PAREN_PATTERN.sub("", s)
    s = _CLEANUP_REPEAT_PUNCT_RE.sub(r"\1", s)

    # 5. 清理擦除敏感病史/症状后遗留的孤立前缀、后缀与时间短语
    s = _CLEANUP_TIME_PREFIX_PUNCT_RE.sub(r"\1", s)
    s = _CLEANUP_TIME_PREFIX_RE.sub("", s)
    s = _CLEANUP_HISTORY_START_PUNCT_RE.sub(r"\1", s)
    s = _CLEANUP_HISTORY_START_RE.sub("", s)
    s = _CLEANUP_SEEK_CARE_PUNCT_RE.sub(r"\1", s)
    s = _CLEANUP_SEEK_CARE_RE.sub("", s)
    s = _CLEANUP_SYMPTOM_ITCH_RE.sub(r"\1", s)
    # 5.05 医嘱残余自愈：如果医嘱内容被高敏专科医院与特种用药整体擦除，自愈润色为常规健康管理描述
    s = _CLEANUP_DOCTOR_ORDER_RE.sub("医嘱：遵医嘱常规治疗与健康管理。", s)

    # 5.1 死因孤立介词自愈重构 ("因去世" -> "因病去世")、括号年龄清理 ("死于(62岁)" -> "死于62岁") 与孤立"患"补全 ("母亲患(55岁确诊)" -> "母亲患病(55岁确诊)")
    s = _CLEANUP_DIE_PAREN_RE.sub(r"\1\2", s)
    s = _CLEANUP_ILL_PUNCT_RE.sub(r"患病\1", s)
    s = _CLEANUP_BECAUSE_DIE_RE.sub(r"因病\1", s)
    s = _CLEANUP_FAMILY_DIE_RE.sub(r"\1因病去世\2", s)
    s = _CLEANUP_DIAG_PREFIX_COMMA_RE.sub(r"\1", s)

    # 5.2 单条记录准标识符自适应年龄 K-匿名泛化 (<60岁按3岁区间/age-(age%3)，>=60岁按2岁精细康养区间/age-(age%2))
    def _age_anon_repl(match: re.Match) -> str:
        prefix = match.group(1) or ""
        age_num = match.group(2)
        suffix = match.group(3) or ""
        from ..privacy.kano import adaptive_age_hierarchy
        anon_age = adaptive_age_hierarchy(age_num, under_60_interval=3, senior_interval=2, output_format="floor")
        return f"{prefix}{anon_age}岁{suffix}"

    s = _CLEANUP_AGE_ANON_RE.sub(_age_anon_repl, s)

    s = _CLEANUP_EMPTY_CLAUSE_PATTERN.sub(r"\2", s)
    s = _CLEANUP_LEADING_PUNCT_PATTERN.sub("", s)
    s = _CLEANUP_REPEAT_PUNCT_RE.sub(r"\1", s)

    # 6. 去标识化擦除文本或图片引用路径中包含的高敏感英文词汇（如 syphilis, hiv, cancer 等）
    s = _CLEANUP_IMAGE_EXT_RE.sub(r"\1sanitized_case_image\2", s)

    # 7. 清理开头孤立的连词与无谓主语（含敏感句被整体擦除后残留的 "患者胸片详见..." 中的悬空主语）
    s = _CLEANUP_LEADING_CONJ_RE.sub("", s)
    s = _CLEANUP_LEADING_PATIENT_PUNCT_RE.sub("", s)
    s = _CLEANUP_LEADING_PATIENT_CHEST_RE.sub("", s)

    # 8. 最终判断：若全句抹平后仅剩无主语/无主病因孤立频次、动词或时间状语从句，直接抹平清空。
    # 该正则含多个无界 \s* 槽位与 $ 锚定，对长输入存在组合回溯风险——仅对短残渣（<=30 字符）执行，
    # 长残渣不存在"无主语状语从句"语义，直接跳过既安全又防 ReDoS。
    tail = s.strip()
    if len(tail) <= 30 and re.match(r"^(?:患者)?\s*(?:\d+\s*(?:年|月|天|周|小时|周期|疗程|次)?\s*(?:前)?)?\s*(?:无明显诱因|体检|目前|近期|现|发现|检查出|查出|提示|示|进一步检查|进一步|出现|曾出现|既往|反复发作|发作|持续|存在|明显|自述|口服|服用|给予|使用|予|遵医嘱|服|长期|定期|术后|抗病毒治疗|抗病毒|抗逆转录治疗|抗逆转录|检测不到|低于检测下限|者|为者)*\s*\d*\s*(?:年|月|天|周|小时|周期|疗程|次)?\s*(?:余)?\s*(?:年|月|天|周)?\s*[。；;，,]*$", tail):
        return ""

    return s.strip()


# ---------------------------------------------------------------------------
# ICD-10 高危诊断编码段治理（§9 规约：L5 强抹平，L4 替换为范畴码）
# 诊断名称抹平后，编码本身（如 B20.900=HIV、C34.900=肺恶性肿瘤）仍会泄露病种，
# 因此对编码字段按 ICD-10 章节码段独立定级与脱敏。
# ---------------------------------------------------------------------------

# 诊断编码字段名集合（命中这些字段名时才按 ICD-10 码段规则处理，避免误伤普通文本）
ICD10_FIELD_NAMES: frozenset[str] = frozenset({
    "icd10_code", "icd10", "icd_code", "icd", "diagnosis_code", "诊断编码",
})

# ICD-10 编码形态：字母 + 两位类目数字 + 可选亚目（如 C34.900 / I10.x00 / G10）
_ICD10_CODE_PATTERN = re.compile(r"^\s*([A-Za-z])(\d{2})(?:\.[xX\d]\d*)?\s*$")


def classify_icd10_code(code: str) -> tuple[str, str] | None:
    """按 ICD-10 章节码段判定诊断编码的风险等级与范畴。

    Returns:
        (level, category) 元组；level 为 "L5"（极高敏，需整值抹平）或 "L4"（高敏，
        替换为范畴码）；category 为范畴代码（用于脱敏替换标签与审计追溯）。
        非高危编码或非法编码形态返回 None。
    """
    match = _ICD10_CODE_PATTERN.match(code or "")
    if not match:
        return None
    letter, number = match.group(1).upper(), int(match.group(2))
    # L5 极高敏：HIV(B20-B24)、精神分裂症(F20-F29)、亨廷顿舞蹈病(G10)
    if letter == "B" and 20 <= number <= 24:
        return ("L5", "MEDICAL_ICD10_HIV")
    if letter == "F" and 20 <= number <= 29:
        return ("L5", "MEDICAL_ICD10_PSYCHIATRIC")
    if letter == "G" and number == 10:
        return ("L5", "MEDICAL_ICD10_GENERAL")
    # L4 高敏：性传播疾病(A50-A64)、肿瘤(C00-C97/D00-D48)、病毒性肝炎(B15-B19)、
    # 急性心肌梗死(I21-I22)、慢性肾病/尿毒症(N18-N19)、慢阻肺(J44)
    if letter == "A" and 50 <= number <= 64:
        return ("L4", "MEDICAL_ICD10_STD")
    if (letter == "C" and 0 <= number <= 97) or (letter == "D" and 0 <= number <= 48):
        return ("L4", "MEDICAL_ICD10_CANCER")
    if letter == "B" and 15 <= number <= 19:
        return ("L4", "MEDICAL_ICD10_GENERAL")
    if letter == "I" and 21 <= number <= 22:
        return ("L4", "MEDICAL_ICD10_GENERAL")
    if letter == "N" and 18 <= number <= 19:
        return ("L4", "MEDICAL_ICD10_GENERAL")
    if letter == "J" and number == 44:
        return ("L4", "MEDICAL_ICD10_GENERAL")
    return None


def redact_icd10_code(code: str) -> str:
    """ICD-10 编码脱敏：L5/L4 均整值抹平（无痕），非高危原样返回。

    与 Go ``medical.RedactICD10Code`` 对齐：L4 范畴码标签同样会暴露敏感病种归属，
    因此与 L5 一并整值擦除。
    """
    result = classify_icd10_code(code)
    if result is None:
        return code
    level, _category = result
    if level in ("L4", "L5"):
        return ""
    return code


# ---------------------------------------------------------------------------
# 日期准标识符泛化（§9 规约：出生/入院/出院日期 L2，截断为年月）
# ---------------------------------------------------------------------------

# 日期泛化字段名集合：完整精度日期属于准标识符（组合重识别风险），截断为年月
DATE_GENERALIZATION_FIELDS: frozenset[str] = frozenset({
    "birth_date", "admission_date", "discharge_date", "出生日期", "入院日期", "出院日期",
})

_DATE_PREFIX_PATTERN = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.]\d{1,2}")


def truncate_date_to_month(date_str: str) -> str:
    """将 YYYY-MM-DD / YYYY/MM/DD 等完整日期截断为 YYYY-MM（无法解析时原样返回）。

    与 Go ``medical.TruncateDateToMonth`` 对齐：仅匹配字符串开头的日期，月份补零。
    """
    if not date_str or not isinstance(date_str, str):
        return date_str
    match = _DATE_PREFIX_PATTERN.match(date_str)
    if match is None:
        return date_str
    return f"{match.group(1)}-{int(match.group(2)):02d}"


_FULLWIDTH_TO_HALFWIDTH = str.maketrans({
    i: i - 0xFEE0
    for i in list(range(0xFF10, 0xFF1A)) + list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B))
})


def normalize_fullwidth_alphanumeric(text: str) -> str:
    """仅将全角英文字母与数字（如 ＨＩＶ、１２３）转换为半角（HIV、123），保留中文标点（，。；“”）。"""
    if not text:
        return text
    return text.translate(_FULLWIDTH_TO_HALFWIDTH)


def redact_medical_text(
    text: str, strategy: RedactionStrategyConfig | None = None
) -> str:
    """全场景高级无痕抹平算法，泛化决策由 ``strategy`` 控制。"""
    if not text:
        return text

    norm_text = normalize_fullwidth_alphanumeric(text)

    # 超长文本保护：降级为简单词库擦除
    if len(norm_text) > _REDACT_MAX_TEXT_LENGTH:
        return _redact_terms_only(norm_text)

    # 首字符预筛：不含任何词库首字符（且无脱敏标签）的文本直接原样返回（毫秒级），
    # 避免对干净文本逐位置尝试数百分支的全量词库匹配
    if not _TERMS_FIRST_CHARS_PATTERN.search(text) and not _MASKED_LABEL_RE.search(text):
        return text

    # 三级检测：单次 DFA 扫描匹配
    if not (_TERMS_ONLY_PATTERN.search(norm_text) or _MASKED_LABEL_RE.search(text)):
        stripped_norm = _STRIPPED_NOISE_RE.sub("", norm_text)
        if stripped_norm == norm_text or not _TERMS_ONLY_PATTERN.search(stripped_norm):
            return text

    s = norm_text
    # 注：词库正则已通过 _flex_escape 实现字符间分隔符容忍（"H I V"/"H.I.V"/"艾-滋-病" 等变体
    # 直接命中词库），无需再为个别词手工编写"先拼合再匹配"的补丁式规则。

    # ReDoS 全局防护：折叠连续水平空白串（>=2 个空格/制表符 → 单空格）。
    # 句法正则的可选修饰组之间以 \s* 连接，恶意构造的长空白串会在多个 \s* 槽位间
    # 引发组合级回溯分配（实测 "梅毒，患者"+2000空格 在用药句法正则上挂死 >10s）；
    # 折叠后每个 \s* 槽位至多消费 1 个字符，回溯空间降为常数，一次性切断全模式攻击面。
    # 仅影响已进入敏感路径的文本（干净文本在上方 Fast-Path 已原样返回，零篡改）。
    s = re.sub(r"[ \t]{2,}", " ", s)

    def _death_age_replace(match: re.Match) -> str:
        action = match.group(1) or "死于"
        raw_age = match.group(2)
        from ..privacy.kano import adaptive_age_hierarchy
        anon_age = adaptive_age_hierarchy(raw_age, under_60_interval=3, senior_interval=2, output_format="floor")
        if action in ("因", "由于"):
            return f"因病去世({anon_age}岁)"
        return f"{action}{anon_age}岁"

    def _death_replace(match: re.Match) -> str:
        return "因病去世"

    # 1. 优先重构死因完整句法（因病去世/含年龄泛化）
    s = _REDACT_DEATH_WITH_AGE_PATTERN.sub(_death_age_replace, s)
    s = _REDACT_CAUSE_DEATH_PATTERN.sub(_death_replace, s)
    s = _REDACT_SUFFER_DEATH_PATTERN.sub(_death_replace, s)

    # 2. 擦除 CD4 计数、遗传缺陷、性病及肝炎综合句法
    s = _REDACT_CD4_PATTERN.sub("", s)
    s = _REDACT_GENETIC_CLAUSE_PATTERN.sub("", s)
    s = _REDACT_STD_FEATURE_CLAUSE_PATTERN.sub("", s)
    s = _REDACT_HEPATITIS_FEATURE_CLAUSE_PATTERN.sub("", s)
    s = _apply_category_generalizations(s, strategy)

    # 2. 优先擦除完整服药用药句法
    s = _REDACT_MEDICATION_FULL_PATTERN.sub("", s)

    # 3. 擦除就诊机构句法短语
    s = _REDACT_HOSPITAL_PATTERN.sub("", s)

    # 4. 擦除独立诊断句法短语
    s = _REDACT_DIAGNOSIS_STANDALONE_PATTERN.sub("", s)

    # 5. 擦除敏感特征倾向短语
    s = _REDACT_FEATURE_TENDENCY_PATTERN.sub("", s)

    # 6. 复合疾病场景
    s = _REDACT_PAIRED_PATTERN.sub(r"\1", s)
    s = _REDACT_PAIRED_SUFFIX_PATTERN.sub("", s)

    # 7. 单敏感疾病场景
    s = _REDACT_SINGLE_SUFFER_PATTERN.sub("患病", s)

    # 8. 擦除既往史/病史带前缀与后缀的完整词组
    s = _REDACT_HISTORY_PATTERN.sub("", s)

    # 9. 清理孤立残余介词、连词与标点
    s = _CLEANUP_DEVELOP_AND_PATTERN.sub("发展为", s)
    s = _CLEANUP_ORPHAN_PREP_PATTERN.sub(r"\1", s)
    s = _CLEANUP_VERB_PUNCT_PATTERN.sub("", s)

    s = _CLEANUP_PATIENT_TIME_PREFIX_PATTERN.sub("", s)
    s = _CLEANUP_FAMILY_VERB_HEAL_PATTERN.sub(r"\1患病\2", s)

    s = _CLEANUP_ORPHAN_VERB_PATTERN.sub(r"\1", s)
    s = _CLEANUP_ORPHAN_SUBJECT_PATTERN.sub(r"\1", s)

    s = _CLEANUP_EMPTY_QUOTES_PATTERN.sub("", s)
    s = _CLEANUP_PUNCTUATION_PATTERN.sub(r"\1", s)
    s = _CLEANUP_EMPTY_CLAUSE_PATTERN.sub(r"\2", s)

    return _clean_orphan_syntax(s)


# ---------------------------------------------------------------------------
# L4/L5 重大高敏疾病（及关联高敏处置/药物）提示词指南与判定核心逻辑
# ---------------------------------------------------------------------------
L4_L5_MAJOR_SENSITIVE_PROMPT_GUIDELINE = """
【Layer-2 NER & Rule 级医疗敏感数据四柱无痕脱敏提示词准则 / L4-L5 Four-Pillar Redaction Guidelines】
对于 L4/L5 级别的重大高敏疾病（性病、恶性肿瘤、HIV/AIDS、重度精神障碍、重大遗传缺陷、病毒性肝炎等），必须遵循【四柱强剥离/无痕抹平原则】：
1. 病因/诱因描述；
2. 现象/体征描述；
3. 诊断/检查描述；
4. 用药/处置描述。
"""

_MAJOR_SENSITIVE_KEYWORDS = (
    # L5 Keywords
    "HIV", "AIDS", "艾滋", "免疫缺陷", "CD4+", "CD4", "抗逆转录", "病毒载量", "齐多夫定",
    "精神分裂", "幻听", "妄想", "自伤", "砸物", "保护性约束", "奥氮平", "喹硫平", "奎硫平", "阿立哌唑", "利培酮", "氯氮平", "氨磺必利", "精神卫生",
    "亨廷顿", "CAG重复", "CAG扩增", "CAG", "HTT基因", "HTT", "四苯嗪", "舞蹈病", "舞蹈样",
    # L4 Keywords
    "梅毒", "密螺旋体", "TPPA", "RPR", "淋病", "淋球菌", "尖锐湿疣", "疱疹", "软下疳", "性病", "不洁性接触", "硬下疳", "菜花状", "鸡冠状", "赘生物", "醋酸白", "咪喹莫特", "苄星青霉素",
    "恶性肿瘤", "腺癌", "肺癌", "胃癌", "肝癌", "乳腺癌", "宫颈癌", "癌症", "转移性肿瘤", "转移瘤", "癌", "肉瘤", "奥希替尼", "EGFR", "化疗", "放疗", "靶向治疗",
    "乙型肝炎", "乙肝", "丙型肝炎", "丙肝", "HBV-DNA", "HBV", "HCV-RNA", "HCV", "恩替卡韦", "干扰素", "肝硬化", "蜘蛛痣", "肝掌", "肝硬化腹水", "门静脉高压", "门脉高压", "食管静脉曲张", "脾大", "脾功能亢进", "肝穿刺", "G3S4",
    "急性心肌梗死", "心肌梗死", "冠状动脉重度狭窄", "重度狭窄", "COPD", "阻塞性肺", "尿毒症", "肾功能衰竭"
)


def _is_major_sensitive_entity(term: str, ent_type: str = "") -> bool:
    """判定实体词是否属于 L4/L5 重大高敏疾病或关联高敏处置/药物。"""
    if not term or len(term) < 2:
        return False

    term_clean = term.strip()
    term_upper = term_clean.upper()

    if _TERMS_ONLY_PATTERN.search(term_clean):
        return True

    for kw in _MAJOR_SENSITIVE_KEYWORDS:
        if kw.upper() in term_upper:
            return True

    return False


def redact_medical_text_with_ner(
    text: str,
    ner_adapter: Any = None,
    strategy: RedactionStrategyConfig | None = None,
) -> str:
    """Layer-2 Small-NER 驱动的高级命名实体识别无痕抹平引擎 (Gold Standard Implementation)."""
    if not text:
        return text

    norm_text = normalize_fullwidth_alphanumeric(text)

    # 超长文本保护：与规则路径一致降级为词库级单次擦除，
    # 防复杂句法正则 ReDoS 与超长文本的 NER 推理资源耗尽
    if len(norm_text) > _REDACT_MAX_TEXT_LENGTH:
        return _redact_terms_only(norm_text)

    stripped_norm = re.sub(r"(?<=[a-zA-Z0-9\u4e00-\u9fa5])[\s\.\-_]+(?=[a-zA-Z0-9\u4e00-\u9fa5])", "", norm_text)

    # 当未传入 NER 适配器时执行 Fast-path 前置校验（首字符预筛 → 全量词库三级检测，变体去重）
    if ner_adapter is None:
        if not _TERMS_FIRST_CHARS_PATTERN.search(text) and not _MASKED_LABEL_RE.search(text):
            return text
        if not any(_TERMS_ONLY_PATTERN.search(v) for v in {text, norm_text, stripped_norm}) and not _MASKED_LABEL_RE.search(text):
            return text

    entities = []
    if ner_adapter is not None:
        try:
            raw_entities = ner_adapter.extract(text)
            if isinstance(raw_entities, list):
                entities = raw_entities
        except Exception:
            entities = []

    sensitive_entities = [
        e for e in entities
        if isinstance(e, dict)
        and e.get("text", "").strip()
        and _is_major_sensitive_entity(e.get("text", ""), str(e.get("type") or e.get("label") or ""))
    ]

    if sensitive_entities:
        # ReDoS 全局防护：折叠连续水平空白串（同 redact_medical_text 主路径）
        collapsed = re.sub(r"[ \t]{2,}", " ", norm_text)
        sorted_entities = sorted(
            sensitive_entities,
            key=lambda x: len(x.get("text", "")),
            reverse=True,
        )

        # 先执行规则全量句法擦除（复用主路径：已知词库 + 全部句法步骤 + 语法自愈）。
        # NER 实体锚定擦除单独工作时只擦实体本身，会遗留 "180/μL。行+。"、"确诊" 等
        # 句法残渣（实测泄露）；完整规则路径保证已知特征零残渣，
        # 随后再以 NER 实体为锚点擦除词库外的高敏实体（NER 的核心增量价值）。
        s = redact_medical_text(collapsed, strategy=strategy)

        for ent in sorted_entities:
            term = ent.get("text", "").strip()
            ent_type = str(ent.get("type") or ent.get("label") or "").upper()
            if not term or len(term) < 2:
                continue

            quoted_term = rf"['\"“‘'”’]?{re.escape(term)}['\"”’]?"

            if any(t in ent_type for t in ["DRUG", "MED", "CHEM"]):
                pat = rf"(?:长期|定期|口服|服用|给予|使用|予|遵医嘱)?\s*{quoted_term}{_DOSE}{_FREQ}\s*(?:口服|服用)?\s*(?:及|与|和|合并|\+)?\s*(?:控制舞蹈样症状|控制症状|抗逆转录治疗|抗逆转录|抗病毒治疗|抗病毒|对症治疗|治疗|对症处理|口服|方案)?"
                s = re.sub(pat, "", s, flags=re.IGNORECASE)
            elif any(t in ent_type for t in ["HOSPITAL", "ORG", "LOC"]):
                pat = rf"(?:曾?就诊于|就诊于|收治于|转诊至|住院于|门诊于)\s*{quoted_term}\s*(?:，|,|。|；|;)?"
                s = re.sub(pat, "", s, flags=re.IGNORECASE)
            else:
                pat_paired = rf"((?:因|患有?|确诊(?:为)?|诊断(?:为)?|患|有|合并|伴有?)\s*){quoted_term}\s*[、,，]\s*"
                s = re.sub(pat_paired, r"\1", s, flags=re.IGNORECASE)
                s = re.sub(quoted_term, "", s, flags=re.IGNORECASE)

        return _clean_orphan_syntax(s)

    # NER 未识别出高敏实体（或推断异常）时，降级由规则引擎兜底。
    # 注意：不得在此再套一层 _clean_orphan_syntax——那些清理正则（删"出现/进一步/伴瘙痒"等）
    # 的设计前提是"文本已经历敏感词擦除、只剩残渣"，对干净文本会误删合法用词
    # （实测 "患者出现皮疹3天，伴瘙痒。" 被篡改为 "患者皮疹3天。"）；
    # redact_medical_text 内部已对敏感文本完成语法自愈，对干净文本走 Fast-Path 原样返回。
    return redact_medical_text(text, strategy=strategy)
