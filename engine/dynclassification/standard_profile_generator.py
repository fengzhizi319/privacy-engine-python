"""标准文档到 YAML 配置自动生成器 / Standard Document YAML Generator.

输入一个 Markdown 格式的分类分级标准文档（如《四川省健康医疗大数据应用指南.md》），
Input a Markdown formatted classification and grading standard document,
自动解析文档中的分类树定义、等级划分矩阵与词条举例，并输出对应的全量字段参考 YAML 配置：
1. taxonomies/<standard_id>.yaml - 分类分级元数据定义 / Classification metadata definition
2. domains/<standard_id>.yaml    - 领域匹配规则包 / Domain matching rule profile
3. standards/<standard_id>.yaml  - 标准组合定义 / Standard combination definition

===================================================================================
              自动生成执行流程 / Auto-Generation Execution Flow
===================================================================================

  StandardProfileGenerator(doc_path="docs/standard/XXX.md")
    │
    ├─① parse()
    │   ├─ _extract_standard_id()    → 从“标准编号”行或文件名提取 (如 sc_health_db51)
    │   ├─ _extract_levels()         → 识别等级体系 (L1~L5 医疗 / C1~C4 金融)
    │   ├─ _extract_categories()     → 识别分类目录树 (PERSONAL_BASIC, MEDICAL_TREATMENT...)
    │   └─ _generate_rules()         → 从文档关键词正向命中生成规则
    │       ├─ 身份证件 → id_card_checksum + keyword_contains
    │       ├─ 手机号码 → regex + keyword_contains
    │       ├─ 银行卡号 → luhn_checksum + keyword_contains
    │       ├─ 敏感病种 → keyword_contains (AND 逻辑)
    │       ├─ 基因遗传 → keyword_contains (AND 逻辑)
    │       └─ 运营统计 → DowngradeRuleDef (降级规则)
    │
    ├─② generate_files(output_dir="rules")
    │   ├─ 写入 taxonomies/{id}.yaml  → DomainTaxonomy.model_dump()
    │   ├─ 写入 domains/{id}.yaml     → RuleProfile.model_dump()
    │   ├─ 写入 standards/{id}.yaml   → StandardDef.model_dump()
    │   └─ 反向校验: 重新加载并 model_validate() 确保输出合法
    │
    └─ CLI: python -m ...standard_profile_generator --doc xxx.md --output rules
===================================================================================
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from .models import CategoryDef, DomainTaxonomy, SensitivityLevelDef
from .rule_schema import DowngradeRuleDef, MatcherDef, RuleDef, RuleProfile, StandardDef


# 否定/排除语义模式：只要关键词所在语义单元出现这些短语，即视为排除。
# 注意：像“禁止/严禁/非”这类词表示数据受监管，是正向信号，不放在这里。
_EXCLUSION_PATTERNS = [
    "不包括", "不包含", "不含", "不适用", "不适用于", "不属于", "不涉及",
    "未涉及", "除外", "例外", "例外情况", "仅作", "仅作为", "仅作为示例",
    "不做", "不做分级",
]

# 应跳过的非正文章节标题关键词
_IGNORED_HEADER_KEYWORDS = [
    "参考文献", "前言", "引言", "起草说明", "规范性引用文件",
]


class StandardProfileGenerator:
    """分类分级标准文档解析与 YAML 规则包自动生成器 / Classification standard document parser and YAML Profile auto-generator.

    支持对符合国标/行标/地方标准格式的 Markdown 文档进行分析，
    抽取数据分类目录、分级定义及字段模式，自动构建全量字段的三套 YAML 参考规则。
    """

    def __init__(self, doc_path: str | Path):
        self.doc_path = Path(doc_path)
        if not self.doc_path.exists():
            raise FileNotFoundError(f"标准文档不存在: {self.doc_path}")
        self.content = self.doc_path.read_text(encoding="utf-8")

    def parse(self) -> tuple[DomainTaxonomy, RuleProfile, StandardDef]:
        """解析文档并生成元数据体系、规则 Profile 和 Standard 组合模型 / Parse document and generate taxonomy, Rule Profile, and Standard definition models."""
        standard_id = self._extract_standard_id()
        description = self._extract_description()

        levels = self._extract_levels()
        categories = self._extract_categories()

        taxonomy = DomainTaxonomy(
            domain=standard_id,
            standard_id=standard_id,
            version="1.0.0",
            description=description,
            levels=levels,
            categories=categories,
            default_level=self._determine_default_level(levels),
        )

        rules, downgrade_rules = self._generate_rules(standard_id, levels, categories)

        profile = RuleProfile(
            domain=standard_id,
            version="1.0.0",
            description=f"从文档 {self.doc_path.name} 自动抽取的分类分级规则包",
            rules=rules,
            downgrade_rules=downgrade_rules,
        )

        standard_def = StandardDef(
            standard_id=standard_id,
            description=description,
            taxonomy=standard_id,
            domains=[standard_id],
        )

        return taxonomy, profile, standard_def

    def generate_files(self, output_dir: str | Path = "rules") -> dict[str, Path]:
        """解析文档并将自动生成的 3 个 YAML 文件写入 output_dir 对应目录 / Parse document and write the 3 auto-generated YAML files to output_dir."""
        output_dir = Path(output_dir)
        taxonomy, profile, standard_def = self.parse()

        tax_dir = output_dir / "taxonomies"
        dom_dir = output_dir / "domains"
        std_dir = output_dir / "standards"

        tax_dir.mkdir(parents=True, exist_ok=True)
        dom_dir.mkdir(parents=True, exist_ok=True)
        std_dir.mkdir(parents=True, exist_ok=True)

        tax_path = tax_dir / f"{taxonomy.standard_id}.yaml"
        dom_path = dom_dir / f"{profile.domain}.yaml"
        std_path = std_dir / f"{standard_def.standard_id}.yaml"

        tax_path.write_text(
            yaml.safe_dump(taxonomy.model_dump(by_alias=True, exclude_none=True), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        dom_path.write_text(
            yaml.safe_dump(profile.model_dump(by_alias=True, exclude_none=True), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        std_path.write_text(
            yaml.safe_dump(standard_def.model_dump(by_alias=True, exclude_none=True), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        # 生成后反向校验反序列化模型合法性 (Post-generation Schema validation)
        DomainTaxonomy.model_validate(yaml.safe_load(tax_path.read_text(encoding="utf-8")))
        RuleProfile.model_validate(yaml.safe_load(dom_path.read_text(encoding="utf-8")))
        StandardDef.model_validate(yaml.safe_load(std_path.read_text(encoding="utf-8")))

        return {
            "taxonomy": tax_path,
            "domain": dom_path,
            "standard": std_path,
        }

    # ------------------------------------------------------------------
    # 内部抽词与解析辅助函数
    # ------------------------------------------------------------------

    def _extract_standard_id(self) -> str:
        """抽取标准标识符，如 DB51/T 2989 -> sc_health_db51 / Extract standard identifier."""
        match = re.search(r"标准编号[：:]\s*([^\n]+)", self.content)
        if match:
            code = match.group(1).upper()
            if "DB51" in code:
                return "sc_health_db51"
            elif "JR/T" in code or "JRT" in code or "0197" in code:
                return "jrt0197"
            elif "35273" in code:
                return "gb35273"
            elif "43697" in code:
                return "gb43697"

        name = self.doc_path.stem
        if "四川" in name or "DB51" in name:
            return "sc_health_db51"
        elif "金融" in name or "JR" in name or "0197" in name:
            return "jrt0197"
        elif "35273" in name:
            return "gb35273"
        elif "43697" in name:
            return "gb43697"
        elif "广东" in name:
            return "gd_health_db44"

        slug = re.sub(r"[^\w\-]", "_", name).lower()
        return slug or "auto_generated_standard"

    def _extract_description(self) -> str:
        """抽取标准简短描述 / Extract standard short description."""
        lines = self.content.strip().split("\n")
        title = lines[0].replace("#", "").strip() if lines else self.doc_path.stem
        match = re.search(r"标准编号[：:]\s*([^\n]+)", self.content)
        code_str = match.group(1).strip() if match else ""
        return f"{title} ({code_str})" if code_str else title

    def _extract_levels(self) -> dict[str, SensitivityLevelDef]:
        """抽取敏感度等级定义字典 / Extract sensitivity level definitions dictionary."""
        levels: dict[str, SensitivityLevelDef] = {}

        if "第1级" in self.content or "第 1 级" in self.content or "L1" in self.content:
            levels["L1"] = SensitivityLevelDef(id="L1", name="公开数据/第1级", rank=1, description="低敏感度或经脱敏的数据")
            levels["L2"] = SensitivityLevelDef(id="L2", name="内部数据/第2级", rank=2, description="机构运营生产相关数据")
            levels["L3"] = SensitivityLevelDef(id="L3", name="敏感数据/第3级", rank=3, description="个人标识与身份信息")
            levels["L4"] = SensitivityLevelDef(id="L4", name="高敏感数据/第4级", rank=4, description="敏感病种与诊疗数据")
            levels["L5"] = SensitivityLevelDef(id="L5", name="极敏感数据/第5级", rank=5, description="基因与遗传数据")
        elif re.search(r"C[1-4]\s*[级类]", self.content) or "第四级" in self.content or "JR/T 0197" in self.content:
            levels["C1"] = SensitivityLevelDef(id="C1", name="第一级（不敏感）", rank=1, description="公开金融数据")
            levels["C2"] = SensitivityLevelDef(id="C2", name="第二级（低敏感）", rank=2, description="内部使用金融数据")
            levels["C3"] = SensitivityLevelDef(id="C3", name="第三级（敏感）", rank=3, description="个人金融信息")
            levels["C4"] = SensitivityLevelDef(id="C4", name="第四级（高敏感）", rank=4, description="核心金融账户")
        else:
            levels["L1"] = SensitivityLevelDef(id="L1", name="公开数据", rank=1)
            levels["L2"] = SensitivityLevelDef(id="L2", name="内部数据", rank=2)
            levels["L3"] = SensitivityLevelDef(id="L3", name="敏感数据", rank=3)
            levels["L4"] = SensitivityLevelDef(id="L4", name="高敏感数据", rank=4)
            levels["L5"] = SensitivityLevelDef(id="L5", name="极敏感数据", rank=5)

        return levels

    def _extract_categories(self) -> dict[str, CategoryDef]:
        """抽取分类目录树 / Extract category tree."""
        categories: dict[str, CategoryDef] = {}

        if "个人基本信息" in self.content or "PERSONAL_BASIC" in self.content:
            categories["PERSONAL_BASIC"] = CategoryDef(id="PERSONAL_BASIC", name="个人基本信息数据", description="能够识别特定自然人的数据")
        if "诊疗信息" in self.content or "MEDICAL_TREATMENT" in self.content:
            categories["MEDICAL_TREATMENT"] = CategoryDef(id="MEDICAL_TREATMENT", name="诊疗信息数据", description="患者医疗服务过程产生的数据")
        if "费用信息" in self.content or "FEE_BILLING" in self.content:
            categories["FEE_BILLING"] = CategoryDef(id="FEE_BILLING", name="费用信息数据", description="医疗服务费用相关数据")
        if "公共卫生" in self.content or "PUBLIC_HEALTH" in self.content:
            categories["PUBLIC_HEALTH"] = CategoryDef(id="PUBLIC_HEALTH", name="公共卫生信息数据", description="疾病控制与公共卫生事业数据")
        if "管理信息" in self.content or "MANAGEMENT" in self.content:
            categories["MANAGEMENT"] = CategoryDef(id="MANAGEMENT", name="管理信息数据", description="反映机构运营管理状况的数据")

        if "基因" in self.content or "遗传" in self.content:
            categories["GENOMIC"] = CategoryDef(id="GENOMIC", name="基因遗传数据", parent_id="MEDICAL_TREATMENT", description="个人或家族基因/多组学检测数据")
        if "金融账户" in self.content or "银行卡" in self.content:
            categories["FINANCIAL_ACCOUNT"] = CategoryDef(id="FINANCIAL_ACCOUNT", name="金融账户数据", parent_id="PERSONAL_BASIC")

        if not categories:
            categories["GENERAL_PII"] = CategoryDef(id="GENERAL_PII", name="通用个人信息")
            categories["BUSINESS_DATA"] = CategoryDef(id="BUSINESS_DATA", name="业务数据")

        return categories

    def _determine_default_level(self, levels: dict[str, SensitivityLevelDef]) -> str:
        """确定未匹配任何规则的字段的默认等级 ID。"""
        if "L3" in levels:
            return "L3"
        elif "C3" in levels:
            return "C3"
        return list(levels.keys())[0] if levels else "L1"

    def _is_effective_positive_hit(self, keywords: list[str]) -> bool:
        """检查文档中是否存在指定关键词的有效正向命中。

        改进点：
        1. 按中文逗号、分号、句号等标点，以及转折连词（但/但是）拆分长句，
           避免转折后半句的否定误杀前半句关键词。
        2. 在每个语义单元内做整段否定判断：只要该单元含有排除/否定短语，
           单元内所有关键词均视为被否定，解决长距离否定（如“本规范不适用于基因数据”）。
        3. 把“禁止/严禁”等监管语气词从排除列表中移除，避免把敏感数据要求误判为排除。
        4. 特殊处理“非+关键词”前缀（如“非基因数据”），避免被单列的“非”误杀。
        """
        in_ignored_section = False

        for line in self.content.split("\n"):
            line_str = line.strip()
            if not line_str:
                continue

            # 检测标题层级，隔离无关章节
            if line_str.startswith("#"):
                header_title = line_str.lstrip("#").strip()
                if any(ign_kw in header_title for ign_kw in _IGNORED_HEADER_KEYWORDS):
                    in_ignored_section = True
                    continue
                else:
                    in_ignored_section = False

            if in_ignored_section:
                continue

            # 拆分为语义单元：标点 + 常见转折连词
            segments = re.split(r"[，,；;。！!\n]|但是|但", line_str)
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue

                # 如果该语义单元含有排除/否定短语，则整段视为否定
                if any(ex in segment for ex in _EXCLUSION_PATTERNS):
                    continue

                for kw in keywords:
                    start = 0
                    while True:
                        idx = segment.find(kw, start)
                        if idx == -1:
                            break
                        # 特殊前缀："非身份证" / "非基因数据"
                        if idx > 0 and segment[idx - 1] == "非":
                            start = idx + len(kw)
                            continue
                        return True
        return False

    def _generate_rules(
        self, standard_id: str, levels: dict[str, SensitivityLevelDef], categories: dict[str, CategoryDef]
    ) -> tuple[list[RuleDef], list[DowngradeRuleDef]]:
        """从标准词条举例提取特征算子并自动构建全量字段规则 / Extract features and auto-build full schema rules."""
        rules: list[RuleDef] = []
        downgrade_rules: list[DowngradeRuleDef] = []

        # 1. 身份证件特征匹配（扩展同义词 + 否定过滤）
        idcard_kws = ["身份证", "身份证件", "公民身份号码", "居民身份证", "护照号码"]
        if self._is_effective_positive_hit(idcard_kws):
            rules.append(
                RuleDef(
                    id=f"RULE_{standard_id.upper()}_IDCARD",
                    name="身份证件号码检测",
                    category="PERSONAL_BASIC" if "PERSONAL_BASIC" in categories else list(categories.keys())[0],
                    level="L3" if "L3" in levels else "C3",
                    match_logic="OR",
                    matchers=[
                        MatcherDef(target="field_value", operator="id_card_checksum", params={}),
                        MatcherDef(target="field_name", operator="keyword_contains", params={"keywords": ["idcard", "sfz", "identity", "id_card", "身份证"]}),
                    ],
                    priority=90,
                )
            )

        # 2. 手机号码特征匹配（扩展同义词 + 否定过滤）
        phone_kws = ["电话", "手机", "联系电话", "手机号码", "移动电话", "固定电话"]
        if self._is_effective_positive_hit(phone_kws):
            rules.append(
                RuleDef(
                    id=f"RULE_{standard_id.upper()}_PHONE",
                    name="手机号码检测",
                    category="PERSONAL_BASIC" if "PERSONAL_BASIC" in categories else list(categories.keys())[0],
                    level="L3" if "L3" in levels else "C3",
                    match_logic="OR",
                    matchers=[
                        MatcherDef(target="field_value", operator="regex", params={"pattern": r"^1[3-9]\d{9}$"}),
                        MatcherDef(target="field_name", operator="keyword_contains", params={"keywords": ["mobile", "phone", "cell", "电话", "手机"]}),
                    ],
                    priority=80,
                )
            )

        # 3. 支付金融账户特征匹配（扩展同义词 + 否定过滤）
        bankcard_kws = ["金融账户", "支付卡号", "银行卡号", "银行账号", "结算账户", "资金账户"]
        if self._is_effective_positive_hit(bankcard_kws):
            target_lvl = "C4" if "C4" in levels else "L3"
            rules.append(
                RuleDef(
                    id=f"RULE_{standard_id.upper()}_BANKCARD",
                    name="支付金融账户检测",
                    category="FINANCIAL_ACCOUNT" if "FINANCIAL_ACCOUNT" in categories else "PERSONAL_BASIC",
                    level=target_lvl,
                    match_logic="OR",
                    matchers=[
                        MatcherDef(target="field_value", operator="luhn_checksum", params={"min_length": 13, "max_length": 19}),
                        MatcherDef(target="field_name", operator="keyword_contains", params={"keywords": ["bankcard", "cardno", "card_no", "bank_card", "支付卡号"]}),
                    ],
                    priority=85,
                )
            )

        # 4. 敏感病种特征匹配（扩展同义词 + 否定过滤）
        disease_kws = ["艾滋病", "性病", "精神病", "传染病", "恶性肿瘤", "精神分裂症"]
        if self._is_effective_positive_hit(disease_kws):
            rules.append(
                RuleDef(
                    id=f"RULE_{standard_id.upper()}_DISEASE",
                    name="敏感病种检测",
                    category="MEDICAL_TREATMENT" if "MEDICAL_TREATMENT" in categories else list(categories.keys())[0],
                    level="L4" if "L4" in levels else "L3",
                    match_logic="AND",
                    matchers=[
                        MatcherDef(
                            target="field_name",
                            operator="keyword_contains",
                            params={"keywords": ["hiv", "aids", "std", "syphilis", "psychiatric", "schizophrenia", "艾滋病", "性病", "精神病"]},
                        )
                    ],
                    priority=95,
                )
            )

        # 5. 个人遗传基因特征匹配（扩展同义词 + 否定过滤）
        genomic_kws = ["基因", "染色体", "地中海贫血", "基因组", "DNA序列", "分子遗传"]
        if self._is_effective_positive_hit(genomic_kws):
            rules.append(
                RuleDef(
                    id=f"RULE_{standard_id.upper()}_GENOMIC",
                    name="个人遗传基因数据检测",
                    category="GENOMIC" if "GENOMIC" in categories else "MEDICAL_TREATMENT",
                    level="L5" if "L5" in levels else "L4",
                    match_logic="AND",
                    matchers=[
                        MatcherDef(
                            target="field_name",
                            operator="keyword_contains",
                            params={"keywords": ["gene", "genomic", "brca", "tp53", "snp", "cnv", "chromosome", "thalassemia", "基因", "染色体", "地中海贫血"]},
                        )
                    ],
                    priority=100,
                )
            )

        downgrade_rules.append(
            DowngradeRuleDef(
                id=f"DOWN_{standard_id.upper()}_OPS",
                name="运营统计指标降级",
                keywords=["turnover", "inventory", "device_usage", "开机次数", "门诊人次", "运行时间"],
                level="L2" if "L2" in levels else "C2",
                category="MANAGEMENT" if "MANAGEMENT" in categories else list(categories.keys())[0],
                force_suppress=False,              # 默认显式标注是否开启强行覆盖
                max_force_suppress_level="",       # 默认显式标注强行覆盖上限等级 (空=使用自身 level)
                exempt_rules=[],                   # 默认显式标注压制豁免例外名单 (空=全额压制)
            )
        )

        return rules, downgrade_rules


def main():
    """CLI 工具命令行入口 / CLI tool entry point.

    Usage:
        python -m engine.dynclassification.standard_profile_generator \\
            --doc docs/standard/四川省健康医疗大数据应用指南.md \\
            --output rules
    """
    parser = argparse.ArgumentParser(description="从分类分级标准 Markdown 文档生成全量参考字段的 YAML 配置文件")
    parser.add_argument("--doc", required=True, help="标准 Markdown 文档路径，例如 docs/standard/四川省健康医疗大数据应用指南.md")
    parser.add_argument("--output", default="rules", help="YAML 输出规则根目录，默认 rules/")

    args = parser.parse_args()

    doc_parser = StandardProfileGenerator(args.doc)
    generated = doc_parser.generate_files(args.output)

    print("=== 自动生成 YAML 配置文件成功 ===")
    for key, path in generated.items():
        print(f"[{key.upper()}] -> {path}")


if __name__ == "__main__":
    main()
