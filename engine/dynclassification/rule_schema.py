"""声明式规则 Profile 数据模型 / Declarative Rule Profile Schema.

本模块定义了动态分类分级系统中所有规则与配置的 Pydantic v2 强类型数据结构契约（Data Schema Contract）。
包含匹配器（MatcherDef）、单条规则（RuleDef）、降级与压制规则（DowngradeRuleDef）、
记录级复合规则（CompositeRuleDef）、领域规则包（RuleProfile）以及组合标准（StandardDef）。

===================================================================================
                       动态分类分级架构与 rule_schema 的位置
               Dynamic Classification Architecture & rule_schema Role
===================================================================================

 [ 外部配置源 / External Configurations ]
   ├── YAML Profiles (rules/domains/*.yaml)     -> 领域规则包 (general-pii, medical, finance...)
   └── YAML Standards (rules/standards/*.yaml) -> 行业合规标准 (sc_health_db51, jrt0197...)
                           │
                           │  YAML 反序列化 & Schema 强类型校验 (Pydantic Validation)
                           ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                       ★ rule_schema.py (数据契约层 / Data Schemas)              │
 │                                                                                 │
 │  将文本配置转换为 Pydantic v2 内存数据模型:                                      │
 │   ├── MatcherDef        : 匹配算子元数据 (target: field_name/field_value, op...)  │
 │   ├── RuleDef           : 字段级分类规则 (id, category, level, matchers...)      │
 │   ├── DowngradeRuleDef  : 降级/压制规则 (keywords, level, force_suppress...)       │
 │   ├── CompositeRuleDef  : 记录级组合升级规则 (patterns, min_matches, target_level) │
 │   ├── RuleProfile       : 领域规则包容器 (domain, rules, downgrade_rules...)     │
 │   └── StandardDef       : 标准集组合与覆盖 (domains, rule_overrides, 追加规则)    │
 └────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          │  读取模型构建规则集 (Pass Model Objects)
                                          ▼
 [ 规则加载与合并器 / Profile Loader ] (profile_loader.py)
   ├── 加载 StandardDef 引用的所有领域规则包 (RuleProfile)
   ├── 执行规则级覆盖 (rule_overrides)
   │   注意：StandardDef.global_params（旧 alias: overrides）当前无消费方，
   │   属于死配置；profile_loader 加载时会记录 warning 提示其不会生效。
   └── 输出合并后的规则集合 list[RuleDef] / list[DowngradeRuleDef]
                           │
                           │  初始化与引擎加载 (Initialize Runtime Engines)
                           ▼
 [ 规则评估引擎 / Execution Engines ] (engine.py / composite.py)
   ├── ConfigurableRuleEngine (评估字段规则 & 执行降级压制)
   │      │
   │      ├─► 调度 OperatorRegistry 算子 (regex, keyword_contains, checksum...)
   │      └─► 计算并输出字段标签 list[SecurityTag]
   └── CompositeRuleEngine (评估记录级多字段组合升级规则)
                           │
                           │  3层分类漏斗调度 (3-Layer Classification Funnel)
                           ▼
 [ 3层分类分级漏斗 / Funnel System ] (funnel.py)
   ├── Layer-1: 规则引擎 (Rule Engine - rule_schema 驱动) ---> 命中直接输出高置信度 Tag
   ├── Layer-2: 实体识别 (Small-NER Engine) -----------------> 未命中/低置信度时提取数据实体
   └── Layer-3: 本地大模型 (Local LLM/VLM Adapter) -----------> 疑难/长尾字段做自然语言语义仲裁
===================================================================================
"""

from __future__ import annotations

from typing import Any, Optional

# Pydantic v2 imports: BaseModel for schema definition, ConfigDict for model config,
# Field for metadata annotations (description, alias, default).
# Pydantic v2 导入：BaseModel 用于定义数据模型，ConfigDict 用于模型配置，
# Field 用于字段元数据注解（description 描述、alias 别名、default 默认值）。
from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# 匹配器定义 / Matcher Definition
# ===========================================================================


class MatcherDef(BaseModel):
    """单个匹配器定义。 / Single Matcher Definition.

    描述对字段名或字段值执行何种算子匹配。 / Describes what operator matching to perform on field name or field value.
    一个规则可包含多个匹配器，通过 match_logic 决定组合逻辑。 / A rule can contain multiple matchers, and the combination logic is determined by match_logic.
    """

    # Allow instantiation using both Python field names and JSON/YAML alias names.
    # 允许同时使用 Python 字段名与 JSON/YAML 别名进行实例化。
    model_config = ConfigDict(populate_by_name=True)

    # Matching target: either "field_name" (match against column name) or
    # "field_value" (match against the actual data value).
    # 匹配目标："field_name"（匹配列名）或 "field_value"（匹配实际数据值）。
    target: str = Field(description="匹配目标: 'field_name' | 'field_value'")
    # Operator name registered in OperatorRegistry, e.g. 'regex', 'keyword_contains',
    # 'id_card_checksum', 'icd10_range', 'luhn_checksum', etc.
    # 在 OperatorRegistry 中注册的算子名称，如 'regex'、'keyword_contains'、
    # 'id_card_checksum'、'icd10_range'、'luhn_checksum' 等。
    operator: str = Field(description="算子名称: 'regex' | 'keyword_contains' | 'id_card_checksum' 等")
    # Operator-specific parameters dict (e.g. {"pattern": "..."} for regex,
    # {"keywords": [...]} for keyword_contains, {"intervals": [...]} for icd10_range).
    # 算子专属参数字典（如 regex 的 {"pattern": "..."}、keyword_contains 的
    # {"keywords": [...]}、icd10_range 的 {"intervals": [...]}）。
    params: dict[str, Any] = Field(default_factory=dict, description="算子参数（如 pattern、keywords、intervals）")


# ===========================================================================
# 规则定义 / Rule Definitions
# ===========================================================================


class RuleDef(BaseModel):
    """单条声明式规则定义。 / Single Declarative Rule Definition.

    规则是分类引擎的最小执行单元，包含一组匹配器和命中后的标签信息。 / A rule is the smallest execution unit of the classification engine, containing a set of matchers and label info upon hit.
    """

    # Allow both snake_case and camelCase field access.
    # 允许同时使用 snake_case 与 camelCase 字段名访问。
    model_config = ConfigDict(populate_by_name=True)

    # Unique rule identifier, used for metrics labeling and audit trail.
    # 规则唯一标识，用于指标打标与审计追踪。
    id: str = Field(description="规则唯一标识，如 'RULE_PII_IDCARD'")
    # Human-readable rule name for display in management UIs.
    # 人类可读的规则名称，用于管理界面展示。
    name: str = Field(default="", description="规则名称（人类可读）")
    # Category ID assigned to the field when this rule hits (e.g. 'PII_ID_CARD').
    # 规则命中时赋予字段的分类类别 ID（如 'PII_ID_CARD'）。
    category: str = Field(description="命中后的分类类别 ID")
    # Sensitivity level ID assigned when this rule hits (e.g. 'L3', 'C4').
    # 规则命中时赋予的敏感度等级 ID（如 'L3'、'C4'）。
    level: str = Field(description="命中后的敏感度等级 ID")
    # List of matchers to evaluate; combined via match_logic (AND/OR).
    # 待评估的匹配器列表，通过 match_logic（AND/OR）决定组合逻辑。
    matchers: list[MatcherDef] = Field(default_factory=list, description="匹配器列表")
    # Logic for combining multiple matchers: "AND" = all must hit, "OR" = any one hit suffices.
    # 多匹配器组合逻辑："AND" 表示全部命中，"OR" 表示任一命中即可。
    match_logic: str = Field(default="AND", description="多匹配器逻辑: 'AND'(全部命中) | 'OR'(任一命中)")
    # Execution priority: higher values execute first. Used for sorting after merge.
    # 执行优先级：数值越大越先执行，用于合并后排序。
    priority: int = Field(default=0, description="优先级（数值越大越先执行）")
    # Enable/disable toggle: disabled rules are skipped during profile merge.
    # 启用/禁用开关：被禁用的规则在 Profile 合并时会被跳过。
    enabled: bool = Field(default=True, description="是否启用")
    # Extension tags for custom metadata (key-value pairs, not interpreted by engine).
    # 扩展标签，用于自定义元数据（键值对，引擎不解释）。
    tags: dict[str, str] = Field(default_factory=dict, description="扩展标签（自定义元数据）")


class DowngradeRuleDef(BaseModel):
    """降级规则定义。 / Downgrade Rule Definition.

    当字段名匹配指定关键词时，将等级降级到目标等级。 / Downgrades the level to the target level when the field name matches specified keywords.
    典型场景：公开字段降为 L1，运营统计字段降为 L2。 / Typical scenarios: public fields downgraded to L1, operation stat fields downgraded to L2.

    强制覆盖模式（force_suppress=true） / Force Override Mode (force_suppress=true):
        默认情况下，降级规则仅作为"兜底归属"——在无普通规则命中时替代默认等级。 / By default, downgrade rules only serve as "fallback" - replacing default level when no normal rules hit.
        当设置 force_suppress=true 后，降级规则可强制压制 rank <= max_force_suppress_level 的 / When force_suppress=true, downgrade rules can forcefully suppress normal rule tags with rank <= max_force_suppress_level,
        普通规则标签，解决宽泛规则误中运营/公开字段的问题。 / solving the issue of broad rules falsely hitting operation/public fields.

        执行流程 / Execution Flow:
        ┌──────────────────────────────────────────────────────────────┐
        │  force_suppress=false (默认/Default):                          │
        │    降级标签 + 普通标签 → 取 max → 降级无效（仅兜底）                 │
        │                                                              │
        │  force_suppress=true:                                        │
        │    先移除 rank <= cap 的普通标签 → 再取 max → 降级生效             │
        └──────────────────────────────────────────────────────────────┘
    """

    # Allow instantiation using both Python field names and JSON/YAML alias names.
    # 允许同时使用 Python 字段名与 JSON/YAML 别名进行实例化。
    model_config = ConfigDict(populate_by_name=True)

    # Unique identifier for this downgrade rule.
    # 降级规则的唯一标识。
    id: str = Field(description="规则唯一标识")
    # Human-readable name for display purposes.
    # 人类可读的名称，用于展示。
    name: str = Field(default="", description="规则名称")
    # Keywords to match against normalized field names (case-insensitive, underscore-stripped).
    # 用于匹配归一化字段名（不区分大小写、去除下划线）的关键词。
    keywords: list[str] = Field(default_factory=list, description="关键词列表")
    # Target level to downgrade to (e.g. 'L1' for public fields, 'L2' for operational stats).
    # 降级目标等级（如公开字段降为 'L1'，运营统计字段降为 'L2'）。
    level: str = Field(description="降级目标等级")
    # Category assigned when downgrade rule fires.
    # 降级规则触发时赋予的分类类别。
    category: str = Field(description="分类类别")
    # What to match against (currently only 'field_name' is supported).
    # 匹配目标（当前仅支持 'field_name'）。
    match_target: str = Field(default="field_name", description="匹配目标")
    # Whether this downgrade rule can forcibly suppress normal rule tags.
    # When true, normal tags with rank <= max_force_suppress_level's rank are removed.
    # Default false: backward-compatible, downgrade tag only serves as fallback.
    # 该降级规则是否可强制压制普通规则标签。
    # 为 true 时，rank <= max_force_suppress_level 等级的普通标签会被移除。
    # 默认 false：向后兼容，降级标签仅作兜底归属。
    # 注：alias="override" 保留旧 YAML key 的向后兼容性。
    force_suppress: bool = Field(default=False, alias="override", description="是否启用强制覆盖（压制低等级普通规则标签）")
    # Maximum level this override can suppress (inclusive).
    # E.g. 'L3' means only normal tags with rank <= rank('L3') can be suppressed.
    # Empty string defaults to using this rule's own 'level' field as the cap.
    # 该覆盖规则可压制的最高等级（含）。
    # 如 'L3' 表示仅压制 rank <= rank('L3') 的普通标签。
    # 空字符串时默认使用本规则自身的 'level' 字段作为上限。
    max_force_suppress_level: str = Field(default="", description="覆盖等级上限（空=使用 level 字段）")
    # Fine-grained suppression exemption list: normal rules whose ID or wildcard pattern
    # matches any entry in this list are EXEMPT from suppression (protected and preserved).
    # Empty list (default) = NO exemptions, all eligible normal tags with rank <= max_force_suppress_level are suppressed.
    # Non-empty = rules listed here (or matching wildcards like '*_EXACT') are EXEMPT and protected from suppression.
    # 压制豁免例外名单：列表中的规则 ID / 通配符表达式为豁免例外（受保护、绝对不被压制）。
    # 核心用途：默认按等级全额压制所有误报标签；仅在需要保护极少数精准检验规则（如身份证正则）时手写例外。
    # - 空列表（默认）= 没有例外，按等级区间全额压制。 / Empty list (default) = NO exemptions, all eligible tags suppressed.
    # - 非空 = 列表中列出（或匹配通配符）的规则 ID 属于例外，保护保留。 / Non-empty = listed/wildcard rules are EXEMPT.
    # 注：alias="exclude_rules" 提供 YAML 极简语义别名。
    exempt_rules: list[str] = Field(
        default_factory=list,
        alias="exclude_rules",
        description="压制豁免例外名单: 列表中的规则 ID / 通配符豁免保护（空=没有例外全额压制）",
    )


class CompositeRuleDef(BaseModel):
    """复合规则定义（记录级）。 / Composite Rule Definition (Record-Level).

    当一条记录中同时有 >= min_matches 个字段匹配指定的字段模式时， / When a record has >= min_matches fields matching specified field patterns simultaneously,
    将整条记录的敏感度升级为 target_level。 / upgrades the sensitivity of the entire record to target_level.
    """

    # Allow instantiation using both Python field names and JSON/YAML alias names.
    # 允许同时使用 Python 字段名与 JSON/YAML 别名进行实例化。
    model_config = ConfigDict(populate_by_name=True)

    # Unique identifier for this composite rule.
    # 复合规则的唯一标识。
    id: str = Field(description="规则唯一标识")
    # Human-readable name.
    # 人类可读的名称。
    name: str = Field(default="", description="规则名称")
    # List of regex patterns to match against field names in a record.
    # Each pattern that matches at least one field counts as one "hit".
    # 用于匹配记录中字段名的正则表达式列表。
    # 每个至少匹配到一个字段的正则计为一次“命中”。
    field_patterns: list[str] = Field(default_factory=list, description="字段名正则列表")
    # Minimum number of pattern hits required to trigger this composite rule.
    # E.g. min_matches=3 means at least 3 different patterns must match fields.
    # 触发本复合规则所需的最低命中数。
    # 如 min_matches=3 表示至少需有 3 个不同的正则匹配到字段。
    min_matches: int = Field(default=1, description="最低匹配数")
    # The upgraded sensitivity level when this composite rule fires (e.g. 'L5').
    # 复合规则触发时升级到的敏感度等级（如 'L5'）。
    target_level: str = Field(description="升级目标等级")
    # Category assigned when composite rule fires.
    # 复合规则触发时赋予的分类类别。
    category: str = Field(description="分类类别")


# ===========================================================================
# 脱敏治理策略 / Redaction Treatment Strategy
# ===========================================================================


class RedactionStrategyDef(BaseModel):
    """脱敏治理策略定义（YAML 可配置）。 / Redaction Treatment Strategy Definition (YAML-configurable).

    定义哪些高敏疾病范畴需要彻底抹平（Purge），哪些适用范畴化泛化（Generalization）。
    Defines which high-sensitivity disease categories require purge-only treatment
    and which are eligible for categorical generalization.

    该模型将原本硬编码在 rules.py 中的 L4/L5 治理策略提升为 YAML 可配置项，
    使运维/合规团队可在不修改代码的情况下调整治理策略。
    """

    model_config = ConfigDict(populate_by_name=True)

    # 彻底抹平范畴列表（严禁泛化，零痕迹擦除）。
    # Purge-only categories (strictly no generalization, zero-trace erasure).
    # 对应 rules.py 中 L5_TERMS_MAP / L4_TERMS_MAP 的键（如 HIV_AIDS, PSYCHIATRIC_DISORDER, STD_VENEREAL）。
    purge_categories: list[str] = Field(
        default_factory=list,
        description="彻底抹平范畴列表（如 HIV_AIDS, PSYCHIATRIC_DISORDER, STD_VENEREAL）",
    )
    # 范畴化泛化范畴列表（重构为低敏通用大类疾病）。
    # Generalization-eligible categories (rewrite to lower-sensitivity general disease classes).
    # 对应 rules.py 中 L4_TERMS_MAP 的键（如 MALIGNANT_NEOPLASM, HEPATITIS_VIRUS, SEVERE_ORGAN_DAMAGE）。
    generalization_categories: list[str] = Field(
        default_factory=list,
        description="范畴化泛化范畴列表（如 MALIGNANT_NEOPLASM, HEPATITIS_VIRUS）",
    )
    # 替换标签映射：范畴名 -> 抽象标签名（防止替换文本泄露原始敏感词）。
    # Replacement label mapping: category -> abstract label (prevents leaking sensitive terms in replacements).
    # 如 {"HIV_AIDS": "IMMUNODEFICIENCY"} 使替换标签为 [L5-IMMUNODEFICIENCY-SENSITIVE-MASKED] 而非 [L5-HIV_AIDS-...]。
    replacement_labels: dict[str, str] = Field(
        default_factory=dict,
        description="替换标签映射: 范畴名 -> 抽象标签名",
    )


# ===========================================================================
# 规则 Profile 与标准组合 / Rule Profile & Standard Definition
# ===========================================================================


class RuleProfile(BaseModel):
    """规则 Profile 完整定义（一个领域包）。 / Complete Rule Profile Definition (A Domain Pack).

    一个 RuleProfile 对应一个行业领域的全部规则集合， / A RuleProfile corresponds to the entire rule set of an industry domain,
    包含普通规则、降级规则和复合规则三类。 / containing three types: normal rules, downgrade rules, and composite rules.
    """

    # Allow instantiation using both Python field names and JSON/YAML alias names.
    # 允许同时使用 Python 字段名与 JSON/YAML 别名进行实例化。
    model_config = ConfigDict(populate_by_name=True)

    # Domain identifier this profile belongs to (e.g. 'medical', 'finance', 'general-pii').
    # 本 Profile 所属的领域标识（如 'medical'、'finance'、'general-pii'）。
    domain: str = Field(description="所属领域，如 'medical', 'finance', 'general-pii'")
    # Semantic version of this rule pack for audit trail.
    # 本规则包的语义化版本号，用于审计追踪。
    version: str = Field(default="1.0.0", description="版本号")
    # Human-readable description of this domain pack.
    # 本领域包的人类可读说明。
    description: str = Field(default="", description="领域包说明")
    # Optional: specify the default taxonomy this domain should be validated against.
    # If not provided, falls back to "default" taxonomy.
    # 可选：指定本领域应校验所依据的默认分类体系。
    # 未提供时回退到 "default" 分类体系。
    default_taxonomy: Optional[str] = Field(
        default=None, description="默认关联的分类体系（用于单领域校验）"
    )
    # Primary classification rules evaluated per-field.
    # 按字段评估的主要分类规则。
    rules: list[RuleDef] = Field(default_factory=list, description="普通规则列表")
    # Downgrade rules that lower sensitivity for specific field patterns.
    # 针对特定字段模式降低敏感度的降级规则。
    downgrade_rules: list[DowngradeRuleDef] = Field(default_factory=list, description="降级规则列表")
    # Composite rules evaluated at record level (multi-field combination).
    # 在记录级评估的复合规则（多字段组合）。
    composite_rules: list[CompositeRuleDef] = Field(default_factory=list, description="复合规则列表")
    # Optional: redaction treatment strategy (which disease categories to purge vs generalize).
    # Only used by domain profiles that include medical treatment redaction rules (e.g. 'medical').
    # 可选：脱敏治理策略（哪些疾病范畴需彻底抹平 vs 范畴化泛化）。
    # 仅用于包含医疗脱敏规则的领域包（如 'medical'）。
    redaction_strategy: Optional[RedactionStrategyDef] = Field(
        default=None, description="脱敏治理策略（抹平 vs 泛化范畴配置）"
    )


class StandardDef(BaseModel):
    """标准组合定义。 / Standard Combination Definition.

    一个标准 = 多个领域包组合 + 参数覆盖 + 规则级覆盖 + 追加规则。 / A standard = multiple domain pack combinations + parameter overrides + rule-level overrides + appended rules.
    例如 sc_health_db51 = general-pii + medical + 四川指南特有规则。 / E.g., sc_health_db51 = general-pii + medical + Sichuan guide-specific rules.
    """

    # Allow instantiation using both Python field names and JSON/YAML alias names.
    # 允许同时使用 Python 字段名与 JSON/YAML 别名进行实例化。
    model_config = ConfigDict(populate_by_name=True)

    # Standard identifier (e.g. 'sc_health_db51', 'jrt0197', 'gbt35273').
    # 标准标识（如 'sc_health_db51'、'jrt0197'、'gbt35273'）。
    standard_id: str = Field(description="标准标识，如 'sc_health_db51', 'jrt0197'")
    # Human-readable description of this standard combination.
    # 本标准组合的人类可读说明。
    description: str = Field(default="", description="标准说明")
    # Reference to the taxonomy YAML file name (without .yaml extension).
    # 引用的 taxonomy YAML 文件名（不含 .yaml 扩展名）。
    taxonomy: str = Field(default="default", description="引用的 taxonomy 文件名")
    # List of domain profile names to combine (loaded from domains/ directory).
    # 待组合的领域 Profile 名称列表（从 domains/ 目录加载）。
    domains: list[str] = Field(default_factory=list, description="组合的领域包列表")
    # Global parameter adjustments applied to all rules in this standard.
    # E.g. {"default_level": "C3"} overrides the engine's default level.
    # 应用于本标准下所有规则的全局参数调整（如 default_level）。
    # 注意：当前为预留字段，profile_loader 无消费方（死配置），
    # 加载含非空 global_params 的标准时会记录 warning 提示其不会生效。
    # 注：alias="overrides" 保留旧 YAML key 的向后兼容性。
    global_params: dict[str, Any] = Field(default_factory=dict, alias="overrides", description="全局参数覆盖（预留字段，当前未生效）")
    # Per-rule attribute overrides: {rule_id: {field_name: new_value}}.
    # Allows a standard to adjust specific rules' level/category/priority.
    # 规则级属性覆盖：{rule_id: {field_name: new_value}}。
    # 允许标准调整特定规则的 level/category/priority。
    rule_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="规则级覆盖: {rule_id: {field: value}}"
    )
    # Additional rules appended to the combined profile (standard-specific).
    # 追加到组合 Profile 的额外规则（标准专属）。
    extra_rules: list[RuleDef] = Field(default_factory=list, description="追加规则")
    # Additional downgrade rules appended to the combined profile.
    # 追加到组合 Profile 的额外降级规则。
    extra_downgrade_rules: list[DowngradeRuleDef] = Field(
        default_factory=list, description="追加降级规则"
    )
    # Additional composite rules appended to the combined profile.
    # 追加到组合 Profile 的额外复合规则。
    extra_composite_rules: list[CompositeRuleDef] = Field(
        default_factory=list, description="追加复合规则"
    )