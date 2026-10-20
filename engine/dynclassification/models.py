"""动态分类分级数据模型 / Dynamic Classification Data Models.

定义分类体系元数据（Taxonomy）、敏感度等级、分类类别、安全标签等核心模型。 / Defines core models for taxonomy metadata, sensitivity levels, classification categories, and security tags.
所有模型均为 Pydantic v2 BaseModel，支持 YAML/JSON 序列化与校验。 / All models are Pydantic v2 BaseModels, supporting YAML/JSON serialization and validation.

本模块完全独立于旧分类引擎（privacy/classification/），无交叉依赖。 / This module is completely independent of the old classification engine (privacy/classification/) with no cross-dependencies.

===================================================================================
              数据模型层次与在分类流程中的位置 / Data Model Hierarchy
===================================================================================

  YAML 配置文件                         运行时数据流
  (rules/**/*.yaml)                   (Request → Response)
       │                                      │
       │  ProfileLoader 反序列化                │  service.classify_field()
       ▼                                      ▼
  ┌────────────────────────┐            ┌──────────────────────────────┐
  │  DomainTaxonomy        │            │  FieldClassificationResult   │
  │  ├── levels (L1~L5)    │──引用───▶   │  ├── tags: list[SecurityTag] │
  │  ├── categories        │            │  ├── final_level: str        │
  │  ├── confidence_policy │            │  ├── confidence: float       │
  │  └── default_level     │            │  ├── engine_layer: str       │
  └────────────────────────┘            │  └── sanitized_value: str    │
       │                                └──────────────────────────────┘
       │ 被引用                                  │ 包含
       ▼                                        ▼
  ┌───────────────────────┐    ┌────────────────────────────────┐
  │  SensitivityLevelDefv │    │  SecurityTag (单次规则命中)       │
  │  id / name / rankv    │    │  level / category / confidence │
  └───────────────────────┘    │  source_engine / rule_id       │
       ▲                       │  is_override / is_downgrade    │
       │ rank 比较              └────────────────────────────────┘
       │                                │
  ┌───────────────────────┐             │ 聚合
  │  ConfidencePolicy     │             ▼
  │  conflict_confidence  │    ┌──────────────────────────────┐
  │  llm_threshold        │    │  RecordClassificationResult  │
  │  enable_ner / llm     │    │  (多字段 → 记录级)             │
  └───────────────────────┘    │      ▼                       │
                               │  TableClassificationResult   │
                               │  (多记录 → 表级)               │
                               └──────────────────────────────┘
  模型依赖方向（从上到下）：
  ConfidencePolicy → SensitivityLevelDef → DomainTaxonomy → SecurityTag → FieldResult → RecordResult → TableResult → AuditInfo → ClassificationResponse
===================================================================================
"""

from __future__ import annotations

# datetime/timezone for generating UTC timestamps in audit info
from datetime import datetime, timezone
import os
from typing import Optional

# Pydantic v2: BaseModel for schema, ConfigDict for model settings, Field for annotations
from pydantic import BaseModel, ConfigDict, Field

from ..observability.logging_config import get_logger

logger = get_logger(__name__)


# ===========================================================================
# 引擎层级与置信度策略 / Engine Layer & Confidence Policy
# ===========================================================================


class EngineLayer:
    """分类引擎层级常量。 / Classification engine layer constants.

    标识分类结果由哪一层引擎产出，用于审计追踪和结果溯源。 / Identifies which engine layer produced the classification result, for audit trails and result tracing.
    三层漏斗架构：规则引擎 → Small-NER → LLM，逐层递进。 / 3-layer funnel architecture: Rule Engine → Small-NER → LLM, progressing layer by layer.
    """

    L1_RULE = "L1_RULE"            # 第一层：可配置规则引擎（字段名 + 值模式匹配）
    L2_SMALL_NER = "L2_SMALL_NER"  # 第二层：小型 NER 模型（ONNX 实体识别）
    L3_LLM = "L3_LLM"             # 第三层：本地大语言模型（多模态分类/仲裁）


class ConfidencePolicy(BaseModel):
    """置信度衰减与 LLM 触发策略配置。 / Confidence Decay & LLM Trigger Policy Configuration.

    定义多层引擎之间的冲突判定规则、置信度衰减系数、 / Defines conflict detection rules between multi-layer engines, confidence decay coefficients,
    以及触发 Layer-2 NER 和 Layer-3 LLM 的置信度阈值。 / and confidence thresholds for triggering Layer-2 NER and Layer-3 LLM.
    支持从环境变量进行全局运维配置控制。
    """

    model_config = ConfigDict(populate_by_name=True)

    # Confidence score assigned when a normal rule and downgrade rule conflict,
    # and LLM arbitration is disabled or unavailable.
    conflict_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        alias="conflictConfidence",
        description="冲突时的降级置信度",
    )
    # Whether to flag for human review when conflict occurs.
    conflict_needs_review: bool = Field(
        default=True,
        alias="conflictNeedsReview",
        description="冲突时是否标记人工复核",
    )
    # Whether to invoke LLM arbitration when conflict is detected.
    enable_llm_arbitration: bool = Field(
        default_factory=lambda: os.environ.get("PRIVACY_LLM_ENABLE_ARBITRATION", "true").lower() == "true",
        alias="enableLlmArbitration",
        description="是否启用 LLM 仲裁（默认 true，需 ML 镜像，支持 PRIVACY_LLM_ENABLE_ARBITRATION 环境变量）",
    )
    # LLM trigger threshold: invoke LLM when confidence falls below this value.
    llm_confidence_threshold: float = Field(
        default_factory=lambda: float(os.environ.get("PRIVACY_LLM_CONFIDENCE_THRESHOLD", "0.75")),
        ge=0.0, le=1.0,
        alias="llmConfidenceThreshold",
        description="LLM 触发阈值（置信度低于此值时触发，默认 0.75，支持 PRIVACY_LLM_CONFIDENCE_THRESHOLD 环境变量）",
    )
    # Whether to enable Layer-2 NER entity extraction.
    enable_ner: bool = Field(
        default_factory=lambda: os.environ.get("PRIVACY_NER_ENABLE", "false").lower() == "true",
        alias="enableNer",
        description="是否启用 NER 层",
    )
    # Whether to explicitly enable Layer-3 LLM (regardless of conflict).
    enable_llm: bool = Field(
        default_factory=lambda: os.environ.get("PRIVACY_LLM_ENABLE", "false").lower() == "true",
        alias="enableLlm",
        description="是否显式启用 LLM 层",
    )
    # Whether to automatically trigger Layer-3 Multimodal LLM when image/DICOM is detected.
    auto_llm_on_image: bool = Field(
        default_factory=lambda: os.environ.get("PRIVACY_LLM_AUTO_ON_IMAGE", "true").lower() == "true",
        alias="autoLlmOnImage",
        description="检测到图片病例或图像字段时是否自动强制触发多模态 LLM 层（默认 true）",
    )
    # NER trigger threshold: invoke NER when current level rank <= this value.
    # Default 3 means NER triggers when field is classified at rank 3 or below.
    # For C1~C4 systems (4 levels), set to 2 to limit NER to C1/C2 only.
    ner_trigger_max_rank: int = Field(
        default=3, ge=0,
        alias="nerTriggerMaxRank",
        description="NER 触发阈值：当前等级 rank <= 此值时触发 NER（默认 3）",
    )
    # Minimum tag confidence to participate in final level resolution.
    # Tags below this threshold are preserved for audit but excluded from max_level calculation.
    min_tag_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        alias="minTagConfidence",
        description="参与最终等级裁定的最低标签置信度（低于此值的标签仅作审计记录，不影响等级）",
    )


# ===========================================================================
# 分类体系元数据模型 / Taxonomy Metadata Models
# ===========================================================================


class SensitivityLevelDef(BaseModel):
    """动态敏感度等级定义。 / Dynamic Sensitivity Level Definition.

    替代硬编码的 SensitivityLevel 枚举，支持任意等级体系。 / Replaces hardcoded SensitivityLevel enum, supporting arbitrary level systems.
    不同行业可使用不同等级标识（L1~L5 / C1~C4 / 1~4级）。 / Different industries can use different level identifiers.
    """

    # Allow population by both field name and alias for flexible YAML/JSON input.
    model_config = ConfigDict(populate_by_name=True)

    # Unique level identifier string, e.g. 'L1', 'C4', 'LEVEL_3'.
    id: str = Field(description="级别唯一标识，如 'L1', 'C4', 'LEVEL_3'")
    # Display name shown in UIs and reports, e.g. '高敏感数据'.
    name: str = Field(description="显示名称，如 '高敏感数据'")
    # Numeric rank for comparison: higher rank = more sensitive.
    # Used by DomainTaxonomy.max_level() to determine the highest severity.
    rank: int = Field(description="排序权重，数值越大越敏感（用于 max_level 比较）")
    # Optional human-readable explanation of what this level means.
    description: Optional[str] = Field(default=None, description="等级说明")


class CategoryDef(BaseModel):
    """动态分类类别定义。 / Dynamic Category Definition.

    替代硬编码的 BusinessCategory 枚举，支持多级分类树结构。 / Replaces hardcoded BusinessCategory enum, supporting multi-level category tree structures.
    通过 parent_id 建立父子关系。 / Establishes parent-child relationships via parent_id.
    """

    model_config = ConfigDict(populate_by_name=True)

    # Category identifier, e.g. 'PERSONAL_BASIC', 'FINANCIAL_ACCOUNT'.
    id: str = Field(description="分类 ID，如 'PERSONAL_BASIC', 'FINANCIAL_ACCOUNT'")
    # Display name for the category, e.g. '个人基本信息'.
    name: str = Field(description="分类名称，如 '个人基本信息'")
    # Parent category ID for building hierarchical tree structures.
    # None means this is a root-level category.
    parent_id: Optional[str] = Field(default=None, description="父分类 ID，支持多级树结构")
    # Optional description explaining what data belongs in this category.
    description: Optional[str] = Field(default=None, description="分类说明")


class DomainTaxonomy(BaseModel):
    """领域分类体系完整定义。 / Complete Domain Taxonomy Definition.

    一个 Taxonomy 对应一个行业标准的分类分级元数据， / A Taxonomy corresponds to an industry standard's classification metadata,
    包含该标准下的所有等级定义和分类目录树。 / containing all level definitions and category directory trees under that standard.
    支持通过 confidence_policy 节配置置信度策略。 / Supports configuring confidence policy via the confidence_policy section.
    """

    # Allow extra fields from YAML to be preserved (forward compatibility).
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # Domain identifier (e.g. 'healthcare', 'finance', 'gov').
    domain: str = Field(description="领域标识，如 'healthcare', 'finance', 'gov'")
    # Standard code this taxonomy implements (e.g. 'DB51_T_2989', 'JR_T_0197').
    standard_id: str = Field(description="标准编号，如 'DB51_T_2989', 'JR_T_0197'")
    # Semantic version of this taxonomy definition.
    version: str = Field(default="1.0.0", description="体系版本号")
    # Optional human-readable description.
    description: Optional[str] = Field(default=None, description="体系说明")
    # Mapping of level ID -> level definition. Contains all sensitivity levels
    # defined by this standard (e.g. L1~L5 for healthcare, C1~C4 for finance).
    levels: dict[str, SensitivityLevelDef] = Field(
        default_factory=dict, description="等级 ID → 等级定义的映射"
    )
    # Mapping of category ID -> category definition. Forms the classification tree.
    categories: dict[str, CategoryDef] = Field(
        default_factory=dict, description="分类 ID → 分类定义的映射"
    )
    # Fallback level ID used when no rule hits a field (safe default).
    default_level: str = Field(default="L3", description="无规则命中时的默认等级 ID")
    # Explicit confidence policy configuration (loaded from taxonomy YAML).
    confidence_policy: Optional[ConfidencePolicy] = Field(
        default=None, description="置信度策略配置（冲突衰减 + LLM 仲裁触发条件）"
    )
    # NER entity-to-level mapping for multi-standard support.
    # Keys are entity labels (e.g. "GENOMIC_HINT"), values are level IDs.
    ner_entity_mapping: Optional[dict[str, str]] = Field(
        default=None, description="NER 实体类型→等级 ID 映射（支持多标准体系）"
    )
    # NER sensitive keywords: entities containing these keywords are upgraded
    # to the second-highest level. Configurable per taxonomy/domain.
    ner_sensitive_keywords: Optional[list[str]] = Field(
        default=None,
        description="NER 敏感关键词列表（命中时升级为次高等级），默认内置医疗敏感病种",
    )
    # LLM arbitration prompt template. Supports placeholders:
    # {field_name}, {value}, {domain}, {standard_id}, {conflict_desc}, {levels_desc}
    llm_arbitration_prompt_template: Optional[str] = Field(
        default=None,
        description="LLM 仲裁 prompt 模板（支持占位符），None 时使用内置默认模板",
    )
    # NER raw label mapping: maps raw NER engine output labels to standard labels.
    # E.g. {"dis": "MEDICAL_DISEASE", "dru": "MEDICATION", "pro": "SURGERY"}
    # When None, uses built-in medical NER label mapping.
    ner_label_mapping: Optional[dict[str, str]] = Field(
        default=None,
        description="NER 原始标签→标准标签映射（如 dis→MEDICAL_DISEASE），None 时使用内置医疗映射",
    )
    # NER model file path (ONNX model or ModelScope model directory).
    ner_model_path: Optional[str] = Field(
        default=None,
        description="NER 模型文件路径（ONNX 或 ModelScope 目录），None 时自动检测",
    )
    # NER vocabulary file path (for ONNX engine tokenizer).
    ner_vocab_path: Optional[str] = Field(
        default=None,
        description="NER 词表文件路径（ONNX 引擎用），None 时自动检测",
    )
    # LLM model directory path (e.g. .models/Qwen3.5-0.8B-Privacy-Classifier-Smoother).
    llm_model_path: Optional[str] = Field(
        default=None,
        description="LLM 模型目录路径，None 时使用默认路径",
    )
    # LLM classification system prompt template.
    # Supports placeholders: {domain}, {standard_id}, {levels_desc}
    # When None, uses built-in medical-domain default prompt.
    llm_classify_prompt_template: Optional[str] = Field(
        default=None,
        description="LLM 分类 system prompt 模板（支持占位符 {domain}/{standard_id}/{levels_desc}），None 时使用内置医疗默认 prompt",
    )

    def max_level(self, *level_ids: str) -> str:
        """返回等级集合中 rank 最高的等级 ID。 / Returns the level ID with the highest rank in the level set.

        Args:
            *level_ids: 一个或多个等级 ID。 / One or more level IDs.

        Returns:
            rank 最大的等级 ID；无有效输入时返回 default_level。 / The level ID with the highest rank; returns default_level when no valid input is provided.
        """
        # Step 1: If no level IDs provided, return the configured default.
        if not level_ids:
            return self.default_level
        # Step 2: Filter to only IDs that exist in our levels dict (ignore unknown IDs).
        # 未知等级静默过滤是 fail-open 行为：补 warning 保证可观测性（含等级名与 taxonomy）。
        unknown = [lid for lid in level_ids if lid not in self.levels]
        if unknown:
            logger.warning(
                "taxonomy_unknown_level_filtered",
                extra={
                    "unknown_levels": unknown,
                    "taxonomy": self.standard_id,
                    "domain": self.domain,
                },
            )
        valid = [lid for lid in level_ids if lid in self.levels]
        # Step 3: If none of the provided IDs are valid, fall back to default.
        if not valid:
            return self.default_level
        # Step 4: Return the level with the highest rank value (most sensitive).
        return max(valid, key=lambda lid: self.levels[lid].rank)

    def get_level_rank(self, level_id: str) -> int:
        """获取等级的排序权重。 / Gets the rank of the level.

        Args:
            level_id: 等级 ID。 / Level ID.

        Returns:
            rank 值；未找到时返回 0。 / The rank value; returns 0 if not found.
        """
        # Look up the level in the levels dict; return its rank or 0 if not found.
        if level_id in self.levels:
            return self.levels[level_id].rank
        # 未找到时返回 0 属于 fail-open 兜底：补 warning 保证可观测性（含等级名与 taxonomy）。
        logger.warning(
            "taxonomy_unknown_level_rank_fallback",
            extra={
                "level_id": level_id,
                "taxonomy": self.standard_id,
                "domain": self.domain,
            },
        )
        return 0


    def get_category_path(self, category_id: str) -> list[str]:
        """获取分类的完整路径（从根到叶）。 / Gets the full path of the category (from root to leaf).

        Args:
            category_id: 分类 ID。 / Category ID.

        Returns:
            从根分类到目标分类的 ID 列表。 / List of IDs from root category to target category.
        """
        # Build path by walking up the parent_id chain from leaf to root.
        path: list[str] = []
        current: Optional[str] = category_id
        # Visited set prevents infinite loops if there's a circular parent reference.
        visited: set[str] = set()
        # Walk up the tree: append current node, then move to its parent.
        while current and current in self.categories and current not in visited:
            visited.add(current)       # Mark as visited to detect cycles
            path.append(current)       # Collect current category ID
            current = self.categories[current].parent_id  # Move to parent
        # Reverse to get root-to-leaf order (we built it leaf-to-root).
        return list(reversed(path))


# ===========================================================================
# 安全标签与分类结果 / Security Tag & Classification Results
# ===========================================================================


class SecurityTag(BaseModel):
    """安全标签，描述单次规则命中的分类结果。 / Security Tag, describing the classification result of a single rule hit.

    每次规则/算子命中都会产出一个 SecurityTag，记录命中的等级、类别、 / Every rule/operator hit produces a SecurityTag, recording the hit level, category,
    来源引擎、规则 ID 等审计信息。 / source engine, rule ID, and other audit information.
    """

    model_config = ConfigDict(populate_by_name=True)

    # Sensitivity level ID assigned by the rule hit (e.g. 'L3', 'C4').
    level: str = Field(description="敏感度等级 ID（如 'L3', 'C4'）")
    # Classification category ID (e.g. 'PII_ID_CARD', 'GENOMIC').
    category: str = Field(description="分类类别 ID（如 'PII_ID_CARD', 'GENOMIC'）")
    # Confidence score [0.0, 1.0]: rule engine always produces 1.0 (deterministic).
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度 [0,1]")
    # Which engine produced this tag: 'RULE' (configurable engine) or 'COMPOSITE'.
    source_engine: str = Field(default="RULE", alias="sourceEngine", description="来源引擎标识")
    # The specific rule ID that triggered this tag (for audit/debugging).
    rule_id: str = Field(default="", alias="ruleId", description="触发的规则 ID")
    # Domain context this tag was generated under.
    domain: str = Field(default="", description="所属领域")
    # Standard context this tag was generated under.
    standard_id: str = Field(default="", alias="standardId", description="所属标准")
    # Tag schema version for forward compatibility.
    version: str = Field(default="1.0.0", description="标签版本")
    # Flag indicating this tag needs human review (e.g. low-confidence ML results).
    needs_human_review: bool = Field(default=False, alias="needsHumanReview", description="是否需人工复核")
    # Flag indicating this tag was produced by an override-enabled downgrade rule.
    # When true, this tag has suppression power over lower-rank normal rule tags.
    is_override: bool = Field(default=False, alias="isOverride", description="是否为覆盖型降级标签")
    # Flag indicating this tag was produced by a downgrade rule (regardless of override).
    # Used by the funnel's conflict detection: normal + downgrade coexist = conflict.
    is_downgrade: bool = Field(default=False, alias="isDowngrade", description="是否由降级规则产生")
    # Records what the rule matched against: "field_name" or "field_value".
    # Value-level hits (field_value) are exempt from override suppression because
    # they represent high-confidence evidence (e.g. checksum validation, regex match).
    match_target: str = Field(default="field_name", alias="matchTarget", description="匹配目标: field_name | field_value")

    def __str__(self) -> str:
        # Compact string representation: "L3_PII_ID_CARD" format.
        return f"{self.level}_{self.category}"


class FieldClassificationResult(BaseModel):
    """单个字段的分类结果。 / Single Field Classification Result.

    记录对数据集中某一个字段（列）的完整分类信息，包括： / Records the complete classification info for a specific field (column) in the dataset, including:
    - 所有命中的安全标签列表 / List of all matched security tags
    - 最终裁定的敏感度等级 / Final adjudicated sensitivity level
    - 产出该结果的引擎层级（三层漏斗中的哪一层） / The engine layer that produced this result (which layer in the 3-layer funnel)
    - 置信度（可能因冲突衰减或 LLM 修正） / Confidence (may be decayed by conflict or corrected by LLM)
    - 推理说明（LLM 层会填充详细推理过程） / Reasoning (LLM layer will populate detailed reasoning process)
    """

    model_config = ConfigDict(populate_by_name=True)

    # The field/column name that was classified.
    field_name: str = Field(alias="fieldName", description="字段名称")
    # Optional sample value of the field (for display/debugging, may be truncated).
    field_value: Optional[str] = Field(default=None, alias="fieldValue", description="字段示例值")
    # All security tags produced by rules that hit this field.
    tags: list[SecurityTag] = Field(default_factory=list, description="命中的安全标签列表")
    # Final adjudicated sensitivity level (highest rank among all tags).
    final_level: str = Field(alias="finalLevel", description="最终裁定的敏感度等级")
    # Aggregate confidence: may be decayed on conflict or corrected by LLM.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="综合置信度")
    # Whether any tag in this result requires human review.
    needs_human_review: bool = Field(default=False, alias="needsHumanReview")
    # Which engine layer produced the final decision (L1_RULE / L2_SMALL_NER / L3_LLM).
    engine_layer: str = Field(
        default="L1_RULE", alias="engineLayer",
        description="产出最终决策的引擎层级",
    )
    # Human-readable reasoning explanation (populated by LLM layer or conflict detection).
    reasoning: str = Field(default="", description="分类推理说明")
    # Smart sanitized/masked value produced when sanitize=True.
    sanitized_value: Optional[str] = Field(
        default=None, alias="sanitizedValue",
        description="智能抹平/脱敏后的字段值（当 sanitize=True 时填充）",
    )
    # Tags that were suppressed by override downgrade rules (for audit trail).
    # When non-empty, indicates that override suppression occurred and explains
    # why certain rules did not contribute to the final level.
    suppressed_tags: list[SecurityTag] = Field(
        default_factory=list, alias="suppressedTags",
        description="被 override 压制的标签列表（审计用）",
    )


class RecordClassificationResult(BaseModel):
    """单条记录（多字段）的分类结果。 / Single Record (Multi-field) Classification Result."""

    model_config = ConfigDict(populate_by_name=True)

    # Zero-based index of this record within the batch/table.
    record_index: int = Field(default=0, alias="recordIndex")
    # Per-field classification results: field_name -> FieldClassificationResult.
    field_results: dict[str, FieldClassificationResult] = Field(
        default_factory=dict, alias="fieldResults"
    )
    # All tags aggregated from all fields plus composite rules.
    aggregated_tags: list[SecurityTag] = Field(default_factory=list, alias="aggregatedTags")
    # Record-level final sensitivity level (max of all field levels + composite upgrades).
    final_level: str = Field(alias="finalLevel", description="记录级最终等级")
    # Aggregate confidence for the entire record.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Whether any field or composite tag requires human review.
    needs_human_review: bool = Field(default=False, alias="needsHumanReview")


class TableClassificationResult(BaseModel):
    """整张表/批次的分类结果。 / Entire Table/Batch Classification Result."""

    model_config = ConfigDict(populate_by_name=True)

    # Column names (schema) of the table being classified.
    schema_: list[str] = Field(default_factory=list, alias="schema")
    # Per-record classification results for all rows in the table.
    record_results: list[RecordClassificationResult] = Field(
        default_factory=list, alias="recordResults"
    )
    # All tags aggregated across all records.
    aggregated_tags: list[SecurityTag] = Field(default_factory=list, alias="aggregatedTags")
    # Table-level final sensitivity (max across all records).
    final_level: str = Field(alias="finalLevel")
    # Aggregate confidence for the entire table.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Whether any record requires human review.
    needs_human_review: bool = Field(default=False, alias="needsHumanReview")

    @property
    def columns(self) -> list[str]:
        """Convenience property for column names (alias for schema_)."""
        return self.schema_



class AuditInfo(BaseModel):
    """审计信息，记录分类请求的执行元数据。 / Audit Info, recording execution metadata of classification requests."""

    model_config = ConfigDict(populate_by_name=True)

    # Schema version of this audit info structure.
    version: str = "1.0.0"
    # Domain context used during classification.
    domain: str = ""
    # Standard context used during classification.
    standard_id: str = Field(default="", alias="standardId")
    # ISO 8601 UTC timestamp when the classification was performed.
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Version of the rule set that was used for evaluation.
    rule_set_version: str = Field(default="1.0.0", alias="ruleSetVersion")
    # Total number of rules evaluated during this request.
    rules_evaluated: int = Field(default=0, alias="rulesEvaluated")
    # Number of rules that actually produced a hit.
    rules_hit: int = Field(default=0, alias="rulesHit")
    # Total execution time in milliseconds.
    duration_ms: float = Field(default=0.0, alias="durationMs")


class ClassificationResponse(BaseModel):
    """分类响应包装器。 / Classification Response Wrapper.

    Wraps exactly one of field_result / record_result / table_result / 根据请求的分类粒度包装
    depending on the classification granularity requested. / field_result / record_result / table_result 中的恰好一个。
    """

    model_config = ConfigDict(populate_by_name=True)

    # Populated for field-level classification requests.
    field_result: Optional[FieldClassificationResult] = Field(default=None, alias="fieldResult")
    # Populated for record-level classification requests.
    record_result: Optional[RecordClassificationResult] = Field(default=None, alias="recordResult")
    # Populated for table-level classification requests.
    table_result: Optional[TableClassificationResult] = Field(default=None, alias="tableResult")
    # Always populated: execution metadata for auditing and debugging.
    audit_info: AuditInfo = Field(default_factory=AuditInfo, alias="auditInfo")
