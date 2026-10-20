"""复合规则引擎 / Composite Rule Engine.

用于识别“单字段不敏感、多字段组合后敏感”的上下文场景。
在单条记录的字段级分类完成后执行，根据字段名组合升级敏感度等级。
Used to identify context scenarios where "a single field is not sensitive, but sensitive when combined with multiple fields".
Executed after field-level classification of a single record, upgrading sensitivity level based on field name combinations.

典型场景 / Typical scenarios:
- 单独一个 "name" 字段可能只是 L3，但如果同一条记录中同时存在
  "name" + "id_card" + "mobile"，则组合后应升级为 L5。
  A single "name" field might only be L3, but if "name" + "id_card" + "mobile"
  exist simultaneously in the same record, the combination should be upgraded to L5.

===================================================================================
              复合规则执行流程 / Composite Rule Execution Flow
===================================================================================

  service.classify_record(record)
       │
       ├─① 对每个字段调用 funnel.classify_field() → 字段级结果
       │
       ├─② CompositeRuleEngine.evaluate(record, field_results)
       │     │
       │     │  for each rule in composite_rules:
       │     │    for each pattern in rule.field_patterns:
       │     │      正则匹配字段名 → 命中数 += 1
       │     │    if 命中数 >= rule.min_matches:
       │     │      生成 SecurityTag(level=target_level, source_engine="COMPOSITE")
       │     │
       │     └─→ 返回 list[SecurityTag]
       │
       ├─③ apply_to_record_level(current_level, composite_tags, taxonomy)
       │     → 只升不降: final = max(current_level, *composite_tag_levels)
       │
       └─④ 聚合为 RecordClassificationResult
===================================================================================
"""

from __future__ import annotations

# re module for compiling and matching field name patterns
import re
from typing import Any

from .models import FieldClassificationResult, SecurityTag
from .rule_schema import CompositeRuleDef
from ..observability.logging_config import get_logger

logger = get_logger(__name__)


def _normalize(name: str) -> str:
    """规范化字段名用于模式匹配 / Normalize field names for pattern matching.

    Normalization steps:
    1. Convert to lowercase.
    2. Strip spaces, underscores, and hyphens so 'id_card', 'id-card', 'idcard' match seamlessly.
    """
    return re.sub(r"[\s_\-]+", "", str(name).lower())


class CompositeRuleEngine:
    """复合规则引擎 / Composite Rule Engine."""

    def __init__(
        self,
        rules: list[CompositeRuleDef] | None = None,
        domain: str = "",
        standard_id: str = "",
    ):
        self.rules = list(rules) if rules else []
        self.domain = domain
        self.standard_id = standard_id

        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for rule in self.rules:
            compiled_list: list[re.Pattern] = []
            for pattern in rule.field_patterns:
                try:
                    # 将 pattern 正确进行原子分组包覆，防止 | 交替符破坏词边界；
                    # 匹配词边界或下划线边界
                    bounded_pattern = rf"(?:\b|_)(?:{pattern})(?:\b|_)"
                    compiled_list.append(re.compile(bounded_pattern, re.IGNORECASE))
                except re.error as e:
                    logger.error(f"Invalid regex pattern in composite rule '{rule.id}': '{pattern}'. Error: {e}")
                    pass
            self._compiled_patterns[rule.id] = compiled_list

    def evaluate(
        self,
        record: dict[str, Any],
        field_results: dict[str, FieldClassificationResult] | None = None,
    ) -> list[SecurityTag]:
        """评估单条记录是否命中复合规则 / Evaluate if a single record hits composite rules.

        Algorithm:
        1. Normalize all field names in the record for pattern matching.
        2. For each composite rule, count how many of its patterns match
           at least one field name in the record.
        3. If matched count >= rule.min_matches, the rule fires and produces a tag.

        Args:
            record: 原始记录字典（字段名 → 字段值） / Original record dictionary (field_name → field_value).
            field_results: 字段级分类结果（可选，预留扩展） / Field-level classification results (optional, reserved for extension).

        Returns:
            命中的 SecurityTag 列表 / List of hit SecurityTags.
        """
        tags: list[SecurityTag] = []
        # Build a normalized-name -> original-name mapping for all fields in the record.
        # This allows regex matching on normalized names while preserving original names for output.
        norm_fields = {_normalize(name): name for name in record}

        for rule in self.rules:
            # Counter for how many distinct patterns have matched at least one field.
            matched = 0
            # Collect original field names that matched (for potential audit use).
            matched_names: list[str] = []
            # Retrieve pre-compiled patterns for this rule (avoids re-compilation).
            compiled_patterns = self._compiled_patterns.get(rule.id, [])

            for compiled in compiled_patterns:
                # For each pattern, check if ANY field in the record matches it.
                for norm_name, original_name in norm_fields.items():
                    # 同时匹配规范化名称与原始名称
                    if compiled.search(norm_name) or compiled.search(original_name):
                        matched += 1
                        matched_names.append(original_name)
                        break

                # Early exit optimization: if threshold already met, skip remaining patterns.
                if matched >= rule.min_matches:
                    break

            # If the rule's threshold is satisfied, generate a SecurityTag.
            if matched >= rule.min_matches:
                tags.append(
                    SecurityTag(
                        level=rule.target_level,       # Upgraded sensitivity level
                        category=rule.category,        # Category from composite rule
                        confidence=1.0,                # Deterministic rule = full confidence
                        source_engine="COMPOSITE",     # Tag produced by composite engine
                        rule_id=rule.id,               # Which composite rule fired
                        domain=self.domain,
                        standard_id=self.standard_id,
                    )
                )

        return tags


    def apply_to_record_level(
        self,
        current_level: str,
        composite_tags: list[SecurityTag],
        taxonomy: Any = None,
    ) -> str:
        """将复合规则标签应用到记录级等级（取最高） / Apply composite rule tags to record-level sensitivity (taking the highest).

        Logic: The final record level is the maximum of the current level
        and all composite tag levels. This ensures composite rules can only
        UPGRADE sensitivity, never downgrade it.

        Args:
            current_level: 当前记录级等级 / Current record-level sensitivity.
            composite_tags: 复合规则命中的标签 / Hit tags from composite rules.
            taxonomy: 分类体系（用于比较等级） / Taxonomy (used for level comparison).

        Returns:
            升级后的等级 ID / Upgraded level ID.
        """
        # If no composite tags fired, the current level remains unchanged.
        if not composite_tags:
            return current_level

        # Collect all candidate levels: current + all composite tag levels.
        levels = [current_level] + [tag.level for tag in composite_tags]

        # If a taxonomy is available, use its rank-based comparison (accurate).
        if taxonomy is not None:
            return taxonomy.max_level(*levels)

        # Fallback without taxonomy: simple string max (lexicographic ordering).
        # This works for standard naming like L1 < L2 < ... < L5.
        return max(levels)