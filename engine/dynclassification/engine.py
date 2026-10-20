"""通用可配置规则引擎 / Configurable Rule Engine.

根据声明式 RuleProfile 动态执行规则匹配。
引擎本身不包含任何领域知识，仅负责解释执行规则配置。
Dynamically executes rule matching based on declarative RuleProfile.
The engine itself contains no domain knowledge and is only responsible for interpreting and executing rule configurations.

执行流程 / Execution flow:
1. 合并多个领域包的规则列表（按 priority 降序） / Merge rule lists from multiple domain packages (descending by priority)
2. 遍历规则，对每条规则的 matchers 执行算子匹配 / Iterate rules, execute operator matching for each rule's matchers
3. 根据 match_logic (AND/OR) 判断是否命中 / Determine hit based on match_logic (AND/OR)
4. 命中则生成 SecurityTag（支持 ICD-10 动态等级） / Generate SecurityTag on hit (supports ICD-10 dynamic levels)
5. 执行降级规则 / Execute downgrade rules
6. 去重返回 / Deduplicate and return
"""

from __future__ import annotations

from collections import OrderedDict
import fnmatch
import hashlib
import os
import threading
from typing import Any, Tuple

# Structured logger for this module (JSON/text output based on config)
from ..observability.logging_config import get_logger
# Side-effect import: importing operators module triggers @OperatorRegistry.register
# decorators, ensuring all built-in operators are available before engine evaluation.
from . import operators  # noqa: F401 - Ensure built-in operators are registered
from .models import DomainTaxonomy, SecurityTag
from .operator_registry import OperatorRegistry, OperatorResult, normalize_result
from .rule_schema import DowngradeRuleDef, MatcherDef, RuleDef, RuleProfile

logger = get_logger(__name__)


# Prometheus metrics for monitoring engine behavior in production:
from ..observability.metrics import (
    CLASSIFICATION_RULE_HITS_TOTAL,
    DYNCLASSIFICATION_OPERATOR_CALLS_TOTAL,
    DYNCLASSIFICATION_OPERATOR_ERRORS_TOTAL,
    DYNCLASSIFICATION_OVERRIDE_SUPPRESSED_TOTAL,
    DYNCLASSIFICATION_RULE_HITS_TOTAL,
)


class ConfigurableRuleEngine:
    """通用可配置规则引擎 / Configurable Rule Engine.

    替代 DefaultRuleEngine 的硬编码逻辑，根据声明式 RuleProfile
    动态执行规则匹配。引擎本身不包含任何领域知识。
    Replaces the hardcoded logic of DefaultRuleEngine, dynamically executing rule matching based on declarative RuleProfile. The engine itself contains no domain knowledge.

    Attributes:
        taxonomy: 分类体系定义（等级 + 类别） / Taxonomy definition (level + category).
        rules: 合并后的规则列表（按 priority 降序） / Merged rule list (descending by priority).
        downgrade_rules: 合并后的降级规则列表 / Merged downgrade rule list.
        domain: 当前引擎绑定的领域标识 / Domain identifier bound to the current engine.
        standard_id: 当前引擎绑定的标准标识 / Standard identifier bound to the current engine.
    """

    def __init__(
        self,
        taxonomy: DomainTaxonomy,
        profiles: list[RuleProfile],
        domain: str = "",
        standard_id: str = "",
        cache_max_size: int | None = None,
    ):
        """初始化规则引擎 / Initialize the rule engine.

        Args:
            taxonomy: 分类体系定义 / Taxonomy definition.
            profiles: 领域规则包列表（将被合并） / List of domain rule packages (to be merged).
            domain: 领域标识（写入标签） / Domain identifier (written to tags).
            standard_id: 标准标识（写入标签） / Standard identifier (written to tags).
            cache_max_size: 字段评估 LRU 缓存容量上限（可选，默认读取 PRIVACY_ENGINE_CACHE_MAX_SIZE 环境变量，缺省 4096） / Max capacity for evaluation LRU cache.
        """
        # Store taxonomy for level comparison and default level resolution.
        self.taxonomy = taxonomy
        # Domain and standard identifiers are embedded in every produced SecurityTag.
        self.domain = domain
        self.standard_id = standard_id
        # Merge all profiles' rules into a single priority-sorted list.
        self.rules = self._merge_rules(profiles)
        # Merge all profiles' downgrade rules into a flat list.
        self.downgrade_rules = self._merge_downgrade_rules(profiles)

        # Thread lock for thread-safe concurrent evaluation cache mutations
        self._cache_lock = threading.Lock()
        # Configurable LRU Evaluation Cache capacity
        if cache_max_size is not None:
            self._eval_cache_max_size = max(1, cache_max_size)
        else:
            raw_env = os.environ.get("PRIVACY_ENGINE_CACHE_MAX_SIZE", "4096")
            try:
                parsed_val = int(raw_env)
                self._eval_cache_max_size = max(1, parsed_val)
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid PRIVACY_ENGINE_CACHE_MAX_SIZE '%s', falling back to default 4096.",
                    raw_env,
                )
                self._eval_cache_max_size = 4096

        # True LRU Evaluation Cache using OrderedDict: (field_name, str_value[:200]) -> (final_tags, suppressed_tags)
        self._eval_cache: OrderedDict[tuple[str, str], Tuple[list[SecurityTag], list[SecurityTag]]] = OrderedDict()
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def clear_cache(self) -> None:
        """清空规则引擎字段评估缓存 / Clear field evaluation cache."""
        with self._cache_lock:
            self._eval_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    def cache_info(self) -> dict[str, int]:
        """获取评估缓存统计信息 / Get evaluation cache statistics."""
        with self._cache_lock:
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "size": len(self._eval_cache),
                "max_size": self._eval_cache_max_size,
            }

    def _merge_rules(self, profiles: list[RuleProfile]) -> list[RuleDef]:
        """合并多个领域包的规则列表，按 priority 降序排列 / Merge rule lists from multiple domain packages, sorted descending by priority.

        Only enabled rules are included. Higher priority rules execute first,
        ensuring more specific rules take precedence over general ones.
        """
        all_rules: list[RuleDef] = []
        for profile in profiles:
            # Filter: only include rules with enabled=True.
            all_rules.extend(r for r in profile.rules if r.enabled)
        # Sort by priority descending: highest priority rules are evaluated first.
        return sorted(all_rules, key=lambda r: r.priority, reverse=True)

    def _merge_downgrade_rules(self, profiles: list[RuleProfile]) -> list[DowngradeRuleDef]:
        """合并多个领域包的降级规则列表 / Merge downgrade rule lists from multiple domain packages.

        降级策略（Downgrade Strategy）说明 / Downgrade Strategy Description:
            降级规则是一种"反向修正"机制。当普通规则将某字段判定为较高敏感等级后，
            降级规则会根据字段名中的特定关键词，将其敏感度"下调"到更低的等级。
            Downgrade rules are a "reverse correction" mechanism. When normal rules determine a field as a higher sensitivity level, downgrade rules will "downgrade" its sensitivity to a lower level based on specific keywords in the field name.

            典型场景 / Typical scenarios:
            - 字段名包含 "turnover"（营业额）、"门诊人次" 等运营统计关键词时，
              虽然可能被通用规则误判为敏感数据，但实际属于机构运营指标，
              应降级为 L2（内部数据）而非 L3（敏感数据）。
              When a field name contains operational statistic keywords like "turnover" or "outpatient visits", although it might be misjudged as sensitive data by general rules, it actually belongs to organizational operational metrics and should be downgraded to L2 (internal data) instead of L3 (sensitive data).
            - 字段名包含 "public"、"公开" 等关键词时，应降级为 L1（公开数据）。
              When a field name contains keywords like "public", it should be downgraded to L1 (public data).

            执行时机 / Execution timing:
            降级规则在所有普通规则评估完毕后统一执行，
            产生的低等级标签会与普通规则标签一同参与最终等级裁定（取最高），
            因此降级标签本身不会"覆盖"高等级标签，而是为无其他规则命中的字段
            提供一个合理的低等级归属。
            Downgrade rules are executed uniformly after all normal rules are evaluated. The generated low-level tags will participate in the final level determination (taking the highest) along with normal rule tags, so downgrade tags themselves will not "override" high-level tags, but provide a reasonable low-level attribution for fields not hit by other rules.

        注意 / Note: 降级规则不按优先级排序，所有降级规则对每个字段均会被评估。 / Downgrade rules are not sorted by priority; all downgrade rules are evaluated for every field.
        """
        # 初始化空列表，用于汇聚所有领域包中的降级规则
        all_rules: list[DowngradeRuleDef] = []
        for profile in profiles:
            # 将当前领域包的降级规则追加到汇总列表中
            all_rules.extend(profile.downgrade_rules)
        return all_rules

    def evaluate(
        self,
        field_name: str,
        value: Any,
        context: dict[str, Any] | None = None,
    ) -> Tuple[list[SecurityTag], list[SecurityTag]]:
        """评估单个字段，返回命中的安全标签列表和被压制的标签列表 / Evaluate a single field, returning lists of hit security tags and suppressed tags.
    
        Execution flow:
        1. Convert value to string for uniform operator processing.
        2. Phase 1: Iterate all rules (priority order) and evaluate each.
        3. Phase 2: Evaluate downgrade rules.
        4. Phase 3: Apply override suppression (force_suppress=true downgrade rules
           forcibly remove normal tags with rank <= cap).
        5. Phase 4: Merge + deduplicate by (level, category) and return.
    
        Args:
            field_name: 字段名 / Field name.
            value: 字段值 / Field value.
            context: 可选的执行上下文（预留扩展） / Optional execution context (reserved for extension).
    
        Returns:
            一个元组 (final_tags, suppressed_tags)，分别包含最终生效的标签和被压制的标签 / A tuple (final_tags, suppressed_tags), containing the final effective tags and suppressed tags respectively.
    
        字段评估流程（含强制覆盖） / Field evaluation flow (including forced override):
        ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        │  evaluate(field_name, value)                                                                                       │
        │                                                                                                                    │
        │  Phase 1: 普通规则评估 / Normal rule evaluation → normal_tags = [L5, L4, L3, ...]                                     │
        │  Phase 2: 降级规则评估 / Downgrade rule evaluation → downgrade_tags = [L2, L1, ...]                                   │
        │                                                                                                                    │
        │  Phase 3: 强制覆盖裁定 / Forced override determination (force_suppress=true 的降级规则 / downgrade rules)               │
        │    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐        │
        │    │ 对每条 override 降级标签 / For each override downgrade tag:                                              │        │
        │    │   cap_rank = rank(max_force_suppress_level)                                                           │        │
        │    │   从 normal_tags 中移除 rank <= cap_rank 的标签 / Remove tags from normal_tags with rank <= cap_rank     │        │
        │    │   (被移除标签记入 suppressed_tags 用于审计 / Removed tags are recorded in suppressed_tags for auditing)    │        │
        │    └───────────────────────────────────────────────────────────────────────────────────────────────────────┘        │
        │                                                                                                                     │
        │  Phase 4: 合并 / Merge normal_tags + downgrade_tags → 去重 / Deduplicate → return                                    │
        └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                   │
                                                                   ▼
        ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        │  _resolve_final_level(tags, engine)                                                                                │
        │    if not tags → return default_level ("L3")                                                                       │
        │    else → return max_level(*levels)  ← 取所有标签中最高 / Take the highest among all tags                             │
        └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        """
        # Convert value to string once; all operators work on string representation.
        str_value = str(value) if value is not None else ""

        # Check Evaluation Cache for instant lookup (PII hygiene: SHA-256 truncated 128-bit hash key to prevent plaintext PII memory retention)
        val_digest = hashlib.sha256(str_value.encode("utf-8")).hexdigest()[:32]
        cache_key = (field_name, val_digest)
        if context is None:
            with self._cache_lock:
                if cache_key in self._eval_cache:
                    self._cache_hits += 1
                    self._eval_cache.move_to_end(cache_key)
                    cached_final, cached_suppressed = self._eval_cache[cache_key]
                    return list(cached_final), list(cached_suppressed)
                self._cache_misses += 1

        # Phase 1: Evaluate all normal rules in priority order.
        normal_tags: list[SecurityTag] = []
        for rule in self.rules:
            tag = self._evaluate_rule(rule, field_name, str_value)
            if tag is not None:
                normal_tags.append(tag)

        # Phase 2: Evaluate downgrade rules (produces downgrade tags + override info).
        downgrade_tags = self._evaluate_downgrade(field_name)

        # Phase 3: Apply override suppression.
        surviving_tags, suppressed_tags = self._apply_override_suppression(
            normal_tags, downgrade_tags
        )

        # Phase 4: Merge all surviving tags and deduplicate.
        all_tags = surviving_tags + downgrade_tags
        final_tags = self._unique_tags(all_tags)

        # Store in LRU Evaluation Cache if no custom context
        if context is None:
            with self._cache_lock:
                if cache_key not in self._eval_cache and len(self._eval_cache) >= self._eval_cache_max_size:
                    self._eval_cache.popitem(last=False)
                self._eval_cache[cache_key] = (list(final_tags), list(suppressed_tags))
                self._eval_cache.move_to_end(cache_key)

        return final_tags, suppressed_tags

    def evaluate_batch(
        self,
        fields: list[tuple[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Tuple[list[SecurityTag], list[SecurityTag]]]:
        """批量评估多个字段 / Evaluate multiple fields in batch.

        Convenience wrapper that calls evaluate() for each field sequentially.
        Future optimization: could add vectorized/batch operator execution.

        Args:
            fields: (字段名, 字段值) 元组列表 / List of (field_name, field_value) tuples.
            context: 可选的执行上下文 / Optional execution context.

        Returns:
            字段名 → (标签列表, 被压制标签列表) 的映射 / Mapping of field_name → (list of tags, list of suppressed tags).
        """
        results: dict[str, Tuple[list[SecurityTag], list[SecurityTag]]] = {}
        # Iterate each (field_name, value) pair and evaluate independently.
        for field_name, value in fields:
            results[field_name] = self.evaluate(field_name, value, context)
        return results

    def _evaluate_rule(self, rule: RuleDef, field_name: str, str_value: str) -> SecurityTag | None:
        """评估单条规则，命中则返回 SecurityTag，否则返回 None / Evaluate a single rule; return SecurityTag on hit, otherwise None.

        Algorithm:
        1. Execute each matcher in the rule and collect boolean results.
        2. Apply match_logic (AND/OR) to determine if the rule fires.
        3. If fired, resolve dynamic level/category (via OperatorResult) and build tag.
        4. Track the actual hit matcher's target for accurate match_target.
        """
        if not rule.matchers:
            return None

        results: list[bool] = []
        dynamic_level: str | None = None
        dynamic_category: str | None = None
        # 记录首个实际命中的匹配器 target（用于精确标注 match_target 归属）
        hit_target: str = "field_name"

        is_or_logic = rule.match_logic.upper() == "OR"

        for matcher in rule.matchers:
            op_result = self._execute_matcher(matcher, field_name, str_value)
            is_hit = op_result.hit
            results.append(is_hit)

            # 命中时捕获由算子返回的动态等级/类别 (如 ICD-10 动态匹配)
            if is_hit and op_result.level is not None:
                dynamic_level = op_result.level
                dynamic_category = op_result.category

            # 记录首个命中算子的 target 目标 (field_name 或 field_value)
            if is_hit and hit_target == "field_name":
                hit_target = matcher.target

            # 算法短路优化 (Short-Circuit Optimization):
            # 1. OR 逻辑下：只要有一个算子命中 (is_hit=True)，可立即短路成立；
            # 2. AND 逻辑下：只要有一个算子未命中 (is_hit=False)，可立即短路失败，跳过后续冗余正则计算。
            if is_or_logic and is_hit:
                break
            elif not is_or_logic and not is_hit:
                break

        if is_or_logic:
            matched = any(results)
        else:  # AND (默认)
            matched = all(results)

        if not matched:
            return None

        DYNCLASSIFICATION_RULE_HITS_TOTAL.labels(
            rule_id=rule.id,
            domain=self.domain or "default",
            standard=self.standard_id or "default",
        ).inc()
        CLASSIFICATION_RULE_HITS_TOTAL.labels(rule_id=rule.id).inc()

        level = dynamic_level if dynamic_level is not None else rule.level
        category = dynamic_category if dynamic_category is not None else rule.category

        # Use the actual hit matcher's target instead of inferring from rule definition.
        # For AND logic, if any matcher targets field_value, the rule is value-driven.
        # For OR logic, use the first hit matcher's target.
        if rule.match_logic.upper() == "AND":
            match_target = "field_value" if any(
                m.target == "field_value" for m in rule.matchers
            ) else "field_name"
        else:
            match_target = hit_target

        return SecurityTag(
            level=level,
            category=category,
            source_engine="RULE",
            rule_id=rule.id,
            domain=self.domain,
            standard_id=self.standard_id,
            match_target=match_target,
        )

    def _execute_matcher(self, matcher: MatcherDef, field_name: str, str_value: str) -> OperatorResult:
        """执行单个匹配器，返回归一化的 OperatorResult / Execute a single matcher, returning a normalized OperatorResult.

        Steps:
        1. Look up the operator function from the registry.
        2. Determine target value (field_name or str_value based on matcher.target).
        3. Execute the operator, normalize result, and record metrics.
        4. Handle exceptions gracefully (fail-safe: return a "miss" result).
        """
        try:
            op_func = OperatorRegistry.get(matcher.operator)
        except KeyError:
            DYNCLASSIFICATION_OPERATOR_CALLS_TOTAL.labels(
                operator=matcher.operator, result="miss"
            ).inc()
            return OperatorResult(hit=False)

        target_value = field_name if matcher.target == "field_name" else str_value
        if target_value is None or target_value == "":
            DYNCLASSIFICATION_OPERATOR_CALLS_TOTAL.labels(
                operator=matcher.operator, result="miss"
            ).inc()
            return OperatorResult(hit=False)

        try:
            raw = op_func(target_value, matcher.params)
            op_result = normalize_result(raw)
            DYNCLASSIFICATION_OPERATOR_CALLS_TOTAL.labels(
                operator=matcher.operator, result="hit" if op_result.hit else "miss"
            ).inc()
            return op_result
        except Exception as exc:
            DYNCLASSIFICATION_OPERATOR_ERRORS_TOTAL.labels(
                operator=matcher.operator, rule_id=""
            ).inc()
            logger.warning(
                "operator_execution_failed",
                extra={
                    "operator": matcher.operator,
                    "field_name": field_name,
                    "error": str(exc),
                },
            )
            return OperatorResult(hit=False)

    def _evaluate_downgrade(self, field_name: str) -> list[SecurityTag]:
        """执行降级规则 / Execute downgrade rules.

        Downgrade rules check if the normalized field name contains any of the
        rule's keywords. If so, a lower-sensitivity tag is produced.
        Typical use: operational/public fields downgraded to L1/L2.

        输入 / Input: field_name = "Turnover_Rate"
          │
          ▼ 归一化 / Normalize
        norm_name = "turnoverrate"
          │
          ▼ 遍历每条降级规则 / Iterate each downgrade rule
          ├─ RULE_DOWN_PUBLIC: keywords=["publicreport","annualsummary","科普"] → 不包含 / Not included → 跳过 / Skip
          └─ RULE_DOWN_OPS:    keywords=["turnoverrate","deviceusage","inventory"] → 包含! / Included! → 生成 L2 标签 / Generate L2 tag
          │
          ▼
        输出 / Output: [SecurityTag(level="L2", category="OPERATIONAL_STAT", is_override=True, ...)]
        """
        tags: list[SecurityTag] = []
        # Normalize field name: lowercase + remove underscores/spaces for matching.
        norm_name = field_name.lower().replace("_", "").replace(" ", "")

        for rule in self.downgrade_rules:
            # Normalize each keyword the same way for consistent comparison.
            keywords = [kw.lower().replace("_", "").replace(" ", "") for kw in rule.keywords]
            # Check if any keyword is a substring of the normalized field name.
            if any(kw in norm_name for kw in keywords):
                # Keyword matched: produce a downgrade tag with the rule's target level.
                # Mark is_override=True if this rule has force_suppress capability enabled.
                # Mark is_downgrade=True for all downgrade tags (used by funnel conflict detection).
                tags.append(
                    SecurityTag(
                        level=rule.level,
                        category=rule.category,
                        source_engine="RULE",
                        rule_id=rule.id,
                        domain=self.domain,
                        standard_id=self.standard_id,
                        is_override=rule.force_suppress,
                        is_downgrade=True,
                    )
                )
        return tags

    def _apply_override_suppression(
        self,
        normal_tags: list[SecurityTag],
        downgrade_tags: list[SecurityTag],
    ) -> Tuple[list[SecurityTag], list[SecurityTag]]:
        """对普通规则标签执行强制覆盖压制 / Apply forced override suppression to normal rule tags.

        返回一个元组 (surviving_tags, suppressed_tags) / Returns a tuple (surviving_tags, suppressed_tags).
        """
        # Step 1: Filter to only override-enabled downgrade tags.
        override_tags = [t for t in downgrade_tags if t.is_override]
        # Early exit: no override tags means no suppression needed.
        if not override_tags:
            return normal_tags, []

        # Step 2: Determine the suppression cap rank across all override tags.
        cap_ranks: list[int] = []
        for tag in override_tags:
            cap_level = self._get_override_cap_level(tag.rule_id, tag.level)
            cap_rank = self.taxonomy.get_level_rank(cap_level)
            if cap_rank > 0:
                cap_ranks.append(cap_rank)
        
        if not cap_ranks:
            return normal_tags, []
        # 安全保守原则：override 是对“安全优先”的例外豁免（降级），例外应从严解释。
        # 多条 override 规则同时命中时取最小 cap_rank（最弱压制能力），
        # 确保高敏感标签（rank > min_cap）不被错误降级而非法逃逸。
        min_cap_rank = min(cap_ranks)

        # Step 2.5: 合并所有 override 规则的 exempt_rules 豁免例外名单（取并集）。
        # 只要在任一命中的 override 规则的 exempt_rules 列表中（或匹配其通配符模式），
        # 该普通规则 ID 即可获得豁免保护、绝对不被压制。
        exempt_patterns: set[str] = set()
        for tag in override_tags:
            rule_def = self._find_downgrade_rule(tag.rule_id)
            if rule_def and rule_def.exempt_rules:
                exempt_patterns.update(rule_def.exempt_rules)

        # Step 3: Filter out normal tags whose rank <= cap_rank. / 过滤并压制 rank <= cap_rank 的普通标签。
        # 压制 4 重判定条件 / 4 Suppression Conditions:
        # 1. 非降级标签（is_override=False，降级标签自身不会互相压制）
        # 2. 不能是值级匹配标签（tag.match_target != 'field_value'，数据值扫描默认豁免保底）
        # 3. 标签敏感等级 <= 覆盖上限（tag_rank <= min_cap_rank）
        # 4. 豁免例外校验（如果规则 ID 匹配了 exempt_rules 豁免模式，则属于例外、受保护不被压制）
        surviving_tags: list[SecurityTag] = []
        suppressed_tags: list[SecurityTag] = []
        for tag in normal_tags:
            # 条件 1: 值级命中防护，永远不被降级规则强行压制 / Value-level hits are protected
            if tag.match_target == "field_value":
                surviving_tags.append(tag)
                continue
            tag_rank = self.taxonomy.get_level_rank(tag.level)
            # 条件 2: 确认敏感等级未超出压制上限 / Check level <= cap_rank
            if tag_rank <= min_cap_rank:
                # 条件 3: exempt_rules 豁免例外校验 (支持精确匹配及 fnmatch 通配符，如 'RULE_IDCARD_EXACT' 或 '*_EXACT')
                # 如果匹配了豁免名单，属于例外受保护，跳过压制、保留标签
                if exempt_patterns:
                    is_exempt = any(
                        p == tag.rule_id or fnmatch.fnmatch(tag.rule_id, p)
                        for p in exempt_patterns
                    )
                    if is_exempt:
                        surviving_tags.append(tag)
                        continue
                suppressed_tags.append(tag)
            else:
                surviving_tags.append(tag)

        # Step 4: Log suppressed tags + Prometheus metric.
        if suppressed_tags:
            logger.info(
                "downgrade_override_suppression",
                extra={
                    "suppressed_count": len(suppressed_tags),
                    "suppressed_tags": [str(t) for t in suppressed_tags],
                    "suppressed_rule_ids": [t.rule_id for t in suppressed_tags],
                    "override_rules": [t.rule_id for t in override_tags],
                    "cap_rank": min_cap_rank,
                    "has_exemptions": bool(exempt_patterns),
                },
            )

            for tag in suppressed_tags:
                DYNCLASSIFICATION_OVERRIDE_SUPPRESSED_TOTAL.labels(
                    domain=self.domain or "default",
                    suppressed_rule_id=tag.rule_id,
                ).inc()

        # Step 5: Return surviving and suppressed tags separately.
        return surviving_tags, suppressed_tags

    def _get_override_cap_level(self, rule_id: str, fallback_level: str) -> str:
        """获取降级规则的覆盖等级上限 / Get the override cap level of a downgrade rule."""
        rule = self._find_downgrade_rule(rule_id)
        if rule:
            return rule.max_force_suppress_level if rule.max_force_suppress_level else rule.level
        return fallback_level

    def _find_downgrade_rule(self, rule_id: str) -> DowngradeRuleDef | None:
        """根据 rule_id 查找降级规则定义 / Find downgrade rule definition by rule_id."""
        for rule in self.downgrade_rules:
            if rule.id == rule_id:
                return rule
        return None

    def _unique_tags(self, tags: list[SecurityTag]) -> list[SecurityTag]:
        """按 (level, category) 去重，保留首次出现 / Deduplicate by (level, category), keeping the first occurrence."""
        seen: set[tuple[str, str]] = set()
        result: list[SecurityTag] = []
        for tag in tags:
            key = (tag.level, tag.category)
            if key not in seen:
                seen.add(key)
                result.append(tag)
        return result

    @property
    def rule_count(self) -> int:
        """当前引擎的规则总数（仅包含已启用的普通规则） / Total number of rules in the current engine (only including enabled normal rules)."""
        return len(self.rules)

    @property
    def downgrade_rule_count(self) -> int:
        """当前引擎的降级规则总数 / Total number of downgrade rules in the current engine."""
        return len(self.downgrade_rules)