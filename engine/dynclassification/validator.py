"""规则配置校验工具与拼写推荐 / Rule Profile Validator & Fuzzy Recommender.

提供对 rules/ 目录下 YAML 文件的离线/在线校验，检查：
Provides offline/online validation for YAML files under rules/ directory, checking:
1. YAML 语法正确性与 Pydantic Schema 模型校验 / YAML syntax correctness and Pydantic Schema model validation.
2. 规则配置中使用的算子是否已在 OperatorRegistry 中注册 / Whether operators used in rule configs are registered in OperatorRegistry.
3. 规则配置中使用的分类类别和敏感等级是否在对应的 Taxonomy 中有定义 / Whether categories and sensitivity levels used in rule configs are defined in the corresponding Taxonomy.
4. 如果算子拼写错误，提供拼写相似度建议 (Did you mean 'regex'?) / If an operator is misspelled, provide fuzzy spelling suggestions.
"""

from __future__ import annotations

# difflib provides fuzzy string matching for "did you mean?" suggestions
import difflib
from pathlib import Path

# PyYAML for parsing rule configuration files
import yaml

from .models import DomainTaxonomy
from .operator_registry import OperatorRegistry
from .rule_schema import RuleProfile, StandardDef


class ValidationResult:
    """校验结果模型 / Validation Result Model.

    Accumulates errors and warnings during validation.
    在校验期间积累错误和警告。
    is_valid is set to False as soon as any error is added.
    一旦添加任何错误，is_valid 就会设置为 False。
    """

    def __init__(self):
        # Overall validity flag: True until first error is encountered.
        # 整体有效性标志：在遇到第一个错误之前为 True。
        self.is_valid: bool = True
        # Critical issues that prevent correct engine operation.
        # 阻止引擎正确运行的严重问题。
        self.errors: list[str] = []
        # Non-critical issues that may indicate misconfiguration.
        # 可能表明配置错误的非严重问题。
        self.warnings: list[str] = []

    def add_error(self, msg: str) -> None:
        """Record a validation error and mark result as invalid / 记录校验错误并将结果标记为无效。"""
        self.is_valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        """Record a non-critical warning (does not affect is_valid) / 记录非严重警告（不影响 is_valid）。"""
        self.warnings.append(msg)


def _suggest_similar_operator(op_name: str) -> str:
    """使用 Levenshtein/difflib 推荐最相似的算子名称 / Recommend the most similar operator name using Levenshtein/difflib.

    Uses difflib.get_close_matches with a 0.5 similarity cutoff to find
    the closest registered operator name. Returns a formatted suggestion
    string or empty string if no close match exists.
    使用 cutoff=0.5 的 difflib.get_close_matches 寻找最接近的已注册算子名称。如果存在相似匹配，则返回格式化的建议字符串，否则返回空字符串。
    """
    # Get all registered operator names as the candidate pool.
    available = OperatorRegistry.list_operators()
    # Find the single closest match with at least 50% similarity.
    matches = difflib.get_close_matches(op_name, available, n=1, cutoff=0.5)
    if matches:
        return f"（您是否想输入 '{matches[0]}'？）"
    return ""


def validate_rules_dir(rules_dir: str | Path = "rules") -> ValidationResult:
    """校验规则目录下的所有 YAML 文件合法性 / Validate the legality of all YAML files in the rules directory.

    Validation pipeline / 校验流程:
    1. Validate all taxonomy YAML files against DomainTaxonomy schema / 根据 DomainTaxonomy 模式校验所有 taxonomy YAML 文件。
    2. Validate all domain profile YAML files against RuleProfile schema,
       and check that every operator referenced in matchers is registered / 根据 RuleProfile 模式校验所有 domain profile YAML 文件，并检查 matchers 中引用的每个算子是否已注册。
    3. Validate all standard YAML files against StandardDef schema,
       and check that referenced taxonomy files exist / 根据 StandardDef 模式校验所有 standard YAML 文件，并检查引用的 taxonomy 文件是否存在。

    Args:
        rules_dir: 规则配置根目录 / Rule configuration root directory.

    Returns:
        ValidationResult 对象包含校验结果、错误列表与警告列表 / ValidationResult object containing validation result, error list, and warning list.
    """
    rules_path = Path(rules_dir)
    res = ValidationResult()

    # Early exit if the rules directory does not exist at all.
    if not rules_path.exists():
        res.add_error(f"规则配置目录不存在: {rules_path}")
        return res

    # Resolve the three expected subdirectories.
    tax_dir = rules_path / "taxonomies"
    dom_dir = rules_path / "domains"
    std_dir = rules_path / "standards"

    # --- Phase 1: Validate taxonomy definitions ---
    taxonomies: dict[str, DomainTaxonomy] = {}
    if tax_dir.exists():
        for yaml_file in tax_dir.glob("*.yaml"):
            try:
                # Parse YAML content and validate against DomainTaxonomy Pydantic schema.
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                tax = DomainTaxonomy.model_validate(data)
                # Cache valid taxonomies for cross-referencing in Phase 3.
                taxonomies[yaml_file.stem] = tax
            except Exception as exc:
                # Any parse/validation failure is a critical error.
                res.add_error(f"[Taxonomy 校验失败] {yaml_file.name}: {exc}")

    # --- Phase 1.5: Parse standards first to map domain -> referenced taxonomies ---
    # 领域包本身不强制声明 taxonomy，其等级体系由引用它的 StandardDef 决定
    # （如 gd_health 的 G1~G4 由 standards/gd_health.yaml 的 taxonomy 指定）。
    # 必须先解析 standards，才能按各 profile 自身归属的 taxonomy 分别校验等级引用，
    # 避免用单一 taxonomy（如 default 的 L1~L5）校验所有领域包造成误报。
    domain_taxonomy_refs: dict[str, set[str]] = {}
    if std_dir.exists():
        for yaml_file in std_dir.glob("*.yaml"):
            try:
                # Parse and validate against StandardDef schema.
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                std = StandardDef.model_validate(data)

                # Check that the referenced taxonomy file exists (either already loaded
                # in Phase 1 or physically present on disk).
                if std.taxonomy not in taxonomies and not (tax_dir / f"{std.taxonomy}.yaml").exists():
                    res.add_warning(
                        f"[Standard 引用提醒] 文件 {yaml_file.name} 引用了未找到的 Taxonomy: '{std.taxonomy}'"
                    )
                # Record domain -> taxonomy association for per-profile validation in Phase 2.
                for domain_name in std.domains:
                    domain_taxonomy_refs.setdefault(domain_name, set()).add(std.taxonomy)
            except Exception as exc:
                res.add_error(f"[Standard 校验失败] {yaml_file.name}: {exc}")

    # --- Phase 2: Validate domain profiles and their operator references ---
    if dom_dir.exists():
        for yaml_file in dom_dir.glob("*.yaml"):
            try:
                # Parse and validate against RuleProfile schema.
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                profile = RuleProfile.model_validate(data)

                # Check each rule's matchers reference valid registered operators.
                for rule in profile.rules:
                    for matcher in rule.matchers:
                        op_name = matcher.operator
                        # Verify the operator exists in the global registry.
                        if not OperatorRegistry.has(op_name):
                            # Provide a fuzzy suggestion for common typos.
                            suggestion = _suggest_similar_operator(op_name)
                            res.add_error(
                                f"[Domain 规则算子未找到] 文件 {yaml_file.name}, 规则 '{rule.id}', 算子 '{op_name}' 未在注册表中找到{suggestion}"
                            )

                # --- 降级规则字段合法性（按该 profile 自身归属的 taxonomy 分别校验） ---
                _validate_downgrade_rules(
                    profile,
                    yaml_file.name,
                    _resolve_profile_taxonomies(profile, yaml_file.stem, taxonomies, domain_taxonomy_refs),
                    res,
                )

                # --- 新增校验: 规则 ID 唯一性 ---
                _validate_rule_id_uniqueness(profile, yaml_file.name, res)

            except Exception as exc:
                res.add_error(f"[Domain Profile 校验失败] {yaml_file.name}: {exc}")

    return res


def _resolve_profile_taxonomies(
    profile: RuleProfile,
    file_stem: str,
    taxonomies: dict[str, DomainTaxonomy],
    domain_taxonomy_refs: dict[str, set[str]],
) -> dict[str, DomainTaxonomy]:
    """确定校验某个领域包等级引用时所依据的候选 Taxonomy 集合。

    解析规则（与 profile_loader 的运行时行为对齐）:
    1. profile.default_taxonomy 显式声明的 taxonomy；
    2. 引用了该领域包的 StandardDef 所声明的 taxonomy（一个领域包可被多个标准组合引用，
       如 general-pii 同时用于 default 引擎与 jrt0197 标准）；
    3. 内置 "default" taxonomy 兜底（_build_engine_from_domain 在 profile 未声明
       default_taxonomy 时回退到 "default"）。

    等级引用只要在任一候选 taxonomy 中存在即视为合法，避免跨体系组合（L/C/G 混用）产生误报。
    """
    candidates: dict[str, DomainTaxonomy] = {}
    if profile.default_taxonomy and profile.default_taxonomy in taxonomies:
        candidates[profile.default_taxonomy] = taxonomies[profile.default_taxonomy]
    for tax_name in domain_taxonomy_refs.get(file_stem, ()):
        if tax_name in taxonomies:
            candidates[tax_name] = taxonomies[tax_name]
    if "default" in taxonomies:
        candidates.setdefault("default", taxonomies["default"])
    return candidates


# ===========================================================================
# 内部校验工具函数 / Internal Validation Helpers
# ===========================================================================


def _validate_downgrade_rules(
    profile: RuleProfile,
    file_name: str,
    taxonomies: dict[str, DomainTaxonomy],
    res: ValidationResult,
) -> None:
    """校验降级规则的新字段合法性 / Validate the legality of new fields in downgrade rules.

    检查项 / Checks:
    1. max_force_suppress_level 在 taxonomy 中存在（拼错如 "L33" 会导致 force_suppress 静默失效） / max_force_suppress_level exists in taxonomy.
    2. force_suppress=false 却配置了 max_force_suppress_level 属于死配置（告警） / force_suppress=false but configured max_force_suppress_level is a dead config (warning).
    3. 降级规则的 level 在 taxonomy 中存在 / Downgrade rule's level exists in taxonomy.
    4. 普通规则的 level 在 taxonomy 中存在（rank=0 会被任何 force_suppress 规则压制） / Normal rule's level exists in taxonomy (rank=0 will be suppressed by any force_suppress rule).

    多 taxonomy 说明 / Multi-taxonomy note:
        `taxonomies` 为该 profile 的候选 taxonomy 集合（见 _resolve_profile_taxonomies），
        等级引用只要在任一候选中存在即视为合法，避免用单一 taxonomy 校验所有领域包
        造成跨体系误报（如 gd_health 的 G1~G4 被 default 的 L1~L5 误判为不存在）。
    """
    # 候选 taxonomy 为空时跳过等级存在性检查（无参考体系可校验）
    candidate_names = sorted(taxonomies.keys())

    def _level_exists(level_id: str) -> bool:
        """等级在任一候选 taxonomy 中存在即视为合法。"""
        return any(level_id in tax.levels for tax in taxonomies.values())

    # 校验降级规则
    for rule in profile.downgrade_rules:
        # 检查 1: max_force_suppress_level 存在性
        if rule.max_force_suppress_level and taxonomies:
            if not _level_exists(rule.max_force_suppress_level):
                res.add_error(
                    f"[降级规则等级未找到] 文件 {file_name}, 规则 '{rule.id}': "
                    f"max_force_suppress_level='{rule.max_force_suppress_level}' 在候选 taxonomy {candidate_names} 中均不存在"
                )

        # 检查 2: 死配置告警
        if not rule.force_suppress and rule.max_force_suppress_level:
            res.add_warning(
                f"[死配置] 文件 {file_name}, 规则 '{rule.id}': "
                f"force_suppress=false 但配置了 max_force_suppress_level='{rule.max_force_suppress_level}'，该配置不会生效"
            )

        # 检查 2.1: 开启强制覆盖但未配置 max_force_suppress_level 提示告警
        if rule.force_suppress and not rule.max_force_suppress_level:
            res.add_warning(
                f"[配置提示] 文件 {file_name}, 规则 '{rule.id}': "
                f"已开启 force_suppress=true，但未指定 max_force_suppress_level（默认仅压制 <= '{rule.level}' 的低/同级标签）。"
                f"若需将更高等级误报（如更高等级 L3/L4）强行降级为 '{rule.level}'，请显式配置 max_force_suppress_level: 'L3' (或更高等级)"
            )

        # 检查 3: 降级规则 level 存在性
        if taxonomies and not _level_exists(rule.level):
            res.add_error(
                f"[降级规则等级未找到] 文件 {file_name}, 规则 '{rule.id}': "
                f"level='{rule.level}' 在候选 taxonomy {candidate_names} 中均不存在"
            )

        # 检查 5: exempt_rules 豁免例外名单校验
        if rule.exempt_rules:
            if not rule.force_suppress:
                res.add_warning(
                    f"[死配置] 文件 {file_name}, 规则 '{rule.id}': "
                    f"force_suppress=false 但配置了 exempt_rules/exclude_rules，该配置不会生效"
                )
            normal_rule_ids = {r.id for r in profile.rules}
            for ref_id in rule.exempt_rules:
                # 支持通配符，非通配符时检查精确匹配存在性
                if "*" not in ref_id and "?" not in ref_id and ref_id not in normal_rule_ids:
                    res.add_warning(
                        f"[豁免名单引用未找到] 文件 {file_name}, 规则 '{rule.id}': "
                        f"exempt_rules/exclude_rules 中引用的 '{ref_id}' 在普通规则中不存在"
                    )

    # 检查 4: 普通规则 level 存在性
    if taxonomies:
        for rule in profile.rules:
            if not _level_exists(rule.level):
                res.add_error(
                    f"[规则等级未找到] 文件 {file_name}, 规则 '{rule.id}': "
                    f"level='{rule.level}' 在候选 taxonomy {candidate_names} 中均不存在"
                    f"（rank=0 会被任何 force_suppress 规则压制）"
                )


def _validate_rule_id_uniqueness(
    profile: RuleProfile,
    file_name: str,
    res: ValidationResult,
) -> None:
    """校验规则 ID 唯一性 / Validate rule ID uniqueness.

    多领域包合并后若 id 重复，_get_override_cap_level 会取第一个匹配项，
    可能导致非预期行为。
    If IDs are duplicated after merging multiple domains, _get_override_cap_level will pick the first match,
    which may lead to unexpected behavior.
    """
    seen_ids: dict[str, str] = {}  # rule_id -> rule type
    all_rules = (
        [(r.id, "普通规则") for r in profile.rules]
        + [(r.id, "降级规则") for r in profile.downgrade_rules]
        + [(r.id, "复合规则") for r in profile.composite_rules]
    )
    for rule_id, rule_type in all_rules:
        if rule_id in seen_ids:
            res.add_warning(
                f"[规则 ID 重复] 文件 {file_name}: "
                f"'{rule_id}' 同时用于{seen_ids[rule_id]}和{rule_type}"
            )
        else:
            seen_ids[rule_id] = rule_type
