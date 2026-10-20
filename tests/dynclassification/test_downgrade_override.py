"""降级规则强制覆盖与豁免例外能力测试 / Downgrade Override & Exemption Feature Tests.

测试覆盖场景 / Test Scenarios Covered:
- force_suppress=false（默认）：行为与修改前完全一致，仅做降级兜底（向后兼容） / Default behavior, fallback only
- force_suppress=true + 普通规则 L3：L3 标签被强行压制，最终等级降为降级目标 L2 / Forced suppression down to L2
- force_suppress=true + 普通规则 L5：L5 标签不被压制（超出 max_force_suppress_level 上限），最终等级保持 L5 / Out-of-cap levels kept
- force_suppress=true + 无普通规则命中：行为与兜底模式一致 / Fallback mode when no normal rules match
- 多条覆盖规则同时命中：取最保守的覆盖上限（最小 max_force_suppress_level） / Take min cap rank for safety
- exempt_rules (exclude_rules) 豁免例外名单：默认空列表全额压制；列表内的规则 ID/通配符作为例外豁免保留 / Exemption whitelist & wildcards
"""

from __future__ import annotations

import pytest

from engine.dynclassification import (
    ConfigurableRuleEngine,
    DomainTaxonomy,
    SecurityTag,
)
from engine.dynclassification.rule_schema import (
    DowngradeRuleDef,
    MatcherDef,
    RuleDef,
    RuleProfile,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def taxonomy() -> DomainTaxonomy:
    """构造 L1~L5 分类体系。"""
    return DomainTaxonomy(
        domain="test",
        standard_id="TEST",
        levels={
            "L1": {"id": "L1", "name": "公开", "rank": 1},
            "L2": {"id": "L2", "name": "内部", "rank": 2},
            "L3": {"id": "L3", "name": "敏感", "rank": 3},
            "L4": {"id": "L4", "name": "高敏感", "rank": 4},
            "L5": {"id": "L5", "name": "极敏感", "rank": 5},
        },
        default_level="L3",
    )


@pytest.fixture()
def profile_with_broad_rule() -> RuleProfile:
    """构造一个包含宽泛关键词规则 + 覆盖型降级规则的 Profile。

    模拟场景：宽泛规则匹配关键词 "report" → L3，
    但运营字段降级规则 force_suppress=true 可将其压制到 L2。
    """
    return RuleProfile(
        domain="test-override",
        rules=[
            # 宽泛规则：字段名包含 "report" 就判为 L3（可能误中运营字段）
            RuleDef(
                id="RULE_BROAD_REPORT",
                name="宽泛报告规则",
                category="GENERAL_REPORT",
                level="L3",
                priority=50,
                matchers=[
                    MatcherDef(
                        target="field_name",
                        operator="keyword_contains",
                        params={"use_word_boundaries": False, "keywords": ["report"]},
                    )
                ],
            ),
            # 高敏感规则：字段名包含 "genome" 判为 L5（不应被降级）
            RuleDef(
                id="RULE_GENOME",
                name="基因组规则",
                category="GENOMIC",
                level="L5",
                priority=200,
                matchers=[
                    MatcherDef(
                        target="field_name",
                        operator="keyword_contains",
                        params={"use_word_boundaries": False, "keywords": ["genome"]},
                    )
                ],
            ),
        ],
        downgrade_rules=[
            # 非覆盖型降级规则（默认行为，仅兜底）
            DowngradeRuleDef(
                id="RULE_DOWN_PUBLIC",
                name="公开数据降级",
                keywords=["public_summary"],
                level="L1",
                category="PUBLIC_REPORT",
                force_suppress=False,  # 显式标注为 false
            ),
            # 覆盖型降级规则：可压制 L3 及以下的普通规则
            DowngradeRuleDef(
                id="RULE_DOWN_OPS",
                name="运营统计强制降级",
                keywords=["turnover", "device_usage", "annual_report"],
                level="L2",
                category="OPERATIONAL_STAT",
                force_suppress=True,
                max_force_suppress_level="L3",  # 仅能压制 L3 及以下
            ),
        ],
    )


@pytest.fixture()
def engine(taxonomy, profile_with_broad_rule) -> ConfigurableRuleEngine:
    """构造带有覆盖型降级规则的引擎。"""
    return ConfigurableRuleEngine(
        taxonomy=taxonomy,
        profiles=[profile_with_broad_rule],
        domain="test",
        standard_id="TEST",
    )


# ===========================================================================
# 测试：向后兼容（force_suppress=false）
# ===========================================================================


class TestBackwardCompatibility:
    """验证 force_suppress=false 时行为与修改前完全一致。"""

    def test_non_override_downgrade_cannot_suppress(self, taxonomy):
        """非覆盖型降级规则不能压制普通规则标签。

        场景：普通规则命中 L3 + 非覆盖降级规则命中 L1
        期望：两个标签都保留，最终等级取 max = L3
        """
        profile = RuleProfile(
            domain="compat",
            rules=[
                RuleDef(
                    id="RULE_A",
                    category="CAT_A",
                    level="L3",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["data"]})],
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_A",
                    keywords=["data"],
                    level="L1",
                    category="PUBLIC",
                    force_suppress=False,  # 非覆盖型
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, _ = engine.evaluate("some_data_field", "value")

        # 两个标签都应存在
        levels = {t.level for t in tags}
        assert "L3" in levels, "普通规则标签不应被压制"
        assert "L1" in levels, "降级标签应存在"

    def test_old_yaml_without_override_field(self, taxonomy):
        """无 force_suppress 字段的旧配置应正常加载（默认 false）。"""
        # 模拟旧 YAML：不传 force_suppress 参数
        rule = DowngradeRuleDef(
            id="OLD_RULE",
            keywords=["test"],
            level="L2",
            category="OLD_CAT",
        )
        assert rule.force_suppress is False
        assert rule.max_force_suppress_level == ""


# ===========================================================================
# 测试：强制覆盖核心功能
# ===========================================================================


class TestOverrideSuppression:
    """验证 force_suppress=true 时的强制覆盖行为。"""

    def test_override_suppresses_l3_normal_tag(self, engine):
        """覆盖型降级规则应压制 L3 普通规则标签。

        场景：字段 "annual_report_turnover" 同时命中：
          - 宽泛规则 RULE_BROAD_REPORT（含 "report"）→ L3
          - 降级规则 RULE_DOWN_OPS（含 "turnover"）→ L2, force_suppress=true, cap=L3
        期望：L3 标签被压制，仅保留 L2 降级标签
        """
        tags, _ = engine.evaluate("annual_report_turnover", "some_value")

        # L3 标签应被压制
        levels = [t.level for t in tags]
        assert "L3" not in levels, "L3 普通规则标签应被覆盖型降级规则压制"
        # L2 降级标签应存在
        assert "L2" in levels, "降级标签应保留"
        # 验证降级标签的 is_override 标记
        override_tags = [t for t in tags if t.is_override]
        assert len(override_tags) == 1
        assert override_tags[0].rule_id == "RULE_DOWN_OPS"

    def test_override_cannot_suppress_l5(self, engine):
        """覆盖型降级规则不能压制超出上限的高等级标签。

        场景：字段 "genome_report" 同时命中：
          - 基因组规则 RULE_GENOME（含 "genome"）→ L5
          - 宽泛规则 RULE_BROAD_REPORT（含 "report"）→ L3
          - 降级规则 RULE_DOWN_OPS（含 "turnover"）→ 不命中
        期望：L5 不被压制（超出 cap=L3），L3 也保留（无 override 命中）
        """
        tags, _ = engine.evaluate("genome_report", "some_value")

        levels = [t.level for t in tags]
        assert "L5" in levels, "L5 标签不应被压制（超出覆盖上限）"
        assert "L3" in levels, "无 override 降级命中时 L3 应保留"

    def test_override_with_no_normal_rules(self, engine):
        """无普通规则命中时，覆盖型降级规则作为兜底。

        场景：字段 "turnover_rate" 仅命中降级规则（无普通规则含 "turnover"）
        期望：降级标签正常产出
        """
        tags, _ = engine.evaluate("turnover_rate", "15.3")

        levels = [t.level for t in tags]
        assert "L2" in levels
        # 无普通规则标签
        normal_tags = [t for t in tags if not t.is_override]
        assert len(normal_tags) == 0

    def test_override_suppresses_l3_but_keeps_l5(self, taxonomy):
        """同时存在 L3 和 L5 普通规则时，仅 L3 被压制。

        场景：字段名同时匹配 L3 规则和 L5 规则，且 override cap=L3
        期望：L3 被压制，L5 保留
        """
        profile = RuleProfile(
            domain="mixed",
            rules=[
                RuleDef(
                    id="RULE_L3",
                    category="CAT_L3",
                    level="L3",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["stats"]})],
                ),
                RuleDef(
                    id="RULE_L5",
                    category="CAT_L5",
                    level="L5",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["genome"]})],
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_OPS",
                    keywords=["stats"],
                    level="L2",
                    category="OPS",
                    force_suppress=True,
                    max_force_suppress_level="L3",
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, _ = engine.evaluate("genome_stats_field", "value")

        levels = [t.level for t in tags]
        assert "L3" not in levels, "L3 应被压制"
        assert "L5" in levels, "L5 不应被压制"
        assert "L2" in levels, "降级标签应存在"


# ===========================================================================
# 测试：边界情况
# ===========================================================================


class TestEdgeCases:
    """边界情况测试。"""

    def test_multiple_override_rules_fire(self, taxonomy):
        """多条覆盖型降级规则同时命中时，取最小 cap_rank（安全保守原则）。

        安全保守语义: override 是对“安全优先”的例外豁免，例外应从严解释。
        两条规则 cap=L3(保守) 和 cap=L4(激进) 同时命中 → 取 min=L3 →
        仅压制 rank<=3 的标签，L4 存活。
        """
        profile = RuleProfile(
            domain="multi",
            rules=[
                RuleDef(
                    id="RULE_L4",
                    category="CAT_L4",
                    level="L4",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["data"]})],
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_A",
                    keywords=["data"],
                    level="L1",
                    category="PUBLIC",
                    force_suppress=True,
                    max_force_suppress_level="L3",  # cap=L3（保守）
                ),
                DowngradeRuleDef(
                    id="DOWN_B",
                    keywords=["data"],
                    level="L2",
                    category="OPS",
                    force_suppress=True,
                    max_force_suppress_level="L4",  # cap=L4（激进）
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, _ = engine.evaluate("some_data_field", "value")

        # 安全保守: min(L3, L4) = L3 → L4 (rank=4) > cap_rank(3) → L4 存活
        levels = [t.level for t in tags]
        assert "L4" in levels, "安全保守原则: L4 不应被压制(min_cap=L3)"

    def test_empty_max_force_suppress_level_uses_rule_level(self, taxonomy):
        """max_force_suppress_level 为空时，使用规则自身 level 作为 cap。"""
        profile = RuleProfile(
            domain="fallback",
            rules=[
                RuleDef(
                    id="RULE_L2",
                    category="CAT_L2",
                    level="L2",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["info"]})],
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_X",
                    keywords=["info"],
                    level="L1",
                    category="PUBLIC",
                    force_suppress=True,
                    max_force_suppress_level="",  # 空 → 使用 level="L1" 作为 cap
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, _ = engine.evaluate("some_info_field", "value")

        # cap=L1 (rank=1)，L2 (rank=2) > cap，不应被压制
        levels = [t.level for t in tags]
        assert "L2" in levels, "L2 不应被 cap=L1 的覆盖规则压制"

    def test_is_override_flag_on_tags(self, engine):
        """验证 is_override 标记正确设置在降级标签上。"""
        tags, _ = engine.evaluate("turnover_rate", "value")
        override_tags = [t for t in tags if t.is_override]
        non_override_tags = [t for t in tags if not t.is_override]

        assert all(t.rule_id == "RULE_DOWN_OPS" for t in override_tags)
        # 非覆盖标签不应有 is_override=True
        assert all(not t.is_override for t in non_override_tags)


# ===========================================================================
# 测试：exempt_rules 豁免例外名单精细控制
# ===========================================================================


class TestExemptRulesExceptions:
    """验证 exempt_rules (exclude_rules) 豁免例外名单的行为。"""

    def test_exempt_list_protects_listed_rules(self, taxonomy):
        """豁免名单非空时，列表中的规则作为例外获得保护，其他规则压制。

        场景：字段同时命中 RULE_A(L3) 和 RULE_B(L3)，
        降级规则 exempt_rules=["RULE_B"] 保护 RULE_B 不被压制。
        期望：RULE_A 被压制，RULE_B 属于例外被豁免保留。
        """
        profile = RuleProfile(
            domain="exempt-list",
            rules=[
                RuleDef(
                    id="RULE_A",
                    category="CAT_A",
                    level="L3",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["data"]})],
                ),
                RuleDef(
                    id="RULE_B",
                    category="CAT_B",
                    level="L3",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["data"]})],
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_EX",
                    keywords=["data"],
                    level="L2",
                    category="OPS",
                    force_suppress=True,
                    max_force_suppress_level="L3",
                    exempt_rules=["RULE_B"],  # RULE_B 为例外豁免保护
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, _ = engine.evaluate("some_data_field", "value")

        rule_ids = [t.rule_id for t in tags]
        assert "RULE_A" not in rule_ids, "RULE_A 应被压制"
        assert "RULE_B" in rule_ids, "RULE_B 属于例外名单，应豁免保留"
        assert "DOWN_EX" in rule_ids, "降级标签应存在"

    def test_empty_exempt_list_suppresses_all(self, taxonomy):
        """豁免名单为空时（默认），没有例外，全额压制所有符合条件的规则。

        场景：字段同时命中 RULE_A(L3) 和 RULE_B(L3)，
        降级规则 exempt_rules=[]（没有例外）。
        期望：两个规则都被压制。
        """
        profile = RuleProfile(
            domain="no-exempt-list",
            rules=[
                RuleDef(
                    id="RULE_A",
                    category="CAT_A",
                    level="L3",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["data"]})],
                ),
                RuleDef(
                    id="RULE_B",
                    category="CAT_B",
                    level="L3",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["data"]})],
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_ALL",
                    keywords=["data"],
                    level="L2",
                    category="OPS",
                    force_suppress=True,
                    max_force_suppress_level="L3",
                    exempt_rules=[],  # 空 = 没有例外全额压制
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, _ = engine.evaluate("some_data_field", "value")

        rule_ids = [t.rule_id for t in tags]
        assert "RULE_A" not in rule_ids, "RULE_A 应被压制"
        assert "RULE_B" not in rule_ids, "RULE_B 应被压制"
        assert "DOWN_ALL" in rule_ids, "降级标签应存在"

    def test_whitelist_still_respects_cap_rank(self, taxonomy):
        """白名单内的规则如果等级超出 cap，仍然不被压制。

        场景：RULE_HIGH(L5) 在白名单中，但 cap=L3，
        期望：L5 不被压制（rank 超出 cap）。
        """
        profile = RuleProfile(
            domain="cap-check",
            rules=[
                RuleDef(
                    id="RULE_HIGH",
                    category="CAT_HIGH",
                    level="L5",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["genome"]})],
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_CAP",
                    keywords=["genome"],
                    level="L2",
                    category="OPS",
                    force_suppress=True,
                    max_force_suppress_level="L3",  # cap=L3，不能压制 L5
                    exempt_rules=[],  # 没有例外，全额压制符合条件的规则
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, _ = engine.evaluate("genome_data", "value")

        levels = [t.level for t in tags]
        assert "L5" in levels, "L5 不应被压制（rank 超出 cap=L3）"

    def test_value_level_hits_always_protected(self, taxonomy):
        """值级扫描命中标签受引擎内置机制保底保护，不受强制压制影响。

        场景：RULE_PHONE 为基于实际数据值（field_value）匹配出的真实手机号标签，
        即使降级规则配置了 force_suppress=true 且 exempt_rules=[]（未配置任何显式豁免），
        期望：值级扫描出来的敏感数据仍受内置机制保护，绝对不被压制擦除。
        """
        profile = RuleProfile(
            domain="value-exempt",
            rules=[
                RuleDef(
                    id="RULE_PHONE",
                    category="PHONE",
                    level="L3",
                    matchers=[MatcherDef(target="field_value", operator="regex", params={"pattern": r"^1[3-9]\d{9}$"})],
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_VL",
                    keywords=["contact"],
                    level="L2",
                    category="OPS",
                    force_suppress=True,
                    max_force_suppress_level="L3",
                    exempt_rules=[],  # 即使未配置任何显式豁免规则 / Even with empty exempt_rules
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, _ = engine.evaluate("contact_info", "13800138000")

        rule_ids = [t.rule_id for t in tags]
        assert "RULE_PHONE" in rule_ids, "基于数据采样值(field_value)命中的敏感标签应受内置保底机制保护，绝对不被压制"

    def test_multiple_override_rules_exempt_union(self, taxonomy):
        """多条 override 规则的豁免例外名单取并集。

        场景：DOWN_A exempt_rules=["RULE_Z"]，DOWN_B exempt_rules=["RULE_Y"]，
        字段命中 RULE_X + RULE_Y + RULE_Z，
        期望：RULE_Y 和 RULE_Z 均属于例外名单保护保留，RULE_X 被压制。
        """
        profile = RuleProfile(
            domain="union",
            rules=[
                RuleDef(
                    id="RULE_X",
                    category="CAT_X",
                    level="L3",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["data"]})],
                ),
                RuleDef(
                    id="RULE_Y",
                    category="CAT_Y",
                    level="L3",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["data"]})],
                ),
                RuleDef(
                    id="RULE_Z",
                    category="CAT_Z",
                    level="L3",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"use_word_boundaries": False, "keywords": ["data"]})],
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_A",
                    keywords=["data"],
                    level="L2",
                    category="OPS_A",
                    force_suppress=True,
                    max_force_suppress_level="L3",
                    exempt_rules=["RULE_Z"],
                ),
                DowngradeRuleDef(
                    id="DOWN_B",
                    keywords=["data"],
                    level="L2",
                    category="OPS_B",
                    force_suppress=True,
                    max_force_suppress_level="L3",
                    exempt_rules=["RULE_Y"],
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, _ = engine.evaluate("some_data_field", "value")

        rule_ids = [t.rule_id for t in tags]
        assert "RULE_X" not in rule_ids, "RULE_X 不在任何豁免名单中，应被压制"
        assert "RULE_Y" in rule_ids, "RULE_Y 在 DOWN_B 豁免名单中，应保留"
        assert "RULE_Z" in rule_ids, "RULE_Z 在 DOWN_A 豁免名单中，应保留"

    def test_exempt_rules_wildcard_matching(self, taxonomy: DomainTaxonomy):
        """测试 exempt_rules 中的通配符（如 'pkg:*_EXACT'）豁免保护能力。"""
        profile = RuleProfile(
            domain="test",
            rules=[
                RuleDef(
                    id="pkg:RULE_BROAD_1",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"keywords": ["data"]})],
                    level="L3",
                    category="C1",
                ),
                RuleDef(
                    id="pkg:RULE_EXACT_2",
                    matchers=[MatcherDef(target="field_name", operator="keyword_contains", params={"keywords": ["data"]})],
                    level="L3",
                    category="C2",
                ),
            ],
            downgrade_rules=[
                DowngradeRuleDef(
                    id="DOWN_WILD",
                    keywords=["data"],
                    level="L2",
                    category="OPS",
                    force_suppress=True,
                    max_force_suppress_level="L3",
                    exempt_rules=["pkg:*_EXACT_2"],  # 使用通配符将 EXACT 规则作为例外保护保留
                ),
            ],
        )
        engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
        tags, suppressed = engine.evaluate("my_data_field", "sample")

        surviving_ids = [t.rule_id for t in tags]
        suppressed_ids = [t.rule_id for t in suppressed]

        assert "pkg:RULE_BROAD_1" in suppressed_ids, "pkg:RULE_BROAD_1 不在豁免名单中，应被压制"
        assert "pkg:RULE_EXACT_2" in surviving_ids, "pkg:RULE_EXACT_2 匹配通配符豁免名单，应被保留"

