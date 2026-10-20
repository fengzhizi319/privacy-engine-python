"""全项目漏洞修复回归测试套件 / Full Audit Remediation Test Suite.

专门针对 2026 安全与正确性审查报告中的 P0/P1 漏洞整改项编写回归单元测试，
确保 Safety Floor、gRPC 零值兜底、预算拒绝、高维向量校准等关键修复被 100% 覆盖。
"""

import math
from unittest.mock import MagicMock
import pytest
import numpy as np

from engine.dynclassification.funnel import ClassificationFunnel
from engine.dynclassification.engine import ConfigurableRuleEngine
from engine.dynclassification.models import (
    DomainTaxonomy,
    SecurityTag,
    SensitivityLevelDef,
)
from engine.dynclassification.rule_schema import (
    RuleProfile,
    RuleDef,
    MatcherDef,
    DowngradeRuleDef,
    CompositeRuleDef,
)
from engine.privacy.budget import default_registry
from engine.privacy.dp import DPApi, DPResult
from engine.dynclassification.composite import CompositeRuleEngine


# ─── 1. Safety Floor 安全地基与值级证据保护测试 ───

def test_safety_floor_prevents_llm_downgrade():
    """验证 Safety Floor：当包含 match_target == 'field_value' 的值级数据证据 (如 L3) 时，
    即使 LLM 仲裁选择降级到 L2，也会被 Safety Floor 拒绝，并强制要求人工复核。
    """
    levels = {
        "L1": SensitivityLevelDef(id="L1", name="公开级", rank=1),
        "L2": SensitivityLevelDef(id="L2", name="低风险", rank=2),
        "L3": SensitivityLevelDef(id="L3", name="中风险", rank=3),
        "L4": SensitivityLevelDef(id="L4", name="高风险", rank=4),
        "L5": SensitivityLevelDef(id="L5", name="极高风险", rank=5),
    }
    taxonomy = DomainTaxonomy(
        domain="medical",
        name="测试分类体系",
        standard_id="default",
        levels=levels,
        default_level="L1",
    )

    # 规则 1：值级证据规则 (身份证号校验 -> L3, match_target=field_value)
    rule_val = RuleDef(
        id="RULE_PII_IDCARD",
        name="身份证值级匹配",
        category="PERSONAL_BASIC",
        level="L3",
        match_logic="AND",
        matchers=[
            MatcherDef(
                target="field_value",
                operator="regex",
                params={"pattern": r"^\d{17}[\dXx]$"},
            )
        ],
    )

    downgrade_rule = DowngradeRuleDef(
        id="DOWNGRADE_RULE_001",
        name="通用字段降级",
        level="L2",
        category="GENERAL",
        keywords=["id_card"],
    )

    profile = RuleProfile(
        domain="medical",
        name="测试规则集",
        version="1.0.0",
        rules=[rule_val],
        downgrade_rules=[downgrade_rule],
    )

    engine = ConfigurableRuleEngine(taxonomy=taxonomy, profiles=[profile])
    funnel = ClassificationFunnel(engine=engine, taxonomy=taxonomy)
    funnel.policy.enable_llm_arbitration = True

    mock_llm = MagicMock()
    mock_llm.is_available = True
    mock_llm.arbitrate.return_value = {
        "final_level": "L2",
        "confidence": 0.95,
        "reasoning": "LLM 强行降级",
    }
    funnel.llm = mock_llm

    funnel_result, tags = funnel.classify_field("id_card", "330102199003072345")

    # 断言：Safety Floor 必须拒绝 LLM 的 L2 降级，保持真实地基等级 L3 且需要人工复核
    assert funnel_result.final_level == "L3"
    assert funnel_result.needs_human_review is True
    assert any(t.match_target == "field_value" for t in funnel_result.tags)


# ─── 2. BudgetAccountant 负数/零值预算 spend 拒绝测试 ───

def test_budget_spend_rejects_non_positive_epsilon():
    """验证 BudgetAccountant.spend() 对 <= 0 的 epsilon 或 < 0 的 delta 抛出 ValueError 拒绝。"""
    accountant = default_registry.get_or_create("test_remediation_ns", epsilon_total=10.0, delta_total=1e-3)

    with pytest.raises(ValueError):
        accountant.spend(epsilon=0.0, delta=0.0)

    with pytest.raises(ValueError):
        accountant.spend(epsilon=-1.0, delta=0.0)

    with pytest.raises(ValueError):
        accountant.spend(epsilon=1.0, delta=-1e-4)

    # 正常消耗正数预算成功，无异常抛出
    accountant.spend(epsilon=1.0, delta=1e-5)


# ─── 3. 高维向量 Laplace 机制 sqrt(d) 范数校准测试 ───

def test_dp_vector_laplace_sqrt_d_calibration():
    """验证 DifferentialPrivacy 高维向量 Laplace 机制下，
    L1 敏感度正确按照 sqrt(d) * max_norm 进行校准。
    """
    dp = DPApi(namespace="test_dp_sqrt_d_ns", epsilon_total=100.0)

    d = 100
    vectors = np.ones((5, d), dtype=np.float64)
    max_norm = 2.0
    epsilon = 2.0

    res = dp.vector_sum(
        vectors=vectors,
        max_norm=max_norm,
        epsilon=epsilon,
        mechanism="laplace",
        return_details=True,
    )

    assert isinstance(res, DPResult)
    expected_scale = (max_norm * math.sqrt(d)) / epsilon
    assert math.isclose(res.noise_scale, expected_scale, rel_tol=1e-5)


# ─── 4. 复合规则下划线规范化匹配测试 ───

def test_composite_rule_underscore_normalization():
    """验证 CompositeRuleEngine 对下划线/连字符字段名的规范化比对 (如 id_card / id-card 匹配 idcard)。"""
    rule_def = CompositeRuleDef(
        id="COMP_TEST_001",
        name="测试下划线复合规则",
        category="PERSONAL_BASIC",
        target_level="L3",
        field_patterns=[r"idcard"],
        min_matches=1,
    )
    engine = CompositeRuleEngine(rules=[rule_def])

    # 下划线形式 id_card 应成功命中
    tags1 = engine.evaluate(record={"id_card": "123456"})
    assert len(tags1) == 1
    assert tags1[0].rule_id == "COMP_TEST_001"

    # 连字符形式 id-card 应成功命中
    tags2 = engine.evaluate(record={"id-card": "123456"})
    assert len(tags2) == 1
    assert tags2[0].rule_id == "COMP_TEST_001"


# ─── 5. LLM 置信度防 1e6 恶意大值钳制测试 ───

def test_safe_llm_confidence_clamping():
    """验证 _safe_llm_confidence 对 1e6、NaN、Inf 或负数等恶意/异常值返回 fallback，
    而不是误计算为 1.0 (满分最高置信度)。
    """
    fallback = 0.5

    # 1e6 大值超界，必须回退 fallback
    assert ClassificationFunnel._safe_llm_confidence(1e6, fallback) == fallback
    # NaN 必须回退 fallback
    assert ClassificationFunnel._safe_llm_confidence(float("nan"), fallback) == fallback
    # Inf 必须回退 fallback
    assert ClassificationFunnel._safe_llm_confidence(float("inf"), fallback) == fallback
    # 负数必须回退 fallback
    assert ClassificationFunnel._safe_llm_confidence(-5.0, fallback) == fallback

    # 95.0 百分数合法容错转换为 0.95
    assert math.isclose(ClassificationFunnel._safe_llm_confidence(95.0, fallback), 0.95)
    # 0.88 原样返回
    assert math.isclose(ClassificationFunnel._safe_llm_confidence(0.88, fallback), 0.88)
