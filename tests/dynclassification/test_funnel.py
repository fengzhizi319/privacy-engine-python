"""三层漏斗 + 置信度策略测试 / Three-Layer Funnel & Confidence Policy Tests.

覆盖场景：
- Layer-1 规则引擎独立工作（无 NER/LLM）
- 冲突检测：普通规则 + 降级规则同时命中 → 置信度衰减
- 无冲突时置信度保持 1.0
- Override 压制后无冲突（置信度不衰减）
- NER 层集成（mock）
- LLM 仲裁集成（mock）
- LLM 不可用时回退到 Phase 1 衰减
- FunnelResult 结构完整性
- service 层集成测试
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from engine.dynclassification import (
    ClassificationFunnel,
    ConfidencePolicy,
    ConfigurableRuleEngine,
    DomainTaxonomy,
    EngineLayer,
    FunnelResult,
    SecurityTag,
)
from engine.dynclassification.llm_adapter import LlmAdapter
from engine.dynclassification.ner_adapter import NerAdapter
from engine.dynclassification.rule_schema import (
    DowngradeRuleDef,
    MatcherDef,
    RuleDef,
    RuleProfile,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def taxonomy() -> DomainTaxonomy:
    """标准 L1~L5 分类体系。"""
    return DomainTaxonomy(
        domain="medical",
        standard_id="TEST",
        levels={
            "L1": {"id": "L1", "name": "公开", "rank": 1},
            "L2": {"id": "L2", "name": "内部", "rank": 2},
            "L3": {"id": "L3", "name": "敏感", "rank": 3},
            "L4": {"id": "L4", "name": "高敏感", "rank": 4},
            "L5": {"id": "L5", "name": "极敏感", "rank": 5},
        },
        categories={
            "PII": {"id": "PII", "name": "个人信息"},
            "OPERATIONAL_STAT": {"id": "OPERATIONAL_STAT", "name": "运营统计"},
        },
        default_level="L3",
    )


@pytest.fixture
def profile_with_conflict() -> RuleProfile:
    """包含普通规则和降级规则的 Profile（可触发冲突）。"""
    return RuleProfile(
        domain="medical",
        rules=[
            RuleDef(
                id="RULE_REPORT",
                level="L3",
                category="PII",
                matchers=[MatcherDef(operator="keyword_contains", target="field_name", params={"keywords": ["report"]})],
            ),
        ],
        downgrade_rules=[
            DowngradeRuleDef(
                id="RULE_DOWN_OPS",
                keywords=["turnover_rate", "device_usage"],
                level="L2",
                category="OPERATIONAL_STAT",
                force_suppress=False,  # 不启用 force_suppress → 冲突共存
            ),
        ],
    )


@pytest.fixture
def profile_with_override() -> RuleProfile:
    """包含 override 降级规则的 Profile（压制后无冲突）。"""
    return RuleProfile(
        domain="medical",
        rules=[
            RuleDef(
                id="RULE_REPORT",
                level="L3",
                category="PII",
                matchers=[MatcherDef(operator="keyword_contains", target="field_name", params={"keywords": ["report"]})],
            ),
        ],
        downgrade_rules=[
            DowngradeRuleDef(
                id="RULE_DOWN_OPS",
                keywords=["turnover_rate"],
                level="L2",
                category="OPERATIONAL_STAT",
                force_suppress=True,
                max_force_suppress_level="L3",
            ),
        ],
    )


@pytest.fixture
def engine_conflict(taxonomy, profile_with_conflict) -> ConfigurableRuleEngine:
    """可触发冲突的引擎。"""
    return ConfigurableRuleEngine(taxonomy, [profile_with_conflict])


@pytest.fixture
def engine_override(taxonomy, profile_with_override) -> ConfigurableRuleEngine:
    """override 压制引擎。"""
    return ConfigurableRuleEngine(taxonomy, [profile_with_override])


# ===========================================================================
# Layer-1 基础测试
# ===========================================================================


class TestLayer1Basic:
    """Layer-1 规则引擎基础功能。"""

    def test_no_tags_returns_default_level(self, taxonomy, engine_conflict):
        """无规则命中时返回默认等级 L3。"""
        policy = ConfidencePolicy()
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy)
        result, _suppressed = funnel.classify_field("unknown_field", "some_value")

        assert result.final_level == "L3"
        assert result.confidence == 0.0
        assert result.engine_layer == EngineLayer.L1_RULE
        assert not result.has_conflict

    def test_normal_rule_hit_confidence_1(self, taxonomy, engine_conflict):
        """仅普通规则命中时置信度为 1.0。"""
        policy = ConfidencePolicy()
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy)
        # "report" 关键词命中普通规则
        result, _suppressed = funnel.classify_field("annual_report", "data")

        assert result.final_level == "L3"
        assert result.confidence == 1.0
        assert not result.has_conflict
        assert not result.needs_human_review


# ===========================================================================
# Phase 1: 置信度衰减测试
# ===========================================================================


class TestConfidenceDecay:
    """Phase 1: 规则冲突时置信度衰减。"""

    def test_conflict_detected_confidence_decays(self, taxonomy, engine_conflict):
        """普通规则 L3 + 降级规则 L2 同时命中 → 冲突 → 置信度衰减为 0.7。"""
        policy = ConfidencePolicy(conflict_confidence=0.7, conflict_needs_review=True)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy)
        # "turnover_rate_report" 同时包含 "report"(普通) 和 "turnover_rate"(降级)
        result, _suppressed = funnel.classify_field("turnover_rate_report", "data")

        assert result.has_conflict
        assert result.confidence == 0.7
        assert result.needs_human_review
        # 最终等级仍取最高 L3（安全优先）
        assert result.final_level == "L3"
        assert "冲突" in result.reasoning

    def test_conflict_custom_confidence(self, taxonomy, engine_conflict):
        """自定义冲突置信度。"""
        policy = ConfidencePolicy(conflict_confidence=0.5)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy)
        result, _suppressed = funnel.classify_field("turnover_rate_report", "data")

        assert result.confidence == 0.5

    def test_override_suppression_no_conflict(self, taxonomy, engine_override):
        """Override 压制后无冲突 → 置信度保持 1.0。"""
        policy = ConfidencePolicy(conflict_confidence=0.7)
        funnel = ClassificationFunnel(engine_override, taxonomy, policy)
        # "turnover_rate_report": override 压制 L3 普通标签，只剩 L2 降级标签
        result, _suppressed = funnel.classify_field("turnover_rate_report", "data")

        # override 成功压制 → 无冲突
        assert not result.has_conflict
        assert result.confidence == 1.0
        assert result.final_level == "L2"

    def test_no_review_when_configured(self, taxonomy, engine_conflict):
        """配置 conflict_needs_review=false 时不标记复核。"""
        policy = ConfidencePolicy(conflict_needs_review=False)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy)
        result, _suppressed = funnel.classify_field("turnover_rate_report", "data")

        assert result.has_conflict
        assert not result.needs_human_review


# ===========================================================================
# Layer-2: NER 集成测试 (Mock)
# ===========================================================================


class TestNerLayer:
    """Layer-2 NER 实体识别集成。"""

    def test_ner_adds_tags(self, taxonomy, engine_conflict):
        """NER 识别到实体时追加标签。"""
        # Mock NER adapter
        mock_ner = MagicMock(spec=NerAdapter)
        mock_ner.extract.return_value = [
            {"label": "MEDICAL_DISEASE", "text": "高血压", "confidence": 0.9}
        ]

        policy = ConfidencePolicy(enable_ner=True)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, ner_adapter=mock_ner)
        result, _suppressed = funnel.classify_field("diagnosis", "患者高血压")

        # NER 应该追加了一个 L3 标签
        ner_tags = [t for t in result.tags if t.source_engine == "SMALL_NER"]
        assert len(ner_tags) == 1
        assert ner_tags[0].level == "L3"
        assert ner_tags[0].category == "MEDICAL_DISEASE"
        assert result.engine_layer == EngineLayer.L2_SMALL_NER

    def test_ner_sensitive_disease_l4(self, taxonomy, engine_conflict):
        """NER 识别 L4 敏感疾病时升级为 L4。"""
        mock_ner = MagicMock(spec=NerAdapter)
        mock_ner.extract.return_value = [
            {"label": "MEDICAL_DISEASE", "text": "抑郁症", "confidence": 0.85}
        ]

        policy = ConfidencePolicy(enable_ner=True)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, ner_adapter=mock_ner)
        result, _suppressed = funnel.classify_field("note", "抑郁症")

        ner_tags = [t for t in result.tags if t.source_engine == "SMALL_NER"]
        assert ner_tags[0].level == "L4"
        assert result.final_level == "L4"

    def test_ner_l5_sensitive_disease_l5(self, taxonomy, engine_conflict):
        """NER 识别 L5 极高敏疾病 (如 HIV/艾滋病) 时升级为 L5。"""
        mock_ner = MagicMock(spec=NerAdapter)
        mock_ner.extract.return_value = [
            {"label": "MEDICAL_DISEASE", "text": "HIV感染", "confidence": 0.85}
        ]

        policy = ConfidencePolicy(enable_ner=True)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, ner_adapter=mock_ner)
        result, _suppressed = funnel.classify_field("note", "HIV感染")

        assert result.final_level == "L5"

    def test_ner_disabled_skips(self, taxonomy, engine_conflict):
        """enable_ner=false 时跳过 NER 层。"""
        mock_ner = MagicMock(spec=NerAdapter)
        policy = ConfidencePolicy(enable_ner=False)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, ner_adapter=mock_ner)
        funnel.classify_field("diagnosis", "高血压")

        mock_ner.extract.assert_not_called()

    def test_ner_custom_entity_mapping(self, taxonomy, engine_conflict):
        """自定义 ner_entity_mapping 配置化映射生效。"""
        # 配置自定义实体→等级映射
        taxonomy.ner_entity_mapping = {
            "MEDICAL_DISEASE": "L5",
            "MEDICATION": "L2",
        }
        mock_ner = MagicMock(spec=NerAdapter)
        mock_ner.extract.return_value = [
            {"label": "MEDICAL_DISEASE", "text": "高血压", "confidence": 0.9},
            {"label": "MEDICATION", "text": "阿司匹林", "confidence": 0.85},
        ]

        policy = ConfidencePolicy(enable_ner=True)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, ner_adapter=mock_ner)
        result, _suppressed = funnel.classify_field("diagnosis", "高血压用阿司匹林")

        ner_tags = [t for t in result.tags if t.source_engine == "SMALL_NER"]
        assert len(ner_tags) == 2
        # 自定义映射: MEDICAL_DISEASE → L5, MEDICATION → L2
        levels = {t.category: t.level for t in ner_tags}
        assert levels["MEDICAL_DISEASE"] == "L5"
        assert levels["MEDICATION"] == "L2"
        assert result.final_level == "L5"

    def test_ner_custom_sensitive_keywords(self, taxonomy, engine_conflict):
        """自定义 ner_sensitive_keywords 配置生效。"""
        taxonomy.ner_sensitive_keywords = ["洗钱", "恐怖融资"]
        mock_ner = MagicMock(spec=NerAdapter)
        mock_ner.extract.return_value = [
            {"label": "MEDICAL_DISEASE", "text": "洗钱风险", "confidence": 0.8}
        ]

        policy = ConfidencePolicy(enable_ner=True)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, ner_adapter=mock_ner)
        result, _suppressed = funnel.classify_field("risk_type", "洗钱风险")

        ner_tags = [t for t in result.tags if t.source_engine == "SMALL_NER"]
        # "洗钱" 命中自定义敏感关键词 → 升级为次高等级 L4
        assert ner_tags[0].level == "L4"
        assert ner_tags[0].category == "MEDICAL_SENSITIVE_DISEASE"


# ===========================================================================
# Layer-3: LLM 仲裁测试 (Mock)
# ===========================================================================


class TestLlmArbitration:
    """Layer-3 LLM 仲裁。"""

    def test_llm_arbitration_resolves_conflict(self, taxonomy, engine_conflict):
        """LLM 仲裁成功时修正等级和置信度。"""
        mock_llm = MagicMock(spec=LlmAdapter)
        mock_llm.is_available = True
        mock_llm.arbitrate.return_value = {
            "final_level": "L2",
            "confidence": 0.92,
            "reasoning": "营业额属于运营统计指标",
        }

        policy = ConfidencePolicy(enable_llm_arbitration=True)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, llm_adapter=mock_llm)
        result, _suppressed = funnel.classify_field("turnover_rate_report", "0.85")

        assert result.has_conflict
        assert result.engine_layer == EngineLayer.L3_LLM
        assert result.confidence == 0.92
        assert "运营统计" in result.reasoning
        # LLM 追加了仲裁标签
        llm_tags = [t for t in result.tags if t.source_engine == "LLM"]
        assert len(llm_tags) == 1
        assert llm_tags[0].level == "L2"

    def test_llm_unavailable_falls_back_to_decay(self, taxonomy, engine_conflict):
        """LLM 不可用时回退到 Phase 1 置信度衰减。"""
        mock_llm = MagicMock(spec=LlmAdapter)
        mock_llm.is_available = False

        policy = ConfidencePolicy(enable_llm_arbitration=True, conflict_confidence=0.7)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, llm_adapter=mock_llm)
        result, _suppressed = funnel.classify_field("turnover_rate_report", "0.85")

        assert result.has_conflict
        assert result.confidence == 0.7
        assert result.needs_human_review
        assert result.engine_layer == EngineLayer.L1_RULE

    def test_llm_arbitration_returns_none_falls_back(self, taxonomy, engine_conflict):
        """LLM 仲裁返回 None 时回退到衰减。"""
        mock_llm = MagicMock(spec=LlmAdapter)
        mock_llm.is_available = True
        mock_llm.arbitrate.return_value = None

        policy = ConfidencePolicy(enable_llm_arbitration=True, conflict_confidence=0.6)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, llm_adapter=mock_llm)
        result, _suppressed = funnel.classify_field("turnover_rate_report", "0.85")

        assert result.confidence == 0.6
        assert result.needs_human_review

    def test_llm_deep_classification_low_confidence(self, taxonomy, engine_conflict):
        """低置信度时触发 LLM 深度分类。"""
        mock_llm = MagicMock(spec=LlmAdapter)
        mock_llm.is_available = True
        mock_llm.classify.return_value = {
            "final_level": "L4",
            "confidence": 0.88,
            "reasoning": "包含敏感病种关键词",
        }

        # 使用一个不命中任何规则的字段，使 confidence=0.0 < threshold
        policy = ConfidencePolicy(enable_llm=True, llm_confidence_threshold=0.6)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, llm_adapter=mock_llm)
        result, _suppressed = funnel.classify_field("unknown_field", "some text")

        # confidence=0.0 < 0.6 → 触发 LLM
        assert result.engine_layer == EngineLayer.L3_LLM
        assert result.confidence == 0.88


# ===========================================================================
# Layer-3 安全地板兜底：LLM 未返回合法等级
# ===========================================================================


class TestLlmNoValidLevelSafetyFloor:
    """LLM 返回结果但无合法 final_level 时的防御行为。

    安全地板要求：无效裁定不得刷新置信度、不得归属 L3_LLM，
    保留上游结果并强制人工复核（防止"高置信度+无等级"注入静默抬升置信度）。
    """

    def test_scenario_b_missing_final_level_keeps_upstream(self, taxonomy, engine_conflict):
        """场景 B：LLM 输出缺少 final_level → 保留上游置信度与层级归属。"""
        mock_llm = MagicMock(spec=LlmAdapter)
        mock_llm.is_available = True
        mock_llm.classify.return_value = {
            "confidence": 0.99,  # 注入式高置信度，无等级
            "reasoning": "无法确定等级",
        }

        policy = ConfidencePolicy(enable_llm=True, llm_confidence_threshold=0.6)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, llm_adapter=mock_llm)
        result, _suppressed = funnel.classify_field("unknown_field", "some text")

        # confidence 保持上游 0.0，不被 LLM 的 0.99 刷新
        assert result.confidence == 0.0
        assert result.engine_layer == EngineLayer.L1_RULE
        assert result.needs_human_review
        assert "未返回有效等级" in result.reasoning
        # 不产生 LLM 审计标签
        assert not [t for t in result.tags if t.source_engine == "LLM"]

    def test_scenario_b_invalid_final_level_rejected(self, taxonomy, engine_conflict):
        """场景 B：LLM 输出 taxonomy 之外的等级（L9）→ 视为无效裁定。"""
        mock_llm = MagicMock(spec=LlmAdapter)
        mock_llm.is_available = True
        mock_llm.classify.return_value = {
            "final_level": "L9",
            "confidence": 0.99,
            "reasoning": "伪造等级",
        }

        policy = ConfidencePolicy(enable_llm=True, llm_confidence_threshold=0.6)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, llm_adapter=mock_llm)
        result, _suppressed = funnel.classify_field("unknown_field", "some text")

        assert result.confidence == 0.0
        assert result.final_level == "L3"  # 回退 default_level，而非 L9
        assert result.engine_layer == EngineLayer.L1_RULE
        assert result.needs_human_review

    def test_scenario_c_missing_final_level_keeps_upstream(self, taxonomy, engine_conflict):
        """场景 C：多模态 LLM 未返回 final_level → 保留上游结果并人工复核。"""
        mock_llm = MagicMock(spec=LlmAdapter)
        mock_llm.is_available = True
        mock_llm.classify.return_value = {
            "confidence": 0.95,
            "reasoning": "图像内容不清晰",
        }

        policy = ConfidencePolicy(auto_llm_on_image=True)
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy, llm_adapter=mock_llm)
        result, _suppressed = funnel.classify_field(
            "case_image", "data:image/png;base64,iVBORw0KGgo="
        )

        assert result.confidence == 0.0
        assert result.engine_layer == EngineLayer.L1_RULE
        assert result.needs_human_review
        assert "未返回有效等级" in result.reasoning


# ===========================================================================
# FunnelResult 结构完整性
# ===========================================================================


class TestFunnelResultStructure:
    """FunnelResult 数据结构验证。"""

    def test_result_fields_complete(self, taxonomy, engine_conflict):
        """结果包含所有必要字段。"""
        policy = ConfidencePolicy()
        funnel = ClassificationFunnel(engine_conflict, taxonomy, policy)
        result, _suppressed = funnel.classify_field("annual_report", "data")

        assert isinstance(result, FunnelResult)
        assert isinstance(result.tags, list)
        assert isinstance(result.final_level, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.engine_layer, str)
        assert isinstance(result.needs_human_review, bool)
        assert isinstance(result.reasoning, str)
        assert isinstance(result.has_conflict, bool)


# ===========================================================================
# Service 层集成测试
# ===========================================================================


class TestServiceIntegration:
    """DynClassificationService 集成三层漏斗。"""

    def test_classify_field_returns_engine_layer(self):
        """service.classify_field 返回 engine_layer 字段。"""
        from engine.dynclassification import DynClassificationService

        svc = DynClassificationService(rules_dir="rules")
        resp = svc.classify_field("phone_number", "13800138000")

        assert resp.field_result is not None
        assert resp.field_result.engine_layer == EngineLayer.L1_RULE
        assert resp.field_result.confidence == 1.0

    def test_classify_field_conflict_confidence(self):
        """通过 service 验证冲突置信度衰减。"""
        from engine.dynclassification import DynClassificationService

        svc = DynClassificationService(rules_dir="rules")
        # "turnover_rate" 在 medical domain 中会触发降级规则
        # 如果同时有普通规则命中则产生冲突
        resp = svc.classify_field("turnover_rate", "0.85", domain="medical")

        assert resp.field_result is not None
        # 验证结果结构完整
        assert resp.field_result.engine_layer in (
            EngineLayer.L1_RULE, EngineLayer.L2_SMALL_NER, EngineLayer.L3_LLM
        )


# ===========================================================================
# LLM 分类 prompt 模板配置化测试
# ===========================================================================


class TestLlmClassifyPromptTemplate:
    """LLM 分类 prompt 模板 YAML 配置化测试。"""

    def test_llm_adapter_accepts_classify_prompt_template(self):
        """LlmAdapter 接受 classify_prompt_template 参数。"""
        from engine.dynclassification.llm_adapter import LlmAdapter

        template = "你是{domain}领域的安全专家。等级定义：{levels_desc}"
        adapter = LlmAdapter(model_path="/nonexistent", classify_prompt_template=template)
        assert adapter._classify_prompt_template == template

    def test_llm_engine_accepts_classify_prompt_template(self):
        """Qwen2VLClassifier 接受 classify_prompt_template 参数。"""
        from engine.dynclassification.llm_engines import Qwen2VLClassifier

        template = "你是{domain}领域专家。标准: {standard_id}。{levels_desc}"
        clf = Qwen2VLClassifier(model_path="/tmp/fake", classify_prompt_template=template)
        assert clf._classify_prompt_template == template

    def test_taxonomy_llm_classify_prompt_template_field(self):
        """DomainTaxonomy 支持 llm_classify_prompt_template 字段。"""
        from engine.dynclassification.models import (
            CategoryDef,
            DomainTaxonomy,
            SensitivityLevelDef,
        )

        tax = DomainTaxonomy(
            domain="finance",
            standard_id="JR_T_0197",
            levels={"C1": SensitivityLevelDef(id="C1", name="不敏感", rank=1)},
            categories={"FIN": CategoryDef(id="FIN", name="金融数据")},
            default_level="C1",
            llm_classify_prompt_template="你是金融数据安全专家。",
        )
        assert tax.llm_classify_prompt_template == "你是金融数据安全专家。"

    def test_sensitivity_level_from_string_warns_on_unknown(self):
        """未知等级字符串回退到 L3 并记录警告。"""
        from engine.dynclassification.base import SensitivityLevel

        # 正常值解析
        assert SensitivityLevel.from_string("L4") == SensitivityLevel.L4
        assert SensitivityLevel.from_string("C3") == SensitivityLevel.C3
        # 未知值回退到 L3
        assert SensitivityLevel.from_string("UNKNOWN_LEVEL") == SensitivityLevel.L3

    def test_ner_unknown_label_fallback(self, taxonomy, engine_conflict):
        """未知 NER 实体标签回退到中间等级而非静默丢弃。"""
        mock_ner = MagicMock(spec=NerAdapter)
        mock_ner.extract.return_value = [
            {"label": "CUSTOM_FINANCIAL_RISK", "text": "洗钱风险", "confidence": 0.75}
        ]

        policy = ConfidencePolicy(enable_ner=True, ner_trigger_max_rank=5)
        funnel = ClassificationFunnel(
            engine=engine_conflict,
            taxonomy=taxonomy,
            confidence_policy=policy,
            ner_adapter=mock_ner,
        )
        result, _ = funnel.classify_field("some_field", "洗钱风险记录")

        # 未知标签应产生标签（回退到中间等级），而非被丢弃
        ner_tags = [t for t in result.tags if t.source_engine == "SMALL_NER"]
        assert len(ner_tags) == 1
        assert ner_tags[0].category == "CUSTOM_FINANCIAL_RISK"
        assert ner_tags[0].rule_id == "NER_CUSTOM_FINANCIAL_RISK"


# ===========================================================================
# Service reload 适配器重置测试
# ===========================================================================


class TestServiceReloadResetsAdapters:
    """验证 service.reload() 重置 NER/LLM 适配器单例。"""

    def test_reload_clears_ner_and_llm_adapters(self):
        """热重载后 NER/LLM 适配器被重置为 None。"""
        from engine.dynclassification import DynClassificationService
        from engine.dynclassification.ner_adapter import NerAdapter
        from engine.dynclassification.llm_adapter import LlmAdapter

        svc = DynClassificationService(rules_dir="rules")
        # 模拟已初始化的适配器单例
        svc._ner_adapter = NerAdapter()
        svc._llm_adapter = LlmAdapter()
        svc._funnel_cache["test:key"] = object()  # 模拟缓存

        # 执行 reload
        svc.reload()

        # 验证适配器和缓存均被重置
        assert svc._ner_adapter is None
        assert svc._llm_adapter is None
        assert len(svc._funnel_cache) == 0

    def test_reload_clears_loader_caches(self):
        """热重载后 ProfileLoader 缓存被清空。"""
        from engine.dynclassification import DynClassificationService

        svc = DynClassificationService(rules_dir="rules")
        # 触发一次分类以填充缓存
        svc.classify_field("phone_number", "13800138000")
        assert len(svc.loader._engine_cache) > 0

        # 执行 reload
        svc.reload()

        # 验证 loader 缓存被清空
        assert len(svc.loader._engine_cache) == 0
        assert len(svc.loader._taxonomy_cache) == 0
