"""真实 NER / LLM 模型冒烟测试 / Real Model Smoke Tests.

这些测试要求 .models/ 目录下已下载 NER 与 Qwen3.5-0.8B 模型，并且当前环境安装了
[ml] 可选依赖（torch/transformers/onnxruntime/modelscope 等）。

由于加载本地大模型非常耗时，所有测试都标记为 slow + real_models，
常规 CI 运行时应使用 `-m "not real_models"` 跳过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.dynclassification.base import SensitivityLevel
from engine.dynclassification.llm_adapter import LlmAdapter
from engine.dynclassification.ner_adapter import NerAdapter

_MODELS_DIR = Path(__file__).resolve().parents[2] / ".models"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.real_models,
    pytest.mark.skipif(not _MODELS_DIR.is_dir(), reason=".models/ 未下载模型权重，跳过真实模型冒烟测试"),
]


class TestRealNerAdapter:
    """使用 .models/raner_cmeee 真实模型验证 NER 实体抽取。"""

    def test_real_ner_extracts_medical_entities(self):
        """NerAdapter 应能加载 ModelScope 后端并识别出医疗实体。"""
        adapter = NerAdapter()
        assert adapter.is_available, "NER 后端初始化失败，请检查模型文件与依赖"

        text = "患者诊断为糖尿病和高血压"
        entities = adapter.extract(text)
        print(f"Extracted entities: {entities}")
        # Extracted entities: [{'text': '糖尿病', '糖尿病label': 'MEDICAL_DISEASE', 'confidence': 1.0},{'text': '高血压', 'label': 'MEDICAL_DISEASE', 'confidence': 1.0}]

        labels = {ent["label"] for ent in entities}
        assert "MEDICAL_DISEASE" in labels
        texts = {ent["text"] for ent in entities}
        assert "糖尿病" in texts or "高血压" in texts


class TestRealLlmAdapter:
    """使用 .models/Qwen3.5-0.8B-Privacy-Classifier-Smoother 真实模型验证 LLM 分类。"""

    def test_real_llm_loads_and_returns_structured_result(self, monkeypatch):
        """LlmAdapter 应能加载本地 Qwen3.5-0.8B 并返回含 final_level 的结构化结果。"""
        # 强制使用本地 PyTorch 后端，避免 vllm.env 将 provider 设为 HTTP API
        # （本地无 vLLM 服务会导致请求失败返回 None）。
        # 清空 PRIVACY_ENV_PROFILE 阻止 load_env_file() 加载 config/env/vllm.env，
        # 重置 env loader 缓存，防止 load_env_file() 覆盖 monkeypatch 的值。
        import engine.env_loader as _env_mod
        monkeypatch.setenv("PRIVACY_ENV_PROFILE", "qwen3")
        monkeypatch.setenv("PRIVACY_LLM_PROVIDER", "qwen3")
        _env_mod._ENV_LOADED = False

        adapter = LlmAdapter(".models/Qwen3.5-0.8B-Privacy-Classifier-Smoother")
        assert adapter.is_available, "LLM 初始化失败，请检查模型文件与依赖"

        result = adapter.classify(
            text="身份证号：510101199001011234",
            upstream_level="L3",
            upstream_confidence=0.6,
        )
        print(f"LLM result: {result}")
        assert result is not None
        assert "final_level" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_real_llm_arbitrate_returns_structured_result(self, monkeypatch):
        """LlmAdapter.arbitrate 应能组装冲突上下文并返回裁定结果。"""
        import engine.env_loader as _env_mod
        monkeypatch.setenv("PRIVACY_ENV_PROFILE", "qwen3")
        monkeypatch.setenv("PRIVACY_LLM_PROVIDER", "qwen3")
        _env_mod._ENV_LOADED = False

        from engine.dynclassification.models import (
            CategoryDef,
            DomainTaxonomy,
            SecurityTag,
            SensitivityLevelDef,
        )

        adapter = LlmAdapter()
        assert adapter.is_available, "LLM 初始化失败，请检查模型文件与依赖"

        tag1 = SecurityTag(rule_id="RULE_IDCARD", level="L3", category="PERSONAL_BASIC")
        tag2 = SecurityTag(rule_id="DOWN_OPS", level="L2", category="MANAGEMENT")
        taxonomy = DomainTaxonomy(
            domain="test",
            standard_id="test",
            levels={
                "L2": SensitivityLevelDef(id="L2", name="内部", rank=2),
                "L3": SensitivityLevelDef(id="L3", name="敏感", rank=3),
            },
            categories={
                "PERSONAL_BASIC": CategoryDef(id="PERSONAL_BASIC", name="个人基本信息"),
                "MANAGEMENT": CategoryDef(id="MANAGEMENT", name="管理信息"),
            },
            default_level="L1",
        )

        result = adapter.arbitrate("user_id", "510101199001011234", [tag1, tag2], taxonomy)
        assert result is not None
        assert "final_level" in result
        assert "confidence" in result
