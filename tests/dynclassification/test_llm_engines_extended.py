"""LLM 引擎深度补充测试 / LLM Engine Deep Supplement Tests.

补充 test_llm_adapter.py 未覆盖的场景：
- Qwen2VLClassifier._lazy_init(): 模型目录不存在、init_error 缓存机制
- Qwen2VLClassifier.is_ready: 各状态下的就绪判断
- Qwen2VLClassifier.warmup(): 成功与失败场景
- Qwen2VLClassifier._select_device(): custom_device 参数、环境变量覆盖、CUDA 不兼容回退
- Qwen2VLClassifier._parse_json_result(): 缺失 final_level、空 JSON、嵌套 JSON
- LlmAdapter.arbitrate(): 自定义 prompt 模板、异常处理
- LlmAdapter.classify(): 多种等级字符串转换（C1-C4）
- Qwen2VLClassifier._classify_inner(): 自定义 prompt 模板格式化

运行方式：
    PYTHONPATH=. pytest tests/dynclassification/test_llm_engines_extended.py -v
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from engine.dynclassification.llm_adapter import LlmAdapter
from engine.dynclassification.llm_engines import Qwen2VLClassifier
from engine.dynclassification.models import (
    CategoryDef,
    DomainTaxonomy,
    SecurityTag,
    SensitivityLevelDef,
)


# =========================================================================== #
# Qwen2VLClassifier 初始化与生命周期测试
# =========================================================================== #


class TestQwen2VLInitLifecycle:
    """测试 Qwen2VLClassifier 的初始化、错误缓存和生命周期管理。"""

    def test_lazy_init_model_dir_not_found(self):
        """模型目录不存在时应抛出异常（FileNotFoundError 或 ImportError）。"""
        classifier = Qwen2VLClassifier(model_path="/nonexistent/qwen_model")
        with pytest.raises(Exception):
            classifier._lazy_init()

    def test_lazy_init_error_caching(self):
        """初始化失败后再次调用应直接抛出缓存的错误（不重试加载）。"""
        classifier = Qwen2VLClassifier(model_path="/nonexistent/qwen_model")
        with pytest.raises(Exception):
            classifier._lazy_init()
        # 第二次调用应抛出相同错误
        with pytest.raises(Exception):
            classifier._lazy_init()
        # 验证 _init_error 被缓存
        assert classifier._init_error is not None

    def test_is_ready_false_before_init(self):
        """未初始化时 is_ready 应为 False。"""
        classifier = Qwen2VLClassifier(model_path="/tmp/fake")
        assert classifier.is_ready is False

    def test_is_ready_true_after_successful_init(self):
        """成功初始化后 is_ready 应为 True。"""
        classifier = Qwen2VLClassifier(model_path="/tmp/fake")
        classifier._initialized = True
        classifier._init_error = None
        assert classifier.is_ready is True

    def test_is_ready_false_after_init_error(self):
        """初始化出错后 is_ready 应为 False。"""
        classifier = Qwen2VLClassifier(model_path="/tmp/fake")
        classifier._initialized = False
        classifier._init_error = FileNotFoundError("model not found")
        assert classifier.is_ready is False

    def test_warmup_success(self):
        """warmup 成功时应返回 True。"""
        classifier = Qwen2VLClassifier(model_path="/tmp/fake")
        with patch.object(classifier, "_lazy_init"):
            result = classifier.warmup()
        assert result is True

    def test_warmup_failure(self):
        """warmup 失败时应返回 False（不抛异常）。"""
        classifier = Qwen2VLClassifier(model_path="/nonexistent/model")
        result = classifier.warmup()
        assert result is False

    def test_classify_init_failure_returns_none(self):
        """classify 时初始化失败应返回 None（优雅降级）。"""
        classifier = Qwen2VLClassifier(model_path="/nonexistent/model")
        from engine.dynclassification.base import SensitivityLevel

        result = classifier.classify("测试文本", SensitivityLevel.L3, 0.5)
        assert result is None


# =========================================================================== #
# Qwen2VLClassifier._select_device 补充测试
# =========================================================================== #


class TestSelectDeviceExtended:
    """补充 _select_device 的参数优先级和环境变量测试。"""

    def test_custom_device_cpu(self):
        """custom_device='cpu' 应直接返回 cpu。"""
        import torch

        classifier = Qwen2VLClassifier()
        result = classifier._select_device(torch, custom_device="cpu")
        assert result == "cpu"

    def test_custom_device_cuda_incompatible_fallback(self):
        """custom_device='cuda' 但 CUDA 不兼容时应继续级联。"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.mem_get_info.return_value = (10 * 1024**3, 12 * 1024**3)
        # MPS 不可用
        mock_torch.backends.mps.is_available.return_value = False

        classifier = Qwen2VLClassifier()
        # Mock _is_cuda_compatible 返回 False（算力不兼容）
        with patch.object(Qwen2VLClassifier, "_is_cuda_compatible", return_value=False):
            result = classifier._select_device(mock_torch, custom_device="cuda")
        # 应回退到 cpu（因为 CUDA 不兼容且无 MPS）
        assert result == "cpu"

    def test_env_var_device_override(self, monkeypatch):
        """环境变量 PRIVACY_LLM_DEVICE 应覆盖自动检测。"""
        import torch

        monkeypatch.setenv("PRIVACY_LLM_DEVICE", "cpu")
        classifier = Qwen2VLClassifier()
        result = classifier._select_device(torch)
        assert result == "cpu"

    def test_env_var_privacy_device_fallback(self, monkeypatch):
        """PRIVACY_DEVICE 环境变量作为次级覆盖。"""
        import torch

        monkeypatch.delenv("PRIVACY_LLM_DEVICE", raising=False)
        monkeypatch.setenv("PRIVACY_DEVICE", "cpu")
        classifier = Qwen2VLClassifier()
        result = classifier._select_device(torch)
        assert result == "cpu"

    def test_mps_selected_on_apple_silicon(self):
        """macOS MPS 可用时应选择 mps。"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        classifier = Qwen2VLClassifier()
        with patch.object(Qwen2VLClassifier, "_is_cuda_compatible", return_value=False):
            result = classifier._select_device(mock_torch)
        assert result == "mps"

    def test_vram_check_exception_still_uses_cuda(self):
        """显存检查异常但 CUDA 兼容时仍应使用 cuda。"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.mem_get_info.side_effect = RuntimeError("driver error")
        mock_torch.backends.mps.is_available.return_value = False

        classifier = Qwen2VLClassifier()
        with patch.object(Qwen2VLClassifier, "_is_cuda_compatible", return_value=True):
            result = classifier._select_device(mock_torch)
        assert result == "cuda"


# =========================================================================== #
# Qwen2VLClassifier._parse_json_result 补充测试
# =========================================================================== #


class TestParseJsonResultExtended:
    """补充 _parse_json_result 的边界情况测试。"""

    def setup_method(self):
        self.classifier = Qwen2VLClassifier(model_path="/tmp/fake")

    def test_missing_final_level_returns_none(self):
        """JSON 中缺少 final_level 字段应返回 None。"""
        from engine.dynclassification.base import SensitivityLevel

        text = '{"confidence": 0.9, "reasoning": "缺少等级字段"}'
        result = self.classifier._parse_json_result(text, SensitivityLevel.L3, 0.5)
        assert result is None

    def test_empty_json_object_returns_none(self):
        """空 JSON 对象 {} 应返回 None。"""
        from engine.dynclassification.base import SensitivityLevel

        result = self.classifier._parse_json_result("{}", SensitivityLevel.L3, 0.5)
        assert result is None

    def test_nested_json_with_final_level(self):
        """包含嵌套结构但有 final_level 的 JSON 应正确解析。"""
        from engine.dynclassification.base import SensitivityLevel

        text = '{"final_level": "L5", "confidence": 0.99, "reasoning": "基因数据", "extra": {"key": "val"}}'
        result = self.classifier._parse_json_result(text, SensitivityLevel.L4, 0.6)
        assert result is not None
        assert result["final_level"] == "L5"
        assert result["confidence"] == 0.99

    def test_json_with_surrounding_text(self):
        """JSON 前后有额外文字时应正确提取。"""
        from engine.dynclassification.base import SensitivityLevel

        text = '经过分析，结果如下：{"final_level": "L3", "confidence": 0.85, "reasoning": "PII"} 以上。'
        result = self.classifier._parse_json_result(text, SensitivityLevel.L3, 0.5)
        assert result is not None
        assert result["final_level"] == "L3"

    def test_empty_string_returns_none(self):
        """空字符串应返回 None。"""
        from engine.dynclassification.base import SensitivityLevel

        result = self.classifier._parse_json_result("", SensitivityLevel.L3, 0.5)
        assert result is None

    def test_json_with_needs_human_review_field(self):
        """包含 needs_human_review 字段的 JSON 应完整保留。"""
        from engine.dynclassification.base import SensitivityLevel

        text = '{"final_level": "L4", "confidence": 0.7, "reasoning": "不确定", "needs_human_review": true}'
        result = self.classifier._parse_json_result(text, SensitivityLevel.L3, 0.5)
        assert result is not None
        assert result["needs_human_review"] is True


# =========================================================================== #
# LlmAdapter 补充测试
# =========================================================================== #


class TestLlmAdapterExtended:
    """补充 LlmAdapter 的仲裁模板、等级转换和异常处理测试。"""

    def _make_taxonomy(self, **kwargs) -> DomainTaxonomy:
        """构建测试用分类体系。"""
        defaults = {
            "domain": "sc_health_db51",
            "standard_id": "DB51_T_2989",
            "levels": {
                "L1": SensitivityLevelDef(id="L1", name="公开", rank=1),
                "L2": SensitivityLevelDef(id="L2", name="内部", rank=2),
                "L3": SensitivityLevelDef(id="L3", name="敏感", rank=3),
                "L4": SensitivityLevelDef(id="L4", name="高敏感", rank=4),
            },
            "categories": {
                "PERSONAL": CategoryDef(id="PERSONAL", name="个人信息"),
                "MEDICAL": CategoryDef(id="MEDICAL", name="诊疗信息"),
            },
            "default_level": "L1",
        }
        defaults.update(kwargs)
        return DomainTaxonomy(**defaults)

    def test_arbitrate_custom_prompt_template(self):
        """taxonomy 中配置的自定义仲裁 prompt 模板应被使用。"""
        adapter = LlmAdapter()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {
            "final_level": "L2",
            "confidence": 0.88,
            "reasoning": "运营数据",
        }
        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        taxonomy = self._make_taxonomy(
            llm_arbitration_prompt_template=(
                "字段: {field_name}, 值: {value}, 领域: {domain}, "
                "标准: {standard_id}\n冲突:\n{conflict_desc}\n等级:\n{levels_desc}"
            )
        )
        tags = [
            SecurityTag(rule_id="R1", level="L3", category="PERSONAL"),
            SecurityTag(rule_id="R2", level="L2", category="MEDICAL"),
        ]

        result = adapter.arbitrate("test_field", "test_value", tags, taxonomy)
        assert result is not None
        assert result["final_level"] == "L2"

        # 验证 classify 被调用，且传入的文本包含自定义模板内容
        call_args = mock_classifier.classify.call_args
        arbitration_text = call_args[0][0]
        assert "test_field" in arbitration_text
        assert "test_value" in arbitration_text
        assert "sc_health_db51" in arbitration_text

    def test_arbitrate_exception_returns_none(self):
        """仲裁过程中异常应返回 None。"""
        adapter = LlmAdapter()
        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = RuntimeError("model crashed")
        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        taxonomy = self._make_taxonomy()
        tags = [SecurityTag(rule_id="R1", level="L3", category="PERSONAL")]

        result = adapter.arbitrate("field", "value", tags, taxonomy)
        assert result is None

    def test_arbitrate_unavailable_returns_none(self):
        """LLM 不可用时 arbitrate 应返回 None。"""
        adapter = LlmAdapter()
        adapter._initialized = True
        adapter._available = False

        taxonomy = self._make_taxonomy()
        tags = [SecurityTag(rule_id="R1", level="L3", category="PERSONAL")]

        result = adapter.arbitrate("field", "value", tags, taxonomy)
        assert result is None

    def test_classify_various_level_strings(self):
        """classify 应支持多种等级字符串（L1-L5, C1-C4）。"""
        adapter = LlmAdapter()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {
            "final_level": "L3",
            "confidence": 0.9,
            "reasoning": "test",
        }
        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        # 测试各种等级字符串
        for level_str in ["L1", "L2", "L3", "L4", "L5"]:
            result = adapter.classify("测试", level_str, 0.5)
            assert result is not None, f"等级 {level_str} 转换失败"

    def test_classify_unknown_level_string_fallback(self):
        """未知等级字符串应回退到默认值（不崩溃）。"""
        adapter = LlmAdapter()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {
            "final_level": "L3",
            "confidence": 0.9,
            "reasoning": "test",
        }
        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        # SensitivityLevel.from_string 对未知值回退到 L3
        result = adapter.classify("测试", "UNKNOWN_LEVEL", 0.5)
        assert result is not None

    def test_classify_unavailable_returns_none(self):
        """LLM 不可用时 classify 应返回 None。"""
        adapter = LlmAdapter()
        adapter._initialized = True
        adapter._available = False

        result = adapter.classify("测试", "L3", 0.5)
        assert result is None


# =========================================================================== #
# Qwen2VLClassifier._classify_inner 自定义 prompt 测试
# =========================================================================== #


class TestClassifyInnerPromptTemplate:
    """测试 _classify_inner 中自定义 prompt 模板的格式化。"""

    def _make_classifier_with_template(self, template: str | None):
        """构建带自定义模板的 mock 分类器。"""
        classifier = Qwen2VLClassifier(
            model_path="/tmp/fake",
            classify_prompt_template=template,
        )
        classifier._initialized = True

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "<prompt>"

        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.__len__ = lambda self: 1
        mock_tensor.__iter__ = lambda self: iter([[1, 2, 3]])
        mock_processor.return_value = {"input_ids": mock_tensor}

        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model.device = "cpu"
        mock_processor.batch_decode.return_value = [
            '{"final_level": "L3", "confidence": 0.85, "reasoning": "test"}'
        ]

        classifier._model = mock_model
        classifier._processor = mock_processor
        return classifier

    def test_custom_template_used_in_classify_inner(self):
        """自定义 classify_prompt_template 应在 _classify_inner 中被格式化使用。"""
        template = "你是{domain}领域专家。标准: {standard_id}。{levels_desc}"
        classifier = self._make_classifier_with_template(template)

        from engine.dynclassification.base import SensitivityLevel

        result = classifier._classify_inner("测试文本", SensitivityLevel.L3, 0.5)
        assert result is not None
        assert result["final_level"] == "L3"

        # 验证 apply_chat_template 被调用时 messages 中的 system content 包含模板内容
        call_args = classifier._processor.apply_chat_template.call_args
        messages = call_args[0][0]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "medical" in system_msg["content"]  # domain 默认值
        assert "DB51_T_2989" in system_msg["content"]  # standard_id 默认值

    def test_default_template_used_when_none(self):
        """未配置模板时应使用内置默认 prompt。"""
        classifier = self._make_classifier_with_template(None)

        from engine.dynclassification.base import SensitivityLevel

        result = classifier._classify_inner("身份证号123", SensitivityLevel.L3, 0.5)
        assert result is not None

        # 验证 system prompt 包含默认医疗分级描述
        call_args = classifier._processor.apply_chat_template.call_args
        messages = call_args[0][0]
        system_content = messages[0]["content"]
        assert "分类分级" in system_content
        assert "L5" in system_content
        assert "L1" in system_content
