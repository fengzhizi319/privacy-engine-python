"""Layer-3 LLM 与 vLLM / OpenAI HTTP 服务集成冒烟测试。

测试场景涵盖：
1. .env 环境变量解析与多模式切换
2. OpenAILlmClassifier / VLLMLlmClassifier HTTP API 通信与 JSON 结果解析
3. LlmAdapter 动态后端选择 (vllm / qwen3 / mlx / openai)
4. ClassificationFunnel 三层漏斗与 vLLM 后端联动及优雅降级
"""

from __future__ import annotations

import json
import os
import io
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from engine.env_loader import load_env_file
from engine.dynclassification.base import SensitivityLevel
from engine.dynclassification.llm_adapter import LlmAdapter
from engine.dynclassification.llm_engines import (
    OpenAILlmClassifier,
    VLLMLlmClassifier,
)


class TestEnvLoader:
    """测试 .env 文件解析与自动加载。"""

    def test_load_env_file_parses_kv(self, tmp_path):
        env_file = tmp_path / ".env.test"
        env_file.write_text(
            '# 模拟配置文件\n'
            'PRIVACY_LLM_PROVIDER=vllm\n'
            'PRIVACY_LLM_API_BASE="http://127.0.0.1:8000/v1"\n'
            'PRIVACY_LLM_MODEL_NAME=Qwen3.5-0.8B-Privacy-Classifier-Smoother\n',
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            loaded = load_env_file(dotenv_path=env_file, override=True)
            assert loaded is True
            assert os.environ.get("PRIVACY_LLM_PROVIDER") == "vllm"
            assert os.environ.get("PRIVACY_LLM_API_BASE") == "http://127.0.0.1:8000/v1"
            assert (
                os.environ.get("PRIVACY_LLM_MODEL_NAME")
                == "Qwen3.5-0.8B-Privacy-Classifier-Smoother"
            )


class TestOpenAILlmClassifier:
    """测试 OpenAILlmClassifier 与 vLLM OpenAI 兼容 HTTP 服务通信。"""

    def test_init_and_url_formatting(self):
        classifier = OpenAILlmClassifier(
            api_base="http://127.0.0.1:8000/v1",
            model_name="Qwen3.5-Test",
            api_key="test_key",
        )
        assert classifier.chat_url == "http://127.0.0.1:8000/v1/chat/completions"
        assert classifier.model_name == "Qwen3.5-Test"
        assert classifier.api_key == "test_key"
        assert classifier.is_ready is True

    def test_classify_success_http_mock(self):
        """模拟 vLLM HTTP API 成功返回包含结构化 JSON 的 HTTP 响应。"""
        classifier = OpenAILlmClassifier(
            api_base="http://127.0.0.1:8000/v1",
            model_name="Qwen3.5-0.8B-Privacy-Classifier-Smoother",
        )

        mock_vllm_response = {
            "id": "chatcmpl-vllm-12345",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "```json\n"
                            "{\n"
                            '  "final_level": "L4",\n'
                            '  "sub_category": "INFECTIOUS_DISEASE",\n'
                            '  "confidence": 0.96,\n'
                            '  "reasoning": "识别出明确的 HIV 阳性诊断结果",\n'
                            '  "needs_human_review": false\n'
                            "}\n"
                            "```"
                        ),
                    }
                }
            ],
        }
        mock_body_bytes = json.dumps(mock_vllm_response).encode("utf-8")

        mock_response_obj = MagicMock()
        mock_response_obj.status = 200
        mock_response_obj.read.return_value = mock_body_bytes
        mock_response_obj.__enter__.return_value = mock_response_obj

        with patch("urllib.request.urlopen", return_value=mock_response_obj) as mock_urlopen:
            res = classifier.classify(
                text="患者确诊为 HIV 阳性住院治疗",
                upstream_level=SensitivityLevel.L3,
                upstream_confidence=0.6,
            )

            assert res is not None
            assert res["final_level"] == "L4"
            assert res["confidence"] == 0.96
            assert "HIV" in res["reasoning"]

            # 验证请求参数正确发送到了 vLLM endpoint
            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://127.0.0.1:8000/v1/chat/completions"
            sent_payload = json.loads(req.data.decode("utf-8"))
            assert sent_payload["model"] == "Qwen3.5-0.8B-Privacy-Classifier-Smoother"
            assert "患者确诊为 HIV 阳性" in sent_payload["messages"][1]["content"]

    def test_classify_connection_error_degradation(self):
        """测试当 vLLM 服务未启动或网络故障时，优雅降级返回 None。"""
        import urllib.error

        classifier = OpenAILlmClassifier(
            api_base="http://127.0.0.1:59999/v1",
            timeout=1.0,
        )

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            res = classifier.classify(
                text="连接超时测试",
                upstream_level=SensitivityLevel.L2,
                upstream_confidence=0.5,
            )
            assert res is None


class TestLlmAdapterVllmIntegration:
    """测试 LlmAdapter 根据 PRIVACY_LLM_PROVIDER 动态加载 vLLM HTTP 引擎。"""

    def test_adapter_vllm_provider_selection(self):
        with patch.dict(
            os.environ,
            {
                "PRIVACY_LLM_PROVIDER": "vllm",
                "PRIVACY_LLM_API_BASE": "http://127.0.0.1:8000/v1",
            },
            clear=True,
        ):
            adapter = LlmAdapter()
            assert adapter.is_available is True
            assert isinstance(adapter._classifier, OpenAILlmClassifier)
            assert adapter._classifier.chat_url == "http://127.0.0.1:8000/v1/chat/completions"

    def test_adapter_openai_alias_selection(self):
        with patch.dict(
            os.environ,
            {
                "PRIVACY_LLM_PROVIDER": "openai",
                "PRIVACY_LLM_API_BASE": "http://127.0.0.1:11434/v1",
            },
            clear=True,
        ):
            adapter = LlmAdapter()
            assert adapter.is_available is True
            assert isinstance(adapter._classifier, VLLMLlmClassifier)
            assert adapter._classifier.chat_url == "http://127.0.0.1:11434/v1/chat/completions"

    def test_funnel_arbitration_with_vllm_adapter(self):
        """测试在 ClassificationFunnel 流程中，Layer-3 使用 vLLM 服务进行分类通信。"""
        adapter = LlmAdapter()
        mock_classifier = MagicMock(spec=OpenAILlmClassifier)
        mock_classifier.classify.return_value = {
            "final_level": "L4",
            "confidence": 0.94,
            "reasoning": "vLLM 模型二次判决为高风险病历",
        }

        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        result = adapter.classify("患者重度精神分裂症诊疗记录", SensitivityLevel.L3, 0.6)
        assert result is not None
        assert result["final_level"] == "L4"
        assert result["confidence"] == 0.94
