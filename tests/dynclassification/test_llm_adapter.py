"""dynclassification 模块 Layer-3 LLM 适配器与引擎单元测试 / LLM Adapter & Engine Unit Tests.

测试覆盖场景：
- LlmAdapter 延迟加载与优雅降级（模型依赖/目录未就绪时不崩溃，返回 None）
- LlmAdapter.classify() 接口转发与级联分类测试
- LlmAdapter.arbitrate() 冲突仲裁测试（规则冲突时组装上下文并裁定等级）
- Qwen2VLClassifier 响应解析器测试（提取 markdown 包裹的 JSON 字典）
"""

from __future__ import annotations

import os
from typing import Any, Dict
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


class TestLlmAdapter:
    """测试 LlmAdapter 的初始化、优雅降级、classify() 与 arbitrate() 冲突仲裁。"""

    def test_adapter_initialization_defaults(self):
        """测试 LlmAdapter 默认未初始化状态。"""
        adapter = LlmAdapter()
        assert adapter._initialized is False
        assert adapter._classifier is None
        assert adapter._available is True

    def test_lazy_init_unavailable_degradation(self):
        """测试当底层的 Qwen2VLClassifier 无法加载时，is_available 为 False 且返回 None。"""
        adapter = LlmAdapter(model_path="/non_existent/qwen_model_path")

        with patch.object(Qwen2VLClassifier, "__init__", side_effect=ImportError("PyTorch not installed")):
            assert adapter.is_available is False
            # classify 与 arbitrate 应安全降级返回 None
            assert adapter.classify("敏感字段", "L3", 0.6) is None
            assert adapter.arbitrate("user_id", "12345", [], None) is None

    def test_classify_mocked_success(self):
        """测试当 LLM 分类器可用时，classify() 正确转换参数并返回结果字典。"""
        adapter = LlmAdapter()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {
            "final_level": "L3",
            "confidence": 0.92,
            "reasoning": "语义识别为身份证号码",
        }

        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        result = adapter.classify("510101199001011234", "L3", 0.6)
        assert result is not None
        assert result["final_level"] == "L3"
        assert result["confidence"] == 0.92
        assert "身份证号码" in result["reasoning"]

    def test_arbitrate_mocked_conflict_resolution(self):
        """测试 arbitrate() 方法遇到分类冲突时的仲裁流程。"""
        adapter = LlmAdapter()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {
            "final_level": "L3",
            "confidence": 0.95,
            "reasoning": "该字段虽名为 turnover，但实际值为个人敏感号码",
        }

        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        # 构造上下文：规则命中 L3，降级规则匹配 L2
        tag1 = SecurityTag(rule_id="RULE_IDCARD", level="L3", category="PERSONAL_BASIC")
        tag2 = SecurityTag(rule_id="DOWN_OPS", level="L2", category="MANAGEMENT")

        taxonomy = DomainTaxonomy(
            domain="sc_health_db51",
            standard_id="sc_health_db51",
            version="1.0.0",
            description="四川健康指南",
            levels={
                "L2": SensitivityLevelDef(id="L2", name="内部数据", rank=2),
                "L3": SensitivityLevelDef(id="L3", name="敏感数据", rank=3),
            },
            categories={
                "PERSONAL_BASIC": CategoryDef(id="PERSONAL_BASIC", name="个人基本信息数据"),
                "MANAGEMENT": CategoryDef(id="MANAGEMENT", name="管理信息数据"),
            },
            default_level="L1",
        )

        res = adapter.arbitrate("turnover_id", "510101199001011234", [tag1, tag2], taxonomy)
        assert res is not None
        assert res["final_level"] == "L3"
        assert res["confidence"] == 0.95

    def test_classify_exception_fallback(self):
        """测试当底层 LLM 推理或网络超时抛出异常时，classify 优雅捕获并返回 None。"""
        adapter = LlmAdapter()
        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = TimeoutError("Inference timeout (>180s)")

        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        result = adapter.classify("超时测试文本", "L4", 0.5)
        assert result is None


class TestQwen2VLClassifier:
    """测试 Qwen2VLClassifier 的解析器与工具辅助方法。"""

    def test_parse_json_response_clean_json(self):
        """测试解析纯 JSON 串。"""
        from engine.dynclassification.base import SensitivityLevel

        classifier = Qwen2VLClassifier()
        raw_json = '{"final_level": "L3", "confidence": 0.9, "reasoning": "纯文本诊断"}'
        parsed = classifier._parse_json_result(raw_json, SensitivityLevel.L3, 0.5)
        assert parsed is not None
        assert parsed["final_level"] == "L3"
        assert parsed["confidence"] == 0.9

    def test_parse_json_response_markdown_codeblock(self):
        """测试解析带有 markdown 块标记 (```json ... ```) 的 LLM 文本响应。"""
        from engine.dynclassification.base import SensitivityLevel

        classifier = Qwen2VLClassifier()
        raw_text = """依据标准评估，分析结果如下：
```json
{
  "final_level": "L4",
  "confidence": 0.95,
  "reasoning": "识别出艾滋病阳性结果"
}
```
"""
        parsed = classifier._parse_json_result(raw_text, SensitivityLevel.L4, 0.5)
        assert parsed is not None
        assert parsed["final_level"] == "L4"
        assert parsed["confidence"] == 0.95

    def test_parse_json_response_invalid_json_returns_none(self):
        """测试当 LLM 响应不符合 JSON 语法时优雅兜底返回 None。"""
        from engine.dynclassification.base import SensitivityLevel

        classifier = Qwen2VLClassifier()
        invalid_text = "抱歉，由于上下文不足，我无法以 JSON 格式输出结果。"
        parsed = classifier._parse_json_result(invalid_text, SensitivityLevel.L3, 0.5)
        assert parsed is None

    def test_select_device_detects_cuda(self):
        """测试自动检测 CUDA 设备。"""
        import torch

        mock_torch = MagicMock()
        mock_torch.cuda = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        # 10 GB free
        mock_torch.cuda.mem_get_info.return_value = (10 * 1024**3, 12 * 1024**3)
        mock_torch.backends = torch.backends

        classifier = Qwen2VLClassifier()
        with patch.dict(os.environ, {}, clear=True):
            assert classifier._select_device(mock_torch) == "cuda"

    def test_select_device_fallback_cpu_when_vram_insufficient(self):
        """显存不足时 _select_device 回退到 cpu（无 MPS 时）。"""
        mock_torch = MagicMock()
        mock_torch.cuda = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        # 1 GB free (低于 1.6 GB 最低显存要求)
        mock_torch.cuda.mem_get_info.return_value = (1 * 1024**3, 12 * 1024**3)
        # Mock MPS 不可用
        mock_torch.backends.mps.is_available.return_value = False

        classifier = Qwen2VLClassifier()
        with patch.dict(os.environ, {}, clear=True), patch.object(Qwen2VLClassifier, "_is_cuda_compatible", return_value=True):
            assert classifier._select_device(mock_torch) == "cpu"

    def test_select_device_fallback_cpu_when_no_cuda(self):
        """没有 CUDA 时回退到 cpu/mps。"""
        import torch

        mock_torch = MagicMock()
        mock_torch.cuda = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends = torch.backends

        classifier = Qwen2VLClassifier()
        with patch.dict(os.environ, {}, clear=True):
            device = classifier._select_device(mock_torch)
            assert device in ("cpu", "mps")

    def test_classify_with_single_pass_sanitize(self):
        """测试 LlmAdapter.classify 开启 sanitize=True 时的单次融合脱敏转发出参。"""
        adapter = LlmAdapter()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = {
            "final_level": "L4",
            "confidence": 0.95,
            "reasoning": "识别为一期梅毒与RPR阳性描述",
            "sanitized_text": "患者自述外阴溃疡，已给予抗感染治疗",
        }

        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        result = adapter.classify("患者自述外阴溃疡，RPR 1:16阳性", "L4", 0.8, sanitize=True)
        assert result is not None
        assert result["final_level"] == "L4"
        assert "sanitized_text" in result
        assert "RPR" not in result["sanitized_text"]
        mock_classifier.classify.assert_called_once()
        assert mock_classifier.classify.call_args.kwargs.get("sanitize") is True

    def test_dynclassification_service_batch_100_records(self):
        """测试 DynClassificationService 对 100 条扩充模拟记录执行批量分类评测。"""
        import csv
        from pathlib import Path
        from engine.dynclassification import DynClassificationService

        csv_path = Path("data/kangyang.csv")
        if not csv_path.exists():
            pytest.skip("data/kangyang.csv 不存在，跳过批量评测")

        service = DynClassificationService()
        records = []
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        assert len(records) == 100
        # 对 100 条数据执行批量分类评测 (指定 jrt0197 标准)
        high_risk_count = 0
        for idx, rec in enumerate(records):
            resp = service.classify_record(rec, record_index=idx, standard="jrt0197")
            assert resp.record_result is not None
            if resp.record_result.final_level in ["L3", "L4", "L5", "C3", "C4"]:
                high_risk_count += 1

        # 扩充数据集中包含 30% L4 与 20% L5，断言识别出的高敏记录大于 0
        assert high_risk_count > 0
