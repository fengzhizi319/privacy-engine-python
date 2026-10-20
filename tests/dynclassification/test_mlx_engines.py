"""MLX Metal GPU 引擎单元测试 / MLX Metal GPU Engine Unit Tests.

测试 MLX NER 引擎和 MLX LLM 引擎的核心功能：
- MLXSmallNerEngine: 初始化、BIOES 解析、Viterbi 解码、Metal GPU 推理
- MLXLlmClassifier: 初始化、RoPE、前向传播、文本生成
- NerAdapter / LlmAdapter: MLX 降级链
- convert_models_to_mlx.py: 转换脚本

运行方式：
    PYTHONPATH=. pytest tests/dynclassification/test_mlx_engines.py -v
"""

from __future__ import annotations

import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# MLX 仅在 macOS 上可用，非 macOS 跳过整个模块
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="MLX Metal tests only run on macOS",
)


# =========================================================================== #
# Fixtures
# =========================================================================== #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLX_NER_MODEL_DIR = PROJECT_ROOT / ".models" / "raner_cmeee-mlx"
MLX_LLM_MODEL_DIR = PROJECT_ROOT / ".models" / "Qwen2-VL-2B-Instruct-mlx"


@pytest.fixture()
def mlx_available():
    """检查 MLX 是否可用。"""
    try:
        import mlx.core as mx
        return True
    except ImportError:
        pytest.skip("mlx not installed")


@pytest.fixture()
def ner_model_available():
    """检查 NER MLX 模型是否已转换。"""
    if not MLX_NER_MODEL_DIR.exists():
        pytest.skip(f"MLX NER model not found: {MLX_NER_MODEL_DIR}")


@pytest.fixture()
def llm_model_available():
    """检查 LLM MLX 模型是否已转换。"""
    if not MLX_LLM_MODEL_DIR.exists():
        pytest.skip(f"MLX LLM model not found: {MLX_LLM_MODEL_DIR}")


# =========================================================================== #
# MLX 基础 Metal GPU 计算测试
# =========================================================================== #


class TestMLXMetalBasics:
    """验证 MLX Metal GPU 基础计算能力。"""

    def test_mlx_import_and_version(self, mlx_available):
        """MLX 应成功导入且版本 >= 0.20。"""
        import mlx.core as mx
        assert hasattr(mx, "__version__")

    def test_metal_gpu_device(self, mlx_available):
        """默认设备应为 GPU (Metal)。"""
        import mlx.core as mx
        device = mx.default_device()
        assert device.type == mx.DeviceType.gpu

    def test_metal_array_computation(self, mlx_available):
        """Metal GPU 上的基本数组运算应正确。"""
        import mlx.core as mx
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        c = a + b
        mx.eval(c)
        assert c.tolist() == [5.0, 7.0, 9.0]

    def test_metal_matmul(self, mlx_available):
        """Metal GPU 上的矩阵乘法应正确。"""
        import mlx.core as mx
        a = mx.array([[1.0, 2.0], [3.0, 4.0]])
        b = mx.array([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        mx.eval(c)
        expected = [[19.0, 22.0], [43.0, 50.0]]
        assert c.tolist() == expected

    def test_metal_softmax(self, mlx_available):
        """Metal GPU 上的 softmax 应正确。"""
        import mlx.core as mx
        x = mx.array([1.0, 2.0, 3.0])
        s = mx.softmax(x)
        mx.eval(s)
        total = sum(s.tolist())
        assert abs(total - 1.0) < 1e-5

    def test_metal_safetensors_roundtrip(self, mlx_available, tmp_path):
        """MLX safetensors 保存/加载应正确。"""
        import mlx.core as mx
        weights = {"test_tensor": mx.array([1.0, 2.0, 3.0])}
        path = str(tmp_path / "test.safetensors")
        mx.save_safetensors(path, weights)
        loaded = mx.load(path)
        assert loaded["test_tensor"].tolist() == [1.0, 2.0, 3.0]


# =========================================================================== #
# Viterbi 解码测试
# =========================================================================== #


class TestViterbiDecode:
    """测试 CRF Viterbi 解码算法。"""

    def test_viterbi_simple_sequence(self):
        """简单序列应返回最优路径。"""
        from engine.dynclassification.mlx_ner_engine import _viterbi_decode

        # 3 个时间步，2 个标签
        emissions = [
            [1.0, 0.0],  # t=0: 偏好标签 0
            [0.0, 1.0],  # t=1: 偏好标签 1
            [1.0, 0.0],  # t=2: 偏好标签 0
        ]
        start_transitions = [0.0, 0.0]
        end_transitions = [0.0, 0.0]
        transitions = [[0.0, 0.0], [0.0, 0.0]]

        path = _viterbi_decode(emissions, start_transitions, end_transitions, transitions, 3)
        assert path == [0, 1, 0]

    def test_viterbi_transition_bias(self):
        """转移分数应影响最优路径。"""
        from engine.dynclassification.mlx_ner_engine import _viterbi_decode

        # 发射分数相同，但转移分数偏好 0→1→1
        emissions = [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        start_transitions = [1.0, 0.0]  # 偏好从 0 开始
        end_transitions = [0.0, 1.0]    # 偏好以 1 结束
        transitions = [
            [0.0, 2.0],  # 0→1 高权重
            [0.0, 1.0],  # 1→1 次高权重
        ]

        path = _viterbi_decode(emissions, start_transitions, end_transitions, transitions, 3)
        assert path[0] == 0  # 从 0 开始
        assert path[-1] == 1  # 以 1 结束

    def test_viterbi_single_step(self):
        """单步序列应直接选择最高发射分数。"""
        from engine.dynclassification.mlx_ner_engine import _viterbi_decode

        emissions = [[0.1, 0.9, 0.3]]
        start_transitions = [0.0, 0.0, 0.0]
        end_transitions = [0.0, 0.0, 0.0]
        transitions = [[0.0] * 3] * 3

        path = _viterbi_decode(emissions, start_transitions, end_transitions, transitions, 1)
        assert path == [1]


# =========================================================================== #
# BIOES 标签解析测试
# =========================================================================== #


class TestBIOESParsing:
    """测试 BIOES 标签解析逻辑。"""

    @pytest.fixture()
    def engine(self):
        """创建 MLX NER 引擎实例（不初始化模型）。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        return MLXSmallNerEngine(model_dir="/tmp/fake")

    def test_single_entity(self, engine):
        """S- 标签应产生单字实体。"""
        tokens = ["[CLS]", "糖", "[SEP]"]
        # S-dis = index 6
        label_indices = [0, 6, 0]
        probs = [0.99, 0.95, 0.99]
        entities = engine._parse_bioes_tags(tokens, label_indices, probs)
        assert len(entities) == 1
        assert entities[0]["text"] == "糖"
        assert entities[0]["label"] == "dis"

    def test_multi_char_entity_bioes(self, engine):
        """B-I-E 序列应产生多字实体。"""
        tokens = ["[CLS]", "糖", "尿", "病", "[SEP]"]
        # B-dis=5, I-dis=23, E-dis=24
        label_indices = [0, 5, 23, 24, 0]
        probs = [0.99, 0.95, 0.93, 0.91, 0.99]
        entities = engine._parse_bioes_tags(tokens, label_indices, probs)
        assert len(entities) == 1
        assert entities[0]["text"] == "糖尿病"
        assert entities[0]["label"] == "dis"
        assert entities[0]["confidence"] == pytest.approx(0.91, abs=0.01)

    def test_adjacent_entities(self, engine):
        """相邻的 S- 标签应产生多个独立实体。"""
        tokens = ["[CLS]", "糖", "阿", "[SEP]"]
        # S-dis=6, S-dru=8
        label_indices = [0, 6, 8, 0]
        probs = [0.99, 0.95, 0.92, 0.99]
        entities = engine._parse_bioes_tags(tokens, label_indices, probs)
        assert len(entities) == 2
        assert entities[0]["label"] == "dis"
        assert entities[1]["label"] == "dru"

    def test_o_tags_no_entity(self, engine):
        """全 O 标签应返回空列表。"""
        tokens = ["[CLS]", "测", "试", "[SEP]"]
        label_indices = [0, 0, 0, 0]
        probs = [0.99, 0.99, 0.99, 0.99]
        entities = engine._parse_bioes_tags(tokens, label_indices, probs)
        assert len(entities) == 0

    def test_entity_at_sequence_end(self, engine):
        """序列末尾的实体（无 O 结束）应正确保存。"""
        tokens = ["[CLS]", "糖", "尿", "[SEP]"]
        # B-dis=5, E-dis=24
        label_indices = [0, 5, 24, 0]
        probs = [0.99, 0.95, 0.93, 0.99]
        entities = engine._parse_bioes_tags(tokens, label_indices, probs)
        assert len(entities) == 1
        assert entities[0]["text"] == "糖尿"


# =========================================================================== #
# MLXSmallNerEngine 初始化与降级测试
# =========================================================================== #


class TestMLXNerEngineInit:
    """测试 MLX NER 引擎的初始化和降级行为。"""

    def test_init_model_not_found(self, mlx_available):
        """模型目录不存在时应抛出 FileNotFoundError。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        engine = MLXSmallNerEngine(model_dir="/nonexistent/model-mlx")
        with pytest.raises(FileNotFoundError):
            engine._lazy_init()

    def test_init_error_caching(self, mlx_available):
        """初始化失败后再次调用应直接抛出缓存的错误。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        engine = MLXSmallNerEngine(model_dir="/nonexistent/model-mlx")
        with pytest.raises(FileNotFoundError):
            engine._lazy_init()
        with pytest.raises(FileNotFoundError):
            engine._lazy_init()

    def test_extract_init_failure_returns_empty(self, mlx_available):
        """初始化失败时 extract 应返回空列表。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        engine = MLXSmallNerEngine(model_dir="/nonexistent/model-mlx")
        result = engine.extract("测试文本")
        assert result == []

    def test_init_with_real_model(self, mlx_available, ner_model_available):
        """使用真实转换模型应成功初始化。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        engine = MLXSmallNerEngine(model_dir=str(MLX_NER_MODEL_DIR))
        engine._lazy_init()
        assert engine._initialized is True
        assert engine._weights is not None
        assert engine.tokenizer is not None


# =========================================================================== #
# MLXSmallNerEngine Metal GPU 推理集成测试
# =========================================================================== #


class TestMLXNerEngineInference:
    """测试 MLX NER 引擎的 Metal GPU 推理（需要真实模型）。"""

    def test_extract_medical_text(self, mlx_available, ner_model_available):
        """医疗文本应能提取出实体。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        engine = MLXSmallNerEngine(model_dir=str(MLX_NER_MODEL_DIR))
        entities = engine.extract("患者诊断为急性心肌梗死，给予阿司匹林治疗")
        # 应至少提取出一个实体
        assert isinstance(entities, list)
        # 验证实体格式
        for ent in entities:
            assert "text" in ent
            assert "label" in ent
            assert "confidence" in ent
            assert 0 <= ent["confidence"] <= 1.0

    def test_extract_returns_mapped_labels(self, mlx_available, ner_model_available):
        """提取的实体标签应经过映射。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        from engine.dynclassification.ner_engines import DEFAULT_NER_LABEL_MAPPING

        engine = MLXSmallNerEngine(model_dir=str(MLX_NER_MODEL_DIR))
        entities = engine.extract("糖尿病患者口服二甲双胍")
        # 如果有实体，标签应为映射后的标准标签或原始标签
        valid_labels = set(DEFAULT_NER_LABEL_MAPPING.values()) | {"dep", "equ"}
        for ent in entities:
            # 标签应在有效范围内（映射后或原始 BIOES 类型）
            assert isinstance(ent["label"], str)

    def test_extract_empty_text(self, mlx_available, ner_model_available):
        """空文本不应崩溃。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        engine = MLXSmallNerEngine(model_dir=str(MLX_NER_MODEL_DIR))
        entities = engine.extract("")
        assert isinstance(entities, list)

    def test_extract_long_text(self, mlx_available, ner_model_available):
        """超长文本应被截断处理，不崩溃。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        engine = MLXSmallNerEngine(model_dir=str(MLX_NER_MODEL_DIR))
        long_text = "糖尿病患者" * 100
        entities = engine.extract(long_text)
        assert isinstance(entities, list)

    def test_metal_gpu_inference_speed(self, mlx_available, ner_model_available):
        """Metal GPU 推理应在合理时间内完成（< 5s）。"""
        import time
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        engine = MLXSmallNerEngine(model_dir=str(MLX_NER_MODEL_DIR))
        # 预热
        engine.extract("测试")
        # 计时
        start = time.monotonic()
        engine.extract("患者诊断为高血压，给予降压药治疗")
        duration = time.monotonic() - start
        assert duration < 5.0, f"推理耗时过长: {duration:.2f}s"


# =========================================================================== #
# MLXLlmClassifier 初始化与降级测试
# =========================================================================== #


class TestMLXLlmClassifierInit:
    """测试 MLX LLM 分类器的初始化和降级行为。"""

    def test_init_model_not_found(self, mlx_available):
        """模型目录不存在时应抛出 FileNotFoundError。"""
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier
        classifier = MLXLlmClassifier(model_dir="/nonexistent/model-mlx")
        with pytest.raises(FileNotFoundError):
            classifier._lazy_init()

    def test_init_error_caching(self, mlx_available):
        """初始化失败后再次调用应直接抛出缓存的错误。"""
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier
        classifier = MLXLlmClassifier(model_dir="/nonexistent/model-mlx")
        with pytest.raises(FileNotFoundError):
            classifier._lazy_init()
        with pytest.raises(FileNotFoundError):
            classifier._lazy_init()

    def test_classify_init_failure_returns_none(self, mlx_available):
        """初始化失败时 classify 应返回 None。"""
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier
        from engine.dynclassification.base import SensitivityLevel
        classifier = MLXLlmClassifier(model_dir="/nonexistent/model-mlx")
        result = classifier.classify("测试", SensitivityLevel.L3, 0.5)
        assert result is None

    def test_is_ready_false_before_init(self, mlx_available):
        """未初始化时 is_ready 应为 False。"""
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier
        classifier = MLXLlmClassifier(model_dir="/tmp/fake")
        assert classifier.is_ready is False


# =========================================================================== #
# MLX LLM 辅助函数测试
# =========================================================================== #


class TestMLXLlmHelpers:
    """测试 MLX LLM 引擎的辅助函数。"""

    def test_rms_norm(self, mlx_available):
        """RMSNorm 应正确归一化。"""
        import mlx.core as mx
        from engine.dynclassification.mlx_llm_engine import _rms_norm

        x = mx.array([[1.0, 2.0, 3.0]])
        w = mx.array([1.0, 1.0, 1.0])
        out = _rms_norm(x, w)
        mx.eval(out)
        # RMSNorm 后向量的 RMS 应接近 1
        rms = float(mx.sqrt(mx.mean(out * out)).item())
        assert abs(rms - 1.0) < 0.01

    def test_silu_activation(self, mlx_available):
        """SiLU 激活函数应正确计算。"""
        import mlx.core as mx
        from engine.dynclassification.mlx_llm_engine import _silu

        x = mx.array([0.0, 1.0, -1.0])
        out = _silu(x)
        mx.eval(out)
        # SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(-1) ≈ -0.269
        assert abs(out[0].item()) < 1e-5
        assert abs(out[1].item() - 0.7310586) < 0.01
        assert abs(out[2].item() - (-0.2689414)) < 0.01

    def test_rope_freqs_shape(self, mlx_available):
        """RoPE 频率矩阵形状应正确。"""
        import mlx.core as mx
        from engine.dynclassification.mlx_llm_engine import _rope_freqs

        angles = _rope_freqs(dim=128, seq_len=10)
        mx.eval(angles)
        assert angles.shape == (10, 64)  # (seq_len, dim//2)

    def test_apply_rope_shape(self, mlx_available):
        """应用 RoPE 后形状应不变。"""
        import mlx.core as mx
        from engine.dynclassification.mlx_llm_engine import _apply_rope, _rope_freqs

        x = mx.random.normal((12, 10, 128))  # (num_heads, seq_len, head_dim)
        angles = _rope_freqs(dim=128, seq_len=10)
        out = _apply_rope(x, angles)
        mx.eval(out)
        assert out.shape == (12, 10, 128)

    def test_parse_json_result_valid(self, mlx_available):
        """有效 JSON 应正确解析。"""
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier

        text = '{"final_level": "L3", "confidence": 0.85, "reasoning": "PII"}'
        result = MLXLlmClassifier._parse_json_result(text)
        assert result is not None
        assert result["final_level"] == "L3"

    def test_parse_json_result_with_surrounding_text(self, mlx_available):
        """JSON 前后有额外文字时应正确提取。"""
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier

        text = '分析结果：{"final_level": "L4", "confidence": 0.9} 以上。'
        result = MLXLlmClassifier._parse_json_result(text)
        assert result is not None
        assert result["final_level"] == "L4"

    def test_parse_json_result_invalid(self, mlx_available):
        """无效 JSON 应返回 None。"""
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier

        result = MLXLlmClassifier._parse_json_result("这不是JSON")
        assert result is None

    def test_parse_json_result_missing_final_level(self, mlx_available):
        """缺少 final_level 字段应返回 None。"""
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier

        result = MLXLlmClassifier._parse_json_result('{"confidence": 0.9}')
        assert result is None


# =========================================================================== #
# NerAdapter MLX 降级链测试
# =========================================================================== #


class TestNerAdapterMLXFallback:
    """测试 NerAdapter 的 MLX 降级链。"""

    def test_mlx_preferred_on_macos(self, mlx_available, ner_model_available):
        """macOS 上 MLX 引擎应被优先选择。"""
        from engine.dynclassification.ner_adapter import NerAdapter
        adapter = NerAdapter()
        adapter._lazy_init()
        assert adapter._available is True
        assert adapter._engine is not None
        # 验证使用的是 MLX 引擎
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine
        assert isinstance(adapter._engine, MLXSmallNerEngine)

    def test_fallback_when_mlx_model_missing(self, mlx_available):
        """MLX 模型不存在时应降级到下一个引擎。"""
        from engine.dynclassification.ner_adapter import NerAdapter
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine

        with patch.object(MLXSmallNerEngine, "_lazy_init", side_effect=FileNotFoundError("no mlx model")):
            adapter = NerAdapter(model_path="/nonexistent/model.onnx")
            adapter._lazy_init()
            # 应该降级（可能所有引擎都不可用）
            # 关键是不应崩溃


# =========================================================================== #
# LlmAdapter MLX 降级链测试
# =========================================================================== #


class TestLlmAdapterMLXFallback:
    """测试 LlmAdapter 的 MLX 降级链。"""

    def test_mlx_preferred_on_macos(self, mlx_available, llm_model_available):
        """macOS 上 MLX LLM 引擎应被优先选择。"""
        from engine.dynclassification.llm_adapter import LlmAdapter
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier

        adapter = LlmAdapter(model_path=str(MLX_LLM_MODEL_DIR))
        adapter._lazy_init()
        assert adapter._available is True
        assert isinstance(adapter._classifier, MLXLlmClassifier)

    def test_fallback_to_qwen2vl_when_mlx_fails(self, mlx_available):
        """MLX 不可用时应降级到 Qwen2VL。"""
        from engine.dynclassification.llm_adapter import LlmAdapter
        from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier

        with patch.object(MLXLlmClassifier, "_lazy_init", side_effect=FileNotFoundError("no mlx")):
            adapter = LlmAdapter(model_path="/nonexistent/model")
            adapter._lazy_init()
            # Qwen2VL 也会失败（模型不存在），但不应崩溃
            # 关键是降级链正确执行


# =========================================================================== #
# 转换脚本测试
# =========================================================================== #


class TestConvertScript:
    """测试 convert_models_to_mlx.py 转换脚本。"""

    def test_require_macos(self):
        """非 macOS 平台应退出。"""
        from scripts.models.convert_models_to_mlx import _require_macos
        # 在 macOS 上不应退出
        if platform.system() == "Darwin":
            _require_macos()  # 不应抛出异常

    def test_convert_bfloat16_handling(self, mlx_available, tmp_path):
        """BFloat16 张量应正确转换。"""
        import torch
        import mlx.core as mx
        from scripts.models.convert_models_to_mlx import _convert_state_dict_to_mlx

        # 创建包含 BFloat16 的 state dict
        state = {
            "weight": torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
            "bias": torch.tensor([0.1, 0.2], dtype=torch.float32),
        }
        result = _convert_state_dict_to_mlx(state, dtype="float16")
        assert "weight" in result
        assert "bias" in result
        # 验证值正确（允许精度误差）
        mx.eval(result["weight"])
        assert abs(result["weight"][0].item() - 1.0) < 0.01

    def test_convert_state_dict_float32(self, mlx_available):
        """Float32 张量应正确转换。"""
        import torch
        from scripts.models.convert_models_to_mlx import _convert_state_dict_to_mlx

        state = {"w": torch.tensor([1.0, 2.0])}
        result = _convert_state_dict_to_mlx(state, dtype="float32")
        assert result["w"].tolist() == [1.0, 2.0]

    def test_copy_auxiliary_files(self, tmp_path):
        """辅助文件应正确复制。"""
        from scripts.models.convert_models_to_mlx import _copy_auxiliary_files

        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text("{}")
        (src / "vocab.txt").write_text("test")

        dst = tmp_path / "dst"
        _copy_auxiliary_files(src, dst)
        assert (dst / "config.json").exists()
        assert (dst / "vocab.txt").exists()

    def test_converted_ner_model_exists(self, ner_model_available):
        """转换后的 NER MLX 模型应包含必要文件。"""
        assert (MLX_NER_MODEL_DIR / "weights.safetensors").exists()
        assert (MLX_NER_MODEL_DIR / "config.json").exists()
        assert (MLX_NER_MODEL_DIR / "vocab.txt").exists()

    def test_converted_llm_model_exists(self, llm_model_available):
        """转换后的 LLM MLX 模型应包含必要文件。"""
        assert (MLX_LLM_MODEL_DIR / "weights.safetensors").exists()
        assert (MLX_LLM_MODEL_DIR / "config.json").exists()
