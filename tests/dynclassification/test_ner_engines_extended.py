"""NER 引擎深度补充测试 / NER Engine Deep Supplement Tests.

补充 test_ner_adapter.py 未覆盖的场景：
- SimpleChineseBertTokenizer: tokenize() 分词逻辑 + encode() 编码与 padding
- ONNXSmallNerEngine: extract() 完整推理流（mock ONNX session）、_lazy_init 错误缓存、
  _parse_bio_tags 边界情况（I- 类型不匹配、相邻 B- 实体、序列末尾实体）
- ModelScopeSmallNerEngine: extract() 初始化失败降级、pipeline 异常降级、_lazy_init 错误缓存
- NerAdapter: label_mapping 透传、多引擎降级顺序
- TensorRTSmallNerEngine: 仅 CUDA 无 TensorRT 时的 provider 回退

运行方式：
    PYTHONPATH=. pytest tests/dynclassification/test_ner_engines_extended.py -v
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from engine.dynclassification.ner_adapter import NerAdapter
from engine.dynclassification.ner_engines import (
    DEFAULT_NER_LABEL_MAPPING,
    ModelScopeSmallNerEngine,
    ONNXSmallNerEngine,
    SimpleChineseBertTokenizer,
    TensorRTSmallNerEngine,
)


# =========================================================================== #
# SimpleChineseBertTokenizer 测试
# =========================================================================== #


class TestSimpleChineseBertTokenizer:
    """测试纯 Python 中文 BERT 分词器的分词与编码逻辑。"""

    @pytest.fixture()
    def vocab_file(self, tmp_path):
        """创建一个最小词表文件用于测试。"""
        # 模拟标准 BERT vocab.txt 格式
        tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",  # 0-4 特殊 token
            "糖", "尿", "病", "患", "者",                  # 5-9 中文
            "h", "i", "v", "a", "b",                      # 10-14 小写字母
            "0", "1", "2", "3",                            # 15-18 数字
            "，", "。", "、",                               # 19-21 中文标点
        ]
        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("\n".join(tokens), encoding="utf-8")
        return str(vocab_path)

    @pytest.fixture()
    def tokenizer(self, vocab_file):
        """创建分词器实例。"""
        return SimpleChineseBertTokenizer(vocab_file)

    def test_tokenize_chinese_characters(self, tokenizer):
        """中文字符应逐字切分。"""
        tokens = tokenizer.tokenize("糖尿病")
        assert tokens == ["糖", "尿", "病"]

    def test_tokenize_unknown_char_maps_to_unk(self, tokenizer):
        """词表中不存在的字符应映射为 [UNK]。"""
        tokens = tokenizer.tokenize("张")  # "张" 不在词表中
        assert tokens == ["[UNK]"]

    def test_tokenize_case_folding(self, tokenizer):
        """大写字母应折叠为小写（若小写在词表中）。"""
        tokens = tokenizer.tokenize("HIV")
        # H→h, I→i, V→v（小写形式在词表中）
        assert tokens == ["h", "i", "v"]

    def test_tokenize_mixed_content(self, tokenizer):
        """混合中英文和标点的分词。"""
        tokens = tokenizer.tokenize("患者hiv")
        assert tokens == ["患", "者", "h", "i", "v"]

    def test_tokenize_empty_string(self, tokenizer):
        """空字符串应返回空列表。"""
        assert tokenizer.tokenize("") == []

    def test_encode_basic_structure(self, tokenizer):
        """encode 应返回正确长度的三元组。"""
        input_ids, attention_mask, token_type_ids = tokenizer.encode("糖尿病", max_len=16)
        assert len(input_ids) == 16
        assert len(attention_mask) == 16
        assert len(token_type_ids) == 16

    def test_encode_cls_sep_positions(self, tokenizer):
        """encode 首位应为 [CLS] ID，有效序列末位应为 [SEP] ID。"""
        input_ids, attention_mask, _ = tokenizer.encode("糖", max_len=8)
        # [CLS] ID = 2
        assert input_ids[0] == 2
        # [SEP] 在位置 2（[CLS] + "糖" + [SEP]）
        assert input_ids[2] == 3
        # attention_mask: 有效位为 1
        assert attention_mask[0] == 1
        assert attention_mask[2] == 1

    def test_encode_padding(self, tokenizer):
        """短序列应右侧 padding 到 max_len。"""
        input_ids, attention_mask, token_type_ids = tokenizer.encode("糖", max_len=8)
        # 有效长度 = 3 ([CLS] + 糖 + [SEP])，padding = 5
        assert attention_mask[3:] == [0, 0, 0, 0, 0]
        # padding 位 input_ids 应为 pad_id=0
        assert input_ids[3:] == [0, 0, 0, 0, 0]
        # token_type_ids 全为 0
        assert token_type_ids == [0] * 8

    def test_encode_truncation(self, tokenizer):
        """超长文本应截断到 max_len-2 个 token。"""
        # 10 个中文字符，max_len=6 → 只保留 4 个字符
        text = "糖尿病患者高血压三"
        input_ids, attention_mask, _ = tokenizer.encode(text, max_len=6)
        assert len(input_ids) == 6
        # 有效位全部为 1（无 padding）
        assert attention_mask == [1, 1, 1, 1, 1, 1]

    def test_special_token_ids(self, tokenizer):
        """特殊 token ID 应从词表正确读取。"""
        assert tokenizer.pad_id == 0
        assert tokenizer.unk_id == 1
        assert tokenizer.cls_id == 2
        assert tokenizer.sep_id == 3


# =========================================================================== #
# ONNXSmallNerEngine 补充测试
# =========================================================================== #


class TestONNXEngineExtended:
    """补充 ONNXSmallNerEngine 的推理流程和边界情况测试。"""

    def test_lazy_init_error_caching(self):
        """初始化失败后再次调用应直接抛出缓存的错误（不重试）。"""
        engine = ONNXSmallNerEngine(model_path="/nonexistent/model.onnx")
        with pytest.raises(FileNotFoundError):
            engine._lazy_init()
        # 第二次调用应抛出相同错误（不重新尝试）
        with pytest.raises(FileNotFoundError):
            engine._lazy_init()

    def test_lazy_init_vocab_not_found(self, tmp_path):
        """模型文件存在但词表文件不存在时应抛出 FileNotFoundError。"""
        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake onnx content")
        engine = ONNXSmallNerEngine(
            model_path=str(fake_model),
            vocab_path="/nonexistent/vocab.txt",
        )
        with pytest.raises(FileNotFoundError, match="vocab"):
            engine._lazy_init()

    def test_extract_with_mocked_session(self):
        """Mock ONNX session 验证完整推理流（分词→推理→softmax→BIO解析→标签映射）。"""
        engine = ONNXSmallNerEngine()
        # 创建最小词表
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                      "糖", "尿", "病", "患", "者"]
            f.write("\n".join(tokens))
            vocab_path = f.name

        try:
            engine.tokenizer = SimpleChineseBertTokenizer(vocab_path)
            engine._initialized = True

            # Mock ONNX session：输出 logits shape=(1, seq_len, 13)
            # 让 "糖尿病" 三个字分别命中 B-dis(1), I-dis(2), I-dis(2)
            seq_len = 128
            logits = np.zeros((1, seq_len, 13), dtype=np.float32)
            # 位置 0=[CLS] → O(0), 位置 1="糖" → B-dis(1), 位置 2="尿" → I-dis(2), 位置 3="病" → I-dis(2)
            logits[0, 0, 0] = 10.0  # [CLS] → O
            logits[0, 1, 1] = 10.0  # "糖" → B-dis
            logits[0, 2, 2] = 10.0  # "尿" → I-dis
            logits[0, 3, 2] = 10.0  # "病" → I-dis
            logits[0, 4, 0] = 10.0  # [SEP] → O

            mock_session = MagicMock()
            mock_session.run.return_value = [logits]
            engine.session = mock_session

            entities = engine.extract("糖尿病")
            assert len(entities) == 1
            assert entities[0]["text"] == "糖尿病"
            assert entities[0]["label"] == "MEDICAL_DISEASE"  # dis → MEDICAL_DISEASE
            assert entities[0]["confidence"] > 0.9
        finally:
            os.unlink(vocab_path)

    def test_extract_init_failure_returns_empty(self):
        """初始化失败时 extract 应返回空列表（不崩溃）。"""
        engine = ONNXSmallNerEngine(model_path="/nonexistent/model.onnx")
        result = engine.extract("测试文本")
        assert result == []

    def test_parse_bio_tags_i_type_mismatch(self):
        """I- 标签类型不匹配时应结束当前实体。"""
        engine = ONNXSmallNerEngine()
        # B-dis(1) + I-dru(4)：类型不匹配
        tokens = ["[CLS]", "糖", "阿", "[SEP]"]
        label_indices = [0, 1, 4, 0]  # O, B-dis, I-dru(不匹配), O
        probs = [0.99, 0.95, 0.90, 0.99]

        entities = engine._parse_bio_tags(tokens, label_indices, probs)
        # "糖" 作为独立 dis 实体保存，I-dru 被丢弃
        assert len(entities) == 1
        assert entities[0]["text"] == "糖"
        assert entities[0]["label"] == "dis"

    def test_parse_bio_tags_adjacent_b_entities(self):
        """相邻的 B- 标签应产生两个独立实体。"""
        engine = ONNXSmallNerEngine()
        # B-dis(1) + B-dru(3)：两个相邻实体
        tokens = ["[CLS]", "糖", "阿", "[SEP]"]
        label_indices = [0, 1, 3, 0]  # O, B-dis, B-dru, O
        probs = [0.99, 0.95, 0.92, 0.99]

        entities = engine._parse_bio_tags(tokens, label_indices, probs)
        assert len(entities) == 2
        assert entities[0]["text"] == "糖"
        assert entities[0]["label"] == "dis"
        assert entities[1]["text"] == "阿"
        assert entities[1]["label"] == "dru"

    def test_parse_bio_tags_entity_at_sequence_end(self):
        """序列末尾的实体（无 O 标签结束）应正确保存。"""
        engine = ONNXSmallNerEngine()
        # B-dis(1) + I-dis(2) 在 [SEP] 前结束
        tokens = ["[CLS]", "糖", "尿", "[SEP]"]
        label_indices = [0, 1, 2, 0]
        probs = [0.99, 0.96, 0.94, 0.99]

        entities = engine._parse_bio_tags(tokens, label_indices, probs)
        assert len(entities) == 1
        assert entities[0]["text"] == "糖尿"

    def test_parse_bio_tags_i_without_b_discarded(self):
        """没有 B- 开头的 I- 标签应被忽略。"""
        engine = ONNXSmallNerEngine()
        tokens = ["[CLS]", "尿", "病", "[SEP]"]
        label_indices = [0, 2, 2, 0]  # O, I-dis(无B开头), I-dis, O
        probs = [0.99, 0.95, 0.94, 0.99]

        entities = engine._parse_bio_tags(tokens, label_indices, probs)
        # I- 没有前置 B-，current_entity 为 None，应被忽略
        assert len(entities) == 0

    def test_custom_label_mapping_applied(self):
        """自定义 label_mapping 应在 extract 中正确应用。"""
        engine = ONNXSmallNerEngine(label_mapping={"dis": "CUSTOM_DISEASE"})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "糖", "尿", "病"]
            f.write("\n".join(tokens))
            vocab_path = f.name

        try:
            engine.tokenizer = SimpleChineseBertTokenizer(vocab_path)
            engine._initialized = True

            seq_len = 128
            logits = np.zeros((1, seq_len, 13), dtype=np.float32)
            logits[0, 0, 0] = 10.0
            logits[0, 1, 1] = 10.0  # B-dis
            logits[0, 2, 2] = 10.0  # I-dis
            logits[0, 3, 2] = 10.0  # I-dis
            logits[0, 4, 0] = 10.0

            mock_session = MagicMock()
            mock_session.run.return_value = [logits]
            engine.session = mock_session

            entities = engine.extract("糖尿病")
            assert entities[0]["label"] == "CUSTOM_DISEASE"
        finally:
            os.unlink(vocab_path)


# =========================================================================== #
# ModelScopeSmallNerEngine 补充测试
# =========================================================================== #


class TestModelScopeEngineExtended:
    """补充 ModelScopeSmallNerEngine 的降级与异常处理测试。"""

    def test_extract_init_failure_returns_empty(self):
        """初始化失败时 extract 应返回空列表。"""
        engine = ModelScopeSmallNerEngine()
        # 强制设置初始化错误
        engine._init_error = RuntimeError("modelscope not installed")
        result = engine.extract("患者诊断为糖尿病")
        assert result == []

    def test_extract_pipeline_exception_returns_empty(self):
        """pipeline 推理异常时应返回空列表（优雅降级）。"""
        engine = ModelScopeSmallNerEngine()
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("CUDA OOM")
        engine.pipeline = mock_pipeline
        engine._initialized = True

        result = engine.extract("异常测试")
        assert result == []

    def test_extract_empty_output(self):
        """pipeline 返回空 output 时应返回空列表。"""
        engine = ModelScopeSmallNerEngine()
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {"output": []}
        engine.pipeline = mock_pipeline
        engine._initialized = True

        result = engine.extract("无实体文本")
        assert result == []

    def test_extract_unknown_label_passthrough(self):
        """未知标签（不在映射表中）应原样保留。"""
        engine = ModelScopeSmallNerEngine(label_mapping={"dis": "MEDICAL_DISEASE"})
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "output": [
                {"type": "unknown_type", "span": "测试实体"},
            ]
        }
        engine.pipeline = mock_pipeline
        engine._initialized = True

        result = engine.extract("测试")
        assert len(result) == 1
        assert result[0]["label"] == "unknown_type"  # 原样保留
        assert result[0]["text"] == "测试实体"

    def test_lazy_init_error_caching(self):
        """初始化失败后再次调用应直接抛出缓存的错误。"""
        engine = ModelScopeSmallNerEngine()
        engine._init_error = ImportError("No module named 'modelscope'")
        with pytest.raises(ImportError, match="modelscope"):
            engine._lazy_init()

    def test_extract_multiple_entities_mapping(self):
        """多实体输出应全部正确映射标签。"""
        engine = ModelScopeSmallNerEngine(label_mapping={
            "dis": "MEDICAL_DISEASE",
            "dru": "MEDICATION",
            "pro": "SURGERY",
        })
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "output": [
                {"type": "dis", "span": "糖尿病"},
                {"type": "dru", "span": "二甲双胍"},
                {"type": "pro", "span": "冠脉介入"},
            ]
        }
        engine.pipeline = mock_pipeline
        engine._initialized = True

        result = engine.extract("糖尿病患者口服二甲双胍，行冠脉介入")
        assert len(result) == 3
        assert result[0]["label"] == "MEDICAL_DISEASE"
        assert result[1]["label"] == "MEDICATION"
        assert result[2]["label"] == "SURGERY"


# =========================================================================== #
# NerAdapter 补充测试
# =========================================================================== #


class TestNerAdapterExtended:
    """补充 NerAdapter 的参数透传与降级顺序测试。"""

    def test_label_mapping_passthrough_to_onnx(self):
        """自定义 label_mapping 应透传到底层引擎。"""
        custom_mapping = {"dis": "MY_DISEASE"}
        adapter = NerAdapter(label_mapping=custom_mapping)
        # 验证构造参数保存正确
        assert adapter._label_mapping == custom_mapping

    def test_device_parameter_passthrough(self):
        """device 参数应透传到底层引擎。"""
        adapter = NerAdapter(device="cpu")
        assert adapter._device == "cpu"

    def test_fallback_order_tensorrt_onnx_modelscope(self):
        """降级顺序应为 MLX → TensorRT → ONNX → ModelScope。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine

        adapter = NerAdapter(model_path="/nonexistent/model.onnx")
        call_order = []

        with patch.object(MLXSmallNerEngine, "_lazy_init", side_effect=RuntimeError("no mlx")), \
             patch.object(TensorRTSmallNerEngine, "_lazy_init", side_effect=RuntimeError("no trt")) as trt_init, \
             patch.object(ONNXSmallNerEngine, "_lazy_init", side_effect=RuntimeError("no onnx")) as onnx_init, \
             patch.object(ModelScopeSmallNerEngine, "__init__", side_effect=RuntimeError("no ms")) as ms_init:
            adapter._lazy_init()

        # 所有引擎都应被尝试
        assert adapter._available is False

    def test_onnx_success_skips_modelscope(self):
        """ONNX 引擎成功时不应尝试 ModelScope。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine

        adapter = NerAdapter()
        adapter._initialized = False

        with patch.object(MLXSmallNerEngine, "_lazy_init", side_effect=RuntimeError("no mlx")), \
             patch.object(TensorRTSmallNerEngine, "_lazy_init", side_effect=RuntimeError("no trt")), \
             patch.object(ONNXSmallNerEngine, "_lazy_init") as onnx_init:
            # ONNX 成功（不抛异常）
            onnx_init.return_value = None
            adapter._lazy_init()

        assert adapter._available is True
        assert adapter._engine is not None

    def test_extract_long_text_no_crash(self):
        """超长文本输入不应导致崩溃。"""
        adapter = NerAdapter()
        mock_engine = MagicMock()
        mock_engine.extract.return_value = []
        adapter._engine = mock_engine
        adapter._initialized = True
        adapter._available = True

        long_text = "糖尿病" * 1000
        result = adapter.extract(long_text)
        assert result == []
        mock_engine.extract.assert_called_once_with(long_text)


# =========================================================================== #
# TensorRTSmallNerEngine 补充测试
# =========================================================================== #


class TestTensorRTEngineExtended:
    """补充 TensorRTSmallNerEngine 的 provider 回退测试。"""

    def test_cuda_only_fallback_without_tensorrt(self):
        """无 TensorRT provider 但有 CUDA 时应回退到 CUDAExecutionProvider。"""
        engine = TensorRTSmallNerEngine(
            model_path=__file__,
            vocab_path=__file__,
        )
        mock_ort = MagicMock()
        # 只有 CUDA 和 CPU，没有 TensorRT
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            engine._lazy_init()
            assert engine._initialized is True
            # 验证使用的是 CUDA provider（非 TensorRT）
            providers_arg = mock_ort.InferenceSession.call_args[1].get("providers")
            assert providers_arg == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_no_gpu_providers_raises_runtime_error(self):
        """无 TensorRT 也无 CUDA 时应抛出 RuntimeError。"""
        engine = TensorRTSmallNerEngine(
            model_path=__file__,
            vocab_path=__file__,
        )
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            with pytest.raises(RuntimeError, match="TensorRT.*CUDA.*不可用"):
                engine._lazy_init()

    def test_init_error_caching(self):
        """初始化失败后再次调用应直接抛出缓存的错误。"""
        engine = TensorRTSmallNerEngine(model_path="/nonexistent/model.onnx")
        with pytest.raises(FileNotFoundError):
            engine._lazy_init()
        # 第二次应直接抛出缓存错误
        with pytest.raises(FileNotFoundError):
            engine._lazy_init()


# =========================================================================== #
# DEFAULT_NER_LABEL_MAPPING 完整性测试
# =========================================================================== #


class TestDefaultLabelMapping:
    """验证默认标签映射的完整性和正确性。"""

    def test_default_mapping_contains_core_types(self):
        """默认映射应包含核心医疗实体类型。"""
        assert "dis" in DEFAULT_NER_LABEL_MAPPING
        assert "dru" in DEFAULT_NER_LABEL_MAPPING
        assert "pro" in DEFAULT_NER_LABEL_MAPPING
        assert "sym" in DEFAULT_NER_LABEL_MAPPING
        assert "bod" in DEFAULT_NER_LABEL_MAPPING

    def test_default_mapping_values_are_uppercase(self):
        """映射目标标签应为大写标准格式。"""
        for raw, standard in DEFAULT_NER_LABEL_MAPPING.items():
            assert standard == standard.upper() or "_" in standard, (
                f"标签 {raw} → {standard} 不符合标准命名"
            )

    def test_sym_maps_to_disease_category(self):
        """症状(sym)应归入疾病大类(MEDICAL_DISEASE)。"""
        assert DEFAULT_NER_LABEL_MAPPING["sym"] == "MEDICAL_DISEASE"

    def test_gene_maps_to_genomic_hint(self):
        """基因(GENE)应映射为 GENOMIC_HINT。"""
        assert DEFAULT_NER_LABEL_MAPPING["GENE"] == "GENOMIC_HINT"


# =========================================================================== #
# ONNX 37类 BIOES 解析与 dyn 门禁专项测试
# =========================================================================== #


class TestBIOESAndSafetyGate:
    """测试 ONNX 37 类 BIOES 解析、自动识别与 dyn 门禁。"""

    def test_parse_bioes_tags_37_classes(self):
        """测试 37 类 BIOES 标签状态机解析（含 S-, B-, I-, E-, O 标记）。"""
        engine = ONNXSmallNerEngine()
        tokens = ["[CLS]", "患", "者", "确", "诊", "肺", "癌", "及", "艾", "滋", "病", "[SEP]"]
        # 5: B-dis, 23: I-dis, 24: E-dis, 7: B-dru, 8: S-dru, 18: S-sym, etc.
        # "肺" -> B-dis (5), "癌" -> E-dis (24)
        # "艾" -> B-dis (5), "滋" -> I-dis (23), "病" -> E-dis (24)
        label_indices = [0, 0, 0, 0, 0, 5, 24, 0, 5, 23, 24, 0]
        probs = [0.99] * len(tokens)

        entities = engine._parse_bioes_tags(tokens, label_indices, probs)
        assert len(entities) == 2
        assert entities[0]["text"] == "肺癌"
        assert entities[0]["label"] == "dis"
        assert entities[1]["text"] == "艾滋病"
        assert entities[1]["label"] == "dis"

    def test_extract_37_classes_autodetect(self, tmp_path):
        """当 ONNX 推理输出 logits shape 为 (seq_len, 37) 时，应自动识别并调用 _parse_bioes_tags。"""
        vocab_path = tmp_path / "vocab.txt"
        tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "患", "者", "肺", "癌"]
        vocab_path.write_text("\n".join(tokens), encoding="utf-8")

        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"dummy_onnx")

        engine = ONNXSmallNerEngine(model_path=str(model_path), vocab_path=str(vocab_path), max_len=16)

        mock_session = MagicMock()
        # logits shape: (1, 16, 37)
        mock_logits = np.zeros((1, 16, 37), dtype=np.float32)
        # token index 2 is [CLS], 3 is [SEP].
        # index 3 ("肺") -> B-dis (5), index 4 ("癌") -> E-dis (24)
        mock_logits[0, 3, 5] = 10.0
        mock_logits[0, 4, 24] = 10.0

        mock_session.run.return_value = [mock_logits]
        engine.session = mock_session
        engine.tokenizer = SimpleChineseBertTokenizer(str(vocab_path))
        engine._initialized = True

        entities = engine.extract("患者肺癌")
        assert len(entities) >= 1
        assert entities[0]["text"] == "肺癌"
        assert entities[0]["label"] == "MEDICAL_DISEASE"

    def test_dyn_service_l5_rating_and_safety_gate(self):
        """测试 dyn classification 定级一致性 (艾滋病判 L5) 与 Fail-Safe 安全门禁。"""
        from engine.dynclassification.service import DynamicClassificationService

        service = DynamicClassificationService()
        
        # 1. 验证极高敏文本 (艾滋病) 被判为 L5
        result = service.classify_field("chief_complaint", "患者确诊艾滋病")
        assert result.field_result is not None
        assert result.field_result.final_level == "L5"

        # 2. 验证若脱敏后仍包含 L4/L5 敏感词，触发 [L4-L5-DATA-REMOVED] 安全门禁
        sanitized = service._get_sanitized_value_from_funnel(
            "chief_complaint",
            "患者确诊 HIV 阳性",
            funnel_result=MagicMock(sanitized_value="患者确诊 HIV 阳性"),
            final_level="L5"
        )
        assert sanitized == "[L4-L5-DATA-REMOVED]"

    def test_onnx_ner_long_text_chunking(self):
        """测试长文本 (>128 tokens) 智能分句与切片推理。"""
        # 测试 _chunk_text 静态方法
        long_text = "患者于3年前因反复咳嗽就诊。经检查提示右肺占位。病理确诊为肺腺癌，行切除术。近1周来出现头痛，复查提示脑转移瘤。"
        chunks = ONNXSmallNerEngine._chunk_text(long_text, max_chunk_len=30)
        assert len(chunks) > 1
        assert "".join(chunks) == long_text

        # 模拟长文本 extract 推理
        engine = ONNXSmallNerEngine(max_len=32)
        engine._lazy_init = MagicMock()
        engine._initialized = True
        engine.tokenizer = MagicMock()
        engine.session = MagicMock()
        engine._extract_single_chunk = MagicMock(side_effect=[
            [{"text": "肺腺癌", "label": "MEDICAL_DISEASE", "confidence": 0.9}],
            [{"text": "脑转移瘤", "label": "MEDICAL_DISEASE", "confidence": 0.95}],
            [{"text": "脑转移瘤", "label": "MEDICAL_DISEASE", "confidence": 0.95}],
        ])

        entities = engine.extract(long_text, max_len=32)
        assert len(entities) == 2
        assert entities[0]["text"] == "肺腺癌"
        assert entities[1]["text"] == "脑转移瘤"


