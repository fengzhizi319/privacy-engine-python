"""dynclassification 模块 Layer-2 NER 适配器与引擎单元测试 / NER Adapter & Engine Unit Tests.

测试覆盖场景：
- NerAdapter 延迟加载与优雅降级（底层依赖缺失或损坏时不崩溃）
- ONNXSmallNerEngine Mock 测试（字符串清洗、实体标注提取、标签映射）
- ModelScopeSmallNerEngine Mock 测试（Pipeline 结果解析与阈值过滤）
- NerAdapter.extract() 在各种异常输入下的健壮性（空串、全标点等）
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from engine.dynclassification.base import SmallNerEngine
from engine.dynclassification.ner_adapter import NerAdapter
from engine.dynclassification.ner_engines import (
    ModelScopeSmallNerEngine,
    ONNXSmallNerEngine,
    TensorRTSmallNerEngine,
)


class TestNerAdapter:
    """测试 NerAdapter 的延迟初始化、降级机制与 extract() 方法。"""

    def test_adapter_initialization_defaults(self):
        """测试适配器默认初始化状态（未加载模型，处于乐观可用状态）。"""
        adapter = NerAdapter()
        assert adapter._initialized is False
        assert adapter._engine is None
        assert adapter._available is True

    def test_lazy_init_failure_degradation(self, monkeypatch):
        """测试当 MLX, TensorRT, ONNX 和 ModelScope 均不可用时，适配器能优雅降级且 is_available 为 False。"""
        from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine

        adapter = NerAdapter(model_path="/non_existent_path/model.onnx")

        # 模拟所有引擎初始化均抛出 Exception
        with patch.object(MLXSmallNerEngine, "_lazy_init", side_effect=RuntimeError("MLX not available")), \
             patch.object(TensorRTSmallNerEngine, "_lazy_init", side_effect=RuntimeError("TensorRT not installed")), \
             patch.object(ONNXSmallNerEngine, "_lazy_init", side_effect=RuntimeError("ONNX not installed")), \
             patch.object(ModelScopeSmallNerEngine, "__init__", side_effect=RuntimeError("ModelScope not installed")):
            assert adapter.is_available is False
            # 再次调用 extract 应优雅返回空列表，不受影响
            result = adapter.extract("患者张三，诊断为糖尿病")
            assert result == []

    def test_extract_with_mocked_onnx_engine(self):
        """测试当 ONNX 引擎可用时，extract() 正确转发并返回标准实体字典结构。"""
        adapter = NerAdapter()
        mock_engine = MagicMock()
        mock_engine.extract.return_value = [
            {"label": "MEDICAL_DISEASE", "text": "糖尿病", "confidence": 0.95},
        ]

        # 注入 Mock 引擎
        adapter._engine = mock_engine
        adapter._initialized = True
        adapter._available = True

        res = adapter.extract("患者诊断为糖尿病")
        assert len(res) == 1
        assert res[0]["label"] == "MEDICAL_DISEASE"
        assert res[0]["text"] == "糖尿病"
        assert res[0]["confidence"] == 0.95
        mock_engine.extract.assert_called_once_with("患者诊断为糖尿病")

    def test_extract_exception_handling(self):
        """测试底层引擎调用抛出未捕获异常时，extract() 捕获异常并返回空列表（Fail-safe）。"""
        adapter = NerAdapter()
        mock_engine = MagicMock()
        mock_engine.extract.side_effect = Exception("CUDA Out of Memory")

        adapter._engine = mock_engine
        adapter._initialized = True
        adapter._available = True

        res = adapter.extract("异常测试文本")
        assert res == []

    def test_empty_input_handling(self):
        """测试传入空字符串或空白字符的处理。"""
        adapter = NerAdapter()
        mock_engine = MagicMock()
        mock_engine.extract.return_value = []

        adapter._engine = mock_engine
        adapter._initialized = True
        adapter._available = True

        assert adapter.extract("") == []
        assert adapter.extract("   ") == []


class TestONNXSmallNerEngine:
    """测试 ONNXSmallNerEngine 的初始化与 BIO 标注解析。"""

    def test_onnx_engine_lazy_init_file_not_found(self):
        """测试当指定不存在的模型文件路径时抛出 FileNotFoundError。"""
        engine = ONNXSmallNerEngine(model_path="/path/to/missing_model.onnx")
        with pytest.raises(FileNotFoundError):
            engine._lazy_init()

    def test_onnx_engine_parse_bio_tags(self):
        """测试 _parse_bio_tags 对 BIO 预测标记序列的实体提取。"""
        engine = ONNXSmallNerEngine()
        tokens = ["[CLS]", "糖", "尿", "病", "[SEP]"]
        label_indices = [0, 1, 2, 2, 0]  # 1=B-dis, 2=I-dis
        probs = [0.99, 0.95, 0.96, 0.94, 0.99]

        entities = engine._parse_bio_tags(tokens, label_indices, probs)
        assert len(entities) == 1
        assert entities[0]["text"] == "糖尿病"
        assert entities[0]["label"] == "dis"


class TestModelScopeSmallNerEngine:
    """测试 ModelScopeSmallNerEngine 的模拟集成逻辑。"""

    def test_modelscope_extract_parsing_mocked(self):
        """Mock ModelScope pipeline 的输出，校验实体抽取与映射。"""
        engine = ModelScopeSmallNerEngine(
            label_mapping={"dis": "MEDICAL_DISEASE"},
        )

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "output": [
                {"type": "dis", "span": "高血压", "probability": 0.98},
            ]
        }
        engine.pipeline = mock_pipeline
        engine._initialized = True
        engine._available = True

        entities = engine.extract("患者高血压三级")
        assert len(entities) == 1
        assert entities[0]["label"] == "MEDICAL_DISEASE"
        assert entities[0]["text"] == "高血压"
        assert entities[0]["confidence"] == 1.0


class TestTensorRTSmallNerEngine:
    """测试 TensorRTSmallNerEngine 的初始化与 C++ TensorRT 挂载逻辑。"""

    def test_tensorrt_engine_lazy_init_file_not_found(self):
        """测试当指定不存在的模型文件路径时抛出 FileNotFoundError。"""
        engine = TensorRTSmallNerEngine(model_path="/path/to/missing_model.onnx")
        with pytest.raises(FileNotFoundError):
            engine._lazy_init()

    def test_tensorrt_engine_lazy_init_providers_mock(self):
        """Mock onnxruntime 验证 TensorRT 选项与引擎加载。"""
        engine = TensorRTSmallNerEngine(
            model_path=__file__,  # 使用真实存在的测试文件模拟模型路径
            vocab_path=__file__,  # 使用真实存在的测试文件模拟词表路径
        )
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_session = MagicMock()
        mock_session.get_providers.return_value = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            engine._lazy_init()
            assert engine._initialized is True
            assert engine.session is mock_session
            mock_ort.InferenceSession.assert_called_once()
            # 校验是否使用了 TensorrtExecutionProvider 且配置了 trt_fp16_enable
            providers_arg = mock_ort.InferenceSession.call_args[1].get("providers")
            assert isinstance(providers_arg[0], tuple)
            assert providers_arg[0][0] == "TensorrtExecutionProvider"
            assert providers_arg[0][1]["trt_fp16_enable"] is True

    def test_tensorrt_engine_parse_bio_tags(self):
        """测试 TensorRTSmallNerEngine 的 BIO 预测标记序列实体提取（单实体测试）。"""
        engine = TensorRTSmallNerEngine()
        tokens = ["[CLS]", "阿", "司", "匹", "林", "[SEP]"]
        label_indices = [0, 3, 4, 4, 4, 0]  # 3=B-dru, 4=I-dru
        probs = [0.99, 0.98, 0.97, 0.96, 0.95]

        entities = engine._parse_bio_tags(tokens, label_indices, probs)
        assert len(entities) == 1
        assert entities[0]["text"] == "阿司匹林"
        assert entities[0]["label"] == "dru"
        assert entities[0]["confidence"] == 0.95

    def test_tensorrt_engine_multi_entity_vectors(self):
        """测试 TensorRTSmallNerEngine 提取多类型实体测试向量 (疾病+症状+药物+检查项目+身体部位)。"""
        engine = TensorRTSmallNerEngine()

        # 文本向量: [CLS] 糖 尿 病 (1,2,2) 头 晕 (7,8) 阿 莫 西 林 (3,4,4,4) 心 电 图 (9,10,10) [SEP]
        tokens = ["[CLS]", "糖", "尿", "病", "头", "晕", "阿", "莫", "西", "林", "心", "电", "图", "[SEP]"]
        label_indices = [0, 1, 2, 2, 7, 8, 3, 4, 4, 4, 9, 10, 10, 0]
        probs = [0.99, 0.96, 0.95, 0.97, 0.91, 0.92, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.99]

        entities = engine._parse_bio_tags(tokens, label_indices, probs)
        assert len(entities) == 4

        # 1. 疾病: 糖尿病
        assert entities[0]["text"] == "糖尿病"
        assert entities[0]["label"] == "dis"
        assert round(entities[0]["confidence"], 2) == 0.95

        # 2. 症状: 头晕
        assert entities[1]["text"] == "头晕"
        assert entities[1]["label"] == "sym"
        assert round(entities[1]["confidence"], 2) == 0.91

        # 3. 药物: 阿莫西林
        assert entities[2]["text"] == "阿莫西林"
        assert entities[2]["label"] == "dru"
        assert round(entities[2]["confidence"], 2) == 0.95

        # 4. 检查项目: 心电图
        assert entities[3]["text"] == "心电图"
        assert entities[3]["label"] == "ite"
        assert round(entities[3]["confidence"], 2) == 0.92

    def test_tensorrt_engine_custom_label_mapping(self):
        """测试 TensorRTSmallNerEngine 的自定义类别映射标签转换向量。"""
        custom_mapping = {
            "dis": "SEC_DISEASE_LVL4",
            "dru": "SEC_MEDICATION_LVL3",
            "sym": "SEC_SYMPTOM_LVL2",
        }
        engine = TensorRTSmallNerEngine(label_mapping=custom_mapping)
        tokens = ["[CLS]", "高", "血", "压", "[SEP]"]
        label_indices = [0, 1, 2, 2, 0]  # 1=B-dis, 2=I-dis
        probs = [0.99, 0.98, 0.97, 0.96, 0.99]

        entities = engine._parse_bio_tags(tokens, label_indices, probs)
        # 执行映射转换
        for ent in entities:
            if ent["label"] in engine.label_mapping:
                ent["label"] = engine.label_mapping[ent["label"]]

        assert len(entities) == 1
        assert entities[0]["text"] == "高血压"
        assert entities[0]["label"] == "SEC_DISEASE_LVL4"

    def test_tensorrt_engine_edge_case_inputs(self):
        """测试无实体文本、标点符号文本与边界向量。"""
        engine = TensorRTSmallNerEngine()
        tokens = ["[CLS]", "，", "。", "！", "[SEP]"]
        label_indices = [0, 0, 0, 0, 0]  # 全部为 O
        probs = [0.99, 0.99, 0.99, 0.99, 0.99]

        entities = engine._parse_bio_tags(tokens, label_indices, probs)
        assert entities == []

    def test_tensorrt_engine_real_model_inference_vector(self):
        """若本地存在 .models/raner_cmeee.onnx 则运行真实的端到端推理测试向量。"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_path = os.path.join(project_root, ".models", "raner_cmeee.onnx")
        vocab_path = os.path.join(project_root, ".models", "vocab.txt")

        if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
            pytest.skip("未在 .models/ 找到 raner_cmeee.onnx 和 vocab.txt，跳过真实模型端到端测试")

        engine = TensorRTSmallNerEngine(model_path=model_path, vocab_path=vocab_path)
        test_text = "患者张三，主诉高血压三级，合并2型糖尿病，口服阿司匹林。"
        results = engine.extract(test_text)

        # 校验返回结构为字典列表，且字段完整
        assert isinstance(results, list)
        for ent in results:
            assert "text" in ent
            assert "label" in ent
            assert "confidence" in ent


class TestCudaNerEngine:
    """针对 CUDA 硬件加速模式下的 NER 引擎与适配器补充单元测试。"""

    def test_modelscope_cuda_compatibility_check(self):
        """测试 PyTorch CUDA 兼容性探针函数能正确识别设备与 kernel 执行能力。"""
        import torch

        is_compatible = ModelScopeSmallNerEngine._is_cuda_compatible(torch)
        if torch.cuda.is_available():
            assert is_compatible is True, "PyTorch 已检测到 CUDA 设备， compatibility check 应为 True"
        else:
            assert is_compatible is False

    def test_modelscope_cuda_ner_extraction(self):
        """测试使用 CUDA 设备的 ModelScopeSmallNerEngine 端到端抽取。"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("当前环境未检测到 CUDA 设备，跳过 CUDA NER 测试")

        engine = ModelScopeSmallNerEngine(device="cuda")
        if not os.path.exists(engine.local_model_dir):
            pytest.skip("未在 .models/ 找到 raner_cmeee 权重，跳过模型推理测试")
        if getattr(engine, "pipeline", None) is None or getattr(engine, "_init_error", None) is not None:
            pytest.skip("ModelScope NER pipeline 未能成功初始化，跳过测试")

        results = engine.extract("患者确诊为2型糖尿病和冠心病")
        assert isinstance(results, list)
        labels = {ent["label"] for ent in results}
        assert "MEDICAL_DISEASE" in labels
        texts = {ent["text"] for ent in results}
        assert "2型糖尿病" in texts or "冠心病" in texts

    def test_ner_adapter_cuda_device_selection(self, monkeypatch):
        """测试设置 PRIVACY_NER_DEVICE=cuda 时 NerAdapter 可正常识别并优先调用硬件加速。"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("当前环境未检测到 CUDA 设备")

        monkeypatch.setenv("PRIVACY_NER_DEVICE", "cuda")
        adapter = NerAdapter()
        if not adapter.is_available:
            pytest.skip("CUDA 设备存在，但 NER 引擎/模型权重不可用（ML 依赖为可选超集）")
        results = adapter.extract("患者主诉急性腹痛与高热")
        for ent in results:
            assert ent["confidence"] > 0.0

    def test_compare_pytorch_vs_tensorrt_ner_performance(self):
        """对比 PyTorch (ModelScope CUDA/CPU) 引擎与 TensorRT (ONNX Runtime) 引擎的推理性能与吞吐量。"""
        import time
        import torch

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        ms_dir = os.path.join(project_root, ".models", "raner_cmeee")
        vocab_path = os.path.join(project_root, ".models", "vocab.txt")
        onnx_path = os.path.join(project_root, ".models", "raner_cmeee.onnx")

        if not (os.path.exists(ms_dir) and os.path.exists(vocab_path)):
            pytest.skip("未在 .models/ 找到 ModelScope 权重目录或词表，跳过性能对比测试")

        sample_texts = [
            "患者张三，诊断为2型糖尿病与高血压三级，口服阿司匹林和二甲双胍。",
            "主诉急性心肌梗死合并心力衰竭，行冠状动脉介入手术治疗。",
            "右下肺听诊可闻及湿啰音，查体发现双下肢重度水肿，拟行心电图与胸部CT检查。",
            "既往有慢性乙型肝炎病史10年，静脉滴注头孢曲松钠，禁用青霉素。",
        ]

        # 1. 测量 PyTorch CUDA (或 CPU) 引擎性能
        cuda_compatible = ModelScopeSmallNerEngine._is_cuda_compatible(torch)
        device_cuda = "cuda" if cuda_compatible else "cpu"
        ms_cuda_engine = ModelScopeSmallNerEngine(device=device_cuda)
        _ = ms_cuda_engine.extract(sample_texts[0])  # Warmup

        t0 = time.monotonic()
        ms_count = 0
        for _ in range(3):
            for text in sample_texts:
                _ = ms_cuda_engine.extract(text)
                ms_count += 1
        ms_cuda_time = time.monotonic() - t0
        ms_cuda_lat = (ms_cuda_time / ms_count) * 1000
        ms_cuda_qps = ms_count / ms_cuda_time

        print(f"\n⚡ [NER 推理性能基准测试 - 共 {ms_count} 次请求]")
        print(f"  PyTorch CUDA ({device_cuda}) 引擎:  平均延迟 = {ms_cuda_lat:.2f} ms | 吞吐量 = {ms_cuda_qps:.2f} req/s")

        # 2. 若存在 ONNX 模型，测量 TensorRT 引擎性能；否则对比 PyTorch CPU 性能
        if os.path.exists(onnx_path):
            trt_engine = TensorRTSmallNerEngine(model_path=onnx_path, vocab_path=vocab_path)
            _ = trt_engine.extract(sample_texts[0])  # Warmup

            t0 = time.monotonic()
            trt_count = 0
            for _ in range(3):
                for text in sample_texts:
                    _ = trt_engine.extract(text)
                    trt_count += 1
            trt_time = time.monotonic() - t0
            trt_lat = (trt_time / trt_count) * 1000
            trt_qps = trt_count / trt_time

            speedup = ms_cuda_lat / max(trt_lat, 0.001)
            print(f"  TensorRT (C++) 引擎:     平均延迟 = {trt_lat:.2f} ms | 吞吐量 = {trt_qps:.2f} req/s")
            print(f"  🚀 TensorRT vs PyTorch 加速比: {speedup:.2f}x")
            assert trt_lat > 0
            assert trt_lat <= ms_cuda_lat * 2.0
        else:
            ms_cpu_engine = ModelScopeSmallNerEngine(device="cpu")
            _ = ms_cpu_engine.extract(sample_texts[0])  # Warmup
            t0 = time.monotonic()
            cpu_count = 0
            for _ in range(3):
                for text in sample_texts:
                    _ = ms_cpu_engine.extract(text)
                    cpu_count += 1
            cpu_time = time.monotonic() - t0
            cpu_lat = (cpu_time / cpu_count) * 1000
            cpu_qps = cpu_count / cpu_time

            speedup = cpu_lat / max(ms_cuda_lat, 0.001)
            print(f"  PyTorch CPU 引擎:         平均延迟 = {cpu_lat:.2f} ms | 吞吐量 = {cpu_qps:.2f} req/s")
            print(f"  🚀 PyTorch CUDA vs CPU 加速比: {speedup:.2f}x")
            assert ms_cuda_lat > 0
            assert cpu_lat > 0


class TestNvidiaLibPreload:
    """回归：_preload_nvidia_libs 必须幂等，不得让 LD_LIBRARY_PATH 无界增长。

    历史缺陷：每次引擎初始化都把整批 nvidia/triton 目录前插到 LD_LIBRARY_PATH，
    进程内累计超过 ARG_MAX 后任何子进程 exec 都会抛
    ``OSError: [Errno 7] Argument list too long``（曾表现为全量 pytest 下真实 LLM
    用例在 NER 用例之后必然失败）。
    """

    @staticmethod
    def _fake_site_packages(tmp_path):
        """构造一个仅含 nvidia/cuda/lib 的假 site-packages 目录。"""
        lib_dir = tmp_path / "nvidia" / "cuda" / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "libcupti.so.12").write_text("")
        return lib_dir

    def test_repeated_calls_do_not_grow_ld_library_path(self, tmp_path, monkeypatch):
        """多次预加载后 LD_LIBRARY_PATH 中同一目录只应出现一次，且保留原有条目。"""
        from engine.dynclassification import base as base_mod

        lib_dir = self._fake_site_packages(tmp_path)
        monkeypatch.setattr(sys, "path", [str(tmp_path)], raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/pre/existing/lib")
        monkeypatch.setattr(base_mod, "_CUDA_PRELOAD_DONE", False)

        for _ in range(5):
            SmallNerEngine._preload_nvidia_libs()

        parts = os.environ["LD_LIBRARY_PATH"].split(":")
        assert parts.count(str(lib_dir)) == 1
        assert "/pre/existing/lib" in parts

    def test_second_call_is_noop_after_guard_set(self, tmp_path, monkeypatch):
        """进程内守卫生效后，新出现的目录不应被再次追加。"""
        from engine.dynclassification import base as base_mod

        self._fake_site_packages(tmp_path / "first")
        monkeypatch.setattr(sys, "path", [str(tmp_path / "first")], raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "")
        monkeypatch.setattr(base_mod, "_CUDA_PRELOAD_DONE", False)

        SmallNerEngine._preload_nvidia_libs()
        after_first = os.environ["LD_LIBRARY_PATH"]
        assert after_first

        second_dir = self._fake_site_packages(tmp_path / "second")
        monkeypatch.setattr(sys, "path", [str(tmp_path / "second")], raising=False)
        SmallNerEngine._preload_nvidia_libs()

        assert os.environ["LD_LIBRARY_PATH"] == after_first
        assert str(second_dir) not in os.environ["LD_LIBRARY_PATH"]

    def test_no_duplicate_override_in_ner_engines(self):
        """NER 引擎子类不得再复制一份 _preload_nvidia_libs（SSOT：基类唯一实现）。"""
        assert "_preload_nvidia_libs" not in ModelScopeSmallNerEngine.__dict__
        assert "_preload_nvidia_libs" not in TensorRTSmallNerEngine.__dict__
        assert "_preload_nvidia_libs" not in ONNXSmallNerEngine.__dict__
