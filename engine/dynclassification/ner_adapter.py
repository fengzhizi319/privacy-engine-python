"""NER 引擎适配器 / NER Engine Adapter.

为 dynclassification 三层漏斗提供 Layer-2 NER 实体识别能力。
采用 lazy-load 策略：仅在首次调用时加载底层 NER 引擎（ONNX/ModelScope），
避免核心路径引入重量级 ML 依赖。
Provides Layer-2 NER entity recognition capabilities for the dynclassification three-layer funnel.
Adopts a lazy-load strategy: loads the underlying NER engine (ONNX/ModelScope) only on the first call,
avoiding heavy ML dependencies in the core path.

执行逻辑 / Execution logic:
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  NerAdapter.extract(text)                                                                            │
│    │                                                                                                 │
│    ├─ 首次调用 / First call → _lazy_init()                                                             │
│    │   ├─ 尝试加载 / Try loading ONNXSmallNerEngine (轻量, 推荐 / lightweight, recommended)              │
│    │   ├─ 失败 / Failed → 尝试 / Try ModelScopeSmallNerEngine                                          │
│    │   └─ 均失败 / All failed → 标记不可用, 后续直接返回 [] / Mark unavailable, subsequently return []      │
│    │                                                                                                  │
│    └─ 调用底层引擎 / Call underlying engine extract(text) → 实体列表 / Entity list                        │
│       [{"label": "MEDICAL_DISEASE", "text": "...", "confidence": 0.9}]                                │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘

降级策略 / Degradation strategy:
- onnxruntime 未安装 / uninstalled → 尝试 / Try ModelScope
- modelscope 未安装 / uninstalled → 标记不可用, 返回空列表 / Mark unavailable, return empty list
- 模型文件不存在 / Model file does not exist → 标记不可用, 返回空列表 / Mark unavailable, return empty list
- 任何异常 / Any exception → 返回空列表 / Return empty list (fail-safe)
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from ..observability.logging_config import get_logger

logger = get_logger(__name__)


class NerAdapter:
    """NER 引擎适配器（Layer-2） / NER Engine Adapter (Layer-2).

    封装旧模块 privacy/classification/classification_ner.py 的 NER 引擎，
    提供统一的 extract() 接口供 ClassificationFunnel 调用。
    Encapsulates the NER engine from the old module privacy/classification/classification_ner.py,
    providing a unified extract() interface for ClassificationFunnel to call.

    Attributes:
        _engine: 底层 NER 引擎实例（延迟初始化） / Underlying NER engine instance (lazy initialized).
        _available: 引擎是否可用（初始化失败后标记为 False） / Whether the engine is available (marked as False after initialization failure).
        _initialized: 是否已尝试过初始化 / Whether initialization has been attempted.
    """

    def __init__(
        self,
        model_path: str | None = None,
        vocab_path: str | None = None,
        label_mapping: dict[str, str] | None = None,
        device: str | None = None,
    ):
        """初始化适配器（不加载模型） / Initialize adapter (without loading model).

        Args:
            model_path: ONNX 模型文件路径（可选，默认自动检测） / ONNX model file path (optional, auto-detect by default).
            vocab_path: 词表文件路径（可选，默认自动检测） / Vocab file path (optional, auto-detect by default).
            label_mapping: 原始标签→标准标签映射（可选，默认使用内置医疗映射） / Raw label to standard label mapping (optional, uses built-in medical mapping by default).
            device: 目标计算设备（"cuda" / "cpu" / None）。
        """
        self._model_path = model_path
        self._vocab_path = vocab_path
        self._label_mapping = label_mapping
        self._device = device
        self._engine: Any = None
        self._available = True  # 乐观假设可用，初始化失败后改为 False
        self._initialized = False
        self._init_lock = threading.Lock()

    def _lazy_init(self) -> None:
        """延迟初始化 NER 引擎 / Lazy initialize NER engine.

        尝试顺序 / Attempt order:
        0. MLXSmallNerEngine (Apple Silicon Metal GPU, macOS 优先 / macOS Metal GPU preferred)
        1. TensorRTSmallNerEngine (NVIDIA GPU 硬件加速 / NVIDIA GPU hardware acceleration)
        2. ONNXSmallNerEngine (轻量, 无需 PyTorch / Lightweight, no PyTorch required)
        3. ModelScopeSmallNerEngine (需 PyTorch + modelscope / Requires PyTorch + modelscope)
        4. 均失败 → 标记不可用 / All failed → Mark as unavailable

        线程安全：使用 Lock + double-check 防止并发请求重复初始化。
        """
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            # 尝试 0: MLX 引擎（Apple Silicon Metal GPU，仅 macOS）
            if sys.platform == "darwin":
                try:
                    from .mlx_ner_engine import MLXSmallNerEngine
                    engine = MLXSmallNerEngine(
                        label_mapping=self._label_mapping,
                    )
                    engine._lazy_init()
                    self._engine = engine
                    logger.info("ner_adapter_initialized", extra={"backend": "mlx_metal"})
                    self._initialized = True
                    return
                except Exception as e:
                    logger.debug("ner_mlx_unavailable", extra={"error": str(e)})

            # 尝试 1: TensorRT 引擎（纯 C++ 硬件加速，FP16 模式）
            try:
                from .ner_engines import TensorRTSmallNerEngine
                engine = TensorRTSmallNerEngine(
                    model_path=self._model_path,
                    vocab_path=self._vocab_path,
                    label_mapping=self._label_mapping,
                    device=self._device,
                )
                engine._lazy_init()
                self._engine = engine
                logger.info("ner_adapter_initialized", extra={"backend": "tensorrt"})
                self._initialized = True
                return
            except Exception as e:
                logger.debug("ner_tensorrt_unavailable", extra={"error": str(e)})

            # 尝试 2: ONNX Runtime 引擎 (CUDA / CPU)
            try:
                from .ner_engines import ONNXSmallNerEngine
                engine = ONNXSmallNerEngine(
                    model_path=self._model_path,
                    vocab_path=self._vocab_path,
                    label_mapping=self._label_mapping,
                    device=self._device,
                )
                # 触发模型加载验证（如果文件不存在会抛出异常）
                engine._lazy_init()
                self._engine = engine
                logger.info("ner_adapter_initialized", extra={"backend": "onnx"})
                self._initialized = True
                return
            except Exception as e:
                logger.debug("ner_onnx_unavailable", extra={"error": str(e)})

            # 尝试 3: ModelScope 引擎（需 PyTorch + modelscope）
            try:
                from .ner_engines import ModelScopeSmallNerEngine
                engine = ModelScopeSmallNerEngine(
                    label_mapping=self._label_mapping,
                    device=self._device,
                )
                # 触发模型加载验证（与其他引擎保持一致）
                engine._lazy_init()
                self._engine = engine
                logger.info("ner_adapter_initialized", extra={"backend": "modelscope"})
                self._initialized = True
                return
            except Exception as e:
                logger.debug("ner_modelscope_unavailable", extra={"error": str(e)})

            # 均不可用
            self._available = False
            self._initialized = True
            logger.info("ner_adapter_unavailable", extra={"reason": "no backend available"})

    @property
    def is_available(self) -> bool:
        """NER 引擎是否可用 / Whether NER engine is available."""
        self._lazy_init()
        return self._available

    def extract(self, text: str) -> list[dict[str, Any]]:
        """从文本中提取命名实体 / Extract named entities from text.

        Args:
            text: 待分析的文本内容 / Text content to analyze.

        Returns:
            实体字典列表，每个字典包含 / List of entity dictionaries, each containing:
            - label: 实体标签 / Entity label (e.g. "MEDICAL_DISEASE", "MEDICATION")
            - text: 实体文本 / Entity text
            - confidence: 识别置信度 / Recognition confidence

            引擎不可用或异常时返回空列表 / Returns empty list when engine is unavailable or on exception.
        """
        self._lazy_init()
        if not self._available or self._engine is None:
            return []

        try:
            return self._engine.extract(text)
        except Exception as e:
            logger.warning("ner_extract_failed", extra={"error": str(e)})
            return []
