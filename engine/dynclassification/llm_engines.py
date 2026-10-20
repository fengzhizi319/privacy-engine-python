"""基于本地纯文本大模型 Qwen3.5-0.8B-Privacy-Classifier-Smoother 的数据分类分级器。

中文说明：
支持纯文本数据的零样本敏感定级推理（微调后的医疗隐私分类专用模型）。
具备延迟加载、自动降级等企业级能力。
注意：该模型为纯文本 CausalLM，不支持图像/视觉输入。

架构设计：
- Qwen3Classifier：纯文本分类器主类，继承 LlmClassifier 抽象基类
- OpenAILlmClassifier：通过 HTTP API 调用 vLLM/Ollama/OpenAI 兼容服务
- 延迟加载：首次调用 classify() 时才加载模型权重（避免启动阻塞和显存浪费）
- 双重检查锁定：线程安全的模型初始化（gRPC 多线程环境）
- 专用推理线程池：隔离推理与 gRPC 工作线程，配合超时机制防止永久阻塞
- JSON 结果解析：正则提取 + 容错降级

降级策略：
- torch/transformers 未安装 → 初始化失败 → classify() 返回 None → 上层降级
- 模型目录不存在 → FileNotFoundError → 降级
- 推理超时（默认 180s）→ 放弃本次推理 → 返回 None → 降级
- JSON 解析失败 → 返回 None → 降级

English Description:
Data classification and grading engine based on local text-only LLM
Qwen3.5-0.8B-Privacy-Classifier-Smoother (fine-tuned medical privacy classifier).
Supports zero-shot sensitivity grading for plain text data. Features lazy-loading
and graceful degradation. Note: text-only model, no image/vision input support.

===================================================================================
              LLM 分类推理流程 / LLM Classification Inference Flow
===================================================================================

  Qwen3Classifier.classify(text, upstream_level, upstream_confidence)
    │
    ├─① 安全检查
    │   ├─ 图片输入检测 → 返回 None (纯文本模型不支持)
    │   └─ _lazy_init() (双重检查锁定)
    │       ├─ 检查已初始化 → 快速返回
    │       ├─ 检查初始化失败缓存 → 抛出缓存异常
    │       └─ 加锁 → 导入 torch/transformers → 加载模型 → 检测设备
    │
    ├─② 构建 Prompt
    │   ├─ system_prompt: 医疗数据分类分级规则 (L1~L5)
    │   └─ user_prompt: wrap_untrusted_text(text) (Prompt 注入防护)
    │
    ├─③ 推理执行 (专用线程池 + 超时保护)
    │   └─ _executor.submit(_classify_inner) → future.result(timeout=180s)
    │       ├─ tokenizer.apply_chat_template(messages) → prompt_ids
    │       ├─ model.generate(prompt_ids, max_new_tokens=512) → output_ids
    │       └─ tokenizer.decode(output_ids) → output_text
    │
    ├─④ 解析 JSON 结果
    │   └─ re.search(r'{.*}', output_text) → json.loads()
    │       → {"final_level": "L3", "confidence": 0.9, "reasoning": "..."}
    │
    └─⑤ 返回 dict | None (None = 降级，上层使用 Phase 1 置信度衰减)

  OpenAILlmClassifier.classify(text, ...)
    └─ HTTP POST → api_base/chat/completions → JSON 响应 → 解析结果
===================================================================================
"""

# 启用延迟注解求值，允许在类型提示中引用尚未定义的类名
from __future__ import annotations

# 导入 JSON 解析模块，用于解析大模型返回的 JSON 结构化结果
import json
# 导入操作系统接口，用于文件路径拼接、目录存在性检查、环境变量读取
import os
# 导入正则表达式模块，用于 JSON 提取
import re
# 导入路径库，用于类型提示中引用 Path 类型
from pathlib import Path
# 导入线程模块，用于创建互斥锁保护模型初始化和推理的线程安全
import threading
# 导入时间模块，用于测量推理耗时（monotonic 单调时钟）
import time
# 导入线程池执行器，用于将模型推理隔离到专用线程（配合超时机制）
from concurrent.futures import ThreadPoolExecutor
# 导入线程池超时异常类型，用于捕获推理超时事件
from concurrent.futures import TimeoutError as FuturesTimeoutError
# 导入类型注解工具：Any 通用类型，cast 类型断言
from typing import Any, cast

# 导入结构化日志工厂函数（支持 JSON 格式日志输出）
from ..observability.logging_config import get_logger
# 导入 Prometheus 指标：
# - CLASSIFICATION_LLM_DURATION：LLM 推理延迟直方图（按引擎标签）
# - CLASSIFICATION_LLM_TOTAL：LLM 调用次数计数器（按状态标签：success/error/timeout/init_failed）
from ..observability.metrics import (
    CLASSIFICATION_LLM_DURATION,
    CLASSIFICATION_LLM_TOKENS_TOTAL,
    CLASSIFICATION_LLM_TOTAL,
)
from ..env_loader import load_env_file
# 导入 LLM 分类器抽象基类和敏感度等级枚举
from .base import LlmClassifier, SensitivityLevel
# 导入日志脱敏工具函数（对敏感路径/值进行掩码处理后再记录日志）
# 以及不可信文本 prompt 中和工具（Prompt 注入防护）
from .utils import sanitize_for_prompt, wrap_untrusted_text

# 创建模块级结构化日志器，用于记录 LLM 分类器相关事件
logger = get_logger(__name__)

# 默认模型目录名 / Default model directory name
_DEFAULT_MODEL_DIR = "Qwen3.5-0.8B-Privacy-Classifier-Smoother"

# 微调模型训练侧 system prompt（与 llmlora/src/dataset/loader.py SYSTEM_PROMPT 保持一致）。
# 注意：不要在末尾追加 JSON 格式规范等训练时未出现的内容——
# 0.8B 微调模型对 prompt 分布漂移极其敏感，任何训练分布外的附加文本
# 都会导致生成漂移（提前 EOS / JSON 截断）。输出 JSON schema 已在训练样本中内化。
_FINETUNED_SYSTEM_PROMPT = (
    "你是一个专业的隐私安全Sidecar助手。请分析输入的文本，识别敏感信息，"
    "输出分类分级结果（JSON格式），并提供语义连贯的无痕抹平脱敏重写文本。\n\n"
    "【数据分类分级标准指南】\n"
    "- L1 (公开数据): 无敏感信息的公开资讯、通用日常文本。\n"
    "- L2 (内部数据): 业务统计指标、系统日志、设备运维等低敏感内部数据。\n"
    "- L3 (敏感数据/个人基本信息): 姓名、身份证号、手机号、银行卡号、电子邮箱等个人基础标识与资产信息。\n"
    "- L4 (高敏感数据/诊疗与金融敏感): 疾病诊断（如重度抑郁症、高血压、冠心病）、病历主诉、处方药品等医疗健康敏感信息。\n"
    "- L5 (极敏感数据): 基因组、生物特征、特级商业机密等核心数据。"
)


class Qwen3Classifier(LlmClassifier):
    """基于本地部署 Qwen3.5-0.8B-Privacy-Classifier-Smoother 的纯文本分类器。

    中文说明：
    支持对纯文本数据进行敏感等级评估。
    本类是三层分类漏斗的第三层（Layer-3），在规则引擎（Layer-1）和 NER（Layer-2）
    之后执行，作为最终的兜底分类手段。
    注意：该模型为纯文本 CausalLM，不支持图像输入。

    线程安全设计：
    - _lock：互斥锁，保护模型初始化（双重检查锁定）和推理过程（串行化）
    - _executor：单线程池，将推理隔离到独立线程，配合超时机制

    English Description:
    Text-only classifier based on the fine-tuned Qwen3.5-0.8B medical privacy model.
    Supports sensitivity level assessment for plain text inputs only (no image/vision).
    """

    # 推理超时（秒）：0.8B 纯文本模型在 CPU 上推理通常较快，
    # 超时后放弃本次推理并返回 None 触发降级，避免无限阻塞 gRPC 工作线程。
    # 可通过环境变量 PRIVACY_VLM_TIMEOUT 覆盖，默认 180 秒。
    _INFERENCE_TIMEOUT = int(os.environ.get("PRIVACY_VLM_TIMEOUT", "180"))

    def __init__(
        self,
        model_path: str | None = None,
        classify_prompt_template: str | None = None,
        device: str | None = None,
    ):
        """初始化分类器 / Initialize Classifier.

        仅设置路径和状态标志，不实际加载模型（延迟加载策略）。
        模型权重加载推迟到首次 classify() 调用时的 _lazy_init() 中执行。

        Args:
            model_path: 模型本地路径 / Local model path.
                如果不指定，默认使用项目根目录下的 .models/Qwen3.5-0.8B-Privacy-Classifier-Smoother。
                (Defaults to .models/Qwen3.5-0.8B-Privacy-Classifier-Smoother under project root)
            classify_prompt_template: 自定义分类 system prompt 模板 / Custom classification prompt template.
                支持占位符: {domain}, {standard_id}, {levels_desc}。
                None 时使用内置医疗领域默认 prompt。
            device: 目标计算设备（"cuda" / "cpu" / "mps" / None）。
        """
        # 如果未指定模型路径，自动计算默认路径 (.models/Qwen3.5-0.8B-Privacy-Classifier-Smoother)
        if not model_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            model_path = os.path.join(project_root, ".models", _DEFAULT_MODEL_DIR)

        # 保存模型路径供后续 _lazy_init 使用
        self.model_path = model_path
        # 保存自定义分类 prompt 模板（None 时使用内置默认）
        self._classify_prompt_template = classify_prompt_template
        # 显式指定的计算设备
        self.device = device
        # 模型实例占位（延迟初始化后赋值）
        self._model: Any = None
        # Tokenizer 实例占位（用于构建模型输入张量）
        self._tokenizer: Any = None
        # 初始化完成标志（False 表示尚未加载模型）
        self._initialized = False
        # 初始化错误缓存（记录首次失败原因，后续直接抛出不重试）
        self._init_error: Exception | None = None
        # 线程锁：gRPC 使用线程池处理请求，多个工作线程可能并发调用
        # _lazy_init / classify，需要互斥保护以防止：
        #   1. 多线程同时初始化模型导致重复加载或竞态
        #   2. 多线程同时推理导致显存/内存争用引发 OOM 崩溃
        self._lock = threading.Lock()
        # 专用推理线程池：将模型推理隔离到单独线程，配合超时机制，
        # 即使推理卡死也不会永久阻塞 gRPC 工作线程。
        # max_workers=1 确保同一时刻只有一个推理任务在执行（串行化）。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-infer")
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.cumulative_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _lazy_init(self):
        """延迟初始化模型 / Lazy-Initialize Model.

        中文说明：避免导入时或非 LLM 运行时占用显存或因缺少依赖报错。
        使用双重检查锁定（double-checked locking）确保线程安全：
        仅首次调用时加锁初始化，后续调用直接返回，避免锁竞争开销。

        双重检查锁定流程：
        1. 第一次检查（无锁）：快速路径，已初始化则直接返回
        2. 获取锁
        3. 第二次检查（有锁）：防止等锁期间另一线程已完成初始化
        4. 执行实际初始化逻辑

        English Description: Avoids occupying GPU memory at import time or when LLM
        is not needed, and prevents errors from missing dependencies.
        Uses double-checked locking for thread safety.

        Raises:
            FileNotFoundError: 本地模型目录不存在 / Local model directory not found.
        """
        # === 第一次检查（无锁快速路径）===
        # 已初始化则直接返回，避免不必要的锁竞争
        if self._initialized:
            return
        # 之前初始化失败过，直接抛出缓存的错误（不重复尝试加载）
        if self._init_error:
            raise self._init_error

        # === 获取互斥锁 ===
        with self._lock:
            # === 第二次检查（有锁）===
            # 另一个线程可能已在等锁期间完成初始化
            if self._initialized:
                return
            if self._init_error:
                raise self._init_error

            try:
                # 延迟导入 PyTorch（避免模块顶层导入导致的启动延迟和依赖问题）
                import torch
                # 延迟导入 transformers 库中的 CausalLM 模型类和 tokenizer
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # 验证模型目录是否存在（用户需先下载微调模型）
                if not os.path.exists(self.model_path) or not os.path.isdir(self.model_path):
                    raise FileNotFoundError(
                        f"本地模型未找到，请先下载微调模型至: {self.model_path}"
                    )

                # 检测计算设备，优先级：显式/环境变量指定 > CUDA GPU > macOS MPS > CPU
                device = self._select_device(torch, custom_device=self.device)

                # 记录模型加载开始的结构化日志
                logger.info(
                    "qwen3_model_loading",
                    extra={"model_path": self.model_path, "device": device},
                )

                # 选择模型精度：优先使用模型 config 声明的 dtype（如 Qwen3.5 为 bfloat16）。
                # 注意：Qwen3.5 的 linear-attention/mamba 混合层在 FP16 下会数值溢出，
                # 导致生成内容损坏截断（JSON 解析失败），CUDA 下禁止无脑降级为 FP16。
                is_cuda = device.startswith("cuda")
                dtype = self._resolve_cuda_dtype(torch) if is_cuda else torch.float32

                # transformers 5.x 将 torch_dtype 弃用改用 dtype；旧版本仅接受 torch_dtype。
                # 按版本选择参数名，避免告警与兼容性问题。
                import transformers as _transformers

                _use_new_kw = _transformers.__version__.split(".")[0].isdigit() and int(
                    _transformers.__version__.split(".")[0]
                ) >= 5
                dtype_kw = {"dtype": dtype} if _use_new_kw else {"torch_dtype": dtype}

                # 从本地目录加载预训练 CausalLM 模型权重
                if is_cuda:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="auto" if device == "cuda" else device,
                        trust_remote_code=True,
                        **dtype_kw,
                    )
                elif device == "mps":
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="mps",
                        trust_remote_code=True,
                        **dtype_kw,
                    )
                else:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        **dtype_kw,
                    )
                    self._model = self._model.to("cpu")

                # 加载模型对应的 tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
                # 标记初始化完成
                self._initialized = True
                # 记录模型初始化成功的结构化日志
                logger.info(
                    "qwen3_model_initialized",
                    extra={"model_path": self.model_path, "device": device, "engine": "qwen3"},
                )

            except Exception as e:
                # 初始化失败：缓存错误对象，后续调用直接抛出（不重复尝试）
                self._init_error = e
                # 记录初始化失败的警告日志
                logger.warning(
                    "qwen3_model_init_failed",
                    extra={"error": str(e), "model_path": self.model_path},
                )
                # 重新抛出异常，让调用方（classify）捕获并触发降级
                raise e

    @property
    def _processor(self) -> Any:
        return self._tokenizer

    @_processor.setter
    def _processor(self, value: Any) -> None:
        self._tokenizer = value

    @property
    def is_ready(self) -> bool:
        """模型是否已完成初始化且未发生错误 / Whether Model Is Ready.

        用于健康检查接口 /readyz/llm 判断 LLM 层是否可用。

        Returns:
            模型就绪状态 / Model readiness status.
        """
        # 两个条件同时满足才视为就绪：已初始化 且 无错误
        return self._initialized and self._init_error is None

    def warmup(self) -> bool:
        """主动触发模型加载 / Proactively Trigger Model Loading.

        中文说明：同步阻塞调用，建议在后台线程/协程中调用。
        服务启动时可通过 PRIVACY_WARMUP_LLM=true 环境变量触发异步预热，
        避免首次请求时因模型加载导致的高延迟。

        English Description: Synchronous blocking call; recommended to invoke in a
        background thread or coroutine.

        Returns:
            是否成功完成初始化 / Whether initialization succeeded.
        """
        try:
            # 触发延迟初始化（加载模型权重）
            self._lazy_init()
            return True  # 初始化成功
        except Exception:
            return False  # 初始化失败（依赖缺失/模型不存在等）

    def _is_finetuned_model(self) -> bool:
        """判断当前加载的是否为项目微调的 Privacy-Classifier-Smoother 模型。

        微调模型的推理 prompt 必须与训练侧（llmlora/src/dataset/loader.py）严格一致：
        短 system prompt + 裸用户文本。训练分布外的附加文本（JSON schema 说明、
        "请评估以下文本…"前导语、««« 分隔符包裹）会导致 0.8B 小模型生成漂移、
        提前 EOS 造成 JSON 截断（llm_json_parse_failed）。

        Returns:
            模型目录名匹配默认微调模型名且未配置自定义模板时返回 True。
        """
        base = os.path.basename(os.path.normpath(self.model_path))
        return base == _DEFAULT_MODEL_DIR and self._classify_prompt_template is None

    def _resolve_cuda_dtype(self, torch: Any) -> Any:
        """选择 CUDA 推理精度：优先模型 config 声明的 dtype，其次 bf16，最后 fp16。

        Qwen3.5 等含 linear-attention/mamba 层的混合架构在 FP16 下会数值溢出，
        导致生成内容损坏截断；因此当模型 config 声明 bfloat16 且设备支持 bf16 时，
        必须使用 bf16；仅在不支持 bf16 的旧 GPU 上才回退 FP16。

        Returns:
            torch dtype（bfloat16 / float16）。
        """
        bf16_supported = False
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False

        # 读取模型 config 声明的训练精度（如 Qwen3.5 的 text_config.dtype = bfloat16）
        cfg_dtype_name: str | None = None
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            text_cfg = getattr(cfg, "text_config", None) or cfg
            raw = getattr(text_cfg, "dtype", None) or getattr(text_cfg, "torch_dtype", None)
            cfg_dtype_name = str(raw).replace("torch.", "") if raw is not None else None
        except Exception:
            cfg_dtype_name = None

        # config 声明 bf16 且设备支持 → 必须 bf16（fp16 会导致混合层数值溢出）
        if cfg_dtype_name in ("bfloat16", "bf16"):
            return torch.bfloat16 if bf16_supported else torch.float32
        # config 声明 fp16 → 遵循
        if cfg_dtype_name in ("float16", "fp16"):
            return torch.float16
        # 未声明时：设备支持 bf16 则默认 bf16（更宽的动态范围，更安全）
        return torch.bfloat16 if bf16_supported else torch.float16

    @staticmethod
    def _is_cuda_compatible(torch: Any) -> bool:
        """验证当前 PyTorch 是否真能在检测到的 CUDA 设备上执行 kernel。

        某些 GPU 的算力（compute capability）比当前 PyTorch 构建支持的范围更新
        （例如 RTX 50 系列的 sm_120 与 PyTorch 2.6+cu124）。此时
        ``torch.cuda.is_available()`` 仍会返回 True，但任何真正的 CUDA 运算都会抛出
        ``RuntimeError: no kernel image is available for execution on the device``。

        本方法执行一次微小的张量运算来确认 CUDA 不仅"可见"而且"可用"，避免后续
        模型加载时因不兼容架构而崩溃。

        Returns:
            当前 PyTorch 能在 CUDA 上执行 kernel 时返回 True，否则 False。
        """
        if not torch.cuda.is_available():
            return False
        try:
            # 执行一次需要 CUDA kernel 的微小运算，捕获算力不兼容等真实错误。
            a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            b = torch.tensor([1.0, 1.0, 1.0], device="cuda")
            _ = (a + b).sum().item()
            return True
        except RuntimeError:
            return False

    @staticmethod
    def _select_device(torch: Any, custom_device: str | None = None) -> str:
        """根据硬件环境选择推理设备 / Select Inference Device.

        设备选择级联策略 (Cascading Order):
        1. 显式配置优先：custom_device 参数 或 PRIVACY_LLM_DEVICE / PRIVACY_DEVICE 环境变量。
        2. CUDA GPU 探测（NVIDIA 卡优先）：开启 CUDA 算力与显存校验，通过则选择 "cuda"。
        3. Mac Metal 探测（Apple Silicon 芯片）：若 macOS MPS 可用，选择 "mps"。
        4. CPU 回退：无 GPU 或加速器时降级至 "cpu"。

        Returns:
            设备字符串："cuda" / "mps" / "cpu" 等。
        """
        # 1. 显式配置或环境变量优先
        target_device = (
            custom_device
            or os.environ.get("PRIVACY_LLM_DEVICE")
            or os.environ.get("PRIVACY_DEVICE")
        )
        if target_device:
            target_device_lower = target_device.lower()
            if target_device_lower in ("cpu", "mps") or target_device_lower.startswith("cuda"):
                if target_device_lower.startswith("cuda") and not Qwen3Classifier._is_cuda_compatible(torch):
                    logger.warning(
                        "qwen3_custom_cuda_not_compatible_fallback_next",
                        extra={"target_device": target_device},
                    )
                else:
                    return target_device_lower

        # 2. 级联 1：优先检测 NVIDIA CUDA GPU
        if torch.cuda.is_available() and Qwen3Classifier._is_cuda_compatible(torch):
            try:
                total_free = sum(
                    torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())
                )
                # Qwen3.5-0.8B FP16 约需 1.6GB 显存，可通过 PRIVACY_VLM_MIN_VRAM_GB 配置（默认 1.6GB）
                min_vram_gb = float(os.environ.get("PRIVACY_VLM_MIN_VRAM_GB", "1.6"))
                min_vram_bytes = min_vram_gb * 1024 * 1024 * 1024
                if total_free >= min_vram_bytes:
                    logger.info(
                        "qwen3_select_cuda",
                        extra={
                            "device_count": torch.cuda.device_count(),
                            "free_vram_gb": round(total_free / (1024**3), 2),
                        },
                    )
                    return "cuda"
                logger.info(
                    "qwen3_cuda_vram_insufficient_checking_mps_or_cpu",
                    extra={
                        "free_vram_gb": round(total_free / (1024**3), 2),
                        "required_vram_gb": min_vram_gb,
                    },
                )
            except Exception as e:
                logger.warning(
                    "qwen3_vram_check_failed_checking_mps_or_cpu",
                    extra={"error": str(e)},
                )
                # 即使显存获取异常，CUDA 本身依然兼容可用，直接使用 CUDA
                return "cuda"

        # 3. 级联 2：检测 Apple Silicon Mac Metal (MPS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("qwen3_select_mps_metal", extra={"engine": "qwen3", "device": "mps"})
            return "mps"

        # 4. 级联 3：无可用 GPU 时降级至 CPU
        logger.info("qwen3_select_cpu_fallback", extra={"engine": "qwen3", "device": "cpu"})
        return "cpu"

    def classify(
        self,
        text: str,
        upstream_level: SensitivityLevel,
        upstream_confidence: float,
        sanitize: bool = False,
    ) -> dict[str, Any] | None:
        """使用本地 Qwen3.5 微调模型对输入进行分类。"""
        # 检测图片输入：纯文本模型不支持图像，直接返回 None 触发降级
        from .image_redaction import is_image_input
        if is_image_input(text):
            logger.debug("qwen3_skip_image_input", extra={"reason": "text_only_model"})
            return None

        try:
            self._lazy_init()
        except Exception:
            # 初始化失败（依赖缺失/模型不存在），递增失败计数并返回 None 触发降级
            CLASSIFICATION_LLM_TOTAL.labels(status="init_failed").inc()
            return None  # 初始化失败，直接返回 None，自动触发底层降级逻辑

        # 记录推理开始时间（monotonic 单调时钟，不受系统时间调整影响）
        start_time = time.monotonic()
        try:
            # 将实际推理提交到专用线程池，设置超时保护。
            # 如果推理超时（如模型卡死），放弃本次推理并返回 None 触发降级，
            # 避免永久阻塞 gRPC 工作线程导致后续所有请求排队失败。
            future = self._executor.submit(
                self._do_classify, text, upstream_level, upstream_confidence
            )
            # 等待推理结果，超过 _INFERENCE_TIMEOUT 秒则抛出 FuturesTimeoutError
            result = future.result(timeout=self._INFERENCE_TIMEOUT)

            # 计算推理耗时并记录到 Prometheus 直方图指标
            duration = time.monotonic() - start_time
            CLASSIFICATION_LLM_DURATION.labels(engine="qwen3").observe(duration)
            # 记录推理完成的 debug 日志
            logger.debug(
                "llm_classify_completed",
                extra={
                    "duration_s": round(duration, 4),
                    "has_result": result is not None,
                },
            )
            return result

        except FuturesTimeoutError:
            # 推理超时：模型可能卡死或输入过于复杂
            duration = time.monotonic() - start_time
            # 递增超时状态计数器
            CLASSIFICATION_LLM_TOTAL.labels(status="timeout").inc()
            # 记录超时耗时到直方图
            CLASSIFICATION_LLM_DURATION.labels(engine="qwen3").observe(duration)
            # 记录超时错误日志
            logger.error(
                "llm_classify_timeout",
                extra={
                    "timeout_s": self._INFERENCE_TIMEOUT,
                    "duration_s": round(duration, 4),
                },
            )
            return None  # 返回 None 触发降级

        except Exception as e:
            # 推理过程中发生其他异常（OOM/模型错误等）
            duration = time.monotonic() - start_time
            # 递增错误状态计数器
            CLASSIFICATION_LLM_TOTAL.labels(status="error").inc()
            # 记录错误耗时到直方图
            CLASSIFICATION_LLM_DURATION.labels(engine="qwen3").observe(duration)
            # 记录错误详情日志
            logger.error(
                "llm_classify_error",
                extra={"error": str(e), "duration_s": round(duration, 4)},
            )
            return None  # 返回 None 触发降级

    def _do_classify(
        self, text: str, upstream_level: SensitivityLevel, upstream_confidence: float
    ) -> dict[str, Any] | None:
        """实际执行模型推理的内部方法（在专用线程中运行）。

        使用 self._lock 保护推理过程，确保同一时刻只有一个线程在执行
        模型推理，避免多线程并发推理导致显存/内存争用引发 OOM 崩溃。

        注意：此方法在 _executor 线程池的工作线程中执行，而非 gRPC 线程。
        锁的获取可能阻塞（当另一个推理正在进行时），但外层的超时机制
        会确保 gRPC 线程不会永久等待。

        Args:
            text: 待分类文本。
            upstream_level: 上游敏感度等级。
            upstream_confidence: 上游置信度。

        Returns:
            分类结果字典或 None。
        """
        # 获取互斥锁，串行化推理（防止并发推理导致 OOM）
        with self._lock:
            # 委托给 _classify_inner 执行实际的推理逻辑
            return self._classify_inner(text, upstream_level, upstream_confidence)

    def _classify_inner(
        self, text: str, upstream_level: SensitivityLevel, upstream_confidence: float
    ) -> dict[str, Any] | None:
        """模型推理核心逻辑（已持有锁）。

        执行步骤：
        1. 构建 system prompt（定义评估标准和输出格式）
        2. 构建 user content（纯文本输入）
        3. 使用 tokenizer 构建模型输入张量
        4. 执行模型 generate 推理
        5. 解码生成 token 为文本
        6. 从文本中提取 JSON 结构化结果

        Args:
            text: 待分类文本。
            upstream_level: 上游敏感度等级（供 prompt 参考）。
            upstream_confidence: 上游置信度（供 prompt 参考）。

        Returns:
            解析后的分类结果字典或 None。
        """
        try:
            # 构建 system prompt：定义角色、评估标准和输出 JSON 格式
            # 优先使用自定义模板（支持 {domain}/{standard_id}/{levels_desc} 占位符）
            #
            # 训练/推理对齐说明（仅 Qwen3Classifier 加载微调模型，其他引擎保留原 prompt）：
            # 本引擎的 L1~L5 等级定义与输出字段契约（final_level/confidence/reasoning/
            # sanitized_text）与微调训练侧 llmlora/src/dataset/loader.py 的
            # SYSTEM_PROMPT 及输出 schema 保持一致，避免训练/推理分布偏移导致
            # 微调模型输出漂移；渲染 prompt 时亦与训练侧一致地关闭 thinking 模式。
            is_finetuned = self._is_finetuned_model()
            if is_finetuned:
                # 微调模型：system prompt 必须与训练侧完全一致（不含 JSON schema 说明），
                # 否则 0.8B 小模型会生成漂移、提前 EOS 导致 JSON 截断。
                system_prompt = _FINETUNED_SYSTEM_PROMPT
            elif self._classify_prompt_template:
                system_prompt = self._classify_prompt_template.format(
                    domain="medical",
                    standard_id="DB51_T_2989",
                    levels_desc=(
                        "- L1 (公开数据): 无敏感信息的公开资讯、通用日常文本。\n"
                        "- L2 (内部数据): 业务统计指标、系统日志、设备运维等低敏感内部数据。\n"
                        "- L3 (敏感数据/个人基本信息): 姓名、身份证号、手机号、银行卡号、电子邮箱等个人基础标识与资产信息。\n"
                        "- L4 (高敏感数据/诊疗与金融敏感): 疾病诊断（如重度抑郁症、高血压、冠心病）、病历主诉、处方药品等医疗健康敏感信息。\n"
                        "- L5 (极敏感数据): 基因组、生物特征、特级商业机密等核心数据。"
                    ),
                )
            else:
                system_prompt = (
                    "你是一个专业的隐私安全Sidecar助手。请分析输入的文本，识别敏感信息，"
                    "输出分类分级结果（JSON格式），并提供语义连贯的无痕抹平脱敏重写文本。\n\n"
                    "【数据分类分级标准指南】\n"
                    "- L1 (公开数据): 无敏感信息的公开资讯、通用日常文本。\n"
                    "- L2 (内部数据): 业务统计指标、系统日志、设备运维等低敏感内部数据。\n"
                    "- L3 (敏感数据/个人基本信息): 姓名、身份证号、手机号、银行卡号、电子邮箱等个人基础标识与资产信息。\n"
                    "- L4 (高敏感数据/诊疗与金融敏感): 疾病诊断（如重度抑郁症、高血压、冠心病）、病历主诉、处方药品等医疗健康敏感信息。\n"
                    "- L5 (极敏感数据): 基因组、生物特征、特级商业机密等核心数据。\n\n"
                    "请严格根据上述标准进行定级，并仅输出符合以下 JSON 格式的结构化内容，不要包含额外的解释文字或 ``` 块：\n"  # noqa: E501
                    "{\n"
                    '  "final_level": "L1/L2/L3/L4/L5",\n'
                    '  "confidence": 0.0到1.0之间的浮点数,\n'
                    '  "reasoning": "定级判别的推理过程说明",\n'
                    '  "sanitized_text": "语义连贯的无痕抹平脱敏重写文本"\n'
                    "}"
                )

            # 构建 user content：
            # - 微调模型：训练样本的 user 消息为裸文本（无前导语、无分隔符包裹），
            #   推理必须保持一致，否则生成漂移/截断。仅剥离 chat-template 控制 token
            #   防止伪造对话轮次（不改变可见文本分布）。
            # - 非微调模型：保留前导语 + ««« 分隔符包裹的 Prompt 注入防护。
            if is_finetuned:
                user_text = sanitize_for_prompt(text)
            else:
                user_text = f"请评估以下文本数据的敏感数据等级：\n{wrap_untrusted_text(text)}"

            # 组装完整的对话消息列表（system + user）
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]

            # 使用 tokenizer 将对话消息转换为模型可接受的文本 prompt 格式
            # apply_chat_template 会按照模型的对话模板格式化消息。
            # 与训练侧（llmlora/src/dataset/loader.py render_prompt_text）保持一致：
            # Qwen3.5 模板在 add_generation_prompt 时默认注入 <think> 前缀，
            # 必须显式传入 enable_thinking=False 走非思考分支，
            # 否则推理输出会带思考标记导致 JSON 解析失败；
            # 不支持该 kwarg 的旧模板回退为不传（TypeError 兼容）。
            try:
                text_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                text_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            # 使用 tokenizer 将文本转换为模型输入张量（input_ids 等）
            inputs = self._tokenizer(
                text=[text_prompt], padding=True, return_tensors="pt"
            )
            # 将所有输入张量移动到模型所在设备（CUDA/MPS/CPU）
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # 执行模型推理生成
            import torch

            generated_ids = None
            try:
                # 禁用梯度计算（推理模式，节省显存和计算资源）
                with torch.no_grad():
                    try:
                        generated_ids = self._model.generate(**inputs, max_new_tokens=512)
                    except RuntimeError as err:
                        if "cudnn" in str(err).lower() or "SUBLIBRARY_VERSION_MISMATCH" in str(err):
                            logger.warning("cuDNN 版本冲突，自动禁用 cuDNN 加速改用 PyTorch 原生 CUDA 卷积...")
                            torch.backends.cudnn.enabled = False
                            generated_ids = self._model.generate(**inputs, max_new_tokens=512)
                        else:
                            raise

                # 裁剪生成结果：去掉输入 prompt 部分，只保留新生成的 token
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = self._tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                prompt_tokens = len(inputs["input_ids"][0])
                completion_tokens = len(generated_ids_trimmed[0])
                total_tokens = prompt_tokens + completion_tokens

                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
                self.cumulative_usage["prompt_tokens"] += prompt_tokens
                self.cumulative_usage["completion_tokens"] += completion_tokens
                self.cumulative_usage["total_tokens"] += total_tokens

                CLASSIFICATION_LLM_TOKENS_TOTAL.labels(type="prompt", engine="qwen3").inc(prompt_tokens)
                CLASSIFICATION_LLM_TOKENS_TOTAL.labels(type="completion", engine="qwen3").inc(completion_tokens)
                CLASSIFICATION_LLM_TOKENS_TOTAL.labels(type="total", engine="qwen3").inc(total_tokens)

                logger.info(
                    "qwen3_token_usage",
                    extra={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                )

                result = self._parse_json_result(output_text, upstream_level, upstream_confidence)
                if result is not None:
                    result["usage"] = dict(self.last_usage)
                CLASSIFICATION_LLM_TOTAL.labels(status="success").inc()
                return result
            finally:
                # 显式释放临时张量与 CUDA 显存缓存（防 CUDA VRAM OOM 积累）
                del inputs, generated_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            CLASSIFICATION_LLM_TOTAL.labels(status="error").inc()
            logger.error(
                "llm_classify_inner_error",
                extra={"error": str(e)},
            )
            return None  # 返回 None 触发上层降级逻辑

    def _parse_json_result(
        self, output_text: str, upstream_level: SensitivityLevel, upstream_confidence: float
    ) -> dict[str, Any] | None:
        """解析大模型返回的 JSON / Parse LLM JSON Output.

        中文说明：使用正则表达式清洗并解析大模型返回的 JSON。
        大模型可能在 JSON 前后包含额外文字或 ```json``` 代码块标记，
        因此使用正则提取 {} 区间内容进行解析。

        容错策略：
        - 优先提取 {} 包裹的 JSON 内容
        - 如果提取失败，尝试直接解析整个输出文本
        - 解析失败或关键字段缺失则返回 None 触发降级

        English Description: Cleans and parses JSON from LLM output using regex.

        Args:
            output_text: 模型生成的原始文本 / Raw generated text from model.
            upstream_level: 上游敏感度等级 / Upstream sensitivity level.
            upstream_confidence: 上游置信度 / Upstream confidence.

        Returns:
            解析后的结果字典（含 final_level 等字段）或 None / Parsed result dict or None.
        """
        # 先剥离 Qwen3.5 等思考模型可能输出的 <think>...</think> 思考链
        cleaned_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()

        # 使用正则表达式提取 {} 包裹的 JSON 内容
        json_match = re.search(r"(\{.*\})", cleaned_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else cleaned_text

        try:
            res = json.loads(json_str)
            if isinstance(res, dict):
                if "final_level" not in res and "level" in res:
                    res["final_level"] = res["level"]
                if "confidence" not in res:
                    res["confidence"] = 0.9
                if "final_level" in res:
                    cat = res.get("category") or res.get("sub_category") or "GENERAL"
                    res["category"] = cat
                    res["sub_category"] = cat
                    return cast("dict[str, Any]", res)
        except Exception:
            try:
                short_match = re.search(r"(\{[^{}]+\})", cleaned_text, re.DOTALL)
                if short_match:
                    res = json.loads(short_match.group(1))
                    if isinstance(res, dict):
                        if "final_level" not in res and "level" in res:
                            res["final_level"] = res["level"]
                        if "confidence" not in res:
                            res["confidence"] = 0.9
                        if "final_level" in res:
                            cat = res.get("category") or res.get("sub_category") or "GENERAL"
                            res["category"] = cat
                            res["sub_category"] = cat
                            return cast("dict[str, Any]", res)
            except Exception as e:
                logger.warning(
                    "llm_json_parse_failed",
                    extra={"error": str(e)},
                )

        return None


# 向后兼容别名：旧代码可能通过 Qwen2VLClassifier 引用 / Backward-compatible alias
Qwen2VLClassifier = Qwen3Classifier


class OpenAILlmClassifier(LlmClassifier):
    """基于 OpenAI 兼容 HTTP API（如 vLLM、Ollama、DeepSeek、Qwen API）的数据分类分级器。

    支持通过 HTTP POST 服务与部署在 8000 端口的 vLLM OpenAI API 通信，
    执行 Layer-3 LLM 敏感度评估。

    使用 Python 标准库 urllib.request 实现，无需额外的 ML/PyTorch/httpx 依赖。
    """

    _INFERENCE_TIMEOUT = int(
        os.environ.get("PRIVACY_VLM_TIMEOUT", os.environ.get("PRIVACY_LLM_TIMEOUT", "180"))
    )

    def __init__(
        self,
        api_base: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        classify_prompt_template: str | None = None,
        timeout: float | None = None,
    ):
        """初始化 OpenAI/vLLM 接口分类器。

        Args:
            api_base: OpenAI API 基础 URL，默认从 PRIVACY_LLM_API_BASE 或 PRIVACY_VLLM_URL 读取，
                回退至 "http://127.0.0.1:8000/v1"。
            model_name: 模型名称，默认从 PRIVACY_LLM_MODEL_NAME 读取，
                回退至 "Qwen3.5-0.8B-Privacy-Classifier-Smoother"。
            api_key: API 密钥（可选，默认 PRIVACY_LLM_API_KEY 或 "EMPTY"）。
            classify_prompt_template: 自定义分类 system prompt 模板。
            timeout: 超时时间（秒）。
        """
        load_env_file()

        base_url = (
            api_base
            or os.environ.get("PRIVACY_LLM_API_BASE")
            or os.environ.get("PRIVACY_VLLM_URL")
            or "http://127.0.0.1:8000/v1"
        )
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1") and not base_url.endswith("/chat/completions"):
            self.chat_url = f"{base_url}/v1/chat/completions"
        elif base_url.endswith("/v1"):
            self.chat_url = f"{base_url}/chat/completions"
        else:
            self.chat_url = base_url

        self.api_base = base_url
        self.model_name = (
            model_name
            or os.environ.get("PRIVACY_LLM_MODEL_NAME")
            or _DEFAULT_MODEL_DIR
        )
        self.api_key = api_key or os.environ.get("PRIVACY_LLM_API_KEY", "EMPTY")
        self._classify_prompt_template = classify_prompt_template
        self.timeout = timeout or float(self._INFERENCE_TIMEOUT)

        self._initialized = True
        self._init_error: Exception | None = None
        self._lock = threading.Lock()
        max_http_workers = int(os.environ.get("PRIVACY_LLM_HTTP_WORKERS", "16"))
        self._executor = ThreadPoolExecutor(max_workers=max_http_workers, thread_name_prefix="vllm-http-infer")
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.cumulative_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 三态熔断器与并发试探控制 (借鉴 Go 引擎 CircuitBreaker 设计)
        self._cb_state = 0  # 0: Closed, 1: Open, 2: HalfOpen
        self._cb_failures = 0
        self._cb_last_failure = 0.0
        self._cb_cooldown = float(os.environ.get("PRIVACY_LLM_CB_COOLDOWN", "15.0"))
        self._cb_half_open_max = 3
        self._cb_half_open_inflight = 0
        self._cb_lock = threading.Lock()

    def _check_circuit(self) -> tuple[bool, Any]:
        """检查三态熔断器状态，返回 (allowed, release_callback)。"""
        now = time.monotonic()
        with self._cb_lock:
            if self._cb_state == 0:  # Closed
                return True, None

            if self._cb_state == 1:  # Open
                if now - self._cb_last_failure > self._cb_cooldown:
                    self._cb_state = 2  # HalfOpen
                    self._cb_half_open_inflight = 1
                    released = False

                    def _release() -> None:
                        nonlocal released
                        if not released:
                            released = True
                            with self._cb_lock:
                                self._cb_half_open_inflight = max(0, self._cb_half_open_inflight - 1)

                    return True, _release
                return False, None

            # Half-Open 状态限制并发试探配额
            if self._cb_half_open_inflight < self._cb_half_open_max:
                self._cb_half_open_inflight += 1
                released = False

                def _release() -> None:
                    nonlocal released
                    if not released:
                        released = True
                        with self._cb_lock:
                            self._cb_half_open_inflight = max(0, self._cb_half_open_inflight - 1)

                return True, _release
            return False, None

    def _record_success(self) -> None:
        """记录成功调用，自愈重置熔断器。"""
        with self._cb_lock:
            self._cb_failures = 0
            self._cb_state = 0
            self._cb_half_open_inflight = 0

    def _record_failure(self) -> None:
        """记录失败调用，连续超阈值触发熔断。"""
        with self._cb_lock:
            self._cb_failures += 1
            self._cb_last_failure = time.monotonic()
            if self._cb_failures >= 3 or self._cb_state == 2:
                self._cb_state = 1
                self._cb_half_open_inflight = 0

    def _is_finetuned_model(self) -> bool:
        """判断当前请求的模型是否为项目微调的 Privacy-Classifier-Smoother 模型。

        当 model_name 匹配微调模型名时，使用与训练侧一致的 system prompt 和
        裸用户文本，避免 0.8B 小模型因 prompt 分布漂移导致 JSON 截断/定级错误。
        """
        if self._classify_prompt_template is not None:
            return False
        base = os.path.basename(self.model_name.rstrip("/"))
        return base == _DEFAULT_MODEL_DIR

    @property
    def is_ready(self) -> bool:
        """服务器是否已就绪。"""
        return self._initialized and self._init_error is None

    def warmup(self) -> bool:
        """预热连接（保持与基类一致）。"""
        return True

    def classify(
        self,
        text: str,
        upstream_level: SensitivityLevel,
        upstream_confidence: float,
        sanitize: bool = False,
    ) -> dict[str, Any] | None:
        """通过 HTTP 调用 vLLM / OpenAI 服务对文本进行定级。"""
        # 检测图片输入：纯文本模型不支持图像，直接返回 None 触发降级
        from .image_redaction import is_image_input
        if is_image_input(text):
            logger.debug("openai_llm_skip_image_input", extra={"reason": "text_only_model"})
            return None

        # 熔断器快速拦截（Half-Open 下限制并发试探配额）
        allowed, release_probe = self._check_circuit()
        if not allowed:
            logger.debug("openai_llm_circuit_open_fast_fail", extra={"url": self.chat_url})
            return None

        start_time = time.monotonic()
        try:
            future = self._executor.submit(
                self._do_classify_http, text, upstream_level, upstream_confidence
            )
            result = future.result(timeout=self.timeout)
            duration = time.monotonic() - start_time
            CLASSIFICATION_LLM_DURATION.labels(engine="vllm").observe(duration)
            if result is not None:
                self._record_success()
            else:
                self._record_failure()
            return result
        except FuturesTimeoutError:
            self._record_failure()
            duration = time.monotonic() - start_time
            CLASSIFICATION_LLM_TOTAL.labels(status="timeout").inc()
            CLASSIFICATION_LLM_DURATION.labels(engine="vllm").observe(duration)
            logger.error("vllm_http_classify_timeout", extra={"timeout_s": self.timeout, "url": self.chat_url})
            return None
        except Exception as e:
            self._record_failure()
            duration = time.monotonic() - start_time
            CLASSIFICATION_LLM_TOTAL.labels(status="error").inc()
            CLASSIFICATION_LLM_DURATION.labels(engine="vllm").observe(duration)
            logger.error("vllm_http_classify_error", extra={"error": str(e), "url": self.chat_url})
            return None
        finally:
            if release_probe is not None:
                release_probe()
    def sanitize_text(self, text: str) -> str:
        """纯脱敏与无痕抹平接口：仅对文本进行 PII 与敏感信息抹平重写，不考虑分类分级逻辑。

        Args:
            text: 待脱敏的原始文本段落。

        Returns:
            脱敏抹平后的重写文本。
        """
        result = self.classify(text, SensitivityLevel.L3, 0.6)
        sanitized: str | None = None
        if result and isinstance(result, dict) and result.get("sanitized_text"):
            sanitized = str(result["sanitized_text"])

        if not sanitized:
            sanitized = text

        # 规则双重兜底防护：
        # 1. 硬敏感 PII 特征（18位身份证号、11位手机号、常见中文姓名如"张三"）正则打码
        sanitized = re.sub(
            r"([1-9]\d{5})\d{8}(\d{3}[\dXx])",
            r"\1********\2",
            sanitized,
        )
        sanitized = re.sub(
            r"(1[3-9]\d)\d{4}(\d{4})",
            r"\1****\2",
            sanitized,
        )
        # 中文姓名格式化掩码（如 "患者张三" -> "患者张*"；"张三" -> "张*"）
        sanitized = re.sub(
            r"(?<=姓名[：:])\s*([\u4e00-\u9fa5])[\u4e00-\u9fa5]{1,2}",
            r"\1*",
            sanitized,
        )
        sanitized = re.sub(
            r"(?<=患者)\s*([\u4e00-\u9fa5])[\u4e00-\u9fa5]{1,2}",
            r"\1*",
            sanitized,
        )

        # 2. L5 级特种极高敏病种（HIV/艾滋、性病、重度精神障碍、特异性抗病毒药物）无痕擦除（替换为空字符串或顺滑连词，严禁输出"[已抹平]"等标签）
        l5_purge_patterns = [
            r"(?i)\bHIV(?:-1|-2)?\b",
            r"(?i)\bAIDS\b",
            r"艾滋病?",
            r"人免疫缺陷病毒",
            r"获得性免疫缺陷综合征",
            r"梅毒",
            r"(?i)\bsyphilis\b",
            r"淋病",
            r"(?i)\bgonorrhea\b",
            r"精神分裂症?",
            r"(?i)\bschizophrenia\b",
            r"抗逆转录[治疗]*",
            r"HAART[治疗]*",
            r"替诺福韦",
            r"拉米夫定",
            r"依非韦伦",
            r"多替拉韦",
        ]
        for pat in l5_purge_patterns:
            sanitized = re.sub(pat, "", sanitized)

        # 3. 清理擦除特种药物与病名后遗留的悬空运算符与修饰括号（如 "（ +  + ）" -> ""；"（感染期）" -> ""）
        sanitized = re.sub(r"[\(（][\s\+\-\*\/]*[\)）]", "", sanitized)
        sanitized = re.sub(
            r"[\(（]\s*(?:HIV\s*)?(?:[\u4e00-\u9fa5]{0,6}(?:期|型|阶段|试验)|期|型)?\s*[\)）]",
            "",
            sanitized,
        )

        # 4. 修正残缺的治疗动作短语与悬空谓语（如 "开展  抗病毒治疗" -> "开展常规对症治疗"；"诊断为，" -> "，"）
        sanitized = re.sub(r"开展\s*(?:HAART\s*)?抗病毒治疗", "开展常规对症治疗", sanitized)
        sanitized = re.sub(r"(?:HAART\s*)?抗病毒治疗", "常规对症治疗", sanitized)
        sanitized = re.sub(r"(?:诊断为|确诊为|提示为?|显示为?|检查出|主诉为?)\s*([，,。；;])", r"\1", sanitized)

        # 5. 【关键】：在所有敏感词与修饰括号擦除完成后，执行标点碰撞修复（消除 "初步诊断：，" -> "初步诊断："）
        sanitized = re.sub(r"([：:])\s*[，,、]", r"\1", sanitized)
        sanitized = re.sub(r"([：:])\s*[。；;]", r"。", sanitized)
        sanitized = re.sub(r"([，,])\s*([。；;])", r"\2", sanitized)
        sanitized = re.sub(r"([，,。；;])\s*\1+", r"\1", sanitized)

        # 6. 规范空白字符：合并多重空格，消除中文标点前后的多余悬空空格
        sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
        sanitized = re.sub(r"\s+([，,。；;：:\)）])", r"\1", sanitized)
        sanitized = re.sub(r"([\(（])\s+", r"\1", sanitized)

        return sanitized.strip()

    def _do_classify_http(
        self, text: str, upstream_level: SensitivityLevel, upstream_confidence: float
    ) -> dict[str, Any] | None:
        """执行 HTTP POST 请求与 JSON 解析。"""
        if self._classify_prompt_template:
            system_prompt = self._classify_prompt_template.format(
                domain="medical",
                standard_id="DB51_T_2989",
                levels_desc=(
                    "- L5 (极高风险): 包含人类基因序列、遗传信息、基因突变或罕见病样本。\n"
                    "- L4 (高风险): 包含精神疾病、敏感传染病或完整的住院病历。\n"
                    "- L3 (中风险): 包含个人身份信息（PII）、普通的门诊诊疗记录或常规检验指标数值。\n"
                    "- L2 (低风险): 仅包含医院科室运营、设备使用率或脱敏后的去标识化统计数据。\n"
                    "- L1 (公开级): 年度门诊总量等医院公开宣传、无任何敏感特征的统计指标。"
                ),
            )
        elif self._is_finetuned_model():
            # 微调模型：必须与训练侧（llmlora/src/dataset/loader.py）的 system prompt
            # 以及裸用户文本保持一致，否则 0.8B 小模型会生成漂移/提前 EOS。
            system_prompt = _FINETUNED_SYSTEM_PROMPT
        else:
            system_prompt = (
                "你是一个医疗数据分类分级与隐私脱敏领域的资深安全专家。\n"
                "请对输入的医疗数据进行敏感等级评估，并生成语义流畅、通顺自然的无痕抹平脱敏文本。\n\n"
                "评估标准如下：\n"
                "- L5 (极高风险): 包含人类基因序列、遗传信息、基因突变（如 BRCA1/TP53）或罕见病样本。\n"
                "- L4 (高风险): 包含精神疾病（如精神分裂）、敏感传染病（如 HIV/AIDS/梅毒）或完整的住院病历。\n"
                "- L3 (中风险): 包含个人身份信息（PII，如身份证号、手机号）、普通的门诊诊疗记录或常规检验指标数值（如血常规）。\n"
                "- L2 (低风险): 仅包含医院科室运营、设备使用率或脱敏后的去标识化统计数据。\n"
                "- L1 (公开级): 年度门诊总量等医院公开宣传、无任何敏感和特征的统计指标。\n\n"
                "【脱敏重写语法与通顺度要求】：\n"
                "1. 语法自愈润色：擦除或抹平敏感信息（如 HIV、梅毒、特定高敏诊断或 PII 姓名/身份证号）后，必须对前后连接词、介词、标点进行自然重构与润色，保证上下文语法通顺流畅，严禁遗留“诊断为，”、“结果：”等悬空断句残渣。\n"
                "2. 处方整句重构：遇到多药联合处方（如“替诺福韦 + 拉米夫定 + 依非韦伦”）或抗病毒方案时，必须将整句用药平滑重构为“开展常规对症治疗与健康管理”，严禁逐字擦除后残留加号“+”、“（ + + ）”空括号或多余空格。\n"
                "3. 孤立修饰词与括号清理：擦除主病名（如“艾滋病”、“梅毒”）时，其紧随其后的病期、分型、试验或变体修饰括号（如“（HIV 感染期）”、“（确证试验）”、“（早期隐性梅毒）”）必须一并连同括号整块擦除，严禁留存“（感染期）”、“（确证试验）”等悬空修饰短语。\n"
                "4. 标点符号语法自愈：擦除主病名或小标题后的敏感词时，严禁输出“：，”或“：。”等非法标点组合（如“初步诊断：，伴...”必须平滑润色为“初步诊断：伴卡氏肺孢子虫肺炎。”或“初步诊断：卡氏肺孢子虫肺炎。”）。\n"
                "5. 语义平滑替代：彻底抹平的敏感范畴可重构为常规身体检查或常规指标评估（如“诊断为 HIV 阳性”平滑润色为“开展常规健康体检与常规指标监测”），保证整段文本读起来完全通顺。\n"
                "6. 严禁生硬标记：重写结果 sanitized_text 中严禁包含“[已抹平]”、“[脱敏]”、“[泛化]”等生硬的人工占位标记，必须输出自然可读的完整段落。\n\n"
                "请仅输出符合以下 JSON 格式的结构化内容：\n"
                "{\n"
                '  "final_level": "L1/L2/L3/L4/L5",\n'
                '  "sub_category": "分类标签简称",\n'
                '  "confidence": 0.0到1.0之间的浮点数,\n'
                '  "reasoning": "定级判别的推理过程说明",\n'
                '  "sanitized_text": "自然通顺、语法连贯的无痕抹平脱敏重写文本",\n'
                '  "needs_human_review": true/false\n'
                "}"
            )

        # 微调模型：user 消息必须是裸文本，与训练样本一致；非微调模型保留注入防护包装。
        if self._is_finetuned_model():
            user_text = sanitize_for_prompt(text)
        else:
            user_text = f"请评估以下文本数据的敏感数据等级：\n{wrap_untrusted_text(text)}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        import urllib.error
        import urllib.request

        req_data = json.dumps(payload).encode("utf-8")
        # noqa: S310 —— chat_url 来自显式运维配置（vLLM API base），非用户输入拼接
        req = urllib.request.Request(  # noqa: S310
            self.chat_url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                if resp.status != 200:
                    logger.warning("vllm_http_non_200", extra={"status": resp.status})
                    CLASSIFICATION_LLM_TOTAL.labels(status="error").inc()
                    return None
                resp_body = resp.read().decode("utf-8")
                resp_json = json.loads(resp_body)
                content = resp_json["choices"][0]["message"]["content"]

                # 统计 Token 消耗（输入、输出、总计）
                usage = resp_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
                self.cumulative_usage["prompt_tokens"] += prompt_tokens
                self.cumulative_usage["completion_tokens"] += completion_tokens
                self.cumulative_usage["total_tokens"] += total_tokens

                if prompt_tokens:
                    CLASSIFICATION_LLM_TOKENS_TOTAL.labels(type="prompt", engine="vllm").inc(prompt_tokens)
                if completion_tokens:
                    CLASSIFICATION_LLM_TOKENS_TOTAL.labels(type="completion", engine="vllm").inc(completion_tokens)
                if total_tokens:
                    CLASSIFICATION_LLM_TOKENS_TOTAL.labels(type="total", engine="vllm").inc(total_tokens)

                logger.info(
                    "vllm_token_usage",
                    extra={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                )

                result = self._parse_json_result(content, upstream_level, upstream_confidence)
                if result:
                    result["usage"] = dict(self.last_usage)
                    CLASSIFICATION_LLM_TOTAL.labels(status="success").inc()
                else:
                    CLASSIFICATION_LLM_TOTAL.labels(status="error").inc()
                return result
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as err:
            logger.warning("vllm_http_request_failed", extra={"error": str(err)})
            CLASSIFICATION_LLM_TOTAL.labels(status="error").inc()
            return None

    def _parse_json_result(
        self, output_text: str, upstream_level: SensitivityLevel, upstream_confidence: float
    ) -> dict[str, Any] | None:
        cleaned_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
        json_match = re.search(r"(\{.*\})", cleaned_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else cleaned_text
        try:
            res = json.loads(json_str)
            if isinstance(res, dict):
                if "final_level" not in res and "level" in res:
                    res["final_level"] = res["level"]
                if "confidence" not in res:
                    res["confidence"] = 0.9
                if "final_level" in res:
                    return cast("dict[str, Any]", res)
        except Exception:
            try:
                short_match = re.search(r"(\{[^{}]+\})", cleaned_text, re.DOTALL)
                if short_match:
                    res = json.loads(short_match.group(1))
                    if isinstance(res, dict):
                        if "final_level" not in res and "level" in res:
                            res["final_level"] = res["level"]
                        if "confidence" not in res:
                            res["confidence"] = 0.9
                        if "final_level" in res:
                            return cast("dict[str, Any]", res)
            except Exception as e:
                logger.warning("llm_json_parse_failed", extra={"error": str(e)})
        return None


def build_prompt_from_domain_and_taxonomy_yaml(
    domain_yaml_path: str | Path,
    taxonomy_yaml_path: str | Path,
) -> str:
    """从领域规则 YAML (如 medical.yaml) 和分类体系 YAML (如 default.yaml) 动态解析并构建完整的 System Prompt。

    解析内容包含：
    1. 【数据分类分级标准指南】：分类等级 ID、名称、Rank 及描述；
    2. 【敏感数据脱敏抹平与泛化治理策略指南】：Purge 彻底抹平范畴与 Generalization 抽象泛化范畴。

    Args:
        domain_yaml_path: 领域规则配置文件路径 (如 rules/domains/medical.yaml)。
        taxonomy_yaml_path: 分类体系配置文件路径 (如 rules/taxonomies/default.yaml)。

    Returns:
        包含完整分类标准与脱敏抹平策略指南的 System Prompt 文本。
    """
    import yaml

    with open(domain_yaml_path, encoding="utf-8") as f:
        domain_cfg = yaml.safe_load(f) or {}

    with open(taxonomy_yaml_path, encoding="utf-8") as f:
        taxonomy_cfg = yaml.safe_load(f) or {}

    # 1. 动态生成【数据分类分级标准指南】
    levels_desc_lines = ["【数据分类分级标准指南】"]
    levels = taxonomy_cfg.get("levels", {})
    for lvl_id in sorted(levels.keys(), key=lambda k: levels[k].get("rank", 0)):
        lvl_info = levels[lvl_id]
        levels_desc_lines.append(f"- {lvl_info['id']} ({lvl_info['name']}): {lvl_info['description']}")

    # 2. 动态生成【敏感数据脱敏抹平与泛化治理策略指南】
    redaction_cfg = domain_cfg.get("redaction_strategy", {})
    purge_cats = redaction_cfg.get("purge_categories", [])
    gen_cats = redaction_cfg.get("generalization_categories", [])

    strategy_lines = ["【敏感数据脱敏抹平与泛化治理策略指南】"]
    if purge_cats:
        strategy_lines.append(f"- 彻底抹平范畴 (Purge - 零痕迹擦除/替换为通用掩码): {', '.join(purge_cats)}")
    if gen_cats:
        strategy_lines.append(f"- 范畴化泛化范畴 (Generalization - 重构为系统器官大类疾病): {', '.join(gen_cats)}")
    strategy_lines.append("- 个人基础标识 (PII - 姓名/身份证号/手机号/住址): 进行脱敏掩码抹平处理")

    fluency_lines = [
        "【脱敏重写语法与通顺度要求】",
        "1. 语法自愈润色：擦除或抹平敏感信息（如 HIV、梅毒、特定高敏诊断或 PII 姓名/身份证号）后，必须对前后连接词、介词、标点进行自然重构与润色，保证上下文语法通顺流畅，严禁遗留“诊断为，”、“结果：”等悬空断句残渣。",
        "2. 处方整句重构：遇到多药联合处方（如“替诺福韦 + 拉米夫定 + 依非韦伦”）或抗病毒方案时，必须将整句用药平滑重构为“开展常规对症治疗与健康管理”，严禁逐字擦除后残留加号“+”、“（ + + ）”空括号或多余空格。",
        "3. 孤立修饰词与括号清理：擦除主病名（如“艾滋病”、“梅毒”）时，其紧随其后的病期、分型、试验或变体修饰括号（如“（HIV 感染期）”、“（确证试验）”、“（早期隐性梅毒）”）必须一并连同括号整块擦除，严禁留存“（感染期）”、“（确证试验）”等悬空修饰短语。",
        "4. 标点符号语法自愈：擦除主病名或小标题后的敏感词时，严禁输出“：，”或“：。”等非法标点组合（如“初步诊断：，伴...”必须平滑润色为“初步诊断：伴卡氏肺孢子虫肺炎。”或“初步诊断：卡氏肺孢子虫肺炎。”）。",
        "5. 语义平滑替代：彻底抹平的敏感范畴可重构为常规身体检查或常规指标评估（如“诊断为 HIV 阳性”平滑润色为“开展常规健康体检与常规指标监测”），保证整段文本读起来完全通顺。",
        "6. 严禁生硬标记：重写结果 sanitized_text 中严禁包含“[已抹平]”、“[脱敏]”、“[泛化]”等生硬的人工占位标记，必须输出自然可读的完整段落。",
    ]

    guidance_str = "\n".join(levels_desc_lines) + "\n\n" + "\n".join(strategy_lines) + "\n\n" + "\n".join(fluency_lines)

    system_prompt = (
        f"你是一个专业的隐私安全Sidecar助手。请分析输入的文本，识别敏感信息，"
        f"输出分类分级结果（JSON格式），并提供语义流畅、语法通顺的无痕抹平脱敏重写文本。\n\n"
        f"{guidance_str}\n\n"
        f"请严格根据上述标准进行定级与脱敏抹平，并仅输出符合以下 JSON 格式的结构化内容：\n"
        f"{{\n"
        f'  "final_level": "L1/L2/L3/L4/L5",\n'
        f'  "confidence": 0.0到1.0之间的浮点数,\n'
        f'  "reasoning": "定级判别的推理过程说明",\n'
        f'  "sanitized_text": "自然通顺、语法连贯的无痕抹平脱敏重写文本"\n'
        f"}}"
    )
    return system_prompt


# 别名定义
VLLMLlmClassifier = OpenAILlmClassifier

