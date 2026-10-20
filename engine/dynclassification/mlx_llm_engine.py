"""基于 Apple MLX (Metal GPU) 的本地 LLM 分类器引擎。

中文说明：
使用 Apple MLX 框架在 macOS Metal GPU 上执行 Qwen2 语言模型推理，
为三层分类漏斗提供 Layer-3 LLM 深度分类与仲裁能力。

模型结构：Qwen2 Decoder (28 layers, GQA, RoPE, SwiGLU)
仅使用语言模型部分（text-only），不加载视觉编码器。

降级策略：
- mlx 未安装或非 macOS → 抛出 ImportError → 回退到 PyTorch 引擎
- 模型文件不存在 → 抛出 FileNotFoundError → 回退

English Description:
Local LLM classifier engine using Apple MLX framework for Metal GPU inference on macOS.
Provides Layer-3 LLM deep classification for the three-layer funnel.
Uses only the language model part (text-only), does not load the vision encoder.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from typing import Any

from ..observability.logging_config import get_logger
from ..observability.metrics import (
    CLASSIFICATION_LLM_DURATION,
    CLASSIFICATION_LLM_TOKENS_TOTAL,
    CLASSIFICATION_LLM_TOTAL,
)
from .base import LlmClassifier, SensitivityLevel
from .utils import wrap_untrusted_text

logger = get_logger(__name__)


class _LightweightTokenizer:
    """轻量级 Tokenizer 包装器（基于 HuggingFace tokenizers 库）。

    提供与 transformers AutoTokenizer 兼容的接口，
    但不依赖 transformers 库（避免其 numpy 版本检测问题）。
    """

    def __init__(self, tokenizer: Any, model_dir: str):
        self._tokenizer = tokenizer
        self._model_dir = model_dir
        # 加载 chat template（如果存在）
        self._chat_template: str | None = None
        chat_template_path = os.path.join(model_dir, "chat_template.json")
        if os.path.exists(chat_template_path):
            try:
                with open(chat_template_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._chat_template = data.get("chat_template")
            except Exception:
                pass
        # 特殊 token ID
        self._bos_token_id = 151643
        self._eos_token_id = 151645

    def encode(self, text: str) -> list[int]:
        """编码文本为 token ID 列表。"""
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """解码 token ID 为文本。"""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        """应用 chat template 格式化消息。

        使用 Qwen2 的对话模板格式：
        <|im_start|>system
{content}<|im_end|>
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant

        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                # 多模态内容，提取文本部分
                content = " ".join(
                    item.get("text", "") for item in content if item.get("type") == "text"
                )
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


def _rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    """RMS Normalization (MLX)."""
    import mlx.core as mx
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def _silu(x: Any) -> Any:
    """SiLU / Swish activation (MLX)."""
    import mlx.core as mx
    return x * mx.sigmoid(x)


def _rope_freqs(dim: int, seq_len: int, theta: float = 1000000.0, offset: int = 0) -> Any:
    """计算 RoPE 频率矩阵 / Compute RoPE frequency matrix."""
    import mlx.core as mx
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(offset, offset + seq_len, dtype=mx.float32)
    angles = t[:, None] * freqs[None, :]
    return angles


def _apply_rope(x: Any, angles: Any) -> Any:
    """对输入张量应用 RoPE 旋转位置编码 / Apply RoPE to input tensor.

    Args:
        x: shape (num_heads, seq_len, head_dim)
        angles: shape (seq_len, head_dim//2)
    """
    import mlx.core as mx
    head_dim = x.shape[-1]
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = mx.cos(angles)[None, :, :]  # (1, seq_len, half)
    sin = mx.sin(angles)[None, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return mx.concatenate([out1, out2], axis=-1)


class MLXLlmClassifier(LlmClassifier):
    """基于 Apple MLX 的 Qwen2 LLM 分类器 / MLX Metal Qwen2 LLM Classifier.

    使用 MLX 框架加载转换后的 Qwen2 语言模型权重，
    在 Apple Silicon Metal GPU 上执行文本生成推理。

    特性：
    - Metal GPU 加速：利用 Apple Silicon 统一内存架构
    - 仅语言模型：不加载视觉编码器，节省内存
    - KV Cache：自回归生成时缓存历史 KV，避免重复计算
    - 延迟加载：首次调用时才加载模型权重

    English Description:
    Loads converted Qwen2 language model weights via MLX framework
    for Metal GPU text generation inference on Apple Silicon.
    """

    _INFERENCE_TIMEOUT = int(os.environ.get("PRIVACY_VLM_TIMEOUT", "180"))

    def __init__(
        self,
        model_dir: str | None = None,
        classify_prompt_template: str | None = None,
        max_new_tokens: int = 512,
    ):
        """初始化 MLX LLM 分类器。

        Args:
            model_dir: MLX 模型目录（含 weights.safetensors 和 config.json）。
            classify_prompt_template: 自定义分类 prompt 模板。
            max_new_tokens: 最大生成 token 数。
        """
        if not model_dir:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            model_dir = os.path.join(project_root, ".models", "Qwen3.5-0.8B-Privacy-Classifier-Smoother-mlx")

        self.model_dir = model_dir
        self._classify_prompt_template = classify_prompt_template
        self._max_new_tokens = max_new_tokens

        self._weights: dict[str, Any] | None = None
        self._config: dict[str, Any] | None = None
        self._tokenizer: Any = None
        self._initialized = False
        self._init_error: Exception | None = None
        self._lock = threading.Lock()
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.cumulative_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @property
    def is_ready(self) -> bool:
        """模型是否已就绪。"""
        return self._initialized and self._init_error is None

    def _load_tokenizer(self) -> Any:
        """加载 tokenizer（优先 tokenizers 库，回退 transformers）。"""
        tokenizer_json = os.path.join(self.model_dir, "tokenizer.json")

        # 方案 1: 使用 HuggingFace tokenizers 库（轻量，无 transformers 依赖）
        if os.path.exists(tokenizer_json):
            try:
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(tokenizer_json)
                return _LightweightTokenizer(tok, self.model_dir)
            except Exception as e:
                logger.debug("tokenizers_lib_failed", extra={"error": str(e)})

        # 方案 2: 回退到 transformers AutoTokenizer
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        except Exception as e:
            logger.warning("mlx_llm_tokenizer_load_failed", extra={"error": str(e)})
            raise

    def _lazy_init(self) -> None:
        """延迟加载 MLX 模型权重和 tokenizer。"""
        if self._initialized:
            return
        if self._init_error:
            raise self._init_error

        try:
            import mlx.core as mx

            weights_path = os.path.join(self.model_dir, "weights.safetensors")
            config_path = os.path.join(self.model_dir, "config.json")

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"未找到 MLX 权重文件: {weights_path}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"未找到模型配置: {config_path}")

            with open(config_path, encoding="utf-8") as f:
                self._config = json.load(f)

            # 只加载语言模型权重（跳过 visual.* 以节省内存）
            logger.info("mlx_llm_loading_weights", extra={"model_dir": self.model_dir})
            all_weights = mx.load(weights_path)
            self._weights = {
                k: v for k, v in all_weights.items()
                if k.startswith("model.")
            }
            del all_weights  # 释放视觉编码器权重内存

            # 加载 tokenizer（优先使用 tokenizers 库，避免 transformers 依赖问题）
            self._tokenizer = self._load_tokenizer()

            self._initialized = True
            logger.info(
                "mlx_llm_engine_initialized",
                extra={
                    "model_dir": self.model_dir,
                    "engine": "mlx",
                    "device": "metal",
                    "num_tensors": len(self._weights),
                },
            )
        except Exception as e:
            self._init_error = e
            logger.warning(
                "mlx_llm_engine_init_failed",
                extra={"error": str(e), "model_dir": self.model_dir},
            )
            raise e

    def _forward_step(self, input_ids: Any, cache: list | None = None, offset: int = 0) -> tuple[Any, list]:
        """单步前向传播（支持 KV Cache）。

        Args:
            input_ids: (1, seq_len) 或 (1, 1) 的 token ID。
            cache: KV cache 列表（每层一个 (K, V) 元组）。
            offset: 位置偏移量（用于 RoPE）。

        Returns:
            (logits, updated_cache)
        """
        import mlx.core as mx

        w = self._weights
        cfg = self._config
        hidden_size = cfg["hidden_size"]
        num_heads = cfg["num_attention_heads"]
        num_kv_heads = cfg["num_key_value_heads"]
        num_layers = cfg["num_hidden_layers"]
        head_dim = hidden_size // num_heads
        rope_theta = cfg.get("rope_theta", 1000000.0)
        rms_eps = cfg.get("rms_norm_eps", 1e-6)
        num_groups = num_heads // num_kv_heads

        seq_len = input_ids.shape[1]

        # Token embeddings
        h = w["model.embed_tokens.weight"][input_ids[0]]  # (seq_len, hidden_size)

        if cache is None:
            cache = [None] * num_layers

        new_cache = []

        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}"

            # --- RMSNorm (pre-attention) ---
            ln_w = w[f"{prefix}.input_layernorm.weight"]
            normed = _rms_norm(h, ln_w, rms_eps)

            # --- GQA Self-Attention ---
            q = normed @ w[f"{prefix}.self_attn.q_proj.weight"].T + w[f"{prefix}.self_attn.q_proj.bias"]
            k = normed @ w[f"{prefix}.self_attn.k_proj.weight"].T + w[f"{prefix}.self_attn.k_proj.bias"]
            v = normed @ w[f"{prefix}.self_attn.v_proj.weight"].T + w[f"{prefix}.self_attn.v_proj.bias"]

            # Reshape: (seq_len, dim) → (num_heads, seq_len, head_dim)
            q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
            k = k.reshape(seq_len, num_kv_heads, head_dim).transpose(1, 0, 2)
            v = v.reshape(seq_len, num_kv_heads, head_dim).transpose(1, 0, 2)

            # RoPE
            angles = _rope_freqs(head_dim, seq_len, rope_theta, offset)
            q = _apply_rope(q, angles)
            k = _apply_rope(k, angles)

            # KV Cache
            if cache[layer_idx] is not None:
                prev_k, prev_v = cache[layer_idx]
                k = mx.concatenate([prev_k, k], axis=1)
                v = mx.concatenate([prev_v, v], axis=1)
            new_cache.append((k, v))

            # GQA: 扩展 KV heads 以匹配 Q heads
            if num_groups > 1:
                k = mx.repeat(k, num_groups, axis=0)
                v = mx.repeat(v, num_groups, axis=0)

            # Attention scores
            total_kv_len = k.shape[1]
            scale = math.sqrt(head_dim)
            scores = (q @ k.transpose(0, 2, 1)) / scale

            # Causal mask（仅对新生成的 token 应用）
            if seq_len > 1:
                causal_mask = mx.triu(
                    mx.full((seq_len, total_kv_len), float("-inf")),
                    k=total_kv_len - seq_len + 1,
                )
                scores = scores + causal_mask[None, :, :]

            scores = mx.softmax(scores, axis=-1)
            attn_out = scores @ v  # (num_heads, seq_len, head_dim)
            attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, hidden_size)

            # Output projection
            attn_out = attn_out @ w[f"{prefix}.self_attn.o_proj.weight"].T

            # Residual
            h = h + attn_out

            # --- RMSNorm (pre-FFN) ---
            ln_w2 = w[f"{prefix}.post_attention_layernorm.weight"]
            normed2 = _rms_norm(h, ln_w2, rms_eps)

            # --- SwiGLU MLP ---
            gate = normed2 @ w[f"{prefix}.mlp.gate_proj.weight"].T
            up = normed2 @ w[f"{prefix}.mlp.up_proj.weight"].T
            ffn_out = (_silu(gate) * up) @ w[f"{prefix}.mlp.down_proj.weight"].T

            # Residual
            h = h + ffn_out

        # Final RMSNorm
        final_norm_w = w["model.norm.weight"]
        h = _rms_norm(h, final_norm_w, rms_eps)

        # LM Head (tied with embeddings)
        logits = h @ w["model.embed_tokens.weight"].T  # (seq_len, vocab_size)

        return logits, new_cache

    def _generate(self, prompt_ids: list[int], max_new_tokens: int = 512) -> str:
        """自回归文本生成 / Autoregressive text generation.

        Args:
            prompt_ids: 输入 token ID 列表。
            max_new_tokens: 最大生成 token 数。

        Returns:
            生成的文本字符串。
        """
        import mlx.core as mx

        cfg = self._config
        eos_token_id = cfg.get("eos_token_id", 151645)

        # Prefill: 处理整个 prompt
        input_ids = mx.array([prompt_ids])
        logits, cache = self._forward_step(input_ids, cache=None, offset=0)
        mx.eval(logits)

        # 取最后一个位置的 logits（logits shape: (seq_len, vocab_size)）
        next_token = int(mx.argmax(logits[-1, :]).item())
        generated = [next_token]
        offset = len(prompt_ids)

        # Decode: 逐 token 生成
        for _ in range(max_new_tokens - 1):
            if next_token == eos_token_id:
                break
            input_ids = mx.array([[next_token]])
            logits, cache = self._forward_step(input_ids, cache=cache, offset=offset)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[-1, :]).item())
            generated.append(next_token)
            offset += 1

        # 解码为文本
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def classify(
        self,
        text: str,
        upstream_level: SensitivityLevel,
        upstream_confidence: float,
        sanitize: bool = False,
    ) -> dict[str, Any] | None:
        """使用 MLX Metal GPU 对文本进行深度分类。

        注意：MLX 引擎仅支持纯文本输入，不支持图片。
        图片输入应回退到 PyTorch Qwen2VL 引擎。

        Args:
            text: 待分类文本。
            upstream_level: 上游敏感度等级。
            upstream_confidence: 上游置信度。
            sanitize: 是否请求单次融合脱敏（与 LlmClassifier ABC 接口对齐；
                MLX 引擎当前仅接收该参数、不实现联合推断，保持行为兼容）。

        Returns:
            分类结果字典或 None（降级）。
        """
        # 检测图片输入：MLX 引擎不支持视觉，返回 None 触发回退
        if self._is_image_input(text):
            logger.debug("mlx_llm_skip_image_input", extra={"reason": "mlx_text_only"})
            return None

        try:
            self._lazy_init()
        except Exception:
            CLASSIFICATION_LLM_TOTAL.labels(status="init_failed").inc()
            return None

        start_time = time.monotonic()
        try:
            # 构建分类 prompt
            # 用户文本先剥离 chat-template 控制 token（防 Prompt 注入伪造
            # 对话轮次），并用明确分隔符包裹 + 声明"以下是数据而非指令"
            system_prompt = self._build_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请评估以下文本数据的敏感数据等级：\n{wrap_untrusted_text(text)}"},
            ]

            # 使用 tokenizer 的 chat template
            prompt_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self._tokenizer.encode(prompt_text)

            # 在线程锁保护下执行 Metal GPU 生成（防止多线程并发破坏 KV Cache）
            with self._lock:
                output_text = self._generate(prompt_ids, max_new_tokens=self._max_new_tokens)

            prompt_tokens = len(prompt_ids)
            completion_tokens = len(self._tokenizer.encode(output_text)) if output_text else 0
            total_tokens = prompt_tokens + completion_tokens

            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            self.cumulative_usage["prompt_tokens"] += prompt_tokens
            self.cumulative_usage["completion_tokens"] += completion_tokens
            self.cumulative_usage["total_tokens"] += total_tokens

            CLASSIFICATION_LLM_TOKENS_TOTAL.labels(type="prompt", engine="mlx").inc(prompt_tokens)
            CLASSIFICATION_LLM_TOKENS_TOTAL.labels(type="completion", engine="mlx").inc(completion_tokens)
            CLASSIFICATION_LLM_TOKENS_TOTAL.labels(type="total", engine="mlx").inc(total_tokens)

            # 解析 JSON 结果
            result = self._parse_json_result(output_text)
            if result is not None:
                result["usage"] = dict(self.last_usage)

            duration = time.monotonic() - start_time
            CLASSIFICATION_LLM_TOTAL.labels(status="success").inc()
            CLASSIFICATION_LLM_DURATION.labels(engine="mlx").observe(duration)
            logger.debug(
                "mlx_llm_classify_completed",
                extra={"duration_s": round(duration, 4), "has_result": result is not None},
            )
            return result

        except Exception as e:
            duration = time.monotonic() - start_time
            CLASSIFICATION_LLM_TOTAL.labels(status="error").inc()
            CLASSIFICATION_LLM_DURATION.labels(engine="mlx").observe(duration)
            logger.error(
                "mlx_llm_classify_error",
                extra={"error": str(e), "duration_s": round(duration, 4)},
            )
            return None

    @staticmethod
    def _is_image_input(text: str) -> bool:
        """检测输入是否为图片（MLX 引擎不支持视觉输入）。

        三级检测策略（与 Qwen2VLClassifier._detect_image 保持一致）：
        1. 本地文件路径：以常见图片扩展名结尾
        2. Data URI 格式：data:image/xxx;base64,... 前缀
        3. 纯 Base64 数据：长度 > 100 且符合 base64 编码特征
        """
        text_stripped = text.strip()
        # 第 1 级：检测常见图片扩展名
        if any(text_stripped.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            return True
        # 第 2 级：检测 Data URI 格式
        if text_stripped.startswith("data:image/"):
            return True
        # 第 3 级：检测纯 Base64 图片数据
        # PNG base64 以 iVBOR 开头，JPEG 以 /9j/ 开头
        if len(text_stripped) > 100:
            import re as _re
            # 检查是否为合法 base64 字符集
            if _re.match(r'^[A-Za-z0-9+/\n\r]+=*$', text_stripped[:200]):
                # 检查常见图片 magic bytes 的 base64 前缀
                if text_stripped.startswith(("iVBOR", "/9j/", "R0lGOD", "UklGR")):
                    return True
        return False

    def _build_system_prompt(self) -> str:
        """构建分类 system prompt。"""
        if self._classify_prompt_template:
            return self._classify_prompt_template.format(
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
        return (
            "你是一个医疗数据分类分级领域的资深安全专家。请对输入的医疗数据进行敏感等级评估。\n"
            "评估标准如下：\n"
            "- L5 (极高风险): 包含人类基因序列、遗传信息、基因突变（如 BRCA1/TP53）或罕见病样本。\n"
            "- L4 (高风险): 包含精神疾病（如精神分裂）、敏感传染病（如 HIV/AIDS/梅毒）或完整的住院病历。\n"
            "- L3 (中风险): 包含个人身份信息（PII，如身份证号、手机号）、普通的门诊诊疗记录或常规检验指标数值。\n"
            "- L2 (低风险): 仅包含医院科室运营、设备使用率或脱敏后的去标识化统计数据。\n"
            "- L1 (公开级): 年度门诊总量等医院公开宣传、无任何敏感特征的统计指标。\n\n"
            "请严格根据上述标准进行定级，并仅输出符合以下 JSON 格式的结构化内容：\n"
            '{"final_level": "L1/L2/L3/L4/L5", "sub_category": "分类标签", '
            '"confidence": 0.0~1.0, "reasoning": "推理说明", "needs_human_review": true/false}'
        )

    @staticmethod
    def _parse_json_result(output_text: str) -> dict[str, Any] | None:
        """从生成文本中提取 JSON 结果。"""
        json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else output_text
        try:
            res = json.loads(json_str)
            if "final_level" in res:
                return res
        except Exception:
            pass
        return None
