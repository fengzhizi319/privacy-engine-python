"""基于 Apple MLX (Metal GPU) 的本地轻量级命名实体识别引擎。

中文说明：
使用 Apple MLX 框架在 macOS Metal GPU 上执行 BERT NER 推理，
相比 ONNX Runtime CPU 推理可显著降低延迟（Apple Silicon 加速）。

模型结构：BERT Encoder → Linear(768→37) → CRF (Viterbi 解码)
标签体系：BIOES（Begin/Inside/Outside/End/Single）共 37 类

降级策略：
- mlx 未安装或非 macOS → 抛出 ImportError → 回退到 ONNX/ModelScope
- 模型文件不存在 → 抛出 FileNotFoundError → 回退

English Description:
Local lightweight NER engine using Apple MLX framework for Metal GPU inference on macOS.
Provides significantly lower latency compared to ONNX Runtime CPU inference on Apple Silicon.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

from ..observability.logging_config import get_logger
from ..observability.metrics import (
    CLASSIFICATION_NER_DURATION,
    CLASSIFICATION_NER_TOTAL,
)
from .base import SmallNerEngine
from .ner_engines import DEFAULT_NER_LABEL_MAPPING, SimpleChineseBertTokenizer, observe_ner_inference

logger = get_logger(__name__)

# BIOES 标签索引映射（与 raner_cmeee config.json 中 id2label 一致）
_BIOES_LABEL_MAP: dict[int, str] = {
    0: "O",
    1: "B-bod", 2: "S-bod",
    3: "B-dep", 4: "S-dep",
    5: "B-dis", 6: "S-dis",
    7: "B-dru", 8: "S-dru",
    9: "B-equ", 10: "S-equ",
    11: "B-ite", 12: "S-ite",
    13: "B-mic", 14: "S-mic",
    15: "B-pro", 16: "S-pro",
    17: "B-sym", 18: "S-sym",
    19: "I-bod", 20: "E-bod",
    21: "I-dep", 22: "E-dep",
    23: "I-dis", 24: "E-dis",
    25: "I-dru", 26: "E-dru",
    27: "I-equ", 28: "E-equ",
    29: "I-ite", 30: "E-ite",
    31: "I-mic", 32: "E-mic",
    33: "I-pro", 34: "E-pro",
    35: "I-sym", 36: "E-sym",
}


def _viterbi_decode(
    emissions: list[list[float]],
    start_transitions: list[float],
    end_transitions: list[float],
    transitions: list[list[float]],
    seq_len: int,
) -> list[int]:
    """Viterbi 解码 CRF 层 / Viterbi decode for CRF layer.

    使用动态规划找到最优标签序列。
    Uses dynamic programming to find the optimal label sequence.

    Args:
        emissions: 发射分数 (seq_len, num_labels)。
        start_transitions: 起始转移分数 (num_labels,)。
        end_transitions: 结束转移分数 (num_labels,)。
        transitions: 转移分数矩阵 (num_labels, num_labels)。
        seq_len: 有效序列长度。

    Returns:
        最优标签索引序列。
    """
    num_labels = len(start_transitions)
    NEG_INF = float("-inf")

    # 初始化：score[j] = start_transitions[j] + emissions[0][j]
    score = [start_transitions[j] + emissions[0][j] for j in range(num_labels)]
    history: list[list[int]] = []

    for t in range(1, seq_len):
        new_score = [NEG_INF] * num_labels
        backptr = [0] * num_labels
        for j in range(num_labels):
            # 找使 score[i] + transitions[i][j] 最大的 i
            best_val = NEG_INF
            best_i = 0
            emit = emissions[t][j]
            for i in range(num_labels):
                val = score[i] + transitions[i][j]
                if val > best_val:
                    best_val = val
                    best_i = i
            new_score[j] = best_val + emit
            backptr[j] = best_i
        score = new_score
        history.append(backptr)

    # 终止：加上 end_transitions
    final_score = [score[j] + end_transitions[j] for j in range(num_labels)]
    best_last = max(range(num_labels), key=lambda j: final_score[j])

    # 回溯
    path = [best_last]
    for t in range(len(history) - 1, -1, -1):
        path.append(history[t][path[-1]])
    path.reverse()
    return path


class MLXSmallNerEngine(SmallNerEngine):
    """基于 Apple MLX 的本地医疗 NER 模型推理引擎 / MLX Metal Medical NER Engine.

    使用 MLX 框架加载转换后的 BERT NER 模型，在 Apple Silicon Metal GPU 上执行推理。
    模型结构：BERT → Linear → CRF (Viterbi)。

    特性：
    - Metal GPU 加速：利用 Apple Silicon 统一内存架构实现低延迟推理
    - 延迟加载：首次调用 extract() 时才加载模型
    - 自动降级：mlx 未安装或模型不存在时抛出异常

    English Description:
    Loads converted BERT NER model via Apple MLX framework for Metal GPU inference
    on Apple Silicon. Features lazy-loading and graceful degradation.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        vocab_path: str | None = None,
        label_mapping: dict[str, str] | None = None,
    ):
        """初始化 MLX NER 引擎 / Initialize MLX NER Engine.

        Args:
            model_dir: MLX 模型目录（含 weights.safetensors 和 config.json）。
            vocab_path: vocab.txt 词表文件路径。
            label_mapping: 原始标签→标准标签映射。
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        self.model_dir = model_dir or os.path.join(project_root, ".models", "raner_cmeee-mlx")
        self.vocab_path = vocab_path or os.path.join(self.model_dir, "vocab.txt")
        self.label_mapping = label_mapping or DEFAULT_NER_LABEL_MAPPING

        self._weights: dict[str, Any] | None = None
        self._config: dict[str, Any] | None = None
        self.tokenizer: SimpleChineseBertTokenizer | None = None
        self._initialized = False
        self._init_error: Exception | None = None

    def _lazy_init(self) -> None:
        """延迟加载 MLX 模型权重 / Lazy-load MLX model weights."""
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
            if not os.path.exists(self.vocab_path):
                raise FileNotFoundError(f"未找到词表文件: {self.vocab_path}")

            # 加载权重和配置
            self._weights = dict(mx.load(weights_path))
            with open(config_path, encoding="utf-8") as f:
                self._config = json.load(f)

            # 初始化分词器
            self.tokenizer = SimpleChineseBertTokenizer(self.vocab_path)
            self._initialized = True

            logger.info(
                "mlx_ner_engine_initialized",
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
                "mlx_ner_engine_init_failed",
                extra={"error": str(e), "model_dir": self.model_dir},
            )
            raise e

    def _bert_forward(self, input_ids: list[int], attention_mask: list[int], token_type_ids: list[int]) -> Any:
        """执行 BERT 前向传播（MLX Metal GPU）/ Run BERT forward pass on MLX Metal GPU.

        Args:
            input_ids: 输入 token ID 序列 (seq_len,)。
            attention_mask: 注意力掩码 (seq_len,)。
            token_type_ids: token 类型 ID (seq_len,)。

        Returns:
            BERT 最后一层隐藏状态 (seq_len, hidden_size)。
        """
        import mlx.core as mx

        w = self._weights
        cfg = self._config
        hidden_size = cfg["hidden_size"]
        num_heads = cfg["num_attention_heads"]
        num_layers = cfg["num_hidden_layers"]
        head_dim = hidden_size // num_heads

        # === Embeddings ===
        ids = mx.array(input_ids)
        tids = mx.array(token_type_ids)
        seq_len = len(input_ids)
        positions = mx.arange(seq_len)

        word_emb = w["encoder.embeddings.word_embeddings.weight"][ids]
        pos_emb = w["encoder.embeddings.position_embeddings.weight"][positions]
        type_emb = w["encoder.embeddings.token_type_embeddings.weight"][tids]

        hidden = word_emb + pos_emb + type_emb
        # LayerNorm
        ln_w = w["encoder.embeddings.LayerNorm.weight"]
        ln_b = w["encoder.embeddings.LayerNorm.bias"]
        mean = mx.mean(hidden, axis=-1, keepdims=True)
        var = mx.var(hidden, axis=-1, keepdims=True)
        hidden = (hidden - mean) / mx.sqrt(var + 1e-12) * ln_w + ln_b

        # === Transformer Layers ===
        mask = mx.array(attention_mask, dtype=mx.float32)
        # 构建 attention mask: (1, seq_len) 用于广播到 (num_heads, seq_len, seq_len)
        attn_mask = (1.0 - mask) * (-1e9)  # (seq_len,)

        for layer_idx in range(num_layers):
            prefix = f"encoder.encoder.layer.{layer_idx}"

            # --- Multi-Head Self-Attention ---
            q_w = w[f"{prefix}.attention.self.query.weight"]
            q_b = w[f"{prefix}.attention.self.query.bias"]
            k_w = w[f"{prefix}.attention.self.key.weight"]
            k_b = w[f"{prefix}.attention.self.key.bias"]
            v_w = w[f"{prefix}.attention.self.value.weight"]
            v_b = w[f"{prefix}.attention.self.value.bias"]

            # Linear projections: (seq_len, hidden_size) @ (hidden_size, hidden_size)^T
            q = hidden @ q_w.T + q_b
            k = hidden @ k_w.T + k_b
            v = hidden @ v_w.T + v_b

            # Reshape to (seq_len, num_heads, head_dim) → (num_heads, seq_len, head_dim)
            q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
            k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
            v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

            # Attention scores: (num_heads, seq_len, seq_len)
            scale = math.sqrt(head_dim)
            scores = (q @ k.transpose(0, 2, 1)) / scale
            # 应用 attention mask: 广播 (seq_len,) → (1, 1, seq_len)
            scores = scores + attn_mask.reshape(1, 1, seq_len)
            # Softmax
            scores = mx.softmax(scores, axis=-1)
            # Weighted sum
            attn_out = scores @ v  # (num_heads, seq_len, head_dim)
            attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, hidden_size)

            # Attention output dense + LayerNorm
            ao_w = w[f"{prefix}.attention.output.dense.weight"]
            ao_b = w[f"{prefix}.attention.output.dense.bias"]
            attn_out = attn_out @ ao_w.T + ao_b

            # Residual + LayerNorm
            hidden = hidden + attn_out
            ln_w = w[f"{prefix}.attention.output.LayerNorm.weight"]
            ln_b = w[f"{prefix}.attention.output.LayerNorm.bias"]
            mean = mx.mean(hidden, axis=-1, keepdims=True)
            var = mx.var(hidden, axis=-1, keepdims=True)
            hidden = (hidden - mean) / mx.sqrt(var + 1e-12) * ln_w + ln_b

            # --- Feed-Forward Network ---
            inter_w = w[f"{prefix}.intermediate.dense.weight"]
            inter_b = w[f"{prefix}.intermediate.dense.bias"]
            inter = hidden @ inter_w.T + inter_b
            # GELU activation
            inter = inter * mx.sigmoid(1.702 * inter)

            out_w = w[f"{prefix}.output.dense.weight"]
            out_b = w[f"{prefix}.output.dense.bias"]
            ff_out = inter @ out_w.T + out_b

            # Residual + LayerNorm
            hidden = hidden + ff_out
            ln_w = w[f"{prefix}.output.LayerNorm.weight"]
            ln_b = w[f"{prefix}.output.LayerNorm.bias"]
            mean = mx.mean(hidden, axis=-1, keepdims=True)
            var = mx.var(hidden, axis=-1, keepdims=True)
            hidden = (hidden - mean) / mx.sqrt(var + 1e-12) * ln_w + ln_b

        return hidden

    def _parse_bioes_tags(self, tokens: list[str], label_indices: list[int], probs: list[float]) -> list[dict[str, Any]]:
        """解析 BIOES 序列标注 / Parse BIOES Sequence Labels.

        BIOES 标注方案：
        - B-XXX: 实体起始 (Begin)
        - I-XXX: 实体内部 (Inside)
        - E-XXX: 实体结束 (End)
        - S-XXX: 单字实体 (Single)
        - O: 非实体 (Outside)
        """
        entities: list[dict[str, Any]] = []
        current_entity: dict[str, Any] | None = None

        for idx in range(1, len(tokens)):
            token = tokens[idx]
            if token in ("[SEP]", "[PAD]"):
                break

            label_idx = label_indices[idx]
            prob = probs[idx]
            tag = _BIOES_LABEL_MAP.get(label_idx, "O")

            if tag == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif tag.startswith("S-"):
                # 单字实体
                if current_entity:
                    entities.append(current_entity)
                ent_type = tag.split("-")[1]
                entities.append({
                    "text": token,
                    "label": ent_type,
                    "confidence": prob,
                })
                current_entity = None
            elif tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                ent_type = tag.split("-")[1]
                current_entity = {
                    "text": token,
                    "label": ent_type,
                    "confidence": prob,
                }
            elif tag.startswith("I-") and current_entity:
                ent_type = tag.split("-")[1]
                if ent_type == current_entity["label"]:
                    current_entity["text"] += token
                    current_entity["confidence"] = min(current_entity["confidence"], prob)
                else:
                    entities.append(current_entity)
                    current_entity = None
            elif tag.startswith("E-") and current_entity:
                ent_type = tag.split("-")[1]
                if ent_type == current_entity["label"]:
                    current_entity["text"] += token
                    current_entity["confidence"] = min(current_entity["confidence"], prob)
                    entities.append(current_entity)
                    current_entity = None
                else:
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    @staticmethod
    def _chunk_text(text: str, max_chunk_len: int = 120) -> list[str]:
        """按句号/分号/换行智能分句；超长单句采用 120 字符带 20 字符重叠的滑动窗口切片。"""
        import re
        if not text or len(text) <= max_chunk_len:
            return [text]

        chunks: list[str] = []
        raw_sentences = [s for s in re.split(r"(?<=[。；;\n\r])", text) if s]
        curr_chunk = ""

        for s in raw_sentences:
            if len(s) > max_chunk_len:
                if curr_chunk:
                    chunks.append(curr_chunk)
                    curr_chunk = ""
                step = max(max_chunk_len - 20, 10)
                for i in range(0, len(s), step):
                    chunks.append(s[i:i + max_chunk_len])
            elif len(curr_chunk) + len(s) <= max_chunk_len:
                curr_chunk += s
            else:
                chunks.append(curr_chunk)
                curr_chunk = s

        if curr_chunk:
            chunks.append(curr_chunk)

        return chunks if chunks else [text]

    def _extract_single_chunk(self, text: str, max_len: int = 128) -> list[dict[str, Any]]:
        """单块文本的 MLX Metal GPU NER 推理。"""
        import mlx.core as mx
        assert self.tokenizer is not None and self._weights is not None

        input_ids, attention_mask, token_type_ids = self.tokenizer.encode(text, max_len=max_len)

        # BERT 前向传播（Metal GPU）
        hidden = self._bert_forward(input_ids, attention_mask, token_type_ids)

        # Linear classifier: (seq_len, 768) → (seq_len, 37)
        cls_w = self._weights["linear.weight"]
        cls_b = self._weights["linear.bias"]
        logits = hidden @ cls_w.T + cls_b  # (seq_len, 37)

        # 计算有效序列长度
        tokens = ["[CLS]", *self.tokenizer.tokenize(text)[:max_len - 2], "[SEP]"]
        seq_len = len(tokens)

        # 发射分数（取有效序列部分）
        logits_np = logits[:seq_len]
        mx.eval(logits_np)
        emissions = logits_np.tolist()

        # CRF Viterbi 解码
        crf_start = self._weights["crf.start_transitions"]
        crf_end = self._weights["crf.end_transitions"]
        crf_trans = self._weights["crf.transitions"]
        mx.eval(crf_start, crf_end, crf_trans)

        label_indices = _viterbi_decode(
            emissions=emissions,
            start_transitions=crf_start.tolist(),
            end_transitions=crf_end.tolist(),
            transitions=crf_trans.tolist(),
            seq_len=seq_len,
        )

        # 计算每个位置的 softmax 概率（用于置信度）
        import numpy as np
        logits_arr = np.array(emissions)
        exp_logits = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
        probs_arr = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        token_probs = [float(probs_arr[i, label_indices[i]]) for i in range(seq_len)]

        # 解析 BIOES 标签
        entities = self._parse_bioes_tags(tokens, label_indices, token_probs)

        # 映射标签到统一标准类别
        for ent in entities:
            raw_label = ent["label"]
            if raw_label in self.label_mapping:
                ent["label"] = self.label_mapping[raw_label]

        return entities

    def extract(self, text: str) -> list[dict[str, Any]]:
        """提取输入文本中的医疗实体（MLX Metal GPU 推理，支持超长文本分句切片）。

        Args:
            text: 目标文本片段。

        Returns:
            命名实体字典列表，每个字典含 text/label/confidence。
        """
        try:
            self._lazy_init()
        except Exception:
            CLASSIFICATION_NER_TOTAL.labels(status="init_failed").inc()
            return []

        start_time = time.monotonic()
        try:
            max_chunk_len = 120
            if len(text) > max_chunk_len:
                chunks = self._chunk_text(text, max_chunk_len=max_chunk_len)
                merged_entities: list[dict[str, Any]] = []
                seen_keys: set[tuple[str, str]] = set()

                for chunk in chunks:
                    chunk_entities = self._extract_single_chunk(chunk, max_len=128)
                    for ent in chunk_entities:
                        key = (ent["text"], ent["label"])
                        if key not in seen_keys:
                            seen_keys.add(key)
                            merged_entities.append(ent)

                duration = time.monotonic() - start_time
                CLASSIFICATION_NER_TOTAL.labels(status="success").inc()
                CLASSIFICATION_NER_DURATION.labels(engine="mlx").observe(duration)
                observe_ner_inference(duration, "mps")
                logger.debug(
                    "mlx_ner_extract_completed",
                    extra={"entity_count": len(merged_entities), "duration_s": round(duration, 4)},
                )
                return merged_entities

            entities = self._extract_single_chunk(text, max_len=128)
            duration = time.monotonic() - start_time
            CLASSIFICATION_NER_TOTAL.labels(status="success").inc()
            CLASSIFICATION_NER_DURATION.labels(engine="mlx").observe(duration)
            observe_ner_inference(duration, "mps")
            logger.debug(
                "mlx_ner_extract_completed",
                extra={"entity_count": len(entities), "duration_s": round(duration, 4)},
            )
            return entities

        except Exception as e:
            duration = time.monotonic() - start_time
            CLASSIFICATION_NER_TOTAL.labels(status="error").inc()
            CLASSIFICATION_NER_DURATION.labels(engine="mlx").observe(duration)
            observe_ner_inference(duration, "mps")
            logger.warning(
                "mlx_ner_extract_error",
                extra={"error": str(e), "duration_s": round(duration, 4)},
            )
            return []
