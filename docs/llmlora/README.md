# LLM / NER 模型适配（llmlora）

> 代码：`engine/dynclassification/llm_*.py`、`ner_*.py`、`mlx_*.py`、`engine/privacy/download_model.py`、`download_ner_model.py` · 测试：`tests/dynclassification/test_llm_*`、`test_ner_*`、`test_mlx_*`

## 定位

本目录对应 Go 引擎的 `docs/llmlora/`，但 **Python 侧没有 LoRA 训练工程**：训练位于 Go 仓库的 `services/privacy-engine/model-training/llmlora/`，Python 只负责加载该训练产出的微调模型并执行推理。Python 的额外能力是 Apple Silicon MLX、TensorRT、ModelScope 与 OpenAI 兼容/vLLM HTTP 后端。

模型：`Qwen3.5-0.8B-Privacy-Classifier-Smoother`（纯文本医疗隐私分类模型），默认目录 `.models/Qwen3.5-0.8B-Privacy-Classifier-Smoother`。

## 代码地图

| 文件/符号 | 职责 |
|---|---|
| `llm_adapter.LlmAdapter` | Layer-3 适配器：`classify()`、`arbitrate()`；懒加载、进程级并发信号量、超时降级 |
| `llm_engines.Qwen3Classifier` | PyTorch/transformers 本地推理；专用推理线程池 + 180s 超时；JSON 容错解析 |
| `llm_engines.OpenAILlmClassifier` | OpenAI 兼容 HTTP 后端（vLLM/Ollama/远端 API）；15s 熔断冷却 |
| `mlx_llm_engine.MLXLlmClassifier` | Apple Silicon Metal 推理；Qwen2 文本解码器；失败回退 PyTorch |
| `ner_adapter.NerAdapter` | Layer-2 适配器：`extract()`；按 MLX → TensorRT → ONNX → ModelScope 顺序尝试 |
| `ner_engines.ONNXSmallNerEngine` | CMeEE BERT NER，ONNX Runtime（CUDA/CPU） |
| `ner_engines.TensorRTSmallNerEngine` | TensorRT FP16 加速（继承 ONNX 引擎接口） |
| `ner_engines.ModelScopeSmallNerEngine` | ModelScope RaNER 医疗模型（需 PyTorch + modelscope） |
| `mlx_ner_engine.MLXSmallNerEngine` | BERT Encoder + Linear + CRF（BIOES，37 类）的 MLX 实现 |
| `download_model.download_*` | 从 ModelScope（首选）或 Hugging Face 镜像下载 LLM |
| `download_ner_model.download_*` | 下载 CMeEE NER 的 ONNX、vocab 与 ModelScope 仓库 |

## 推理后端选择

`PRIVACY_LLM_PROVIDER`（回退 `PRIVACY_LLM_BACKEND`，默认 `auto`）：

| 值 | 行为 |
|---|---|
| `auto` | 配置了 `PRIVACY_LLM_API_BASE` / `PRIVACY_VLLM_URL` → HTTP；macOS → MLX；否则 PyTorch Qwen3 |
| `vllm` / `openai` | 强制 OpenAI 兼容 HTTP（失败即标记不可用，不继续回退） |
| `mlx` | 强制 MLX（失败即标记不可用） |
| `qwen3` / `transformers` | 强制 PyTorch 本地模型 |

NER 没有显式 provider 开关，按平台与依赖可用性自动降级；最终都失败时 `extract()` 返回空结果，分类漏斗继续使用 Layer-1。

## 关键配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_LLM_ENABLE` | `false` | 漏斗是否允许 Layer-3 深度分类 |
| `PRIVACY_LLM_ENABLE_ARBITRATION` | `true` | 规则冲突时是否启用 LLM 仲裁 |
| `PRIVACY_LLM_CONFIDENCE_THRESHOLD` | `0.75` | 低于该置信度触发 LLM |
| `PRIVACY_LLM_MODEL_PATH` | `.models/Qwen3.5-0.8B-Privacy-Classifier-Smoother` | 本地模型目录 |
| `PRIVACY_LLM_MODEL_NAME` | `Qwen/Qwen3.5-0.8B-Privacy-Classifier-Smoother` | 下载/远程模型名 |
| `PRIVACY_LLM_API_BASE` / `PRIVACY_VLLM_URL` | 空 | OpenAI 兼容服务地址 |
| `PRIVACY_LLM_API_KEY` | `EMPTY` | HTTP 后端密钥 |
| `PRIVACY_LLM_TIMEOUT` / `PRIVACY_VLM_TIMEOUT` | `180` | 推理超时（秒） |
| `PRIVACY_LLM_HTTP_WORKERS` | `16` | HTTP 后端线程池大小 |
| `PRIVACY_LLM_CB_COOLDOWN` | `15.0` | HTTP 熔断冷却（秒） |
| `PRIVACY_LLM_MAX_CONCURRENCY` | 按 CPU 推导 | 进程级推理并发上限 |
| `PRIVACY_LLM_SEMAPHORE_WAIT_SECONDS` | `30.0` | 等待推理槽位超时 |
| `PRIVACY_LLM_MIN_FREE_MEM_MB` | `512` | 低于该可用内存时不加载模型 |
| `PRIVACY_LLM_DEVICE` / `PRIVACY_DEVICE` | 自动 | `cuda` / `cpu` / `mps` |
| `PRIVACY_NER_ENABLE` | `false` | 启用 Layer-2 NER |
| `PRIVACY_MODELS_DIR` | `.models` | 模型根目录 |
| `PRIVACY_LLM_AUTO_ON_IMAGE` | 见 config.md | 图片输入自动触发视觉模型 |
| `PRIVACY_VLM_MIN_VRAM_GB` | 见 config.md | 视觉模型最低显存 |

## 下载模型

```bash
python -m engine.privacy.download_model       # LLM → .models/Qwen3.5-0.8B-Privacy-Classifier-Smoother
python -m engine.privacy.download_ner_model   # NER  → .models/raner_cmeee{,.onnx}, .models/vocab.txt
```

下载依赖 `modelscope` 或 `huggingface_hub`；两者都不可用时命令失败。模型权重不随核心镜像分发。

## 安全与降级

- **Prompt 注入防护**：用户文本经 `wrap_untrusted_text()` 包裹后再进入 prompt。
- **Prompt 与训练一致**：微调模型使用 `_FINETUNED_SYSTEM_PROMPT`，注释明确要求与 `llmlora/src/dataset/loader.py` 的训练 prompt 保持一致。
- **fail-closed 仲裁**：LLM 只能在与规则冲突等级集合内裁定；低于上游等级会被拒绝并要求人工复核。
- **降级路径**：依赖缺失、模型缺失、超时、JSON 解析失败均返回 `None`，上层使用置信度衰减，不伪造分类结果。
- **内存保护**：进程级信号量限制并发推理，避免多模型实例叠加导致 OOM；HTTP 后端另有熔断冷却。
- **模型完整性**：无摘要/签名校验，加载的是本地目录内容；发布时需在镜像或卷层保证完整性。

## 测试

```bash
pytest tests/dynclassification/test_llm_adapter.py tests/dynclassification/test_llm_engines_extended.py -q
pytest tests/dynclassification/test_ner_adapter.py tests/dynclassification/test_ner_engines_extended.py -q
pytest tests/dynclassification/test_mlx_engines.py -q          # 需要 macOS + MLX
pytest tests/dynclassification/test_real_models.py -q          # 需要本地模型
```

## 与 Go 的差异与边界

| 项 | Go 引擎 | Python 引擎 |
|---|---|---|
| LoRA 训练 | 有（`model-training/llmlora/`） | **无**，只做推理 |
| 本地 LLM | 外部 LLM / 微调模型调用约定 | PyTorch、MLX、vLLM/OpenAI 兼容 |
| NER | ONNX | ONNX / TensorRT / ModelScope / MLX |
| 熔断 | 三态熔断 | HTTP 后端简易冷却；本地推理靠信号量+超时 |
| 模型验真 | 镜像/部署层 | 无摘要校验 |

相关文档：[dynclassification/README.md](../dynclassification/README.md) · [config.md](../config.md) · [architecture.md](../architecture.md)
