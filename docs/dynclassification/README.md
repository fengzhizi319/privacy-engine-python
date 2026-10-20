# 动态分类分级（Dynclassification）模块

> 代码：`engine/dynclassification/` · REST：`/v1/dynclassification/*` · 测试：`tests/dynclassification/`

## 定位

三层分类分级漏斗：Layer-1 YAML 规则引擎 → Layer-2 Small-NER → Layer-3 LLM。按 `domain`（领域）与 `standard`（合规标准）动态加载规则包，输出字段级/记录级/表级分类结果，并可附带智能抹平值。规则文件为本地 YAML，没有版本清单签名机制（见 [policy_bundles.md](../policy_bundles.md)）。

## 代码地图

| 文件/符号 | 职责 |
|---|---|
| `service.DynClassificationService` | 高层入口：`classify_field` / `classify_record` / `classify_table` / `dry_run`、缓存、热重载、标准/领域/算子查询、文档转 Profile |
| `profile_loader.ProfileLoader` | 加载 `taxonomies/`、`domains/`、`standards/`；mtime 热重载；引擎实例缓存 |
| `engine.ConfigurableRuleEngine` | 规则求值（匹配器 + 权重/置信度 + 降级规则），字段级规则缓存 |
| `composite.CompositeRuleEngine` | 组合规则（字段间条件）求值 |
| `funnel.ClassificationFunnel` | 三层编排、冲突检测、置信度衰减、LLM 仲裁、fail-closed 约束 |
| `operators.py` / `operator_registry.py` | 14 个内置算子与运行时可扩展注册表 |
| `base.py` | `SensitivityLevel`、`SmallNerEngine` / `LlmClassifier` 抽象与 NoOp 实现 |
| `ner_adapter.py` / `ner_engines.py` | NER 适配层；ONNX / TensorRT / ModelScope 后端 |
| `mlx_ner_engine.py` / `mlx_llm_engine.py` | Apple Silicon MLX 后端（可选） |
| `llm_adapter.py` / `llm_engines.py` | LLM 适配层；Qwen3Classifier、OpenAI 兼容后端、vLLM |
| `models.py` / `rule_schema.py` | Pydantic 结果模型与规则/标准 Schema |
| `domain_registry.py` | 领域文本脱敏回调注册表（医疗模块注入） |
| `image_redaction.py` | DICOM 匿名化与图片路径/Base64 打码；`PRIVACY_IMAGE_ALLOWED_DIRS` 目录护栏 |
| `validator.py` / `standard_profile_generator.py` | 规则目录校验、标准 Markdown 转 YAML |

## 内置算子

`regex`、`keyword_contains`、`prefix_match`、`suffix_match`、`id_card_checksum`、`medical_card_checksum`、`icd10_range`、`luhn_checksum`、`length_range`、`exact_match`、`ip_address`、`mac_address`、`chinese_name`、`email`。

## REST 接口（前缀 `/v1/dynclassification`）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/classify` | 单字段分类 |
| POST | `/classify/batch` | 批量字段 |
| POST | `/eval`、`/eval_record`、`/eval_table` | 记录/表级评估 |
| POST | `/dry_run` | 只返回命中规则与中间结果 |
| GET | `/standards`、`/domains`、`/operators` | 运行时可用项 |
| POST/GET | `/validate` | 校验规则目录 |
| POST | `/profiles/reload` | 清空缓存并重建引擎 |
| POST | `/generate_profile` | 从 Markdown 生成规则 YAML（路径受 `_safe_path` 限制） |

读取 scope 为 `dynclassification:read`，变更/重载/生成 scope 为 `dynclassification:write`。

## 规则目录与热重载

```text
rules/
├── taxonomies/   # default.yaml、finance_jrt0197.yaml、gd_health.yaml、sc_health_db51.yaml
├── domains/      # general-pii.yaml、medical.yaml、finance.yaml、gd_health.yaml、sc_health_db51.yaml
└── standards/    # jrt0197.yaml、gd_health.yaml、sc_health_db51.yaml
```

- 环境变量 `PRIVACY_DYNCLASSIFICATION_RULES_DIR`（默认 `rules`，相对进程工作目录）。
- `PRIVACY_DYNCLASSIFICATION_HOT_RELOAD=true`（默认）时，每次取引擎前按 mtime 检查；`PRIVACY_DYNCLASSIFICATION_RELOAD_INTERVAL` 控制最小检查间隔（默认 0，即每次检查）。
- **修改已有文件**：两阶段提交——先解析到临时缓存，全部成功才原子替换；失败保留旧缓存且不记录新 mtime，下次继续重试。
- **新增/删除文件**：检测到文件集合变化后立即 `invalidate_cache()`，下次请求整体重新加载；若新文件解析失败，旧缓存已被清空，错误会在请求时暴露。
- `POST /profiles/reload` 只清缓存并返回成功，不校验新规则是否可解析；错误会在下一次请求才暴露。多 worker 部署时必须逐进程调用。
- 没有 `configuration_version` / `configuration_release` 字段，无法从运行时诊断确认当前生效版本。

## 三层漏斗要点

- **Layer-1 规则**：字段名、值、算子参数、权重与置信度；支持降级规则；求值结果有 4096 条 LRU 缓存（键含字段名与值摘要）。
- **Layer-2 NER**：`PRIVACY_NER_ENABLE=true` 时启用；模型缺失/依赖缺失时降级，不阻塞分类。
- **Layer-3 LLM**：规则冲突时仲裁，或置信度低于阈值时深度分类；LLM 不可用/超时/JSON 解析失败时走置信度衰减。
- **fail-closed 仲裁**：LLM 裁定等级必须落在冲突等级集合内；低于上游等级时拒绝降级并要求人工复核。
- **结果字段**：`final_level`、`confidence`、`engine_layer`、`needs_human_review`、`reasoning`、`sanitized_value`、`suppressed_tags`，以及 `AuditInfo`（domain、standard、规则版本、耗时）。
- **统一缓存**：`PRIVACY_CLASSIFICATION_CACHE_SIZE`（默认 10000）控制 `HighConcurrencyLRUCache` 容量。

## 重要边界

- **P0 隐私风险**：分类结果缓存键包含原始字段值（长值取前 100 字符 + 长度 + `hash(value)`），缓存对象本身也保存 `field_value` 原值；进程内存中会留存敏感明文。生产应缩短 TTL/降低容量，或改造为仅缓存摘要（见 [算法优化.md](../算法优化.md)）。
- **无策略清单/签名**：规则目录没有整包摘要、路径约束或发布版本绑定；磁盘上的任何 YAML 变更都会在热重载后生效。
- **手动重载不验证**：`profiles/reload` 与 `validate` 是分离操作，发布流程必须先校验再逐进程重载。
- **图片输入**：路径输入受允许目录约束；Base64 输入无目录概念但受大小/类型检查。
- **多进程缓存独立**：每 worker 各自持有规则缓存与模型实例；`engine.launcher --warmup` 可在 fork 前预加载适配器以共享只读页。

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_DYNCLASSIFICATION_RULES_DIR` | `rules` | 规则根目录 |
| `PRIVACY_DYNCLASSIFICATION_HOT_RELOAD` | `true` | mtime 热重载 |
| `PRIVACY_DYNCLASSIFICATION_RELOAD_INTERVAL` | `0` | 最小检查间隔（秒） |
| `PRIVACY_CLASSIFICATION_CACHE_SIZE` | `10000` | 分类结果 LRU 容量 |
| `PRIVACY_NER_ENABLE` | `false` | 启用 Layer-2 NER |
| `PRIVACY_IMAGE_ALLOWED_DIRS` | `data/uploads/samples/medical_images/tmp` | 图片路径白名单（`os.pathsep` 分隔） |
| `PRIVACY_MODELS_DIR` | `.models` | 本地模型根目录 |
| `PRIVACY_LLM_*` / `PRIVACY_VLM_*` | 见 [config.md](../config.md) | LLM 后端、模型、超时、并发与回退 |

## 测试

```bash
pytest tests/dynclassification -q
pytest tests/dynclassification/test_mlx_engines.py -q   # 仅 macOS/MLX 环境
```

## 与 Go 的差异与边界

| 项 | Go 引擎 | Python 引擎 |
|---|---|---|
| 规则引擎 | AC 自动机 + Regex | 正则/关键字算子逐条求值 + 缓存 |
| 策略清单 | `PRIVACY_CONFIG_MANIFEST_FILE` 整包验真 | **无** |
| NER | ONNX | ONNX / TensorRT / ModelScope / MLX |
| LLM | 外部 LLM + 三态熔断 | Qwen3 / OpenAI 兼容 / vLLM / MLX |
| 热重载 | 清单 + 原子快照 | mtime 清缓存 |
| 配置版本诊断 | `configuration_version` | 无 |

相关文档：[architecture.md](../architecture.md) · [policy_bundles.md](../policy_bundles.md) · [api.md](../api.md) · [llmlora/README.md](../llmlora/README.md)
