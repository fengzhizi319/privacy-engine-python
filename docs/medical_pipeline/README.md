# 医疗数据治理流水线（Medical Pipeline）

> 代码：`engine/medical_pipeline/` · REST：`/v1/medical/*`、`/v1/agent/process` · 测试：`tests/test_medical_pipeline.py`、`tests/perf/test_medical_pipeline_perf.py`

## 定位

在一次请求内完成医疗记录的**分类分级 + PII/高敏字段脱敏**，输出“分级报告 + 合规脱敏数据”双结构。它是 Go 引擎医疗链路的 Python 对齐实现，也是 app-lz / governance-portal 展示的五阶段医疗链路的后端能力来源；不包含差分隐私或 K-匿名阶段。

## 代码地图

| 文件/符号 | 职责 |
|---|---|
| `pipeline.MedicalPrivacyPipeline` | 主流水线：分类、PII 掩码、L4/L5 文本抹平、最终泄露门禁、批次级去重 |
| `pipeline.process_records(records, sanitize=True)` | 返回 `MedicalPipelineResult`（report / sanitized_data / raw_data / summary） |
| `pipeline.sanitize_text()` / `sanitize_field()` | 单文本 / 单字段抹平入口 |
| `pipeline.get_default_pipeline()` | 进程级单例（线程安全） |
| `rules.PII_FIELD_ALIASES` / `canonicalize_pii_field()` | 中英文字段别名归一（如 `身份证号 (id_card_no)`） |
| `rules.L5_TERMS_MAP` | L5 词库：`HIV_AIDS`、`PSYCHIATRIC_DISORDER`、`GENETIC_DEFECT` |
| `rules.L4_TERMS_MAP` | L4 词库：`STD_VENEREAL`、`MALIGNANT_NEOPLASM`、`HEPATITIS_VIRUS`、`SEVERE_ORGAN_DAMAGE` |
| `rules.load_redaction_strategy()` | 从 `rules/domains/medical.yaml::redaction_strategy` 加载抹平策略；缺失回退代码默认值 |
| `rules.compile_l4_l5_patterns()` | 编译带自定义替换标签的 L4/L5 正则 |
| `rules.contains_high_risk_text()` | 输出回扫门禁：检测是否仍含高风险词 |
| `samples/yibao.csv`、`kangyang.csv` | 医保 / 康养示例数据 |

## REST 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/medical/sanitize` | 单记录脱敏（Go 契约：`{record, domain}` → `{result}`） |
| POST | `/v1/medical/sanitize/batch` | 批量脱敏（`{records, domain}` → `{results}`） |
| POST | `/v1/medical/process` | **已弃用**：完整双输出；响应头 `Deprecation`、`Sunset`、`Link` 指向 `/v1/agent/process` |
| POST | `/v1/agent/process` | 规范后继路径（scope：`agent:process` 或 `medical:process`） |

`domain` 用于选择数据源语义（示例为 `yibao` / `kangyang`），未知 domain 返回 400。单请求上限：500 条记录、每记录 100 字段、单字段值 100,000 字符。

## 处理流程

1. **字段归一**：`canonicalize_pii_field` 识别中英文别名与括号组合。
2. **分类分级**：调用 `DynClassificationService` 三层漏斗，得到 `L1~L5`、标签、置信度、命中规则。
3. **PII 掩码**：身份类字段按类型掩码（复用 `engine.privacy.masking`）。
4. **L4/L5 抹平**：`redact_engine="rule"`（默认）使用编译后的复杂正则；`"ner"` 走 Small-NER。替换标签使用抽象类别（如 HIV → `IMMUNODEFICIENCY`），避免替换文本再次泄露病名。
5. **最终门禁**：对输出回扫 `contains_high_risk_text`；未抹净字段整值删除并计入 `fail_safe_triggered_fields`。
6. **汇总**：等级计数、PII 字段数、打码失败数、耗时、`guarantee_no_l4_l5_raw_data`。

`MedicalPipelineResult.classification_report` 中每条记录包含 `field_details`（`FieldClassification`：等级、安全标签、命中规则、原始值、最终脱敏值、规则结果、NER 结果）。**原始值快照会随响应返回**，调用方必须按敏感数据处置。

## 性能与缓存

- `_sanitized_cache` 与 `_ner_cache` 各上限 2048（FIFO 淘汰）；`_field_class_cache` 容量 4096。
- NER 双检模式：锁内查/写缓存、锁外推理，重复文本 0ms 命中。
- 批次级规则抹平去重表使用线程局部存储，`process_records` 结束即释放，不跨请求增长。
- 超过 50,000 字符的文本跳过复杂句法正则，避免 ReDoS 与超长耗时。
- 默认 `rule` 引擎比 NER 快得多；`PRIVACY_NER_ENABLE=true` 会显著增加单字段延迟。

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_NER_ENABLE` | `false` | `true` 时抹平引擎改为 NER |
| `PRIVACY_DYNCLASSIFICATION_RULES_DIR` | `rules` | 规则根目录；抹平策略读取 `domains/medical.yaml` |

## 测试

```bash
pytest tests/test_medical_pipeline.py -q
pytest tests/perf/test_medical_pipeline_perf.py -q
```

## 与 Go 的差异与边界

| 项 | Go 引擎 | Python 引擎 |
|---|---|---|
| 分类 + 脱敏 | 单请求完成 | 单请求完成 |
| 抹平实现 | L4/L5 词库 + 正则 | 同类词库 + 正则，可选 NER |
| 审计存证 | 可写 audit-log | **不写**；仅返回 summary |
| 数据源适配 | 医保/康养领域包 | `domain` 选择示例数据语义 |
| 图像/DICOM | DICOM 解析 | `dynclassification.image_redaction` 支持 DICOM 与图片打码 |

相关文档：[architecture.md](../architecture.md) · [dynclassification/README.md](../dynclassification/README.md) · [api.md](../api.md)
