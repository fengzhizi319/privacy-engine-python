# Python 隐私引擎 (privacy-engine-python) — 接口契约规范

> **组件归属**：`services/privacy-engine-python`
> **服务协议**：REST (FastAPI) `:8079` / gRPC (grpcio) `:50051`；网关 HTTP `:8000` / gRPC `:50000`
> **与 Go 的关系**：共享行为以 `contracts/conformance/` 为准，Go 实现是事实源；Python 在此之上保留了若干历史别名与扩展端点。

---

## 1. 通用规则

| 项目 | 说明 |
|---|---|
| 编码 | 请求/响应均为 UTF-8 JSON；Arrow 端点例外（见 §3.3） |
| 认证 | `PRIVACY_AUTH_ENABLED=true` 时要求 `Authorization: Bearer <key>`；健康探针默认豁免 |
| 限流 | `PRIVACY_RATE_LIMIT_ENABLED=true` 时按身份/端点执行滑动窗口限流 |
| 错误信封 | 非 2xx 统一返回 `code/message/detail/trace_id/timestamp` 五字段 |
| 请求体上限 | 默认 32 MiB（`PRIVACY_MAX_BODY_SIZE`），超限返回 `413 PAYLOAD_TOO_LARGE` |
| 上传上限 | `/v1/privacy/process_file` 与 `/v1/pipeline/process_csv` 默认 10 MiB（`PRIVACY_MAX_UPLOAD_BYTES`） |
| 追踪 | 响应头回写 `X-Request-ID` / `X-Trace-ID`；日志自动注入 request_id |
| OpenAPI | 认证关闭时 `/docs`、`/openapi.json` 可用；`/docs/openapi.json` 始终要求 `ops:diagnostics` |

端点权限在路由层显式声明（`require_permission(...)`）。认证关闭时使用匿名身份 `["*"]`，全部放行，仅适合本地开发。

---

## 2. REST 接口清单

### 2.1 动态分类分级（前缀 `/v1/dynclassification`）

| 方法 | 路径 | Scope | 功能 |
|---|---|---|---|
| POST | `/classify` | `dynclassification:read` | 字段/记录分类分级（三层漏斗） |
| POST | `/classify/batch` | `dynclassification:read` | 批量分类 |
| POST | `/eval` | `dynclassification:read` | 单字段评估 |
| POST | `/eval_record` | `dynclassification:read` | 单记录评估 |
| POST | `/eval_table` | `dynclassification:read` | 整表评估 |
| POST | `/dry_run` | `dynclassification:read` | 规则试运行（不落盘） |
| POST | `/profiles/reload` | `dynclassification:write` | 热加载规则/画像 |
| POST | `/generate_profile` | `dynclassification:write` | 由标准生成个性化画像 |
| GET | `/standards` | `dynclassification:read` | 已加载标准列表 |
| GET | `/domains` | `dynclassification:read` | 已加载领域列表 |
| GET | `/operators` | `dynclassification:read` | 可用操作符列表 |
| GET/POST | `/validate` | `dynclassification:read` | 校验规则目录/规则文本 |

### 2.2 脱敏与哈希

| 方法 | 路径 | Scope | 功能 |
|---|---|---|---|
| POST | `/v1/privacy/mask` | `privacy:mask` | 单值字段名感知脱敏 |
| POST | `/v1/privacy/mask/record` | `privacy:mask` | 整记录脱敏 |
| POST | `/v1/privacy/mask_record` | `privacy:mask` | 历史别名（同上） |
| POST | `/v1/privacy/mask/batch` | `privacy:mask` | 批量记录脱敏 |
| POST | `/v1/privacy/mask_batch` | `privacy:mask` | 历史别名（同上） |
| POST | `/v1/privacy/mask/dataframe` | `privacy:mask` | DataFrame/表格脱敏 |
| POST | `/v1/privacy/mask_dataframe` | `privacy:mask` | 历史别名（同上） |
| POST | `/v1/privacy/hash` | `privacy:hash` | HMAC-SM3 键控哈希（兼容 SHA-256 开关） |

### 2.3 差分隐私与本地差分隐私

| 方法 | 路径 | Scope | 功能 |
|---|---|---|---|
| POST | `/v1/privacy/dp/count` | `privacy:dp` | 计数加噪（敏感度 1） |
| POST | `/v1/privacy/dp/sum` | `privacy:dp` | 求和加噪（需截断边界） |
| POST | `/v1/privacy/dp/mean` | `privacy:dp` | 均值加噪 |
| POST | `/v1/privacy/dp/histogram` | `privacy:dp` | 直方图加噪 |
| POST | `/v1/privacy/dp/noisy_count` | `privacy:dp` | 直接对已加噪计数做组合 |
| POST | `/v1/privacy/dp/noisy_sum` | `privacy:dp` | 直接对已加噪和做组合 |
| POST | `/v1/privacy/dp/noisy_mean` | `privacy:dp` | 由加噪和/计数求均值 |
| POST | `/v1/privacy/dp/noisy_histogram` | `privacy:dp` | 已加噪直方图组合 |
| POST | `/v1/privacy/dp/aggregate` | `privacy:dp` | 多规格聚合（count/sum/mean/histogram） |
| POST | `/v1/privacy/dp/vector_sum` | `privacy:dp` | 向量和（L2 截断） |
| POST | `/v1/privacy/dp/vector_mean` | `privacy:dp` | 向量均值 |
| POST | `/v1/privacy/dp/adaptive_clip` | `privacy:dp` | 自适应截断边界 |
| POST | `/v1/privacy/dp/groupby` | `privacy:dp` | 分组聚合加噪 |
| POST | `/v1/privacy/dp/chunked_count` | `privacy:dp` | 分块流式计数 |
| POST | `/v1/privacy/dp/chunked_sum` | `privacy:dp` | 分块流式求和 |
| POST | `/v1/privacy/dp/chunked_mean` | `privacy:dp` | 分块流式均值 |
| POST | `/v1/privacy/dp/chunked_histogram` | `privacy:dp` | 分块流式直方图 |
| POST | `/v1/privacy/dp/arrow_ipc` | `privacy:dp` | Arrow IPC 二进制聚合（低序列化开销） |
| POST | `/v1/privacy/ldp/randomized_response` | `privacy:dp` | 单值 randomized response |
| POST | `/v1/privacy/ldp/orr` | `privacy:dp` | 优化 randomized response（ORR） |
| POST | `/v1/privacy/ldp/perturb/binary` | `privacy:dp` | 批量二值扰动 |
| POST | `/v1/privacy/ldp/perturb/categorical` | `privacy:dp` | 批量类别扰动 |
| POST | `/v1/privacy/ldp/estimate/binary` | `privacy:dp` | 二值频率估计 |
| POST | `/v1/privacy/ldp/estimate/categorical` | `privacy:dp` | 类别直方图估计 |

### 2.4 K-匿名

| 方法 | 路径 | Scope | 功能 |
|---|---|---|---|
| POST | `/v1/privacy/k_anonymize` | `privacy:kano` | 数据集级 Mondrian K-匿名 |
| POST | `/v1/privacy/k_anonymize/record` | `privacy:kano` | 单记录自适应层级泛化 |
| POST | `/v1/privacy/k_anonymize/table` | `privacy:kano` | 整表 Mondrian K-匿名 |
| POST | `/v1/privacy/k_anonymize_table` | `privacy:kano` | 历史别名（同上） |
| POST | `/v1/privacy/k_anonymize/dataframe` | `privacy:kano` | DataFrame Mondrian K-匿名 |

### 2.5 查询混淆、预算与画像

| 方法 | 路径 | Scope | 功能 |
|---|---|---|---|
| POST | `/v1/privacy/qol/obfuscate` | `privacy:qol` | 查询混淆 + 诱饵注入 |
| POST | `/v1/privacy/qol/obfuscate/batch` | `privacy:qol` | 批量查询混淆 |
| GET | `/v1/privacy/budget` | `privacy:budget` | 查询当前命名空间剩余预算 |
| POST | `/v1/privacy/budget/reset` | `privacy:budget` | 重置预算 |
| DELETE | `/v1/privacy/budget` | `privacy:budget` | 重置预算（DELETE 别名） |
| GET/POST | `/v1/privacy/profile/recommend` | `privacy:profile` | 推荐并保存隐私参数 |

### 2.6 文件、流水线与医疗/业务处理

| 方法 | 路径 | Scope | 功能 |
|---|---|---|---|
| POST | `/v1/privacy/process_file` | `privacy:mask` | CSV/JSON/XLSX/XLS 文件脱敏或 K-匿名（≤10 MiB） |
| POST | `/v1/pipeline/process_records` | `pipeline:process` | 记录数组分类分级 + L4/L5 脱敏 |
| POST | `/v1/pipeline/process_csv` | `pipeline:process` | CSV 上传流水线（≤10 MiB） |
| POST | `/v1/agent/process` | `agent:process` 或 `medical:process` | **canonical** 业务数据评估与脱敏，含输入/输出哈希 |
| POST | `/v1/medical/sanitize` | `medical:process` | 单记录医疗脱敏（Go 契约） |
| POST | `/v1/medical/sanitize/batch` | `medical:process` | 批量医疗脱敏 |
| POST | `/v1/medical/process` | `medical:process` | 历史端点，响应头标记废弃（Sunset 2026-12-31） |

`/v1/agent/process` 与医疗端点的输入上限：单请求 500 条记录、单记录 100 字段、单字段值 100,000 字符。`process_file` 的 `operation` 仅支持 `mask_dataframe` / `k_anonymize`。

### 2.7 健康、运维与网关

| 方法 | 路径 | 认证 | 功能 |
|---|---|---|---|
| GET/POST | `/health`、`/v1/privacy/health` | 豁免（默认） | 存活 + 命名空间 |
| GET | `/livez` | 豁免（默认） | 存活探针 |
| GET | `/readyz` | 豁免（默认） | 就绪探针（检查 resolver 与预算 SQLite） |
| GET | `/readyz/llm` | 豁免（默认） | LLM 可用性探针（始终 200） |
| GET | `/v1/ops/diagnostics`、`/ops/diagnostics` | `ops:diagnostics`；`refresh=true` 另需 `ops:admin` | 依赖/模型/硬件诊断 |
| GET | `/metrics` | `ops:diagnostics`（认证开启时） | Prometheus 文本指标 |
| POST | `/v1/gateway/register` 等 5 个管理端点 | `PRIVACY_GATEWAY_ADMIN_API_KEY`（网关进程；缺省继承 metrics key） | 节点注册/注销/隔离/排空/激活 |
| GET | `:9000/health`、`:9000/readyz`、`:9000/gateway/backends`、`:9000/metrics` | 运维口 `/metrics` 需 metrics key、`/gateway/backends` 需 admin key（环回开发态可匿名） | 网关独立运维面 |

> 健康端点默认免认证、免限流由 `PRIVACY_HEALTH_NO_AUTH` / `PRIVACY_HEALTH_NO_RATE_LIMIT` 控制（默认 `true`）。关闭后按普通端点鉴权。

---

## 3. 代表性请求/响应

### 3.1 脱敏

```json
POST /v1/privacy/mask
{"field_name": "id_card", "value": "450201199001011234"}
```

```json
{"field_name": "id_card", "masked_value": "450201********1234", "field_type": "id_card"}
```

支持 `field`/`type`/`context` 双别名；`fail_mode` 等批量参数以 `engine/schemas.py` 为准。

### 3.2 差分隐私

```json
POST /v1/privacy/dp/count
{"values": [1, 1, 0, 1], "params": {"epsilon": 1.0, "delta": 0.0, "mechanism": "laplace"}}
```

```json
{"result": 3.27, "mechanism": "laplace", "epsilon": 1.0, "delta": 0.0}
```

`sum`/`mean` 必须提供 `clip_lower`/`clip_upper`；向量聚合使用 `max_norm`。预算耗尽返回 `429`（gRPC `RESOURCE_EXHAUSTED`）。

### 3.3 Arrow IPC

`POST /v1/privacy/dp/arrow_ipc?aggregation=mean&epsilon=1.0&clip_lower=0&clip_upper=100`
请求体为 `application/vnd.apache.arrow.stream`，响应为带 DP 元数据的 Arrow IPC Stream；认证仍走 Bearer 头。

### 3.4 K-匿名

```json
POST /v1/privacy/k_anonymize/table
{"rows": [{"age": 31, "zip": "450201"}, {"age": 32, "zip": "450202"}], "qi_cols": ["age", "zip"], "k": 2}
```

响应含 `records`/`result`/`k`/`group_count`。`k < 1` 按 Go 口径收敛为 2；`k=1` 在表级 Mondrian 中会被拒绝。

### 3.5 查询混淆

```json
POST /v1/privacy/qol/obfuscate
{"query": "高血压治疗", "num_dummies": 3, "domain": "medical", "seed": 42}
```

返回原始查询与诱饵查询的合并列表；`domain` 仅支持 `medical` / `generic`，诱饵数上限 64。

### 3.6 业务处理

```json
POST /v1/agent/process
{"records": [{"name": "张三", "diagnosis": "HIV"}], "api_code": "api1_yibao", "datasource_id": "ds_yibao"}
```

响应固定为 `classification_report` + `sanitized_data` + `summary`；`summary` 附加 `input_hash` / `output_hash`（SHA-256）与请求方 API/数据源标识。

### 3.7 预算

```json
GET /v1/privacy/budget
{"epsilon_remaining": 8.5, "delta_remaining": 0.0001, "epsilon_total": 10.0, "delta_total": 0.0001}
```

具体字段以 `BudgetAccountant.remaining()` 为准；重置接口返回重置后的预算快照。

---

## 4. gRPC 契约 (`proto/privacy.proto`)

包名 `privacy.local`，服务 `PrivacyService`，共 41 个 RPC：

| 分组 | 方法 |
|---|---|
| 脱敏 | `Mask` `MaskRecord` `MaskBatch` `MaskDataFrame` `Hash` |
| DP | `DPCount` `DPSum` `DPMean` `DPHistogram` `DPNoisyCount` `DPNoisySum` `DPNoisyMean` `DPNoisyHistogram` `DPChunkedCount` `DPChunkedSum` `DPChunkedMean` `DPChunkedHistogram` `DPAggregate` `DPVectorSum` `DPVectorMean` `DPAdaptiveClip` `DPGroupBy` |
| LDP | `PerturbBinaryBatch` `PerturbCategoricalBatch` `EstimateBinaryFrequency` `EstimateCategoricalHistogram` |
| K-匿名 | `KAnonymizeRecord` `KAnonymizeTable` `KAnonymizeDataFrame` |
| 查询混淆 | `ObfuscateQuery` `ObfuscateQueryBatch` |
| 分类分级 | `DynClassify` `DynEval` `DynEvalRecord` `DynStandards` `DynDomains` `DynOperators` `DynValidate` `DynGenerateProfile` |
| 运维 | `Health` `RecommendParams` |

- 消息收发上限 64 MiB（`engine/grpc_server.py::serve`）。
- 可选 `AuthInterceptor`（认证开启时）与 `RateLimitInterceptor`；未映射方法 fail-closed 到 `*`。
- `KAnonymizeRecord` 为逐行自适应泛化（历史命名），`KAnonymizeTable` 才是 Mondrian；Python 未实现 L-Diversity，与 Go 存在能力差。

---

## 5. 错误信封与状态码映射

```json
{
  "code": "INVALID_ARGUMENT",
  "message": "Invalid request parameters",
  "detail": null,
  "trace_id": "c1f7a0482b8e4a9f8f1d7201b17b6a12",
  "timestamp": "2026-10-19T09:30:00.123Z"
}
```

| HTTP | code |
|---|---|
| 400 / 422 | `INVALID_ARGUMENT` |
| 401 | `UNAUTHORIZED` |
| 403 | `FORBIDDEN` |
| 404 | `NOT_FOUND` |
| 409 | `CONFLICT` |
| 413 | `PAYLOAD_TOO_LARGE` |
| 429 | `RATE_LIMITED`（预算耗尽同为 429） |
| 500 | `INTERNAL_ERROR` |
| 503 | `UPSTREAM_UNAVAILABLE` |

gRPC 映射：`PrivacyBudgetExhausted` → `RESOURCE_EXHAUSTED`；`ValueError` → `INVALID_ARGUMENT`；其余 → `INTERNAL`。认证失败为 `UNAUTHENTICATED`，权限不足为 `PERMISSION_DENIED`。

422 默认不回传 Pydantic 字段细节；`PRIVACY_DEBUG_VALIDATION=true` 时才附带校验原因。

---

## 6. Scope 与 fail-closed 口径

- REST 路由在 `engine/routers/*.py` 中**逐路由显式**声明 Scope；`engine/security/identity.py::permission_for_rest_path` 仅为参考映射，未接线。
- gRPC 通过 `permission_for_grpc_method` 映射；未识别方法返回 `*`（最高权限），不会默认放行低权限 Key。
- `/metrics` 由 ASGI 中间件独立校验 `ops:diagnostics`；`/docs/openapi.json` 同样校验 `ops:diagnostics`。
- 动态 Key 的 `"*"` 不穿透 `audit:` 前缀（三权分立），与 Go 一致。

新增端点时必须在装饰器中同步声明 Scope，并补充对应测试；不要依赖参考映射兜底。

---

## 7. 本版本明确不支持

| 能力 | 状态 |
|---|---|
| `POST /v1/medical/process` | 已废弃，2026-12-31 后移除；请改用 `/v1/agent/process` |
| L-Diversity | Python 未实现（Go SDK 有） |
| gRPC 流式 RPC | proto 中无 streaming 方法 |
| TLCP 国密传输 | Python 进程不终结 TLCP，由 Go 终结器侧车提供，见 [`tlcp.md`](./tlcp.md) |
| 策略清单（bundle manifest） | Python 未实现，见 [`policy_bundles.md`](./policy_bundles.md) |
| 引擎独立运维端口 | 引擎仍无 9079 端口（健康/指标与业务共用 `:8079`）；网关已提供 `:9000` 独立运维口 |
