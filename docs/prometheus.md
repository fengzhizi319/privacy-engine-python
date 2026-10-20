# 数盾 PrivShield · Python 隐私计算引擎 — Prometheus 监控与可观测性指南

> **组件归属**：`services/privacy-engine-python`
> **适用服务**：`engine.server` / `engine.launcher`（REST `:8079` + gRPC `:50051`）、`engine.gateway.server`（REST `:8000` + gRPC `:50000`）
> **事实源**：`engine/observability/metrics.py`（指标定义）、`engine/observability/middleware.py`（RED 埋点）、`engine/security/auth.py`（`/metrics` 鉴权）
> **配套文档**：[architecture.md](architecture.md) · [api.md](api.md) · [config.md](config.md) · [ops.md](ops.md) · [testing.md](testing.md)

---

## 1. 架构定位与监控拓扑

### 1.1 与 Go 参考实现的差异（先读）

Python 引擎在指标模型上与 Go 引擎刻意保持同名语义，但部署拓扑不同，抓取方案不能照搬 Go 文档：

| 维度 | Go `privacy-engine` | Python `privacy-engine-python` |
|---|---|---|
| 引擎运维口 | 业务口 + 1000（`:9079`） | ✅ `:9079`（业务口 +1000，路径白名单）；终结器拓扑下经终结器运维口抓取 |
| 网关运维口 | `:9000`（业务口 + 1000） | ✅ `:9000`（`PRIVACY_GATEWAY_OPS_PORT`），挂载 `/metrics` `/gateway/backends` `/readyz` `/health` |
| `/metrics` 权限 | `ops:diagnostics` | 引擎：`ops:diagnostics` 或专属指标 Key/mTLS CN；网关：`PRIVACY_GATEWAY_METRICS_API_KEY` |
| 多进程聚合 | 单进程 | `launcher` 可拉起多个 SO_REUSEPORT worker，**未启用 `PROMETHEUS_MULTIPROC_DIR`** |
| gRPC 指标 | 与 REST 同 Registry 暴露于运维口 | 引擎同 Registry 暴露于运维口；网关同 Registry 暴露于运维口 |

> ✅ **网关可观测性已闭环**：`engine/gateway/{http_proxy,grpc_proxy,balancer}.py` 更新 `privshield_gateway_*` 指标，`engine.gateway.server` 在独立运维口导出，并提供 `/gateway/backends` 拓扑快照。第 4.4 节的查询与告警可直接使用。
>
> ⚠️ **共享但未接线的指标**：`privshield_budget_consumed_total` 与 `privshield_classification_total` 已在 Python 注册，但与 Go 当前实现一样**没有生产调用点**；不要基于它们建告警。`privshield_ner_inference_seconds` 已接线。

### 1.2 双进程拓扑

```text
                    ┌──────────────────────────────────────────────┐
                    │  privacy-engine-python (:8079 REST 业务口)    │
 Prometheus ───────▶│  /metrics  (ASGI mount, 需 ops:diagnostics)   │
                    │                                              │
 REST 调用方 ───────▶│  ObservabilityMiddleware ─ RED 指标           │
 gRPC 调用方 ───────▶│  :50051 gRPC ─ 写入同一进程 Registry          │
                    │  默认 Registry: privshield_* / privacy_*      │
                    │                 + process_* / python_*        │
                    └──────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────┐
                    │  gateway (:8000 / :50000 / ops :9000)        │
 Prometheus ───────▶│  :9000/metrics (metrics key)                 │
                    │  :9000/gateway/backends (admin key)          │
                    │  :9000/readyz /health                        │
                    └──────────────────────────────────────────────┘
```

### 1.3 指标分层

| 层 | 内容 | 代表指标 |
|---|---|---|
| 传输 RED | REST/gRPC 请求量、延迟、字节流量 | `privshield_requests_total`、`privshield_request_duration_seconds`、`privshield_traffic_bytes_total` |
| 隐私原语 | 脱敏、差分隐私、K-匿名、查询混淆 | `privacy_masking_*`、`privacy_dp_*`、`privacy_kano_*`、`privacy_qol_*` |
| 分类漏斗 | 三层漏斗命中与逐层调用 | `privacy_classification_*`、`classification_*`、`privshield_ner_inference_seconds` |
| 预算资产 | 剩余 ε/δ（Gauge） | `privacy_budget_remaining`（共享累计计数 `privshield_budget_consumed_total` 当前未接线） |
| 安全 | 认证/授权/限流拒绝与鉴权耗时 | `privshield_auth_failures_total`、`privshield_auth_forbidden_total`、`privacy_auth_denials_total`、`privacy_auth_duration_seconds` |
| 网关 | 转发、在途、EWMA、熔断、节点状态 | `privshield_gateway_*`（运维口可抓取） |

---

## 2. `/metrics` 端点与安全防线

### 2.1 挂载方式

`engine/main.py` 通过 `app.mount("/metrics", ApiKeyAuthAsgiMiddleware(make_asgi_app()))` 挂载 `prometheus_client` 的标准 ASGI 应用。由于 ASGI mount **绕过 FastAPI 依赖注入**，鉴权由 `ApiKeyAuthAsgiMiddleware` 在 ASGI 层独立完成：

1. `PRIVACY_AUTH_ENABLED=false`：完全透传（本地开发）；
2. 开启认证：
   - 缺失或无效 Bearer Token → `401`，`privacy_auth_denials_total{reason="unauthenticated"}` +1；
   - 合法 Token 但缺少 `ops:diagnostics`（且无 `*`）→ `403`，`reason="forbidden"` +1；
   - 通过后进入 `prometheus_client` 的 `Gather()`。

> **信封差异**：`/metrics` 的 401/403 是 **纯文本** `text/plain`，不是其余 REST 接口的 5 字段 JSON 信封。抓取端只关心状态码；安全审计请以 `privacy_auth_denials_total` 与结构化日志为准。

### 2.2 凭据配置

静态 Key 由 JSON 环境变量注入，格式为「Token → `{name, scopes}`」：

```bash
export PRIVACY_AUTH_ENABLED=true
export PRIVACY_AUTH_INTERNAL_KEYS_JSON='{
  "psk_prom_engine_7b8c9d0e1f2a3456": {
    "name": "prometheus-privacy-engine-scraper",
    "scopes": ["ops:diagnostics"]
  }
}'
```

- Key 比对使用 `hmac.compare_digest`，并对全部 Key 做无短路遍历，避免时序侧信道。
- `*` 通配 scope 可访问 `/metrics`；但 `audit:` 前缀不被 `*` 覆盖（见 [architecture.md](architecture.md)）。
- 生产环境只给抓取凭据 `ops:diagnostics`，**不要**发放 `privacy:*`、`privacy:budget` 或 `ops:admin`。

### 2.3 健康探针豁免

`/health`、`/healthz`、`/livez`、`/readyz`、`/readyz/llm`、`/v1/privacy/health` 属于健康路径，在 `PRIVACY_HEALTH_NO_AUTH=true`（默认）时免认证、免限速；`/metrics` **不在豁免名单内**。注意 `/healthz` 仅出现在豁免判定中，Python 进程并未注册该路由，探针请使用 `/health` 或 `/livez`。

### 2.4 阶悌验证

```bash
# 1) 认证关闭（仅本地）
curl -i http://127.0.0.1:8079/metrics
# 预期 200，文本指标

# 2) 认证开启但未带 Token
curl -i http://127.0.0.1:8079/metrics
# 预期 401 Unauthorized: missing or invalid credentials

# 3) 持业务 Key（无 ops:diagnostics）
curl -i -H "Authorization: Bearer psk_low_privilege" http://127.0.0.1:8079/metrics
# 预期 403 Forbidden: insufficient scope for metrics access

# 4) 持抓取专用 Key
curl -s -H "Authorization: Bearer $PROM_TOKEN" http://127.0.0.1:8079/metrics \
  | grep -E '^privacy_(requests_total|request_duration_seconds|budget_remaining)'
```

---

## 3. 采集流水线与进程语义

### 3.1 请求处理顺序

1. `SecurityHeadersMiddleware` → 注入安全响应头；
2. `MaxBodySizeMiddleware` → `Content-Length > PRIVACY_MAX_BODY_SIZE`（默认 32 MiB）直接 413；
3. `GZipMiddleware` → ≥1 KB 响应压缩（Prometheus 支持 gzip）；
4. `CORSMiddleware`（可选）；
5. `ObservabilityMiddleware` → **`/metrics` 直接放行，不记录自身 RED 指标**，避免自引用；
6. 路由 / mounted ASGI app。

`ObservabilityMiddleware` 的 REST 标签取**路由模板**（如 `/v1/privacy/mask`），未匹配路径统一归入 `endpoint="unmatched"`，防止随机 404 扫描制造无界标签基数。gRPC 请求使用 `protocol="grpc"`，`endpoint` 为完整方法名。

### 3.2 默认 Registry 与多进程

- 所有指标注册在 `prometheus_client` 的**进程内默认 Registry**；`/metrics` 同时包含 `process_*`、`python_*` 等运行时指标。
- `engine.launcher` 在 Linux 上通过 SO_REUSEPORT 启动 `PRIVACY_WORKERS` 个独立进程（默认 `min(CPU, 8)`）。每个 worker 有**独立 Registry**，且代码未设置 `PROMETHEUS_MULTIPROC_DIR`：
  - Prometheus 每次抓取可能落到不同 worker，**计数器会来回跳变，直方图与预算 Gauge 只是单个 worker 的局部值**；
  - 需要精确全量指标时，部署侧应固定 `PRIVACY_WORKERS=1`，或按 worker 分别抓取（例如每 worker 独立端口/实例），不要直接把多 worker 聚合值当作全局真值；
  - 当前版本**不支持** `prometheus_client` 多进程聚合模式，文档不承诺该能力。

### 3.3 采样开销

- RED 埋点使用 `prometheus_client` 进程内原子操作，无网络与磁盘 I/O；
- 请求体被 `ObservabilityMiddleware` 读取以统计 `direction="request"` 字节数；FastAPI 会缓存 body，不会二次消费；
- `privshield_traffic_bytes_total{direction="response"}` 优先取 `Content-Length`，流式响应可能记为 0；
- `/metrics` 自身的抓取流量不计入业务 RED 指标。

---

## 4. 指标字典

> 所有指标均为进程启动时注册；未发生的标签组合不会出现在输出中。

### 4.1 传输 RED（共享 `privshield_*`）

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `privshield_requests_total` | Counter | `protocol`、`endpoint`、`status` | `protocol=http|grpc`；`endpoint` 取路由模板（未匹配为 `unmatched`）；`status` 为字符串状态码 |
| `privshield_request_duration_seconds` | Histogram | `protocol`、`endpoint` | Go `prometheus.DefBuckets`：`0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10` |
| `privshield_traffic_bytes_total` | Counter | `protocol`、`endpoint`、`direction` | `direction=request\|response`（Python 超集） |
| `privacy_auth_duration_seconds` | Histogram | `result` | `disabled` / `exempt` / `denied` / `success`；桶低至 `0.0001s` |
| `privshield_auth_failures_total` | Counter | `reason` | Go 同名：认证失败原因 |
| `privshield_auth_forbidden_total` | Counter | — | Go 同名：授权失败（scope/策略拒绝） |
| `privshield_auth_allowed_total` | Counter | `subject`、`permission` | Go 同名：放行计数（Python 超集接线） |
| `privacy_auth_denials_total` | Counter | `reason` | Python 原生补充：`unauthenticated` / `forbidden` / `rate_limited` / `rate_limit_storage_error` |

### 4.2 动态分类分级

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `privacy_classification_total` | Counter | `final_level`、`layer` | 漏斗最终定级次数；`layer` 取实际裁决层，未知为 `unknown` |
| `privacy_classification_duration_seconds` | Histogram | `operation` | `field` / `record` / `table` 等 |
| `privacy_classification_jobs_total` | Counter | `status` | 异步分类任务状态（`success` / `error`） |
| `privacy_classification_jobs_duration_seconds` | Histogram | `status` | 桶：`0.1 … 300s` |
| `privacy_classification_rule_hits_total` | Counter | `rule_id` | 规则引擎命中 |
| `privacy_classification_composite_hits_total` | Counter | `rule_id` | 组合规则命中 |
| `privacy_classification_ner_total` | Counter | `status` | NER 调用（`success` / `error` / `init_failed`） |
| `privacy_classification_ner_duration_seconds` | Histogram | `engine` | `mlx` 等 |
| `privacy_classification_llm_total` | Counter | `status` | LLM 调用（`success` / `error` / `timeout` / `init_failed`） |
| `privacy_classification_llm_duration_seconds` | Histogram | `engine` | `qwen3` / `vllm` / `mlx` |
| `privacy_classification_llm_tokens_total` | Counter | `type`、`engine` | `prompt` / `completion` / `total` |

动态分类子模块另有 **不带 `privacy_` 前缀** 的指标（Go 契约同名）：

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `classification_rule_hits_total` | Counter | `rule_id`、`domain`、`standard` | YAML 规则命中 |
| `classification_operator_calls_total` | Counter | `operator`、`result` | 操作符执行 |
| `classification_operator_errors_total` | Counter | `operator`、`rule_id` | 操作符错误 |
| `classification_engine_load_duration_seconds` | Histogram | `domain`、`standard` | 引擎装载耗时 |
| `classification_profile_cache_size` | Gauge | — | 引擎缓存条目数 |
| `classification_override_suppressed_total` | Counter | `domain`、`suppressed_rule_id` | 降级覆盖抑制计数 |

> 命名不一致是既成事实：`classification_*` 为跨语言契约名，`privacy_classification_*` 为 Python 侧补充指标。告警表达式必须按实际名字书写。

#### 共享分类/治理指标（`privshield_*`）

| 指标 | 类型 | 标签 | 接线状态 |
|---|---|---|---|
| `privshield_ner_inference_seconds` | Histogram | `device`、`batch_size` | ✅ ONNX / TensorRT / ModelScope / MLX 推理路径均已埋点 |
| `privshield_classification_total` | Counter | `engine`、`level`、`domain` | ⚠️ 已注册，未接线（Go 亦无生产调用点） |
| `privshield_budget_consumed_total` | Counter | `namespace`、`mechanism` | ⚠️ 已注册，未接线（Go 亦无生产调用点） |
| `privshield_api_alias_requests_total` | Counter | `alias`、`canonical`、`target`、`module` | ✅ `engine/naming.py` 标识符别名解析命中时递增 |
| `privshield_datasource_normalize_errors_total` | Counter | `reason`、`module` | ✅ 标识符归一化失败/拒绝时递增 |

### 4.3 隐私原语与预算

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `privacy_masking_operations_total` | Counter | `operation` | `mask_value`、`mask_value_batch`、`mask_record`、`mask_dataframe`、`hash_value`、`truncate`、`chunked_mask_records` |
| `privacy_masking_duration_seconds` | Histogram | `operation` | 脱敏耗时 |
| `privacy_dp_queries_total` | Counter | `mechanism`、`aggregation` | `laplace`/`gaussian` × `count`/`sum`/`mean`/`histogram`/… |
| `privacy_dp_duration_seconds` | Histogram | `aggregation`、`mechanism` | DP 计算耗时 |
| `privacy_budget_remaining` | Gauge | `namespace`、`budget_type` | `epsilon` / `delta` 剩余预算；每次记账后更新 |
| `privacy_kano_operations_total` | Counter | `operation` | `record`、`record_batch`、表级等 |
| `privacy_kano_duration_seconds` | Histogram | `operation` | K-匿名耗时 |
| `privacy_qol_operations_total` | Counter | `domain` | 查询混淆调用 |
| `privacy_qol_duration_seconds` | Histogram | `domain` | 查询混淆耗时 |
| `privacy_profile_resolve_total` | Counter | `primitive`、`status` | Profile 参数解析（`success` / `validation_failed`） |
| `privacy_data_extraction_total` | Counter | `format`、`status` | 数据适配器（`pandas_dataframe`、`arrow_ipc`、`numpy` 等） |

### 4.4 网关指标（运维口 `:9000/metrics` 可抓取）

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `privshield_gateway_requests_total` | Counter | `node_id`、`status` | 转发结果；REST 为 HTTP 状态码，gRPC 为 `200`（流正常结束）/ `502`（失败）；`node_id="aggregate"` 聚合全部转发 |
| `privshield_gateway_backend_in_flight` | Gauge | `node_id`、`backend_addr` | 每节点在途请求数 |
| `privshield_gateway_backend_ewma_latency_seconds` | Gauge | `node_id` | 每节点 EWMA 延迟（REST α=0.3 / gRPC α=0.2） |
| `privshield_gateway_circuit_breaker_state` | Gauge | `node_id`、`state` | `closed=0`、`half_open=1`、`open=2`；`state` 标签保留当前状态，便于按状态查询 |
| `privacy_gateway_latency_seconds` | Histogram | `protocol` | Python 超集：转发耗时 |
| `privacy_gateway_healthy_nodes` | Gauge | — | Python 超集：健康节点数 |
| `privacy_gateway_node_admin_state` | Gauge | `node` | Python 超集：`0=active`、`1=isolated`、`2=drained` |

抓取示例：

```yaml
scrape_configs:
  - job_name: "privshield-gateway"
    metrics_path: "/metrics"
    static_configs:
      - targets: ["127.0.0.1:9000"]
    authorization:
      type: "Bearer"
      credentials: "psk_gateway_metrics_dev_token"
```

`/gateway/backends` 返回拓扑快照（`address` / `in_flight` / `ewma_ms` / `cb_state`），适合排障时人工核验；Prometheus 告警仍以时序指标为准。

---

## 5. Prometheus 抓取配置

### 5.1 本地 / 单机

```yaml
scrape_configs:
  - job_name: "privacy-engine-python"
    metrics_path: "/metrics"
    scheme: "http"
    scrape_interval: 15s
    scrape_timeout: 10s
    static_configs:
      - targets: ["127.0.0.1:9079"]
    authorization:
      type: "Bearer"
      credentials: "psk_prom_engine_dev_token"
```

认证关闭的本地环境可删除 `authorization`。推荐抓取**运维口 `:9079`**（业务口 +1000，白名单只放行探针/指标/诊断）；`/metrics` 在业务口 `:8079` 同样可用，但运维口能避免把业务路由暴露给抓取器。

### 5.2 生产 mTLS

Python 进程本身只支持标准 TLS，TLCP 由 Go 终结器终结（见 [tlcp.md](tlcp.md)）。生产推荐抓取终结器运维口 `:9079`（`PRIVACY_OPS_TLS_STACK=both|tls` 时 mTLS），Python 回环运维口 `:19079` 不对 Pod 外暴露。抓取端必须校验证书，不得 `insecure_skip_verify`：

```yaml
scrape_configs:
  - job_name: "privacy-engine-python"
    metrics_path: "/metrics"
    scheme: "https"
    scrape_interval: 15s
    scrape_timeout: 10s
    static_configs:
      - targets: ["privacy-engine-python:9079"]
    authorization:
      type: "Bearer"
      credentials_file: "/etc/prometheus/secrets/python-engine-scrape-token"
    tls_config:
      ca_file: "/certs/ca.crt"
      cert_file: "/certs/prometheus/client.crt"
      key_file: "/certs/prometheus/client.key"
      insecure_skip_verify: false
```

也可用客户端证书替代 Bearer：引擎侧设置 `PRIVACY_METRICS_ALLOW_MTLS=true`（要求同时启用 `PRIVACY_TERMINATOR_IDENTITY_ENABLED=true`）与 `PRIVACY_METRICS_ALLOWED_CLIENT_CNS`；抓取请求经终结器验签后按 CN 放行。专属 Key 仍由 `PRIVACY_METRICS_API_KEY` / `_PREVIOUS` 配置，与业务静态 Key 完全独立。

### 5.3 Kubernetes

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: privshield-python-engine
  namespace: privshield
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: privacy-engine-python
  endpoints:
    - port: ops
      path: /metrics
      scheme: https
      interval: 15s
      scrapeTimeout: 10s
      bearerTokenSecret:
        name: python-engine-prom-token
        key: token
      tlsConfig:
        caFile: /certs/ca.crt
        certFile: /certs/prometheus/client.crt
        keyFile: /certs/prometheus/client.key
        insecureSkipVerify: false
```

若 Service 背后是多 worker 容器，先按 §3.2 的说明处理聚合口径，再对大盘数值下结论。

---

## 6. 告警与 PromQL

### 6.1 推荐告警

```yaml
groups:
  - name: PrivShield.python-engine
    rules:
      - alert: PythonEngineHighErrorRate
        expr: |
          sum(rate(privshield_requests_total{status=~"5.."}[5m]))
          /
          sum(rate(privshield_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
          service: privacy-engine-python
        annotations:
          summary: "Python 引擎 5xx 比例超过 5%"

      - alert: PythonEngineLLMFailureSpike
        expr: |
          sum(rate(privacy_classification_llm_total{status=~"error|timeout|init_failed"}[5m]))
          /
          sum(rate(privacy_classification_llm_total[5m])) > 0.2
        for: 10m
        labels:
          severity: warning
          service: privacy-engine-python
        annotations:
          summary: "LLM 分类失败率超过 20%，检查本地模型/上游 vLLM"

      - alert: PythonEngineRateLimitStorageError
        expr: increase(privacy_auth_denials_total{reason="rate_limit_storage_error"}[5m]) > 0
        for: 1m
        labels:
          severity: warning
          service: privacy-engine-python
        annotations:
          summary: "限流存储不可用，限流当前 fail-open"

      - alert: PythonEngineBudgetNearlyExhausted
        expr: privacy_budget_remaining{budget_type="epsilon"} < 1
        for: 10m
        labels:
          severity: warning
          service: privacy-engine-python
        annotations:
          summary: "命名空间 {{ $labels.namespace }} 剩余 ε < 1"
```

> 预算默认总额 ε=10.0、δ=1e-4（代码默认，可通过画像或预算 API 覆盖），阈值需结合窗口重置策略设定；`privacy_budget_remaining` 只在发生记账后更新，进程刚启动且未消费时可能没有样本。

### 6.2 常用面板查询

| 面板 | PromQL |
|---|---|
| 总 QPS | `sum(rate(privshield_requests_total[1m])) by (protocol)` |
| 成功率 | `sum(rate(privshield_requests_total{status=~"2.."}[5m])) / sum(rate(privshield_requests_total[5m]))` |
| P99 延迟 | `histogram_quantile(0.99, sum(rate(privshield_request_duration_seconds_bucket[5m])) by (le))` |
| 各端点 P95 | `histogram_quantile(0.95, sum(rate(privshield_request_duration_seconds_bucket[5m])) by (le, endpoint))` |
| NER 调用量 | `sum(rate(privacy_classification_ner_total[5m])) by (status)` |
| NER 推理延迟 | `histogram_quantile(0.95, sum(rate(privshield_ner_inference_seconds_bucket[5m])) by (le, device))` |
| LLM token 速率 | `sum(rate(privacy_classification_llm_tokens_total{type="total"}[5m])) by (engine)` |
| 预算剩余 | `privacy_budget_remaining` |
| 鉴权拒绝 | `sum(rate(privshield_auth_failures_total[5m])) by (reason)` |
| 脱敏操作分布 | `sum(rate(privacy_masking_operations_total[5m])) by (operation)` |
| 网关转发速率 | `sum(rate(privshield_gateway_requests_total{node_id!="aggregate"}[5m])) by (node_id, status)` |
| 网关在途请求 | `sum(privshield_gateway_backend_in_flight) by (node_id)` |
| 网关熔断节点 | `count(privshield_gateway_circuit_breaker_state == 2)` |

---

## 7. 排障与验证

| 现象 | 排查 |
|---|---|
| 抓取 401 | 检查 `Authorization: Bearer` 是否与 `PRIVACY_AUTH_INTERNAL_KEYS_JSON` / `EXTERNAL_KEYS_JSON` 中的 Token 一致；两个 JSON 变量解析失败会静默回退空表，合法 Key 全部失效 |
| 抓取 403 | Key 缺少 `ops:diagnostics` 且无 `*`；不要复用 `privacy:*` 业务 Key |
| 指标为空/跳变 | 多 worker 独立 Registry（§3.2）；固定单 worker 或分 worker 抓取 |
| 看不到 `privshield_gateway_*` | 抓到的是引擎口；网关指标只在 `PRIVACY_GATEWAY_OPS_PORT`（默认 `:9000`）导出 |
| 网关 `/metrics` 401 | 未带匹配 `PRIVACY_GATEWAY_METRICS_API_KEY` 的 Bearer Token |
| 网关 `/gateway/backends` 401 | 未带匹配 `PRIVACY_GATEWAY_ADMIN_API_KEY` 的 Bearer Token |
| 共享指标缺失 | `privshield_budget_consumed_total` / `privshield_classification_total` 当前未接线，属预期 |
| 看不到 `process_*` | 若未来启用多进程模式，默认 Registry 的进程指标行为会变化；当前版本未启用 |
| 指标名找不到 | 动态分类使用 `classification_*`（无 `privacy_` 前缀），见 §4.2 |

自查命令：

```bash
# 校验指标可被解析
curl -s -H "Authorization: Bearer $PROM_TOKEN" http://127.0.0.1:8079/metrics \
  | promtool check metrics

# 校验抓取目标与权限（Prometheus 侧）
promtool check config /etc/prometheus/prometheus.yml
```

相关运行态信息（版本、规则缓存、LLM 状态等）通过 `GET /v1/ops/diagnostics` 获取；`/metrics` 只承载时序数值，不替代诊断接口。
