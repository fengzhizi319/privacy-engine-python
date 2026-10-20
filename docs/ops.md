# Python 隐私引擎 (privacy-engine-python) — 生产运维与故障自愈指南

> **组件归属**：`services/privacy-engine-python`
> **默认端口**：REST `8079` / gRPC `50051` / 运维口 `9079`；网关 REST `8000` / gRPC `50000` / 运维口 `9000`
> **与 Go 的差异**：Python 引擎已提供独立运维口（业务口 +1000），仅放行探针、`/metrics`、`/ops/diagnostics` 与 `/debug/pprof/`；无真实 pprof profile、无策略清单。TLCP 由 [`terminator/`](../terminator/README.md) 的 Go 终结器提供。

---

## 1. 启停方式

| 模式 | 命令 | 适用场景 | 说明 |
|---|---|---|---|
| 单进程双协议 | `python -m engine.server` | 开发、小流量 | REST 在守护线程，gRPC 线程池，运维口独立线程；信号统一关闭 |
| 多进程双协议 | `python -m engine.launcher --workers 4` | 多核生产 | Linux SO_REUSEPORT，每 worker 独立 REST+gRPC+ops |
| 预热后 fork | `python -m engine.launcher --warmup --workers 4` | 大模型冷启动敏感 | fork 前加载模型，依靠 CoW 共享权重 |
| 仅 gRPC | `python -m engine.grpc_server --host 0.0.0.0 --port 50051` | 纯 RPC 场景 | 独立信号处理（无 ops 口） |
| 仅 REST | `uvicorn engine.main:app --host 0.0.0.0 --port 8079` | 调试 | 不启动 gRPC 与 ops |
| 网关 | `python -m engine.gateway.server` | 前置代理 | 独立 REST `:8000` + gRPC `:50000` + 运维口 `:9000` |

Makefile：`make install-dev`、`make test`、`make test-cov`、`make lint`、`make typecheck`、`make check`、
`make docker-core`、`make docker-ml`、`make terminator-check`、`make docker-terminator`。

### 环境引导

启动时 `load_env_file()` 会加载 `.env` 与 `config/env/<profile>.env`（见 [`config.md`](./config.md) §1）。容器内 entrypoint 额外将 `PRIVACY_LOG_FORMAT` 默认为 `json`、监听地址默认为 `0.0.0.0`。

---

## 2. 线程/进程模型

```text
engine.server（单进程）
├── 主线程：信号监听 + 优雅关闭调度
├── uvicorn-rest-server（daemon 线程）：FastAPI REST
└── gRPC 线程池（PRIVACY_GRPC_MAX_WORKERS=64）
    └── 每个 RPC 一个 worker；阻塞型隐私计算在此执行

engine.launcher（多进程）
├── launcher 主进程：fork N 个 worker + 信号转发
└── worker × N
    ├── REST 线程（非 daemon）
    └── gRPC 线程池
```

- 单进程模式下 REST 线程为 `daemon=True`，即使 join 超时也不会阻止进程退出。
- gRPC 线程池大小由 `PRIVACY_GRPC_MAX_WORKERS` 控制；低于 1 直接拒绝启动。
- 多进程模式仅在 Linux 启用 `SO_REUSEPORT`；macOS/Windows 会退化为普通绑定语义。

---

## 3. 优雅关闭

```text
SIGTERM / SIGINT
  → shutdown_event.set()
  → gRPC stop(grace=5)，最多等待 10s 排空在途 RPC
  → REST should_exit=True，join 10s（超时告警后随主进程退出）
  → sys.exit(0)
```

网关进程额外：

```text
取消 health_check_loop 并 await 其退出
→ 三个监听面（REST / gRPC / 运维口）并发退出
→ gRPC stop(grace=1.0)
→ balancer.close_all() 释放所有回源通道
→ 日志 "Gateway services safely stopped."
```

注意事项：

- 单进程模式下 `asyncio.gather` 中任一服务退出会取消另一服务；请通过信号关闭。
- 多进程 launcher 由主进程负责向 worker 传播信号并等待回收。
- 如果使用 `docker stop`，确保 `stop_grace_period` 大于 gRPC grace + REST join（建议 ≥ 20s）。

---

## 4. 健康检查与就绪探针

| 端点 | 监听 | 行为 | 关键响应 |
|---|---|---|---|
| `GET/POST /health`、`/v1/privacy/health` | 业务口 + 运维口 | 固定存活 | `{"status":"ok","engine":"python","namespace":"default"}` |
| `GET /livez` | 业务口 + 运维口 | 固定存活 | `{"status":"ok"}` |
| `GET /readyz` | 业务口 + 运维口 | 检查 `service.resolver` 与 `PRIVACY_BUDGET_DB`（如设置） | `{"status":"ok","tls_stack":"<env 原样>"}`；失败 503 |
| `GET /readyz/llm` | 业务口 + 运维口 | 探测 LLM 适配器 | 始终 200：`{"status":"ready","llm":"available|unavailable"}` |
| `GET /metrics` | 业务口 + 运维口 | Prometheus 指标；需 `ops:diagnostics` 或 `PRIVACY_METRICS_ALLOW_MTLS` | 文本指标 |
| `GET /ops/diagnostics` | 业务口 + 运维口 | 引擎/依赖/模型诊断 | JSON |
| `/debug/pprof/*` | 仅运维口白名单 | 同名诊断入口（不是 Go profile 格式） | 由应用路由决定 |

- 运维口默认 `REST_PORT + 1000`（`PRIVACY_OPS_PORT` 可覆盖），只放行上表路径，其余 404。
- 运维口与业务口共享同一份中间件与鉴权栈；不存在「运维口少一道门」。
- 探针默认免认证/免限流（`PRIVACY_HEALTH_NO_AUTH` / `PRIVACY_HEALTH_NO_RATE_LIMIT`）。
- `/readyz` 的 `tls_stack` **不是实际协议栈**，只是回显 `PRIVACY_TLS_STACK`。进程内只有标准 TLS；
  `tlcp|both` 由前置 Go 终结器终结。
- 预算 SQLite 损坏、锁死或不可读会导致 `/readyz` 503；这是有意的 fail-closed 行为。
- 没有 `/readyz` 的深度规则校验端点；规则加载状态请用 `/v1/ops/diagnostics` 或 `/v1/dynclassification/standards`。

### Docker HEALTHCHECK

镜像默认每 30s 执行：`curl -sf http://localhost:8079/health` + gRPC `:50051` TCP 连通性检查。
终结器拓扑中 Python 监听回环 `18079/15051`，必须在 compose/K8s 中覆盖探针（示例已覆盖）；
K8s 部署应使用 `/livez` 与 `/readyz`。

### 终结器探针

| 端点 | 监听 | 行为 |
|---|---|---|
| `GET /health` `/healthz` `/livez` | 终结器运维口 `:9079` | 本地存活；排空期间 503 |
| `GET /readyz` | 终结器运维口 `:9079` | 聚合检查 Python ops/REST `/readyz` 与 gRPC TCP；未就绪返回 503 |
| `/metrics` `/ops/diagnostics` `/debug/pprof/` | 终结器运维口 `:9079` | 原样代理到 Python 运维口 |

### 网关探针

| 端点 | 监听 | 行为 |
|---|---|---|
| `GET /health` | 业务口 `:8000` | 网关本地存活，不转发后端 |
| `GET /health` | 运维口 `:9000` | 同上 |
| `GET /readyz` | 运维口 `:9000` | REST 池无健康节点时 503；返回 `strategy`、`tls_stack`、`metrics_auth_mode` 与健康计数 |
| `GET /gateway/backends` | 运维口 `:9000` | 后端拓扑；配置 admin key 时需 Bearer |
| `GET /metrics` | 运维口 `:9000` | Prometheus；配置 metrics key 时需 Bearer |

---

## 5. 运维诊断

```bash
curl -s http://127.0.0.1:8079/v1/ops/diagnostics | jq
curl -s -X GET 'http://127.0.0.1:8079/v1/ops/diagnostics?refresh=true' \
  -H "Authorization: Bearer $OPS_ADMIN_KEY"
```

返回结构：

| 字段 | 内容 |
|---|---|
| `service` | 名称、命名空间、Python 版本、REST/gRPC 端口 |
| `engines.ner` | 实际激活引擎（MLX/TensorRT/ONNX/ModelScope）、依赖、模型路径、原因 |
| `engines.llm` | 后端、可用性判定来源（runtime/probe/static）、模型与依赖 |
| `dependencies` | onnxruntime/torch/transformers/modelscope/tensorrt/mlx 安装状态与安装指引 |
| `models` | NER ONNX/vocab/ModelScope/Qwen2-VL 模型文件是否存在 |
| `hardware` | 平台、架构、`nvidia-smi`、CUDA 状态（避免为诊断首次导入 torch） |

`refresh=true` 会失效探测缓存并真实加载模型，需 `ops:admin`；普通诊断只需 `ops:diagnostics`。认证关闭时匿名身份 `["*"]` 放行全部。

---

## 6. 指标抓取

- 引擎：`GET http://127.0.0.1:9079/metrics`（运维口，默认业务口 +1000；业务口同名路径仍存在，但终结器拓扑下业务口只监听回环）。认证开启时需 `ops:diagnostics`；`PRIVACY_METRICS_ALLOW_MTLS=true` 时可用已验签的终结器 CN 准入。
- 终结器拓扑：外部抓取 `https://<host>:9079/metrics`（`PRIVACY_OPS_TLS_STACK=both|tls`）或 TLCP 客户端（`tlcp`），由终结器代理到 Python 回环运维口。
- 网关：`GET http://127.0.0.1:9000/metrics`（`PRIVACY_GATEWAY_OPS_PORT`）；配置 `PRIVACY_GATEWAY_METRICS_API_KEY` 后需 `Authorization: Bearer <key>`，环回开发态可匿名。
- 网关 `/gateway/backends` 与 `/metrics` 共用观测面；`PRIVACY_GATEWAY_ADMIN_API_KEY` 缺省继承 metrics key。
- `/metrics` 请求本身不进入 RED 计数，避免自引用。
- 指标字典、抓取配置与告警示例见 [`prometheus.md`](./prometheus.md)。

---

## 7. 预算治理

```bash
# 查询剩余预算
curl -s http://127.0.0.1:8079/v1/privacy/budget -H "Authorization: Bearer $KEY"

# 重置（POST 或 DELETE 等价）
curl -s -X POST http://127.0.0.1:8079/v1/privacy/budget/reset -H "Authorization: Bearer $KEY"
```

- 默认 `epsilon_total=10.0`、`delta_total=1e-4`；命名空间由 `PRIVACY_NAMESPACE` 或请求画像决定。
- `PRIVACY_BUDGET_DB` 设置后使用 SQLite（WAL、busy_timeout=10s、启动 `PRAGMA integrity_check`）；损坏时拒绝初始化而不是回退内存。
- `PRIVACY_BUDGET_REDIS_URL` / `PRIVACY_BUDGET_BACKEND=redis` 使用 Redis；连接失败回退 SQLite/内存并告警。
- `PRIVACY_BUDGET_WINDOW_SECONDS` 控制自动重置窗口。
- 每次消耗写入 HMAC-SHA256 审计日志（`PRIVACY_BUDGET_AUDIT_LOG`，默认 `/tmp/budget_audit.log`）。生产必须设置 `PRIVACY_AUDIT_KEY`，否则使用进程级随机密钥，重启后旧签名不可校验。

验证审计链：

```bash
PRIVACY_AUDIT_KEY="$PRIVACY_AUDIT_KEY" \
PRIVACY_BUDGET_AUDIT_LOG=/var/log/privacy/budget_audit.log \
python -m engine.privacy.verify_audit
```

预算耗尽：REST `429`，gRPC `RESOURCE_EXHAUSTED`，并递增相应拒绝指标。

---

## 8. 规则与画像热重载 SOP

1. 在**新目录**准备规则 YAML（`rules/domains`、`rules/standards`、`rules/taxonomies`），不要原地覆盖。
2. 校验：
   ```bash
   curl -s -X POST http://127.0.0.1:8079/v1/dynclassification/validate \
     -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"rules_dir":"rules"}'
   ```
3. 先 `dry_run` 抽样，确认等级与命中规则符合预期。
4. 原子切换目录/符号链接（同文件系统 rename），再对每个副本调用：
   ```bash
   curl -s -X POST http://127.0.0.1:8079/v1/dynclassification/profiles/reload \
     -H "Authorization: Bearer $KEY"
   ```
5. 通过 `/v1/ops/diagnostics`、`/v1/dynclassification/standards`、`/v1/dynclassification/domains` 核对生效版本。
6. 回滚：切回旧目录并再次 reload。失败时旧缓存保持不变（不会回退到半加载状态）。

热重载默认开启（mtime 检测），`PRIVACY_DYNCLASSIFICATION_RELOAD_INTERVAL` 可节流。LLM/NER 启动参数变化仍需重启进程。

> Python **没有**清单签名/整包摘要验真；发布物完整性需由部署系统（镜像摘要、只读卷、签名）保证。见 [`policy_bundles.md`](./policy_bundles.md)。

---

## 9. 故障排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 启动即退出，日志 `PRIVACY_TLS_CERT_FILE ... required` | 开了 TLS 但缺证书/私钥 | 补齐 `PRIVACY_TLS_CERT_FILE`/`KEY_FILE` |
| 启动即退出，`PRIVACY_TLS_CA_FILE is required` | `tls_client_auth` 为 optional/require 但无 CA | 配置 CA 或改回 `none` |
| REST 正常但 gRPC 不可用 | 端口冲突/防火墙/未装 grpcio | 检查 `:50051`、`grpcio` 依赖 |
| `/readyz` 503 `Database connectivity check failed` | 预算 SQLite 不可读/损坏 | 修复挂载与权限；必要时从备份恢复 |
| 所有 mTLS 调用 401/UNAUTHENTICATED | 白名单为空或 CN 不匹配 | 配置 `PRIVACY_AUTH_MTLS_ALLOWED_CNS` 或 YAML；确认未开 `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED` |
| 管理端点 503 | `PRIVACY_GATEWAY_ADMIN_API_KEY`（及 fallback metrics key）未配置 | 设置 Key 后重启网关 |
| 网关 `/readyz` 503 | REST 池无健康节点 | 检查后端探针、`PRIVACY_GATEWAY_BACKENDS` 与 `PRIVACY_GATEWAY_HEALTH_CHECK_INTERVAL` |
| `/metrics` 401/403 | 未带 Key 或缺 `ops:diagnostics` | 使用监控专用 Key |
| 诊断页一直显示引擎不可用 | 探测缓存陈旧 | `?refresh=true` + `ops:admin` |
| LLM 调用全部降级 | 并发信号量满/内存不足/熔断 | 检查 `PRIVACY_LLM_MAX_CONCURRENCY`、`PRIVACY_LLM_MIN_FREE_MEM_MB`、`PRIVACY_LLM_CB_COOLDOWN` |
| 分类结果未随规则更新 | 未 reload 或 mtime 未变 | 调 reload；确认 rules dir 指向正确 |
| 上传 413 | 超过 50 MiB / 32 MiB | 调 `PRIVACY_MAX_UPLOAD_BYTES`/`PRIVACY_MAX_BODY_SIZE`，并评估内存 |
| `Task was destroyed but it is pending` | 旧版健康任务未等待 | 当前版本已在关闭时 await 健康任务；确认未运行旧镜像 |
| 日志非 JSON | `PRIVACY_LOG_FORMAT` 未设为 json | 设置并重启；确认安装 `python-json-logger` |
| 终结器启动即退出，`no HMAC key is configured` | 开启身份注入但无密钥 | 配置 `PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE` 或显式 `..._ENABLED=false` |
| 终结器启动即退出，`must point at a loopback upstream` | 上游填了非回环地址 | 改为 `127.0.0.1`/`localhost`，Python 只监听回环 |
| 终结器启动即退出，`invalid listener stack` | `PRIVACY_TLS_STACK` 取值非法 | 仅接受 `tlcp|both|tls` |
| 所有请求 401，Python 日志 `identity signature mismatch` | 两侧 HMAC 算法/密钥不一致或时钟偏移 | 同步 `PRIVACY_TERMINATOR_IDENTITY_*`；校对 NTP；检查 `MAX_SKEW_SECONDS` |
| 运维口 404 | 请求路径不在白名单 | 仅访问 `/health*` `/livez` `/readyz` `/metrics` `/ops/diagnostics` `/debug/pprof/` |
| 终结器 `/readyz` 503 `upstream not ready` | Python REST/ops 或 gRPC 未就绪 | 查看 Python 容器日志与 `/readyz`；确认 18079/15051/19079 与上游配置一致 |

---

## 10. 部署与资源建议

- **core 镜像**不包含 torch/transformers，仅适合规则/原语路径；启用 NER/LLM 请用 `--target ml` 或自建镜像。
- 大表/图片分类的 gRPC 消息上限为 64 MiB；HTTP 请求体默认 32 MiB。超出会明确拒绝。
- 多进程 worker 各自持有独立缓存与预算连接；共享预算必须使用 SQLite（同机）或 Redis。
- 生产必须：显式开启 Auth/TLS/RateLimit；设置 `PRIVACY_AUDIT_KEY`；限制 `PRIVACY_IMAGE_ALLOWED_DIRS`；只读挂载规则与模型目录。
- 终结器拓扑：Python 只监听 `127.0.0.1`，对外端口全部由 [`terminator/`](../terminator/README.md) 持有；
  两侧共享同一份 HMAC 密钥与 mTLS 白名单。部署示例见
  [`deploy/docker-compose.terminator.yml`](../deploy/docker-compose.terminator.yml) 与
  [`deploy/k8s/terminator-sidecar.yaml`](../deploy/k8s/terminator-sidecar.yaml)。
- 当前版本仍无真实 pprof profile、无策略清单；引擎已有独立运维口 `:9079`，网关运维口 `:9000`。
