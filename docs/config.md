# Python 隐私引擎 (privacy-engine-python) — 配置字典

> **组件归属**：`services/privacy-engine-python`
> **事实源**：`engine/env_loader.py`、`engine/server.py`、`engine/launcher.py`、`engine/main.py`、`engine/security/config.py`、`engine/gateway/*`
> **原则**：未设置即使用代码默认值；生产环境必须显式开启安全项。

---

## 1. 配置加载顺序

```text
1. 进程环境变量（最高优先级，永远不被 .env 覆盖）
2. 基础 .env：项目根目录 .env → 当前工作目录 .env（二选一，先存在者）
3. 场景 profile：PRIVACY_ENV_PROFILE / ENV_PROFILE / APP_ENV
   → config/env/<profile>.env（override=True，覆盖基础 .env）
   → .env.<profile> 变体（见 env_loader.py）
```

仓库内置场景：`config/env/mlx.env`、`qwen3.env`、`vllm.env`、`openai.env`。`engine.main`、`engine.security.config` 在导入期都会调用 `load_env_file()`，因此从第一行业务导入起即可读到配置。

> 环境变量名统一使用 `PRIVACY_*`（引擎）与 `PRIVACY_GATEWAY_*`（Python 网关进程），与 Go 参考实现同名同义。不要引入 `GATEWAY_*` / `ENGINE_GATEWAY_*` 等历史别名。

---

## 2. 传输与运行时

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_REST_HOST` | `0.0.0.0` | REST 监听地址（注意与 Go 默认 `127.0.0.1` 不同） |
| `PRIVACY_REST_PORT` | `8079` | REST 端口 |
| `PRIVACY_GRPC_HOST` | `0.0.0.0` | gRPC 监听地址 |
| `PRIVACY_GRPC_PORT` | `50051` | gRPC 端口 |
| `PRIVACY_GRPC_MAX_WORKERS` | `64` | gRPC 线程池大小，必须 ≥1 |
| `PRIVACY_WORKERS` | `min(cpu_count, 8)` | 仅 `engine.launcher` 多进程模式；SO_REUSEPORT 需 Linux |
| `PRIVACY_LIMIT_CONCURRENCY` | `10000` | Uvicorn 最大并发连接 |
| `PRIVACY_LIMIT_MAX_REQUESTS` | `100000` | 单 worker 最大请求数（防内存泄漏） |
| `PRIVACY_TIMEOUT_KEEP_ALIVE` | `30` | Keep-Alive 秒数 |
| `PRIVACY_TIMEOUT_GRACEFUL_SHUTDOWN` | `10` | SIGTERM 后优雅关闭等待秒数 |
| `PRIVACY_MAX_BODY_SIZE` | `33554432`（32 MiB） | REST 请求体上限，超限 413 |
| `PRIVACY_MAX_UPLOAD_BYTES` | `52428800`（50 MiB） | 文件/CSV 上传上限，与 Go `ParseMultipartForm(50<<20)` 同口径 |
| `PRIVACY_ENV_PROFILE` | 未设置 | 场景 profile 名称（兼容 `ENV_PROFILE` / `APP_ENV`） |
| `PRIVACY_PROFILE` | `privacy-profile.yaml` | 隐私参数画像路径 |
| `PRIVACY_NAMESPACE` | `default` | 默认预算/画像命名空间 |
| `PRIVACY_SERVICE_NAME` | `PrivShield` | 日志/诊断服务名 |

### 日志与追踪

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR/CRITICAL |
| `PRIVACY_LOG_FORMAT` | `text` | `json` 需安装 `python-json-logger`；容器 entrypoint 默认改为 `json` |
| `PRIVACY_DEBUG_VALIDATION` | `false` | 422 响应是否附带 Pydantic 校验细节 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 未设置 | 设置后启用 OpenTelemetry 导出 |
| `OTEL_SERVICE_NAME` | `PRIVACY_SERVICE_NAME` | 追踪服务名 |

### CORS（默认关闭）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_CORS_ENABLED` | `false` | 是否启用 CORS |
| `PRIVACY_CORS_ALLOW_ORIGINS` | `*` | 逗号分隔；为 `*` 时强制 `allow_credentials=false` |
| `PRIVACY_CORS_ALLOW_CREDENTIALS` | `true` | 通配符场景自动降为 `false` 并告警 |
| `PRIVACY_CORS_ALLOW_METHODS` | `*` | 允许方法 |
| `PRIVACY_CORS_ALLOW_HEADERS` | `*` | 允许请求头 |
| `PRIVACY_CORS_EXPOSE_HEADERS` | `x-request-id` | 暴露响应头 |
| `PRIVACY_CORS_MAX_AGE` | `600` | 预检缓存秒数 |

---

## 3. 安全配置（`engine/security/config.py`）

所有安全开关默认关闭。`get_security_settings()` 每次调用重新读取环境变量，因此测试/运维修改进程环境后无需重启即可生效（但已按启动时值装配的 Uvicorn TLS/文档开关除外）。

### TLS（标准 TLS，非 TLCP）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_TLS_ENABLED` | `false` | 启用后必须同时提供证书与私钥，否则启动期 `ValueError` |
| `PRIVACY_TLS_CERT_FILE` | — | 服务端证书 |
| `PRIVACY_TLS_KEY_FILE` | — | 服务端私钥 |
| `PRIVACY_TLS_CA_FILE` | — | 客户端 CA；`tls_client_auth` 为 optional/require 时必填 |
| `PRIVACY_TLS_CLIENT_AUTH` | `none` | `none` / `optional` / `require` |
| `PRIVACY_TLS_KEY_PASSWORD` | — | 私钥口令 |
| `PRIVACY_TLS_STACK` | `""` | **仅由 `/readyz` 原样回显，不选择协议栈**；Python 不实现 TLCP |
| `PRIVACY_OPS_HOST` | 继承 `PRIVACY_REST_HOST` | 独立运维口监听地址 |
| `PRIVACY_OPS_PORT` | REST 端口 + 1000 | 独立运维口端口；REST=0 时同样随机 |
| `PRIVACY_OPS_TLS_STACK` | `""` | `tlcp` / `both` / `tls` / 空（继承业务口）；`tlcp|both` 要求绑定环回 |
| `PRIVACY_OPS_TLS_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` | 继承 `PRIVACY_TLS_*` | 运维口材料；`tls` 档三项必填 |
| `PRIVACY_OPS_TLS_KEY_PASSWORD` | 继承 `PRIVACY_TLS_KEY_PASSWORD` | 运维口私钥口令 |

### 终结器身份（`engine/security/terminator.py`）

仅当部署 Go TLCP 终结器时启用；终结器与 Python 必须配置同一组值。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_TERMINATOR_IDENTITY_ENABLED` | `false` | 启用后只接受已验签的终结器身份头；缺密钥启动即 `ValueError` |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO` | `sm3` | `sm3` / `sha256` |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY` | — | ≥16 字节共享密钥 |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE` | — | 密钥文件，优先于环境变量；mtime 缓存 |
| `PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS` | `60` | 时间戳窗口，必须 >0 |

启用后，来源必须是回环；任一身份头存在但缺失配对头、重复、hex 畸形、签名错误、时间戳越窗或
`sha256(SPKI-DER)` 与签名摘要不符，都会整体拒绝。`PRIVACY_METRICS_ALLOW_MTLS=true` 也要求先启用终结器验签。

### 认证与授权

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_AUTH_ENABLED` | `false` | 启用后 REST/gRPC 都要求凭据 |
| `PRIVACY_AUTH_INTERNAL_KEYS_JSON` | `{}` | 内部 Key JSON：`{"key": {"name": "...", "scopes": [...]}}` |
| `PRIVACY_AUTH_EXTERNAL_KEYS_JSON` | `{}` | 外部 Key JSON，同结构 |
| `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED` | `false` | 允许直连 gRPC mTLS 证书或已验签终结器身份换取内部身份（REST 业务身份仍只来自 Bearer） |
| `PRIVACY_AUTH_MTLS_ALLOWED_CNS` | `[]` | 静态 CN 白名单（逗号分隔或 JSON 数组），命中授予 `["*"]` |
| `PRIVACY_AUTH_MTLS_WHITELIST_FILE` | — | YAML per-CN scope 白名单，设置后忽略静态列表；支持 mtime 热重载 |
| `PRIVACY_HEALTH_NO_AUTH` | `true` | 健康端点免认证 |
| `PRIVACY_HEALTH_NO_RATE_LIMIT` | `true` | 健康端点免限流 |

白名单为空时**拒绝所有** mTLS 证书（fail-closed）。`*` scope 不穿透 `audit:` 前缀。

### 指标准入（`/metrics`）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_METRICS_API_KEY` | — | 专属指标 Key，与业务静态 Key 独立；配置后匿名抓取一律拒绝 |
| `PRIVACY_METRICS_API_KEY_PREVIOUS` | — | 轮换期并行接受的旧指标 Key |
| `PRIVACY_METRICS_ALLOW_MTLS` | `false` | 允许已验签终结器对端按 CN 抓取；要求先启用终结器验签 |
| `PRIVACY_METRICS_ALLOWED_CLIENT_CNS` | `[]` | 允许抓取的客户端 CN；为空表示任意已验签 CN |

认证开启且未配置专属指标 Key 时，抓取方仍需持有带 `ops:diagnostics` scope 的普通凭据；生产姿态无任何准入配置时 fail-closed 401。

### IAM 用户库（条件启用）

仅当 `PRIVACY_AUTH_USER_STORE_BACKEND` 显式配置时，`/v1/auth/*` 路由才会挂载，线路层 Bearer 解析才接入用户库。`memory` / `sqlite` 可用；`postgres` 在当前构建因缺少驱动在启动期拒绝。后端已配置但 `PRIVACY_AUTH_ENABLED=false` 会把用户管理接口暴露给开发身份，生产必须同时启用认证。完整变量表、沙箱与令牌语义见 [auth.md](./auth.md) §3。

### 限流

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_RATE_LIMIT_ENABLED` | `false` | 启用滑动窗口限流 |
| `PRIVACY_RATE_LIMIT_DEFAULT_RPS` | `10.0` | 默认持续速率 |
| `PRIVACY_RATE_LIMIT_DEFAULT_BURST` | `20.0` | 默认突发额度 |
| `PRIVACY_RATE_LIMIT_PER_ENDPOINT_JSON` | `{}` | 例：`{"/v1/privacy/mask": {"rps": 50, "burst": 100}}` |
| `PRIVACY_RATE_LIMIT_REDIS_URL` | — | 设置后使用 Redis 存储，否则进程内内存 |

限流存储异常时 fail-open，同时递增 `privacy_auth_denials_total{reason="rate_limit_storage_error"}`。

---

## 4. 分类分级（三层漏斗）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_DYNCLASSIFICATION_RULES_DIR` | `rules` | 领域/标准/分类体系 YAML 根目录（相对进程 CWD） |
| `PRIVACY_DYNCLASSIFICATION_HOT_RELOAD` | `true` | mtime 变更自动重载 |
| `PRIVACY_DYNCLASSIFICATION_RELOAD_INTERVAL` | `0` | 重载检查最小间隔（秒），0 不节流 |
| `PRIVACY_DYNCLASSIFICATION_PATH_BASE` | `.` | `generate_profile` 等路径的允许根目录（防穿越） |
| `PRIVACY_ENGINE_CACHE_MAX_SIZE` | `4096` | 规则引擎求值缓存条目数 |
| `PRIVACY_CLASSIFICATION_CACHE_SIZE` | `10000` | 漏斗结果 LRU 条目数 |
| `PRIVACY_NER_ENABLE` | `false` | 启用 L2 Small-NER |
| `PRIVACY_LLM_ENABLE` | `false` | 显式启用 L3 LLM |
| `PRIVACY_LLM_ENABLE_ARBITRATION` | `true` | 冲突时启用 LLM 仲裁 |
| `PRIVACY_LLM_CONFIDENCE_THRESHOLD` | `0.75` | 低于阈值触发 LLM |
| `PRIVACY_LLM_AUTO_ON_IMAGE` | `true` | 图片/DICOM 自动触发多模态 LLM |
| `PRIVACY_IMAGE_ALLOWED_DIRS` | `<cwd>/{data,uploads,samples,medical_images}` + 临时目录 | 图片读取目录白名单（`os.pathsep` 分隔） |

### LLM 引擎

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_LLM_BACKEND` / `PRIVACY_LLM_PROVIDER` | `auto` / 空 | `auto`、`vllm`、`openai`、`local` 等 |
| `PRIVACY_LLM_MODEL_NAME` | `Qwen/Qwen3.5-0.8B-Privacy-Classifier-Smoother`（下载脚本） | 模型标识 |
| `PRIVACY_LLM_MODEL_PATH` | 下载脚本推导 | 本地模型目录 |
| `PRIVACY_VLLM_URL` / `PRIVACY_LLM_API_BASE` | — | 远程推理服务地址 |
| `PRIVACY_LLM_API_KEY` | `EMPTY` | 远程服务 Key |
| `PRIVACY_LLM_DEVICE` / `PRIVACY_DEVICE` | — | 本地推理设备（如 `cuda`、`mps`） |
| `PRIVACY_LLM_TIMEOUT` / `PRIVACY_VLM_TIMEOUT` | `180` | 推理超时秒数 |
| `PRIVACY_LLM_HTTP_WORKERS` | `16` | 远程 HTTP 客户端线程数 |
| `PRIVACY_LLM_CB_COOLDOWN` | `15.0` | LLM 熔断冷却秒数 |
| `PRIVACY_LLM_MAX_CONCURRENCY` | 远程 16 / 本地 1 | 推理并发上限 |
| `PRIVACY_LLM_SEMAPHORE_WAIT_SECONDS` | `30.0` | 并发排队超时，超时降级跳过 LLM |
| `PRIVACY_LLM_MIN_FREE_MEM_MB` | `512.0` | 低于此可用内存跳过本地 LLM |
| `PRIVACY_VLM_MIN_VRAM_GB` | `1.6` | 多模态模型最低显存 |
| `PRIVACY_MODELS_DIR` | `<project>/.models` | 模型缓存目录（下载脚本） |

---

## 5. 预算、哈希与画像

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_BUDGET_DB` | 未设置 | SQLite 预算库路径；设置后跨进程/重启持久化，并在 `/readyz` 中被检查 |
| `PRIVACY_BUDGET_REDIS_URL` | 未设置 | 设置后使用 Redis 分布式预算 |
| `PRIVACY_BUDGET_BACKEND` | 空 | 设为 `redis` 时使用 Redis（即使未设置 URL 也会尝试默认地址） |
| `PRIVACY_BUDGET_WINDOW_SECONDS` | 未设置 | 预算窗口秒数；未设置/≤0 不自动重置 |
| `PRIVACY_BUDGET_AUDIT_LOG` | `/tmp/budget_audit.log` | HMAC 审计日志路径 |
| `PRIVACY_AUDIT_KEY` | 未设置 | 审计链 HMAC 密钥；未设置时使用开发态默认并告警 |
| `PRIVACY_HASH_LEGACY_SHA256` | 空 | 设为真值时哈希回退 SHA-256；默认 HMAC-SM3 |
| `PRIVACY_PERSONALIZED_PROFILE` | `config/personalized-profiles.yaml` | 个性化画像路径 |

预算默认总额：`epsilon_total=10.0`、`delta_total=1e-4`；可通过预算 API/画像覆盖。Redis/DB 连接失败会回退 SQLite/内存并告警，不会静默清零已持久化预算。

---

## 6. Python 网关进程（`engine.gateway.server`）

网关是独立进程，配置全部来自 `PRIVACY_GATEWAY_*` 环境变量与 CLI 参数，不再读取 YAML 或 `GATEWAY_*` 别名（事实源 `engine/gateway/settings.py`）。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_GATEWAY_HOST` | `127.0.0.1` | HTTP 代理监听地址 |
| `PRIVACY_GATEWAY_PORT` | `8000` | HTTP 代理端口 |
| `PRIVACY_GATEWAY_GRPC_HOST` | `127.0.0.1` | gRPC 代理监听地址 |
| `PRIVACY_GATEWAY_GRPC_PORT` | `50000` | gRPC 代理端口 |
| `PRIVACY_GATEWAY_OPS_PORT` | 业务口 + 1000 | 独立运维口；业务口为 0 时也为 0 |
| `PRIVACY_GATEWAY_BACKENDS` | `127.0.0.1:8079` | REST 后端地址，逗号分隔；自动补 `http://` |
| `PRIVACY_GATEWAY_GRPC_BACKENDS` | `127.0.0.1:50051` | gRPC 后端地址，逗号分隔 |
| `PRIVACY_GATEWAY_STRATEGY` | `p2c` | `p2c`/`round_robin`/`weighted_round_robin`/`least_connections`/`weighted_random`/`random` |
| `PRIVACY_GATEWAY_HEALTH_CHECK_INTERVAL` | `5.0` | 主动健康检查间隔秒数（Python 超集，0 关闭） |
| `PRIVACY_GATEWAY_MAX_BODY_BYTES` | `33554432` | 请求体上限（32 MiB） |
| `PRIVACY_GATEWAY_AUTH_ENABLED` | `false` | 逐请求凭据准入（除 `/health` 外必须带 Authorization） |
| `PRIVACY_GATEWAY_BACKEND_API_KEY` | 未设置 | 匿名请求的回源兜底 Bearer Key；非环回绑定下与准入至少配置其一 |
| `PRIVACY_GATEWAY_INBOUND_KEYS_FILE` | 未设置 | 入站 Keyfile scope/路径策略；生产必填 |
| `PRIVACY_GATEWAY_ALLOWED_CIDRS` | 空 | 入站 IP 白名单，逗号分隔或 JSON 数组 |
| `PRIVACY_GATEWAY_TRUSTED_PROXIES` | 空 | 信任的代理 CIDR，仅这些来源的转发头生效 |
| `PRIVACY_GATEWAY_REQUIRE_TLS` / `_REQUIRE_FORWARDED_PROTO` | `false` | 非环回调用方强制 HTTPS（426） |
| `PRIVACY_GATEWAY_METRICS_API_KEY` | 未设置 | 运维口 `/metrics` Bearer Key |
| `PRIVACY_GATEWAY_ADMIN_API_KEY` | 继承 metrics key | 管理端点与 `/gateway/backends` Bearer Key；两者皆空时 fail-closed 503 |
| `PRIVACY_GATEWAY_TLS_STACK` | `tlcp` | `tlcp`/`both`/`tls`；三档均 mTLS，无明文档 |
| `PRIVACY_GATEWAY_TLS_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` | 未设置 | `tls` 档必填（cert/key/CA 成对校验） |
| `PRIVACY_GATEWAY_OPS_TLS_STACK` | `both` | 运维口独立 TLS 档位 |
| `PRIVACY_GATEWAY_OPS_TLS_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` | 继承业务口 | 运维口 TLS 材料；`tls` 档必填 |
| `PRIVACY_GATEWAY_BACKEND_TLS_ENABLED` | 空 | 回源 TLS 开关（遗留布尔，口径见服务间认证文档） |
| `PRIVACY_GATEWAY_BACKEND_TLS_CA` | 未设置 | 回源 CA；启用 TLS 但缺失时构造期 fail-fast |
| `PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_CERT` / `_CLIENT_KEY` | 未设置 | 回源 mTLS 客户端身份，必须成对 |
| `PRIVACY_GATEWAY_LIMIT_CONCURRENCY` | `10000` | Uvicorn 并发连接上限 |
| `PRIVACY_GATEWAY_LIMIT_MAX_REQUESTS` | `100000` | 单 worker 最大请求数 |
| `PRIVACY_GATEWAY_TIMEOUT_KEEP_ALIVE` | `120` | Keep-Alive 秒数 |
| `PRIVACY_GATEWAY_TIMEOUT_GRACEFUL_SHUTDOWN` | `10` | 优雅关闭等待秒数 |

启动期强校验（fail-closed）：

- `TLS_STACK` / `OPS_TLS_STACK` 只接受 `tlcp|both|tls`；`tls` 必须提供 cert/key/CA；
- `tlcp` / `both` 只能绑定环回地址（Python 无原生 TLCP，由 Phase 8 Go 终结器终结）；
- 非环回绑定必须配置 `AUTH_ENABLED=true` 或 `BACKEND_API_KEY`，并配置 metrics/admin Key 之一；
- `APP_ENV` / `PRIVACY_GATEWAY_ENV` 为 production 时必须提供仓库外的 `INBOUND_KEYS_FILE`；
- 所有引用的证书/Keyfile 必须可读，否则拒绝启动。

CLI 参数（`--rest-host` / `--rest-port` / `--grpc-host` / `--grpc-port` / `--ops-port`）覆盖环境变量。

---

## 7. 容器运行

镜像内置默认值：

| 变量 | 默认值 | 来源 |
|---|---|---|
| `PYTHONUNBUFFERED` | `1` | Dockerfile |
| `PRIVACY_REST_HOST` / `PRIVACY_GRPC_HOST` | `0.0.0.0` | Dockerfile + entrypoint |
| `PRIVACY_LOG_FORMAT` | `json` | `docker-entrypoint.sh` |

`ENTRYPOINT` 固定为 `docker-entrypoint.sh`，`CMD` 默认 `python -m engine.server`；entrypoint 仅做提示性检查（profile/rules 路径）后 `exec "$@"`，信号可直接到达主进程。

---

## 8. 与 Go 配置的已知差异

| 差异项 | Go | Python | 说明 |
|---|---|---|---|
| 网关入站前缀 | `PRIVACY_GATEWAY_*` | `PRIVACY_GATEWAY_*` | 已统一，同名同义 |
| `PRIVACY_TLS_STACK=tlcp` | 原生 TLCP | 由 Go 终结器终结 | 引擎同名变量仅回显；终结器读同名变量选择协议栈 |
| 引擎独立运维口 `:9079` | 支持 | 支持 | 业务口 +1000；`/metrics` `/ops/diagnostics` 在运维口白名单内 |
| 终结器身份头 | Go 引擎直接读 mTLS | 验签 Go 终结器注入的头 | REST 仍只认 Bearer；gRPC 可由验签 CN 换取身份 |
| `PRIVACY_RULES_DIR` | 规则目录 | `PRIVACY_DYNCLASSIFICATION_RULES_DIR` | Python 分类/医疗/流水线统一读后者 |
| `PRIVACY_CONFIG_MANIFEST_FILE` | 支持 | 不支持 | 见 [`policy_bundles.md`](./policy_bundles.md) |
| `PRIVACY_PPROF_ENABLED` | 支持 | 不支持 | 运维口只保留同名路径入口，不伪造 Go profile |
| REST 默认 `127.0.0.1` | `127.0.0.1` | `0.0.0.0` | 终结器拓扑必须显式绑定 `127.0.0.1` |
