# 认证、授权与传输安全（Security）

> 代码：`engine/security/`（线路层）· `engine/auth/`（IAM 内核）· 测试：`tests/security/`
> 对照实现：Go `services/privacy-engine` 的 `pkg/auth` + `pkg/middleware` + 拦截器链

## 定位

Python 引擎的入站安全栈由两层组成：

1. **线路层（已全量接线）**：静态 API Key、gRPC mTLS 证书身份、Go 终结器签名身份、凭据策略（含 PoP）、权限 scope、滑动窗口限流、TLS。REST 通过 FastAPI 依赖执行，gRPC 通过服务拦截器执行，`/metrics` 通过纯 ASGI 中间件执行。
2. **IAM 内核（条件接线）**：用户库、登录会话、动态 API Key、TOTP/MFA、恢复码、令牌交换、服务账号。仅当显式配置 `PRIVACY_AUTH_USER_STORE_BACKEND` 时，`/v1/auth/*` 路由才会挂载，线路层的 Bearer 解析才会先查询 IAM 运行时。

线路层的静态 Key 始终有效；IAM 是叠加层，不是替代层。两层共用同一个 `Identity` 与权限判定函数。

## 代码地图

### 线路层 `engine/security/`

| 文件 | 职责 |
|---|---|
| `auth.py` | `get_current_identity` REST 依赖、`require_permission` / `require_any_permission`、`AuthInterceptor` gRPC 拦截器、`ApiKeyAuthAsgiMiddleware`（`/metrics`）、凭据策略与沙箱执行 |
| `identity.py` | `Identity` 数据类与 `has_permission`；REST 路径 / gRPC 方法到权限的参考映射；标准健康路径与公开认证路径白名单 |
| `config.py` | `SecuritySettings`：TLS、静态 Key、mTLS 白名单、限流、健康豁免；`get_security_settings()` 每次调用重新解析环境变量 |
| `ratelimit.py` | 基于 `limits` 的 `MovingWindowRateLimiter`；REST `rate_limit_dependency` 与 gRPC `RateLimitInterceptor` 共用进程级 `Limiter` |
| `whitelist.py` | mTLS CN 白名单：YAML per-CN scope、mtime 被动热重载、两阶段提交、静态 CN 列表回退 |
| `tls.py` | Uvicorn `ssl_*` 参数与 `grpc.ssl_server_credentials` 构造；支持 `none` / `optional` / `require` 三档客户端证书 |
| `terminator.py` | 验签 Go 终结器身份头：环回限制、时间戳 skew、HMAC-SM3/SHA256 常量时间比对、SPKI-DER 与摘要一致性 |
| `pop.py` | 与 Go `pop_decision.go` 同构的 PoP 状态机；对端证书视图来自验签后的终结器身份 |
| `sm3.py` | SM3 国密散列（优先 OpenSSL `hashlib.new("sm3")`，缺失时纯 Python 回退），输出跨环境一致 |

### IAM 内核 `engine/auth/`

| 文件 | 职责 |
|---|---|
| `config.py` | `AuthSettings.from_env()`：用户库后端、会话 TTL、口令策略、TOTP、集群受众、审计出口等 |
| `runtime.py` | 进程级 `AuthRuntime`；显式后端才构建；`memory` / `sqlite` 可用，`postgres` 因缺少驱动在启动期拒绝 |
| `service.py` | `AuthService`：登录、会话、动态 Key、MFA、恢复码、用户/服务账号管理、令牌交换与吊销、`resolve_token`、`enforce_sandboxes` |
| `store.py` / `persistence.py` | 内存用户库与 SQLite 仓储（含 MFA pending 槽、高水位、审计 outbox） |
| `routes.py` | `/v1/auth/*` 与 `/v1/.well-known/service-capabilities/public` 路由；Handler 级 ABAC 在本文件与 `service.py` |
| `password.py` / `totp.py` / `recovery.py` | bcrypt(cost=12) 与 SM3 迭代散列；TOTP 生成/校验（`sha1` / `sm3`）；8 组一次性恢复码 |
| `bound_pubkey.py` / `contract.py` / `idempotency.py` | 服务账号 SPKI 绑定与摘要；凭据契约解析与路径匹配；动态 Key 签发的幂等重放 |
| `migrate.py` / `migrations.py` | SQLite schema 迁移（仅显式迁移路径执行） |
| `security.py` | `sm3_hex` 等 IAM 内部散列辅助 |

## 请求处理链

```text
REST 业务请求
  └─ SECURITY_DEPS = [get_current_identity, rate_limit_dependency]
        ├─ 认证关闭 → DEV_MODE_IDENTITY（["*", "admin"]）
        ├─ 标准健康路径 GET/HEAD + PRIVACY_HEALTH_NO_AUTH → health-probe（health:read）
        ├─ 公开认证路径且未带 Authorization → public-caller（auth:public）
        ├─ Bearer → IAM（会话 → 动态 Key → 交换票据）→ 静态内部 Key → 静态外部 Key
        ├─ 凭据策略（受众 → 契约版本 → drain-only → 路径允许列表）
        ├─ 沙箱（首登改密 / 口令过期 / MFA）
        └─ require_permission("...") 逐路由 scope 校验
  └─ rate_limit_dependency（按身份 + 实际路径滑动窗口）

gRPC 请求
  └─ GrpcObservabilityInterceptor → AuthInterceptor → RateLimitInterceptor → Servicer

/metrics（ASGI mount）
  └─ ApiKeyAuthAsgiMiddleware：认证 → 凭据策略 + 沙箱 → ops:diagnostics
```

### 认证解析细节

- `Authorization` 只接受 `Bearer <token>`。空 header → `missing_token`；非空但格式不符（如裸 token）→ `bare_token`，一律 401，不做宽容解析。
- 静态 Key 查找使用常量时间比较（对排序后的全部 Key 逐一 `hmac.compare_digest`，不短路），内部 Key 先于外部 Key。
- 启用 IAM 后，`_resolve_token` 先调用 `runtime.service.resolve_token`，未命中再回退静态 Key。IAM 内部顺序为 **登录会话 → 用户动态 API Key → 本进程交换票据**。
- 公开认证路径在“完全未带凭据”时获得 `public-caller`；若携带了凭据，则按正常身份校验，凭据无效仍返回 401。
- 健康豁免仅限 `GET`/`HEAD` 且路径命中标准白名单（`/health`、`/healthz`、`/livez`、`/readyz`、`/readyz/llm`、`/v1/privacy/health`）。其中 `/healthz` 只存在于豁免判定中，本进程**未注册该路由**，实际探针应使用 `/health` 或 `/livez`。`POST /health` 在认证开启时需要有效凭据。
- gRPC 身份顺序：直连 mTLS 证书（若启用）→ 验签后的终结器 CN（需 `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED=true`）→ 健康豁免 → metadata 中的 Bearer。健康方法在 `PRIVACY_HEALTH_NO_AUTH=true` 时直接返回 `health-probe` 身份。
- 终结器身份头（`X-PrivShield-Client-*` / `X-PrivShield-Identity-*`）在 REST 与 gRPC 上都会验签；REST 仅把结果用于 PoP/审计/服务账号激活，不改变 Bearer 业务身份。任一身份头存在但校验失败一律 401，不降级。

### 凭据策略

`_enforce_credential_policy` 在 REST、gRPC、`/metrics` 三面共用，顺序固定：

1. **受众**：当服务受众不是 `PrivShield` 且身份声明了受众/受众白名单时，必须匹配，否则 `403 AUDIENCE_MISMATCH`。
2. **契约版本**：请求携带 `X-PrivShield-Contract-Version` 时，必须与凭据登记版本完全一致，否则 `403 CONTRACT_VIOLATION`。
3. **drain-only**：`binding_status == "drain_only"` 的凭据只允许健康探针与公开认证路径。
4. **契约路径**：凭据登记了 `allowed_paths` 时，业务请求必须命中 `contract_allows`（`METHOD /path`），否则 `403 CONTRACT_VIOLATION`。

受众与契约版本检查对健康/公开路径同样生效；drain-only 与路径白名单对它们豁免。

5. **PoP**：`engine/security/pop.py::decide_pop` 与 Go `pkg/auth/pop_decision.go` 同构。凭据声明 CN / 证书指纹 / 公钥绑定时必须出示对端证书并全部匹配；未声明绑定的 workload 身份一律拒绝；登录会话等主体放行。对端证书视图来自 Go 终结器注入并 HMAC 签名的身份头——直连 mTLS 没有签名头，绑定凭据因此 fail-closed；证书指纹仅支持 `sha256:`，公钥绑定比对原始 SPKI DER。

### 沙箱

`_enforce_sandboxes` 仅对带 `subject` 的身份执行（登录会话、用户动态 Key 都可能命中；静态服务 Key 与 mTLS 身份无 subject，跳过）：

| 条件 | 错误 |
|---|---|
| 首登/管理员重置后未改密 | `403 PASSWORD_CHANGE_REQUIRED` |
| 口令超过有效期且 `_PASSWORD_EXPIRE_BLOCK=true` | `403 PASSWORD_EXPIRED_REQUIRE_CHANGE` |
| 登录会话所属账号未绑定 TOTP | `403 MFA_ENROLLMENT_REQUIRED` |
| 登录会话建立时未通过 TOTP | `403 MFA_SESSION_REQUIRED` |

豁免路径：`/health`、`/readyz`、`/livez`、`/v1/auth/login`、`/v1/auth/register`、`/v1/auth/logout`、`/v1/auth/change-password`、`/v1/auth/mfa/status`、`/v1/auth/mfa/setup`、`/v1/auth/mfa/setup/cancel`、`/v1/auth/mfa/enable`，以及用户查看自身 `/v1/auth/users/{username}`。API Key 身份不执行 MFA 检查（非会话身份），但首登改密/口令过期沙箱仍会生效。

### 权限判定

- Scope 为**精确匹配**，不做 glob 或层级展开。
- `*` 授予全部业务权限，但**不穿透** `audit:` 前缀（三权分立：系统管理员不得查看审计）。
- `audit:*` 可授予审计权限；具体 `audit:xxx` 也可单独授予。
- 复合权限 `"a|b"` 按逻辑或拆分。
- 权限不满足返回 `403 Forbidden: insufficient scope`；认证失败返回 `401 Unauthorized`。
- `permission_for_rest_path` 与 `rate_limit_for_path` 是**参考实现、未接线**：REST 路由在各 `routers/*` 中用 `require_permission` 显式声明 scope；`permission_for_grpc_method` 则由 `AuthInterceptor` 实际使用。未映射的 gRPC 方法返回 `"*"`，即只有通配身份可调用。

路由实际使用的 scope：

| Scope | 覆盖接口 |
|---|---|
| `privacy:mask` | `/v1/privacy/mask*`、`/v1/privacy/process_file`、`/v1/privacy/file/*` |
| `privacy:hash` | `/v1/privacy/hash` |
| `privacy:dp` | `/v1/privacy/dp/*`、`/v1/privacy/ldp/*` |
| `privacy:kano` | `/v1/privacy/k_anonymize*` |
| `privacy:qol` | `/v1/privacy/qol/*` |
| `privacy:budget` | `/v1/privacy/budget*` |
| `privacy:profile` | `/v1/privacy/profile/*` |
| `dynclassification:read` | `/v1/dynclassification/*` 分类/评估/校验读接口 |
| `dynclassification:write` | 分类配置热加载 / 生成配置 |
| `medical:process` | `/v1/medical/*` |
| `agent:process` | `/v1/agent/process`（`medical:process` 亦可） |
| `pipeline:process` | `/v1/pipeline/*` |
| `ops:diagnostics` | `/v1/ops/*`、`/metrics`、OpenAPI 别名 |
| `ops:admin` | `GET /v1/ops/diagnostics?refresh=true`（Handler 内二次校验；无独立 `/refresh` 路由） |

健康路由只挂认证 + 限流依赖，不显式要求 `health:read`；认证开启时，任何有效身份都可访问健康端点。

## IAM 条件接线

`engine/main.py` 在导入期检查：

```python
_auth_user_store_enabled = AuthSettings.from_env().enabled  # backend 非空
if _auth_user_store_enabled:
    app.include_router(auth_router)             # /v1/auth/*
    app.include_router(auth_well_known_router)  # /v1/.well-known/service-capabilities/public
```

启动 lifespan 中若后端已配置，会调用 `get_auth_runtime()` 强制构建；配置错误在启动期失败，不静默降级：

| 后端 | 行为 |
|---|---|
| 未设置 | IAM 未启用；`get_auth_runtime()` 返回 `None`，只使用静态 Key / mTLS |
| `memory` | 进程内存用户库；重启即丢失 |
| `sqlite` | 需 `PRIVACY_AUTH_USER_STORE_FILE`；生产标注环境需 `_USER_STORE_ENCRYPTION_KEY` |
| `postgres` | 当前构建无 psycopg 驱动，启动期抛 `RuntimeError`，拒绝启动 |

**部署红线**：`/v1/auth/*` 的挂载只取决于用户库后端是否配置，不取决于 `PRIVACY_AUTH_ENABLED`。若配置了后端却关闭认证，`get_current_identity` 返回带 `*`/`admin` 的开发身份，用户管理接口将对所有调用者开放。生产必须同时启用 `PRIVACY_AUTH_ENABLED=true`。

IAM 路由使用 `get_current_identity` 做主体识别，Handler 内按主体/角色裁决（本人、`user:read`、`user:admin` 等）；`/v1/auth/*` **不挂通用限流依赖**，登录/激活等敏感操作由 `AuthService` 内部按主体与 IP 计数限速。

IAM 运行时的事实边界：

- 交换票据保存在 `AuthService._issued_tokens`（进程内存），重启丢失，多 worker 之间不共享。
- `PRIVACY_AUTH_CLUSTER_TRUSTED_KEYS_JSON` / `_CLUSTER_SIGNING_KEY_FILE` 已解析但**未用于校验集群会话或委托令牌签名**；当前没有跨服务会话校验器。
- 生产标注环境下 `POST /v1/auth/token-exchange` 要求 TLS 终结器从环回注入非空 `X-Client-SPKI`，但只校验“存在”，未将该 SPKI 与调用者绑定。
- 服务账号激活会校验 `X-Client-SPKI` 与预登记公钥摘要；非环回来源的终结器头一律忽略。

## mTLS 与 TLS

- **进程内仅标准 TLS**，没有 TLCP/国密传输栈；`PRIVACY_TLS_STACK` 若设置只被 `/readyz` 回显，不改变监听行为。国密由 Go 终结器终结，见下文与 [tlcp.md](../tlcp.md)。
- REST 与 gRPC 共用 `PRIVACY_TLS_ENABLED` / `_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` / `_CLIENT_AUTH`；启用 TLS 但缺证书/私钥、或要求客户端证书但缺 CA，构造期即报错。
- `_CLIENT_AUTH=none|optional|require` 映射到 `ssl.CERT_NONE|CERT_OPTIONAL|CERT_REQUIRED` 与 gRPC `require_client_auth`。
- REST 的直连客户端证书只用于传输层校验，**不会转成 `Identity`**；REST 业务身份仍来自 Bearer。开启终结器验签后，对端证书视图可用（PoP/审计/服务账号激活），但同样不转成 REST 身份。
- mTLS 身份默认关闭（`PRIVACY_AUTH_INTERNAL_MTLS_ENABLED=false`）：通过 CA 校验只证明“持有合法证书”，不证明“有权访问本服务”。启用后 CN 必须命中白名单，否则拒绝；白名单为空时全部拒绝（fail-closed）。
- gRPC 身份提取有两条路径：直连 mTLS（`auth_context` 的 `x509_common_name`）与终结器验签 CN。直连路径拿不到叶证书，命中 `cert_fingerprints` 钉扎的条目会 fail-closed；终结器路径可核验指纹。
- 白名单 YAML 的 `default_scopes` 字段会被解析但**未接入认证决策**，未显式列出的 CN 永远拒绝，不能依赖它实现 fail-open。
- TLCP 不在 Python 进程内终结：由 Go 终结器提供，Python 只监听环回；详见 [tlcp.md](../tlcp.md)。

## 限流

- 算法：`limits` 的 `MovingWindowRateLimiter`；Key 为 `service_type:identity_name:endpoint`，REST 的 endpoint 是实际请求路径，gRPC 是完整方法名。
- 窗口换算：`window_seconds = max(1, burst / rps)`，配额为 `max(1, ceil(burst))`；因此配置是“每秒 rps、突发 burst”的近似，不是严格令牌桶。
- 未命中 `/` 覆盖端点时使用 `PRIVACY_RATE_LIMIT_DEFAULT_RPS` / `_DEFAULT_BURST`；可按路径/方法配置 `PRIVACY_RATE_LIMIT_PER_ENDPOINT_JSON`。
- 默认 `MemoryStorage` 是**进程内**状态；`PRIVACY_WORKERS > 1` 时每个 worker 独立计数，实际总配额近似乘以 worker 数。多副本需配置 Redis（`PRIVACY_RATE_LIMIT_REDIS_URL`），缺少 `redis` 包时启动期报错。
- 限流存储故障时**降级放行**并递增 `privacy_auth_denials_total{reason="rate_limit_storage_error"}`，不把基础设施故障放大为全站 500。
- 超限：REST `429 Rate limit exceeded`；gRPC `RESOURCE_EXHAUSTED`。
- 健康路径在 `PRIVACY_HEALTH_NO_RATE_LIMIT=true` 时免限流。

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_TLS_ENABLED` | `false` | 启用 REST/gRPC TLS |
| `PRIVACY_TLS_CERT_FILE` / `_KEY_FILE` | 空 | 启用 TLS 时必填 |
| `PRIVACY_TLS_CA_FILE` | 空 | 客户端证书校验 CA；`optional`/`require` 时必填 |
| `PRIVACY_TLS_CLIENT_AUTH` | `none` | `none` / `optional` / `require` |
| `PRIVACY_TLS_KEY_PASSWORD` | 空 | 私钥口令 |
| `PRIVACY_AUTH_ENABLED` | `false` | 线路层认证总开关 |
| `PRIVACY_AUTH_INTERNAL_KEYS_JSON` | `{}` | 内部静态 Key：`{"key": {"name": "...", "scopes": [...]}}` |
| `PRIVACY_AUTH_EXTERNAL_KEYS_JSON` | `{}` | 外部静态 Key |
| `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED` | `false` | gRPC 证书（直连或终结器验签）换身份 |
| `PRIVACY_TERMINATOR_IDENTITY_ENABLED` | `false` | 验签 Go 终结器身份头；开启时必须配置 HMAC Key |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE` / `_HMAC_KEY` | 空 | 验签 Key，文件优先；≥16 字节 |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO` | `sm3` | `sm3` / `sha256` |
| `PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS` | `60` | 身份时间戳允许偏差 |
| `PRIVACY_AUTH_MTLS_ALLOWED_CNS` | 空 | 静态 CN 白名单，命中授予 `["*"]` |
| `PRIVACY_AUTH_MTLS_WHITELIST_FILE` | 空 | YAML 白名单；设置后覆盖静态 CN 列表 |
| `PRIVACY_AUTH_USER_STORE_BACKEND` | 空 | `memory` / `sqlite` / `postgres`；空 = IAM 未启用 |
| `PRIVACY_AUTH_USER_STORE_FILE` | `data/auth.db` | `sqlite` 后端文件路径（该档必填） |
| `PRIVACY_AUTH_USER_STORE_DSN` | 空 | `postgres` 连接串（当前不可用） |
| `PRIVACY_AUTH_USER_STORE_ENCRYPTION_KEY` | 空 | TOTP 密文信封密钥；生产标注环境必填 |
| `PRIVACY_AUTH_USER_SESSION_TTL` | `24h` | 会话有效期（支持 `24h`/`15m`/秒） |
| `PRIVACY_AUTH_USER_SELF_REGISTER` | `false` | 是否开放 `/v1/auth/register` |
| `PRIVACY_AUTH_USER_LOGIN_THROTTLE_PER_MIN` | `20` | 登录限速 |
| `PRIVACY_AUTH_BOOTSTRAP_TOKEN` | 空 | 管理员引导令牌 |
| `PRIVACY_AUTH_BCRYPT_COST` | `12` | bcrypt 成本（4–15） |
| `PRIVACY_AUTH_SM3_ITERATIONS` | 见代码 | SM3 口令迭代次数 |
| `AUTH_TOTP_ALGORITHM` | 空（默认 `sha1`） | `sha1` / `sm3`（注意：无 `PRIVACY_` 前缀） |
| `PRIVACY_AUTH_CLUSTER_AUDIENCE` / `_SERVICE_AUDIENCE` | 空 / `privacy-engine` | 令牌受众约束 |
| `PRIVACY_AUTH_CLUSTER_TRUSTED_KEYS_JSON` | `{}` | 已解析，未用于会话验签 |
| `PRIVACY_RATE_LIMIT_ENABLED` | `false` | 限流总开关 |
| `PRIVACY_RATE_LIMIT_DEFAULT_RPS` | `10.0` | 默认速率 |
| `PRIVACY_RATE_LIMIT_DEFAULT_BURST` | `20.0` | 默认突发 |
| `PRIVACY_RATE_LIMIT_PER_ENDPOINT_JSON` | `{}` | `{"path": {"rps": 5, "burst": 10}}` |
| `PRIVACY_RATE_LIMIT_REDIS_URL` | 空 | 多进程/多副本共享限流 |
| `PRIVACY_HEALTH_NO_AUTH` | `true` | 健康探针免认证 |
| `PRIVACY_HEALTH_NO_RATE_LIMIT` | `true` | 健康探针免限流 |

完整变量口径与生产编排约束见 [config.md](../config.md)、[auth.md](../auth.md)。

## 观测

- `privacy_auth_denials_total{reason}`：`unauthenticated` / `forbidden` / `rate_limited` / `rate_limit_storage_error`。
- `privacy_auth_duration_seconds{result}`：`success` / `denied` / `exempt` / `public` / `disabled`。
- 认证/授权失败写入结构化警告日志，包含路径、身份名、拒绝码；不记录 Token 明文。
- `/metrics` 本身受 `ops:diagnostics` 保护，详见 [prometheus.md](../prometheus.md)。

## 测试

| 文件 | 覆盖 |
|---|---|
| `tests/security/test_security_auth.py` | REST 认证、公开路径、静态 Key、权限拒绝、凭据策略 |
| `tests/security/test_security_grpc.py` | gRPC 拦截器、mTLS 身份、健康方法 |
| `tests/security/test_security_rate_limit.py` | REST/gRPC 限流、存储故障降级 |
| `tests/security/test_security_whitelist.py` | CN 白名单、per-CN scope、热重载、失败保留旧配置 |
| `tests/security/test_security_tls.py` / `test_security_urllib_tls.py` | TLS 参数构造与客户端信任 |
| `tests/security/test_security_unit.py` | 配置解析、Bearer 解析等单元行为 |
| `tests/security/test_security_terminator.py` / `test_security_terminator_auth.py` | 终结器验签、环回限制、skew、SPKI-DER 一致性、REST/gRPC 接线 |
| `tests/security/test_security_pop.py` | PoP 绑定状态机与拒绝形态 |
| `tests/security/test_ops_port.py` | 运维口白名单、端口推导与 TLS 档位 |
| `tests/auth/*` | IAM：登录锁定、TOTP/恢复码/换绑、动态 Key、令牌交换、SQLite 运行时、凭据策略 |

## 与 Go 的差异

| 能力 | Go | Python |
|---|---|---|
| IAM 路由 | 始终由 `RegisterUserRoutes` 注册，用户库必须显式配置 | 仅配置后端时挂载；未配置则 404 |
| 用户库后端 | SQLite / PostgreSQL | memory / SQLite；PostgreSQL 未构建 |
| 集群会话 / 委托令牌验签 | 证书私钥签名 + 集群校验器 | 未实现；交换票据仅进程内 |
| PoP 公钥持有证明 | 有 | 有，需 Go 终结器签名身份；直连 mTLS 无签名头时 fail-closed |
| mTLS 身份 | REST + gRPC | 仅 gRPC（直连或终结器验签）；REST 对端视图仅用于 PoP/审计/激活 |
| 限流 | 32 分片令牌桶、跨实例能力 | `limits` 滑动窗口、进程内存 / Redis |
| TLCP | 支持 | 进程内不支持；由 Go 终结器终结，Python 监听环回 |
| 权限映射 | 中间件集中映射 | REST 显式依赖；gRPC 集中映射 |
| 审计 | 完整审计链 | IAM 仅向 SQLite 仓储写脱敏 outbox 事件（`PRIVACY_AUDIT_LOG_URL` 已解析但未外发）；线路层拒绝只记指标/日志 |
