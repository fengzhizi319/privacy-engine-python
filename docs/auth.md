# 认证、授权与 IAM/MFA

> **组件归属**：`services/privacy-engine-python`
> **对照实现**：Go `services/privacy-engine` 的 `pkg/auth` + `pkg/middleware` 认证栈
> **代码位置**：线路层 `engine/security/`；IAM 内核 `engine/auth/`
> **状态**：线路层已全量接线（含 Go 终结器身份验签）；IAM/MFA 在配置用户库后端后**条件接线**；`tests/auth/` 与 `tests/security/test_security_terminator*.py` 已覆盖登录、MFA 换绑、SQLite 运行时、终结器身份与凭据策略

本文说明 Python 引擎当前真实生效的认证/授权能力、IAM 条件接线的边界，以及两者与 Go 之间的差距。阅读时请以「代码事实」为准，不要按 Go 的部署文档推断 Python 侧行为。

---

## 1. 两层认证模型

Python 引擎有两层认证代码，配置用户库后串联工作：

| 层 | 目录 | 身份来源 | 接线状态 | 作用面 |
|---|---|---|---|---|
| **线路层 (wire-level)** | `engine/security/` | 静态 API Key、gRPC mTLS 客户端证书、Go 终结器签名身份、IAM 动态身份 | ✅ 已接线（REST 依赖 + gRPC 拦截器 + `/metrics` ASGI 中间件） | 每个业务请求 |
| **IAM/MFA 内核** | `engine/auth/` | 用户库 + 登录会话 + 动态 API Key + TOTP | 🟡 **条件接线**：`PRIVACY_AUTH_USER_STORE_BACKEND` 非空时挂载 `/v1/auth/*`，并接入 Bearer 解析 | 认证层 + 用户管理面 |

```text
REST 请求 ──► ApiKeyAuthAsgiMiddleware（仅 /metrics，先验签终结器身份头）
          └─► SECURITY_DEPS = get_current_identity + rate_limit_dependency
                    │
                    ├─ PRIVACY_AUTH_ENABLED=false → DEV_MODE_IDENTITY（internal, ["*","admin"]）
                    ├─ 终结器身份头（环回 + HMAC 验签）→ request.state.terminator_peer
                    │     （仅用于 PoP/审计/服务账号激活，不转成 REST 业务身份）
                    └─ true → Bearer → IAM resolve（若已配置）→ 静态 Key
                                  → 凭据策略（含 PoP）→ 沙箱 → require_permission("...")

gRPC 请求 ──► GrpcObservabilityInterceptor ──► AuthInterceptor ──► RateLimitInterceptor ──► Servicer
                                                    │
                                                    ├─ 直连 mTLS 证书 CN（PRIVACY_AUTH_INTERNAL_MTLS_ENABLED）
                                                    ├─ 终结器验签 CN → internal 身份（同上开关）
                                                    └─ metadata authorization: Bearer <IAM 凭据或静态 Key>
```

**关键结论**：

- 未配置 `PRIVACY_AUTH_USER_STORE_BACKEND` 时，`engine/auth/` 不参与运行时，登录会话与动态 Key 不会被接受，`/v1/auth/*` 返回 404。
- 配置后，`get_current_identity` 先调用 `AuthService.resolve_token`（会话 → 动态 Key → 交换票据），未命中再回退静态 Key 表；IAM 身份携带 `subject` / `role` / `tenant_id` / 会话 MFA 状态等字段，沙箱与 ABAC 因此生效。
- IAM 路由是否挂载**只看用户库后端是否配置**，不看 `PRIVACY_AUTH_ENABLED`。后端已配置但认证关闭时，用户管理接口会接受开发身份，属于部署红线（见 §3.3）。

---

## 2. 线路层认证（已生效）

### 2.1 开关与环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_AUTH_ENABLED` | `false` | 线路层总开关。关闭时所有请求获得开发身份 `["*","admin"]`，限流按 `internal:dev-anonymous` 计 |
| `PRIVACY_AUTH_INTERNAL_KEYS_JSON` | `{}` | 内部服务静态 Key 表；JSON 对象，非法 JSON 静默回退为空 |
| `PRIVACY_AUTH_EXTERNAL_KEYS_JSON` | `{}` | 外部/公开客户端静态 Key 表 |
| `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED` | `false` | 是否允许 gRPC 证书（直连 mTLS 或终结器验签 CN）换取内部身份；REST 不把对端证书转成业务身份 |
| `PRIVACY_AUTH_MTLS_ALLOWED_CNS` | 空 | 静态 CN 白名单（JSON 数组或逗号分隔），命中即授予 `["*"]` |
| `PRIVACY_AUTH_MTLS_WHITELIST_FILE` | 空 | YAML 白名单；设置后**覆盖**静态 CN 列表，支持 per-CN scope、证书指纹钉扎与 mtime 热重载 |
| `PRIVACY_TERMINATOR_IDENTITY_ENABLED` | `false` | 验签 Go 终结器注入的身份头；开启但未配置 Key 时启动/构造期失败 |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE` / `_HMAC_KEY` | 空 | 验签 Key，**文件优先**于环境变量；长度 ≥16 字节，支持 mtime 热加载 |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO` | `sm3` | `sm3` / `sha256`，必须与终结器一致 |
| `PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS` | `60` | 身份时间戳允许偏差 |
| `PRIVACY_RATE_LIMIT_ENABLED` | `false` | 限流总开关 |
| `PRIVACY_RATE_LIMIT_DEFAULT_RPS` | `10.0` | 默认持续速率 |
| `PRIVACY_RATE_LIMIT_DEFAULT_BURST` | `20.0` | 默认突发配额 |
| `PRIVACY_RATE_LIMIT_PER_ENDPOINT_JSON` | `{}` | 按 REST 实际路径或 gRPC 完整方法名覆盖 `{rps, burst}` |
| `PRIVACY_RATE_LIMIT_REDIS_URL` | 空 | 多副本共享限流；配置后需安装 `redis` 包，否则启动即报错 |
| `PRIVACY_HEALTH_NO_AUTH` | `true` | 标准健康路径免认证 |
| `PRIVACY_HEALTH_NO_RATE_LIMIT` | `true` | 标准健康路径免限流 |

所有变量由 `engine/security/config.py::get_security_settings()` **每次调用重新解析**，因此单测可通过修改 `os.environ` 切换行为；`ratelimit.get_limiter()` 在设置变化时重建限流器。

### 2.2 静态 Key 格式

```json
{
  "psk_internal_example": {
    "name": "service-hub",
    "scopes": ["privacy:mask", "privacy:dp", "health:read"]
  },
  "psk_external_example": {
    "name": "partner-portal",
    "scopes": ["privacy:mask", "health:read"]
  }
}
```

- Key 是 JSON 对象的**键**，不是值字段；值只包含 `name` 与 `scopes`。
- 内部表先于外部表匹配，内部 Key 不会被同名外部 Key 覆盖。
- 比对采用排序遍历 + `hmac.compare_digest` 的常量时间查找（`_constant_time_lookup`），不短路。
- 请求头格式严格为 `Authorization: Bearer <key>`；`Bearer` 大小写不敏感，缺 token 或裸 token 均视为未认证。

### 2.3 身份与 scope 语义

`engine/security/identity.py::Identity` 为冻结 dataclass，字段包括 `service_type`（`internal` / `external` / `public`）、`name`、`scopes`，以及 IAM 身份会填充的 `subject` / `role` / `principal_type` / `tenant_id` / `mfa_authenticated` / `session_auth` / `audience` / `binding_status` / `contract_version` / `contract_allowed_paths` 等。静态 Key 与 mTLS 身份不填充 IAM 字段。

`has_permission` 规则（与 Go `Identity.HasPermission` 同口径）：

| 规则 | 示例 |
|---|---|
| 精确匹配，无层级/通配扩展 | `privacy:mask` 不隐含 `privacy:mask:batch` |
| `["*"]` 授予全部**业务**权限 | 但不穿透 `audit:` 前缀（三权分立） |
| `audit:` 前缀需显式授权 | `audit:*` 可覆盖；裸 `*` 不可 |
| `"a\|b"` 复合权限按逻辑或拆分 | gRPC 映射与中间件复用 |

### 2.4 REST 路由权限

所有业务路由在装饰器中**显式**声明权限，`permission_for_rest_path()` 只是文档参考，未接线。当前声明分布：

| 权限 | 覆盖路由（示例） |
|---|---|
| `privacy:mask` | `/v1/privacy/mask`、`mask_record`、`mask/batch`、`mask/dataframe`、`/v1/privacy/process_file`、`/v1/privacy/file/*` |
| `privacy:hash` | `/v1/privacy/hash` |
| `privacy:dp` | `/v1/privacy/dp/*`（17 个）与 `/v1/privacy/ldp/*`（6 个），含 chunked / aggregate / ORR |
| `privacy:kano` | `/v1/privacy/k_anonymize/*` |
| `privacy:qol` | `/v1/privacy/qol/obfuscate`、`.../batch` |
| `privacy:budget` | `GET/POST/DELETE /v1/privacy/budget`、`/budget/reset` |
| `privacy:profile` | `/v1/privacy/profile/*` 参数推荐/画像接口 |
| `dynclassification:read` | `/v1/dynclassification/*` 的 classify/eval/dry_run/standards/domains/operators/validate |
| `dynclassification:write` | `/v1/dynclassification/profiles/reload`、`/generate_profile` |
| `medical:process` | `/v1/medical/*` 三个入口 |
| `agent:process` 或 `medical:process` | `/v1/agent/*` 使用 `require_any_permission` |
| `pipeline:process` | `/v1/pipeline/*` |
| `ops:diagnostics` | `/v1/ops/*`、`/ops/diagnostics`、`/metrics`、`/openapi.json`、`/docs/openapi.json` |
| — | 健康端点（`/health`、`/v1/privacy/health`、`/livez`、`/readyz`、`/readyz/llm`）仅认证+限流，无 scope 校验 |

`/v1/ops/diagnostics?refresh=true` 会真实加载模型，端点内另行要求 `ops:admin`。

### 2.5 gRPC 方法权限

`permission_for_grpc_method()` 按完整方法名的最后一段查表，**已接线**于 `AuthInterceptor`：

| 方法与权限 | 说明 |
|---|---|
| `Mask*` / `Hash` | `privacy:mask` / `privacy:hash` |
| `DP*`、`KAnonymize*`、`ObfuscateQuery*` | `privacy:dp`、`privacy:kano`、`privacy:qol` |
| `PerturbBinaryBatch`、`EstimateCategoricalHistogram` 等 LDP | `privacy:dp` |
| `Classify*` | `classification:read` |
| `DynClassify` | `dynclassification:read` |
| `RecommendParams` | `privacy:profile` |
| `Health` | `health:read` |
| **未登记方法** | 回退 `"*"`，只有通配身份可调用 |

> ⚠️ 未登记方法回退 `"*"` 意味着只持有具体 scope 的 Key 会被拒绝；新增 gRPC 方法时必须同步登记映射。

### 2.6 gRPC mTLS、终结器身份与 CN 白名单

**直连 mTLS**（gRPC 自身终结 TLS）：

- 仅在 `auth_context["transport_security_type"] == b"ssl"` 时提取 `x509_common_name`。
- 证书通过 CA 校验只证明「持有合法证书」，不证明「有权访问本服务」，因此 `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED` 默认关闭。
- 白名单为空 → **拒绝所有证书**（fail-closed）。
- 未列出 CN 一律拒绝；`WhitelistConfig.default_scopes` 虽被解析，但**未接入判定路径**，不要依赖它实现兜底放行。
- 热重载为请求驱动的 mtime 被动检查；YAML 解析失败时保留旧缓存并记录 `last_error`，不会清空白名单。
- 直连 `auth_context` 不暴露叶证书，无法核验 `cert_fingerprints`；命中钉扎条目的直连请求 **fail-closed** 返回 `403 PERMISSION_DENIED`，必须走终结器身份路径。

**Go 终结器身份**（TLCP / TLS 由终结器终结，Python 监听环回）：

- `PRIVACY_TERMINATOR_IDENTITY_ENABLED=true` 时，REST 与 gRPC 都校验终结器注入的 `X-PrivShield-Client-*` / `X-PrivShield-Identity-*` 头：仅接受**环回对端**、时间戳在 skew 内、HMAC-SM3/SHA256 常量时间比对签名、`SPKI-DER` 与已签名 SPKI 摘要一致。
- 头**成组出现**：任一身份头存在但不完整/伪造/超时/来源非回环 → `401`，绝不静默降级为「无证书」；完全无身份头则视为未走终结器路径。
- REST：验签结果只写入 `request.state.terminator_peer`，用于 PoP、审计与服务账号激活（`_terminator_spki`）；**业务身份仍由 Bearer 决定**，与 Go 的 REST 语义一致。
- gRPC：验签 CN 在 `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED=true` 时按白名单换取 `internal` 身份，并可校验指纹钉扎；关闭该开关时仅参与 PoP。身份优先级：直连 mTLS → 终结器身份 → 健康豁免 → Bearer。
- 终结器与引擎的 `PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO` 与 Key 必须一致；轮换 Key 需要协调重启或在文件热加载窗口内完成。

YAML 示例：

```yaml
version: "1.0"
entries:
  - cn: service-hub
    scopes: [hub:dispatch, privacy:mask, privacy:dp]
    description: 调度中枢
    enabled: true
  - cn: retired-client
    scopes: ["*"]
    enabled: false        # 禁用项加载时直接丢弃
default_scopes: []        # 仅解析，不生效
```

### 2.7 `/metrics` 的 ASGI 认证

`/metrics` 通过 `app.mount()` 挂载，绕过 FastAPI 依赖注入，因此单独包裹 `ApiKeyAuthAsgiMiddleware`：

| 场景 | 行为 |
|---|---|
| `PRIVACY_AUTH_ENABLED=false` | 完全透传 |
| 认证开启 + Key 缺失/无效 | `401 text/plain: Unauthorized: missing or invalid credentials` |
| 认证开启 + Key 有效但无 `ops:diagnostics` 或 `*` | `403 text/plain: Forbidden: insufficient scope for metrics access` |
| 认证开启 + 权限满足 | 透传至 Prometheus ASGI app |

该中间件与 REST 共用同一套 Bearer 解析、凭据策略和沙箱，因此 IAM 会话/动态 Key 在 `/metrics` 上同样生效。注意：`/metrics` 不在健康白名单内，健康豁免不适用。

### 2.8 限流语义

- 算法：`limits` 库 `MovingWindowRateLimiter`；窗口长度 = `max(1, burst / rps)` 秒，配额为 `max(1, ceil(burst))`。
- Key：`f"{service_type}:{name}:{endpoint}"`。REST 的 `endpoint` 是**实际请求路径**（因此 `/v1/privacy/mask` 与 `/v1/privacy/mask_record` 分别计数），gRPC 是完整方法名。
- 认证关闭时按 `internal:dev-anonymous` 计；公开认证路径未带凭据时按 `public:public-caller` 计。
- 存储后端异常（如 Redis 断连）**fail-open 放行**，并递增 `privacy_auth_denials_total{reason="rate_limit_storage_error"}`。
- REST 超限 `429`；gRPC 超限 `RESOURCE_EXHAUSTED`。
- 多进程 `engine.launcher` 下，内存限流与静态 Key 均为**每进程独立**，全局限速需 Redis。
- IAM 路由（`/v1/auth/*`）不挂通用限流依赖，登录/激活由 `AuthService` 内部按主体与 IP 限速。

### 2.9 错误信封

| HTTP | detail | 触发 |
|---|---|---|
| 401 | `Unauthorized: missing credentials` | 无 `Authorization` 或非 Bearer |
| 401 | `Unauthorized: invalid credentials` | 凭据不在 IAM 或静态表 |
| 403 | `Forbidden: insufficient scope` | scope 不满足 |
| 403 | `AUDIENCE_MISMATCH` / `CONTRACT_VIOLATION` / `CREDENTIAL_DRAIN_ONLY` | 凭据策略拒绝 |
| 429 | `Rate limit exceeded` | 限流命中 |
| gRPC | `UNAUTHENTICATED` / `PERMISSION_DENIED` / `RESOURCE_EXHAUSTED` | 同名语义 |

认证失败与授权失败统一计入 `privacy_auth_denials_total`，并写入结构化日志（reason 为 `unauthenticated` / `forbidden` / `rate_limited` 等）。

---

## 3. IAM/MFA（条件接线）

### 3.1 挂载与启动

`engine/main.py` 在导入期判断：

```python
_auth_user_store_enabled = AuthSettings.from_env().enabled  # backend 非空
if _auth_user_store_enabled:
    app.include_router(auth_router)             # /v1/auth/*
    app.include_router(auth_well_known_router)  # /v1/.well-known/service-capabilities/*
```

启动 lifespan 中若后端已配置，会调用 `get_auth_runtime()` 强制构建；配置错误在启动期失败，不静默降级：

| 后端 | 行为 |
|---|---|
| 未设置 | IAM 未启用；`get_auth_runtime()` 返回 `None`，只使用静态 Key / mTLS |
| `memory` | 进程内存用户库；重启即丢失，多 worker 不共享 |
| `sqlite` | 需 `PRIVACY_AUTH_USER_STORE_FILE`；直接执行 `deploy/sql/auth/0001–0007.up.sql` |
| `postgres` | 当前构建无 psycopg 驱动，启动期抛 `RuntimeError`，拒绝启动 |

公开认证路径（未带凭据时以 `public-caller` 放行，具体授权在 Handler 内）：`/v1/auth/login`、`/v1/auth/register`、`/v1/auth/users/register`、`/v1/auth/users/bootstrap`、`/v1/auth/service-accounts/activate`、`/v1/.well-known/service-capabilities/public`。

### 3.2 能力清单

`engine/auth/` 覆盖与 Go `pkg/auth` 对齐的用户全生命周期：

| 模块 | 能力 |
|---|---|
| `service.py` | 登录/登出、公开注册、管理员建号、改密/重置、用户 CRUD/状态、MFA setup/enable/cancel/status/disable/regenerate、API Key 签发/列表/吊销、服务账号 CRUD/冻结/配额验收、管理员 step-up、审计员保护（三权分立）、token exchange/revoke、能力发现/协商 |
| `store.py` | 5 次失败锁定 15 分钟、登录节流、会话（上限 8）、用户 API Key（上限 32，TTL 默认 30 天、上限 90 天）、SM3 摘要、角色默认 scope |
| `password.py` | SM3 v2 迭代哈希（16B salt，1,000,000 轮）+ bcrypt(cost=12) 双存、弱口令/键盘序检测、长度 8–72 |
| `totp.py` | RFC 6238 六位/30 秒、±1 窗口、严格单调步长防重放；SHA1 或 SM3 |
| `recovery.py` | 8 组 `XXXX-XXXX-XXXX` 一次性恢复码，仅存 SM3 摘要与 4 位前缀 |
| `persistence.py` | SQLite 权威持久化；TOTP 密钥 SM4-GCM 信封加密（`sm4gcm:v1:`），未配置密钥时在库旁生成 0600 `.key` |
| `runtime.py` | 进程级运行时装配；未配置后端返回 `None`；PostgreSQL 因缺 `psycopg` 显式失败 |
| `bound_pubkey.py` | 服务账号激活的 SPKI 解析、格式化与摘要 |
| `contract.py` / `idempotency.py` | 凭据契约解析与路径匹配、动态 Key 签发幂等 |

请求级 PoP 不在 `engine/auth/`，而在线路层 `engine/security/pop.py`，由 `_enforce_credential_policy` 在 REST/gRPC/`/metrics` 三面统一执行；对端证书视图来自 Go 终结器签名身份。

会话沙箱按优先级生效：`PASSWORD_CHANGE_REQUIRED` → `PASSWORD_EXPIRED_REQUIRE_CHANGE` → `MFA_ENROLLMENT_REQUIRED` → `MFA_SESSION_REQUIRED`。恢复码登录只签发最长 10 分钟、仅可走换绑流程的沙箱会话，并立即作废旧 TOTP 绑定与剩余恢复码。

### 3.3 部署红线

`/v1/auth/*` 的挂载**只取决于用户库后端是否配置**，不取决于 `PRIVACY_AUTH_ENABLED`：

- 配置后端 + 关闭认证 → `get_current_identity` 返回带 `*`/`admin` 的开发身份，用户管理接口对所有调用者开放。
- 生产必须同时设置 `PRIVACY_AUTH_ENABLED=true` 与用户库后端，并在网络层限制 `/v1/auth/*` 的暴露面。
- 未配置后端时不要调用 IAM API；路由不存在，运行时也不会构建。

### 3.4 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_AUTH_USER_STORE_BACKEND` | 空 | `memory` / `sqlite` / `postgres`；**空表示 IAM 未启用**，未知值启动期拒绝 |
| `PRIVACY_AUTH_USER_STORE_FILE` | `data/auth.db` | SQLite 路径；`sqlite` 档必填 |
| `PRIVACY_AUTH_USER_STORE_DSN` | 空 | PostgreSQL DSN（当前构建不可用） |
| `PRIVACY_AUTH_USER_STORE_ENCRYPTION_KEY` | 空 | SM4-GCM 信封密钥；生产标注环境必填，开发态空则生成旁路 `.key` |
| `PRIVACY_AUTH_USER_SELF_REGISTER` | `false` | 是否开放公开注册 |
| `PRIVACY_AUTH_USER_SESSION_TTL` | `24h` | 会话 TTL；支持 Go 风格 `30s/15m/24h` |
| `PRIVACY_AUTH_USER_LOGIN_THROTTLE_PER_MIN` | `20` | 每 IP 登录节流 |
| `PRIVACY_AUTH_USER_PASSWORD_EXPIRE_BLOCK` | `false` | 口令过期是否阻断业务 |
| `PRIVACY_AUTH_USER_PASSWORD_EXPIRE_GRACE_DAYS` | `0` | 过期宽限期 |
| `PRIVACY_AUTH_ADMIN_STEPUP_REQUIRED` | `true` | 管理员敏感操作是否要求 TOTP 复核 |
| `PRIVACY_AUTH_BOOTSTRAP_TOKEN` | 空 | 首个管理员引导令牌 |
| `PRIVACY_AUTH_USER_BOOTSTRAP_REMOTE_ALLOWED` | `false` | 是否允许远程引导 |
| `PRIVACY_AUTH_OUTBOX_ENABLED` | `true` | 写入 SQLite 脱敏事件 outbox |
| `PRIVACY_AUTH_BCRYPT_COST` | `12` | 仅接受 4–15 |
| `PRIVACY_AUTH_SM3_ITERATIONS` | `1000000` | SM3 迭代轮数，必须 > 0 |
| `AUTH_TOTP_ALGORITHM` | 空 | 空 → SHA1；可配 `sha1` / `sm3`（注意无 `PRIVACY_` 前缀） |
| `PRIVACY_AUTH_CLUSTER_SIGNING_KEY_FILE` | 空 | 已解析；当前未用于会话/委托令牌验签 |
| `PRIVACY_AUTH_CLUSTER_ISSUER` | `privacy-engine` | 能力协商 issuer |
| `PRIVACY_AUTH_CLUSTER_AUDIENCE` | 空 | 默认受众约束 |
| `PRIVACY_AUTH_CLUSTER_TENANT_ID` | 空 | 租户标识 |
| `PRIVACY_AUTH_CLUSTER_TRUSTED_KEYS_JSON` | `{}` | 已解析；当前未用于验签 |
| `PRIVACY_AUTH_SERVICE_AUDIENCE` | `privacy-engine` | 入站凭据受众校验目标 |
| `PRIVACY_AUDIT_LOG_URL` | 空 | 已解析；当前未外发审计 |
| `PRIVACY_AUDIT_LOG_FAIL_CLOSED` | `false` | 已解析；当前未接入审计失败阻断 |

### 3.5 令牌与凭据边界

- 交换票据保存在 `AuthService._issued_tokens`（进程内存），重启丢失，多 worker 之间不共享。
- 生产标注环境下 `POST /v1/auth/token-exchange` 要求 TLS 终结器从**环回**注入 SPKI 材料。开启 `PRIVACY_TERMINATOR_IDENTITY_ENABLED` 时只信任已验签的身份头（签名覆盖 SPKI 摘要，raw DER 需与摘要一致），SPKI 因此与调用者证书绑定；未开启时才回退到旧式环回 `X-Client-SPKI`。非环回来源的终结器头一律忽略。
- 服务账号激活会校验 SPKI（验签路径取自已签名头，否则取旧式环回 `X-Client-SPKI`）与预登记公钥摘要；`X-Client-SPIFFE` 参与激活绑定（Go 终结器不签名 SPIFFE，验签路径下恒为空）。
- `negotiate_capabilities` 的 `signature` 字段是 manifest 的 SHA-256 摘要，不是私钥签名，不具备证明力。
- `public_capabilities()` 仍宣称 `tls_modes: ["tls", "tlcp"]`，与 Python 仅支持标准 TLS 的事实不符。

### 3.6 测试状态

`tests/auth/` 覆盖 IAM 主路径：登录锁定、TOTP 单调步/重放、恢复码核销、换绑沙箱、动态 Key 签发/吊销、令牌交换受众约束与 SQLite 重启恢复（`test_mfa_flow.py`、`test_routes_parity.py`、`test_sqlite_runtime.py`、`test_credential_policy.py`）。`tests/security/test_security_terminator*.py`、`test_security_pop.py` 覆盖终结器验签、环回限制、skew、SPKI-DER 一致性与 PoP 判定。

仍需人工/集成验证的边界：Redis 限流多实例一致性、集群会话/委托令牌验签（Python 未实现）、PostgreSQL 用户库（启动期拒绝）。

---

## 4. 与 Go 的差距

| 能力 | Go `privacy-engine` | Python |
|---|---|---|
| 静态 API Key | ✅ | ✅ |
| gRPC mTLS CN 白名单 + per-CN scope | ✅ | ✅ |
| CN 白名单热重载 | ✅ | ✅（请求驱动 mtime） |
| REST mTLS 身份 | 由网关/前置终结器承担 | ✅ 终结器验签对端（仅 PoP/审计/服务账号激活；业务身份仍 Bearer） |
| 限流 | 32 分片令牌桶 | ✅ `limits` 滑动窗口（内存/Redis） |
| 用户库 + 登录会话 | ✅ `/v1/auth/*` | 🟡 配置后端后可用 |
| TOTP / 恢复码 / 换绑 | ✅ | 🟡 配置后端后可用 |
| 动态 API Key / 服务账号 / 配额 | ✅ | 🟡 配置后端后可用 |
| PostgreSQL 用户库 | ✅ | ❌ 启动期拒绝 |
| 集群会话 / 委托令牌验签 | ✅ 证书私钥签名 + 集群校验器 | ❌ 未实现 |
| PoP 公钥持有证明 | ✅ | ✅ 需终结器签名身份（直连 mTLS 无签名头时绑定凭据 fail-closed） |
| 密钥热加载 | ✅ 双通道活密钥 | ❌ 静态 JSON 仅在 `get_security_settings()` 重新解析时生效（进程内环境变量不变则不变） |
| 网关入站认证 | ✅ | ❌ Python 网关无任何认证代码 |

---

## 5. 运维与验证

### 5.1 本地启用静态 Key

```bash
export PRIVACY_AUTH_ENABLED=true
export PRIVACY_AUTH_INTERNAL_KEYS_JSON='{"psk_dev_internal":{"name":"dev","scopes":["*"]}}'
export PRIVACY_AUTH_EXTERNAL_KEYS_JSON='{"psk_dev_low":{"name":"low","scopes":["privacy:mask"]}}'
python -m engine.server

# 无凭据 → 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8079/v1/privacy/mask \
  -H 'content-type: application/json' -d '{"record":{"name":"张三"}}'

# 低权限 Key 调用 DP → 403
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8079/v1/privacy/dp/count \
  -H 'authorization: Bearer psk_dev_low' \
  -H 'content-type: application/json' -d '{"values":[1,2,3],"epsilon":1.0}'

# 全权限 Key → 200
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8079/v1/privacy/mask \
  -H 'authorization: Bearer psk_dev_internal' \
  -H 'content-type: application/json' -d '{"record":{"name":"张三"}}'
```

### 5.2 验证 `/metrics` scope

```bash
# 低权限 Key（无 ops:diagnostics）→ 403
curl -s -o /dev/null -w '%{http_code}\n' localhost:8079/metrics -H 'authorization: Bearer psk_dev_low'
# 内部全权限 Key → 200
curl -s -o /dev/null -w '%{http_code}\n' localhost:8079/metrics -H 'authorization: Bearer psk_dev_internal'
```

### 5.3 验证 gRPC mTLS 与终结器身份

直连 mTLS：

1. 启动 gRPC 时设置 `PRIVACY_TLS_ENABLED=true`、`PRIVACY_TLS_CERT_FILE`、`PRIVACY_TLS_KEY_FILE`、`PRIVACY_TLS_CA_FILE`、`PRIVACY_TLS_CLIENT_AUTH=require`。
2. 设置 `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED=true` 与 `PRIVACY_AUTH_MTLS_WHITELIST_FILE`。
3. 使用白名单 CN 的客户端证书应成功；换用同 CA 签发但 CN 未列出的证书应收到 `UNAUTHENTICATED`。

终结器拓扑（Python 监听回环，TLCP/TLS 由 Go 终结器终结）：

1. 终结器与引擎配置相同的 `PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO` 与 HMAC Key（Key 文件优先）。
2. 引擎设置 `PRIVACY_TERMINATOR_IDENTITY_ENABLED=true`；未配置 Key 时构造期即失败。
3. 经终结器发起的请求应携带完整的 `X-PrivShield-Client-*` / `X-PrivShield-Identity-*` 头；直连 Python 回环口伪造任一身份头 → `401`，篡改时间戳/签名/SPKI-DER → `401`。
4. `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED=true` 时，终结器验签 CN 命中白名单的 gRPC 调用获得 `internal` 身份；关闭时同一调用仍需 Bearer。
5. 验证命令：`tests/security/test_security_terminator_auth.py` 与 `make terminator-check`（Go 终结器验签实现）。

### 5.4 验证 IAM 条件接线

```bash
# 未配置后端：路由不存在
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8079/v1/auth/login \
  -H 'content-type: application/json' -d '{"username":"admin","password":"x"}'
# 预期：404

# 配置 SQLite + 认证后：登录可用；错误口令为 401
export PRIVACY_AUTH_ENABLED=true
export PRIVACY_AUTH_USER_STORE_BACKEND=sqlite
export PRIVACY_AUTH_USER_STORE_FILE=/tmp/privshield-auth.db
curl -s -X POST localhost:8079/v1/auth/login \
  -H 'content-type: application/json' -d '{"username":"admin","password":"wrong"}'
# 预期：401 invalid credentials（或账号未引导）

# 启动日志应包含 auth runtime initialized，backend=sqlite
```

> 首次管理员通过 `/v1/auth/users/bootstrap` 创建，需要 `PRIVACY_AUTH_BOOTSTRAP_TOKEN` 或环回/远程引导许可。测试后删除临时 SQLite 文件与旁路 `.key`。

---

## 6. 已知不一致

| 位置 | 现象 | 影响 |
|---|---|---|
| `engine/auth/service.py::negotiate_capabilities` | `signature` 字段为 manifest 的 SHA-256 摘要，非私钥签名 | 不具备实际证明力 |
| `PRIVACY_AUTH_MTLS_ALLOWED_CNS` | 静态模式固定授予 `["*"]` | 无法按 CN 最小权限；需要 per-CN scope 时必须用 YAML |
| 直连 gRPC mTLS | `auth_context` 不暴露叶证书，无法核验指纹钉扎 | 命中钉扎条目的直连请求 fail-closed；需走终结器身份路径 |
| Python 侧未实现集群会话/委托令牌验签 | `PRIVACY_AUTH_CLUSTER_*` 仅解析，交换票据为进程内存 | 多实例共享会话仍需 Go 侧或网关承担 |
| 旧式 `X-Client-SPIFFE` | Go 终结器不签名 SPIFFE | 开启验签后恒为空，依赖 SPIFFE 的部署不可用 |

---

## 7. 相关文档

- [架构总览](./architecture.md)
- [配置参考](./config.md)
- [安全模块 README](./security/README.md)
- [TLCP 与传输层](./tlcp.md)
- [策略清单](./policy_bundles.md)
- [运维手册](./ops.md)
- [Prometheus 指标](./prometheus.md)
- [测试指南](./testing.md)
