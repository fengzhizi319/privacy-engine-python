# 传输层：标准 TLS 与 TLCP 终结器

> **组件归属**：`services/privacy-engine-python`
> **Go 对照**：`services/privacy-engine`（默认 `PRIVACY_TLS_STACK=tlcp`，GM/T 0024 双证书）
> **状态**：❌ Python 进程内不实现 TLCP；✅ 网关 `PRIVACY_GATEWAY_TLS_STACK` 三档校验；✅ Go 终结器 [`terminator/`](../terminator/README.md) 对外提供国密 mTLS（已实现并接入 Python 验签）

本文说明 Python 引擎与网关真实的传输能力、`*_TLS_STACK` 的语义、国密场景的部署拓扑与验证方法。

---

## 1. 一句话结论

Python 进程内**没有任何国密传输实现**：

- `ssl.PROTOCOL_TLS_SERVER` / `grpc.ssl_server_credentials` 只协商 TLS 1.2/1.3；
- 没有 SM2 签名/加密双证书、没有 GM/T 0024 握手、没有 SM3/SM4 密码套件选择；
- 引擎的 `PRIVACY_TLS_STACK` 仍只被 `/readyz` 回显；
- 网关的 `PRIVACY_GATEWAY_TLS_STACK` **参与启动校验**：`tlcp|both|tls` 三档，`tls` 才由 Python 终止，`tlcp|both` 要求绑定环回并由 Go 终结器终结；
- 引擎的 Go 终结器 [`terminator/`](../terminator/README.md) 提供 `tlcp|both|tls` 三档 mTLS、SM2 双证书、CN/SAN 白名单与证书指纹钉扎，并以 HMAC 签名身份头把对端证书身份传给 Python；
- `AuthService.public_capabilities()` 只声明 `tls_modes: ["tls"]`，不虚报进程内 TLCP。

任何要求「客户端到 Python 引擎全程国密」的合规场景，都必须在前置 Go 终结器（或 Go 网关）完成握手。

---

## 2. Go 基线（对照）

Go 引擎默认开启国密传输（`PRIVACY_TLS_STACK=tlcp`）：

| 档位 | 语义 |
|---|---|
| `tlcp`（默认） | 仅国密 TLCP（GM/T 0024），必须提供 SM2 **签名证书对 + 加密证书对**，缺失拒绝启动 |
| `both` | 同端口自适应国密与 TLS 1.3 |
| `tls` | 仅国际标准 TLS |

三档均为 **mTLS**，无明文档；运维口为业务口 +1000；`/readyz` 的 `tls_stack` 字段报告实际生效档位。证书公钥指纹统一采用 `spki=sm3:<64hex>` 或 `spki=sha256:`。

Python 侧能力对照：

| 项目 | Go | Python |
|---|---|---|
| TLCP / GM/T 0024 | ✅ 进程内 | ✅ 由 Go 终结器提供（进程内 ❌） |
| SM2 双证书 | ✅ | ✅ 终结器 |
| `both` 自适应 | ✅ | ✅ 终结器 |
| 引擎三档 mTLS | ✅ | 标准 TLS 的 `none/optional/require`；三档 mTLS 由终结器提供 |
| 引擎 `PRIVACY_TLS_STACK` | 选择协议栈 | 仅回显（终结器同名变量选择协议栈） |
| 网关 `PRIVACY_GATEWAY_TLS_STACK` | 选择协议栈 | 校验档位；`tls` 自终止，`tlcp/both` 必须环回并由终结器前置 |
| 引擎运维口 +1000 | ✅ | ✅（`:9079`，独立 TLS 档位；`tlcp/both` 时回环并由终结器前置） |
| 网关运维口 +1000 | ✅ | ✅（`:9000`，独立 TLS 档位） |

---

## 3. Python 标准 TLS 配置矩阵

### 3.1 主引擎（REST `:8079` / gRPC `:50051`）

变量定义见 [`config.md`](./config.md) §3 与 `engine/security/config.py`：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_TLS_ENABLED` | `false` | 总开关 |
| `PRIVACY_TLS_CERT_FILE` | — | 服务端证书链（PEM） |
| `PRIVACY_TLS_KEY_FILE` | — | 服务端私钥（PEM，可加密） |
| `PRIVACY_TLS_KEY_PASSWORD` | — | 私钥口令 |
| `PRIVACY_TLS_CA_FILE` | — | 客户端 CA；`optional` / `require` 时必填 |
| `PRIVACY_TLS_CLIENT_AUTH` | `none` | `none` / `optional` / `require` |

启动期校验（`SecuritySettings._check_tls_consistency`）：

- `TLS_ENABLED=true` 但缺证书或私钥 → `ValueError`，拒绝启动；
- `CLIENT_AUTH` 为 `optional`/`require` 但缺 CA → `ValueError`，拒绝启动；
- 不存在「配置不完整时静默降级为明文」的分支。

REST 与 gRPC 都从同一组变量装配：

- REST：`uvicorn_ssl_kwargs()` → `ssl.PROTOCOL_TLS_SERVER` + `ssl_cert_reqs`；
- gRPC：`grpc_server_credentials()` → `grpc.ssl_server_credentials(..., require_client_auth=...)`。

> **重要**：REST 的 `CLIENT_AUTH=require` 只保证「连接持有受信客户端证书」，**不会**把证书 CN/指纹映射成业务身份。REST 身份仍只来自 `Authorization: Bearer` 静态 Key；终结器注入的证书身份只用于 PoP、审计与服务账号激活（与 Go 一致）。gRPC 的证书身份映射另见 [`auth.md`](./auth.md) §2.6。

### 3.1b 独立运维口（`:9079`）

`engine.server` 与 `engine.launcher` 同时启动运维口（默认 `REST_PORT + 1000`）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_OPS_HOST` | 继承 `PRIVACY_REST_HOST` | 运维口监听地址 |
| `PRIVACY_OPS_PORT` | REST 端口 + 1000 | 运维口端口；REST=0 时同样随机 |
| `PRIVACY_OPS_TLS_STACK` | 空（继承业务口） | `tlcp` / `both` / `tls` 或空 |
| `PRIVACY_OPS_TLS_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` | 继承 `PRIVACY_TLS_*` | `tls` 档三项必填 |
| `PRIVACY_OPS_TLS_KEY_PASSWORD` | 继承 `PRIVACY_TLS_KEY_PASSWORD` | 运维口私钥口令 |

- 运维口只放行 `/health` `/healthz` `/livez` `/readyz` `/metrics` `/ops/diagnostics` `/debug/pprof/`，其余 404；
- `PRIVACY_OPS_TLS_STACK=tlcp|both` 只允许绑定环回，由 Go 终结器对外提供国密；监听本身为明文（终结器之后的受信段）；
- `PRIVACY_OPS_TLS_STACK=tls` 由 Python 终结标准 TLS mTLS，缺 cert/key/CA 拒绝启动。

### 3.2 网关（REST `:8000` / gRPC `:50000` / 运维口 `:9000`）

网关与 Go 同名同义使用 `PRIVACY_GATEWAY_*`：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_GATEWAY_TLS_STACK` | `tlcp` | `tlcp` / `both` / `tls`，三档均 mTLS |
| `PRIVACY_GATEWAY_TLS_CERT_FILE` | — | `tls` 档服务端证书 |
| `PRIVACY_GATEWAY_TLS_KEY_FILE` | — | `tls` 档私钥（与 cert 成对校验） |
| `PRIVACY_GATEWAY_TLS_CA_FILE` | — | `tls` 档客户端 CA（mTLS 必填） |
| `PRIVACY_GATEWAY_OPS_TLS_STACK` | `both` | 运维口独立档位 |
| `PRIVACY_GATEWAY_OPS_TLS_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` | 继承业务口 | 运维口材料；`tls` 档三项必填 |

启动期校验：

- `tlcp` / `both` 档要求监听地址为环回（或 `127.0.0.1`）；非环回直接拒绝启动，避免误以为 Python 已终止国密；
- `tls` 档必须提供 cert/key/CA（mTLS），缺一拒绝启动；
- 运维口 `tls` 档同理，且默认继承业务口材料。

回源方向（与 Go 遗留布尔开关同名）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_GATEWAY_BACKEND_TLS_ENABLED` | 空 | 网关 → 后端 TLS/mTLS 开关 |
| `PRIVACY_GATEWAY_BACKEND_TLS_CA` | — | 回源 CA；启用 TLS 但缺失/文件不存在时构造期 `RuntimeError` |
| `PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_CERT` / `_CLIENT_KEY` | — | 回源 mTLS 客户端身份，必须成对 |

网关不向上游传递客户端身份；mTLS 只用于传输认证。

### 3.3 引擎 `PRIVACY_TLS_STACK` 的真实语义

```python
# engine/routers/health.py
return {"status": "ok", "tls_stack": os.environ.get("PRIVACY_TLS_STACK", "")}
```

- 无论设置为 `tlcp`、`both` 还是 `tls`，引擎监听器都不会改变；
- 建议 Python 引擎部署显式设置 `PRIVACY_TLS_STACK=tls` 或留空，并由前置终结器负责对外档位；
- 判断真实协议请查看 Uvicorn/gRPC 启动日志与证书配置，不要只信回显。

验证：

```bash
# 回显值 ≠ 实际协议
curl -s localhost:8079/readyz | jq .tls_stack

# 实际握手只支持 TLS（纯 TLCP 端口会失败）
openssl s_client -connect localhost:8079 -tls1_3 </dev/null 2>/dev/null | grep -E 'Protocol|Cipher'
```

---

## 4. 国密合规场景的部署拓扑（已实现）

```text
client ──TLCP / both / TLS(mTLS)──▶ Go terminator ──loopback──▶ Python REST/gRPC/ops
                                        │                       :18079 / :15051 / :19079
                                        └─ 校验 CN/SAN + 指纹钉扎（可选）
                                           strip 外部同名头 → HMAC 签名后注入
```

- 公网端口：REST `8079`、gRPC `50051`、运维口 `9079`（均可配置）；
- Python 回环端口：REST `18079`、gRPC `15051`、运维口 `19079`；
- 终结器是唯一 TLS 边界，Python 只监听 `127.0.0.1`；
- 终结器复用 Go `pkg/tlsutil` 的 `BuildServerStack`、SM2 双证书、CN 白名单与 `both` 记录层分派；
- gRPC 采用 L7 代理（`UnknownServiceHandler`）而非 L4 透传，保留 streaming / trailer / metadata；
- 终结器先剥离客户端伪造的同名头，再注入自己的签名头；
- Python 只接受来自环回、HMAC 签名正确、时间戳在窗口内且必需头各出现一次的请求，
  任一身份头存在但校验失败即整体拒绝（不会静默退化成“无证书”）。

### 4.1 终结器配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_TERMINATOR_REST_LISTEN` / `_GRPC_LISTEN` / `_OPS_LISTEN` | `:8079` / `:50051` / `:9079` | 对外监听 |
| `PRIVACY_TERMINATOR_REST_UPSTREAM` / `_GRPC_UPSTREAM` / `_OPS_UPSTREAM` | `http://127.0.0.1:18079` / `127.0.0.1:15051` / `http://127.0.0.1:19079` | 回环上游，非环回拒绝启动 |
| `PRIVACY_TLS_STACK` | `tlcp` | 业务口协议栈，三档均 mTLS |
| `PRIVACY_OPS_TLS_STACK` | `both` | 运维口协议栈 |
| `PRIVACY_TLS_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` | — | 标准 TLS 材料 |
| `PRIVACY_TLCP_SIGN_*` / `_ENC_*` / `_CLIENT_CA_FILE` | Go 引擎证书目录 | 国密双证书材料 |
| `PRIVACY_TERMINATOR_IDENTITY_ENABLED` | `true` | 是否注入签名身份头 |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO` | `sm3` | `sm3` / `sha256` |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY[_FILE]` | — | ≥16 字节共享密钥；文件优先 |
| `PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS` | `60` | Python 验签时间窗 |
| `PRIVACY_AUTH_MTLS_WHITELIST_FILE` | — | 终结器先行 CN/SAN + 指纹钉扎 |

Python 侧必须配置同一组 `PRIVACY_TERMINATOR_IDENTITY_*` 变量。完整说明、密钥轮换与安全不变量见
[`terminator/README.md`](../terminator/README.md)。

### 4.2 身份头（签名规范）

| REST 头 | gRPC metadata | 内容 |
|---|---|---|
| `X-PrivShield-Client-CN` | `x-privshield-client-cn` | 证书 Subject CN |
| `X-PrivShield-Client-Cert-SHA256` | `x-privshield-client-cert-sha256` | 叶证书 DER 的 SHA-256 |
| `X-PrivShield-Client-SPKI` | `x-privshield-client-spki` | SPKI DER 的 SHA-256 |
| `X-PrivShield-Client-SPKI-DER` | `x-privshield-client-spki-der` | 原始 SPKI DER（base64） |
| `X-PrivShield-Identity-Timestamp` | `x-privshield-identity-timestamp` | Unix 秒 |
| `X-PrivShield-Identity-Sig` | `x-privshield-identity-sig` | HMAC-SM3 / HMAC-SHA256 |

```text
privshield-identity-v1
<METHOD>
<PATH>
<CN>
<CERT_SHA256>
<SPKI_SHA256>
<TIMESTAMP>
```

原始 DER 不进入签名串，其完整性由 `sha256(DER) == 已签名的 SPKI 摘要` 在 Python 侧校验。
服务账号激活需要 DER 本体，因此不能只注入摘要。

### 4.3 身份如何使用

| 场景 | 行为 |
|---|---|
| REST | PoP 证书绑定、审计、服务账号激活；业务身份仍只来自 Bearer |
| gRPC | `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED=true` 时，验签 CN 经白名单映射为内部身份 |
| `/metrics` | `PRIVACY_METRICS_ALLOW_MTLS=true` 时可用验签 CN 准入 |

Python 侧 mTLS 身份映射只使用 **CN**；SAN URI / SAN DNS 由终结器白名单处理。PoP 绑定比对仅支持
`sha256:<64hex>`（`sm3:` 视为畸形并拒绝）。

### 4.4 部署示例

- Docker Compose：[`deploy/docker-compose.terminator.yml`](../deploy/docker-compose.terminator.yml)（共享网络命名空间）；
- K8s sidecar：[`deploy/k8s/terminator-sidecar.yaml`](../deploy/k8s/terminator-sidecar.yaml)。

### 备选：Go privacy-gateway 前置

```text
调用方 --(TLCP/mTLS)--> Go privacy-gateway(:8000/:50000) --(TLS 或受信内网)--> Python engine
```

由 Go 网关完成国密握手，Python 侧只需标准 TLS 或受信内网明文；网关到 Python 段不是国密，需在合规文档中明确边界。

> 无论哪种拓扑，都必须关闭 Python 的对外明文暴露面：业务端口不得对不可信网络直接可达；如直接暴露，至少启用标准 TLS + mTLS + 静态 Key。

---

## 5. 进程内 TLCP 的迁移清单（不在当前计划内）

若未来要求在 Python 内完成国密握手，至少需要：

1. **TLCP 协议栈**：GM/T 0024 握手、SM2 签名/加密双证书、SM3/SM4 套件。Python 标准库 `ssl` 不支持；需引入国密引擎（Tongsuo/BabaSSL 的 Python 绑定）或 Rust/C 扩展。
2. **配置模型**：`PRIVACY_TLS_STACK=tlcp|both|tls` 真正选择监听器；`tlcp` 缺双证书拒绝启动。
3. **双协议自适应**：`both` 档同端口识别 ClientHello 并分流。
4. **身份提取**：从 SM2 客户端证书提取 CN/公钥指纹，接入现有 `engine/security` 身份链；指纹格式与 Go 的 `spki=sm3:` 对齐。
5. **`/readyz` 如实报告**实际生效档位。
6. **测试**：TLCP 握手、双证书缺失拒启、`both` 分流、mTLS CN 白名单、证书轮换。

在完成前，不要在 Python 文档、Helm values 或能力协商中宣称支持 TLCP。

---

## 6. 常见误配置

| 误配置 | 后果 | 纠正 |
|---|---|---|
| 引擎设置 `PRIVACY_TLS_STACK=tlcp` 但无终结器 | `/readyz` 回显与真实协议不符，客户端国密握手失败 | 前置终结器；引擎显式设 `tls` 或留空 |
| 网关 `PRIVACY_GATEWAY_TLS_STACK=tlcp` 却绑定非环回地址 | 启动即拒绝 | 绑定环回并由终结器对外，或改 `tls` |
| 网关 `tls` 档缺 CA | 启动即拒绝 | 补齐 cert/key/CA（三档均 mTLS） |
| `PRIVACY_TLS_ENABLED=true` 但缺 CA 且 `CLIENT_AUTH=require` | 启动 `ValueError` | 补齐 CA 或改 `none` |
| 认为 REST mTLS 会提供用户身份 | 业务仍要求 Bearer Key，误判 401 | 见 [`auth.md`](./auth.md) §2.6 |
| 使用历史 `GATEWAY_TLS_*` / `ENGINE_GATEWAY_*` | 变量不生效，监听回退默认值 | 统一 `PRIVACY_GATEWAY_*` |
| 回源 `PRIVACY_GATEWAY_BACKEND_TLS_ENABLED=true` 但无 CA | 构造期 `RuntimeError` | 提供可读 CA 文件 |
| 只有在 Ingress 开 TLS，Service 仍明文且网络可达 | 内网可绕过 TLS/认证 | NetworkPolicy + 认证双保险 |
| 终结器已启用但未配置 HMAC 密钥 | 启动失败（fail-closed） | 配置 `PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE` 或显式关闭身份注入 |
| 终结器与 Python 的 HMAC 算法/密钥不一致 | 所有身份头验签失败，PoP 与 mTLS 身份不可用 | 两侧使用同一组 `PRIVACY_TERMINATOR_IDENTITY_*` |
| 终结器上游填非回环地址 | 启动失败 | 改为 `127.0.0.1`/`localhost`；Python 不得直接暴露 |
| Python 在终结器拓扑仍绑定 `0.0.0.0` | 端口可被绕过终结器直连 | 绑定 `127.0.0.1`（compose/K8s 示例已固化） |

---

## 7. 验证命令

```bash
# 1. 确认 Python 进程内不存在 TLCP 实现
grep -rniE 'gm/t|sm2|tongsuo|babassl' engine --include='*.py'

# 2. 网关档位校验
PRIVACY_GATEWAY_TLS_STACK=ssl python -m engine.gateway.server   # 应拒绝启动
PRIVACY_GATEWAY_TLS_STACK=tlcp PRIVACY_GATEWAY_HOST=0.0.0.0 \
  python -m engine.gateway.server                               # 应拒绝启动

# 3. 终结器 Go 测试与静态检查（独立模块）
make terminator-check

# 4. 标准 TLS 握手与 mTLS（终结器 tls/both 档）
openssl s_client -connect localhost:8079 -tls1_3 -CAfile ca.pem \
  -cert client.pem -key client-key.pem </dev/null

# 5. gRPC 通道（标准 TLS）
grpcurl -cacert ca.pem -cert client.pem -key client-key.pem \
  localhost:50051 grpc.health.v1.Health/Check

# 6. 网关入站 mTLS（tls 档）
curl --cacert ca.pem --cert client.pem --key client-key.pem https://localhost:8000/health

# 7. 运维口（终结器 both 档为 TLS；Python 回环运维口为明文）
curl -sk https://localhost:9079/readyz | jq .tls_stack
curl -s http://127.0.0.1:19079/readyz | jq .tls_stack

# 8. 国密握手需要支持 TLCP 的客户端（gmssl / Tongsuo 等）；
#    openssl s_client 不支持 GM/T 0024，不能用来验证 tlcp 档。
```

---

## 8. 相关文档

- [TLCP 终结器实现与运维](../terminator/README.md)
- [架构总览](./architecture.md)
- [配置参考](./config.md)
- [认证与授权](./auth.md)
- [策略清单](./policy_bundles.md)
- [运维手册](./ops.md)
- Go 国密基线：上游 `services/privacy-engine/docs/architecture.md` §6.1
