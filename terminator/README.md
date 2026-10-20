# Python 引擎 TLCP / mTLS 终结器

`terminator/` 是一个独立的 Go 模块，为 Python 引擎提供**唯一的 TLS 边界**：

```text
客户端 ──TLCP / both / TLS（三档均 mTLS）──▶ terminator ──回环明文──▶ Python 引擎
                                              │                       REST  :18079
                                              │                       gRPC  :15051
                                              │                       ops   :19079
                                              ├─ 校验客户端证书（可选 CN/SAN 白名单 + 指纹钉扎）
                                              ├─ 剥离客户端伪造的身份头
                                              └─ 注入 HMAC 签名的证书身份头
```

Python 进程内**不实现国密 TLCP**，也不应绑定非回环地址。所有对外端口由终结器持有：
REST `:8079`、gRPC `:50051`、运维口 `:9079`。

---

## 1. 构建与测试

模块自带 `go.mod`，依赖的 Go `pkg/` 子集已 vendor 在 `third_party/pkg`
（`replace` 指向该目录），不在任何 `go.work` 中。因此所有 Go 命令都必须显式 `GOWORK=off`：

```bash
make terminator-lint       # go vet
make terminator-test       # go test ./...
make terminator-check      # 两者
make terminator-build      # bin/privshield-python-terminator
make docker-terminator     # privshield-python-terminator:<VERSION>（上下文=本服务目录）
```

直接运行：

```bash
PRIVACY_TLS_STACK=tls \
PRIVACY_TLS_CERT_FILE=config/certs/server.crt \
PRIVACY_TLS_KEY_FILE=config/certs/server.key \
PRIVACY_TLS_CA_FILE=config/certs/ca.crt \
PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY='replace-with-32-byte-random-key' \
bin/privshield-python-terminator
```

开发证书生成见 `config/certs/README.md`（标准 TLS）与
`go run ./scripts/dev/tlcp-certgen -dir config/certs/tlcp`（国密双证书）。

---

## 2. 配置

### 2.1 监听与上游

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_TERMINATOR_SERVICE_NAME` | `privshield-python-terminator` | 日志服务名 |
| `PRIVACY_TERMINATOR_REST_LISTEN` | `:8079` | 对外 REST 监听 |
| `PRIVACY_TERMINATOR_GRPC_LISTEN` | `:50051` | 对外 gRPC 监听 |
| `PRIVACY_TERMINATOR_OPS_LISTEN` | `:9079` | 对外运维口监听 |
| `PRIVACY_TERMINATOR_REST_UPSTREAM` | `http://127.0.0.1:18079` | Python REST 回环上游 |
| `PRIVACY_TERMINATOR_GRPC_UPSTREAM` | `127.0.0.1:15051` | Python gRPC 回环上游 |
| `PRIVACY_TERMINATOR_OPS_UPSTREAM` | `http://127.0.0.1:19079` | Python 运维口回环上游 |
| `PRIVACY_SHUTDOWN_DRAIN_SECONDS` | `5` | 收到信号后先置 unready 并排空 |
| `PRIVACY_GRPC_GRACEFUL_STOP_SECONDS` | `15` | gRPC GracefulStop 上限 |

三个监听地址必须互不相同。三个上游**必须是回环地址**（`127.0.0.0/8`、`::1`、`localhost`），
否则启动即拒绝——这是防止绕过终结器直连 Python 的硬不变量。

### 2.2 协议栈与 TLS 材料

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_TLS_STACK` | `tlcp` | 业务口协议栈：`tlcp` / `both` / `tls`，三档均 mTLS |
| `PRIVACY_TLS_CERT_FILE` / `_KEY_FILE` / `_KEY_PASSWORD` | — | `tls` / `both` 的服务端证书与私钥 |
| `PRIVACY_TLS_CA_FILE` | — | 客户端 CA |
| `PRIVACY_TLS_CLIENT_AUTH` | `requireandverify` | 业务口客户端认证档位 |
| `PRIVACY_TLCP_SIGN_CERT_FILE` / `_KEY_FILE` | `config/certs/tlcp/server-sign.*` | SM2 签名证书对 |
| `PRIVACY_TLCP_ENC_CERT_FILE` / `_KEY_FILE` | `config/certs/tlcp/server-enc.*` | SM2 加密证书对 |
| `PRIVACY_TLCP_CLIENT_CA_FILE` | `config/certs/tlcp/ca.crt` | TLCP 客户端 CA |
| `PRIVACY_OPS_TLS_STACK` | `both` | 运维口独立档位 |
| `PRIVACY_OPS_TLS_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` | 继承业务口，再回退 `config/certs/server.*` / `ca.crt` | 运维口标准 TLS 材料 |

- `tlcp` 缺任一 SM2 材料、`tls`/`both` 缺标准 TLS 材料 → 启动失败，不存在明文降级。
- 运维口标准 TLS 固定 `verify-if-given`：kubelet 探针可以不出示证书，但数据路径仍由
  Python 的 `/metrics` 鉴权与路径白名单裁决。
- TLS 材料沿用 Go 引擎的 `PRIVACY_*` 变量名，不引入第二套命名。

### 2.3 身份注入

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_TERMINATOR_IDENTITY_ENABLED` | `true` | 关闭后不再注入身份头，Python 只能用 API Key |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO` | `sm3` | `sm3` 或 `sha256` |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY` | — | 共享 HMAC 密钥，至少 16 字节 |
| `PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE` | — | 密钥文件；**优先于**环境变量，首尾空白会被裁剪 |
| `PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS` | `60` | Python 侧拒绝超出此窗口的签名 |

启用身份注入却没有密钥 → 启动失败。密钥文件方式适合 Docker/K8s secret 挂载；
文件内容必须是原始密钥文本（≥16 字节），不要加引号或换行以外的装饰。

Python 侧必须使用**同一组** `PRIVACY_TERMINATOR_IDENTITY_*` 变量（`ENABLED/HMAC_ALGO/HMAC_KEY[_FILE]/MAX_SKEW`）
才能验签。两侧算法不一致会被判为签名失败。

### 2.4 证书白名单

`PRIVACY_AUTH_MTLS_WHITELIST_FILE` 设置后，终结器在转发前按 Go `DynamicWhitelist` 语义校验对端证书：
SAN URI → SAN DNS → Subject CN，命中条目可配置 `cert_fingerprints` 钉扎与 `allowed_scopes`（gRPC 完整方法名）。
未设置时终结器只做 TLS 握手，授权完全交给 Python 的 `PRIVACY_AUTH_MTLS_WHITELIST_FILE`。
两侧可指向同一份 YAML；终结器负责“早拒绝”，Python 负责“最终裁决”。

```yaml
version: "1.0"
clients:
  - name: service-hub
    allowed_scopes: ["/privacy.local.PrivacyService/*", "/privacy.local.Health/Check"]
    cert_fingerprints:
      - "aa:bb:cc:..."   # 叶证书 SHA-256，冒号/大小写不敏感
  - name: audit-log
    allowed_scopes: ["/privacy.local.AuditService/*"]
entries:
  - cn: legacy-client
    scopes: ["privacy:mask"]
```

---

## 3. 身份契约（跨语言）

终结器为每个通过 mTLS 的请求生成以下头；gRPC 使用同名小写 metadata：

| REST 头 | gRPC metadata | 内容 |
|---|---|---|
| `X-PrivShield-Client-CN` | `x-privshield-client-cn` | 证书 Subject CN |
| `X-PrivShield-Client-Cert-SHA256` | `x-privshield-client-cert-sha256` | 叶证书 DER 的 SHA-256（小写 hex） |
| `X-PrivShield-Client-SPKI` | `x-privshield-client-spki` | SPKI DER 的 SHA-256（小写 hex） |
| `X-PrivShield-Client-SPKI-DER` | `x-privshield-client-spki-der` | 原始 SPKI DER 的 base64 |
| `X-PrivShield-Identity-Timestamp` | `x-privshield-identity-timestamp` | Unix 秒 |
| `X-PrivShield-Identity-Sig` | `x-privshield-identity-sig` | HMAC-SM3 或 HMAC-SHA256（小写 hex） |

签名规范串固定为：

```text
privshield-identity-v1
<METHOD>
<PATH>
<CN>
<CERT_SHA256>
<SPKI_SHA256>
<TIMESTAMP>
```

- REST 的 `METHOD/PATH` 取请求方法与原始路径；gRPC 固定 `POST` + 完整方法名（如
  `/privacy.local.PrivacyService/Mask`）。
- 原始 SPKI DER 不进入规范串；其完整性由接收端校验 `sha256(DER) == 已签名的 SPKI 摘要`。
- 终结器先删除客户端自带的全部同名头，再注入自己的签名头，杜绝“外部写入身份”。
- Python 只接受来自回环、签名正确、时间戳在窗口内且必需头各出现一次的请求；
  只要出现任一身份头但校验失败就整体拒绝，不会静默退化为“无证书”。

### Python 侧如何使用

| 场景 | 行为 |
|---|---|
| REST | 身份头用于 PoP 证书绑定、审计与服务账号激活；**业务身份仍只来自 Bearer**（与 Go 一致） |
| gRPC | `PRIVACY_AUTH_INTERNAL_MTLS_ENABLED=true` 时，已验签 CN 经白名单映射为内部身份；关闭时仍参与 PoP |
| 运维口 `/metrics` | `PRIVACY_METRICS_ALLOW_MTLS=true` 时可用已验签的 CN 准入 mTLS 抓取 |
| 审计/存证 | 终结点身份可作为请求归属证据，但仍需通过 API 层授权 |

> Python 侧 mTLS 身份映射使用 **CN**；SAN URI / SAN DNS 由终结器的白名单先行处理。
> PoP 的证书绑定比对支持 `sha256:<64hex>`；`sm3:` 绑定格式在 Python 侧视为畸形（fail-closed）。

---

## 4. 运维口

| 路径 | 来源 | 说明 |
|---|---|---|
| `/health` `/healthz` `/livez` | 终结器本地 | `{"status":"ok"}`；排空期间 503 |
| `/readyz` | 终结器聚合 | 检查 Python ops + REST 的 `/readyz` 与 gRPC TCP 连通；全部就绪才转发上游结果 |
| `/metrics` | Python 运维口 | 原样代理，鉴权与业务口一致 |
| `/ops/diagnostics` | Python 运维口 | 原样代理，需 `ops:diagnostics` |
| `/debug/pprof/` | Python 运维口 | 仅当 Python 暴露该路径时可用；Python 不伪造 Go profile 格式 |

探针必须按实际协议栈访问：`PRIVACY_OPS_TLS_STACK=both|tlcp` 时用支持对应协议的客户端
并校验证书；不要用明文 HTTP 探测。

---

## 5. 密钥与证书轮换

1. **HMAC 密钥**：终结器只在启动时读取；Python 按文件 stat 缓存。轮换时先更新共享
   secret，再滚动重启终结器与 Python（同一发布窗口内），否则两侧签名不一致会导致
   身份头整体被拒。密钥至少 16 字节，建议 32 字节随机值。
2. **服务端证书**：终结器在启动时 `BuildServerStack`，轮换后需重启容器。
3. **客户端 CA / 白名单**：白名单 `clients:` 支持 mtime 热重载；CA 变更需重启。
4. **指纹钉扎**：`cert_fingerprints` 支持冒号、大小写混写，加载时归一化为 64 位小写 hex。

---

## 6. 安全不变量

- Python 不得直接监听非回环地址；终结器上游只接受回环地址，违反即拒绝启动。
- 三档协议栈都是 mTLS，没有明文档位；配置缺失一律 fail-closed。
- 外部请求的同名身份头在进入回环链路前被无条件剥离。
- 身份注入关闭时启动日志明确告警，此时 Python 仅接受 API Key。
- 终结器拒绝非白名单 CN/SAN 与指纹不匹配的证书；白名单为空时不做额外放行。
