# 网关与负载均衡（Gateway / Balancer）

> 代码：`engine/gateway/` · 进程入口：`python -m engine.gateway.server` · 测试：`tests/gateway/`
> 对齐事实源：Go `services/privacy-engine/cmd/privshield-gateway` 与 `internal/gateway`

## 定位

Python 引擎自带的 REST + gRPC 透明反向代理与负载均衡器，默认监听 REST `127.0.0.1:8000`、gRPC `127.0.0.1:50000`、运维口 `127.0.0.1:9000`，向后端 `127.0.0.1:8079` / `127.0.0.1:50051` 回源。网关只做选路、熔断、准入与转发，不解析业务请求体。

## 代码地图

| 文件/符号 | 职责 |
|---|---|
| `balancer.LoadBalancer` | 节点池、P2C-EWMA / 轮询 / 最少连接等策略、动态注册/注销/隔离/排空、拓扑快照 |
| `balancer.BackendNode` | 节点坐标、健康标志、在途连接数、独立熔断器、EWMA、懒加载 gRPC 通道 |
| `balancer.CircuitBreaker` / `Permit` | Closed → Open → Half-Open；代际绑定的放行许可，过期结果自动丢弃 |
| `balancer.health_check_loop` | Python 超集：周期主动探针 HTTP `/health` 与 gRPC Health，并同步熔断器 |
| `http_proxy.create_http_gateway_app` | REST 单次转发、准入中间件、节点管理端点 |
| `grpc_proxy.GatewayGrpcProxy` | 原生 gRPC 透明代理（任意方法/任意基数），metadata 白名单与 trailer 透传 |
| `server.async_main` / `create_ops_app` | 三端口启动、运维口 `/health` `/readyz` `/metrics` `/gateway/backends`、优雅停机 |

## 调度策略

`PRIVACY_GATEWAY_STRATEGY` 默认 `p2c`，与 Go 一致：

| 策略 | 行为 |
|---|---|
| `p2c`（默认，别名 `power_of_two_choices`） | 随机取两个节点，按 `(inflight+1) * max(EWMA, 0.001)` 取较小者；REST EWMA α=0.3，gRPC α=0.2 |
| `round_robin` | 健康节点轮询 |
| `weighted_round_robin`（别名 `weighted_rr`） | Nginx 平滑加权轮询 |
| `weighted_random` / `random` | 按静态权重随机抽样 |
| `least_connections`（别名 `least_conn`） | 选择在途请求最少的节点 |

熔断器 open 的节点不参与候选；若全部候选熔断，退回 `nodes[0]`，由代理的 `acquire()` 返回 `CIRCUIT_OPEN` / `UNAVAILABLE`，保证失败归因正确。空池返回 `None`（REST 503 / gRPC `UNAVAILABLE`）。

## 健康与故障处理

- **单次尝试（Go parity）**：选中节点后只转发一次，**绝不重试**。传输失败 REST 返回 `502 BAD_GATEWAY`，gRPC 返回 `UNAVAILABLE`；错误文案不含后端异常原文或内网地址。
- **响应记账**：后端状态码 `<500` 记 success，`>=500` 记 failure 并原样透传；4xx 不惩罚节点。
- **熔断**：每节点独立三态；默认连续 5 次 failure 打开，冷却 30s 后由下一次 `acquire()` 转入 half-open，最多 3 个在途探测。Closed 状态下任一 success 将失败计数清零；HalfOpen 累计 3 次 success 关闭，任一 failure 重新打开。旧代际的迟到结果被丢弃。
- **主动探针（Python 超集）**：`PRIVACY_GATEWAY_HEALTH_CHECK_INTERVAL`（默认 5s）周期检查 HTTP `/health`（要求 200 且 `status == "ok"`）与 gRPC Health；失联节点移出候选，恢复后重置熔断器。Go 网关没有主动探针，此为超集能力。
- **本机 `/health`**：网关业务口自行应答 `{"status":"ok","component":"gateway"}`，不转发到后端。
- **请求体上限**：`PRIVACY_GATEWAY_MAX_BODY_BYTES`，默认 32 MiB；`Content-Length` 或实际读取超限均返回 413。
- **逐段头治理**：请求/响应方向剔除 RFC 7230 hop-by-hop 头；响应额外剥离 `content-encoding`（httpx 已自动解压）。重复响应头（如多条 `Set-Cookie`）按原样保留。
- **回源连接池**：`httpx.AsyncClient` 超时 30s，keepalive 100 / 最大连接 500，`trust_env=False`；跨事件循环自动重建。

## 准入与凭据

处理顺序：CIDR 白名单 → HTTPS 强制 → 逐请求凭据准入 → keyfile 策略 → `select_node` → `acquire()`。

| 变量 | 行为 |
|---|---|
| `PRIVACY_GATEWAY_ALLOWED_CIDRS` | 入站 IP 白名单，支持逗号分隔或 JSON 数组 |
| `PRIVACY_GATEWAY_TRUSTED_PROXIES` | 仅信任这些代理的 `X-Forwarded-For` / `X-Forwarded-Proto` |
| `PRIVACY_GATEWAY_REQUIRE_TLS` / `_REQUIRE_FORWARDED_PROTO` | 非 `/health` 且非环回调用方强制 HTTPS，否则 426 |
| `PRIVACY_GATEWAY_AUTH_ENABLED` | 除 `/health` 外必须携带非空 `Authorization`，否则 401（合法性由后端裁决） |
| `PRIVACY_GATEWAY_INBOUND_KEYS_FILE` | 可选 keyfile scope/路径策略，校验失败返回对应 401/403 |
| `PRIVACY_GATEWAY_BACKEND_API_KEY` | 仅当入站匿名且准入关闭时注入 `Bearer <key>`；入站已有凭据（包括畸形凭据）原样透传 |

REST 会为上游追加 `X-Forwarded-For` 与 `X-Real-IP`。gRPC 出站 metadata 只透传链路追踪/幂等键白名单（`x-request-id`、`traceparent`、`baggage`、`b3` 系列等），`authorization` 单独处理，`x-internal-*` 一律剔除。

gRPC 代理使用 `grpc.GenericRpcHandler` + 恒等序列化器，因此 unary / server-stream / client-stream / bidi 全部按 stream-stream 转发，保留 initial metadata、trailing metadata 与半关闭语义。

## 节点管理 API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/gateway/register` | 注册/更新节点；`http_url` 仅允许 http/https（SSRF 防护） |
| POST | `/v1/gateway/deregister` | 注销节点（不存在也返回 200，幂等） |
| POST | `/v1/gateway/isolate` | 强制隔离，不参与调度 |
| POST | `/v1/gateway/drain` | 排空：不再接新请求，在途请求可完成 |
| POST | `/v1/gateway/activate` | 恢复为 active |

管理端点由 `PRIVACY_GATEWAY_ADMIN_API_KEY`（缺省继承 `PRIVACY_GATEWAY_METRICS_API_KEY`）保护：未配置时一律 503（fail-closed），`Authorization: Bearer <key>` 不匹配返回 401（`hmac.compare_digest`）。该鉴权独立于 `engine/security` 的 API Key 体系。

## 运维口

默认绑定业务 Host 的 `业务口 + 1000`（`:9000`），可用 `PRIVACY_GATEWAY_OPS_PORT` 覆盖：

| 路径 | 鉴权 | 内容 |
|---|---|---|
| `/health` | 无 | 进程存活 |
| `/readyz` | 无 | `status`、`strategy`、`tls_stack`、`metrics_auth_mode`、REST 后端健康计数；无健康后端返回 503 |
| `/gateway/backends` | `ADMIN_API_KEY`（配置时） | REST 池拓扑：`address` / `in_flight` / `ewma_ms` / `cb_state` |
| `/metrics` | `METRICS_API_KEY`（配置时） | Prometheus 文本格式 |

环回绑定允许匿名读取；非环回绑定必须配置 metrics/admin Key，否则拒绝启动。`PRIVACY_GATEWAY_OPS_TLS_STACK` 独立于业务口，`tls` 档要求 cert/key/CA 齐全（mTLS）。

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_GATEWAY_HOST` / `_PORT` | `127.0.0.1` / `8000` | REST 监听 |
| `PRIVACY_GATEWAY_GRPC_HOST` / `_GRPC_PORT` | `127.0.0.1` / `50000` | gRPC 监听 |
| `PRIVACY_GATEWAY_OPS_PORT` | 业务口 + 1000 | 运维口 |
| `PRIVACY_GATEWAY_BACKENDS` | `127.0.0.1:8079` | REST 后端，逗号分隔 |
| `PRIVACY_GATEWAY_GRPC_BACKENDS` | `127.0.0.1:50051` | gRPC 后端，逗号分隔 |
| `PRIVACY_GATEWAY_STRATEGY` | `p2c` | 调度策略 |
| `PRIVACY_GATEWAY_HEALTH_CHECK_INTERVAL` | `5.0` | 主动探针周期（秒，0 关闭） |
| `PRIVACY_GATEWAY_MAX_BODY_BYTES` | `33554432` | 请求体上限 |
| `PRIVACY_GATEWAY_TLS_STACK` | `tlcp` | `tlcp` / `both` / `tls`；三档均 mTLS，无明文档 |
| `PRIVACY_GATEWAY_TLS_CERT_FILE` / `_KEY_FILE` / `_CA_FILE` | 空 | `tls` 档必填（mTLS） |
| `PRIVACY_GATEWAY_OPS_TLS_STACK` / `_OPS_TLS_*_FILE` | `both` | 运维口独立 TLS 档位与材料 |
| `PRIVACY_GATEWAY_METRICS_API_KEY` | 空 | `/metrics` Bearer Key |
| `PRIVACY_GATEWAY_ADMIN_API_KEY` | 继承 metrics key | 管理端点与 `/gateway/backends` Bearer Key |
| `PRIVACY_GATEWAY_BACKEND_TLS_ENABLED` | 空 | 回源 TLS 开关；启用时 `_TLS_CA` 必填，否则构造期报错 |
| `PRIVACY_GATEWAY_BACKEND_TLS_CA` | 空 | 后端证书校验 CA |
| `PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_CERT` / `_CLIENT_KEY` | 空 | 回源 mTLS 客户端身份，必须成对配置 |
| `PRIVACY_GATEWAY_LIMIT_CONCURRENCY` | `10000` | Uvicorn 最大并发连接 |
| `PRIVACY_GATEWAY_LIMIT_MAX_REQUESTS` | `100000` | 单 worker 最大请求数 |
| `PRIVACY_GATEWAY_TIMEOUT_KEEP_ALIVE` | `120` | keep-alive 超时（秒） |
| `PRIVACY_GATEWAY_TIMEOUT_GRACEFUL_SHUTDOWN` | `10` | 优雅停机等待（秒） |

> `TLCP` 由 Phase 8 的 Go 终结器侧车对外提供；Python 网关在 `tlcp` / `both` 档只监听环回明文，非环回绑定会拒绝启动。

## 边界

- **无重试**：与 Go 一致，单次转发；幂等性由调用方负责。
- **不终止 TLCP**：Python 无原生国密栈，由 Go 终结器侧车负责；标准 TLS + mTLS 可用 `tls` 档自终止。
- **不转发认证身份**：mTLS 只完成 TLS 校验，不解析 CN 或注入用户身份头；仅追加 `X-Forwarded-For` / `X-Real-IP`。
- **管理 API 无审计服务接入**：仅记录结构化日志。
- **多进程指标不聚合**：多个网关进程各自暴露指标。
- **动态节点状态只在进程内**：重启回到配置值。

相关文档：[architecture.md](../architecture.md) · [tlcp.md](../tlcp.md) · [prometheus.md](../prometheus.md) · [api.md](../api.md)
