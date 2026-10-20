# Python 网关与负载均衡运维指南 (Ops)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 核心环境变量

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_GATEWAY_STRATEGY` | `p2c` | 负载均衡算法：`p2c` / `round_robin` / `least_connections` |
| `PRIVACY_GATEWAY_BACKENDS` | `127.0.0.1:8079` | 后端 REST 节点列表（逗号分隔） |
| `PRIVACY_GATEWAY_GRPC_BACKENDS` | `127.0.0.1:50051` | 后端 gRPC 节点列表 |
| `PRIVACY_GATEWAY_MAX_BODY_BYTES` | `33554432` | 请求体大小限制（默认 32 MiB） |
| `PRIVACY_GATEWAY_HEALTH_CHECK_INTERVAL`| `5.0` | 主动探针周期（秒） |

---

## 2. 关键指标

| 指标 | 说明 |
|---|---|
| `privshield_gateway_requests_total` | 网关接收与转发请求总计数（带 status_code 标签） |
| `privshield_gateway_duration_seconds` | 代理全链路耗时分布 |
| `privshield_gateway_active_backends` | 当前处于可用状态的后端节点数量 |
