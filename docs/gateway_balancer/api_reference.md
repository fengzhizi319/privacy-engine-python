# Python 网关与负载均衡 API 参考手册 (API Reference)

> **组件归属**：`services/privacy-engine-python`
> **服务监听**：REST `:8000` · gRPC `:50000` · 运维口 `:9000`

---

## 1. 运维与管理端点 (Ops Port :9000)

| 方法 | 路径 | 权限要求 | 描述 |
|---|---|---|---|
| `GET` | `/health` | 免密 | 网关本机探活，返回 `{"status": "ok", "component": "gateway"}` |
| `GET` | `/readyz` | 免密 | 就绪检查，检查是否有可用健康后端节点 |
| `GET` | `/metrics` | 免密/BasicAuth | Prometheus 监控指标导出 |
| `GET` | `/gateway/backends` | 管理员 | 查询当前所有后端节点坐标、在途连接、EWMA 与熔断状态 |
| `POST` | `/gateway/backends/register` | 管理员 | 运行时动态注册新后端节点 |
| `POST` | `/gateway/backends/drain` | 管理员 | 优雅排空指定节点流量（准备下线维护） |

---

## 2. 节点状态查询响应示例 (`GET /gateway/backends`)
```json
{
  "strategy": "p2c",
  "active_nodes_count": 2,
  "nodes": [
    {
      "id": "node-1",
      "address": "127.0.0.1:8079",
      "grpc_address": "127.0.0.1:50051",
      "weight": 100,
      "state": "CLOSED",
      "healthy": true,
      "inflight_requests": 3,
      "ewma_latency_ms": 4.2
    },
    {
      "id": "node-2",
      "address": "127.0.0.1:8080",
      "grpc_address": "127.0.0.1:50052",
      "weight": 100,
      "state": "CLOSED",
      "healthy": true,
      "inflight_requests": 1,
      "ewma_latency_ms": 3.8
    }
  ]
}
```
