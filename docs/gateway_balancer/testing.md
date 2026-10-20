# Python 网关与负载均衡测试指南 (Testing)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 测试套件构成

- `tests/gateway/test_gateway.py`：基础反向代理选路、HTTP/gRPC 转发、熔断器代际状态机流转。
- `tests/gateway/test_http_proxy_edge.py`：边界与负向测试（超大请求体 413、hop-by-hop 头剥离、全部节点熔断 503）。
- `tests/gateway/test_server_unit.py`：运维口 `/readyz`、`/metrics`、`/gateway/backends` 响应结构与优雅停机。

---

## 2. 自动化执行

```bash
.venv/bin/python -m pytest tests/gateway/ -v
```
