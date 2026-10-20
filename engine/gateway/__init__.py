"""代理转发与负载均衡网关模块 (PrivShield Gateway & Load Balancer).

本模块作为 PrivShield 隐私计算平台的统一流量调度与安全接入层，提供以下核心能力：
1. **双协议统一代理**：在同进程内统一承接 HTTP/REST (FastAPI/Uvicorn) 与 gRPC 异步通信；
2. **多策略负载均衡**：支持轮询、平滑加权轮询（SWRR）、加权随机及最小连接数（Least Connections）调度；
3. **高可用容灾与自愈**：双协议周期主动探针、毫秒级被动故障下线、三态节点熔断器（Circuit Breaker）与自适应幂等重试；
4. **双向安全防护**：南北向入站 TLS 终结（含客户端 mTLS 验签）与东西向后端 TLS 安全回源，拓扑管理 API 防 SSRF；
5. **云原生双层协同**：完美适配 Kubernetes Ingress + Gateway + Headless Service 架构，攻克 gRPC 长连接倾斜痛点。

模块导出子文件说明：
- ``balancer.py``: 后端工作节点模型 (BackendNode)、熔断器 (CircuitBreaker)、负载均衡调度器 (LoadBalancer) 与主动探针循环；
- ``http_proxy.py``: 基于 FastAPI 的通配 HTTP 反向代理引擎、单例连接池管理与动态拓扑管理路由；
- ``grpc_proxy.py``: 基于 grpc.aio 的泛化 gRPC 反向代理引擎、元数据双向透传与服务启动器；
- ``server.py``: 网关统一启动入口，整合配置加载、双协议服务并发托管与优雅停机排空。
"""

from __future__ import annotations
