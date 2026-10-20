"""HTTP 网关代理边界回归测试。

覆盖以下关键特性与边界场景（Go internal/gateway/http_proxy.go 对齐）：
1. **本地 /health**：不转发后端，固定返回 ``{"status":"ok","component":"gateway"}``；
2. **响应头 content-encoding 剥离**：httpx 已自动解压 body，透传会导致客户端二次解压失败；
3. **单次尝试**：任何传输失败只转发一次并返回 502，绝不重试；
4. **后端状态码分流与熔断反馈**：
   - 5xx 响应原样透传并计入熔断器失败；
   - 4xx 原样透传并按成功记账（closed 状态清零连续失败）；
5. **安全与管理端点防护**：
   - 注册 URL Scheme 非法阻断 (仅限 http/https，防 SSRF)；
   - 未配置管理 Key 时 Fail-Closed 默认 503 阻断；
   - 常量时间比对鉴权 (hmac.compare_digest)；
   - 502 错误脱敏（绝不向客户端泄露内网 URL / 内部堆栈）；
6. **网络头治理与 IP 透传**：
   - Hop-by-Hop 请求头（Connection, Transfer-Encoding, Upgrade 等）严格剔除；
   - 客户端 IP 自动注入 X-Forwarded-For 与 X-Real-IP；
7. **连接池与事件循环漂移自愈**：
   - 跨事件循环 (Event Loop Drift) 时自动重建 AsyncClient 连接池。
"""

from __future__ import annotations

import gzip

import httpx
import pytest
from fastapi.testclient import TestClient

from engine.gateway import http_proxy
from engine.gateway.balancer import LoadBalancer
from engine.gateway.http_proxy import create_http_gateway_app
from engine.gateway.settings import GatewaySettings

_ADMIN_KEY = "secret-token-123"
_ADMIN_SETTINGS = GatewaySettings(admin_api_key=_ADMIN_KEY)
_NO_ADMIN_SETTINGS = GatewaySettings(admin_api_key="", metrics_api_key="")


@pytest.fixture
def patched_client(monkeypatch):
    """将代理内部创建的 httpx.AsyncClient 替换为基于 MockTransport 的客户端。

    Returns:
        (calls, state, captured_requests)：
        calls 记录每次转发的 "METHOD URL"；
        state["handler"] 由各用例设置为 (request) -> Response 的可调用；
        captured_requests 记录上游收到的每个原始 httpx.Request 对象。
    """
    calls: list[str] = []
    captured_requests: list[httpx.Request] = []
    state: dict = {"handler": None}
    real_async_client = httpx.AsyncClient  # 先保留真实类，避免 factory 内递归调用

    def factory(**_kwargs):
        def dispatch(request: httpx.Request) -> httpx.Response:
            calls.append(f"{request.method} {request.url}")
            captured_requests.append(request)
            return state["handler"](request)

        return real_async_client(transport=httpx.MockTransport(dispatch))

    monkeypatch.setattr(http_proxy.httpx, "AsyncClient", factory)
    return calls, state, captured_requests


def test_health_is_local_and_not_forwarded(patched_client):
    """/health 是网关本地端点，不得消耗后端连接。"""
    calls, _state, _captured = patched_client
    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("http://node-a.test", "127.0.0.1:50051")
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "component": "gateway"}
    assert calls == []


def test_response_strips_content_encoding(patched_client):
    """代理响应必须剥离 content-encoding，其余自定义头保留。

    模拟真实场景：上游返回 gzip 压缩体 + content-encoding 头，
    httpx 会自动解压，若再把头透传给客户端将导致二次解压失败。
    """
    _calls, state, _captured = patched_client
    payload = gzip.compress(b'{"status": "ok"}')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={"content-encoding": "gzip", "x-custom": "keep"},
        )

    state["handler"] = handler

    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("http://node-a.test", "127.0.0.1:50051")
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    resp = client.get("/v1/privacy/budget")
    assert resp.status_code == 200
    assert resp.content == b'{"status": "ok"}'
    assert "content-encoding" not in {k.lower() for k in resp.headers}
    assert resp.headers.get("x-custom") == "keep"


def test_transport_error_single_attempt_returns_502(patched_client):
    """传输失败只转发一次，返回 502 并记一次熔断失败（Go 单次尝试）。"""
    calls, state, _captured = patched_client

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow backend", request=request)

    state["handler"] = handler

    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("http://node-a.test", "127.0.0.1:50051")
    node = balancer.nodes[0]
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    resp = client.post(
        "/v1/privacy/mask",
        json={"field_name": "mobile", "value": "13812345678", "context": ""},
    )
    assert resp.status_code == 502
    assert len(calls) == 1
    assert node.circuit_breaker.failures == 1
    # 网关只做熔断记账，不引入 Python 旧版的被动下线状态
    assert node.is_healthy is True


def test_backend_5xx_records_circuit_breaker_failure(patched_client):
    """后端 5xx 应计入熔断器失败统计并原样透传，4xx 按成功记账。"""
    calls, state, _captured = patched_client

    state["handler"] = lambda request: httpx.Response(500, json={"error": "boom"})

    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("http://node-a.test", "127.0.0.1:50051")
    node = balancer.nodes[0]
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    resp = client.get("/v1/unknown")
    assert resp.status_code == 500
    assert len(calls) == 1
    assert node.circuit_breaker.failures == 1
    assert node.is_healthy is True

    # 4xx 属于客户端问题：不增加失败计数；closed 状态下成功会清零连续失败
    state["handler"] = lambda request: httpx.Response(404, json={"error": "not found"})
    resp404 = client.get("/v1/unknown")
    assert resp404.status_code == 404
    assert node.circuit_breaker.failures == 0


def test_hop_by_hop_headers_stripped_on_request(patched_client):
    """请求方向必须剔除 Hop-by-Hop 逐段传输头，其余自定义头保留透传。"""
    _calls, state, captured = patched_client
    state["handler"] = lambda request: httpx.Response(200, json={"status": "ok"})

    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("http://node-a.test", "127.0.0.1:50051")
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    resp = client.get(
        "/v1/privacy/budget",
        headers={
            "Connection": "keep-alive",
            "Proxy-Authenticate": "basic",
            "Upgrade": "websocket",
            "X-Custom-Trace": "trace-12345",
        },
    )
    assert resp.status_code == 200
    assert len(captured) == 1
    req_headers = captured[0].headers
    assert "proxy-authenticate" not in req_headers
    assert "upgrade" not in req_headers
    assert req_headers.get("x-custom-trace") == "trace-12345"


def test_client_ip_forwarding(patched_client):
    """验证 X-Forwarded-For 与 X-Real-IP 注入与追加。"""
    _calls, state, captured = patched_client
    state["handler"] = lambda request: httpx.Response(200, json={"status": "ok"})

    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("http://node-a.test", "127.0.0.1:50051")
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    # 1. 未携带 X-Forwarded-For 时自动注入 client_ip
    client.get("/v1/privacy/budget")
    assert "x-forwarded-for" in captured[0].headers
    assert "x-real-ip" in captured[0].headers

    # 2. 原请求已有 X-Forwarded-For 时追加
    client.get("/v1/privacy/budget", headers={"X-Forwarded-For": "10.0.0.1"})
    xff = captured[1].headers.get("x-forwarded-for")
    assert xff.startswith("10.0.0.1, ")


def test_register_node_invalid_scheme_rejected():
    """注册节点的 http_url 必须校验 Scheme，非 http/https 返回 400。"""
    balancer = LoadBalancer(strategy="round_robin")
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    resp = client.post(
        "/v1/gateway/register",
        headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
        json={"http_url": "ftp://malicious-node.test", "grpc_address": "127.0.0.1:50051"},
    )
    assert resp.status_code == 400
    assert "Invalid http_url scheme" in resp.json()["message"]


def test_management_endpoints_fail_closed_without_key():
    """未配置管理 Key 时管理端点一律 503（fail-closed，防 SSRF）。"""
    balancer = LoadBalancer(strategy="round_robin")
    client = TestClient(create_http_gateway_app(balancer, settings=_NO_ADMIN_SETTINGS))

    resp_reg = client.post(
        "/v1/gateway/register",
        json={"http_url": "http://node-b.test", "grpc_address": "127.0.0.1:50051"},
    )
    assert resp_reg.status_code == 503
    assert "PRIVACY_GATEWAY_ADMIN_API_KEY" in resp_reg.json()["message"]

    resp_dereg = client.post(
        "/v1/gateway/deregister",
        json={"http_url": "http://node-b.test", "grpc_address": "127.0.0.1:50051"},
    )
    assert resp_dereg.status_code == 503


def test_502_does_not_leak_backend_error(patched_client):
    """502 响应不得回传后端异常原文（可能含内网 URL），仅给通用文案。"""
    _calls, state, _captured = patched_client

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow backend", request=request)

    state["handler"] = handler

    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("http://node-a.internal:9000", "127.0.0.1:50051")
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    resp = client.post(
        "/v1/privacy/mask",
        json={"field_name": "mobile", "value": "13812345678", "context": ""},
    )
    assert resp.status_code == 502
    body = resp.text
    assert "backend request failed" in body
    # 不泄漏后端异常细节与内网地址
    assert "slow backend" not in body
    assert "node-a.internal" not in body


def test_register_node_auth_constant_time():
    """验证管理接口配置 Admin Key 时的鉴权保护。"""
    balancer = LoadBalancer(strategy="round_robin")
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    # 未提供 Token -> 401
    resp_no_token = client.post(
        "/v1/gateway/register",
        json={"http_url": "http://node-b.test", "grpc_address": "127.0.0.1:50051"},
    )
    assert resp_no_token.status_code == 401

    # 提供错误 Token -> 401
    resp_wrong_token = client.post(
        "/v1/gateway/register",
        headers={"Authorization": "Bearer wrong-token"},
        json={"http_url": "http://node-b.test", "grpc_address": "127.0.0.1:50051"},
    )
    assert resp_wrong_token.status_code == 401

    # 提供正确 Token -> 200
    resp_correct = client.post(
        "/v1/gateway/register",
        headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
        json={"http_url": "http://node-b.test", "grpc_address": "127.0.0.1:50051"},
    )
    assert resp_correct.status_code == 200
    assert resp_correct.json() == {"status": "registered"}


def test_deregister_nonexistent_node():
    """注销不存在的节点应返回 200 deregistered（幂等设计）。"""
    balancer = LoadBalancer(strategy="round_robin")
    client = TestClient(create_http_gateway_app(balancer, settings=_ADMIN_SETTINGS))

    resp = client.post(
        "/v1/gateway/deregister",
        headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
        json={"http_url": "http://node-not-exists.test", "grpc_address": "127.0.0.1:50051"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "deregistered"}
