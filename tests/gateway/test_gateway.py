"""网关与负载均衡集成测试。

在后台启动真实的引擎实例，并通过网关的 HTTP 与 gRPC 透明代理进行请求分发与
连通性验证。gRPC 侧额外使用一个通用回显后端验证 streaming、metadata 白名单、
凭据透传/兜底注入与逐请求准入。
"""

import asyncio
import json
import socket
import threading
import time

import grpc
import pytest
import uvicorn
from fastapi.testclient import TestClient

from engine import privacy_pb2, privacy_pb2_grpc
from engine.gateway.balancer import LoadBalancer
from engine.gateway.grpc_proxy import start_grpc_gateway
from engine.gateway.http_proxy import create_http_gateway_app
from engine.gateway.settings import GatewaySettings
from engine.grpc_server import serve as grpc_serve
from engine.main import app as rest_app

pytestmark = pytest.mark.integration


def find_free_port() -> int:
    """寻找本地空闲端口。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _identity(value: bytes) -> bytes:
    return value


@pytest.fixture(scope="module")
def backend_agent():
    """启动一个真实的后台引擎实例。"""
    rest_port = find_free_port()
    grpc_port = find_free_port()

    # 直接使用 lambda 调用，动态传入分配的空闲端口，避开 import-time 环境变量求值陷阱
    rest_thread = threading.Thread(
        target=lambda: uvicorn.run(rest_app, host="127.0.0.1", port=rest_port, log_level="error"),
        daemon=True,
    )
    rest_thread.start()

    grpc_thread = threading.Thread(
        target=lambda: grpc_serve(port=grpc_port),
        daemon=True,
    )
    grpc_thread.start()

    # 等待后台线程初始化并开始监听端口
    time.sleep(2.5)

    yield {
        "http_url": f"http://127.0.0.1:{rest_port}",
        "grpc_address": f"127.0.0.1:{grpc_port}",
    }


class _EchoHandler(grpc.GenericRpcHandler):
    """通用回显后端：把入站 metadata 与每个请求字节原样回给客户端。"""

    def __init__(self) -> None:
        self.observed_metadata: list[dict[str, str]] = []

    def service(self, handler_call_details: grpc.HandlerCallDetails):
        async def behavior(request_iterator, context):
            metadata = {key: value for key, value in context.invocation_metadata()}
            self.observed_metadata.append(metadata)
            context.set_trailing_metadata((("x-echo-trailer", "yes"),))
            header = json.dumps(metadata, sort_keys=True).encode("utf-8")
            async for request in request_iterator:
                yield header + b"\n" + request

        return grpc.stream_stream_rpc_method_handler(
            behavior,
            request_deserializer=_identity,
            response_serializer=_identity,
        )


async def _start_echo_backend() -> tuple[grpc.aio.Server, str, _EchoHandler]:
    port = find_free_port()
    server = grpc.aio.server()
    handler = _EchoHandler()
    server.add_generic_rpc_handlers((handler,))
    server.add_insecure_port(f"127.0.0.1:{port}")
    await server.start()
    return server, f"127.0.0.1:{port}", handler


async def _start_gateway(balancer: LoadBalancer, settings: GatewaySettings) -> grpc.aio.Server:
    server = await start_grpc_gateway(settings, balancer)
    return server


def test_load_balancer_strategies(backend_agent):
    """测试负载均衡选择算法。"""
    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node(backend_agent["http_url"], backend_agent["grpc_address"])

    loop = asyncio.new_event_loop()
    try:
        node = loop.run_until_complete(balancer.select_node())
        assert node is not None
        assert node.http_url == backend_agent["http_url"]
        assert node.grpc_address == backend_agent["grpc_address"]
        assert node.is_healthy is True

        # 测试最小连接数策略
        balancer.strategy = "least_conn"
        node.active_connections = 5

        # 添加另一个活跃连接为 0 的节点
        balancer.add_node("http://127.0.0.1:9999", "127.0.0.1:9999")
        selected = loop.run_until_complete(balancer.select_node())
        assert selected.grpc_address == "127.0.0.1:9999"

        # 关闭 gRPC 连接
        loop.run_until_complete(balancer.close_all())
    finally:
        loop.close()


def test_http_proxy_forwarding(backend_agent):
    """测试 HTTP 代理路由请求转发。"""
    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node(backend_agent["http_url"], backend_agent["grpc_address"])
    settings = GatewaySettings()

    gateway_app = create_http_gateway_app(balancer, settings=settings)
    client = TestClient(gateway_app)

    try:
        # 1. /health 是网关本地端点，不转发后端
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json() == {"status": "ok", "component": "gateway"}

        # 2. 验证脱敏接口转发
        res_mask = client.post(
            "/v1/privacy/mask",
            json={"field_name": "mobile", "value": "13812345678", "context": ""},
        )
        assert res_mask.status_code == 200
        assert res_mask.json()["result"] == "138****5678"

        # 3. 清空可用节点后返回 503
        balancer.nodes = []
        res_err = client.get("/v1/privacy/budget")
        assert res_err.status_code == 503
        assert res_err.json()["code"] == "SERVICE_UNAVAILABLE"
    finally:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(balancer.close_all())
        finally:
            loop.close()


@pytest.mark.anyio
async def test_grpc_transparent_unary_forwarding(backend_agent):
    """gRPC 透明代理转发 typed unary RPC（含未手写映射的泛化方法）。"""
    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node(backend_agent["http_url"], backend_agent["grpc_address"])
    settings = GatewaySettings(grpc_port=find_free_port())
    server = await _start_gateway(balancer, settings)
    channel = grpc.aio.insecure_channel(settings.grpc_address())

    try:
        stub = privacy_pb2_grpc.PrivacyServiceStub(channel)

        res = await stub.Mask(
            privacy_pb2.MaskRequest(field_name="mobile", value="13812345678", context="")
        )
        assert res.result == "138****5678"

        res_health = await stub.Health(privacy_pb2.HealthRequest())
        assert res_health.status == "ok"

        res_rec = await stub.RecommendParams(
            privacy_pb2.RecommendRequest(namespace="default", values=[1.0, 2.0, 3.0])
        )
        assert res_rec.status == "success"
        assert res_rec.namespace == "default"
    finally:
        await channel.close()
        await server.stop(grace=0.1)
        await balancer.close_all()


@pytest.mark.anyio
async def test_grpc_streaming_and_metadata_allowlist():
    """任意基数的流式转发 + outbound metadata 白名单。"""
    echo_server, echo_address, echo_handler = await _start_echo_backend()
    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("", echo_address)
    settings = GatewaySettings(grpc_port=find_free_port())
    gateway_server = await _start_gateway(balancer, settings)
    channel = grpc.aio.insecure_channel(settings.grpc_address())

    try:
        call = channel.stream_stream(
            "/echo.v1.Echo/Echo",
            request_serializer=_identity,
            response_deserializer=_identity,
        )

        async def requests():
            yield b"one"
            yield b"two"

        call_obj = call(
            requests(),
            metadata=(
                ("x-request-id", "rid-123"),
                ("x-evil", "drop-me"),
                ("authorization", "Bearer inbound-token"),
            ),
        )

        responses = [response async for response in call_obj]
        assert len(responses) == 2
        assert responses[0].endswith(b"\none")
        assert responses[1].endswith(b"\ntwo")

        forwarded = echo_handler.observed_metadata[0]
        assert forwarded["x-request-id"] == "rid-123"
        assert "x-evil" not in forwarded
        assert forwarded["authorization"] == "Bearer inbound-token"

        trailers = list(await call_obj.trailing_metadata())
        assert ("x-echo-trailer", "yes") in trailers
    finally:
        await channel.close()
        await gateway_server.stop(grace=0.1)
        await echo_server.stop(grace=0.1)
        await balancer.close_all()


@pytest.mark.anyio
async def test_grpc_backend_credential_fallback_only_when_anonymous():
    """匿名调用注入回源凭据；已有凭据（哪怕畸形）原样透传。"""
    echo_server, echo_address, echo_handler = await _start_echo_backend()
    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("", echo_address)
    settings = GatewaySettings(grpc_port=find_free_port(), backend_api_key="fallback-key")
    gateway_server = await _start_gateway(balancer, settings)
    channel = grpc.aio.insecure_channel(settings.grpc_address())

    try:
        call = channel.stream_stream(
            "/echo.v1.Echo/Echo",
            request_serializer=_identity,
            response_deserializer=_identity,
        )

        async def single():
            yield b"x"

        # 1) 匿名 -> 注入 Bearer fallback-key
        first = [response async for response in call(single())]
        assert first[0].endswith(b"\nx")
        assert echo_handler.observed_metadata[0]["authorization"] == "Bearer fallback-key"

        # 2) 携带畸形凭据 -> 不做任何改写或兜底注入
        second = [
            response
            async for response in call(single(), metadata=(("authorization", "Token malformed"),))
        ]
        assert second[0].endswith(b"\nx")
        assert echo_handler.observed_metadata[1]["authorization"] == "Token malformed"
    finally:
        await channel.close()
        await gateway_server.stop(grace=0.1)
        await echo_server.stop(grace=0.1)
        await balancer.close_all()


@pytest.mark.anyio
async def test_grpc_admission_requires_authorization():
    """auth_enabled 时匿名请求在选路/建连前即被 UNAUTHENTICATED 拒绝。"""
    echo_server, echo_address, echo_handler = await _start_echo_backend()
    balancer = LoadBalancer(strategy="round_robin")
    balancer.add_node("", echo_address)
    settings = GatewaySettings(grpc_port=find_free_port(), auth_enabled=True)
    gateway_server = await _start_gateway(balancer, settings)
    channel = grpc.aio.insecure_channel(settings.grpc_address())

    try:
        call = channel.stream_stream(
            "/echo.v1.Echo/Echo",
            request_serializer=_identity,
            response_deserializer=_identity,
        )

        async def single():
            yield b"x"

        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            async for _ in call(single()):
                pass
        assert exc_info.value.code() is grpc.StatusCode.UNAUTHENTICATED
        assert echo_handler.observed_metadata == []

        responses = [
            response
            async for response in call(single(), metadata=(("authorization", "Bearer caller"),))
        ]
        assert responses[0].endswith(b"\nx")
        assert echo_handler.observed_metadata[0]["authorization"] == "Bearer caller"
    finally:
        await channel.close()
        await gateway_server.stop(grace=0.1)
        await echo_server.stop(grace=0.1)
        await balancer.close_all()
