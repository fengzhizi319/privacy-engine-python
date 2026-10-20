"""gRPC 透明流代理 (Go ``internal/gateway/grpc_proxy.go`` parity).

Go 网关用 ``UnknownServiceHandler`` + raw codec 把任意方法名与任意基数
（unary / server-stream / client-stream / bidi）的字节流转发给后端。Python
侧用 ``grpc.GenericRpcHandler`` 注册**每个方法都返回 stream-stream handler**
的泛化处理器，序列化器为恒等函数——客户端是 unary 还是 streaming 由线协议
决定，服务端统一按双向流读写即可，因此一份处理器覆盖全部 RPC 形态。

与 Go 逐语义对齐：

- **单次尝试**：选中节点后只转发一次，失败不重试。
- **准入先于选路**：CIDR → 逐请求凭据门禁 → keyfile 策略 → ``select_node``
  → ``breaker.acquire()``；缺凭据的调用不消耗后端连接与熔断预算。
- **出站 metadata 白名单**：只透传链路追踪/幂等键；``authorization`` 单独
  保留（存在且非空白即原样透传，包括畸形凭据，交给后端按真实原因拒绝），
  仅在匿名且未开启准入时才注入 ``Bearer <PRIVACY_GATEWAY_BACKEND_API_KEY>``。
- **半关闭语义**：客户端发完请求只关闭发送方向，继续排空后端响应；后端
  响应结束即调用结束（与 Go 修复后的 ``CloseSend`` 语义一致）。
- **熔断记账一次**：流正常结束记 success，任何 RPC/传输错误记 failure。
  响应状态 200 / 502 计入 ``privshield_gateway_requests_total{node_id,status}``，
  EWMA alpha=0.2（Go gRPC 侧口径）。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Iterable
from typing import Any

import grpc

from engine.observability.logging_config import get_logger

from .balancer import GRPC_MAX_MESSAGE_BYTES, LoadBalancer
from .http_proxy import NodeMetrics, ip_allowed, parse_networks
from .inbound import InboundKeyStore
from .settings import GatewaySettings

logger = get_logger(__name__)

_GRPC_EWMA_ALPHA = 0.2

# Go outboundMetadataAllowlist：凭据类与 x-internal-* 一律剔除。
_ALLOWED_METADATA = frozenset(
    {
        "x-request-id",
        "x-trace-id",
        "x-idempotency-key",
        "traceparent",
        "tracestate",
        "baggage",
        "b3",
        "x-b3-traceid",
        "x-b3-spanid",
        "x-b3-parentspanid",
        "x-b3-sampled",
        "x-b3-flags",
    }
)

_RESERVED_TRAILER_KEYS = frozenset(
    {"grpc-status", "grpc-message", "grpc-status-details-bin"}
)

_SERVER_OPTIONS = [
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
    ("grpc.max_concurrent_streams", 250),
    ("grpc.max_connection_idle_ms", 5 * 60 * 1000),
    ("grpc.max_connection_age_ms", 2 * 60 * 60 * 1000),
]


def _identity(value: bytes) -> bytes:
    """Zero-copy passthrough serializer: the proxy never decodes payloads."""
    return value


def _metadata_pairs(metadata: Iterable[Any] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in metadata or ():
        if hasattr(item, "key"):
            pairs.append((str(item.key).lower(), str(item.value)))
        else:
            key, value = item
            pairs.append((str(key).lower(), str(value)))
    return pairs


def _filter_outbound_metadata(
    pairs: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    return [(key, value) for key, value in pairs if key in _ALLOWED_METADATA]


def _first(pairs: Iterable[tuple[str, str]], key: str) -> str:
    for item_key, value in pairs:
        if item_key == key:
            return value
    return ""


def _has_authorization(pairs: Iterable[tuple[str, str]]) -> bool:
    return any(key == "authorization" and value.strip() for key, value in pairs)


def grpc_peer_ip(peer: str) -> str:
    """Extract the client IP from a gRPC peer string (``ipv4:``/``ipv6:``)."""
    value = (peer or "").strip()
    for prefix in ("ipv4:", "ipv6:"):
        if value.startswith(prefix):
            rest = value[len(prefix) :]
            if rest.startswith("["):
                end = rest.find("]")
                return rest[1:end] if end > 0 else rest
            return rest.rsplit(":", 1)[0]
    return value


class _ProxyAbort(Exception):
    """Internal signal that the proxy must abort the inbound RPC."""

    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__(details)
        self.code = code
        self.details = details


class GatewayGrpcProxy:
    """Transparent stream proxy with Go-compatible admission and accounting."""

    def __init__(
        self,
        balancer: LoadBalancer,
        settings: GatewaySettings,
        inbound_store: InboundKeyStore | None = None,
    ):
        self.balancer = balancer
        self.settings = settings
        self.inbound_store = inbound_store
        self.allowed_networks = parse_networks(settings.allowed_cidrs)

    async def handle(
        self,
        method: str,
        request_iterator: AsyncIterator[bytes],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[bytes]:
        """Forward one RPC (any cardinality) to a selected backend."""
        inbound_md = _metadata_pairs(context.invocation_metadata())

        # 0) CIDR admission. UnknownServiceHandler bypasses interceptors, so the
        #    gateway must enforce PRIVACY_GATEWAY_ALLOWED_CIDRS here as well.
        if self.allowed_networks:
            client_ip = grpc_peer_ip(context.peer())
            if not ip_allowed(self.allowed_networks, client_ip):
                logger.warning(
                    "gRPC proxy rejected peer outside allowed CIDRs",
                    extra={"peer_ip": client_ip, "method": method},
                )
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    "client IP not in allowed CIDR ranges",
                )

        # 0.5) Per-request credential admission before selection.
        if self.settings.auth_enabled and not _has_authorization(inbound_md):
            logger.warning(
                "gRPC proxy rejected stream without inbound credential",
                extra={"method": method},
            )
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "missing credential: gateway requires an authorization metadata",
            )

        if self.inbound_store is not None:
            decision = self.inbound_store.verify(
                _first(inbound_md, "authorization"),
                "GRPC",
                method,
                _first(inbound_md, "x-privshield-contract-version"),
            )
            if not decision.ok:
                code = (
                    grpc.StatusCode.UNAUTHENTICATED
                    if decision.status == 401
                    else grpc.StatusCode.PERMISSION_DENIED
                )
                await context.abort(code, f"{decision.code}: {decision.message}")

        # 1) Select a backend (read-only filter), then acquire the permit.
        node = await self.balancer.select_node()
        if node is None:
            logger.error("No healthy gRPC backend nodes available", extra={"method": method})
            await context.abort(grpc.StatusCode.UNAVAILABLE, "no backend agent available")

        node_id = node.grpc_node_id
        permit = node.circuit_breaker.acquire()
        if permit is None:
            NodeMetrics.set_cb(node_id, node.circuit_breaker.state)
            await context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"backend {node_id} circuit breaker open",
            )

        # 2) Outbound metadata: allowlist + authorization passthrough/fallback.
        out_metadata = _filter_outbound_metadata(inbound_md)
        if _has_authorization(inbound_md):
            out_metadata.extend(
                (key, value) for key, value in inbound_md if key == "authorization"
            )
        elif not self.settings.auth_enabled and node.backend_auth_header:
            out_metadata.append(("authorization", node.backend_auth_header))

        call: Any = None
        succeeded = False
        failure_detail = ""
        node.active_connections = node.active_connections + 1
        NodeMetrics.set_in_flight(node_id, node_id, node.active_connections)
        start = time.perf_counter()

        try:
            try:
                stream_callable = node.grpc_channel.stream_stream(
                    method,
                    request_serializer=_identity,
                    response_deserializer=_identity,
                )
                # grpc.aio pumps the inbound request iterator itself and closes
                # the send direction on EOS — the same half-close semantics as
                # Go's CloseSend() after serverStream.RecvMsg returns io.EOF.
                call = stream_callable(
                    request_iterator,
                    timeout=context.time_remaining(),
                    metadata=out_metadata,
                )
            except Exception as exc:  # noqa: BLE001 - dial/credential failures
                raise _ProxyAbort(
                    grpc.StatusCode.UNAVAILABLE, f"backend stream error: {exc}"
                ) from exc

            initial_md = await call.initial_metadata()
            forward_initial = [
                (key, value)
                for key, value in _metadata_pairs(initial_md)
                if not key.startswith("grpc-")
            ]
            if forward_initial:
                await context.send_initial_metadata(forward_initial)

            async for response in call:
                yield response

            trailers = await call.trailing_metadata()
            forward_trailers = [
                (key, value)
                for key, value in _metadata_pairs(trailers)
                if key not in _RESERVED_TRAILER_KEYS
            ]
            if forward_trailers:
                context.set_trailing_metadata(forward_trailers)
            succeeded = True
        except _ProxyAbort as exc:
            failure_detail = f"{exc.code.name}: {exc.details}"
            if not context.cancelled():
                await context.abort(exc.code, exc.details)
        except grpc.aio.AioRpcError as exc:
            failure_detail = f"{exc.code().name}: {exc.details()}"
            if not context.cancelled():
                await context.abort(exc.code(), exc.details())
        except asyncio.CancelledError:
            failure_detail = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - proxy-internal failure
            failure_detail = str(exc)
            if not context.cancelled():
                await context.abort(
                    grpc.StatusCode.INTERNAL, "gateway proxy internal error"
                )
        finally:
            if call is not None and not succeeded:
                with contextlib.suppress(Exception):
                    await call.cancel()

            latency = time.perf_counter() - start
            node.active_connections = node.active_connections - 1
            node.update_ewma(latency, _GRPC_EWMA_ALPHA)
            if succeeded:
                permit.success()
            else:
                permit.failure()
                if failure_detail:
                    logger.warning(
                        "gRPC proxy stream failed",
                        extra={
                            "method": method,
                            "backend": node_id,
                            "error": failure_detail,
                        },
                    )
            NodeMetrics.set_in_flight(node_id, node_id, node.active_connections)
            NodeMetrics.set_ewma(node_id, node.ewma)
            NodeMetrics.set_cb(node_id, node.circuit_breaker.state)
            NodeMetrics.record_forwarded(node_id, 200 if succeeded else 502)


class _TransparentHandler(grpc.GenericRpcHandler):
    """Return a stream-stream handler for every method (Go UnknownServiceHandler)."""

    def __init__(self, proxy: GatewayGrpcProxy):
        self._proxy = proxy

    def service(self, handler_call_details: grpc.HandlerCallDetails) -> Any:
        method = handler_call_details.method

        async def behavior(
            request_iterator: AsyncIterator[bytes],
            context: grpc.aio.ServicerContext,
        ) -> AsyncIterator[bytes]:
            async for response in self._proxy.handle(method, request_iterator, context):
                yield response

        return grpc.stream_stream_rpc_method_handler(
            behavior,
            request_deserializer=_identity,
            response_serializer=_identity,
        )


def create_grpc_gateway_server(
    balancer: LoadBalancer,
    settings: GatewaySettings,
    inbound_store: InboundKeyStore | None = None,
) -> grpc.aio.Server:
    """Build the async gRPC server with Go-compatible resource limits."""
    if settings.backend_api_key:
        balancer.set_backend_credential(settings.backend_api_key)
    server = grpc.aio.server(options=_SERVER_OPTIONS)
    server.add_generic_rpc_handlers(
        (_TransparentHandler(GatewayGrpcProxy(balancer, settings, inbound_store)),)
    )
    return server


async def start_grpc_gateway(
    settings: GatewaySettings,
    balancer: LoadBalancer,
    inbound_store: InboundKeyStore | None = None,
) -> grpc.aio.Server:
    """Bind and start the gRPC proxy listener (mTLS when stack=tls)."""
    server = create_grpc_gateway_server(balancer, settings, inbound_store)
    address = settings.grpc_address()

    if settings.tls_enabled:
        if not (settings.tls_cert_file and settings.tls_key_file and settings.tls_ca_file):
            raise ValueError(
                "gateway gRPC TLS requires cert, key and CA files; refusing plaintext"
            )
        with open(settings.tls_cert_file, "rb") as handle:
            cert_chain = handle.read()
        with open(settings.tls_key_file, "rb") as handle:
            private_key = handle.read()
        with open(settings.tls_ca_file, "rb") as handle:
            root_certificates = handle.read()
        credentials = grpc.ssl_server_credentials(
            [(private_key, cert_chain)],
            root_certificates=root_certificates,
            require_client_auth=True,
        )
        bound = server.add_secure_port(address, credentials)
        logger.info(
            "Gateway gRPC server started with TLS", extra={"addr": address, "mtls": True}
        )
    else:
        bound = server.add_insecure_port(address)
        if not bound:
            raise RuntimeError(f"gateway gRPC listen failed: {address}")
        if settings.tls_stack in {"tlcp", "both"}:
            logger.info(
                "Gateway gRPC server bound behind TLS terminator (loopback plaintext)",
                extra={"addr": address, "tls_stack": settings.tls_stack},
            )
        else:
            logger.warning(
                "Gateway gRPC server started WITHOUT TLS — plaintext transport",
                extra={"addr": address},
            )

    await server.start()
    return server
