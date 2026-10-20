"""HTTP 反向代理网关 (Go ``internal/gateway/http_proxy.go`` parity).

与 Go 网关逐语义对齐：

- **单次尝试**：选中节点后只转发一次，绝不重试；失败即 502 ``BAD_GATEWAY``。
- **准入与选路分离**：``select_node`` 只做只读预筛，``breaker.acquire()`` 决定
  本次请求是否放行；``CIRCUIT_OPEN`` / ``SERVICE_UNAVAILABLE`` 均为 503。
- **熔断记账一次**：传输失败只记一次 failure；响应状态 <500 记 success，
  >=500 记 failure。
- **P2C-EWMA**：转发后按 alpha=0.3 更新节点延迟。
- **业务口本地 /health**：不转发到后端。
- **统一错误信封**：所有本地错误与 Go ``middleware.AbortWithError`` 同形。
- **逐请求凭据准入**：``PRIVACY_GATEWAY_AUTH_ENABLED`` 开启时除 /health 外
  必须携带 Authorization（只判存在性，合法性由后端的 scope 裁决）。
- **IP 白名单 / HTTPS 强制 / 请求体上限**：与 Go 中间件同口径。

Python 额外保留（超集）：``/v1/gateway/{register,deregister,isolate,drain,activate}``
拓扑管理端点，由 ``PRIVACY_GATEWAY_ADMIN_API_KEY``（缺省继承 metrics key）
保护；未配置时 fail-closed 503。
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine.observability.logging_config import get_logger
from engine.observability.metrics import (
    GATEWAY_BACKEND_EWMA_LATENCY,
    GATEWAY_BACKEND_IN_FLIGHT,
    GATEWAY_CIRCUIT_BREAKER_STATE,
    GATEWAY_REQUESTS_TOTAL,
)

from .balancer import LoadBalancer, backend_tls_verify
from .inbound import InboundKeyStore
from .settings import GatewaySettings, is_loopback_host

logger = get_logger(__name__)

# RFC 7230 hop-by-hop headers. ``host`` is rebuilt by httpx from the target URL,
# matching Go ReverseProxy (Host is not part of the forwarded Header map).
EXCLUDE_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}
RESPONSE_EXCLUDE_HEADERS = EXCLUDE_HEADERS | {"content-encoding", "content-length"}

_LOCAL_PATHS = {"/health"}
_CB_STATE_VALUES = {"closed": 0.0, "half-open": 1.0, "open": 2.0}
_METRIC_STATE_NAMES = {"closed": "closed", "half-open": "half_open", "open": "open"}


class _HeaderPairs(list[tuple[str, str]]):
    """Starlette expects ``.items()``; a plain list would drop duplicate headers."""

    def items(self):  # type: ignore[override]
        return iter(self)


class RegisterRequest(BaseModel):
    http_url: str
    grpc_address: str
    weight: int = 1


class DeregisterRequest(BaseModel):
    http_url: str
    grpc_address: str


class IsolateRequest(BaseModel):
    http_url: str
    grpc_address: str


class DrainRequest(BaseModel):
    http_url: str
    grpc_address: str


class ActivateRequest(BaseModel):
    http_url: str
    grpc_address: str


def _trace_id(request: Request) -> str:
    return (
        request.headers.get("x-request-id")
        or getattr(request.state, "request_id", "")
        or f"req-{int(time.time())}"
    )


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    detail: object | None = None,
) -> JSONResponse:
    """Unified 5-field error envelope (Go ``AbortWithError`` parity)."""
    trace_id = _trace_id(request)
    return JSONResponse(
        status_code=status_code,
        headers={"X-Request-ID": trace_id, "X-Trace-ID": trace_id},
        content={
            "code": code,
            "message": message,
            "detail": detail,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z",
        },
    )


def _peer_ip(request: Request) -> str:
    client = request.client
    return client.host if client else ""


def parse_networks(cidrs: tuple[str, ...] | list[str]) -> list[ipaddress._BaseNetwork]:
    networks: list[ipaddress._BaseNetwork] = []
    for raw in cidrs:
        entry = raw.strip()
        if not entry:
            continue
        if "/" not in entry:
            entry = entry + ("/32" if ":" not in entry else "/128")
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("IPAllowlist: invalid CIDR skipped", extra={"entry": raw})
    return networks


def ip_allowed(networks: list[ipaddress._BaseNetwork], ip: str) -> bool:
    if not networks:
        return True
    try:
        address = ipaddress.ip_address((ip or "").strip("[]"))
    except ValueError:
        return False
    return any(address in network for network in networks)


def _real_client_ip(request: Request, trusted_proxies: list[ipaddress._BaseNetwork]) -> str:
    peer = _peer_ip(request)
    if trusted_proxies and ip_allowed(trusted_proxies, peer):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
    return peer


def _resolve_proto(
    request: Request, trusted_proxies: list[ipaddress._BaseNetwork]
) -> str:
    if request.url.scheme == "https" or request.scope.get("scheme") == "https":
        return "https"
    if trusted_proxies and ip_allowed(trusted_proxies, _peer_ip(request)):
        forwarded = request.headers.get("x-forwarded-proto", "").strip().lower()
        if forwarded:
            return forwarded
    return "http"


class NodeMetrics:
    """Per-node metric helpers shared by the REST and gRPC proxies."""

    @staticmethod
    def set_in_flight(node_id: str, backend_addr: str, count: int) -> None:
        GATEWAY_BACKEND_IN_FLIGHT.labels(node_id=node_id, backend_addr=backend_addr).set(
            float(count)
        )

    @staticmethod
    def set_ewma(node_id: str, latency_seconds: float) -> None:
        GATEWAY_BACKEND_EWMA_LATENCY.labels(node_id=node_id).set(latency_seconds)

    @staticmethod
    def set_cb(node_id: str, state: str) -> None:
        name = _METRIC_STATE_NAMES.get(state, state.replace("-", "_"))
        GATEWAY_CIRCUIT_BREAKER_STATE.labels(node_id=node_id, state=name).set(
            _CB_STATE_VALUES.get(state, 0.0)
        )

    @staticmethod
    def record_forwarded(node_id: str, status: int) -> None:
        GATEWAY_REQUESTS_TOTAL.labels(node_id=node_id, status=str(status)).inc()


def create_http_gateway_app(
    balancer: LoadBalancer,
    *,
    settings: GatewaySettings | None = None,
    inbound_store: InboundKeyStore | None = None,
) -> FastAPI:
    """Build the REST reverse-proxy app (single attempt, Go semantics)."""
    cfg = settings or GatewaySettings.from_env()
    if cfg.backend_api_key:
        balancer.set_backend_credential(cfg.backend_api_key)
    allowed_networks = parse_networks(cfg.allowed_cidrs)
    trusted_networks = parse_networks(cfg.trusted_proxies)
    if (cfg.require_tls or cfg.require_forwarded_proto) and not trusted_networks:
        trusted_networks = parse_networks(("127.0.0.0/8", "::1/128"))
    enforce_https = cfg.require_tls or cfg.require_forwarded_proto
    admin_key = cfg.admin_api_key

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=500),
            trust_env=False,
            verify=backend_tls_verify(),
        )
        yield
        await app.state.http_client.aclose()

    app = FastAPI(title="PrivShield REST Gateway", lifespan=lifespan)
    app.state.gateway_settings = cfg
    app.state.inbound_store = inbound_store

    def require_management_auth(authorization: str | None) -> None:
        if not admin_key:
            logger.warning(
                "Gateway management endpoint rejected: admin key is not configured"
            )
            raise _ManagementDenied(
                503,
                "SERVICE_UNAVAILABLE",
                "Gateway management API is disabled: PRIVACY_GATEWAY_ADMIN_API_KEY "
                "(or PRIVACY_GATEWAY_METRICS_API_KEY) is not configured",
            )
        expected = f"Bearer {admin_key}"
        if not authorization or not hmac.compare_digest(
            authorization.encode("utf-8"), expected.encode("utf-8")
        ):
            raise _ManagementDenied(
                401, "UNAUTHORIZED", "Unauthorized gateway management request"
            )

    @app.middleware("http")
    async def _access_control(request: Request, call_next):  # type: ignore[no-untyped-def]
        # 1) IP allowlist (Go `middleware.IPAllowlist`), CIDR list shared with gRPC.
        if allowed_networks:
            client_ip = _real_client_ip(request, trusted_networks)
            if not client_ip:
                return error_response(
                    request, 403, "FORBIDDEN", "unable to determine client IP"
                )
            if not ip_allowed(allowed_networks, client_ip):
                return error_response(
                    request, 403, "IP_NOT_ALLOWED", "client IP not in allowed CIDR ranges"
                )

        path = request.url.path
        # 2) HTTPS enforcement (Go step 6). /health is exempt.
        if enforce_https and path != "/health":
            proto = _resolve_proto(request, trusted_networks)
            client_ip = _real_client_ip(request, trusted_networks)
            if proto != "https" and not is_loopback_host(client_ip):
                return error_response(
                    request,
                    426,
                    "HTTPS_REQUIRED",
                    "HTTPS is required by gateway security policy; ensure upstream "
                    "Ingress/LB terminates TLS",
                )

        # 3) Per-request credential admission (Go `requireInboundCredential`).
        if cfg.auth_enabled and path != "/health":
            if not (request.headers.get("authorization") or "").strip():
                return error_response(
                    request,
                    401,
                    "UNAUTHENTICATED",
                    "credential required: the gateway admits only authenticated callers",
                )

        # 4) Optional inbound keyfile verification (Go `InboundPolicy`).
        if inbound_store is not None:
            health_exempt = path == "/health" and request.method in ("GET", "HEAD")
            if not health_exempt:
                decision = inbound_store.verify(
                    request.headers.get("authorization"),
                    request.method,
                    path,
                    request.headers.get("x-privshield-contract-version", ""),
                    health_exempt=False,
                )
                if not decision.ok:
                    return error_response(
                        request, decision.status, decision.code, decision.message
                    )

        return await call_next(request)

    @app.middleware("http")
    async def _aggregate_metrics(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        path = request.url.path
        if path not in _LOCAL_PATHS and not path.startswith("/v1/gateway/"):
            GATEWAY_REQUESTS_TOTAL.labels(
                node_id="aggregate", status=str(response.status_code)
            ).inc()
        return response

    @app.api_route("/health", methods=["GET", "HEAD"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "component": "gateway"}

    @app.post("/v1/gateway/register")
    async def register_node(
        req: RegisterRequest, authorization: str | None = Header(default=None)
    ):
        require_management_auth(authorization)
        if not (req.http_url.startswith("http://") or req.http_url.startswith("https://")):
            raise _ManagementDenied(
                400,
                "INVALID_ARGUMENT",
                "Invalid http_url scheme, must be http:// or https://",
            )
        balancer.add_node(req.http_url, req.grpc_address, req.weight)
        logger.info(
            "Node registered via API",
            extra={"http_url": req.http_url, "grpc_address": req.grpc_address},
        )
        return {"status": "registered"}

    @app.post("/v1/gateway/deregister")
    async def deregister_node(
        req: DeregisterRequest, authorization: str | None = Header(default=None)
    ):
        require_management_auth(authorization)
        balancer.remove_node(req.http_url, req.grpc_address)
        logger.info(
            "Node deregistered via API",
            extra={"http_url": req.http_url, "grpc_address": req.grpc_address},
        )
        return {"status": "deregistered"}

    @app.post("/v1/gateway/isolate")
    async def isolate_node(
        req: IsolateRequest, authorization: str | None = Header(default=None)
    ):
        require_management_auth(authorization)
        if not balancer.isolate_node(req.http_url, req.grpc_address):
            raise _ManagementDenied(404, "NOT_FOUND", "Node not found")
        return {"status": "isolated"}

    @app.post("/v1/gateway/drain")
    async def drain_node(
        req: DrainRequest, authorization: str | None = Header(default=None)
    ):
        require_management_auth(authorization)
        if not balancer.drain_node(req.http_url, req.grpc_address):
            raise _ManagementDenied(404, "NOT_FOUND", "Node not found")
        return {"status": "drained"}

    @app.post("/v1/gateway/activate")
    async def activate_node(
        req: ActivateRequest, authorization: str | None = Header(default=None)
    ):
        require_management_auth(authorization)
        if not balancer.activate_node(req.http_url, req.grpc_address):
            raise _ManagementDenied(404, "NOT_FOUND", "Node not found")
        return {"status": "activated"}

    @app.exception_handler(_ManagementDenied)
    async def _management_denied_handler(
        request: Request, exc: _ManagementDenied
    ) -> JSONResponse:
        return error_response(request, exc.status_code, exc.code, exc.message)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    )
    async def proxy_request(path: str, request: Request):
        node = await balancer.select_node()
        if node is None:
            logger.error(
                "No healthy backend nodes available",
                extra={"path": path, "method": request.method},
            )
            return error_response(
                request, 503, "SERVICE_UNAVAILABLE", "无可用后端节点", "all backends exhausted"
            )

        node_id = node.rest_node_id
        permit = node.circuit_breaker.acquire()
        if permit is None:
            NodeMetrics.set_cb(node_id, node.circuit_breaker.state)
            return error_response(
                request,
                503,
                "CIRCUIT_OPEN",
                f"后端 {node_id} 熔断器开启",
                "circuit breaker is open",
            )

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in EXCLUDE_HEADERS
        }
        client_ip = _real_client_ip(request, trusted_networks) or "127.0.0.1"
        if "x-forwarded-for" in headers:
            headers["x-forwarded-for"] = f"{headers['x-forwarded-for']}, {client_ip}"
        else:
            headers["x-forwarded-for"] = client_ip
        if "x-real-ip" not in headers:
            headers["x-real-ip"] = client_ip

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > cfg.max_body_bytes:
                    permit.release()
                    return error_response(
                        request,
                        413,
                        "PAYLOAD_TOO_LARGE",
                        f"Payload too large: exceeds {cfg.max_body_bytes} bytes",
                    )
            except ValueError:
                pass
        body = await request.body()
        if len(body) > cfg.max_body_bytes:
            permit.release()
            return error_response(
                request,
                413,
                "PAYLOAD_TOO_LARGE",
                f"Payload too large: exceeds {cfg.max_body_bytes} bytes",
            )

        # 入站凭据原样透传；仅在关闭准入且入站未带凭据时注入回源兜底 Key。
        inbound_auth = (headers.get("authorization") or "").strip()
        if not inbound_auth and not cfg.auth_enabled and node.backend_auth_header:
            headers["authorization"] = node.backend_auth_header

        client = _get_http_client(request)
        url = f"{node.http_url}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        node.active_connections = node.active_connections + 1
        NodeMetrics.set_in_flight(node_id, node_id, node.active_connections)
        start = time.perf_counter()
        try:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
            latency = time.perf_counter() - start
            node.update_ewma(latency, 0.3)
            if response.status_code < 500:
                permit.success()
            else:
                permit.failure()
            NodeMetrics.set_ewma(node_id, node.ewma)
            NodeMetrics.set_cb(node_id, node.circuit_breaker.state)
            NodeMetrics.record_forwarded(node_id, response.status_code)
            response_headers = [
                (key, value)
                for key, value in response.headers.multi_items()
                if key.lower() not in RESPONSE_EXCLUDE_HEADERS
            ]
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=_HeaderPairs(response_headers),
            )
        except Exception as exc:  # noqa: BLE001 - transport failures are expected
            latency = time.perf_counter() - start
            node.update_ewma(latency, 0.3)
            permit.failure()
            NodeMetrics.set_ewma(node_id, node.ewma)
            NodeMetrics.set_cb(node_id, node.circuit_breaker.state)
            NodeMetrics.record_forwarded(node_id, 502)
            logger.error(
                "HTTP proxy upstream failure",
                extra={
                    "path": path,
                    "method": request.method,
                    "node": node_id,
                    "error": str(exc),
                },
            )
            return error_response(
                request,
                502,
                "BAD_GATEWAY",
                "upstream gateway unavailable",
                "backend request failed",
            )
        finally:
            node.active_connections = node.active_connections - 1
            NodeMetrics.set_in_flight(node_id, node_id, node.active_connections)

    return app


class _ManagementDenied(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _get_http_client(request: Request) -> httpx.AsyncClient:
    """Return the app-level client, rebuilding it when the event loop changed."""
    current_loop = asyncio.get_running_loop()
    client = getattr(request.app.state, "http_client", None)
    cached_loop = getattr(request.app.state, "http_client_loop", None)
    if client is None or cached_loop is not current_loop:
        if client is not None:
            asyncio.create_task(client.aclose())  # noqa: RUF006
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=500),
            trust_env=False,
            verify=backend_tls_verify(),
        )
        request.app.state.http_client = client
        request.app.state.http_client_loop = current_loop
    return client
