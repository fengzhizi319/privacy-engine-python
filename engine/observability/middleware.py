"""REST middleware and gRPC interceptor for observability.

提供统一的访问日志、metrics 与 request_id 透传。
Provides unified access logs, metrics, and request-id propagation for REST and gRPC.
"""

from __future__ import annotations

import secrets
import time
from typing import TYPE_CHECKING, Any

import grpc
from starlette.middleware.base import BaseHTTPMiddleware

from .context import RequestContext, set_request_context
from .logging_config import get_logger
from .metrics import REQUEST_DURATION, REQUESTS_TOTAL, TRAFFIC_BYTES_TOTAL
from .tracing import context_from_carrier, start_span

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request, Response
    from starlette.middleware.base import RequestResponseEndpoint

logger = get_logger(__name__)

_REQUEST_ID_HEADER = "x-request-id"
_TRACE_ID_HEADER = "x-trace-id"


_MAX_REQUEST_ID_LEN = 128
_REQUEST_ID_SAFE = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-~"
)


def _generate_request_id() -> str:
    """Go ``GenerateRequestID``: ``req-<YYYYMMDDHHMMSS-nanos>-<8hex>``."""
    now_ns = time.time_ns()
    seconds = time.strftime("%Y%m%d%H%M%S", time.localtime(now_ns // 1_000_000_000))
    return f"req-{seconds}-{now_ns % 1_000_000_000:09d}-{secrets.token_hex(4)}"


def _sanitize_request_id(value: str | None) -> str:
    """Go ``SanitizeRequestID``: reject (not truncate) unsafe inbound IDs.

    The ID is echoed in response headers and written to every access-log line,
    so accepting control characters, whitespace or non-ASCII would enable header
    injection, log forging and response amplification. 非法值一律视为未提供，
    由调用方回退到自生成 ID，避免截断后两个请求共用同一追踪前缀。
    """
    if not value or len(value) > _MAX_REQUEST_ID_LEN:
        return ""
    if not value.isascii() or any(ch not in _REQUEST_ID_SAFE for ch in value):
        return ""
    return value


def _request_id_from_headers(request: Request) -> str:
    return _sanitize_request_id(request.headers.get(_REQUEST_ID_HEADER)) or _generate_request_id()


def _metrics_path(request: Request) -> str:
    """Return the route template path for Prometheus labels (bounded cardinality).

    必须在 ``call_next`` 之后调用：Starlette 路由匹配完成后才把命中的
    ``Route`` 写入 ``request.scope["route"]``。未匹配路由（如 404 扫描）
    统一归入 ``"unmatched"``，避免攻击者用随机路径制造无界 label 基数。
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return "unmatched"


def _get_request_id_from_metadata(context: grpc.ServicerContext) -> str:
    """Extract and sanitize x-request-id (or x-trace-id fallback) from metadata."""
    metadata = dict(context.invocation_metadata() or [])
    value = metadata.get(_REQUEST_ID_HEADER, "") or metadata.get(_TRACE_ID_HEADER, "")
    return _sanitize_request_id(value) or _generate_request_id()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that sets request context, logs, and records metrics.

    The ``/metrics`` endpoint is excluded to avoid self-referential noise.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path
        if path == "/metrics":
            return await call_next(request)

        request_id = _request_id_from_headers(request)
        method = request.method

        # Set initial context (identity will be filled after auth dependency runs).
        set_request_context(
            RequestContext(
                request_id=request_id,
                method=method,
                path=path,
                identity_name="",
            )
        )

        start = time.perf_counter()
        status_code = 500
        request_size = 0
        try:
            # 读取请求体以统计流量；FastAPI Request 会缓存 body，后续仍可读取。
            try:
                request_size = len(await request.body())
            except Exception:
                request_size = 0

            # Go ``TraceMiddleware`` 同口径：span 名 ``METHOD FullPath``，
            # 属性 {span.kind=server, http.request.method, http.route}；
            # 父上下文来自入站 W3C traceparent。
            carrier = {key: value for key, value in request.headers.items()}
            span_attributes = {
                "span.kind": "server",
                "http.request.method": method,
                "http.route": "",
            }
            with start_span(
                f"{method} {path}",
                span_attributes,
                context=context_from_carrier(carrier),
            ) as span:
                response = await call_next(request)
                route_path = _metrics_path(request)
                route_template = "" if route_path == "unmatched" else route_path
                if span is not None:
                    span.update_name(f"{method} {route_template}")
                    span.set_attribute("http.route", route_template)
            status_code = response.status_code

            # 从响应头读取内容长度；若不存在则尝试读取响应体。
            response_size = int(response.headers.get("content-length", 0))
            if response_size == 0:
                try:
                    response_size = len(getattr(response, "body", b""))
                except Exception:
                    response_size = 0

            # Try to enrich the log with the authenticated identity.
            identity: Any = getattr(request.state, "identity", None)
            identity_name = identity.name if identity else ""
            set_request_context(
                RequestContext(
                    request_id=request_id,
                    method=method,
                    path=path,
                    identity_name=identity_name,
                )
            )

            duration = time.perf_counter() - start
            # metrics 标签使用路由模板（有界基数）；日志保留原始路径便于排查。
            metric_path = route_path
            REQUESTS_TOTAL.labels(protocol="http", endpoint=metric_path, status=str(status_code)).inc()
            REQUEST_DURATION.labels(protocol="http", endpoint=metric_path).observe(duration)
            TRAFFIC_BYTES_TOTAL.labels(protocol="http", endpoint=metric_path, direction="request").inc(request_size)
            TRAFFIC_BYTES_TOTAL.labels(protocol="http", endpoint=metric_path, direction="response").inc(response_size)

            logger.info(
                "%s %s %s %.3fms request=%dB response=%dB identity=%s",
                method,
                path,
                status_code,
                duration * 1000,
                request_size,
                response_size,
                identity_name or "anonymous",
            )

            response.headers[_REQUEST_ID_HEADER] = request_id
            response.headers[_TRACE_ID_HEADER] = request_id
            return response
        except Exception as exc:
            duration = time.perf_counter() - start
            metric_path = _metrics_path(request)
            REQUESTS_TOTAL.labels(protocol="http", endpoint=metric_path, status="500").inc()
            REQUEST_DURATION.labels(protocol="http", endpoint=metric_path).observe(duration)
            TRAFFIC_BYTES_TOTAL.labels(protocol="http", endpoint=metric_path, direction="request").inc(request_size)
            logger.exception(
                "%s %s 500 %.3fms request=%dB error=%s",
                method,
                path,
                duration * 1000,
                request_size,
                exc,
            )
            # Return a unified error envelope with X-Request-ID/X-Trace-ID headers,
            # aligned with Go middleware.ErrorEnvelope format.
            # 返回统一错误信封格式，携带 X-Request-ID/X-Trace-ID 头，
            # 与 Go 端 middleware.ErrorEnvelope 格式对齐。
            from datetime import datetime, timezone
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=500,
                content={
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "detail": "Internal server error",
                    "trace_id": request_id,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                },
                headers={
                    _REQUEST_ID_HEADER: request_id,
                    _TRACE_ID_HEADER: request_id,
                },
            )


def _grpc_status(context: grpc.ServicerContext) -> str:
    """Return a status label for a gRPC call."""
    code = context.code()
    if code is None:
        return "OK"
    return str(code.name)


def _message_size(message: Any) -> int:
    """尝试获取 protobuf 消息的字节大小；不支持时返回 0。"""
    if message is None:
        return 0
    if hasattr(message, "ByteSize"):
        try:
            return int(message.ByteSize())
        except Exception:
            return 0
    return 0


def _grpc_span(method: str, context: grpc.ServicerContext):
    """Go ``StartSpan`` 同口径：span ``grpc FullMethod`` + rpc.* 属性。"""
    metadata = dict(context.invocation_metadata() or [])
    return start_span(
        f"grpc {method}",
        {"rpc.system": "grpc", "rpc.method": method, "rpc.side": "server"},
        context=context_from_carrier(metadata),
    )


def _wrap_unary_unary(
    handler: Callable[[Any, grpc.ServicerContext], Any],
    method: str,
    request_deserializer: Callable[[bytes], Any] | None,
    response_serializer: Callable[[Any], bytes] | None,
) -> grpc.RpcMethodHandler:
    """Wrap a unary-unary handler with observability."""

    def _wrapper(request: Any, context: grpc.ServicerContext) -> Any:
        request_id = _get_request_id_from_metadata(context)
        set_request_context(
            RequestContext(request_id=request_id, method="gRPC", path=method)
        )
        # Inject dual trace headers as initial metadata for end-to-end correlation.
        context.send_initial_metadata([
            (_REQUEST_ID_HEADER, request_id),
            (_TRACE_ID_HEADER, request_id),
        ])
        start = time.perf_counter()
        request_size = _message_size(request)
        with _grpc_span(method, context):
            response = handler(request, context)
        response_size = _message_size(response)
        _observe_grpc(context, method, start, request_size, response_size)
        return response

    return grpc.unary_unary_rpc_method_handler(
        _wrapper,
        request_deserializer=request_deserializer,
        response_serializer=response_serializer,
    )


def _wrap_unary_stream(
    handler: Callable[[Any, grpc.ServicerContext], Any],
    method: str,
    request_deserializer: Callable[[bytes], Any] | None,
    response_serializer: Callable[[Any], bytes] | None,
) -> grpc.RpcMethodHandler:
    """Wrap a unary-stream handler with observability."""

    def _wrapper(request: Any, context: grpc.ServicerContext) -> Any:
        request_id = _get_request_id_from_metadata(context)
        set_request_context(
            RequestContext(request_id=request_id, method="gRPC", path=method)
        )
        start = time.perf_counter()
        request_size = _message_size(request)
        try:
            with _grpc_span(method, context):
                return handler(request, context)
        finally:
            _observe_grpc(context, method, start, request_size, 0)

    return grpc.unary_stream_rpc_method_handler(
        _wrapper,
        request_deserializer=request_deserializer,
        response_serializer=response_serializer,
    )


def _wrap_stream_unary(
    handler: Callable[[Any, grpc.ServicerContext], Any],
    method: str,
    request_deserializer: Callable[[bytes], Any] | None,
    response_serializer: Callable[[Any], bytes] | None,
) -> grpc.RpcMethodHandler:
    """Wrap a stream-unary handler with observability."""

    def _wrapper(request_iterator: Any, context: grpc.ServicerContext) -> Any:
        request_id = _get_request_id_from_metadata(context)
        set_request_context(
            RequestContext(request_id=request_id, method="gRPC", path=method)
        )
        start = time.perf_counter()
        response = None
        try:
            with _grpc_span(method, context):
                response = handler(request_iterator, context)
            return response
        finally:
            response_size = _message_size(response)
            _observe_grpc(context, method, start, 0, response_size)

    return grpc.stream_unary_rpc_method_handler(
        _wrapper,
        request_deserializer=request_deserializer,
        response_serializer=response_serializer,
    )


def _wrap_stream_stream(
    handler: Callable[[Any, grpc.ServicerContext], Any],
    method: str,
    request_deserializer: Callable[[bytes], Any] | None,
    response_serializer: Callable[[Any], bytes] | None,
) -> grpc.RpcMethodHandler:
    """Wrap a stream-stream handler with observability."""

    def _wrapper(request_iterator: Any, context: grpc.ServicerContext) -> Any:
        request_id = _get_request_id_from_metadata(context)
        set_request_context(
            RequestContext(request_id=request_id, method="gRPC", path=method)
        )
        start = time.perf_counter()
        try:
            with _grpc_span(method, context):
                return handler(request_iterator, context)
        finally:
            _observe_grpc(context, method, start, 0, 0)

    return grpc.stream_stream_rpc_method_handler(
        _wrapper,
        request_deserializer=request_deserializer,
        response_serializer=response_serializer,
    )


def _observe_grpc(
    context: grpc.ServicerContext,
    method: str,
    start: float,
    request_size: int = 0,
    response_size: int = 0,
) -> None:
    """Record gRPC access log and metrics."""
    duration = time.perf_counter() - start
    status = _grpc_status(context)
    REQUESTS_TOTAL.labels(protocol="grpc", endpoint=method, status=status).inc()
    REQUEST_DURATION.labels(protocol="grpc", endpoint=method).observe(duration)
    TRAFFIC_BYTES_TOTAL.labels(protocol="grpc", endpoint=method, direction="request").inc(request_size)
    TRAFFIC_BYTES_TOTAL.labels(protocol="grpc", endpoint=method, direction="response").inc(response_size)
    logger.info(
        "gRPC %s %s %.3fms request=%dB response=%dB",
        method,
        status,
        duration * 1000,
        request_size,
        response_size,
    )


class GrpcObservabilityInterceptor(grpc.ServerInterceptor):
    """gRPC server interceptor for request context, logs, and metrics."""

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        handler = continuation(handler_call_details)
        if handler is None:
            return handler  # type: ignore[return-value]

        method = handler_call_details.method
        kwargs = {
            "request_deserializer": handler.request_deserializer,
            "response_serializer": handler.response_serializer,
        }
        if not handler.request_streaming and not handler.response_streaming:
            return _wrap_unary_unary(handler.unary_unary, method, **kwargs)
        if not handler.request_streaming and handler.response_streaming:
            return _wrap_unary_stream(handler.unary_stream, method, **kwargs)
        if handler.request_streaming and not handler.response_streaming:
            return _wrap_stream_unary(handler.stream_unary, method, **kwargs)
        return _wrap_stream_stream(handler.stream_stream, method, **kwargs)


def record_auth_denial(reason: str) -> None:
    """Record an authentication/authorization/rate-limit denial.

    Increments both the Python-native denial counter and the Go-parity security
    counters (``privshield_auth_failures_total`` / ``privshield_auth_forbidden_total``).
    Reasons ``missing_token`` / ``bare_token`` / ``invalid_token`` are authentication
    failures; ``forbidden`` is an authorization failure; other reasons (e.g. rate
    limit) only feed the Python-native counter.
    """
    from .metrics import AUTH_DENIALS_TOTAL, AUTH_FAILURES_TOTAL, AUTH_FORBIDDEN_TOTAL

    AUTH_DENIALS_TOTAL.labels(reason=reason).inc()
    if reason in ("missing_token", "bare_token", "invalid_token", "unauthenticated"):
        AUTH_FAILURES_TOTAL.labels(reason=reason).inc()
    elif reason == "forbidden":
        AUTH_FORBIDDEN_TOTAL.inc()
    logger.warning("Auth denial: reason=%s", reason)


def record_auth_allowed(subject: str, permission: str) -> None:
    """Record a permission grant for a workload principal (Go parity)."""
    from .metrics import AUTH_ALLOWED_TOTAL

    AUTH_ALLOWED_TOTAL.labels(subject=subject, permission=permission).inc()
