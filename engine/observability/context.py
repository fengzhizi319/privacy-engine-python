"""Request context propagated via contextvars.

通过 contextvars 在异步/线程上下文中透传请求级元数据，供日志、metrics 与 tracing 使用。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    """Per-request metadata used across logs, metrics and traces.

    Attributes:
        request_id: Unique correlation id for the request/RPC.
        method: HTTP method or gRPC method name.
        path: REST path or gRPC full method name.
        identity_name: Authenticated caller name, if available.
    """

    request_id: str
    method: str
    path: str
    identity_name: str = ""


# Module-level ContextVar. Each request/RPC sets its own value; values are
# automatically scoped to the current task/thread.
_request_context: ContextVar[RequestContext | None] = ContextVar(
    "request_context", default=None
)


def set_request_context(ctx: RequestContext) -> None:
    """Set the current request context."""
    _request_context.set(ctx)


def get_request_context() -> RequestContext | None:
    """Return the current request context, or None if outside a request."""
    return _request_context.get(None)
