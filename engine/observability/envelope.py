"""Unified API error envelope for cross-language consistency.

所有 Go 微服务与 Python FastAPI 引擎共享同一错误响应格式：

{
    "code":      "INVALID_ARGUMENT",
    "message":   "请求参数校验失败",
    "detail":    "...",
    "trace_id":  "req-1787554500-abc123",
    "timestamp": "2026-08-27T09:30:00.123Z"
}

迁移过渡期双轨兼容：响应体同时包含 code / message / detail，
响应头强制下发 X-Request-ID 与 X-Trace-ID。

Usage::

    from engine.observability.envelope import register_envelope_exception_handlers
    register_envelope_exception_handlers(app)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# HTTP 422 constant: use int literal to avoid StarletteDeprecationWarning
# HTTP 422 常量：使用整数字面量以避免 Starlette 弃用警告
_HTTP_422 = 422

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

# HTTP status code → machine-readable error code mapping.
# Aligned with Go middleware.ErrorCodeFromStatus() in pkg/middleware/envelope.go.
# HTTP 状态码 → 机器可读错误码映射，与 Go 端 ErrorCodeFromStatus() 对齐。
_STATUS_CODE_MAP: dict[int, str] = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "INVALID_ARGUMENT",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    503: "UPSTREAM_UNAVAILABLE",
}


def _get_trace_id(request: Request) -> str:
    """Extract trace ID from request headers or state.

    优先从请求头获取（上游网关/BFF 已注入），其次从 request.state 获取
    （ObservabilityMiddleware 已设置），最后自动生成。
    """
    # 1. From header (set by upstream gateway/BFF) / 从请求头获取
    trace_id = request.headers.get("X-Request-ID", "")
    if trace_id:
        return trace_id
    # 2. From request state (set by ObservabilityMiddleware) / 从请求状态获取
    if hasattr(request.state, "trace_id"):
        return str(request.state.trace_id)
    if hasattr(request.state, "request_id"):
        return str(request.state.request_id)
    # 3. Generate fallback / 自动生成
    return f"req-{int(time.time())}"


def _build_error_envelope(
    code: str,
    message: str,
    detail: Any,
    request: Request,
    status_code: int,
) -> JSONResponse:
    """Build a JSONResponse with the unified error envelope format.

    构建遵循统一错误信封格式的 JSONResponse，同时下发 X-Request-ID 与 X-Trace-ID 头。
    """
    trace_id = _get_trace_id(request)
    return JSONResponse(
        status_code=status_code,
        headers={
            "X-Request-ID": trace_id,
            "X-Trace-ID": trace_id,
        },
        content={
            "code": code,
            "message": message,
            "detail": detail,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        },
    )


def _build_success_envelope(
    message: str,
    data: Any,
    request: Request,
    status_code: int = 200,
) -> JSONResponse:
    """Build a success response in the Go ``respondSuccess`` envelope shape.

    成功信封：``{code:"OK", message, data, trace_id, timestamp}``，并下发
    ``X-Request-ID`` / ``X-Trace-ID``，与 Go 端逐字段一致。
    """
    trace_id = _get_trace_id(request)
    return JSONResponse(
        status_code=status_code,
        headers={
            "X-Request-ID": trace_id,
            "X-Trace-ID": trace_id,
        },
        content={
            "code": "OK",
            "message": message,
            "data": data,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        },
    )


def _make_validation_handler(debug: bool = False):
    """Create the RequestValidationError handler with envelope format.

    创建遵循统一信封格式的请求校验异常处理器。
    debug=True 时返回详细校验错误（仅限开发环境），否则返回通用消息。
    """

    async def handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        if debug:
            detail = exc.errors()
        else:
            detail = "Invalid request: one or more fields failed validation"
        return _build_error_envelope(
            code="INVALID_ARGUMENT",
            message="请求参数校验失败",
            detail=detail,
            request=request,
            status_code=_HTTP_422,
        )

    return handler


def _make_http_exception_handler():
    """Create the HTTPException handler with envelope format.

    创建遵循统一信封格式的 HTTP 异常处理器。
    将 FastAPI/Starlette HTTPException 映射为统一错误码。
    """

    async def handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        err_code = _STATUS_CODE_MAP.get(exc.status_code, "UNKNOWN_ERROR")
        message = str(exc.detail)
        return _build_error_envelope(
            code=err_code,
            message=message,
            detail=message,
            request=request,
            status_code=exc.status_code,
        )

    return handler


def register_envelope_exception_handlers(
    app: FastAPI,
    *,
    debug_validation: bool = False,
) -> None:
    """Register unified envelope exception handlers on a FastAPI app.

    在 FastAPI 应用上注册统一信封格式的异常处理器。
    替换 main.py 中的内联异常处理器，确保所有错误响应遵循统一格式。

    Args:
        app: FastAPI application instance / FastAPI 应用实例。
        debug_validation: Return detailed validation errors (dev only)
            是否返回详细校验错误（仅开发环境使用）。
            由 PRIVACY_DEBUG_VALIDATION 环境变量控制。
    """
    app.add_exception_handler(
        RequestValidationError,
        _make_validation_handler(debug=debug_validation),
    )
    app.add_exception_handler(
        StarletteHTTPException,
        _make_http_exception_handler(),
    )
