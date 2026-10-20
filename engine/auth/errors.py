"""Auth error taxonomy mapped to the shared REST/gRPC error envelopes."""

from __future__ import annotations

from typing import Any

import grpc


class AuthError(Exception):
    """Raised by the auth kernel; carries the HTTP status and machine code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        detail: Any = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail
        self.headers = headers or {}

    def grpc_status(self) -> grpc.StatusCode:
        return {
            400: grpc.StatusCode.INVALID_ARGUMENT,
            401: grpc.StatusCode.UNAUTHENTICATED,
            403: grpc.StatusCode.PERMISSION_DENIED,
            404: grpc.StatusCode.NOT_FOUND,
            409: grpc.StatusCode.ALREADY_EXISTS,
            429: grpc.StatusCode.RESOURCE_EXHAUSTED,
            503: grpc.StatusCode.UNAVAILABLE,
        }.get(self.status_code, grpc.StatusCode.INTERNAL)


def invalid_argument(message: str, detail: Any = None) -> AuthError:
    return AuthError(400, "INVALID_ARGUMENT", message, detail)


def forbidden(message: str, code: str = "FORBIDDEN", detail: Any = None) -> AuthError:
    return AuthError(403, code, message, detail)


def unauthenticated(message: str, code: str = "UNAUTHENTICATED", detail: Any = None) -> AuthError:
    return AuthError(401, code, message, detail)


def not_found(message: str, code: str = "USER_NOT_FOUND") -> AuthError:
    return AuthError(404, code, message)


def conflict(message: str, code: str) -> AuthError:
    return AuthError(409, code, message)


def rate_limited(message: str, retry_after: int = 60) -> AuthError:
    return AuthError(
        429,
        "RATE_LIMITED",
        message,
        {"retry_after_seconds": retry_after},
        headers={"Retry-After": str(retry_after)},
    )


def internal(message: str = "internal error") -> AuthError:
    return AuthError(500, "INTERNAL_ERROR", message)
