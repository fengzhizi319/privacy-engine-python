"""Security layer for engine.

Provides TLS, authentication/authorization, and rate limiting shared between the
REST (FastAPI) and gRPC servers.
"""

from .auth import (
    AuthInterceptor,
    get_current_identity,
    get_identity_from_grpc_context,
    require_permission,
    require_rest_path_permission,
)
from .config import SecuritySettings, get_security_settings, settings
from .identity import Identity, permission_for_grpc_method, permission_for_rest_path
from .ratelimit import (
    Limiter,
    RateLimitInterceptor,
    get_limiter,
    rate_limit_dependency,
    rate_limit_for_path,
)
from .tls import grpc_server_credentials, uvicorn_ssl_kwargs
from .whitelist import CNEntry, WhitelistConfig, WhitelistManager, get_whitelist_manager, reset_whitelist_manager

__all__ = [
    "AuthInterceptor",
    "CNEntry",
    "Identity",
    "Limiter",
    "RateLimitInterceptor",
    "SecuritySettings",
    "WhitelistConfig",
    "WhitelistManager",
    "get_current_identity",
    "get_identity_from_grpc_context",
    "get_limiter",
    "get_security_settings",
    "get_whitelist_manager",
    "grpc_server_credentials",
    "permission_for_grpc_method",
    "permission_for_rest_path",
    "rate_limit_dependency",
    "rate_limit_for_path",
    "require_permission",
    "require_rest_path_permission",
    "reset_whitelist_manager",
    "settings",
    "uvicorn_ssl_kwargs",
]
