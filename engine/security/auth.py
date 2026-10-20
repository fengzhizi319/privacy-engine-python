"""Authentication and authorization for REST and gRPC.

与 Go ``pkg/auth`` 共享同一套可观测行为：
- 静态 API Key + 用户库动态凭据（登录会话 / 动态 Key / 交换票据）热解析；
- 公开认证路径未携凭证时注入 ``public-caller``，携凭证时按正常身份校验；
- 健康探针仅在 GET/HEAD 且命中标准路径时豁免，身份只授予 ``health:read``；
- 受众 / 契约版本 / 契约路径 / drain-only 的凭据策略判定；
- 首登改密、口令超期与 TOTP 强制沙箱；
- 接口级权限校验（REST dependency 与 gRPC interceptor 同口径）。

Provides FastAPI dependencies and a gRPC server interceptor supporting static API
keys, user-store sessions/dynamic keys, mTLS certificate identity extraction,
credential policy enforcement, sandboxes, and per-method permission checks.
"""

from __future__ import annotations

import hmac
import time
from typing import TYPE_CHECKING, Any

import grpc
from fastapi import Depends, HTTPException, Request

from ..auth.contract import contract_allows
from ..auth.errors import AuthError
from ..observability.logging_config import get_logger
from ..observability.middleware import record_auth_allowed, record_auth_denial
from ..observability.metrics import AUTH_DURATION
from .config import SecuritySettings, get_security_settings, metrics_auth_mode
from .identity import (
    ANONYMOUS_IDENTITY,
    DEV_MODE_IDENTITY,
    Identity,
    is_auth_public_path,
    is_grpc_health_method,
    is_standard_health_path,
    permission_for_grpc_method,
    permission_for_rest_path,
)
from .pop import decide_pop
from .terminator import (
    IDENTITY_HEADERS,
    TerminatorIdentityError,
    TerminatorPeer,
    build_verifier,
    client_host_from_peer,
    client_host_from_scope,
)
from .whitelist import CNEntry, get_whitelist_manager

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

# Go ``ScopeMetricsObservability``: /metrics requires the same scope as /v1/ops/*.
METRICS_SCOPE = "ops:diagnostics"

_HEALTH_IDENTITY = Identity("internal", "health-probe", ["health:read"])
_PUBLIC_IDENTITY = Identity("public", "public-caller", ["auth:public"])

_MISSING_TOKEN = "missing_token"
_BARE_TOKEN = "bare_token"


def _extract_bearer_token(header_value: str | None) -> str | None:
    """Extract the bearer token from an Authorization header value."""
    if not header_value:
        return None
    parts = header_value.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def _parse_bearer(header_value: str | None) -> tuple[str, str]:
    """Go ``ParseBearerCredential``: (token, deny_reason).

    Empty header → missing_token; non-empty but not ``Bearer <token>`` →
    bare_token (never tolerated). A valid credential returns an empty reason.
    """
    raw = (header_value or "").strip()
    if not raw:
        return "", _MISSING_TOKEN
    token = _extract_bearer_token(raw)
    if token:
        return token, ""
    return "", _BARE_TOKEN


def _constant_time_lookup(keys: dict[str, Any], token: str) -> Any | None:
    """Constant-time lookup of ``token`` among ``keys`` to mitigate timing attacks.

    A plain ``dict.get`` is a hash-based probe whose timing can correlate with the
    stored keys, theoretically leaking information about key existence/prefixes.
    Here keys are sorted for deterministic iteration order, and every stored key is
    compared with :func:`hmac.compare_digest` without short-circuiting.
    """
    if not keys or not token:
        return None
    token_bytes = token.encode("utf-8")
    matched = None
    for key in sorted(keys.keys()):
        if hmac.compare_digest(key.encode("utf-8"), token_bytes):
            matched = keys[key]
    return matched


def _authenticate_api_key(settings: SecuritySettings, token: str) -> Identity | None:
    """Look up an API key in internal and external key stores.

    Internal keys are checked first so an internal token can never be shadowed by an
    external one.
    """
    internal = _constant_time_lookup(settings.internal_keys, token)
    if internal:
        return Identity("internal", internal.name, list(internal.scopes))
    external = _constant_time_lookup(settings.external_keys, token)
    if external:
        return Identity("external", external.name, list(external.scopes))
    return None


def _auth_runtime() -> Any | None:
    """Return the process-wide IAM runtime when a user store is configured."""
    from ..auth.runtime import get_auth_runtime

    return get_auth_runtime()


def _resolve_token(settings: SecuritySettings, token: str) -> Identity | None:
    """Resolve a Bearer credential: sessions → dynamic keys → exchanged tokens → static.

    Go 的解析顺序为集群会话 → 活密钥 → 用户库摘要 Key → Break-Glass → 静态密钥；
    Python 无集群校验器，动态层由运行时覆盖，最后回落到静态内部/外部 Key。
    """
    runtime = _auth_runtime()
    if runtime is not None:
        identity = runtime.service.resolve_token(token)
        if identity is not None:
            return identity
    return _authenticate_api_key(settings, token)


def _service_audience() -> str:
    try:
        from ..auth.config import AuthSettings

        return AuthSettings.from_env().service_audience
    except Exception:  # noqa: BLE001 - config failure is surfaced on first auth use
        return "privacy-engine"


def _terminator_peer_from_values(
    settings: SecuritySettings,
    *,
    method: str,
    path: str,
    values: dict[str, list[str]],
    client_host: str | None,
) -> TerminatorPeer | None:
    """校验并返回终结器注入的对端证书视图（未启用时返回 None）。"""
    verifier = build_verifier(settings)
    if verifier is None:
        return None
    try:
        return verifier.verify(
            method=method, path=path, values=values, client_host=client_host
        )
    except TerminatorIdentityError as exc:
        logger.warning(
            "terminator identity rejected",
            extra={"path": path, "error": str(exc)},
        )
        raise AuthError(
            401, "UNAUTHENTICATED", "Unauthorized: invalid terminator identity"
        ) from exc


def _rest_identity_values(request: Request) -> dict[str, list[str]]:
    return {name: list(request.headers.getlist(name)) for name in IDENTITY_HEADERS}


def _grpc_identity_values(context: grpc.ServicerContext) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    try:
        metadata = context.invocation_metadata() or ()
    except Exception:  # noqa: BLE001 - test doubles may not expose metadata
        return values
    for item in metadata:
        if hasattr(item, "key"):
            key, value = item.key, item.value
        else:
            key, value = item[0], item[1]
        values.setdefault(str(key).lower(), []).append(str(value))
    return values


def _grpc_terminator_peer(
    settings: SecuritySettings, context: grpc.ServicerContext, method: str
) -> TerminatorPeer | None:
    if not settings.terminator_identity_enabled:
        return None
    try:
        peer_addr = context.peer()
    except Exception:  # noqa: BLE001 - test doubles may not expose peer()
        peer_addr = None
    return _terminator_peer_from_values(
        settings,
        method="POST",
        path=method,
        values=_grpc_identity_values(context),
        client_host=client_host_from_peer(peer_addr),
    )


def _enforce_credential_policy(
    identity: Identity,
    *,
    method: str,
    path: str,
    contract_header: str,
    peer: TerminatorPeer | None = None,
) -> None:
    """Mirror Go ``CheckCredentialPolicy`` (audience → contract → drain-only → PoP → paths).

    对端证书视图来自 Go 终结器注入的签名头；PoP 判定与 Go ``DecidePoP`` 同构，
    机器凭据无绑定档位、无部署开关。
    """
    health_or_public = (
        is_standard_health_path(path)
        or is_auth_public_path(path)
        or is_grpc_health_method(path)
    )

    audience = _service_audience()
    if audience and audience != "PrivShield" and (
        identity.allowed_audiences or identity.audience
    ):
        if not identity.matches_audience(audience):
            raise AuthError(
                403,
                "AUDIENCE_MISMATCH",
                f"Forbidden: credential audience mismatch, cannot access service {audience}",
            )

    header_version = (contract_header or "").strip()
    if header_version and (
        not identity.contract_version or header_version != identity.contract_version
    ):
        raise AuthError(
            403,
            "CONTRACT_VIOLATION",
            "Forbidden: contract version mismatch",
        )

    if identity.binding_status == "drain_only" and not health_or_public:
        raise AuthError(
            403,
            "CREDENTIAL_DRAIN_ONLY",
            "Forbidden: credential in drain-only mode; only health probes are permitted",
        )

    outcome = decide_pop(identity, peer, health_exempt=health_or_public)
    if not outcome.allowed:
        raise AuthError(outcome.status, outcome.code, outcome.message)

    if identity.contract_allowed_paths and not health_or_public:
        if not contract_allows(identity.contract_allowed_paths, method, path):
            raise AuthError(
                403,
                "CONTRACT_VIOLATION",
                "Forbidden: request path is not in the credential's contract allow-list",
            )


def _enforce_sandboxes(identity: Identity, path: str) -> None:
    """Apply password / expiry / MFA sandboxes for session identities."""
    if not identity.subject:
        return
    if is_standard_health_path(path) or is_grpc_health_method(path):
        return
    runtime = _auth_runtime()
    if runtime is not None:
        runtime.service.enforce_sandboxes(identity, path)


def _identity_from_whitelist(peer: TerminatorPeer, method: str) -> Identity:
    """Map a verified terminator identity onto an internal Identity via the whitelist."""
    manager = get_whitelist_manager()
    entry, reason = manager.authorize(peer.common_name, peer.cert_sha256, method)
    if entry is None:
        logger.warning(
            "mTLS client certificate rejected: not authorized",
            extra={"cn": peer.common_name, "reason": reason, "method": method},
        )
        raise AuthError(
            403, "PERMISSION_DENIED", "Forbidden: client certificate is not authorized"
        )
    return _identity_from_entry(entry, peer.common_name, method)


def _identity_from_entry(entry: CNEntry, name: str, method: str) -> Identity:
    """Go ``allowed_scopes`` are method patterns; legacy ``scopes`` are RBAC grants."""
    scopes = list(entry.scopes)
    if entry.uses_method_scopes:
        # 方法已由白名单匹配；身份只拿本次调用对应的 RBAC 权限，
        # 不能因为方法级条目而获得通配身份。
        scopes = [permission_for_grpc_method(method)]
    return Identity("internal", name, scopes)


def _authenticate_mtls(
    settings: SecuritySettings, auth_context: dict[str, Any], method: str = ""
) -> Identity | None:
    """Derive an internal Identity from a verified mTLS client certificate.

    The gRPC auth_context is populated only when the connection uses TLS and the
    client presented a certificate. A certificate passing CA verification only
    proves possession of *a* certificate, not authorization to access this
    service — so mTLS auth is disabled by default and, when enabled via
    ``PRIVACY_AUTH_INTERNAL_MTLS_ENABLED``, the certificate Common Name must
    additionally match the whitelist.

    Whitelist lookup uses the :class:`WhitelistManager` singleton which supports:
    - YAML config file with per-CN scopes (``PRIVACY_AUTH_MTLS_WHITELIST_FILE``)
    - Go ``clients:`` schema (``allowed_scopes`` + ``cert_fingerprints``)
    - Fallback to static CN list (``PRIVACY_AUTH_MTLS_ALLOWED_CNS``) with ["*"] scope
    - Hot-reload when the config file changes

    An empty whitelist rejects every certificate (fail-closed). Direct mTLS
    cannot present the leaf fingerprint to ``auth_context()``; pinned entries
    therefore reject direct connections (the terminator path has the fingerprint).
    """
    if not settings.auth_internal_mtls_enabled:
        return None
    transport = auth_context.get("transport_security_type", [b""])[0]
    if transport != b"ssl":
        return None
    cn_bytes = auth_context.get("x509_common_name", [b""])[0]
    if not cn_bytes:
        return None
    cn = cn_bytes.decode("utf-8", errors="replace")

    manager = get_whitelist_manager()
    entry, reason = manager.authorize(cn, "", method)
    if entry is None:
        if reason == "cert_fingerprint_missing":
            # 直连 gRPC 的 auth_context 不暴露叶证书，钉扎无法核验 —— fail-closed。
            logger.warning(
                "mTLS client certificate rejected: pinned entry requires terminator identity",
                extra={"cn": cn, "reason": reason},
            )
            raise AuthError(
                403,
                "PERMISSION_DENIED",
                "Forbidden: pinned client certificate requires the terminator identity path",
            )
        logger.warning(
            "mTLS client certificate rejected: CN not in whitelist",
            extra={"cn": cn, "reason": reason},
        )
        return None
    return _identity_from_entry(entry, cn, method)


def _extract_identity_from_grpc_context(
    settings: SecuritySettings,
    context: grpc.ServicerContext,
    method: str,
    peer: TerminatorPeer | None = None,
) -> Identity | None:
    """Extract identity from gRPC metadata and/or mTLS auth context.

    Order: direct mTLS → verified terminator identity → health → Bearer.
    A verified terminator peer is only used as an identity when
    ``auth_internal_mtls_enabled`` is on; otherwise it still feeds PoP.
    """
    # First try direct mTLS because a verified certificate is stronger than a bearer token.
    try:
        auth_context = context.auth_context()
    except Exception:  # noqa: BLE001 - test doubles may not expose auth_context
        auth_context = None
    if auth_context:
        identity = _authenticate_mtls(settings, auth_context, method)
        if identity:
            return identity

    if peer is not None and settings.auth_internal_mtls_enabled:
        return _identity_from_whitelist(peer, method)

    if is_grpc_health_method(method) and settings.health_no_auth:
        return _HEALTH_IDENTITY

    metadata = dict(context.invocation_metadata() or ())
    token, deny_reason = _parse_bearer(metadata.get("authorization", ""))
    if deny_reason:
        return None
    return _resolve_token(settings, token)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_identity(request: Request) -> Identity:
    """FastAPI dependency resolving the caller identity (Go ``AuthMiddleware``).

    Order:
    1. auth disabled → dev identity (all routes reachable);
    2. standard health path on GET/HEAD → ``health-probe`` with ``health:read``;
    3. public auth path without credentials → ``public-caller`` with ``auth:public``;
    4. Bearer token → sessions / dynamic keys / exchanged tokens / static keys;
    5. credential policy (audience / contract / drain-only / allow-list);
    6. public paths bypass sandboxes and scope checks;
    7. password / expiry / MFA sandboxes for the remaining paths.
    """
    start = time.perf_counter()
    settings = get_security_settings()
    path = request.url.path
    normalized = path.rstrip("/") if len(path) > 1 else path

    if not settings.auth_enabled:
        request.state.identity = DEV_MODE_IDENTITY
        AUTH_DURATION.labels(result="disabled").observe(time.perf_counter() - start)
        return DEV_MODE_IDENTITY

    try:
        peer = _terminator_peer_from_values(
            settings,
            method=request.method,
            path=path,
            values=_rest_identity_values(request),
            client_host=request.client.host if request.client else None,
        )
    except AuthError:
        record_auth_denial("invalid_token")
        AUTH_DURATION.labels(result="denied").observe(time.perf_counter() - start)
        raise
    request.state.terminator_peer = peer

    if (
        request.method in ("GET", "HEAD")
        and is_standard_health_path(normalized)
        and settings.health_no_auth
    ):
        request.state.identity = _HEALTH_IDENTITY
        AUTH_DURATION.labels(result="exempt").observe(time.perf_counter() - start)
        return _HEALTH_IDENTITY

    token, deny_reason = _parse_bearer(request.headers.get("authorization"))

    if is_auth_public_path(normalized) and deny_reason == _MISSING_TOKEN:
        request.state.identity = _PUBLIC_IDENTITY
        AUTH_DURATION.labels(result="public").observe(time.perf_counter() - start)
        return _PUBLIC_IDENTITY

    if deny_reason:
        record_auth_denial(deny_reason)
        AUTH_DURATION.labels(result="denied").observe(time.perf_counter() - start)
        logger.warning(
            "Authentication failed: missing credentials",
            extra={"path": path, "reason": deny_reason},
        )
        raise AuthError(401, "UNAUTHENTICATED", "Unauthorized: missing credentials")

    identity = _resolve_token(settings, token)
    if identity is None:
        record_auth_denial("invalid_token")
        AUTH_DURATION.labels(result="denied").observe(time.perf_counter() - start)
        logger.warning(
            "Authentication failed: invalid credentials",
            extra={"path": path, "reason": "invalid_token"},
        )
        raise AuthError(401, "UNAUTHENTICATED", "Unauthorized: invalid credentials")

    try:
        _enforce_credential_policy(
            identity,
            method=request.method,
            path=normalized,
            contract_header=request.headers.get("x-privshield-contract-version", ""),
            peer=peer,
        )
        if not is_auth_public_path(normalized):
            _enforce_sandboxes(identity, normalized)
    except AuthError as exc:
        record_auth_denial("unauthenticated" if exc.status_code == 401 else "forbidden")
        AUTH_DURATION.labels(result="denied").observe(time.perf_counter() - start)
        logger.warning(
            "Authorization failed",
            extra={
                "path": path,
                "identity_name": identity.name,
                "code": exc.code,
            },
        )
        raise

    # Stash identity on request.state so rate limiting can reuse it without
    # re-authenticating.
    request.state.identity = identity
    AUTH_DURATION.labels(result="success").observe(time.perf_counter() - start)
    logger.debug(
        "Authentication successful",
        extra={"path": path, "identity_type": identity.service_type, "identity_name": identity.name},
    )
    return identity


def require_permission(permission: str) -> Any:
    """Return a FastAPI dependency that enforces a specific permission.

    Usage:
        @app.post("/v1/privacy/mask", dependencies=[require_permission("privacy:mask")])
    """

    async def _checker(identity: Identity = Depends(get_current_identity)) -> None:
        if not identity.has_permission(permission):
            record_auth_denial("forbidden")
            logger.warning(
                "Authorization failed: insufficient scope",
                extra={"required_permission": permission, "identity_name": identity.name},
            )
            raise HTTPException(status_code=403, detail="Forbidden: insufficient scope")
        record_auth_allowed(identity.name, permission)

    return Depends(_checker)


def require_any_permission(*permissions: str) -> Any:
    """Return a FastAPI dependency that enforces any one of the specified permissions."""

    async def _checker(identity: Identity = Depends(get_current_identity)) -> None:
        matched = next((p for p in permissions if identity.has_permission(p)), None)
        if matched is None:
            record_auth_denial("forbidden")
            logger.warning(
                "Authorization failed: insufficient scope",
                extra={"required_permissions": permissions, "identity_name": identity.name},
            )
            raise HTTPException(status_code=403, detail="Forbidden: insufficient scope")
        record_auth_allowed(identity.name, matched)

    return Depends(_checker)


def require_rest_path_permission(path: str) -> Any:
    """Convenience wrapper that enforces the permission for a REST path.

    .. note::
        **参考实现，当前未接线**：各路由显式使用 ``require_permission(...)``
        声明权限，本便捷函数依赖 ``permission_for_rest_path`` 的参考映射，
        仅供文档/示例参考。
    """
    return require_permission(permission_for_rest_path(path))


# ---------------------------------------------------------------------------
# gRPC interceptor
# ---------------------------------------------------------------------------

class AuthInterceptor(grpc.ServerInterceptor):
    """gRPC server interceptor enforcing authentication and authorization."""

    def __init__(self, settings: SecuritySettings | None = None):
        self._settings = settings or get_security_settings()

    def _check(self, context: grpc.ServicerContext, method: str) -> Identity:
        """Authenticate and authorize the current gRPC call (Go ``authUnaryInterceptor``)."""
        if not self._settings.auth_enabled:
            return ANONYMOUS_IDENTITY

        try:
            peer = _grpc_terminator_peer(self._settings, context, method)
        except AuthError as exc:
            record_auth_denial("unauthenticated")
            context.abort(exc.grpc_status(), f"{exc.code}: {exc.message}")
            raise  # pragma: no cover - abort raises

        if is_grpc_health_method(method) and self._settings.health_no_auth:
            return _HEALTH_IDENTITY

        try:
            identity = _extract_identity_from_grpc_context(
                self._settings, context, method, peer
            )
            if identity is None:
                record_auth_denial("unauthenticated")
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid credentials")

            metadata = dict(context.invocation_metadata() or ())
            _enforce_credential_policy(
                identity,
                method="GRPC",
                path=method,
                contract_header=metadata.get("x-privshield-contract-version", ""),
                peer=peer,
            )
            if not is_grpc_health_method(method):
                _enforce_sandboxes(identity, method)
        except AuthError as exc:
            record_auth_denial("forbidden")
            context.abort(exc.grpc_status(), f"{exc.code}: {exc.message}")
            raise  # pragma: no cover - abort raises

        permission = permission_for_grpc_method(method)
        if permission:
            if not identity.has_permission(permission):
                record_auth_denial("forbidden")
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "insufficient scope")
            record_auth_allowed(identity.name, permission)
        return identity

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

        def _wrap(handler_fn: Callable[..., Any]) -> Callable[..., Any]:
            def _wrapper(request_or_iterator: Any, context: grpc.ServicerContext) -> Any:
                self._check(context, method)
                return handler_fn(request_or_iterator, context)

            return _wrapper

        if not handler.request_streaming and not handler.response_streaming:
            return grpc.unary_unary_rpc_method_handler(
                _wrap(handler.unary_unary), **kwargs
            )
        if not handler.request_streaming and handler.response_streaming:
            return grpc.unary_stream_rpc_method_handler(
                _wrap(handler.unary_stream), **kwargs
            )
        if handler.request_streaming and not handler.response_streaming:
            return grpc.stream_unary_rpc_method_handler(
                _wrap(handler.stream_unary), **kwargs
            )
        return grpc.stream_stream_rpc_method_handler(
            _wrap(handler.stream_stream), **kwargs
        )


# Helper used by the rate-limit interceptor to avoid duplicating auth extraction.
def get_identity_from_grpc_context(
    context: grpc.ServicerContext, method: str
) -> Identity:
    """Extract identity from a gRPC context, falling back to anonymous if auth off."""
    settings = get_security_settings()
    if not settings.auth_enabled:
        return ANONYMOUS_IDENTITY
    try:
        peer = _grpc_terminator_peer(settings, context, method)
    except AuthError as exc:
        context.abort(exc.grpc_status(), f"{exc.code}: {exc.message}")
        raise  # pragma: no cover - abort raises
    if is_grpc_health_method(method) and settings.health_no_auth:
        return _HEALTH_IDENTITY
    try:
        identity = _extract_identity_from_grpc_context(settings, context, method, peer)
    except AuthError as exc:
        context.abort(exc.grpc_status(), f"{exc.code}: {exc.message}")
        raise  # pragma: no cover - abort raises
    if identity is None:
        # The auth interceptor would have already rejected; this fallback avoids
        # leaking anonymous rate-limit budget if called in isolation.
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing or invalid credentials")
    assert identity is not None
    return identity


# ---------------------------------------------------------------------------
# ASGI API-key middleware (for mounted sub-apps such as /metrics)
# ---------------------------------------------------------------------------

class ApiKeyAuthAsgiMiddleware:
    """Pure-ASGI middleware enforcing API-key auth on a mounted sub-application.

    ``app.mount()`` 挂载的子应用（如 Prometheus ``/metrics``）绕过 FastAPI 路由
    依赖体系，``get_current_identity`` 不会执行。本中间件在 ASGI 层直接校验
    ``Authorization: Bearer <key>``，复用与 REST 相同的解析与常量时间比对逻辑，
    因此用户库动态 Key / 会话 / 交换票据与静态 Key 在 ``/metrics`` 上同样生效。

    When auth is enabled, the middleware also enforces that the authenticated
    identity holds the ``ops:diagnostics`` scope (or wildcard ``*``), preventing
    low-privilege API keys from accessing operational metrics.

    Behaviour:
    - 配置 ``PRIVACY_METRICS_API_KEY`` / ``_PREVIOUS`` 时接受专属指标 Key
      （常量时间比对，独立于业务静态 Key），配置后匿名抓取一律拒绝；
    - ``PRIVACY_AUTH_ENABLED`` 关闭且未配置专属 Key 时，仅开发姿态透传；
      生产姿态无任何准入口径时返回 401（fail-closed），不再静默匿名暴露；
    - 开启认证时缺失/无效 key 返回 401 并计入 ``privacy_auth_denials_total``；
    - 开启认证时合法 key 但缺少 ``ops:diagnostics`` scope 返回 403。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        settings = get_security_settings()
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        identity_values: dict[str, list[str]] = {}
        for name, value in scope.get("headers", []):
            key = name.decode("latin-1").lower()
            if key in IDENTITY_HEADERS:
                identity_values.setdefault(key, []).append(value.decode("latin-1"))

        peer: TerminatorPeer | None = None
        if settings.terminator_identity_enabled:
            try:
                peer = _terminator_peer_from_values(
                    settings,
                    method=str(scope.get("method", "GET")),
                    path=str(scope.get("path", "")),
                    values=identity_values,
                    client_host=client_host_from_scope(scope),
                )
            except AuthError:
                record_auth_denial("unauthenticated")
                await self._respond(send, 401, b"Unauthorized: invalid terminator identity")
                return

        token, deny_reason = _parse_bearer(headers.get("authorization"))
        mode, _strategies = metrics_auth_mode(settings)

        # 1. 开发态免密（与 Go TryMetricsAccess 同序）。
        if mode == "dev_anonymous":
            await self.app(scope, receive, send)
            return

        # 2. 专属指标 Key 优先：与静态/动态业务 Key 完全独立，仅放行 /metrics 抓取。
        if token and any(
            hmac.compare_digest(token.encode("utf-8"), key.encode("utf-8"))
            for key in settings.effective_metrics_keys()
        ):
            await self.app(scope, receive, send)
            return

        # 3. mTLS 传输层准入：对端证书已由 Go 终结器验证并签名注入。
        if settings.metrics_allow_mtls and peer is not None:
            allowed_cns = settings.metrics_allowed_client_cns
            if not allowed_cns or any(
                hmac.compare_digest(
                    peer.common_name.strip().lower().encode("utf-8"),
                    cn.strip().lower().encode("utf-8"),
                )
                for cn in allowed_cns
                if cn.strip()
            ):
                await self.app(scope, receive, send)
                return

        if settings.auth_enabled:
            identity = _resolve_token(settings, token) if token else None
            if identity is None:
                record_auth_denial("unauthenticated")
                logger.warning(
                    "Authentication failed on mounted sub-app",
                    extra={
                        "path": scope.get("path", ""),
                        "reason": deny_reason or "invalid_token",
                    },
                )
                await self._respond(send, 401, b"Unauthorized: missing or invalid credentials")
                return

            try:
                method = scope.get("method", "GET")
                path = scope.get("path", "")
                _enforce_credential_policy(
                    identity,
                    method=method,
                    path=path,
                    contract_header=headers.get("x-privshield-contract-version", ""),
                    peer=peer,
                )
                _enforce_sandboxes(identity, path)
            except AuthError as exc:
                record_auth_denial("forbidden")
                logger.warning(
                    "Authorization failed on mounted sub-app",
                    extra={
                        "path": scope.get("path", ""),
                        "identity_name": identity.name,
                        "code": exc.code,
                    },
                )
                await self._respond(send, 403, b"Forbidden: insufficient scope for metrics access")
                return

            if not identity.has_permission(METRICS_SCOPE):
                record_auth_denial("forbidden")
                logger.warning(
                    "Authorization failed on mounted sub-app: insufficient scope",
                    extra={"path": scope.get("path", ""), "identity_name": identity.name},
                )
                await self._respond(send, 403, b"Forbidden: insufficient scope for metrics access")
                return

            await self.app(scope, receive, send)
            return

        # 认证关闭且无可命中的准入策略：生产姿态或已配置专属 Key（未匹配）
        # 时一律拒绝，避免指标面静默公开。
        record_auth_denial("unauthenticated")
        logger.warning(
            "Authentication failed on mounted sub-app: metrics credential required",
            extra={
                "path": scope.get("path", ""),
                "reason": deny_reason or "invalid_token",
                "metrics_auth_mode": mode,
            },
        )
        await self._respond(send, 401, b"Unauthorized: metrics credential required")

    @staticmethod
    async def _respond(send: Any, status: int, body: bytes) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
