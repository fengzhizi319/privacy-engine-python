"""REST surface for the auth kernel, mirroring Go ``pkg/auth/user_handlers*.go``.

路由层只做载荷绑定、身份注入与响应信封/原始 JSON 的选择；全部授权、
节流与 Step-Up 裁决都在 :mod:`engine.auth.service`。逐端点的响应形态与
Go 保持一致：用户/会话/口令/服务账号管理走成功信封，MFA、权限视图、
服务账号列表与令牌交换走原始 JSON。
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from ..observability.envelope import _build_error_envelope, _build_success_envelope
from ..security.auth import (
    _rest_identity_values,
    _terminator_peer_from_values,
    get_current_identity,
)
from ..security.config import get_security_settings
from ..security.identity import Identity
from .config import AuthSettings
from .contract import parse_contract
from .errors import AuthError
from .idempotency import get_idempotency_store, request_hash
from .models import EgressQuota
from .runtime import get_auth_runtime
from .service import AuthService, is_loopback_peer

router = APIRouter(prefix="/v1/auth", tags=["auth"])
well_known_router = APIRouter(prefix="/v1/.well-known/service-capabilities", tags=["auth"])

# Auth routes are authenticated by the shared REST dependency; public routes
# (login / register / bootstrap / activation) bypass it through the public-path
# allow-list implemented in ``engine.security.auth``.
_identity = get_current_identity

_PRODUCTION_ENV_VARS = ("APP_ENV", "PRIVACY_ENV", "SERVICE_HUB_ENV")


def _error(
    status: int,
    code: str,
    message: str,
    detail: Any = None,
    headers: dict[str, str] | None = None,
) -> AuthError:
    return AuthError(status, code, message, detail, headers=headers)


def _service() -> AuthService:
    runtime = get_auth_runtime()
    if runtime is None:
        raise _error(503, "AUTH_UNAVAILABLE", "authentication subsystem is not enabled")
    return runtime.service


def _settings() -> AuthSettings:
    return _service().settings


def register_auth_exception_handler(app: Any) -> None:
    """Map :class:`AuthError` onto the shared five-field error envelope."""

    @app.exception_handler(AuthError)
    async def _handle_auth_error(request: Request, exc: AuthError) -> JSONResponse:
        response = _build_error_envelope(
            code=exc.code,
            message=exc.message,
            detail=exc.detail,
            request=request,
            status_code=exc.status_code,
        )
        response.headers.update(exc.headers)
        return response


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


def _loopback(request: Request) -> bool:
    return is_loopback_peer(_client_ip(request))


def _bearer_token(request: Request) -> str:
    parts = (request.headers.get("authorization") or "").split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return ""


def _terminator_spki(request: Request) -> tuple[str, str]:
    """TLS 终结器注入的对端证书材料。

    终结器身份校验开启时只信任已验签的 ``request.state.terminator_peer``，
    返回其 raw SPKI DER 的标准 base64；校验关闭时才回退到旧式
    ``x-client-spki`` / ``x-client-spiffe`` 环回头，兼容尚未部署终结器的
    开发环境。SPKIFFE 不由 Go 终结器签名，故验签路径下恒为空。
    """
    settings = get_security_settings()
    if settings.terminator_identity_enabled:
        peer = getattr(request.state, "terminator_peer", None)
        if peer is None:
            peer = _terminator_peer_from_values(
                settings,
                method=request.method,
                path=request.url.path,
                values=_rest_identity_values(request),
                client_host=request.client.host if request.client else None,
            )
        if peer is None:
            return "", ""
        return peer.spki_der_base64, ""
    if not _loopback(request):
        return "", ""
    return (
        (request.headers.get("x-client-spki") or "").strip(),
        (request.headers.get("x-client-spiffe") or "").strip(),
    )


def _production_labeled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in {"prod", "production"}
        for name in _PRODUCTION_ENV_VARS
    )


async def _raw_body(request: Request) -> bytes:
    return await request.body()


def _bind(
    raw: bytes,
    *,
    message: str,
    allow_empty: bool = False,
    code: str = "INVALID_ARGUMENT",
) -> dict[str, Any]:
    if not raw.strip():
        if allow_empty:
            return {}
        raise _error(400, code, message)
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise _error(400, code, message, str(exc)) from exc
    if not isinstance(payload, dict):
        raise _error(400, code, message, "request body must be a JSON object")
    return payload


async def _bind_request(
    request: Request,
    *,
    message: str,
    allow_empty: bool = False,
    allow_form: bool = False,
    code: str = "INVALID_ARGUMENT",
) -> dict[str, Any]:
    """Bind JSON, or form data when ``allow_form`` mirrors Go ``c.ShouldBind``.

    Gin's form binding merges URL query and body parameters with body values
    taking precedence (``http.Request.ParseForm``), so the same merge is done
    here for ``application/x-www-form-urlencoded`` and ``multipart/form-data``.
    """
    if not allow_form:
        return _bind(await _raw_body(request), message=message, allow_empty=allow_empty, code=code)

    content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if content_type not in ("application/x-www-form-urlencoded", "multipart/form-data"):
        return _bind(await _raw_body(request), message=message, allow_empty=allow_empty, code=code)

    raw = await _raw_body(request)
    if not raw.strip():
        if allow_empty:
            return {}
        raise _error(400, code, message)

    payload: dict[str, Any] = dict(request.query_params)
    form = await request.form()
    payload.update(dict(form.items()))
    return payload


def _as_str(
    payload: dict[str, Any], key: str, message: str, code: str = "INVALID_ARGUMENT"
) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise _error(400, code, message)
    return str(value)


def _required_str(
    payload: dict[str, Any], key: str, message: str, code: str = "INVALID_ARGUMENT"
) -> str:
    value = _as_str(payload, key, message, code)
    if not value:
        raise _error(400, code, message)
    return value


def _str_list(payload: dict[str, Any], key: str, message: str) -> list[str]:
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _error(400, "INVALID_ARGUMENT", message)
    return list(value)


def _int_field(payload: dict[str, Any], key: str, message: str) -> int:
    value = payload.get(key)
    if value is None or value == "":
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(400, "INVALID_ARGUMENT", message)
    return value


def _admin_code(request: Request, payload: dict[str, Any]) -> str:
    """Header ``X-MFA-Code`` first, then body ``admin_totp_code`` (Go order)."""
    header = (request.headers.get("x-mfa-code") or "").strip()
    if header:
        return header
    return _as_str(payload, "admin_totp_code", "invalid payload").strip()


def _bootstrap_token(request: Request) -> str:
    return (request.headers.get("x-bootstrap-token") or "").strip()


def _ok(request: Request, message: str, data: Any, status_code: int = 200) -> JSONResponse:
    return _build_success_envelope(message, data, request, status_code=status_code)


def _raw(
    data: Any,
    *,
    status_code: int = 200,
    no_store: bool = False,
    pragma: bool = False,
) -> JSONResponse:
    headers: dict[str, str] = {}
    if no_store:
        headers["Cache-Control"] = "no-store"
    if pragma:
        headers["Pragma"] = "no-cache"
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(data),
        headers=headers,
    )


def _mfa_subject(identity: Identity) -> str:
    if not identity.subject:
        raise _error(401, "UNAUTHENTICATED", "authentication required")
    return identity.subject


def _reject_workload_mfa(identity: Identity, message: str) -> None:
    if (
        identity.role == "service_account"
        or identity.subject.startswith("svc:")
        or identity.principal_type == "workload"
    ):
        raise _error(403, "FORBIDDEN", message)


def _quota(payload: dict[str, Any]) -> EgressQuota | None:
    raw = payload.get("egress_quota")
    quota = EgressQuota.from_dict(raw if isinstance(raw, dict) else None)
    if quota is not None and not (
        quota.rows_per_day or quota.bytes_per_day or quota.max_rows_per_response
    ):
        return None
    return quota


# ---------------------------------------------------------------------------
# Public authentication endpoints
# ---------------------------------------------------------------------------


@router.post("/login")
async def login(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid login payload")
    data = _service().login(
        username=_required_str(payload, "username", "invalid login payload"),
        password=_required_str(payload, "password", "invalid login payload"),
        totp_code=_as_str(payload, "totp_code", "invalid login payload"),
        recovery_code=_as_str(payload, "recovery_code", "invalid login payload"),
        client_ip=_client_ip(request),
    )
    return _ok(request, "login succeeded", data)


@router.post("/register")
async def public_register(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    settings = _settings()
    if not settings.self_register:
        raise _error(
            403,
            "SELF_REGISTER_DISABLED",
            "auth: self-service registration is disabled; ask an administrator to create the account",
        )
    payload = _bind(await _raw_body(request), message="invalid register payload")
    data = _service().public_register(
        username=_required_str(payload, "username", "invalid register payload"),
        password=_required_str(payload, "password", "invalid register payload"),
        display_name=_as_str(payload, "display_name", "invalid register payload"),
        client_ip=_client_ip(request),
    )
    return _ok(request, "user registered", data)


@router.post("/users/register")
async def register_user(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid register payload")
    data = _service().register(
        caller=identity,
        username=_required_str(payload, "username", "invalid register payload"),
        password=_required_str(payload, "password", "invalid register payload"),
        display_name=_as_str(payload, "display_name", "invalid register payload"),
        role=_as_str(payload, "role", "invalid register payload"),
        scopes=_str_list(payload, "scopes", "invalid register payload"),
        must_change_password=bool(payload.get("must_change_password", False)),
        cert_fingerprints=_str_list(payload, "cert_fingerprints", "invalid register payload"),
        client_ip=_client_ip(request),
        bootstrap_token=_bootstrap_token(request),
        loopback_peer=_loopback(request),
    )
    return _ok(request, "user registered", data)


@router.post("/users/bootstrap")
async def bootstrap_admin(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid register payload")
    data = _service().bootstrap_admin(
        username=_required_str(payload, "username", "invalid register payload"),
        password=_required_str(payload, "password", "invalid register payload"),
        display_name=_as_str(payload, "display_name", "invalid register payload"),
        role=_as_str(payload, "role", "invalid register payload"),
        scopes=_str_list(payload, "scopes", "invalid register payload"),
        client_ip=_client_ip(request),
        bootstrap_token=_bootstrap_token(request),
        loopback_peer=_loopback(request),
    )
    return _ok(request, "user registered", data)


@router.post("/logout")
async def logout(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    data = _service().logout(_bearer_token(request))
    return _ok(request, "session revoked", data)


@router.post("/change-password")
async def change_password(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload")
    _service().change_password(
        caller=identity,
        username=_required_str(payload, "username", "invalid payload"),
        old_password=_required_str(payload, "old_password", "invalid payload"),
        new_password=_required_str(payload, "new_password", "invalid payload"),
        client_ip=_client_ip(request),
    )
    return _ok(
        request,
        "password changed; all sessions revoked",
        {"message": "password changed successfully", "sessions_revoked": True},
    )


# ---------------------------------------------------------------------------
# User administration
# ---------------------------------------------------------------------------


@router.get("/permissions")
async def permission_view(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    raw_offset = request.query_params.get("offset", "")
    offset = 0
    if raw_offset:
        try:
            offset = int(raw_offset)
        except ValueError as exc:
            raise _error(400, "INVALID_ARGUMENT", "invalid offset") from exc
        if offset < 0:
            raise _error(400, "INVALID_ARGUMENT", "invalid offset")
    return _raw(_service().permission_view(identity, offset))


@router.post("/users")
async def admin_create_user(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid register payload")
    data = _service().admin_create_user(
        caller=identity,
        username=_required_str(payload, "username", "invalid register payload"),
        password=_required_str(payload, "password", "invalid register payload"),
        display_name=_as_str(payload, "display_name", "invalid register payload"),
        role=_as_str(payload, "role", "invalid register payload"),
        scopes=_str_list(payload, "scopes", "invalid register payload"),
        must_change_password=bool(payload.get("must_change_password", False)),
        cert_fingerprints=_str_list(payload, "cert_fingerprints", "invalid register payload"),
        admin_code=_admin_code(request, payload),
    )
    return _ok(request, "user registered", data)


@router.get("/users")
async def list_users(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    return _ok(request, "users listed", _service().list_users(identity))


@router.get("/users/{username}")
async def get_user(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    return _ok(request, "user profile", _service().get_user(identity, username))


@router.put("/users/{username}/permissions")
async def update_permissions(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload")
    data = _service().update_permissions(
        caller=identity,
        username=username,
        role=_as_str(payload, "role", "invalid payload"),
        scopes=_str_list(payload, "scopes", "invalid payload"),
        admin_code=_admin_code(request, payload),
    )
    return _ok(request, "permissions updated", data)


@router.put("/users/{username}/certificates")
async def set_certificates(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload")
    data = _service().set_certificates(
        caller=identity,
        username=username,
        fingerprints=_str_list(payload, "cert_fingerprints", "invalid payload"),
        admin_code=_admin_code(request, payload),
    )
    return _ok(request, "certificate fingerprints updated", data)


@router.delete("/users/{username}/certificates")
async def clear_certificates(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload", allow_empty=True)
    data = _service().clear_certificates(
        caller=identity,
        username=username,
        admin_code=_admin_code(request, payload),
    )
    return _ok(request, "certificate fingerprints cleared", data)


@router.put("/users/{username}/status")
async def set_status(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload")
    data = _service().set_status(
        caller=identity,
        username=username,
        status=_required_str(payload, "status", "invalid payload"),
        reason=_as_str(payload, "reason", "invalid payload"),
        admin_code=_admin_code(request, payload),
    )
    return _ok(request, "status updated", data)


@router.delete("/users/{username}")
async def delete_user(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload", allow_empty=True)
    data = _service().delete_user(
        caller=identity,
        username=username,
        admin_code=_admin_code(request, payload),
    )
    return _ok(request, "user deleted", data)


@router.post("/users/{username}/reset-password")
async def reset_password(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload", allow_empty=True)
    data = _service().reset_password(
        caller=identity,
        username=username,
        old_password=_as_str(payload, "old_password", "invalid payload"),
        new_password=_as_str(payload, "new_password", "invalid payload"),
        admin_code=_admin_code(request, payload),
        client_ip=_client_ip(request),
    )
    return _ok(request, "password reset successfully", data)


# ---------------------------------------------------------------------------
# User API keys
# ---------------------------------------------------------------------------


@router.post("/users/{username}/keys")
async def issue_key(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload")
    data = _service().issue_key(
        caller=identity,
        username=username,
        key_name=_as_str(payload, "key_name", "invalid payload"),
        scopes=_str_list(payload, "scopes", "invalid payload"),
        audiences=_str_list(payload, "allowed_audiences", "invalid payload"),
        ttl_seconds=_int_field(payload, "ttl_seconds", "invalid payload"),
        admin_code=_admin_code(request, payload),
        client_ip=_client_ip(request),
    )
    return _ok(request, "api key issued", data)


@router.get("/users/{username}/keys")
async def list_keys(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    return _ok(request, "api keys listed", _service().list_keys(identity, username))


@router.delete("/users/{username}/keys/{key_id}")
async def revoke_key(
    username: str, key_id: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    _service().revoke_key(caller=identity, username=username, key_id=key_id)
    return _ok(request, "api key revoked", {"revoked": key_id})


# ---------------------------------------------------------------------------
# MFA lifecycle (raw JSON, Go-compatible shapes)
# ---------------------------------------------------------------------------


@router.post("/mfa/setup")
async def mfa_setup(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    subject = _mfa_subject(identity)
    _reject_workload_mfa(identity, "service accounts cannot configure interactive MFA")
    payload = _bind(await _raw_body(request), message="invalid payload", allow_empty=True)
    data = _service().mfa_setup(
        subject=subject,
        code=_as_str(payload, "code", "invalid payload"),
        recovery_code=_as_str(payload, "recovery_code", "invalid payload"),
        session_auth=identity.session_auth,
        client_ip=_client_ip(request),
    )
    return _raw(data, no_store=True, pragma=True)


@router.post("/mfa/setup/cancel")
async def mfa_setup_cancel(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    if not identity.session_auth:
        raise _error(401, "UNAUTHENTICATED", "session required")
    payload = _bind(await _raw_body(request), message="invalid payload")
    setup_id = _required_str(payload, "setup_id", "invalid payload")
    data = _service().mfa_cancel(subject=identity.subject, setup_id=setup_id)
    return _raw(data, no_store=True)


@router.post("/mfa/enable")
async def mfa_enable(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    subject = _mfa_subject(identity)
    _reject_workload_mfa(identity, "service accounts cannot configure interactive MFA")
    payload = _bind(await _raw_body(request), message="code and setup_id are required")
    data = _service().mfa_enable(
        subject=subject,
        code=_required_str(payload, "code", "code and setup_id are required"),
        setup_id=_required_str(payload, "setup_id", "code and setup_id are required"),
        session_auth=identity.session_auth,
        client_ip=_client_ip(request),
    )
    return _raw(data, no_store=True, pragma=True)


@router.post("/mfa/disable")
async def mfa_disable(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    _mfa_subject(identity)
    payload = _bind(await _raw_body(request), message="invalid payload", allow_empty=True)
    data = _service().mfa_disable(
        caller=identity,
        username=_as_str(payload, "username", "invalid payload"),
        code=_as_str(payload, "code", "invalid payload"),
        admin_code=_admin_code(request, payload),
    )
    return _raw(data)


@router.get("/mfa/status")
async def mfa_status(identity: Identity = Depends(_identity)) -> JSONResponse:
    subject = _mfa_subject(identity)
    return _raw(_service().mfa_status(subject), no_store=True, pragma=True)


@router.post("/mfa/recovery-codes/regenerate")
async def regenerate_recovery_codes(
    request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    subject = _mfa_subject(identity)
    _reject_workload_mfa(
        identity, "service accounts cannot regenerate interactive recovery codes"
    )
    payload = _bind(await _raw_body(request), message="code and setup_id are required")
    data = _service().mfa_regenerate_recovery_codes(
        subject=subject,
        code=_required_str(payload, "code", "code and setup_id are required"),
        client_ip=_client_ip(request),
    )
    return _raw(data, no_store=True, pragma=True)


# ---------------------------------------------------------------------------
# Service accounts / workload IAM
# ---------------------------------------------------------------------------


@router.post("/service-accounts")
async def create_service_account(
    request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload")
    audiences = _str_list(payload, "allowed_audiences", "invalid payload")
    if not audiences:
        raise _error(
            400, "INVALID_ARGUMENT", "allowed_audiences is required and cannot be empty"
        )
    data = _service().create_service_account(
        caller=identity,
        username=_required_str(payload, "username", "invalid payload"),
        display_name=_as_str(payload, "display_name", "invalid payload"),
        tenant_id=_as_str(payload, "tenant_id", "invalid payload"),
        scopes=_str_list(payload, "scopes", "invalid payload"),
        allowed_audiences=audiences,
        cert_fingerprints=_str_list(payload, "cert_fingerprints", "invalid payload"),
        egress_quota=_quota(payload),
        expected_public_key_sha256=_as_str(
            payload, "expected_public_key_sha256", "invalid payload"
        ),
        expected_spiffe_id=_as_str(payload, "expected_spiffe_id", "invalid payload"),
        admin_code=_admin_code(request, payload),
    )
    return _ok(request, "service account created", data)


@router.post("/service-accounts/activate")
async def activate_service_account(
    request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid activation request")
    token = (request.headers.get("x-activation-token") or "").strip()
    if not token:
        token = _as_str(payload, "activation_token", "invalid activation request").strip()
    username = _as_str(payload, "username", "invalid activation request").strip()
    spki, spiffe = _terminator_spki(request)
    data = _service().activate_service_account(
        username=username,
        activation_token=token,
        actual_spki=spki,
        actual_spiffe=spiffe or _as_str(payload, "actual_spiffe", "invalid activation request"),
        tenant_id=_as_str(payload, "tenant_id", "invalid activation request"),
    )
    return _ok(request, "service account activated successfully", data)


@router.get("/service-accounts")
async def list_service_accounts(identity: Identity = Depends(_identity)) -> JSONResponse:
    accounts = _service().list_service_accounts(identity)
    return _raw({"service_accounts": accounts, "count": len(accounts)})


@router.get("/service-accounts/{username}")
async def get_service_account(
    username: str, identity: Identity = Depends(_identity)
) -> JSONResponse:
    try:
        data = _service().get_service_account(identity, username)
    except AuthError as exc:
        if exc.code == "USER_NOT_FOUND":
            raise _error(404, "NOT_FOUND", "service account not found") from exc
        raise
    return _raw(data)


@router.put("/service-accounts/{username}/quota")
async def set_service_account_quota(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid quota payload")
    data = _service().set_service_account_quota(
        caller=identity,
        username=username,
        quota=_quota(payload),
        admin_code=_admin_code(request, payload),
    )
    return _ok(request, "egress quota updated", data)


@router.post("/service-accounts/{username}/freeze")
async def freeze_service_account(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload", allow_empty=True)
    data = _service().freeze_service_account(
        caller=identity,
        username=username,
        reason=_as_str(payload, "reason", "invalid payload"),
        admin_code=_admin_code(request, payload),
    )
    return _raw(data)


@router.post("/service-accounts/{username}/unfreeze")
async def unfreeze_service_account(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload", allow_empty=True)
    data = _service().unfreeze_service_account(
        caller=identity,
        username=username,
        reason=_as_str(payload, "reason", "invalid payload"),
        admin_code=_admin_code(request, payload),
    )
    return _raw(data)


@router.post("/service-accounts/{username}/keys")
async def issue_service_account_key(
    username: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    raw = await _raw_body(request)
    payload = _bind(raw, message="invalid payload")
    audiences = _str_list(payload, "allowed_audiences", "invalid payload")
    if not audiences:
        raise _error(
            400, "INVALID_ARGUMENT", "allowed_audiences is required and cannot be empty"
        )
    contract = parse_contract(payload.get("contract"))

    idempotency = get_idempotency_store()
    idem_key = (request.headers.get("x-idempotency-key") or "").strip()
    record = None
    if idem_key:
        record, replayed = idempotency.begin(
            identity.tenant_id or "default",
            idem_key,
            identity.subject,
            request.url.path,
            request_hash(raw),
        )
        if replayed:
            return Response(
                content=record.response_body,
                status_code=record.response_code,
                media_type="application/json",
            )

    try:
        binding = payload.get("binding")
        binding_pubkey = ""
        if isinstance(binding, dict):
            binding_pubkey = _as_str(binding, "pubkey", "invalid payload")
        data = _service().issue_service_account_key(
            caller=identity,
            username=username,
            key_name=_as_str(payload, "name", "invalid payload"),
            scopes=_str_list(payload, "scopes", "invalid payload"),
            allowed_audiences=audiences,
            binding_pubkey=binding_pubkey,
            ttl_seconds=_int_field(payload, "ttl_seconds", "invalid payload"),
            contract=contract,
            admin_code=_admin_code(request, payload),
        )
    except AuthError:
        idempotency.abort(record)
        raise

    if record is not None:
        sanitized = {
            key: value
            for key, value in data.items()
            if key not in {"token", "created_at", "expires_at"}
        }
        sanitized.update(
            {
                "credential_delivered": False,
                "message": "credential was already delivered in initial request and cannot be replayed for security",
                "created_at": data.get("created_at"),
                "expires_at": data.get("expires_at"),
            }
        )
        try:
            body = json.dumps(jsonable_encoder(sanitized), ensure_ascii=False).encode()
        except (TypeError, ValueError):
            body = json.dumps(sanitized, default=str).encode()
        idempotency.commit(record, 201, body)

    return _raw(data, status_code=201)


@router.get("/service-accounts/{username}/keys")
async def list_service_account_keys(
    username: str, identity: Identity = Depends(_identity)
) -> JSONResponse:
    try:
        data = _service().list_service_account_keys(identity, username)
    except AuthError as exc:
        if exc.code == "USER_NOT_FOUND":
            raise _error(404, "NOT_FOUND", "service account not found") from exc
        raise
    return _raw(data)


@router.delete("/service-accounts/{username}/keys/{key_id}")
async def revoke_service_account_key(
    username: str, key_id: str, request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload", allow_empty=True)
    data = _service().revoke_service_account_key(
        caller=identity,
        username=username,
        key_id=key_id,
        admin_code=_admin_code(request, payload),
    )
    return _raw(data)


@router.post("/service-accounts/{username}/keys/{key_id}/rotate")
async def rotate_service_account_key(
    username: str,
    key_id: str,
    request: Request,
    identity: Identity = Depends(_identity),
) -> JSONResponse:
    raw = await _raw_body(request)
    payload = _bind(raw, message="invalid payload", allow_empty=True)

    idempotency = get_idempotency_store()
    idem_key = (request.headers.get("x-idempotency-key") or "").strip()
    record = None
    if idem_key:
        record, replayed = idempotency.begin(
            identity.tenant_id or "default",
            idem_key,
            identity.subject,
            request.url.path,
            request_hash(raw),
        )
        if replayed:
            return Response(
                content=record.response_body,
                status_code=record.response_code,
                media_type="application/json",
            )

    try:
        data = _service().rotate_service_account_key(
            caller=identity,
            username=username,
            key_id=key_id,
            ttl_seconds=_int_field(payload, "ttl_seconds", "invalid payload"),
            admin_code=_admin_code(request, payload),
        )
    except AuthError:
        idempotency.abort(record)
        raise
    data["credential_delivered"] = True

    if record is not None:
        sanitized = {
            key: value
            for key, value in data.items()
            if key not in {"token", "created_at", "expires_at"}
        }
        sanitized.update(
            {
                "credential_delivered": False,
                "message": "credential was already delivered in initial request and cannot be replayed for security",
                "created_at": data.get("created_at"),
                "expires_at": data.get("expires_at"),
            }
        )
        idempotency.commit(record, 201, json.dumps(sanitized, default=str).encode())

    return _raw(data, status_code=201)


# ---------------------------------------------------------------------------
# Token exchange / revocation
# ---------------------------------------------------------------------------


@router.post("/token-exchange")
async def token_exchange(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    if _production_labeled():
        spki, _ = _terminator_spki(request)
        if not spki:
            raise _error(
                403,
                "CLIENT_CERTIFICATE_REQUIRED",
                "mTLS client certificate is required for workload token exchange in production",
            )
    payload = await _bind_request(
        request,
        message="invalid request payload",
        allow_form=True,
        code="INVALID_REQUEST",
    )
    data = _service().token_exchange(
        identity,
        grant_type=_required_str(
            payload, "grant_type", "invalid request payload", code="INVALID_REQUEST"
        ),
        client_id=_required_str(
            payload, "client_id", "invalid request payload", code="INVALID_REQUEST"
        ),
        audience=_required_str(
            payload, "audience", "invalid request payload", code="INVALID_REQUEST"
        ),
        subject_token=_as_str(
            payload, "subject_token", "invalid request payload", code="INVALID_REQUEST"
        ),
        subject_token_type=_as_str(
            payload, "subject_token_type", "invalid request payload", code="INVALID_REQUEST"
        ),
        scope=_as_str(payload, "scope", "invalid request payload", code="INVALID_REQUEST"),
    )
    return _raw(data)


@router.post("/tokens/revoke")
async def revoke_token(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    payload = _bind(
        await _raw_body(request), message="invalid request payload", code="INVALID_REQUEST"
    )
    data = _service().revoke_token(
        identity,
        token=_as_str(payload, "token", "invalid request payload", code="INVALID_REQUEST"),
        token_type_hint=_as_str(payload, "jti", "invalid request payload", code="INVALID_REQUEST"),
    )
    return _raw(data)


# ---------------------------------------------------------------------------
# Capability discovery
# ---------------------------------------------------------------------------


@well_known_router.get("/public")
async def public_capabilities(request: Request, identity: Identity = Depends(_identity)) -> JSONResponse:
    response = _ok(request, "public service capabilities", _service().public_capabilities())
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


@well_known_router.post("/negotiate")
async def negotiate_capabilities(
    request: Request, identity: Identity = Depends(_identity)
) -> JSONResponse:
    payload = _bind(await _raw_body(request), message="invalid payload")
    data = _service().negotiate_capabilities(
        identity, _as_str(payload, "contract_version", "invalid payload")
    )
    return _ok(request, "service capabilities negotiated", data)
