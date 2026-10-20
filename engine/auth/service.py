"""Authentication service layer mirroring Go ``pkg/auth`` HTTP semantics.

本模块把 :mod:`engine.auth.store` 的裁决能力编排为与 Go ``pkg/auth``
用户端点一致的业务语义：登录 / 会话 / 口令生命周期 / MFA / 用户与
服务账号管理 / 令牌交换。路由层（``routes.py``）只做载荷绑定与响应
信封包装，所有授权判定集中在此处，便于端到端测试。

安全口径（与 Go 对齐）：
- 公开认证路径未携凭证时注入 ``public-caller``（仅 ``auth:public``）；
- 登录会话未完成 TOTP 只能进入 MFA 绑定沙箱；
- 管理员操作他人一律 Step-Up（默认强制），静态机器凭据 fail-closed；
- 「特权不得超越自身」按实际生效 Scope 集合校验；
- 审计员受三权分立保护，管理员不得干预其账号。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..observability.logging_config import get_logger
from ..security.identity import Identity
from .bound_pubkey import public_key_digest
from .config import AuthSettings
from .errors import AuthError
from .models import (
    DEFAULT_API_KEY_TTL_SECONDS,
    DEFAULT_SESSION_TTL_SECONDS,
    MAX_API_KEY_TTL_SECONDS,
    MFA_RECOVERY_SESSION_TTL_SECONDS,
    PRINCIPAL_WORKLOAD,
    ROLE_ADMIN,
    ROLE_AUDITOR,
    ROLE_SERVICE_ACCOUNT,
    STATUS_ACTIVE,
    STATUS_DISABLED,
    EgressQuota,
    default_scopes_for_role,
    is_privileged_role,
    isoformat,
    normalize_username,
    now_utc,
    validate_display_name,
)
from .store import (
    MFASetupState,
    UserStore,
    hash_token,
    issuance_scope_allowed,
    validate_audiences,
    validate_cert_fingerprints,
)
from .totp import default_totp_algorithm

logger = get_logger(__name__)

# 公开认证路径：未携凭证时以 public-caller 放行，具体授权在 Handler 内裁决。
# 改密 / MFA 沙箱的共同豁免白名单（与 Go IsPasswordExpiryExemptPath 同表）。
_SANDBOX_EXEMPT_PATHS = frozenset(
    {
        "/health",
        "/readyz",
        "/livez",
        "/v1/auth/login",
        "/v1/auth/register",
        "/v1/auth/logout",
        "/v1/auth/change-password",
        "/v1/auth/mfa/status",
        "/v1/auth/mfa/setup",
        "/v1/auth/mfa/setup/cancel",
        "/v1/auth/mfa/enable",
    }
)

_MAX_TOKEN_TTL_SECONDS = 24 * 3600
_DEFAULT_WORKLOAD_TOKEN_TTL_SECONDS = 15 * 60
_MAX_WORKLOAD_TOKEN_TTL_SECONDS = 3600

_TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
_ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


def _error(
    status: int,
    code: str,
    message: str,
    detail: Any = None,
    headers: dict[str, str] | None = None,
) -> AuthError:
    return AuthError(status, code, message, detail, headers=headers)


def normalize_exempt_path(path: str) -> str:
    trimmed = path.rstrip("/")
    return trimmed or "/"


def is_password_expiry_exempt_path(path: str, subject: str = "") -> bool:
    normalized = normalize_exempt_path(path)
    if normalized in _SANDBOX_EXEMPT_PATHS:
        return True
    if subject and normalized.lower() == f"/v1/auth/users/{normalize_username(subject)}".lower():
        return True
    return False


def is_caller_admin(identity: Identity | None) -> bool:
    if identity is None:
        return False
    return (
        identity.has_permission("*")
        or identity.has_permission("admin")
        or identity.has_permission("user:admin")
    )


def is_full_admin(identity: Identity | None) -> bool:
    if identity is None:
        return False
    return identity.has_permission("*") or identity.has_permission("admin")


def can_view_user_account(identity: Identity | None, username: str) -> bool:
    if identity is None:
        return False
    return (
        is_caller_admin(identity)
        or identity.has_permission("user:read")
        or identity.is_subject(username)
    )


def can_manage_user_account(identity: Identity | None, username: str) -> bool:
    if identity is None:
        return False
    return is_caller_admin(identity) or identity.is_subject(username)


def caller_label(identity: Identity | None) -> str:
    if identity is None:
        return "anonymous"
    return identity.subject or identity.name or "anonymous"


def is_loopback_peer(host: str | None) -> bool:
    if not host:
        return False
    value = host.strip().strip("[]").lower()
    if value.startswith("::ffff:"):
        value = value[7:]
    return value in {"127.0.0.1", "::1", "localhost"} or value.startswith("127.")


def mfa_fault(exc: AuthError, *, admin: bool = False) -> AuthError:
    """把存储层 MFA 异常映射为与 Go ``respondMFAFault`` 一致的状态与错误码。"""
    if exc.code == "ACCOUNT_LOCKED":
        return _error(429, "ACCOUNT_LOCKED", "account temporarily locked", exc.detail,
                      headers={"Retry-After": "900"})
    if exc.code in {"MFA_SETUP_CHANGED", "MFA_REQUIRED"}:
        return _error(
            409,
            "MFA_SETUP_CHANGED",
            "restart MFA setup; the pending binding was changed, expired, or consumed",
        )
    if exc.code == "MFA_SECRET_UNAVAILABLE":
        return _error(
            503,
            "ADMIN_MFA_SECRET_UNAVAILABLE" if admin else "MFA_SECRET_UNAVAILABLE",
            "MFA key material is temporarily unavailable",
        )
    if exc.code == "MFA_VERIFICATION_UNAVAILABLE":
        return _error(503, "MFA_VERIFICATION_UNAVAILABLE", "MFA verification infrastructure is temporarily unavailable")
    if admin:
        return _error(401, "INVALID_ADMIN_MFA_CODE", "invalid administrator verification code")
    return _error(400, "INVALID_MFA_CODE", "invalid verification code")


@dataclass
class IssuedToken:
    """令牌交换签发的进程内不透明票据（只存摘要索引，不落明文）。"""

    subject: str
    role: str
    scopes: list[str]
    expires_at: datetime
    tenant_id: str = ""
    principal_type: str = "workload"
    client_id: str = ""
    actor: str = ""
    audience: str = ""
    pwd_ver: int = 0
    jti: str = ""


class AuthService:
    """High-level auth flows used by the REST router and the auth middleware."""

    def __init__(self, settings: AuthSettings, store: UserStore) -> None:
        self.settings = settings
        self.store = store
        self._issued_tokens: dict[str, IssuedToken] = {}
        self._token_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Authentication resolution (used by security.auth middleware)
    # ------------------------------------------------------------------

    def resolve_token(self, token: str) -> Identity | None:
        """把 Bearer 凭据解析为身份：登录会话 → 用户 API Key → 交换票据。"""
        if not token:
            return None
        session = self.store.lookup_session(token)
        if session is not None:
            user, entry = session
            return Identity(
                service_type="internal",
                name=f"{user.username}:session",
                scopes=list(entry.scopes),
                subject=user.username,
                role=user.role,
                principal_type=user.principal_type,
                tenant_id=user.tenant_id,
                mfa_authenticated=entry.mfa_authenticated,
                session_auth=True,
                mfa_enabled=user.mfa_enabled,
            )
        api_key = self.store.lookup_api_key(token)
        if api_key is not None:
            user, record = api_key
            contract = record.contract if isinstance(record.contract, dict) else {}
            return Identity(
                service_type="internal",
                name=f"{user.username}:{record.name}",
                scopes=list(record.scopes),
                subject=user.username,
                role=user.role,
                principal_type=user.principal_type,
                tenant_id=user.tenant_id,
                allowed_audiences=list(record.allowed_audiences),
                key_id=record.key_id,
                binding_status=record.binding_status,
                binding_mode=record.binding_mode,
                bound_public_key_der=record.bound_public_key_der,
                contract_version=str(contract.get("version") or ""),
                contract_allowed_paths=[
                    str(item) for item in contract.get("allowed_paths") or []
                ],
            )
        record = self._lookup_issued_token(token)
        if record is not None:
            return Identity(
                service_type="internal",
                name=record.subject,
                scopes=list(record.scopes),
                subject=record.subject,
                role=record.role,
                principal_type=record.principal_type,
                tenant_id=record.tenant_id,
                allowed_audiences=[record.audience] if record.audience else [],
                audience=record.audience,
                key_id=record.jti,
            )
        return None

    def _lookup_issued_token(self, token: str) -> IssuedToken | None:
        digest = hash_token(token)
        with self._token_lock:
            record = self._issued_tokens.get(digest)
            if record is None:
                return None
            if now_utc() >= record.expires_at:
                self._issued_tokens.pop(digest, None)
                return None
            user = self.store.users.get(record.subject)
            if user is None or user.status != STATUS_ACTIVE:
                self._issued_tokens.pop(digest, None)
                return None
            current_pwd_ver = _password_version(user.password_updated_at)
            if record.pwd_ver and current_pwd_ver != record.pwd_ver:
                self._issued_tokens.pop(digest, None)
                return None
            return record

    # ------------------------------------------------------------------
    # Sandboxes
    # ------------------------------------------------------------------

    def enforce_sandboxes(self, identity: Identity | None, path: str) -> None:
        """口令沙箱优先于 MFA 沙箱；两者共享同一豁免白名单。"""
        if identity is None or not identity.subject:
            return
        subject = identity.subject
        if is_password_expiry_exempt_path(path, subject):
            return
        if self.store.must_change_password_for_user(subject):
            raise _error(
                403,
                "PASSWORD_CHANGE_REQUIRED",
                "Forbidden: initial or reset password must be changed at /v1/auth/change-password before accessing business APIs",
                {"username": subject, "must_change_password": True},
            )
        if self.settings.password_expire_block and self.store.is_password_expired_for_user(
            subject, self.settings.password_expire_grace_days
        ):
            raise _error(
                403,
                "PASSWORD_EXPIRED_REQUIRE_CHANGE",
                "Forbidden: password has expired, please change your password at /v1/auth/change-password before continuing",
                {"username": subject, "password_expired": True, "max_password_age_days": 90},
            )
        if not identity.session_auth or identity.mfa_authenticated:
            return
        if not self.store.mfa_enabled(subject):
            raise _error(
                403,
                "MFA_ENROLLMENT_REQUIRED",
                "Forbidden: TOTP multi-factor authentication is mandatory for every account; "
                "enroll at /v1/auth/mfa/setup then /v1/auth/mfa/enable before accessing business APIs",
                {"must_enroll_mfa": True},
            )
        raise _error(
            403,
            "MFA_SESSION_REQUIRED",
            "Forbidden: this session was established without the account's TOTP code; sign in again with totp_code",
            {"mfa_required": True},
        )

    def allow_mfa_write(self, subject: str, client_ip: str) -> None:
        for key in (f"mfa:subject:{subject}", f"mfa:ip:{client_ip}"):
            ok, retry = self.store.login_throttle.allow(key, 20, 60)
            if not ok:
                raise _error(
                    429,
                    "RATE_LIMITED",
                    "too many MFA requests",
                    headers={"Retry-After": str(retry)},
                )

    # ------------------------------------------------------------------
    # Login / session
    # ------------------------------------------------------------------

    def login(
        self,
        *,
        username: str,
        password: str,
        totp_code: str = "",
        recovery_code: str = "",
        client_ip: str = "",
    ) -> dict[str, Any]:
        ok, retry = self.store.login_throttle.allow(
            f"login:{client_ip}",
            self.settings.login_throttle_per_min,
            60,
        )
        if not ok:
            raise _error(
                429,
                "RATE_LIMITED",
                "too many login attempts from this client, slow down",
                detail=None,
                headers={"Retry-After": str(retry)},
            )

        username = normalize_username(username)
        if username.startswith("svc:"):
            raise _error(
                403,
                "ACCOUNT_DISABLED",
                "service account is not allowed for interactive login, use token-exchange or mTLS",
            )

        result = self.store.authenticate(username, password)
        user = result.user
        if user.role == ROLE_SERVICE_ACCOUNT:
            raise _error(
                403,
                "ACCOUNT_DISABLED",
                "service account is not allowed for interactive login, use token-exchange or mTLS",
            )

        mfa_authenticated = False
        recovery_code_used = False
        if self.store.mfa_enabled(username):
            totp_code = totp_code.strip()
            recovery_code = recovery_code.strip()
            if not totp_code and not recovery_code:
                raise _error(
                    401,
                    "MFA_REQUIRED",
                    "multi-factor authentication is enabled for this account; totp_code (or recovery_code) is required",
                    {"mfa_required": True},
                )
            if not totp_code and recovery_code:
                try:
                    consumed = self.store.consume_recovery_code(username, recovery_code)
                except AuthError as exc:
                    raise mfa_fault(exc) from exc
                if not consumed:
                    if self.store.is_locked(username):
                        remaining = self.store.lockout_remaining(username)
                        raise _error(
                            429,
                            "ACCOUNT_LOCKED",
                            "account is temporarily locked due to too many failed attempts",
                            {"retry_after_seconds": remaining},
                            headers={"Retry-After": str(remaining)},
                        )
                    raise _error(
                        401,
                        "INVALID_RECOVERY_CODE",
                        "invalid or already consumed recovery code",
                    )
                recovery_code_used = True
                user = self.store.get_user(username)
            else:
                if not self.store.totp_verifiable(username):
                    raise _error(
                        503,
                        "MFA_SECRET_UNAVAILABLE",
                        "multi-factor verification is temporarily unavailable for this account, contact your administrator",
                    )
                try:
                    self.store.verify_totp(username, totp_code)
                except AuthError as exc:
                    if exc.code in {"INVALID_MFA_CODE", "MFA_REPLAY"}:
                        raise _error(401, "INVALID_MFA_CODE", "invalid verification code") from exc
                    raise mfa_fault(exc) from exc
                mfa_authenticated = True

        ttl = self.settings.session_ttl_seconds or DEFAULT_SESSION_TTL_SECONDS
        if recovery_code_used and (ttl <= 0 or ttl > MFA_RECOVERY_SESSION_TTL_SECONDS):
            ttl = MFA_RECOVERY_SESSION_TTL_SECONDS
        token, expires_at = self.store.create_session(
            user.username, ttl, mfa_authenticated=mfa_authenticated
        )

        password_expired = user.password_expired()
        must_enroll_mfa = not mfa_authenticated
        warning = _login_sandbox_warning(user.must_change_password, must_enroll_mfa)
        if recovery_code_used:
            rebinding = (
                "one-time recovery code consumed: the previous TOTP binding and all remaining codes were revoked; "
                "re-enroll at /v1/auth/mfa/setup + /v1/auth/mfa/enable to receive a fresh binding and a new set of "
                "recovery codes (mandatory before accessing business APIs; this recovery session expires within 10 minutes)"
            )
            warning = f"{warning}; {rebinding}" if warning else rebinding

        return {
            "token": token,
            "token_type": "Bearer",
            "expires_at": isoformat(expires_at),
            "user": user.to_summary(),
            "password_expired": password_expired,
            "must_change_password": user.must_change_password,
            "must_enroll_mfa": must_enroll_mfa,
            "recovery_code_used": recovery_code_used,
            "warning": warning,
        }

    def logout(self, token: str) -> dict[str, Any]:
        if not token:
            raise _error(400, "INVALID_ARGUMENT", "missing bearer token")
        try:
            self.store.revoke_session(token)
        except AuthError as exc:
            raise _error(
                404, "SESSION_NOT_FOUND", "session not found or already revoked"
            ) from exc
        return {"revoked": True}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        caller: Identity | None,
        username: str,
        password: str,
        display_name: str = "",
        role: str = "",
        scopes: list[str] | None = None,
        must_change_password: bool = False,
        cert_fingerprints: list[str] | None = None,
        client_ip: str = "",
        bootstrap_token: str = "",
        loopback_peer: bool = False,
    ) -> dict[str, Any]:
        ok, retry = self.store.login_throttle.allow(
            f"register:{client_ip}",
            self.settings.login_throttle_per_min,
            60,
        )
        if not ok:
            raise _error(
                429,
                "RATE_LIMITED",
                "too many registration attempts from this client, slow down",
                headers={"Retry-After": str(retry)},
            )

        username = normalize_username(username)
        scopes = list(scopes or [])
        is_admin = is_caller_admin(caller)
        bootstrap = not is_admin and not self.store.bootstrap_sealed and self.store.count() == 0

        if is_admin:
            if self.store.must_change_password_for_user(caller.subject):
                raise _error(
                    403,
                    "PASSWORD_CHANGE_REQUIRED",
                    "Forbidden: initial or reset password must be changed at /v1/auth/change-password before provisioning users",
                    {"username": caller.subject, "must_change_password": True},
                )
            self.enforce_no_grant_beyond_self(caller, role, scopes)
        elif bootstrap:
            self._verify_bootstrap_gate(bootstrap_token, loopback_peer)
            if role and role != ROLE_ADMIN:
                raise _error(
                    400,
                    "INVALID_BOOTSTRAP_ROLE",
                    "the first account must be created with role 'admin'",
                )
            if scopes:
                raise _error(
                    400,
                    "INVALID_BOOTSTRAP_SCOPES",
                    "custom scopes are not accepted while bootstrapping the first administrator",
                )
            user = self.store.register_bootstrap_admin(username, password, display_name)
            return user.to_summary()
        elif self.settings.self_register:
            if is_privileged_role(role):
                raise _error(403, "FORBIDDEN", "only administrators can assign privileged roles")
            if scopes:
                raise _error(403, "FORBIDDEN", "custom scopes can only be granted by administrator")
            role = "consumer"
            must_change_password = False
            cert_fingerprints = []
        else:
            raise _error(
                403,
                "SELF_REGISTER_DISABLED",
                "self-registration is disabled; ask an administrator to provision the account",
            )

        user = self.store.register(
            username,
            password,
            display_name,
            role,
            scopes,
            must_change_password=must_change_password,
        )
        if cert_fingerprints:
            try:
                user = self.store.set_certificate_fingerprints(user.username, cert_fingerprints)
            except AuthError as exc:
                raise _error(400, "INVALID_CERT_FINGERPRINTS", exc.message) from exc
        return user.to_summary()

    def public_register(
        self,
        *,
        username: str,
        password: str,
        display_name: str = "",
        client_ip: str = "",
    ) -> dict[str, Any]:
        """公开自注册（Go ``handlePublicRegister``）：开关关闭即拒绝，角色恒为 consumer。"""
        ok, retry = self.store.login_throttle.allow(
            f"register:{client_ip}",
            self.settings.login_throttle_per_min,
            60,
        )
        if not ok:
            raise _error(
                429,
                "RATE_LIMITED",
                "too many registration attempts from this client, slow down",
                headers={"Retry-After": str(retry)},
            )
        if not self.settings.self_register:
            raise _error(
                403,
                "SELF_REGISTER_DISABLED",
                "auth: self-service registration is disabled; ask an administrator to create the account",
            )
        user = self.store.register(
            normalize_username(username),
            password,
            display_name,
            "consumer",
            [],
        )
        return user.to_summary()

    def bootstrap_admin(
        self,
        *,
        username: str,
        password: str,
        display_name: str = "",
        role: str = "",
        scopes: list[str] | None = None,
        client_ip: str = "",
        bootstrap_token: str = "",
        loopback_peer: bool = False,
    ) -> dict[str, Any]:
        ok, retry = self.store.login_throttle.allow(
            f"register:{client_ip}",
            self.settings.login_throttle_per_min,
            60,
        )
        if not ok:
            raise _error(
                429,
                "RATE_LIMITED",
                "too many registration attempts from this client, slow down",
                headers={"Retry-After": str(retry)},
            )
        if self.store.bootstrap_sealed or self.store.count() > 0:
            raise _error(
                409,
                "BOOTSTRAP_CLOSED",
                "the first administrator has already been created; ask an administrator to provision this account",
            )
        self._verify_bootstrap_gate(bootstrap_token, loopback_peer)
        if role and role != ROLE_ADMIN:
            raise _error(400, "INVALID_BOOTSTRAP_ROLE", "the first account must be created with role 'admin'")
        if scopes:
            raise _error(
                400,
                "INVALID_BOOTSTRAP_SCOPES",
                "custom scopes are not accepted while bootstrapping the first administrator",
            )
        user = self.store.register_bootstrap_admin(username, password, display_name)
        return user.to_summary()

    def _verify_bootstrap_gate(self, provided_token: str, loopback_peer: bool) -> None:
        if not self.settings.bootstrap_remote_allowed and not loopback_peer:
            raise _error(
                403,
                "BOOTSTRAP_REMOTE_FORBIDDEN",
                "first administrator bootstrap is only allowed from a loopback peer; run it inside the host/pod, "
                "or set PRIVACY_AUTH_USER_BOOTSTRAP_REMOTE_ALLOWED=true for a temporary exception",
            )
        configured = self.settings.bootstrap_token.strip()
        if not configured:
            return
        provided = provided_token.strip()
        if not provided or not hmac.compare_digest(provided.encode(), configured.encode()):
            raise _error(
                401,
                "INVALID_BOOTSTRAP_TOKEN",
                "valid X-Bootstrap-Token header is required for bootstrap",
            )

    def enforce_no_grant_beyond_self(
        self, caller: Identity | None, role: str, scopes: list[str]
    ) -> None:
        if is_full_admin(caller):
            return
        if role and is_privileged_role(role):
            raise _error(
                403,
                "FORBIDDEN",
                "sub-administrator cannot assign privileged roles (admin/auditor)",
            )
        effective = list(scopes)
        if not effective and role.strip():
            effective = default_scopes_for_role(role)
        for scope in effective:
            if caller is None or not caller.has_permission(scope):
                raise _error(
                    403,
                    "FORBIDDEN",
                    f'cannot grant scope "{scope}" exceeding caller\'s own permissions',
                )

    def admin_create_user(
        self,
        *,
        caller: Identity | None,
        username: str,
        password: str,
        display_name: str = "",
        role: str = "",
        scopes: list[str] | None = None,
        must_change_password: bool = False,
        cert_fingerprints: list[str] | None = None,
        admin_code: str = "",
    ) -> dict[str, Any]:
        """管理员受控开户（POST /v1/auth/users），强制 Step-Up 与权限边界。"""
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if self.store.must_change_password_for_user(caller.subject):
            raise _error(
                403,
                "PASSWORD_CHANGE_REQUIRED",
                "Forbidden: initial or reset password must be changed at /v1/auth/change-password before provisioning users",
                {"username": caller.subject, "must_change_password": True},
            )
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "only administrators can provision users")
        username = normalize_username(username)
        self.enforce_admin_step_up(caller.subject, admin_code, "admin_create_user", username)
        if not is_full_admin(caller):
            if role in {ROLE_ADMIN, ROLE_AUDITOR}:
                raise _error(
                    403,
                    "FORBIDDEN",
                    "sub-administrator cannot assign privileged roles (admin/auditor)",
                )
            for scope in scopes or []:
                if not caller.has_permission(scope):
                    raise _error(
                        403,
                        "FORBIDDEN",
                        f'cannot grant scope "{scope}" exceeding caller\'s own permissions',
                    )
        user = self.store.register(
            username,
            password,
            display_name,
            role,
            list(scopes or []),
            must_change_password=must_change_password,
        )
        if cert_fingerprints:
            try:
                user = self.store.set_certificate_fingerprints(user.username, cert_fingerprints)
            except AuthError as exc:
                raise _error(400, "INVALID_CERT_FINGERPRINTS", exc.message) from exc
        return user.to_summary()

    def permission_view(self, caller: Identity | None, offset: int = 0) -> dict[str, Any]:
        """工作负载凭据授权元数据视图（只含元数据，绝无令牌材料）。"""
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller) and not caller.has_permission("user:read"):
            raise _error(403, "FORBIDDEN", "user:read required")
        if offset < 0:
            raise _error(400, "INVALID_ARGUMENT", "invalid offset")
        limit = 100
        users = self.store.list_user_objects(offset, limit)
        views: list[dict[str, Any]] = []
        for user in users:
            if not user.is_workload():
                continue
            if caller.tenant_id and user.tenant_id != caller.tenant_id:
                continue
            for key in user.api_keys.values():
                if not key.active:
                    status = "revoked"
                elif key.is_expired():
                    status = "expired"
                else:
                    status = user.status
                view: dict[str, Any] = {
                    "source": "userstore",
                    "subject": user.username,
                    "name": key.name,
                    "key_id": key.key_id,
                    "tenant_id": user.tenant_id,
                    "scopes": list(key.scopes),
                    "allowed_audiences": list(key.allowed_audiences),
                    "status": status,
                }
                digest = public_key_digest(key.bound_public_key_der)
                if digest:
                    view["bound_public_key_digest"] = digest
                if key.expires_at:
                    view["expires_at"] = key.expires_at
                if user.egress_quota is not None:
                    view["egress_quota"] = user.egress_quota.to_dict()
                views.append(view)
        views.sort(
            key=lambda item: "|".join(
                str(item.get(field, "")) for field in ("source", "subject", "name", "key_id")
            )
        )
        next_offset = offset + limit if len(users) == limit else None
        return {
            "credentials": views,
            "next_offset": next_offset,
            "scope": "workload_credentials",
            "note": "Metadata view only; runtime scopes, audience, PoP and contract checks remain authoritative.",
        }

    # ------------------------------------------------------------------
    # Password lifecycle
    # ------------------------------------------------------------------

    def change_password(
        self,
        *,
        caller: Identity | None,
        username: str,
        old_password: str,
        new_password: str,
        client_ip: str = "",
    ) -> None:
        ok, retry = self.store.login_throttle.allow(
            f"change-password:{client_ip}",
            self.settings.login_throttle_per_min,
            60,
        )
        if not ok:
            raise _error(
                429,
                "RATE_LIMITED",
                "too many change-password attempts from this client, slow down",
                headers={"Retry-After": str(retry)},
            )
        username = normalize_username(username)
        if username.startswith("svc:") or (
            caller is not None
            and (caller.role == ROLE_SERVICE_ACCOUNT or caller.subject.startswith("svc:"))
        ):
            raise _error(403, "FORBIDDEN", "service accounts cannot have interactive passwords")
        if not can_manage_user_account(caller, username):
            raise _error(403, "FORBIDDEN", "permission denied: cannot change password for this user")
        self.check_auditor_protection(caller, username, "change_password")
        self.store.change_password(username, old_password, new_password)

    def reset_password(
        self,
        *,
        caller: Identity | None,
        username: str,
        old_password: str = "",
        new_password: str = "",
        admin_code: str = "",
        client_ip: str = "",
    ) -> dict[str, Any]:
        ok, retry = self.store.login_throttle.allow(
            f"reset-password:{client_ip}",
            self.settings.login_throttle_per_min,
            60,
        )
        if not ok:
            raise _error(
                429,
                "RATE_LIMITED",
                "too many reset-password attempts from this client, slow down",
                headers={"Retry-After": str(retry)},
            )
        username = normalize_username(username)
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "permission denied: requires user:admin")
        self.check_auditor_protection(caller, username, "reset_password")

        is_self = caller is not None and caller.is_subject(username)
        if is_self and not old_password:
            raise _error(
                400,
                "OLD_PASSWORD_REQUIRED",
                "old_password is required when resetting your own password",
            )
        if not is_self:
            self.enforce_admin_step_up(
                caller.subject if caller else "",
                admin_code,
                "reset_password",
                username,
            )
        try:
            temp_password = self.store.reset_password(
                username, old_password if is_self else "", new_password
            )
        except AuthError as exc:
            if exc.code == "INVALID_PASSWORD" and exc.status_code == 401:
                raise _error(401, "INVALID_CREDENTIALS", "invalid old password") from exc
            raise
        return {
            "username": username,
            "temporary_password": temp_password,
            "sessions_revoked": True,
            "api_keys_revoked": True,
        }

    # ------------------------------------------------------------------
    # User administration
    # ------------------------------------------------------------------

    def list_users(self, caller: Identity | None) -> list[dict[str, Any]]:
        if caller is None:
            raise _error(403, "FORBIDDEN", "permission denied: authentication context required")
        if not caller.has_permission("user:read") and not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "permission denied: requires user:read or user:admin")
        return self.store.list_users()

    def get_user(self, caller: Identity | None, username: str) -> dict[str, Any]:
        if not can_view_user_account(caller, username):
            raise _error(403, "FORBIDDEN", "permission denied: cannot access user profile")
        user = self.store.get_user(username)
        return user.to_detail()

    def update_permissions(
        self,
        *,
        caller: Identity | None,
        username: str,
        role: str,
        scopes: list[str] | None,
        admin_code: str = "",
    ) -> dict[str, Any]:
        username = normalize_username(username)
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "permission denied: requires user:admin")
        self.check_auditor_protection(caller, username, "update_permissions")
        if role == ROLE_AUDITOR:
            raise _error(
                403,
                "FORBIDDEN",
                "permission denied: admin cannot assign auditor role (SoD enforcement)",
            )
        self.enforce_no_grant_beyond_self(caller, role, list(scopes or []))
        self.enforce_admin_step_up(
            caller.subject if caller else "", admin_code, "update_permissions", username
        )
        user = self.store.update_permissions(username, role, scopes)
        return _user_payload(user)

    def set_certificates(
        self,
        *,
        caller: Identity | None,
        username: str,
        fingerprints: list[str],
        admin_code: str = "",
    ) -> dict[str, Any]:
        username = normalize_username(username)
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "permission denied: requires user:admin")
        self.check_auditor_protection(caller, username, "set_certificates")
        try:
            normalized = validate_cert_fingerprints(fingerprints)
        except AuthError as exc:
            raise _error(400, "INVALID_CERT_FINGERPRINTS", exc.message) from exc
        self.enforce_admin_step_up(
            caller.subject if caller else "", admin_code, "set_certificates", username
        )
        user = self.store.set_certificate_fingerprints(username, normalized)
        return _user_payload(user)

    def clear_certificates(
        self,
        *,
        caller: Identity | None,
        username: str,
        admin_code: str = "",
    ) -> dict[str, Any]:
        username = normalize_username(username)
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "permission denied: requires user:admin")
        self.check_auditor_protection(caller, username, "clear_certificates")
        self.enforce_admin_step_up(
            caller.subject if caller else "", admin_code, "clear_certificates", username
        )
        user = self.store.clear_certificate_fingerprints(username)
        return _user_payload(user)

    def set_status(
        self,
        *,
        caller: Identity | None,
        username: str,
        status: str,
        reason: str = "",
        admin_code: str = "",
    ) -> dict[str, Any]:
        username = normalize_username(username)
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "permission denied: requires user:admin")
        self.check_auditor_protection(caller, username, "set_status")
        if status not in {STATUS_ACTIVE, STATUS_DISABLED}:
            raise _error(400, "INVALID_STATUS", f'invalid status "{status}"')
        validate_display_name(reason)
        self.enforce_admin_step_up(
            caller.subject if caller else "", admin_code, "set_status", username
        )
        user = self.store.set_status(username, status)
        return _user_payload(user)

    def delete_user(
        self,
        *,
        caller: Identity | None,
        username: str,
        admin_code: str = "",
    ) -> dict[str, Any]:
        username = normalize_username(username)
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "permission denied: requires user:admin")
        self.check_auditor_protection(caller, username, "delete_user")
        target = self.store.get_user(username)
        if target.is_workload():
            raise _error(
                409,
                "WORKLOAD_DELETION_PROHIBITED",
                "workload service accounts cannot be deleted; freeze them instead",
            )
        self.enforce_admin_step_up(
            caller.subject if caller else "", admin_code, "delete_user", username
        )
        self.store.delete_user(username)
        return {"deleted": username}

    # ------------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------------

    def issue_key(
        self,
        *,
        caller: Identity | None,
        username: str,
        key_name: str = "",
        scopes: list[str] | None = None,
        audiences: list[str] | None = None,
        ttl_seconds: int = 0,
        admin_code: str = "",
        client_ip: str = "",
    ) -> dict[str, Any]:
        username = normalize_username(username)
        if not can_manage_user_account(caller, username):
            raise _error(403, "FORBIDDEN", "permission denied: cannot issue key for this user")
        on_behalf = not (caller is not None and caller.is_subject(username))
        throttle_key = f"issue-key:{caller_label(caller)}:{client_ip}"
        ok, retry = self.store.login_throttle.allow(throttle_key, 10, 300)
        if not ok:
            raise _error(
                429,
                "RATE_LIMITED",
                "too many api key issuance attempts, slow down (max 10 per 5m0s)",
                headers={"Retry-After": str(retry)},
            )
        if on_behalf:
            self.check_auditor_protection(caller, username, "issue_key")
            self.enforce_admin_step_up(
                caller.subject if caller else "", admin_code, "issue_key", username
            )
        if ttl_seconds < 0 or ttl_seconds > MAX_API_KEY_TTL_SECONDS:
            raise _error(
                400,
                "INVALID_ARGUMENT",
                f"invalid ttl_seconds: must be between 0 and {MAX_API_KEY_TTL_SECONDS} (90 days)",
            )
        if ttl_seconds > DEFAULT_API_KEY_TTL_SECONDS and not is_caller_admin(caller):
            raise _error(
                403,
                "TTL_PRIVILEGE_REQUIRED",
                f"ttl_seconds above {DEFAULT_API_KEY_TTL_SECONDS} (default 30 days) requires user:admin privilege",
                {"max_ttl_seconds_for_self_service": DEFAULT_API_KEY_TTL_SECONDS},
            )
        record, token = self.store.issue_api_key(
            username, key_name, scopes, audiences, ttl_seconds
        )
        return {
            "key_id": record.key_id,
            "name": record.name,
            "token": token,
            "token_prefix": record.token_prefix,
            "scopes": list(record.scopes),
            "allowed_audiences": list(record.allowed_audiences),
            "created_at": record.created_at,
            "expires_at": record.expires_at,
        }

    def list_keys(self, caller: Identity | None, username: str) -> list[dict[str, Any]]:
        if not can_view_user_account(caller, username):
            raise _error(403, "FORBIDDEN", "permission denied: cannot list keys for this user")
        return self.store.list_api_key_records(username)

    def revoke_key(
        self,
        *,
        caller: Identity | None,
        username: str,
        key_id: str,
        admin_code: str = "",
    ) -> None:
        username = normalize_username(username)
        if not can_manage_user_account(caller, username):
            raise _error(403, "FORBIDDEN", "permission denied: cannot revoke key for this user")
        self.check_auditor_protection(caller, username, "revoke_key")
        self.store.revoke_api_key(username, key_id)

    # ------------------------------------------------------------------
    # MFA lifecycle
    # ------------------------------------------------------------------

    def mfa_setup(
        self,
        *,
        subject: str,
        code: str = "",
        recovery_code: str = "",
        session_auth: bool = False,
        client_ip: str = "",
    ) -> dict[str, Any]:
        self.allow_mfa_write(subject, client_ip)
        setup_state = self.store.mfa_setup_snapshot(subject)
        recovered = False
        rebinding = setup_state.enabled
        previous_binding = ""
        if rebinding:
            previous_binding = self.store.mfa_binding_fingerprint(subject)
            code = code.strip()
            recovery_code = recovery_code.strip()
            if not code and not recovery_code:
                raise _error(
                    403,
                    "MFA_REBIND_PROOF_REQUIRED",
                    'Forbidden: this account already has an active TOTP binding; replacing it requires proof of possession — '
                    'send the current 6-digit code from the bound device as "code", or an unused one-time recovery code as "recovery_code"',
                    {"rebinding": True, "proof_required": True},
                )
            if code:
                try:
                    self.store.verify_totp(subject, code)
                except AuthError as exc:
                    raise mfa_fault(exc) from exc
            else:
                try:
                    consumed = self.store.consume_recovery_code(subject, recovery_code)
                except AuthError as exc:
                    raise mfa_fault(exc) from exc
                if not consumed:
                    raise _error(
                        401,
                        "INVALID_RECOVERY_CODE",
                        "invalid or already consumed recovery code",
                    )
                setup_state = MFASetupState(
                    enabled=False, binding_id="", epoch=setup_state.epoch + 1
                )
                recovered = True

        secret, uri, expires_at = self.store.setup_totp(subject, setup_state)
        response: dict[str, Any] = {
            "setup_id": hash_token(secret),
            "expires_at": isoformat(expires_at),
            "algorithm": default_totp_algorithm(),
            "status": "pending_verification",
            "secret": secret,
            "otpauth": uri,
            "message": "Scan the QR code / import otpauth into your authenticator app, then call /v1/auth/mfa/enable with a valid 6-digit code to activate MFA",
        }
        if recovered and session_auth:
            token, token_expires = self.store.create_session(
                subject,
                MFA_RECOVERY_SESSION_TTL_SECONDS,
                mfa_authenticated=False,
            )
            response["token"] = token
            response["token_type"] = "Bearer"
            response["expires_at"] = isoformat(token_expires)
            response["must_enroll_mfa"] = True
        if rebinding:
            response["rebinding"] = True
            response["mfa_rebound_notice"] = (
                "this request queued a TOTP rebind: your previous binding stays in force until /v1/auth/mfa/enable "
                "verifies a code from the new device. Abandon it and the old device keeps working; completing it "
                "needs a code from the new device, and replacing this pending rebind needs a code from the old one. "
                "If you did not intend to re-enroll, disable MFA and re-enroll immediately, then rotate your password"
            )
            response["message"] = (
                "Scan the QR code / import otpauth into your new authenticator app, then call "
                "/v1/auth/mfa/enable with a valid 6-digit code from it to complete the rebind"
            )
        return response

    def mfa_enable(
        self,
        *,
        subject: str,
        code: str,
        setup_id: str,
        session_auth: bool = False,
        client_ip: str = "",
    ) -> dict[str, Any]:
        self.allow_mfa_write(subject, client_ip)
        try:
            self.store.enable_totp(subject, code, setup_id)
        except AuthError as exc:
            raise mfa_fault(exc) from exc

        response: dict[str, Any] = {
            "status": "enabled",
            "mfa_enabled": True,
            "message": "Multi-Factor Authentication enabled successfully",
        }
        try:
            codes = self.store.issue_recovery_codes(subject)
        except AuthError as exc:
            logger.warning(
                "auth: recovery code issuance failed after mfa enable",
                extra={"username": subject, "error": exc.message},
            )
            response["recovery_codes_error"] = (
                "failed to issue one-time recovery codes; retry via POST /v1/auth/mfa/recovery-codes/regenerate"
            )
        else:
            response["recovery_codes"] = codes
            response["message"] = (
                "Multi-Factor Authentication enabled; store the one-time recovery codes below now — they are the only "
                "self-service fallback if the authenticator device is lost"
            )
        if session_auth:
            token, expires_at = self.store.create_session(
                subject,
                self.settings.session_ttl_seconds or DEFAULT_SESSION_TTL_SECONDS,
                mfa_authenticated=True,
            )
            response["token"] = token
            response["token_type"] = "Bearer"
            response["expires_at"] = isoformat(expires_at)
            response["message"] = (
                "Multi-Factor Authentication enabled; use the returned token for subsequent calls"
            )
        return response

    def mfa_cancel(self, *, subject: str, setup_id: str) -> dict[str, Any]:
        try:
            self.store.cancel_totp_setup(subject, setup_id)
        except AuthError as exc:
            raise mfa_fault(exc) from exc
        return {"status": "cancelled"}

    def mfa_status(self, subject: str) -> dict[str, Any]:
        return self.store.mfa_status(subject)

    def mfa_regenerate_recovery_codes(self, *, subject: str, code: str, client_ip: str = "") -> dict[str, Any]:
        self.allow_mfa_write(subject, client_ip)
        if not self.store.mfa_enabled(subject):
            raise _error(
                403,
                "MFA_NOT_ENABLED",
                "recovery codes are issued together with a TOTP binding; enroll at /v1/auth/mfa/setup + /v1/auth/mfa/enable first",
            )
        try:
            self.store.verify_totp(subject, code)
        except AuthError as exc:
            raise mfa_fault(exc) from exc
        codes = self.store.issue_recovery_codes(subject)
        return {
            "status": "regenerated",
            "recovery_codes": codes,
            "recovery_codes_remaining": len(codes),
            "message": "previous recovery codes are permanently revoked; store the new one-time codes below now",
        }

    def mfa_disable(
        self,
        *,
        caller: Identity,
        username: str = "",
        code: str = "",
        admin_code: str = "",
    ) -> dict[str, Any]:
        target = normalize_username(username) if username else caller.subject
        bypass_with_admin = False
        if target != caller.subject:
            if not caller.has_permission("user:admin") and not is_full_admin(caller):
                raise _error(
                    403,
                    "FORBIDDEN",
                    "only user:admin can disable MFA for other users",
                )
            self.check_auditor_protection(caller, target, "mfa_disable")
            self.enforce_admin_step_up(
                caller.subject, admin_code, "mfa_disable", target
            )
            bypass_with_admin = True
        try:
            self.store.disable_totp(target, code, bypass_with_admin=bypass_with_admin)
        except AuthError as exc:
            raise mfa_fault(exc) from exc
        return {
            "status": "disabled",
            "mfa_enabled": False,
            "message": "Multi-Factor Authentication disabled successfully",
        }

    # ------------------------------------------------------------------
    # Service accounts
    # ------------------------------------------------------------------

    def create_service_account(
        self,
        *,
        caller: Identity | None,
        username: str,
        display_name: str = "",
        tenant_id: str = "",
        scopes: list[str] | None = None,
        allowed_audiences: list[str] | None = None,
        cert_fingerprints: list[str] | None = None,
        egress_quota: EgressQuota | None = None,
        expected_public_key_sha256: str = "",
        expected_spiffe_id: str = "",
        admin_code: str = "",
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if self.store.must_change_password_for_user(caller.subject):
            raise _error(
                403,
                "PASSWORD_CHANGE_REQUIRED",
                "Forbidden: initial or reset password must be changed at /v1/auth/change-password before provisioning service accounts",
                {"username": caller.subject, "must_change_password": True},
            )
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "only administrators can manage service accounts")
        tenant_id = tenant_id or caller.tenant_id
        username = normalize_username(username)
        if not username.startswith("svc:"):
            username = "svc:" + username
        self.enforce_admin_step_up(caller.subject, admin_code, "create_service_account", username)
        scopes = list(scopes or [])
        if not is_full_admin(caller):
            for scope in scopes:
                if not caller.has_permission(scope):
                    raise _error(
                        403,
                        "FORBIDDEN",
                        f'cannot grant scope "{scope}" exceeding caller\'s own permissions',
                    )
        if not allowed_audiences:
            raise _error(
                400,
                "INVALID_ARGUMENT",
                "allowed_audiences is required and cannot be empty",
            )
        try:
            audiences = validate_audiences(
                allowed_audiences, allow_empty=False, allow_wildcard=False
            )
        except AuthError as exc:
            raise _error(400, "INVALID_ARGUMENT", exc.message) from exc
        if expected_public_key_sha256.strip() or expected_spiffe_id.strip():
            user, activation_token = self.store.pre_register_service_account(
                username,
                display_name,
                tenant_id,
                scopes,
                expected_public_key_sha256,
                expected_spiffe_id,
                audiences,
                egress_quota,
            )
            return {
                "service_account": _user_payload(user),
                "activation_token": activation_token,
            }
        user = self.store.create_service_account(
            username,
            display_name,
            tenant_id,
            scopes,
            cert_fingerprints,
            audiences,
            egress_quota,
        )
        return {"service_account": _user_payload(user)}

    def activate_service_account(
        self,
        *,
        username: str,
        activation_token: str,
        actual_spki: str,
        actual_spiffe: str = "",
        tenant_id: str = "",
    ) -> dict[str, Any]:
        """冷启动一次性激活；对端证书公钥由 TSL 终结层提取后传入。"""
        if not (activation_token or "").strip():
            raise _error(400, "INVALID_ARGUMENT", "activation token is required")
        if not (username or "").strip():
            raise _error(400, "INVALID_ARGUMENT", "username is required")
        if not (actual_spki or "").strip():
            raise _error(
                401,
                "POP_TLS_CERT_REQUIRED",
                "mTLS peer certificate required for service account activation",
            )
        record, raw_token = self.store.activate_service_account(
            username,
            activation_token.strip(),
            actual_spki.strip(),
            actual_spiffe,
            tenant_id,
            self.settings.cluster_audience,
        )
        return {
            "key": {
                "key_id": record.key_id,
                "name": record.name,
                "token_hash": record.token_digest,
                "token_prefix": record.token_prefix,
                "scopes": list(record.scopes),
                "allowed_audiences": list(record.allowed_audiences),
                "binding_mode": record.binding_mode,
                "binding_status": record.binding_status,
                "bound_public_key_der": record.bound_public_key_der,
                "active": record.active,
                "created_at": record.created_at,
                "expires_at": record.expires_at,
            },
            "raw_token": raw_token,
        }

    def list_service_accounts(self, caller: Identity | None) -> list[dict[str, Any]]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller) and not caller.has_permission("user:read"):
            raise _error(403, "FORBIDDEN", "insufficient permissions to list service accounts")
        return self.store.list_service_accounts()

    def get_service_account(self, caller: Identity | None, username: str) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller) and not caller.has_permission("user:read"):
            raise _error(403, "FORBIDDEN", "insufficient permissions")
        user = self.store.get_service_account(username)
        return _user_payload(user)

    def freeze_service_account(
        self,
        *,
        caller: Identity | None,
        username: str,
        reason: str = "",
        admin_code: str = "",
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "only administrators can freeze service accounts")
        username = normalize_username(username)
        validate_display_name(reason)
        self.enforce_admin_step_up(
            caller.subject, admin_code, "freeze_service_account", username
        )
        self.store.freeze_service_account(username)
        return {"username": username, "status": "disabled", "frozen": True}

    def unfreeze_service_account(
        self,
        *,
        caller: Identity | None,
        username: str,
        reason: str = "",
        admin_code: str = "",
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "only administrators can unfreeze service accounts")
        username = normalize_username(username)
        validate_display_name(reason)
        self.enforce_admin_step_up(
            caller.subject, admin_code, "unfreeze_service_account", username
        )
        self.store.unfreeze_service_account(username)
        return {"username": username, "status": "active", "frozen": False}

    def issue_service_account_key(
        self,
        *,
        caller: Identity | None,
        username: str,
        key_name: str = "",
        scopes: list[str] | None = None,
        allowed_audiences: list[str] | None = None,
        binding_pubkey: str = "",
        ttl_seconds: int = 0,
        contract: dict[str, Any] | None = None,
        admin_code: str = "",
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "only administrators can issue service account keys")
        username = normalize_username(username)
        self.enforce_admin_step_up(caller.subject, admin_code, "issue_service_account_key", username)
        if not allowed_audiences:
            raise _error(400, "INVALID_ARGUMENT", "allowed_audiences is required and cannot be empty")
        try:
            audiences = validate_audiences(
                allowed_audiences, allow_empty=False, allow_wildcard=False
            )
        except AuthError as exc:
            raise _error(400, "INVALID_ARGUMENT", exc.message) from exc
        binding_pubkey = binding_pubkey.strip()
        if not binding_pubkey:
            raise _error(
                400,
                "BINDING_PUBKEY_REQUIRED",
                "binding.pubkey (base64 SPKI DER of the caller's client certificate) is required: workload keys are always PoP-bound",
            )
        account = self.store.get_service_account(username)
        if caller.tenant_id and caller.tenant_id != account.tenant_id:
            raise _error(403, "FORBIDDEN", "cross-tenant key issuance denied")
        grant_scopes = list(scopes or account.scopes)
        for scope in grant_scopes:
            if not issuance_scope_allowed(caller.scopes, scope):
                raise _error(403, "FORBIDDEN", "key scope exceeds administrator permissions")
        if ttl_seconds < 0 or ttl_seconds > MAX_API_KEY_TTL_SECONDS:
            raise _error(
                400,
                "INVALID_ARGUMENT",
                f"invalid ttl_seconds: must be between 0 and {MAX_API_KEY_TTL_SECONDS} (90 days)",
            )
        record, token = self.store.issue_api_key(
            username,
            key_name,
            grant_scopes,
            audiences,
            ttl_seconds,
            binding_pubkey,
            contract,
        )
        return {
            "key_id": record.key_id,
            "name": record.name,
            "token": token,
            "token_prefix": record.token_prefix,
            "scopes": list(record.scopes),
            "allowed_audiences": list(record.allowed_audiences),
            "contract": dict(record.contract) if record.contract is not None else None,
            "credential_delivered": True,
            "created_at": record.created_at,
            "expires_at": record.expires_at,
        }

    def list_service_account_keys(
        self, caller: Identity | None, username: str
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller) and not caller.has_permission("user:read"):
            raise _error(403, "FORBIDDEN", "insufficient permissions")
        username = normalize_username(username)
        user = self.store.get_user(username)
        records = sorted(user.api_keys.values(), key=lambda record: record.key_id)
        records.sort(key=lambda record: record.created_at, reverse=True)
        keys = [
            {
                "key_id": record.key_id,
                "name": record.name,
                "token_prefix": record.token_prefix,
                "scopes": list(record.scopes),
                "allowed_audiences": list(record.allowed_audiences),
                "active": record.active,
                "created_at": record.created_at,
                "expires_at": record.expires_at,
                "revoked_at": record.revoked_at,
                "contract": None,
            }
            for record in records
        ]
        return {"keys": keys, "count": len(keys)}

    def revoke_service_account_key(
        self,
        *,
        caller: Identity | None,
        username: str,
        key_id: str,
        admin_code: str = "",
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "only administrators can revoke service account keys")
        username = normalize_username(username)
        self.enforce_admin_step_up(
            caller.subject, admin_code, "revoke_service_account_key", username
        )
        self.store.revoke_api_key(username, key_id)
        return {"revoked": True, "key_id": key_id}

    def rotate_service_account_key(
        self,
        *,
        caller: Identity | None,
        username: str,
        key_id: str,
        ttl_seconds: int = 0,
        admin_code: str = "",
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "only administrators can rotate service account keys")
        username = normalize_username(username)
        self.enforce_admin_step_up(
            caller.subject, admin_code, "rotate_service_account_key", username
        )
        if ttl_seconds < 0 or ttl_seconds > MAX_API_KEY_TTL_SECONDS:
            raise _error(
                400,
                "INVALID_ARGUMENT",
                f"invalid ttl_seconds: must be between 0 and {MAX_API_KEY_TTL_SECONDS} (90 days)",
            )
        record, token = self.store.rotate_api_key(username, key_id, ttl_seconds)
        return {
            "key_id": record.key_id,
            "name": record.name,
            "token": token,
            "token_prefix": record.token_prefix,
            "scopes": list(record.scopes),
            "allowed_audiences": list(record.allowed_audiences),
            "created_at": record.created_at,
            "expires_at": record.expires_at,
        }

    def set_service_account_quota(
        self,
        *,
        caller: Identity | None,
        username: str,
        quota: EgressQuota | None,
        admin_code: str = "",
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required")
        if not is_caller_admin(caller):
            raise _error(403, "FORBIDDEN", "only administrators can set service account egress quota")
        username = normalize_username(username)
        self.enforce_admin_step_up(caller.subject, admin_code, "set_service_account_quota", username)
        self.store.set_egress_quota(username, quota)
        return {
            "username": username,
            "egress_quota": quota.to_dict() if quota else None,
        }

    # ------------------------------------------------------------------
    # Admin step-up
    # ------------------------------------------------------------------

    def enforce_admin_step_up(
        self,
        admin_subject: str,
        provided_code: str,
        event: str,
        target: str,
    ) -> None:
        """「管理员操作他人」的特权二次确认，与 Go ``enforceAdminStepUp`` 同口径。"""
        if not admin_subject:
            if self.settings.admin_stepup_required:
                raise _error(
                    403,
                    "ADMIN_STEPUP_SUBJECT_REQUIRED",
                    "privileged operation on another principal requires an accountable human administrator; "
                    "static service credentials cannot perform step-up authentication",
                )
            return
        if not self.settings.admin_stepup_required:
            if not self.store.mfa_enabled(admin_subject):
                return
            if not provided_code:
                raise _error(
                    401,
                    "ADMIN_MFA_REQUIRED",
                    "privileged operation requires administrator step-up authentication; provide X-MFA-Code header or admin_totp_code",
                )
            self._verify_admin_code(admin_subject, provided_code)
            return
        if not self.store.mfa_enabled(admin_subject):
            raise _error(
                401,
                "ADMIN_MFA_REQUIRED",
                "administrator must enroll MFA before performing privileged operations on other users (管理员须先绑定 MFA)",
            )
        if not provided_code:
            raise _error(
                401,
                "ADMIN_MFA_REQUIRED",
                "privileged operation requires administrator step-up authentication; provide X-MFA-Code header or admin_totp_code",
            )
        self._verify_admin_code(admin_subject, provided_code)

    def _verify_admin_code(self, admin_subject: str, provided_code: str) -> None:
        if not self.store.totp_verifiable(admin_subject):
            raise _error(
                503,
                "ADMIN_MFA_SECRET_UNAVAILABLE",
                "administrator step-up is temporarily unavailable: the MFA secret cannot be verified with the current key material",
            )
        try:
            self.store.verify_totp(admin_subject, provided_code)
        except AuthError as exc:
            raise mfa_fault(exc, admin=True) from exc

    def check_auditor_protection(
        self,
        caller: Identity | None,
        username: str,
        action: str = "privileged operation",
    ) -> None:
        """三权分立：管理员不得对 auditor 账号做特权干预；本人豁免。"""
        if caller is not None and caller.is_subject(username):
            return
        try:
            target = self.store.get_user(username)
        except AuthError:
            return
        if target.role == ROLE_AUDITOR:
            raise _error(
                403,
                "FORBIDDEN",
                f"permission denied: admin is strictly forbidden to perform {action} on auditor account (SoD enforcement)",
            )

    # ------------------------------------------------------------------
    # Capability discovery
    # ------------------------------------------------------------------

    def public_capabilities(self) -> dict[str, Any]:
        return {
            "service": self.settings.cluster_issuer or "privacy-engine",
            "api_versions": ["v1"],
            "protocols": ["http", "grpc"],
            "contract_versions": ["hub.dispatch.v1", "hub.dispatch.v2"],
            "tls_modes": ["tls"],
        }

    def negotiate_capabilities(
        self, caller: Identity | None, contract_version: str
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(
                401, "UNAUTHENTICATED", "authentication and pop verification required"
            )
        now = now_utc()
        expires = now + timedelta(hours=1)
        nonce = "n_" + secrets.token_hex(8)
        service = self.settings.cluster_issuer or "privacy-engine"
        manifest = (
            f"{service}|{caller.tenant_id}|{contract_version}|"
            f"{int(now.timestamp())}|{int(expires.timestamp())}|{nonce}"
        )
        digest = "sha256:" + hashlib.sha256(manifest.encode()).hexdigest()
        signature = hashlib.sha256(manifest.encode()).hexdigest()
        return {
            "issuer": f"https://{service}.privshield.internal",
            "service": service,
            "tenant_id": caller.tenant_id,
            "contract_versions": [contract_version],
            "contract_digest": digest,
            "required_scopes": list(caller.scopes),
            "audience": service,
            "pop_binding_required": True,
            "issued_at": isoformat(now),
            "expires_at": isoformat(expires),
            "nonce": nonce,
            "kid": f"key_{service}_sig_2026",
            "signature": signature,
        }

    # ------------------------------------------------------------------
    # Token exchange / revocation
    # ------------------------------------------------------------------

    def token_exchange(
        self,
        caller: Identity | None,
        *,
        grant_type: str,
        client_id: str,
        audience: str,
        subject_token: str = "",
        subject_token_type: str = "",
        scope: str = "",
        ttl_seconds: int = 0,
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "client authentication required")
        caller_name = caller.subject or caller.name
        if normalize_username(caller_name) != normalize_username(client_id):
            raise _error(
                403,
                "CALLER_CLIENT_ID_MISMATCH",
                f'authenticated caller "{caller_name}" does not match requested client_id "{client_id}"',
            )
        audience = audience.strip()
        if not audience:
            raise _error(400, "INVALID_ARGUMENT", "audience is required")
        if self.settings.cluster_audience and audience != self.settings.cluster_audience:
            raise _error(
                403,
                "AUDIENCE_FORBIDDEN",
                f'caller is not authorized to exchange tokens for audience "{audience}"',
            )
        if caller.allowed_audiences and audience not in caller.allowed_audiences:
            raise _error(
                403,
                "AUDIENCE_FORBIDDEN",
                f'caller is not authorized to exchange tokens for audience "{audience}"',
            )
        if grant_type not in {_TOKEN_EXCHANGE_GRANT, "client_credentials"}:
            raise _error(400, "UNSUPPORTED_GRANT_TYPE", f"unsupported grant_type: {grant_type}")

        ttl = ttl_seconds or _DEFAULT_WORKLOAD_TOKEN_TTL_SECONDS
        if ttl <= 0 or ttl > _MAX_WORKLOAD_TOKEN_TTL_SECONDS:
            ttl = _DEFAULT_WORKLOAD_TOKEN_TTL_SECONDS
        requested_scopes = scope.split()
        now = now_utc()
        expires = now + timedelta(seconds=ttl)

        if grant_type == "client_credentials":
            # 静态服务密钥没有 UserStore 档案：与 Go 同口径，查不到即跳过档案校验。
            try:
                user = self.store.get_user(caller_name)
            except AuthError:
                user = None
            if user is not None and user.status != STATUS_ACTIVE:
                raise _error(403, "ACCOUNT_DISABLED", "service account is inactive")
            final_scopes = _intersect_scopes(caller.scopes, requested_scopes) if requested_scopes else list(caller.scopes)
            record = IssuedToken(
                subject=caller_name,
                role=ROLE_SERVICE_ACCOUNT,
                scopes=final_scopes,
                expires_at=expires,
                tenant_id=caller.tenant_id or "default",
                principal_type=PRINCIPAL_WORKLOAD,
                client_id=client_id,
                audience=audience,
                pwd_ver=_password_version(user.password_updated_at if user is not None else None),
            )
        else:
            if subject_token_type and subject_token_type != _ACCESS_TOKEN_TYPE:
                raise _error(
                    400,
                    "INVALID_REQUEST",
                    f"unsupported subject_token_type: {subject_token_type}",
                )
            if not subject_token.strip():
                raise _error(400, "INVALID_REQUEST", "subject_token is required for token-exchange")
            subject_identity = self.resolve_token(subject_token)
            if subject_identity is None:
                raise _error(400, "INVALID_REQUEST", "invalid subject token")
            if subject_identity.key_id and subject_identity.principal_type == "delegated":
                raise _error(400, "INVALID_TARGET", "delegation reselling is not permitted")
            user = self.store.get_user(subject_identity.subject)
            if user.status != STATUS_ACTIVE:
                raise _error(403, "ACCOUNT_DISABLED", "subject user is inactive")
            if "*" in caller.scopes or "admin" in caller.scopes:
                final_scopes = list(subject_identity.scopes)
            else:
                final_scopes = _intersect_scopes(subject_identity.scopes, caller.scopes)
            if requested_scopes:
                final_scopes = _intersect_scopes(final_scopes, requested_scopes)
            record = IssuedToken(
                subject=user.username,
                role=user.role,
                scopes=final_scopes,
                expires_at=expires,
                tenant_id=subject_identity.tenant_id,
                principal_type="delegated",
                client_id=client_id,
                actor=caller_name,
                audience=audience,
                pwd_ver=_password_version(user.password_updated_at),
            )

        token = "psk_" + secrets.token_urlsafe(48).replace("-", "").replace("_", "")[:64]
        record.jti = secrets.token_hex(16)
        with self._token_lock:
            self._issued_tokens[hash_token(token)] = record
        return {
            "access_token": token,
            "issued_token_type": _ACCESS_TOKEN_TYPE,
            "token_type": "Bearer",
            "expires_in": ttl,
            "scope": " ".join(record.scopes),
        }

    def revoke_token(
        self,
        caller: Identity | None,
        *,
        token: str = "",
        token_type_hint: str = "",
    ) -> dict[str, Any]:
        if caller is None:
            raise _error(401, "UNAUTHENTICATED", "authentication required for token revocation")
        token = token.strip()
        if not token:
            raise _error(400, "INVALID_REQUEST", "token is required")
        digest = hash_token(token)
        with self._token_lock:
            record = self._issued_tokens.get(digest)
            if record is None:
                raise _error(404, "NOT_FOUND", "token not found or already revoked")
            if not (
                is_full_admin(caller)
                or caller.is_subject(record.subject)
                or caller.name == record.client_id
            ):
                raise _error(
                    403,
                    "FORBIDDEN",
                    "caller is not permitted to revoke this token",
                )
            if (
                caller.tenant_id
                and record.tenant_id
                and caller.tenant_id != record.tenant_id
            ):
                raise _error(
                    403,
                    "CROSS_TENANT_REVOCATION_FORBIDDEN",
                    "cross-tenant token revocation is forbidden",
                )
            self._issued_tokens.pop(digest, None)
        return {"revoked": True, "status": "revoked"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _login_sandbox_warning(must_change: bool, must_enroll_mfa: bool) -> str:
    if must_change and must_enroll_mfa:
        return (
            "initial or reset password must be changed at /v1/auth/change-password, then TOTP must be enrolled "
            "at /v1/auth/mfa/setup + /v1/auth/mfa/enable, before any business API can be used"
        )
    if must_change:
        return "initial or reset password must be changed at /v1/auth/change-password before any business API can be used"
    if must_enroll_mfa:
        return (
            "TOTP multi-factor authentication is mandatory: enroll at /v1/auth/mfa/setup + /v1/auth/mfa/enable, "
            "then use the session token returned by mfa/enable"
        )
    return ""


def _user_payload(user: Any) -> dict[str, Any]:
    """序列化脱敏用户档案（排除口令哈希与 TOTP 材料）。"""
    payload = user.to_summary()
    payload["password_updated_at"] = isoformat(user.password_updated_at)
    return payload


def _password_version(updated_at: datetime | None) -> int:
    if updated_at is None:
        return 0
    return int(updated_at.timestamp() * 1_000_000_000)


def _intersect_scopes(left: list[str], right: list[str]) -> list[str]:
    if "*" in right:
        return list(left)
    if "*" in left:
        return list(right)
    return [scope for scope in left if scope in right]
