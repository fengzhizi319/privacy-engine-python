"""Thread-safe user/session/API-key store mirroring Go ``pkg/auth.UserStore``.

权威语义（与 Go 侧对齐）：
- 口令主校验走 SM3 v2 拉伸哈希，bcrypt 仅用于历史口令防重用；
- 连续 5 次失败锁定 15 分钟，锁定态只向能自证口令的本人披露；
- 会话为内存态（重启即失效），API Key 只落 SM3 摘要；
- MFA 绑定、恢复码、epoch 与会话失效联动；
- 可选 SQLite 权威持久化（写穿 + 启动装载），表结构复用
  ``deploy/sql/auth`` 的跨方言 DDL。
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .bound_pubkey import format_bound_public_key_der, parse_bound_public_key_der
from .config import AuthSettings
from .errors import AuthError, invalid_argument, not_found
from .models import (
    ACCOUNT_LOCKOUT_SECONDS,
    API_KEY_ISSUE_THROTTLE_MAX_PER_CALLER,
    API_KEY_ISSUE_THROTTLE_WINDOW_SECONDS,
    DEFAULT_API_KEY_TTL_SECONDS,
    DEFAULT_SESSION_TTL_SECONDS,
    LOGIN_THROTTLE_MAX_PER_IP,
    LOGIN_THROTTLE_WINDOW_SECONDS,
    MAX_API_KEYS_PER_USER,
    MAX_API_KEY_TTL_SECONDS,
    MAX_FAILED_LOGIN_ATTEMPTS,
    MAX_SCOPE_COUNT,
    MAX_SESSIONS_PER_USER,
    MFA_RECOVERY_SESSION_TTL_SECONDS,
    MFA_SETUP_TTL_SECONDS,
    PASSWORD_HISTORY_DEPTH,
    PRINCIPAL_USER,
    PRINCIPAL_WORKLOAD,
    ROLE_ADMIN,
    ROLE_AUDITOR,
    ROLE_SERVICE_ACCOUNT,
    STATUS_ACTIVE,
    STATUS_DISABLED,
    STATUS_PENDING_ACTIVATION,
    WORKLOAD_NON_INTERACTIVE_LOCKED,
    APIKeyRecord,
    EgressQuota,
    User,
    default_scopes_for_role,
    is_privileged_role,
    is_valid_role,
    is_valid_username,
    isoformat,
    normalize_username,
    now_utc,
    parse_iso,
    validate_display_name,
    validate_key_name,
    validate_scopes,
)
from .password import (
    hash_password_bcrypt,
    hash_password_sm3,
    validate_password_strength,
    verify_password_bcrypt,
    verify_password_sm3,
)
from .recovery import (
    RecoveryCodeEntry,
    generate_recovery_codes,
    hash_recovery_code,
    is_valid_recovery_code,
    normalize_recovery_code,
)
from .security import sm3_hex
from .totp import default_totp_algorithm, generate_secret, generate_uri, validate_code

AUDIT_SCOPE_PREFIX = "audit:"
_MANAGEMENT_SCOPES = {"*", "admin", "user:admin", "ops:admin", "hub:admin"}
_ADMIN_FORBIDDEN_SCOPES = {"audit:read", "audit:verify", "audit:export", "audit:*"}
_AUDITOR_FORBIDDEN_SCOPES = {
    "*", "admin", "user:admin", "hub:admin", "ops:admin", "hub:dispatch", "hub:dispatch:*",
}
_AUDIT_SCOPES = {"audit:read", "audit:verify", "audit:export", "audit:*"}

_CERT_FINGERPRINT_LEN = 64


# ---------------------------------------------------------------------------
# Token / error helpers
# ---------------------------------------------------------------------------


def generate_secure_token() -> str:
    """与 Go ``GenerateSecureToken`` 一致：``psk_`` + 16 字节随机 hex。"""
    return "psk_" + secrets.token_hex(16)


def generate_key_id() -> str:
    return "key_" + secrets.token_hex(8)


def token_prefix(token: str) -> str:
    return "***" if len(token) <= 4 else token[:4] + "***"


def hash_token(token: str) -> str:
    """凭据摘要统一走国密 SM3（与 Go ``HashTokenSM3`` 同口径）。"""
    return sm3_hex(token)


def generate_activation_token() -> str:
    return "act_psk_" + secrets.token_hex(16)


def _error(status: int, code: str, message: str, detail: Any = None, headers: dict[str, str] | None = None) -> AuthError:
    return AuthError(status, code, message, detail, headers=headers)


def _user_not_found(username: str = "") -> AuthError:
    detail = {"username": username} if username else None
    return _error(404, "USER_NOT_FOUND", "user not found", detail)


def _invalid_credentials() -> AuthError:
    return _error(401, "UNAUTHENTICATED", "invalid username or password")


def _invalid_password(message: str = "invalid password") -> AuthError:
    return _error(401, "INVALID_PASSWORD", message)


def _account_locked(retry_after: int) -> AuthError:
    seconds = max(1, retry_after)
    return _error(
        429,
        "ACCOUNT_LOCKED",
        "account is temporarily locked due to too many failed attempts",
        {"retry_after_seconds": seconds},
        headers={"Retry-After": str(seconds)},
    )


def _invalid_mfa_code() -> AuthError:
    return _error(400, "INVALID_MFA_CODE", "invalid verification code")


def _mfa_setup_changed() -> AuthError:
    return _error(
        409,
        "MFA_SETUP_CHANGED",
        "restart MFA setup; the pending binding was changed, expired, or consumed",
    )


def _mfa_unavailable() -> AuthError:
    return _error(503, "MFA_VERIFICATION_UNAVAILABLE", "MFA verification infrastructure is temporarily unavailable")


def _forbidden_scope(message: str) -> AuthError:
    return _error(403, "FORBIDDEN_SCOPE", message)


def _conflict(code: str, message: str, detail: Any = None) -> AuthError:
    return _error(409, code, message, detail)


# ---------------------------------------------------------------------------
# Throttle / session entries
# ---------------------------------------------------------------------------


class FixedWindowThrottle:
    """进程内固定窗口限速（登录、注册、密钥签发共用）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, int]] = {}

    def allow(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0
        now = now_utc().timestamp()
        with self._lock:
            started, count = self._entries.get(key, (now, 0))
            if now - started >= window_seconds:
                started, count = now, 0
            count += 1
            self._entries[key] = (started, count)
            if count > limit:
                return False, max(1, int(window_seconds - (now - started)) + 1)
            return True, 0

    def reset(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)


@dataclass
class SessionEntry:
    username: str
    scopes: list[str]
    mfa_authenticated: bool
    mfa_epoch: int
    created_at: datetime
    expires_at: datetime


@dataclass
class AuthResult:
    """Result of password authentication (MFA is a separate step)."""

    user: User
    mfa_enabled: bool
    totp_verifiable: bool


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------


def has_admin_capability(user: User | None) -> bool:
    if user is None or user.status != STATUS_ACTIVE:
        return False
    if user.role == ROLE_ADMIN:
        return True
    return any(scope in {"*", "user:admin"} for scope in user.scopes)


def issuance_scope_allowed(owner_scopes: list[str], requested: str) -> bool:
    if requested in owner_scopes:
        return True
    if requested.startswith(AUDIT_SCOPE_PREFIX):
        return False
    return any(scope in {"*", "admin"} for scope in owner_scopes)


def validate_role_scope_consistency(role: str, scopes: list[str]) -> None:
    if role == ROLE_ADMIN:
        for scope in scopes:
            if scope in _ADMIN_FORBIDDEN_SCOPES or scope.startswith(AUDIT_SCOPE_PREFIX):
                raise _forbidden_scope(
                    f'role "admin" cannot hold audit scope "{scope}" '
                    "(Separation of Duties violation: admin cannot inspect/tamper audit logs)"
                )
        return
    if role == ROLE_AUDITOR:
        for scope in scopes:
            if scope in _AUDITOR_FORBIDDEN_SCOPES:
                raise _forbidden_scope(
                    f'role "auditor" cannot hold administrative/dispatch scope "{scope}" '
                    "(Separation of Duties violation: auditor cannot hold admin or dispatch rights)"
                )
            if scope.endswith(":write") or scope == "privacy:budget":
                raise _forbidden_scope(
                    f'role "auditor" cannot hold write scope "{scope}" '
                    "(Separation of Duties violation: auditor is read-only)"
                )
        return
    for scope in scopes:
        if scope in _MANAGEMENT_SCOPES:
            raise _forbidden_scope(f'role "{role}" cannot hold management scope "{scope}"')
        if scope in _AUDIT_SCOPES or scope.startswith(AUDIT_SCOPE_PREFIX):
            raise _forbidden_scope(f'role "{role}" cannot hold audit scope "{scope}" (auditor only)')


def clamp_scopes_to_owner(key_scopes: list[str], owner_scopes: list[str]) -> list[str]:
    clamped = [scope for scope in key_scopes if issuance_scope_allowed(owner_scopes, scope)]
    return clamped if clamped != key_scopes else list(key_scopes)


def validate_audiences(audiences: list[str], *, allow_empty: bool, allow_wildcard: bool = False) -> list[str]:
    if not audiences:
        if allow_empty:
            return []
        raise invalid_argument("allowed_audiences cannot be empty")
    if len(audiences) > 32:
        raise invalid_argument("too many audiences")
    normalized: list[str] = []
    for raw in audiences:
        value = str(raw).strip()
        if not value or len(value) > 64:
            raise invalid_argument("invalid audience format")
        if value == "*" and not allow_wildcard:
            raise invalid_argument("wildcard '*' audience is strictly prohibited for service account keys")
        allowed_extra = ":_-.*" if allow_wildcard else ":_-."
        if not all(char.isascii() and (char.isalnum() or char in allowed_extra) for char in value):
            raise invalid_argument("invalid characters in audience")
        if value not in normalized:
            normalized.append(value)
    return normalized


def validate_cert_fingerprints(fingerprints: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw in fingerprints:
        value = str(raw).strip().lower().replace(":", "")
        if len(value) != _CERT_FINGERPRINT_LEN:
            raise invalid_argument("certificate fingerprint must be a SHA-256 hex digest (64 hex characters)")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise invalid_argument("certificate fingerprint must be hexadecimal") from exc
        if value not in normalized:
            normalized.append(value)
    return normalized


def check_password_expiry_sandbox_user(user: User, *, block: bool, grace_days: int) -> bool:
    """True 表示业务访问仍被强制改密沙箱阻断。"""
    if user.must_change_password:
        return True
    if not block:
        return False
    grace = max(0, grace_days) * 24 * 3600
    age = (now_utc() - user.password_updated_at).total_seconds()
    return age > 90 * 24 * 3600 + grace


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class UserStore:
    """In-memory authoritative store with optional write-through persistence.

    ``repository`` 仅负责持久化（``load_users`` / ``save_user`` / ``delete_user``）；
    所有裁决均在本类锁内完成，与 Go 内存档位同语义。
    """

    def __init__(self, settings: AuthSettings | None = None, repository: Any = None) -> None:
        self.settings = settings or AuthSettings()
        self.repository = repository
        self._lock = threading.RLock()
        self.users: dict[str, User] = {}
        self.sessions: dict[str, SessionEntry] = {}
        self.active_keys: dict[str, tuple[str, str]] = {}
        self.failed_attempts: dict[str, int] = {}
        self.locked_until: dict[str, datetime] = {}
        self.totp_replay: dict[str, int] = {}
        self.revoked_jti: dict[str, datetime] = {}
        self.bootstrap_sealed = False
        self.login_throttle = FixedWindowThrottle()
        self.issue_throttle = FixedWindowThrottle()
        self._dummy_sm3_hash = ""
        if repository is not None:
            loaded = repository.load_users()
            self.users = loaded
            self.bootstrap_sealed = bool(loaded)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist(self, user: User) -> None:
        if self.repository is not None:
            self.repository.save_user(user)

    def _persist_delete(self, username: str) -> None:
        if self.repository is not None:
            self.repository.delete_user(username)

    def _dummy_hash(self) -> str:
        if not self._dummy_sm3_hash:
            self._dummy_sm3_hash = hash_password_sm3(
                secrets.token_hex(16), iterations=self.settings.sm3_iterations
            )
        return self._dummy_sm3_hash

    def _dummy_verify(self, password: str) -> None:
        # 与真实校验同构（同算法、同迭代轮数），抑制基于时序的账号枚举。
        verify_password_sm3(self._dummy_hash(), password)

    def is_locked(self, username: str) -> bool:
        with self._lock:
            return self._lock_remaining(normalize_username(username)) > 0

    def lockout_remaining(self, username: str) -> int:
        with self._lock:
            return self._lock_remaining(normalize_username(username))

    def _lock_remaining(self, username: str) -> int:
        until = self.locked_until.get(username)
        if until is None:
            return 0
        remaining = (until - now_utc()).total_seconds()
        if remaining <= 0:
            self.locked_until.pop(username, None)
            self.failed_attempts.pop(username, None)
            return 0
        return int(remaining) + 1

    def _record_failure_locked(self, username: str) -> None:
        if len(self.locked_until) > 2048:
            now = now_utc()
            for key in [key for key, until in self.locked_until.items() if now >= until]:
                self.locked_until.pop(key, None)
        if len(self.failed_attempts) > 4096:
            for key in [key for key in self.failed_attempts if key not in self.users][:2048]:
                self.failed_attempts.pop(key, None)
        count = self.failed_attempts.get(username, 0) + 1
        self.failed_attempts[username] = count
        if count >= MAX_FAILED_LOGIN_ATTEMPTS:
            self.locked_until[username] = now_utc() + timedelta(seconds=ACCOUNT_LOCKOUT_SECONDS)

    def _reset_failures_locked(self, username: str) -> None:
        self.failed_attempts.pop(username, None)
        self.locked_until.pop(username, None)

    def _check_mfa_lock_locked(self, username: str) -> None:
        remaining = self._lock_remaining(username)
        if remaining > 0:
            raise _account_locked(remaining)

    def _fail_mfa_locked(self, username: str) -> None:
        self._record_failure_locked(username)
        remaining = self._lock_remaining(username)
        if remaining > 0:
            raise _account_locked(remaining)
        raise _invalid_mfa_code()

    def _get_locked(self, username: str) -> User:
        user = self.users.get(username)
        if user is None:
            raise _user_not_found(username)
        return user

    def _ensure_active(self, user: User) -> None:
        if user.status != STATUS_ACTIVE:
            raise _error(403, "ACCOUNT_DISABLED", "account is disabled")

    def _prune_expired_sessions_locked(self) -> None:
        now = now_utc()
        for digest in [digest for digest, entry in self.sessions.items() if now >= entry.expires_at]:
            self.sessions.pop(digest, None)

    def _revoke_sessions_locked(self, username: str) -> int:
        digests = [digest for digest, entry in self.sessions.items() if entry.username == username]
        for digest in digests:
            self.sessions.pop(digest, None)
        return len(digests)

    def _prune_expired_keys_locked(self, user: User) -> None:
        for key_id in [key_id for key_id, record in user.api_keys.items() if record.is_expired()]:
            record = user.api_keys.pop(key_id)
            self.active_keys.pop(record.token_digest, None)

    def _ensure_not_last_admin_locked(self, target: User) -> None:
        if not has_admin_capability(target):
            return
        others = [
            user for user in self.users.values()
            if user.username != target.username and has_admin_capability(user)
        ]
        if not others:
            raise _conflict("LAST_ADMIN", "operation would remove the last active administrator")

    def _ensure_fingerprints_unique_locked(self, username: str, fingerprints: list[str]) -> None:
        for other in self.users.values():
            if other.username == username:
                continue
            if any(fingerprint in other.cert_fingerprints for fingerprint in fingerprints):
                raise _conflict(
                    "CERT_FINGERPRINT_CONFLICT",
                    "certificate fingerprint is already bound to another account",
                )

    # ------------------------------------------------------------------
    # User lifecycle
    # ------------------------------------------------------------------

    def count(self) -> int:
        with self._lock:
            return len(self.users)

    def get_user(self, username: str) -> User:
        with self._lock:
            return self._get_locked(normalize_username(username)).sanitized_copy()

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock:
            return [user.to_summary() for user in sorted(self.users.values(), key=lambda item: item.username)]

    def list_user_objects(self, offset: int = 0, limit: int = 0) -> list[User]:
        """权限视图分页：按主体名排序，返回脱敏用户副本。"""
        with self._lock:
            names = sorted(self.users)
            if offset < 0:
                offset = 0
            selected = names[offset : offset + limit] if limit else names[offset:]
            return [self.users[name].sanitized_copy() for name in selected]

    def exists(self, username: str) -> bool:
        with self._lock:
            return normalize_username(username) in self.users

    def register(
        self,
        username: str,
        password: str,
        display_name: str = "",
        role: str = "",
        scopes: list[str] | None = None,
        *,
        must_change_password: bool = False,
        bootstrap_only: bool = False,
    ) -> User:
        username = normalize_username(username)
        if not is_valid_username(username):
            raise invalid_argument("username must be 3-32 characters of letters, digits, '_' or '-'")
        try:
            validate_password_strength(password, username)
        except AuthError as exc:
            raise _error(400, "WEAK_PASSWORD", exc.message) from exc
        validate_display_name(display_name)
        role = role or "consumer"
        if not is_valid_role(role):
            raise _error(400, "INVALID_ROLE", f'invalid role "{role}"', {"known_roles": list(_known_roles())})
        final_scopes = validate_scopes(scopes or [])
        if not final_scopes:
            final_scopes = default_scopes_for_role(role)
        validate_role_scope_consistency(role, final_scopes)

        # 重活（bcrypt + SM3）在锁外执行，避免注册风暴阻塞认证热路径。
        bcrypt_hash = hash_password_bcrypt(password, cost=self.settings.bcrypt_cost)
        sm3_hash = hash_password_sm3(password, iterations=self.settings.sm3_iterations)

        with self._lock:
            if bootstrap_only and (self.bootstrap_sealed or self.users):
                raise _conflict(
                    "BOOTSTRAP_CLOSED",
                    "the first administrator has already been created; ask an administrator to provision this account",
                )
            if username in self.users:
                raise _conflict("USER_EXISTS", "username already exists")
            now = now_utc()
            user = User(
                username=username,
                display_name=display_name or username,
                role=role,
                scopes=final_scopes,
                status=STATUS_ACTIVE,
                password_hash=bcrypt_hash,
                password_sm3_hash=sm3_hash,
                password_updated_at=now,
                must_change_password=must_change_password,
                principal_type=PRINCIPAL_USER,
                created_at=now,
                updated_at=now,
            )
            self.users[username] = user
            self.bootstrap_sealed = True
            try:
                self._persist(user)
            except Exception:
                self.users.pop(username, None)
                raise
            return user.sanitized_copy()

    def register_bootstrap_admin(self, username: str, password: str, display_name: str = "") -> User:
        if self.bootstrap_sealed or self.count() > 0:
            raise _conflict(
                "BOOTSTRAP_CLOSED",
                "the first administrator has already been created; ask an administrator to provision this account",
            )
        return self.register(
            username,
            password,
            display_name,
            ROLE_ADMIN,
            [],
            must_change_password=True,
            bootstrap_only=True,
        )

    def authenticate(self, username: str, password: str) -> AuthResult:
        username = normalize_username(username)
        remaining = self._lock_remaining(username)
        if remaining > 0:
            # 锁定态只向能自证口令的本人披露，避免账号枚举预言机。
            with self._lock:
                user = self.users.get(username)
                sm3_hash = user.password_sm3_hash if user else ""
            if sm3_hash and verify_password_sm3(sm3_hash, password):
                raise _account_locked(remaining)
            if not sm3_hash:
                self._dummy_verify(password)
            raise _invalid_credentials()

        with self._lock:
            user = self.users.get(username)
            if user is None:
                sm3_hash, status = "", ""
            else:
                sm3_hash, status = user.password_sm3_hash, user.status

        if not user or not sm3_hash:
            self._dummy_verify(password)
            raise _invalid_credentials()
        if not verify_password_sm3(sm3_hash, password):
            with self._lock:
                self._record_failure_locked(username)
            raise _invalid_credentials()
        if status != STATUS_ACTIVE:
            raise _error(403, "ACCOUNT_DISABLED", "account is disabled")

        with self._lock:
            current = self.users.get(username)
            if current is None:
                raise _invalid_credentials()
            if not current.mfa_enabled and not current.pending_totp_id:
                self._reset_failures_locked(username)
            return AuthResult(
                user=current.sanitized_copy(),
                mfa_enabled=current.mfa_enabled,
                totp_verifiable=not (current.totp_secret_undecryptable or not current.totp_secret),
            )

    def update_permissions(self, username: str, role: str, scopes: list[str] | None) -> User:
        username = normalize_username(username)
        if role and not is_valid_role(role):
            raise _error(400, "INVALID_ROLE", f'invalid role "{role}"', {"known_roles": list(_known_roles())})
        normalized = validate_scopes(scopes or [])
        with self._lock:
            user = self._get_locked(username)
            next_role = role or user.role
            next_scopes = normalized or (default_scopes_for_role(next_role) if role else list(user.scopes))
            validate_role_scope_consistency(next_role, next_scopes)
            if has_admin_capability(user):
                probe = User(username=user.username, role=next_role, scopes=next_scopes, status=user.status)
                if not has_admin_capability(probe):
                    self._ensure_not_last_admin_locked(user)
            changed = next_role != user.role or next_scopes != user.scopes
            user.role = next_role
            user.scopes = list(next_scopes)
            user.updated_at = now_utc()
            for record in user.api_keys.values():
                record.scopes = clamp_scopes_to_owner(record.scopes, user.scopes)
            for entry in self.sessions.values():
                if entry.username == username:
                    entry.scopes = list(user.scopes)
            # 角色变更后刷新主体类型（工作负载账号仍由 svc: 前缀判定）。
            if not user.is_workload():
                user.principal_type = PRINCIPAL_USER
            self._persist(user)
            if changed:
                self._emit_event("update_permissions", user.username, {"role": user.role, "scopes": user.scopes})
            return user.sanitized_copy()

    def set_status(self, username: str, status: str) -> User:
        username = normalize_username(username)
        if status not in {STATUS_ACTIVE, STATUS_DISABLED}:
            raise _error(400, "INVALID_STATUS", f'invalid status "{status}"')
        with self._lock:
            user = self._get_locked(username)
            if status == STATUS_DISABLED:
                self._ensure_not_last_admin_locked(user)
            user.status = status
            user.updated_at = now_utc()
            if status == STATUS_DISABLED:
                for record in user.api_keys.values():
                    record.active = False
                    record.revoked_at = isoformat(now_utc())
                    self.active_keys.pop(record.token_digest, None)
                self._revoke_sessions_locked(username)
            self._persist(user)
            self._emit_event("set_status", user.username, {"status": status})
            return user.sanitized_copy()

    def delete_user(self, username: str) -> None:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            if user.is_workload():
                raise _conflict("WORKLOAD_DELETION_PROHIBITED", "workload service accounts cannot be deleted; freeze them instead")
            self._ensure_not_last_admin_locked(user)
            for record in user.api_keys.values():
                self.active_keys.pop(record.token_digest, None)
            self._revoke_sessions_locked(username)
            self.users.pop(username, None)
            self.failed_attempts.pop(username, None)
            self.locked_until.pop(username, None)
            self._persist_delete(username)
            self._emit_event("delete_user", username, {})

    def set_certificate_fingerprints(self, username: str, fingerprints: list[str]) -> User:
        username = normalize_username(username)
        normalized = validate_cert_fingerprints(fingerprints)
        with self._lock:
            user = self._get_locked(username)
            self._ensure_fingerprints_unique_locked(username, normalized)
            user.cert_fingerprints = normalized
            user.updated_at = now_utc()
            self._persist(user)
            return user.sanitized_copy()

    def clear_certificate_fingerprints(self, username: str) -> User:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            user.cert_fingerprints = []
            user.updated_at = now_utc()
            self._persist(user)
            return user.sanitized_copy()

    def find_user_by_certificate_fingerprint(self, fingerprint: str) -> User | None:
        value = str(fingerprint).strip().lower().replace(":", "")
        with self._lock:
            for user in self.users.values():
                if user.status == STATUS_ACTIVE and value in user.cert_fingerprints:
                    return user.sanitized_copy()
        return None

    # ------------------------------------------------------------------
    # Password lifecycle
    # ------------------------------------------------------------------

    def change_password(self, username: str, old_password: str, new_password: str) -> None:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            if user.role == ROLE_SERVICE_ACCOUNT or user.password_hash == WORKLOAD_NON_INTERACTIVE_LOCKED:
                raise _error(403, "SERVICE_ACCOUNT_NO_PASSWORD", "service accounts cannot have interactive passwords")
            current_sm3 = user.password_sm3_hash
            history = list(user.password_history)

        if not current_sm3:
            raise _invalid_password()
        if not verify_password_sm3(current_sm3, old_password):
            raise _invalid_password("old password does not match")
        if old_password == new_password:
            raise _error(400, "INVALID_PASSWORD", "new password must differ from the current password")
        try:
            validate_password_strength(new_password, username)
        except AuthError as exc:
            raise _error(400, "INVALID_PASSWORD", exc.message) from exc
        for encoded in history:
            if encoded and verify_password_bcrypt(encoded, new_password):
                raise _error(400, "INVALID_PASSWORD", "password was used recently and cannot be reused")

        new_bcrypt = hash_password_bcrypt(new_password, cost=self.settings.bcrypt_cost)
        new_sm3 = hash_password_sm3(new_password, iterations=self.settings.sm3_iterations)

        with self._lock:
            user = self._get_locked(username)
            if user.password_sm3_hash != current_sm3:
                raise _conflict(
                    "PASSWORD_CHANGED_CONCURRENTLY",
                    "password was changed concurrently; retry the request",
                )
            for encoded in user.password_history:
                if encoded and verify_password_bcrypt(encoded, new_password):
                    raise _error(400, "INVALID_PASSWORD", "password was used recently and cannot be reused")
            user.password_history = [user.password_hash, *user.password_history][:PASSWORD_HISTORY_DEPTH]
            user.password_hash = new_bcrypt
            user.password_sm3_hash = new_sm3
            user.password_updated_at = now_utc()
            user.must_change_password = False
            user.updated_at = now_utc()
            self._revoke_sessions_locked(username)
            self._persist(user)
            self._emit_event("change_password", username, {"sessions_revoked": True})

    def reset_password(self, username: str, old_password: str = "", new_password: str = "") -> str:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            if old_password == "" and user.role == ROLE_AUDITOR:
                raise _error(
                    403,
                    "AUDITOR_PROTECTED",
                    "auditor passwords cannot be reset by administrators (separation of duties)",
                )
            target_sm3 = user.password_sm3_hash
            if user.role == ROLE_SERVICE_ACCOUNT or user.password_hash == WORKLOAD_NON_INTERACTIVE_LOCKED:
                raise _error(403, "SERVICE_ACCOUNT_NO_PASSWORD", "service accounts cannot have interactive passwords")
            history = list(user.password_history)

        if old_password:
            if not target_sm3 or not verify_password_sm3(target_sm3, old_password):
                raise _invalid_password("invalid old password")
        if not new_password.strip():
            new_password = generate_temporary_password(username)
        else:
            try:
                validate_password_strength(new_password, username)
            except AuthError as exc:
                raise _error(400, "INVALID_PASSWORD", exc.message) from exc
        for encoded in history:
            if encoded and verify_password_bcrypt(encoded, new_password):
                raise _error(400, "INVALID_PASSWORD", "password was used recently and cannot be reused")

        new_bcrypt = hash_password_bcrypt(new_password, cost=self.settings.bcrypt_cost)
        new_sm3 = hash_password_sm3(new_password, iterations=self.settings.sm3_iterations)

        with self._lock:
            user = self._get_locked(username)
            user.password_history = [user.password_hash, *user.password_history][:PASSWORD_HISTORY_DEPTH]
            user.password_hash = new_bcrypt
            user.password_sm3_hash = new_sm3
            user.password_updated_at = now_utc()
            user.must_change_password = True
            user.updated_at = now_utc()
            self._revoke_sessions_locked(username)
            for record in user.api_keys.values():
                self.active_keys.pop(record.token_digest, None)
            user.api_keys.clear()
            self._reset_failures_locked(username)
            self._persist(user)
            self._emit_event("reset_password", username, {"sessions_revoked": True, "api_keys_revoked": True})
        return new_password

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, username: str, ttl: int = 0, *, mfa_authenticated: bool = False) -> tuple[str, datetime]:
        username = normalize_username(username)
        ttl = ttl or DEFAULT_SESSION_TTL_SECONDS
        ttl = min(ttl, DEFAULT_SESSION_TTL_SECONDS)
        token = generate_secure_token()
        digest = hash_token(token)
        with self._lock:
            user = self._get_locked(username)
            self._ensure_active(user)
            self._prune_expired_sessions_locked()
            existing = [entry for entry in self.sessions.values() if entry.username == username]
            if len(existing) >= MAX_SESSIONS_PER_USER:
                oldest = min(existing, key=lambda entry: entry.created_at)
                for stored_digest, entry in list(self.sessions.items()):
                    if entry is oldest:
                        self.sessions.pop(stored_digest, None)
                        break
            now = now_utc()
            expires_at = now + timedelta(seconds=ttl)
            self.sessions[digest] = SessionEntry(
                username=username,
                scopes=list(user.scopes),
                mfa_authenticated=mfa_authenticated,
                mfa_epoch=user.mfa_epoch,
                created_at=now,
                expires_at=expires_at,
            )
            return token, expires_at

    def lookup_session(self, token: str) -> tuple[User, SessionEntry] | None:
        if not token:
            return None
        digest = hash_token(token)
        with self._lock:
            entry = self.sessions.get(digest)
            if entry is None:
                return None
            if now_utc() >= entry.expires_at:
                self.sessions.pop(digest, None)
                return None
            user = self.users.get(entry.username)
            if user is None or user.mfa_epoch != entry.mfa_epoch or user.status != STATUS_ACTIVE:
                self.sessions.pop(digest, None)
                return None
            return user.sanitized_copy(), entry

    def revoke_session(self, token: str) -> None:
        digest = hash_token(token)
        with self._lock:
            if digest not in self.sessions:
                raise _error(401, "UNAUTHENTICATED", "session not found")
            self.sessions.pop(digest, None)

    def revoke_user_sessions(self, username: str) -> int:
        username = normalize_username(username)
        with self._lock:
            user = self.users.get(username)
            if user is not None:
                # 与 Go 一致：推进口令时间戳，使跨 Pod 集群会话立即失效。
                user.password_updated_at = now_utc()
                self._persist(user)
            return self._revoke_sessions_locked(username)

    # ------------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------------

    def issue_api_key(
        self,
        username: str,
        name: str = "",
        scopes: list[str] | None = None,
        audiences: list[str] | None = None,
        ttl: int = 0,
        bound_public_key_der: str = "",
        contract: dict[str, Any] | None = None,
    ) -> tuple[APIKeyRecord, str]:
        username = normalize_username(username)
        if name:
            validate_key_name(name)
        normalized_scopes = validate_scopes(scopes or [])
        if ttl < 0 or ttl > MAX_API_KEY_TTL_SECONDS:
            raise _error(
                400,
                "INVALID_TTL",
                "invalid API key TTL",
                {"default_ttl_seconds": DEFAULT_API_KEY_TTL_SECONDS, "max_ttl_seconds": MAX_API_KEY_TTL_SECONDS},
            )
        ttl = ttl or DEFAULT_API_KEY_TTL_SECONDS
        is_workload = username.startswith("svc:")
        if is_workload and not bound_public_key_der:
            raise _error(
                400,
                "BINDING_PUBKEY_REQUIRED",
                "workload service account keys must be bound to a client certificate public key",
            )
        normalized_audiences = validate_audiences(
            audiences or [], allow_empty=not is_workload, allow_wildcard=False
        )

        token = generate_secure_token()
        key_id = generate_key_id()
        digest = hash_token(token)
        now = now_utc()
        expires_at = now + timedelta(seconds=ttl)

        with self._lock:
            user = self._get_locked(username)
            self._ensure_active(user)
            self._prune_expired_keys_locked(user)
            if user.is_workload() and len(user.api_keys) >= 2:
                raise _error(
                    409,
                    "KEY_QUOTA_EXCEEDED",
                    "workload service accounts allow at most 2 concurrent keys (1 active + 1 rotation)",
                    {"max_keys_per_user": 2},
                )
            if len(user.api_keys) >= MAX_API_KEYS_PER_USER:
                raise _error(
                    409,
                    "KEY_QUOTA_EXCEEDED",
                    f"max {MAX_API_KEYS_PER_USER} active keys per user",
                    {"max_keys_per_user": MAX_API_KEYS_PER_USER},
                )
            if normalized_scopes:
                for scope in normalized_scopes:
                    if not issuance_scope_allowed(user.scopes, scope):
                        raise _forbidden_scope(f'user does not hold scope "{scope}"')
                final_scopes = normalized_scopes
            else:
                final_scopes = list(user.scopes)
            record = APIKeyRecord(
                key_id=key_id,
                name=name or f"key-{len(user.api_keys) + 1}",
                token_digest=digest,
                token_prefix=token_prefix(token),
                scopes=final_scopes,
                allowed_audiences=normalized_audiences,
                bound_public_key_der=bound_public_key_der,
                binding_mode="required" if bound_public_key_der else "",
                binding_status="active" if bound_public_key_der else "",
                contract=dict(contract) if contract is not None else None,
                active=True,
                created_at=isoformat(now) or "",
                expires_at=isoformat(expires_at),
            )
            user.api_keys[key_id] = record
            user.updated_at = now
            self.active_keys[digest] = (username, key_id)
            self._persist(user)
            self._emit_event("issue_api_key", username, {"key_id": key_id, "scopes": final_scopes})
            return APIKeyRecord(
                **{**record.__dict__, "scopes": list(record.scopes), "allowed_audiences": list(record.allowed_audiences)}
            ), token

    def revoke_api_key(self, username: str, key_id: str) -> None:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            record = user.api_keys.get(key_id)
            if record is None:
                raise _error(404, "KEY_NOT_FOUND", "api key not found")
            record.active = False
            record.revoked_at = isoformat(now_utc())
            self.active_keys.pop(record.token_digest, None)
            user.updated_at = now_utc()
            self._persist(user)
            self._emit_event("revoke_api_key", username, {"key_id": key_id})

    def list_api_keys(self, username: str) -> list[dict[str, Any]]:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            self._prune_expired_keys_locked(user)
            records = _sort_key_records(user.api_keys.values())
            return [record.to_summary() for record in records]

    def list_api_key_records(self, username: str) -> list[dict[str, Any]]:
        """完整密钥记录（含 token_hash / 绑定材料），对齐 Go ``ListAPIKeys``。"""
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            self._prune_expired_keys_locked(user)
            records = _sort_key_records(user.api_keys.values())
            return [record.to_record() for record in records]

    def lookup_api_key(self, token: str) -> tuple[User, APIKeyRecord] | None:
        if not token:
            return None
        digest = hash_token(token)
        with self._lock:
            entry = self.active_keys.get(digest)
            if entry is None:
                return None
            username, key_id = entry
            user = self.users.get(username)
            if user is None or user.status != STATUS_ACTIVE:
                return None
            record = user.api_keys.get(key_id)
            if record is None or not record.active or record.is_expired():
                self.active_keys.pop(digest, None)
                return None
            return user.sanitized_copy(), APIKeyRecord(
                **{**record.__dict__, "scopes": list(record.scopes), "allowed_audiences": list(record.allowed_audiences)}
            )

    def rotate_api_key(self, username: str, old_key_id: str, ttl: int = 0) -> tuple[APIKeyRecord, str]:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            old = user.api_keys.get(old_key_id)
            if old is None:
                raise _error(404, "KEY_NOT_FOUND", "api key not found")
            name, scopes, audiences = old.name, list(old.scopes), list(old.allowed_audiences)
            bound = old.bound_public_key_der
            contract = dict(old.contract) if old.contract is not None else None
        record, token = self.issue_api_key(username, name, scopes, audiences, ttl, bound, contract)
        self.revoke_api_key(username, old_key_id)
        return record, token

    # ------------------------------------------------------------------
    # MFA lifecycle
    # ------------------------------------------------------------------

    def setup_totp(self, username: str, expected: MFASetupState | None = None) -> tuple[str, str, datetime]:
        username = normalize_username(username)
        with self._lock:
            self._check_mfa_lock_locked(username)
            user = self._get_locked(username)
            self._ensure_active(user)
            if expected is not None and not expected.matches(user.mfa_enabled, user.totp_binding_id, user.mfa_epoch):
                raise _mfa_setup_changed()
            secret = generate_secret()
            algorithm = default_totp_algorithm()
            expires_at = now_utc() + timedelta(seconds=MFA_SETUP_TTL_SECONDS)
            user.pending_totp_secret = secret
            user.pending_totp_id = sm3_hex(secret)
            user.pending_totp_algorithm = algorithm
            user.pending_totp_expires_at = expires_at
            user.totp_secret_undecryptable = False
            user.updated_at = now_utc()
            self._persist(user)
            return secret, generate_uri(username, "PrivShield", secret, algorithm), expires_at

    def cancel_totp_setup(self, username: str, setup_id: str) -> None:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            if (
                not setup_id
                or not user.pending_totp_id
                or not hmac.compare_digest(setup_id, user.pending_totp_id)
            ):
                raise _mfa_setup_changed()
            user.pending_totp_secret = ""
            user.pending_totp_id = ""
            user.pending_totp_algorithm = ""
            user.pending_totp_expires_at = None
            user.updated_at = now_utc()
            self._persist(user)

    def enable_totp(self, username: str, code: str, setup_id: str) -> None:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            self._ensure_active(user)
            self._check_mfa_lock_locked(username)
            if (
                not setup_id
                or not user.pending_totp_id
                or not hmac.compare_digest(setup_id, user.pending_totp_id)
            ):
                raise _mfa_setup_changed()
            if user.pending_totp_expires_at is None or now_utc() >= user.pending_totp_expires_at:
                raise _mfa_setup_changed()
            if not user.pending_totp_secret:
                raise _mfa_setup_changed()
            valid, step = validate_code(
                user.pending_totp_secret, code, None, 0, user.pending_totp_algorithm
            )
            if not valid:
                self._fail_mfa_locked(username)
            was_enabled = user.mfa_enabled
            user.totp_secret = user.pending_totp_secret
            user.totp_algorithm = user.pending_totp_algorithm
            user.totp_binding_id = user.pending_totp_id
            user.last_totp_step = step
            user.mfa_enabled = True
            user.recovery_codes = []
            user.pending_totp_secret = ""
            user.pending_totp_id = ""
            user.pending_totp_algorithm = ""
            user.pending_totp_expires_at = None
            if was_enabled:
                user.mfa_epoch += 1
                self._revoke_sessions_locked(username)
            user.updated_at = now_utc()
            self._reset_failures_locked(username)
            self._persist(user)
            self._emit_event("mfa_enable", username, {"rebinding": was_enabled})

    def disable_totp(self, username: str, code: str = "", *, bypass_with_admin: bool = False) -> None:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            self._ensure_active(user)
            if not user.mfa_enabled:
                return
            if not bypass_with_admin:
                self._validate_totp_locked(user, code)
            user.mfa_epoch += 1
            user.mfa_enabled = False
            user.totp_secret = ""
            user.totp_algorithm = ""
            user.totp_binding_id = ""
            user.last_totp_step = 0
            user.pending_totp_secret = ""
            user.pending_totp_id = ""
            user.pending_totp_algorithm = ""
            user.pending_totp_expires_at = None
            user.recovery_codes = []
            user.updated_at = now_utc()
            self._revoke_sessions_locked(username)
            self._persist(user)
            self._emit_event("mfa_disable", username, {"bypass_with_admin": bypass_with_admin})

    def issue_recovery_codes(self, username: str) -> list[str]:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            self._ensure_active(user)
            if not user.mfa_enabled:
                raise _error(
                    409,
                    "MFA_SETUP_CHANGED",
                    "MFA must be enabled before recovery codes can be issued",
                )
            codes, entries = generate_recovery_codes()
            user.recovery_codes = entries
            user.updated_at = now_utc()
            self._persist(user)
            self._emit_event("mfa_recovery_codes_regenerated", username, {})
            return codes

    def consume_recovery_code(self, username: str, raw_code: str) -> bool:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            self._ensure_active(user)
            self._check_mfa_lock_locked(username)
            normalized = normalize_recovery_code(raw_code)
            if is_valid_recovery_code(normalized):
                digest = hash_recovery_code(normalized)
                for entry in user.recovery_codes:
                    if not entry.used_at and hmac.compare_digest(entry.digest, digest):
                        entry.used_at = isoformat(now_utc())
                        user.recovery_codes = [entry]
                        user.mfa_epoch += 1
                        user.mfa_enabled = False
                        user.totp_secret = ""
                        user.totp_algorithm = ""
                        user.totp_binding_id = ""
                        user.last_totp_step = 0
                        user.pending_totp_secret = ""
                        user.pending_totp_id = ""
                        user.pending_totp_algorithm = ""
                        user.pending_totp_expires_at = None
                        user.updated_at = now_utc()
                        self._revoke_sessions_locked(username)
                        self._reset_failures_locked(username)
                        self._persist(user)
                        self._emit_event("mfa_recovery_code_consumed", username, {})
                        return True
            self._record_failure_locked(username)
            return False

    def recovery_codes_remaining(self, username: str) -> int:
        with self._lock:
            return self._get_locked(normalize_username(username)).recovery_codes_remaining()

    def mfa_enabled(self, username: str) -> bool:
        with self._lock:
            user = self.users.get(normalize_username(username))
            return bool(user and user.mfa_enabled)

    def mfa_setup_pending(self, username: str) -> bool:
        with self._lock:
            user = self.users.get(normalize_username(username))
            if user is None or not user.pending_totp_id:
                return False
            return user.pending_totp_expires_at is not None and now_utc() < user.pending_totp_expires_at

    def totp_verifiable(self, username: str) -> bool:
        with self._lock:
            user = self.users.get(normalize_username(username))
            return bool(user and not user.totp_secret_undecryptable and user.totp_secret)

    def verify_totp(self, username: str, code: str) -> None:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            self._validate_totp_locked(user, code)
            self._reset_failures_locked(username)
            self._persist(user)

    def _validate_totp_locked(self, user: User, code: str) -> None:
        self._ensure_active(user)
        self._check_mfa_lock_locked(user.username)
        if not user.mfa_enabled:
            raise _error(409, "MFA_SETUP_CHANGED", "MFA is not enabled for this account")
        if user.totp_secret_undecryptable or not user.totp_secret:
            raise _error(503, "MFA_SECRET_UNAVAILABLE", "MFA key material is temporarily unavailable")
        algorithm = user.totp_algorithm
        binding_id = user.totp_binding_id
        if algorithm not in {"SHA1", "SM3"} or not binding_id:
            raise _mfa_unavailable()
        valid, step = validate_code(user.totp_secret, code, None, user.last_totp_step, algorithm)
        if not valid:
            self._fail_mfa_locked(user.username)
        replay_key = f"{user.username}:{binding_id}"
        if step <= self.totp_replay.get(replay_key, 0):
            raise _invalid_mfa_code()
        self.totp_replay[replay_key] = step
        user.last_totp_step = step
        user.updated_at = now_utc()

    def mfa_status(self, username: str) -> dict[str, Any]:
        with self._lock:
            user = self._get_locked(normalize_username(username))
            return {
                "username": user.username,
                "mfa_enabled": user.mfa_enabled,
                "recovery_codes_remaining": user.recovery_codes_remaining(),
                "mfa_pending": self.mfa_setup_pending(user.username),
            }

    def mfa_setup_snapshot(self, username: str) -> MFASetupState:
        """返回换绑 CAS 快照；后续 setup 必须带回同一快照才允许覆盖。"""
        with self._lock:
            user = self._get_locked(normalize_username(username))
            self._ensure_active(user)
            return MFASetupState(
                enabled=user.mfa_enabled,
                binding_id=user.totp_binding_id,
                epoch=user.mfa_epoch,
            )

    def mfa_binding_fingerprint(self, username: str) -> str:
        """把「被替换的是哪一次绑定」写进审计：仅取摘要前 8 位，不泄露种子。"""
        with self._lock:
            user = self.users.get(normalize_username(username))
            if user is None or not user.totp_secret:
                return ""
            return sm3_hex(user.totp_secret)[:8]

    def must_change_password_for_user(self, username: str) -> bool:
        with self._lock:
            user = self.users.get(normalize_username(username))
            return bool(user and user.must_change_password)

    def is_password_expired_for_user(self, username: str, grace_days: int = 0) -> bool:
        with self._lock:
            user = self.users.get(normalize_username(username))
            if user is None:
                return False
            grace = max(0, grace_days) * 24 * 3600
            return (now_utc() - user.password_updated_at).total_seconds() > 90 * 24 * 3600 + grace

    # ------------------------------------------------------------------
    # Service accounts / quotas
    # ------------------------------------------------------------------

    def create_service_account(
        self,
        username: str,
        display_name: str = "",
        tenant_id: str = "",
        scopes: list[str] | None = None,
        cert_fingerprints: list[str] | None = None,
        allowed_audiences: list[str] | None = None,
        egress_quota: EgressQuota | None = None,
    ) -> User:
        username = normalize_username(username)
        if not username.startswith("svc:"):
            username = "svc:" + username
        if not is_valid_username(username):
            raise invalid_argument("invalid service account username")
        display_name = display_name or username
        validate_display_name(display_name)
        final_scopes = validate_scopes(scopes or [])
        if not final_scopes:
            final_scopes = default_scopes_for_role(ROLE_SERVICE_ACCOUNT)
        validate_role_scope_consistency(ROLE_SERVICE_ACCOUNT, final_scopes)
        fingerprints = validate_cert_fingerprints(cert_fingerprints or [])
        audiences = validate_audiences(allowed_audiences or [], allow_empty=True, allow_wildcard=False)

        with self._lock:
            if username in self.users:
                raise _conflict("USER_EXISTS", "username already exists")
            self._ensure_fingerprints_unique_locked(username, fingerprints)
            now = now_utc()
            user = User(
                username=username,
                display_name=display_name,
                role=ROLE_SERVICE_ACCOUNT,
                scopes=final_scopes,
                status=STATUS_ACTIVE,
                password_hash=WORKLOAD_NON_INTERACTIVE_LOCKED,
                password_sm3_hash=WORKLOAD_NON_INTERACTIVE_LOCKED,
                principal_type=PRINCIPAL_WORKLOAD,
                tenant_id=tenant_id,
                allowed_audiences=audiences,
                egress_quota=egress_quota,
                cert_fingerprints=fingerprints,
                created_at=now,
                updated_at=now,
            )
            self.users[username] = user
            self.bootstrap_sealed = True
            self._persist(user)
            self._emit_event("create_service_account", username, {"tenant_id": tenant_id})
            return user.sanitized_copy()

    def pre_register_service_account(
        self,
        username: str,
        display_name: str = "",
        tenant_id: str = "",
        scopes: list[str] | None = None,
        expected_public_key_sha256: str = "",
        expected_spiffe_id: str = "",
        allowed_audiences: list[str] | None = None,
        egress_quota: EgressQuota | None = None,
    ) -> tuple[User, str]:
        """预分配待激活工作负载账号，明文激活令牌只在返回值出现一次。"""
        if egress_quota is not None and (
            egress_quota.rows_per_day < 0
            or egress_quota.bytes_per_day < 0
            or egress_quota.max_rows_per_response < 0
        ):
            raise _error(400, "INVALID_EGRESS_QUOTA", "egress quota values must be non-negative")
        audiences = validate_audiences(allowed_audiences or [], allow_empty=True, allow_wildcard=False)
        username = normalize_username(username)
        if not username.startswith("svc:"):
            username = "svc:" + username
        if not is_valid_username(username):
            raise invalid_argument("invalid service account username")
        display_name = display_name or username
        validate_display_name(display_name)
        final_scopes = validate_scopes(scopes or [])
        if not final_scopes:
            final_scopes = default_scopes_for_role(ROLE_SERVICE_ACCOUNT)
        validate_role_scope_consistency(ROLE_SERVICE_ACCOUNT, final_scopes)
        if expected_public_key_sha256.strip():
            try:
                expected_public_key_sha256 = format_bound_public_key_der(
                    parse_bound_public_key_der(expected_public_key_sha256)
                )
            except ValueError as exc:
                raise _error(400, "BINDING_PUBKEY_REQUIRED", str(exc)) from exc

        activation_token = generate_activation_token()
        now = now_utc()
        with self._lock:
            if username in self.users:
                raise _conflict("USER_EXISTS", "username already exists")
            user = User(
                username=username,
                display_name=display_name,
                role=ROLE_SERVICE_ACCOUNT,
                scopes=final_scopes,
                status=STATUS_PENDING_ACTIVATION,
                password_hash=WORKLOAD_NON_INTERACTIVE_LOCKED,
                password_sm3_hash="",
                password_updated_at=now,
                principal_type=PRINCIPAL_WORKLOAD,
                tenant_id=tenant_id,
                allowed_audiences=audiences,
                egress_quota=egress_quota,
                expected_public_key_sha256=expected_public_key_sha256.strip(),
                expected_spiffe_id=expected_spiffe_id.strip(),
                activation_token_digest=hash_token(activation_token),
                activation_expires_at=now + timedelta(minutes=10),
                activation_attempts=0,
                created_at=now,
                updated_at=now,
            )
            self.users[username] = user
            self.bootstrap_sealed = True
            try:
                self._persist(user)
            except Exception:
                self.users.pop(username, None)
                raise
            self._emit_event("create_service_account", username, {"tenant_id": tenant_id, "status": user.status})
            return user.sanitized_copy(), activation_token

    def activate_service_account(
        self,
        username: str,
        activation_token: str,
        actual_spki: str,
        actual_spiffe: str = "",
        tenant_id: str = "",
        service_audience: str = "",
    ) -> tuple[APIKeyRecord, str]:
        """五重严验一次性激活：令牌 / 公钥 / SPIFFE / 租户 / 待激活状态。"""
        username = normalize_username(username)
        if not username.startswith("svc:"):
            username = "svc:" + username
        now = now_utc()

        def deny(message: str) -> AuthError:
            return _error(
                401,
                "ACTIVATION_FAILED",
                f"activation failed: {message}",
            )

        with self._lock:
            user = self.users.get(username)
            if user is None or not user.is_workload():
                raise deny("invalid token, mismatched identity, or expired")
            if user.status != STATUS_PENDING_ACTIVATION or not user.activation_token_digest:
                raise deny("invalid token, mismatched identity, or expired")
            if user.activation_locked_until is not None and now < user.activation_locked_until:
                raise deny("activation is locked, retry later")
            if user.activation_expires_at is None or now >= user.activation_expires_at:
                raise deny("invalid token, mismatched identity, or expired")

            def fail(message: str) -> AuthError:
                user.activation_attempts += 1
                user.last_activation_attempt_at = now
                if user.activation_attempts >= 3:
                    user.activation_locked_until = now + timedelta(hours=24)
                user.updated_at = now
                self._persist(user)
                return deny(message)

            if not hmac.compare_digest(hash_token(activation_token or ""), user.activation_token_digest):
                raise fail("invalid token, mismatched identity, or expired")
            try:
                der = parse_bound_public_key_der(actual_spki)
            except ValueError:
                raise fail("client public key is invalid") from None
            formatted_spki = format_bound_public_key_der(der)
            if user.expected_public_key_sha256 and not hmac.compare_digest(
                formatted_spki.encode(), user.expected_public_key_sha256.encode()
            ):
                raise fail("client public key mismatch")
            if user.expected_spiffe_id and user.expected_spiffe_id != (actual_spiffe or ""):
                raise fail("SPIFFE identity mismatch")
            if user.tenant_id and user.tenant_id != (tenant_id or ""):
                raise fail("tenant mismatch")

            token = generate_secure_token()
            key_id = generate_key_id()
            audiences = list(user.allowed_audiences) or [service_audience or "service-hub"]
            record = APIKeyRecord(
                key_id=key_id,
                name=f"{user.username}-pop-key",
                token_digest=hash_token(token),
                token_prefix=token_prefix(token),
                scopes=list(user.scopes),
                allowed_audiences=audiences,
                bound_public_key_der=formatted_spki,
                binding_mode="required",
                binding_status="active",
                active=True,
                created_at=isoformat(now) or "",
                expires_at=isoformat(now + timedelta(seconds=DEFAULT_API_KEY_TTL_SECONDS)),
            )
            user.api_keys[key_id] = record
            user.activation_token_digest = ""
            user.activation_attempts = 0
            user.activation_locked_until = None
            user.activated_at = now
            user.status = STATUS_ACTIVE
            user.password_hash = WORKLOAD_NON_INTERACTIVE_LOCKED
            user.password_sm3_hash = ""
            user.updated_at = now
            self.active_keys[record.token_digest] = (username, key_id)
            self._persist(user)
            self._emit_event("workload.activated", username, {"key_id": key_id})
            return (
                APIKeyRecord(
                    **{
                        **record.__dict__,
                        "scopes": list(record.scopes),
                        "allowed_audiences": list(record.allowed_audiences),
                    }
                ),
                token,
            )

    def list_service_accounts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                user.to_summary() for user in sorted(self.users.values(), key=lambda item: item.username)
                if user.is_workload()
            ]

    def get_service_account(self, username: str) -> User:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            if not user.is_workload():
                raise _user_not_found(username)
            return user.sanitized_copy()

    def freeze_service_account(self, username: str) -> User:
        username = normalize_username(username)
        with self._lock:
            user = self._get_locked(username)
            if not user.is_workload():
                raise _user_not_found(username)
        self.set_status(username, STATUS_DISABLED)
        with self._lock:
            user = self._get_locked(username)
            for record in user.api_keys.values():
                record.active = False
                record.revoked_at = isoformat(now_utc())
                self.active_keys.pop(record.token_digest, None)
            self._persist(user)
            self._emit_event("freeze_service_account", username, {})
            return user.sanitized_copy()

    def unfreeze_service_account(self, username: str) -> User:
        return self.set_status(username, STATUS_ACTIVE)

    def set_egress_quota(self, username: str, quota: EgressQuota | None) -> User:
        username = normalize_username(username)
        if quota is not None and (quota.rows_per_day < 0 or quota.bytes_per_day < 0 or quota.max_rows_per_response < 0):
            raise _error(400, "INVALID_EGRESS_QUOTA", "egress quota values must be non-negative")
        with self._lock:
            user = self._get_locked(username)
            if not user.is_workload():
                raise _user_not_found(username)
            user.egress_quota = quota
            user.updated_at = now_utc()
            self._persist(user)
            self._emit_event("set_egress_quota", username, {"quota": quota.to_dict() if quota else None})
            return user.sanitized_copy()

    def egress_quota_for(self, username: str) -> EgressQuota | None:
        with self._lock:
            user = self.users.get(normalize_username(username))
            return user.egress_quota if user else None

    # ------------------------------------------------------------------
    # Outbox / events (sanitized metadata only)
    # ------------------------------------------------------------------

    def _emit_event(self, event_type: str, username: str, payload: dict[str, Any]) -> None:
        if self.repository is None or not self.settings.outbox_enabled:
            return
        record_event = getattr(self.repository, "record_event", None)
        if record_event is None:
            return
        try:
            record_event(event_type, username, payload)
        except Exception:  # noqa: BLE001 - outbox failures must not break auth mutations
            pass


@dataclass(frozen=True)
class MFASetupState:
    """MFA 绑定快照：仅当启用态、绑定 ID 与 epoch 全部匹配才允许换绑。"""

    enabled: bool
    binding_id: str
    epoch: int

    def matches(self, enabled: bool, binding_id: str, epoch: int) -> bool:
        return (
            self.enabled == enabled
            and self.epoch == epoch
            and (not enabled or self.binding_id == binding_id)
        )


def _sort_key_records(records: Any) -> list[APIKeyRecord]:
    """Go ``ListAPIKeys`` 排序：created_at 倒序，同刻按 key_id 升序。"""
    ordered = sorted(records, key=lambda record: record.key_id)
    ordered.sort(key=lambda record: record.created_at, reverse=True)
    return ordered


def _known_roles() -> tuple[str, ...]:
    return (ROLE_ADMIN, ROLE_AUDITOR, "consumer", "viewer", ROLE_SERVICE_ACCOUNT)


def generate_temporary_password(username: str) -> str:
    """生成满足等保三级复杂度且不含用户名的临时口令。"""
    upper = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    lower = "abcdefghijkmnopqrstuvwxyz"
    digits = "23456789"
    special = "!@#$%^&*()-_=+"
    for _ in range(64):
        chars = [
            secrets.choice(upper),
            secrets.choice(lower),
            secrets.choice(digits),
            secrets.choice(special),
        ]
        alphabet = upper + lower + digits + special
        chars.extend(secrets.choice(alphabet) for _ in range(12))
        secrets.SystemRandom().shuffle(chars)
        candidate = "".join(chars)
        try:
            validate_password_strength(candidate, username)
        except AuthError:
            continue
        return candidate
    raise _error(500, "INTERNAL_ERROR", "failed to generate a compliant temporary password")


def new_user_id() -> str:
    return str(uuid.uuid4())


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return isoformat(value)
    if isinstance(value, RecoveryCodeEntry):
        return value.to_dict()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=json_default, separators=(",", ":"))
