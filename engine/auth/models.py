"""User, API key and quota models mirroring Go ``pkg/auth/user.go``."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .recovery import RecoveryCodeEntry

ROLE_ADMIN = "admin"
ROLE_AUDITOR = "auditor"
ROLE_CONSUMER = "consumer"
ROLE_VIEWER = "viewer"
ROLE_SERVICE_ACCOUNT = "service_account"

STATUS_ACTIVE = "active"
STATUS_DISABLED = "disabled"
STATUS_PENDING_ACTIVATION = "pending_activation"

PRINCIPAL_USER = "user"
PRINCIPAL_WORKLOAD = "workload"

WORKLOAD_NON_INTERACTIVE_LOCKED = "$workload_locked$non_interactive$"

MAX_FAILED_LOGIN_ATTEMPTS = 5
ACCOUNT_LOCKOUT_SECONDS = 15 * 60
MAX_API_KEYS_PER_USER = 32
MAX_SESSIONS_PER_USER = 8
DEFAULT_SESSION_TTL_SECONDS = 24 * 3600
MAX_SESSION_TTL_SECONDS = 24 * 3600
DEFAULT_API_KEY_TTL_SECONDS = 30 * 24 * 3600
MAX_API_KEY_TTL_SECONDS = 90 * 24 * 3600
MAX_SCOPE_COUNT = 64
MAX_DISPLAY_NAME_LENGTH = 64
MAX_KEY_NAME_LENGTH = 64
LOGIN_THROTTLE_WINDOW_SECONDS = 60
LOGIN_THROTTLE_MAX_PER_IP = 20
API_KEY_ISSUE_THROTTLE_WINDOW_SECONDS = 5 * 60
API_KEY_ISSUE_THROTTLE_MAX_PER_CALLER = 10
MFA_SETUP_TTL_SECONDS = 10 * 60
MFA_RECOVERY_SESSION_TTL_SECONDS = 10 * 60
PASSWORD_HISTORY_DEPTH = 3

_KNOWN_ROLES = {ROLE_ADMIN, ROLE_AUDITOR, ROLE_CONSUMER, ROLE_VIEWER, ROLE_SERVICE_ACCOUNT}
_PRIVILEGED_ROLES = {ROLE_ADMIN, ROLE_AUDITOR}
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
_SVC_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,36}$")


def normalize_username(username: str) -> str:
    return username.strip().lower()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def default_scopes_for_role(role: str) -> list[str]:
    if role == ROLE_ADMIN:
        return [
            "admin", "user:admin", "hub:admin", "ops:admin", "privacy:budget",
            "privacy:budget:read", "hub:read", "health:read",
        ]
    if role == ROLE_AUDITOR:
        return [
            "audit:read", "audit:verify", "audit:export", "rule:read",
            "privacy:budget:read", "ops:diagnostics", "user:read", "hub:read", "health:read",
        ]
    if role == ROLE_CONSUMER:
        return [
            "hub:dispatch", "hub:read", "privacy:mask", "privacy:dp", "privacy:kano",
            "medical:process", "file:process", "health:read",
        ]
    if role == ROLE_VIEWER:
        return ["hub:read", "health:read"]
    if role == ROLE_SERVICE_ACCOUNT:
        return ["hub:read", "health:read"]
    return ["health:read"]


def is_valid_role(role: str) -> bool:
    return role in _KNOWN_ROLES


def is_privileged_role(role: str) -> bool:
    return role in _PRIVILEGED_ROLES


def is_valid_username(username: str) -> bool:
    if username.startswith("svc:"):
        return bool(_SVC_USERNAME_RE.fullmatch(username[4:])) and len(username) <= 40
    return bool(_USERNAME_RE.fullmatch(username))


def validate_display_name(name: str) -> None:
    from .errors import invalid_argument

    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        raise invalid_argument("display name is too long (max 64 characters)")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in name):
        raise invalid_argument("display name must not contain control characters")


def validate_key_name(name: str) -> None:
    from .errors import invalid_argument

    if len(name) > MAX_KEY_NAME_LENGTH:
        raise invalid_argument("key name is too long (max 64 characters)")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in name):
        raise invalid_argument("key name must not contain control characters")


def validate_scopes(scopes: list[str]) -> list[str]:
    from .errors import invalid_argument

    if len(scopes) > MAX_SCOPE_COUNT:
        raise invalid_argument(f"too many scopes (max {MAX_SCOPE_COUNT})")
    normalized: list[str] = []
    for scope in scopes:
        value = str(scope).strip()
        if not value:
            raise invalid_argument("scope must not be empty")
        if value not in normalized:
            normalized.append(value)
    return normalized


@dataclass
class EgressQuota:
    rows_per_day: int = 0
    bytes_per_day: int = 0
    max_rows_per_response: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> EgressQuota | None:
        if not payload:
            return None
        return cls(
            rows_per_day=int(payload.get("rows_per_day", 0) or 0),
            bytes_per_day=int(payload.get("bytes_per_day", 0) or 0),
            max_rows_per_response=int(payload.get("max_rows_per_response", 0) or 0),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "rows_per_day": self.rows_per_day,
            "bytes_per_day": self.bytes_per_day,
            "max_rows_per_response": self.max_rows_per_response,
        }


@dataclass
class APIKeyRecord:
    key_id: str
    name: str
    token_digest: str
    token_prefix: str
    scopes: list[str]
    allowed_audiences: list[str] = field(default_factory=list)
    bound_public_key_der: str = ""
    binding_mode: str = ""
    binding_status: str = ""
    contract: dict[str, Any] | None = None
    active: bool = True
    created_at: str = ""
    expires_at: str | None = None
    revoked_at: str | None = None

    def is_expired(self) -> bool:
        expiry = parse_iso(self.expires_at)
        return bool(expiry and now_utc() > expiry)

    def to_summary(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "name": self.name,
            "token_prefix": self.token_prefix,
            "scopes": list(self.scopes),
            "allowed_audiences": list(self.allowed_audiences),
            "binding_mode": self.binding_mode,
            "binding_status": self.binding_status,
            "active": self.active and not self.is_expired(),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
        }

    def to_record(self) -> dict[str, Any]:
        """Full record shape mirroring Go ``APIKeyRecord`` JSON (GET .../keys)."""
        payload: dict[str, Any] = {
            "key_id": self.key_id,
            "name": self.name,
            "token_hash": self.token_digest,
            "token_prefix": self.token_prefix,
            "scopes": list(self.scopes),
            "active": self.active,
            "created_at": self.created_at,
        }
        if self.allowed_audiences:
            payload["allowed_audiences"] = list(self.allowed_audiences)
        if self.binding_mode:
            payload["binding_mode"] = self.binding_mode
        if self.binding_status:
            payload["binding_status"] = self.binding_status
        if self.bound_public_key_der:
            payload["bound_public_key_der"] = self.bound_public_key_der
        if self.contract is not None:
            payload["contract"] = dict(self.contract)
        if self.expires_at:
            payload["expires_at"] = self.expires_at
        if self.revoked_at:
            payload["revoked_at"] = self.revoked_at
        return payload


@dataclass
class User:
    username: str
    display_name: str = ""
    role: str = ROLE_VIEWER
    scopes: list[str] = field(default_factory=list)
    status: str = STATUS_ACTIVE
    password_hash: str = ""
    password_sm3_hash: str = ""
    password_history: list[str] = field(default_factory=list)
    password_updated_at: datetime = field(default_factory=now_utc)
    must_change_password: bool = False
    mfa_enabled: bool = False
    totp_secret: str = ""
    totp_algorithm: str = ""
    totp_binding_id: str = ""
    pending_totp_secret: str = ""
    pending_totp_id: str = ""
    pending_totp_algorithm: str = ""
    pending_totp_expires_at: datetime | None = None
    totp_secret_undecryptable: bool = False
    last_totp_step: int = 0
    mfa_epoch: int = 0
    recovery_codes: list[RecoveryCodeEntry] = field(default_factory=list)
    principal_type: str = PRINCIPAL_USER
    tenant_id: str = ""
    allowed_audiences: list[str] = field(default_factory=list)
    egress_quota: EgressQuota | None = None
    cert_fingerprints: list[str] = field(default_factory=list)
    expected_public_key_sha256: str = ""
    expected_spiffe_id: str = ""
    activation_token_digest: str = ""
    activation_expires_at: datetime | None = None
    activation_attempts: int = 0
    last_activation_attempt_at: datetime | None = None
    activation_locked_until: datetime | None = None
    activated_at: datetime | None = None
    api_keys: dict[str, APIKeyRecord] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)

    def is_workload(self) -> bool:
        return self.role == ROLE_SERVICE_ACCOUNT or self.username.startswith("svc:")

    def recovery_codes_remaining(self) -> int:
        return sum(1 for entry in self.recovery_codes if not entry.used_at)

    def password_expired(self) -> bool:
        return (now_utc() - self.password_updated_at).total_seconds() > 90 * 24 * 3600

    def to_summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "principal_type": self.principal_type,
            "scopes": list(self.scopes),
            "status": self.status,
            "key_count": len(self.api_keys),
            "must_change_password": self.must_change_password,
            "mfa_enabled": self.mfa_enabled,
            "recovery_codes_remaining": self.recovery_codes_remaining(),
            "created_at": isoformat(self.created_at),
            "updated_at": isoformat(self.updated_at),
            "password_updated_at": isoformat(self.password_updated_at),
        }
        if self.tenant_id:
            payload["tenant_id"] = self.tenant_id
        if self.cert_fingerprints:
            payload["cert_fingerprints"] = list(self.cert_fingerprints)
        if self.expected_public_key_sha256:
            payload["expected_public_key_sha256"] = self.expected_public_key_sha256
        if self.expected_spiffe_id:
            payload["expected_spiffe_id"] = self.expected_spiffe_id
        if self.allowed_audiences:
            payload["allowed_audiences"] = list(self.allowed_audiences)
        if self.egress_quota is not None:
            payload["egress_quota"] = self.egress_quota.to_dict()
        return payload

    def to_detail(self) -> dict[str, Any]:
        """Full sanitized profile mirroring Go ``User`` JSON (GET /users/:username)."""
        payload: dict[str, Any] = {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "scopes": list(self.scopes),
            "status": self.status,
            "password_updated_at": isoformat(self.password_updated_at),
            "must_change_password": self.must_change_password,
            "mfa_enabled": self.mfa_enabled,
            "principal_type": self.principal_type,
            "activation_attempts": self.activation_attempts,
            "created_at": isoformat(self.created_at),
            "updated_at": isoformat(self.updated_at),
        }
        if self.tenant_id:
            payload["tenant_id"] = self.tenant_id
        if self.expected_public_key_sha256:
            payload["expected_public_key_sha256"] = self.expected_public_key_sha256
        if self.expected_spiffe_id:
            payload["expected_spiffe_id"] = self.expected_spiffe_id
        if self.allowed_audiences:
            payload["allowed_audiences"] = list(self.allowed_audiences)
        if self.activation_expires_at is not None:
            payload["activation_expires_at"] = isoformat(self.activation_expires_at)
        if self.last_activation_attempt_at is not None:
            payload["last_activation_attempt_at"] = isoformat(self.last_activation_attempt_at)
        if self.activation_locked_until is not None:
            payload["activation_locked_until"] = isoformat(self.activation_locked_until)
        if self.activated_at is not None:
            payload["activated_at"] = isoformat(self.activated_at)
        if self.cert_fingerprints:
            payload["cert_fingerprints"] = list(self.cert_fingerprints)
        if self.egress_quota is not None:
            payload["egress_quota"] = self.egress_quota.to_dict()
        if self.api_keys:
            ordered = sorted(self.api_keys.values(), key=lambda record: record.key_id)
            ordered.sort(key=lambda record: record.created_at, reverse=True)
            payload["api_keys"] = {record.key_id: record.to_record() for record in ordered}
        return payload

    def sanitized_copy(self) -> User:
        clone = copy.copy(self)
        clone.scopes = list(self.scopes)
        clone.password_history = list(self.password_history)
        clone.recovery_codes = [copy.copy(entry) for entry in self.recovery_codes]
        clone.api_keys = {key_id: copy.copy(key) for key_id, key in self.api_keys.items()}
        clone.cert_fingerprints = list(self.cert_fingerprints)
        clone.allowed_audiences = list(self.allowed_audiences)
        clone.egress_quota = copy.copy(self.egress_quota) if self.egress_quota else None
        return clone
