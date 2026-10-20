"""Auth kernel settings read from the canonical ``PRIVACY_AUTH_*`` variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .password import DEFAULT_BCRYPT_COST, DEFAULT_SM3_ITERATIONS


def parse_duration_seconds(raw: str, default: int) -> int:
    """Parse Go-style durations ("24h", "15m", "30s") or plain seconds."""
    value = raw.strip().lower()
    if not value:
        return default
    try:
        if value.endswith("ms"):
            return max(0, int(float(value[:-2]) / 1000))
        if value.endswith("s"):
            return int(float(value[:-1]))
        if value.endswith("m"):
            return int(float(value[:-1]) * 60)
        if value.endswith("h"):
            return int(float(value[:-1]) * 3600)
        return int(float(value))
    except ValueError:
        return default


def _bool(raw: str | None, default: bool = False) -> bool:
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_PRODUCTION_ENV_VARS = ("APP_ENV", "PRIVACY_ENV", "SERVICE_HUB_ENV")


def is_production_labeled() -> bool:
    """Mirror Go ``pkgconfig.IsProductionLabeled``.

    只有显式标注为 ``prod`` / ``production`` 的环境才按生产姿态 fail-closed；
    未知或空值按开发姿态处理。
    """
    for name in _PRODUCTION_ENV_VARS:
        if os.environ.get(name, "").strip().lower() in {"prod", "production"}:
            return True
    return False


@dataclass
class AuthSettings:
    """Runtime settings for the user/MFA subsystem.

    ``backend`` 为空表示未启用用户库；Go 的同等行为是显式要求
    ``PRIVACY_AUTH_USER_STORE_BACKEND``，绝不静默落在临时内存库上。
    """

    backend: str = ""
    sqlite_path: str = "data/auth.db"
    postgres_dsn: str = ""
    self_register: bool = False
    session_ttl_seconds: int = 24 * 3600
    login_throttle_per_min: int = 20
    password_expire_block: bool = False
    password_expire_grace_days: int = 0
    admin_stepup_required: bool = True
    bootstrap_token: str = ""
    bootstrap_remote_allowed: bool = False
    outbox_enabled: bool = True
    bcrypt_cost: int = DEFAULT_BCRYPT_COST
    sm3_iterations: int = DEFAULT_SM3_ITERATIONS
    totp_algorithm: str = ""
    encryption_key: str = ""
    cluster_signing_key_file: str = ""
    cluster_issuer: str = "privacy-engine"
    cluster_audience: str = ""
    cluster_tenant_id: str = ""
    service_audience: str = "privacy-engine"
    trusted_issuer_keys: dict[str, str] = field(default_factory=dict)
    audit_sink_url: str = ""
    audit_fail_closed: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.backend)

    @classmethod
    def from_env(cls) -> AuthSettings:
        prefix = "PRIVACY_AUTH"
        backend = os.environ.get(f"{prefix}_USER_STORE_BACKEND", "").strip().lower()
        if backend and backend not in {"memory", "sqlite", "postgres"}:
            raise ValueError(
                f"{prefix}_USER_STORE_BACKEND must be one of memory/sqlite/postgres, got {backend!r}"
            )
        bcrypt_cost = _int_env(f"{prefix}_BCRYPT_COST", DEFAULT_BCRYPT_COST)
        sm3_iterations = _int_env(f"{prefix}_SM3_ITERATIONS", DEFAULT_SM3_ITERATIONS)
        if bcrypt_cost < 4 or bcrypt_cost > 15:
            raise ValueError(f"{prefix}_BCRYPT_COST must be between 4 and 15")
        if sm3_iterations <= 0:
            raise ValueError(f"{prefix}_SM3_ITERATIONS must be positive")
        sqlite_file = os.environ.get(f"{prefix}_USER_STORE_FILE", "").strip()
        if backend == "sqlite" and not sqlite_file:
            raise ValueError(
                f"{prefix}_USER_STORE_FILE is required when {prefix}_USER_STORE_BACKEND=sqlite"
            )
        encryption_key = os.environ.get(f"{prefix}_USER_STORE_ENCRYPTION_KEY", "").strip()
        if backend in {"sqlite", "postgres"} and not encryption_key and is_production_labeled():
            raise ValueError(
                f"{prefix}_USER_STORE_ENCRYPTION_KEY is required in production-labeled mode; "
                "refusing to start with plaintext TOTP secrets"
            )
        totp_algorithm = os.environ.get("AUTH_TOTP_ALGORITHM", "").strip().lower()
        if totp_algorithm and totp_algorithm not in {"sha1", "sm3"}:
            raise ValueError(
                f"AUTH_TOTP_ALGORITHM must be one of sha1/sm3, got {totp_algorithm!r}"
            )
        return cls(
            backend=backend,
            sqlite_path=sqlite_file or "data/auth.db",
            postgres_dsn=os.environ.get(f"{prefix}_USER_STORE_DSN", "").strip(),
            self_register=_bool(os.environ.get(f"{prefix}_USER_SELF_REGISTER"), False),
            session_ttl_seconds=parse_duration_seconds(
                os.environ.get(f"{prefix}_USER_SESSION_TTL", ""), 24 * 3600
            ),
            login_throttle_per_min=_int_env(f"{prefix}_USER_LOGIN_THROTTLE_PER_MIN", 20),
            password_expire_block=_bool(os.environ.get(f"{prefix}_USER_PASSWORD_EXPIRE_BLOCK"), False),
            password_expire_grace_days=_int_env(f"{prefix}_USER_PASSWORD_EXPIRE_GRACE_DAYS", 0),
            admin_stepup_required=_bool(os.environ.get(f"{prefix}_ADMIN_STEPUP_REQUIRED"), True),
            bootstrap_token=os.environ.get(f"{prefix}_BOOTSTRAP_TOKEN", ""),
            bootstrap_remote_allowed=_bool(
                os.environ.get(f"{prefix}_USER_BOOTSTRAP_REMOTE_ALLOWED"), False
            ),
            outbox_enabled=_bool(os.environ.get(f"{prefix}_OUTBOX_ENABLED"), True),
            bcrypt_cost=bcrypt_cost,
            sm3_iterations=sm3_iterations,
            totp_algorithm=totp_algorithm,
            encryption_key=encryption_key,
            cluster_signing_key_file=os.environ.get(f"{prefix}_CLUSTER_SIGNING_KEY_FILE", ""),
            cluster_issuer=os.environ.get(f"{prefix}_CLUSTER_ISSUER", "privacy-engine"),
            cluster_audience=os.environ.get(f"{prefix}_CLUSTER_AUDIENCE", ""),
            cluster_tenant_id=os.environ.get(f"{prefix}_CLUSTER_TENANT_ID", ""),
            service_audience=os.environ.get(f"{prefix}_SERVICE_AUDIENCE", "privacy-engine").strip()
            or "privacy-engine",
            trusted_issuer_keys=_json_map(os.environ.get(f"{prefix}_CLUSTER_TRUSTED_KEYS_JSON", "")),
            audit_sink_url=os.environ.get("PRIVACY_AUDIT_LOG_URL", ""),
            audit_fail_closed=_bool(os.environ.get("PRIVACY_AUDIT_LOG_FAIL_CLOSED"), False),
        )

    def sqlite_file(self) -> Path:
        return Path(self.sqlite_path).expanduser()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _json_map(raw: str) -> dict[str, str]:
    import json

    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(value) for key, value in parsed.items()}
