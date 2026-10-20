"""SQLite authoritative persistence for the auth kernel.

表结构直接执行仓库权威迁移 ``deploy/sql/auth/*.up.sql``（PostgreSQL/SQLite
通用子集）。应用启动不做隐式 DDL：``migrate`` 命令或显式 ``ensure_schema()``
负责建库；本模块在装载前校验 ``schema_migrations`` 完整性。

TOTP 密钥落盘前使用 SM4-GCM 信封加密；未显式配置
``PRIVACY_AUTH_USER_STORE_ENCRYPTION_KEY`` 时，在数据库旁生成 0600
密钥文件，杜绝明文密钥入库。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .migrations import verify_schema
from .models import (
    APIKeyRecord,
    EgressQuota,
    User,
    isoformat,
    now_utc,
    parse_iso,
)
from .recovery import RecoveryCodeEntry

_ENVELOPE_PREFIX = "sm4gcm:v1:"
_MIGRATIONS_DIR_ENV = "PRIVACY_AUTH_USER_MIGRATIONS_DIR"
_DEFAULT_DB_PATH = "data/auth.db"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_migrations_dir() -> Path:
    override = os.environ.get(_MIGRATIONS_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return _repo_root() / "deploy" / "sql" / "auth"


class SqliteRepository:
    """Write-through repository for :class:`engine.auth.store.UserStore`."""

    def __init__(self, path: str | Path, *, encryption_key: str = "", migrations_dir: str | Path | None = None) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.RLock()
        self._migrations_dir = Path(migrations_dir) if migrations_dir else default_migrations_dir()
        self._key = self._resolve_key(encryption_key)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._ensure_schema_ready()
        if str(self.path) != ":memory:":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Key material
    # ------------------------------------------------------------------

    def _resolve_key(self, configured: str) -> bytes:
        raw = configured.strip()
        if raw:
            return _parse_key_material(raw)
        # 未配置时在数据库旁生成 0600 密钥文件（不把 TOTP 密钥明文落库）。
        key_path = Path(str(self.path) + ".key")
        with self._lock:
            if key_path.exists():
                data = key_path.read_bytes().strip()
                if data:
                    return _parse_key_material(data.decode("ascii", "ignore"))
            return _new_key_file(key_path)

    def _encrypt_secret(self, secret: str) -> str:
        if not secret:
            return ""
        iv = os.urandom(12)
        encryptor = Cipher(algorithms.SM4(self._key), modes.GCM(iv)).encryptor()
        ciphertext = encryptor.update(secret.encode("utf-8")) + encryptor.finalize()
        payload = iv + encryptor.tag + ciphertext
        return _ENVELOPE_PREFIX + base64.b64encode(payload).decode("ascii")

    def _decrypt_secret(self, encoded: str) -> tuple[str, bool]:
        """Return (plaintext, undecryptable)."""
        if not encoded:
            return "", False
        if not encoded.startswith(_ENVELOPE_PREFIX):
            # 兼容既有的明文开发库；生产写路径永不产生该形态。
            return encoded, False
        try:
            payload = base64.b64decode(encoded[len(_ENVELOPE_PREFIX):])
            iv, tag, ciphertext = payload[:12], payload[12:28], payload[28:]
            decryptor = Cipher(algorithms.SM4(self._key), modes.GCM(iv, tag)).decryptor()
            return (decryptor.update(ciphertext) + decryptor.finalize()).decode("utf-8"), False
        except Exception:  # noqa: BLE001 - 信封损坏按服务端密钥故障处理
            return "", True

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema_ready(self) -> None:
        """Verify migrations instead of applying DDL at startup (Go parity)."""
        with self._lock:
            verify_schema(self._conn, self._migrations_dir)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_users(self) -> dict[str, User]:
        with self._lock:
            users: dict[str, User] = {}
            rows = self._conn.execute("SELECT * FROM auth_users").fetchall()
            for row in rows:
                user_id = row["id"]
                mfa = self._conn.execute(
                    "SELECT * FROM auth_user_mfa WHERE user_id = ?", (user_id,)
                ).fetchone()
                secret, undecryptable = self._decrypt_secret(mfa["totp_secret_enc"]) if mfa else ("", False)
                pending_secret, pending_undecryptable = (
                    self._decrypt_secret(mfa["pending_totp_secret_enc"]) if mfa else ("", False)
                )
                user = User(
                    username=row["username_norm"],
                    display_name=row["display_name"],
                    role=row["role"],
                    scopes=[
                        item["scope"]
                        for item in self._conn.execute(
                            "SELECT scope FROM auth_user_scopes WHERE user_id = ? ORDER BY rowid", (user_id,)
                        ).fetchall()
                    ],
                    status=row["status"],
                    password_hash=row["password_hash"],
                    password_sm3_hash=row["password_sm3_hash"] or "",
                    password_history=[
                        item["password_hash"]
                        for item in self._conn.execute(
                            "SELECT password_hash FROM auth_password_history WHERE user_id = ? ORDER BY sequence_no",
                            (user_id,),
                        ).fetchall()
                    ],
                    password_updated_at=parse_iso(row["password_updated_at"]) or now_utc(),
                    must_change_password=bool(row["must_change_password"]),
                    mfa_enabled=bool(mfa["enabled"]) if mfa else False,
                    totp_secret=secret,
                    totp_algorithm=(mfa["algorithm"] or "") if mfa else "",
                    totp_binding_id=(mfa["binding_id"] or "") if mfa else "",
                    pending_totp_secret=pending_secret,
                    pending_totp_id=(mfa["pending_id"] or "") if mfa else "",
                    pending_totp_algorithm=(mfa["pending_algorithm"] or "") if mfa else "",
                    pending_totp_expires_at=parse_iso(mfa["pending_expires_at"]) if mfa else None,
                    totp_secret_undecryptable=undecryptable or pending_undecryptable,
                    last_totp_step=int(mfa["last_totp_step"] or 0) if mfa else 0,
                    mfa_epoch=int(row["mfa_epoch"] or 0),
                    recovery_codes=[
                        RecoveryCodeEntry(
                            digest=item["code_hash"],
                            prefix=item["code_prefix"],
                            created_at=item["created_at"],
                            used_at=item["used_at"],
                        )
                        for item in self._conn.execute(
                            "SELECT * FROM auth_user_recovery_codes WHERE user_id = ? ORDER BY rowid", (user_id,)
                        ).fetchall()
                    ],
                    principal_type=row["principal_type"] if "principal_type" in row.keys() else "user",
                    tenant_id=row["tenant_id"] or "",
                    allowed_audiences=_load_json_list(row["allowed_audiences_json"]),
                    egress_quota=EgressQuota.from_dict(_load_json_obj(row["egress_quota_json"])),
                    cert_fingerprints=[
                        item["fingerprint"]
                        for item in self._conn.execute(
                            "SELECT fingerprint FROM auth_cert_fingerprints WHERE user_id = ? ORDER BY rowid",
                            (user_id,),
                        ).fetchall()
                    ],
                    expected_public_key_sha256=_row_value(row, "expected_public_key_sha256"),
                    expected_spiffe_id=_row_value(row, "expected_spiffe_id"),
                    activation_token_digest=_row_value(row, "activation_token_digest"),
                    activation_expires_at=parse_iso(_row_value(row, "activation_expires_at")),
                    activation_attempts=int(_row_value(row, "activation_attempts") or 0) if _row_value(row, "activation_attempts") is not None else 0,
                    last_activation_attempt_at=parse_iso(_row_value(row, "last_activation_attempt_at")),
                    activation_locked_until=parse_iso(_row_value(row, "activation_locked_until")),
                    activated_at=parse_iso(_row_value(row, "activated_at")),
                    api_keys={
                        item["id"]: APIKeyRecord(
                            key_id=item["id"],
                            name=item["name"],
                            token_digest=item["token_digest"],
                            token_prefix=item["token_prefix"],
                            scopes=_load_json_list(item["scopes_json"]),
                            allowed_audiences=_load_json_list(
                                item["allowed_audiences_json"] if "allowed_audiences_json" in item.keys() else "[]"
                            ),
                            bound_public_key_der=(
                                item["bound_public_key_der"] if "bound_public_key_der" in item.keys() else ""
                            ) or "",
                            binding_mode=(
                                "required"
                                if (item["binding_mode"] if "binding_mode" in item.keys() else "none") == "required"
                                else ""
                            ),
                            binding_status="active" if item["active"] else "revoked",
                            contract=(
                                _load_json_obj(item["contract_json"])
                                if "contract_json" in item.keys() and item["contract_json"]
                                else None
                            ),
                            active=bool(item["active"]),
                            created_at=item["created_at"],
                            expires_at=item["expires_at"],
                            revoked_at=item["revoked_at"],
                        )
                        for item in self._conn.execute(
                            "SELECT * FROM auth_api_keys WHERE user_id = ?", (user_id,)
                        ).fetchall()
                    },
                    created_at=parse_iso(row["created_at"]) or now_utc(),
                    updated_at=parse_iso(row["updated_at"]) or now_utc(),
                )
                users[user.username] = user
            return users

    # ------------------------------------------------------------------
    # Save / delete
    # ------------------------------------------------------------------

    def _user_id(self, username: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM auth_users WHERE username_norm = ?", (username,)
        ).fetchone()
        return row["id"] if row else None

    def save_user(self, user: User) -> None:
        with self._lock, self._conn:
            user_id = self._user_id(user.username) or str(uuid.uuid4())
            now = isoformat(now_utc())
            self._conn.execute(
                """
                INSERT INTO auth_users (
                    id, username, username_norm, display_name, role, status, password_hash,
                    password_sm3_hash, password_version, must_change_password, password_updated_at,
                    version, mfa_epoch, tenant_id, allowed_audiences_json, egress_quota_json,
                    created_at, updated_at, principal_type, expected_public_key_sha256,
                    expected_spiffe_id, activation_token_digest, activation_expires_at, activation_attempts,
                    last_activation_attempt_at, activation_locked_until, activated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    display_name=excluded.display_name,
                    role=excluded.role,
                    status=excluded.status,
                    password_hash=excluded.password_hash,
                    password_sm3_hash=excluded.password_sm3_hash,
                    must_change_password=excluded.must_change_password,
                    password_updated_at=excluded.password_updated_at,
                    mfa_epoch=excluded.mfa_epoch,
                    tenant_id=excluded.tenant_id,
                    allowed_audiences_json=excluded.allowed_audiences_json,
                    egress_quota_json=excluded.egress_quota_json,
                    updated_at=excluded.updated_at,
                    principal_type=excluded.principal_type,
                    expected_public_key_sha256=excluded.expected_public_key_sha256,
                    expected_spiffe_id=excluded.expected_spiffe_id,
                    activation_token_digest=excluded.activation_token_digest,
                    activation_expires_at=excluded.activation_expires_at,
                    activation_attempts=excluded.activation_attempts,
                    last_activation_attempt_at=excluded.last_activation_attempt_at,
                    activation_locked_until=excluded.activation_locked_until,
                    activated_at=excluded.activated_at
                """,
                (
                    user_id,
                    user.username,
                    user.username,
                    user.display_name or user.username,
                    user.role,
                    user.status,
                    user.password_hash,
                    user.password_sm3_hash,
                    1,
                    1 if user.must_change_password else 0,
                    isoformat(user.password_updated_at) or now,
                    1,
                    user.mfa_epoch,
                    user.tenant_id,
                    json.dumps(list(user.allowed_audiences), ensure_ascii=False),
                    json.dumps(user.egress_quota.to_dict() if user.egress_quota else None),
                    isoformat(user.created_at) or now,
                    isoformat(user.updated_at) or now,
                    user.principal_type,
                    user.expected_public_key_sha256,
                    user.expected_spiffe_id,
                    user.activation_token_digest,
                    isoformat(user.activation_expires_at),
                    user.activation_attempts,
                    isoformat(user.last_activation_attempt_at),
                    isoformat(user.activation_locked_until),
                    isoformat(user.activated_at),
                ),
            )

            self._conn.execute("DELETE FROM auth_user_scopes WHERE user_id = ?", (user_id,))
            self._conn.executemany(
                "INSERT INTO auth_user_scopes (user_id, scope) VALUES (?, ?)",
                [(user_id, scope) for scope in user.scopes],
            )

            self._conn.execute("DELETE FROM auth_password_history WHERE user_id = ?", (user_id,))
            self._conn.executemany(
                "INSERT INTO auth_password_history (user_id, sequence_no, password_hash, created_at) VALUES (?, ?, ?, ?)",
                [
                    (user_id, index, encoded, isoformat(user.password_updated_at) or now)
                    for index, encoded in enumerate(user.password_history)
                    if encoded
                ],
            )

            if (
                user.mfa_enabled
                or user.totp_secret
                or user.pending_totp_id
                or user.last_totp_step
            ):
                self._conn.execute(
                    """
                    INSERT INTO auth_user_mfa (
                        user_id, totp_secret_enc, enabled, last_totp_step, algorithm, binding_id,
                        pending_totp_secret_enc, pending_id, pending_algorithm, pending_expires_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        totp_secret_enc=excluded.totp_secret_enc,
                        enabled=excluded.enabled,
                        last_totp_step=excluded.last_totp_step,
                        algorithm=excluded.algorithm,
                        binding_id=excluded.binding_id,
                        pending_totp_secret_enc=excluded.pending_totp_secret_enc,
                        pending_id=excluded.pending_id,
                        pending_algorithm=excluded.pending_algorithm,
                        pending_expires_at=excluded.pending_expires_at,
                        updated_at=excluded.updated_at
                    """,
                    (
                        user_id,
                        self._encrypt_secret(user.totp_secret),
                        1 if user.mfa_enabled else 0,
                        user.last_totp_step,
                        user.totp_algorithm,
                        user.totp_binding_id,
                        self._encrypt_secret(user.pending_totp_secret),
                        user.pending_totp_id,
                        user.pending_totp_algorithm,
                        isoformat(user.pending_totp_expires_at) or "",
                        now,
                    ),
                )
            else:
                self._conn.execute("DELETE FROM auth_user_mfa WHERE user_id = ?", (user_id,))

            self._conn.execute("DELETE FROM auth_api_keys WHERE user_id = ?", (user_id,))
            self._conn.executemany(
                """
                INSERT INTO auth_api_keys (
                    id, user_id, name, token_digest, token_prefix, scopes_json,
                    allowed_audiences_json, active, created_at, expires_at, revoked_at,
                    binding_mode, bound_public_key_der, contract_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.key_id,
                        user_id,
                        record.name,
                        record.token_digest,
                        record.token_prefix,
                        json.dumps(list(record.scopes), ensure_ascii=False),
                        json.dumps(list(record.allowed_audiences), ensure_ascii=False),
                        1 if record.active and not record.is_expired() else 0,
                        record.created_at,
                        record.expires_at or "",
                        record.revoked_at,
                        record.binding_mode or "none",
                        record.bound_public_key_der or None,
                        json.dumps(record.contract, ensure_ascii=False) if record.contract is not None else None,
                    )
                    for record in user.api_keys.values()
                ],
            )

            self._conn.execute("DELETE FROM auth_cert_fingerprints WHERE user_id = ?", (user_id,))
            self._conn.executemany(
                "INSERT INTO auth_cert_fingerprints (user_id, fingerprint, created_at) VALUES (?, ?, ?)",
                [(user_id, fingerprint, now) for fingerprint in user.cert_fingerprints],
            )

            self._conn.execute("DELETE FROM auth_user_recovery_codes WHERE user_id = ?", (user_id,))
            self._conn.executemany(
                """
                INSERT INTO auth_user_recovery_codes (code_hash, user_id, code_prefix, created_at, used_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (entry.digest, user_id, entry.prefix, entry.created_at, entry.used_at)
                    for entry in user.recovery_codes
                ],
            )

    def delete_user(self, username: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM auth_users WHERE username_norm = ?", (username,))

    def record_event(self, event_type: str, username: str, payload: dict[str, Any]) -> None:
        now = isoformat(now_utc()) or ""
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO auth_outbox (
                    id, aggregate_type, aggregate_id, event_type, payload,
                    created_at, attempts, next_attempt_at
                ) VALUES (?, 'auth_user', ?, ?, ?, ?, 0, ?)
                """,
                (
                    str(uuid.uuid4()),
                    username,
                    event_type,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    now,
                    now,
                ),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _new_key_file(path: Path) -> bytes:
    key = os.urandom(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError:
        return _parse_key_material(path.read_bytes().decode("ascii", "ignore"))
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        handle.write(key.hex())
    return key


def _parse_key_material(raw: str) -> bytes:
    value = raw.strip()
    if len(value) == 32:
        try:
            return bytes.fromhex(value)
        except ValueError:
            pass
    if len(value) == 64:
        try:
            return bytes.fromhex(value)[:16]
        except ValueError:
            pass
    try:
        decoded = base64.b64decode(value, validate=True)
        if len(decoded) in (16, 32):
            return decoded[:16]
    except Exception:  # noqa: BLE001
        pass
    # 任意口令材料：SHA-256 派生 128-bit 密钥，保证长度合法且不可逆。
    return hashlib.sha256(value.encode("utf-8")).digest()[:16]


def _load_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _load_json_obj(raw: str | None) -> dict[str, Any] | None:
    if not raw or raw == "null":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _row_value(row: sqlite3.Row, column: str) -> str:
    return row[column] if column in row.keys() else ""
