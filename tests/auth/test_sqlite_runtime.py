"""SQLite user-store parity: migrations, restart persistence, TOTP encryption."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.auth.config import AuthSettings
from engine.auth.migrations import apply_migrations, pending_migrations
from engine.auth.models import isoformat, now_utc
from engine.auth.persistence import SqliteRepository, default_migrations_dir
from engine.auth.runtime import build_runtime, reset_auth_runtime_for_tests
from engine.auth.totp import generate_code

from .conftest import ADMIN, ADMIN_PASSWORD, ADMIN_PASSWORD_NEW


def _settings(db_path: Path) -> AuthSettings:
    return AuthSettings(
        backend="sqlite",
        sqlite_path=str(db_path),
        bootstrap_remote_allowed=True,
        login_throttle_per_min=1000,
        admin_stepup_required=False,
        bcrypt_cost=4,
        sm3_iterations=1000,
    )


def _migrate(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        applied = apply_migrations(
            conn, default_migrations_dir(), applied_at=isoformat(now_utc()) or ""
        )
    finally:
        conn.close()
    assert len(applied) == 7


@pytest.fixture
def totp_clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    clock = SimpleNamespace(now=time.time())
    monkeypatch.setattr(
        "engine.auth.totp.time", SimpleNamespace(time=lambda: clock.now, sleep=time.sleep)
    )
    return clock


def test_startup_refuses_unmigrated_database(tmp_path: Path) -> None:
    db_path = tmp_path / "auth.db"
    with pytest.raises(RuntimeError, match="auth schema is not migrated"):
        SqliteRepository(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert pending_migrations(conn, default_migrations_dir())
    finally:
        conn.close()


def test_sqlite_runtime_persists_users_and_encrypted_totp(
    tmp_path: Path, totp_clock: SimpleNamespace
) -> None:
    db_path = tmp_path / "auth.db"
    _migrate(db_path)

    runtime = build_runtime(_settings(db_path))
    reset_auth_runtime_for_tests(runtime)
    try:
        runtime.service.bootstrap_admin(
            username=ADMIN,
            password=ADMIN_PASSWORD,
            client_ip="127.0.0.1",
            loopback_peer=True,
        )
        runtime.store.change_password(ADMIN, ADMIN_PASSWORD, ADMIN_PASSWORD_NEW)
        setup = runtime.service.mfa_setup(subject=ADMIN)
        enabled = runtime.service.mfa_enable(
            subject=ADMIN,
            code=generate_code(setup["secret"]),
            setup_id=setup["setup_id"],
        )
        assert enabled["mfa_enabled"] is True
        connection_secret = setup["secret"]
    finally:
        reset_auth_runtime_for_tests(None)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM auth_user_mfa").fetchone()
        assert row is not None
        assert row["enabled"] == 1
        assert row["totp_secret_enc"].startswith("sm4gcm:v1:")
        assert connection_secret not in row["totp_secret_enc"]
    finally:
        conn.close()

    key_file = Path(str(db_path) + ".key")
    assert key_file.exists()
    assert oct(key_file.stat().st_mode & 0o777) == "0o600"

    # Restart: users, MFA binding and the monotonic high-water mark survive.
    restarted = build_runtime(_settings(db_path))
    reset_auth_runtime_for_tests(restarted)
    try:
        totp_clock.now += 31
        session = restarted.service.login(
            username=ADMIN,
            password=ADMIN_PASSWORD_NEW,
            totp_code=generate_code(connection_secret),
            client_ip="127.0.0.1",
        )
        assert session["must_enroll_mfa"] is False
        assert session["user"]["mfa_enabled"] is True
        assert restarted.store.recovery_codes_remaining(ADMIN) == 8
    finally:
        reset_auth_runtime_for_tests(None)
