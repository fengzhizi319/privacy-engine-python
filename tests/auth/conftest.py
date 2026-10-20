"""Shared fixtures for auth-kernel conformance tests.

每个用例都用独立的进程级运行时（memory / sqlite），避免用户库状态跨用例泄漏。
测试应用只挂载 auth 路由与一个受保护的业务探针，用于验证沙箱与凭据策略。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from engine.auth.config import AuthSettings
from engine.auth.routes import register_auth_exception_handler, router, well_known_router
from engine.auth.runtime import AuthRuntime, build_runtime, reset_auth_runtime_for_tests
from engine.observability.envelope import register_envelope_exception_handlers
from engine.security.auth import get_current_identity, require_permission
from engine.security.identity import Identity

ADMIN = "rootadmin"
ADMIN_PASSWORD = "Adm1n-Passw0rd!"
ADMIN_PASSWORD_NEW = "Adm1n-Passw0rd-2!"
STATIC_TOKEN = "sk-auth-conformance"
STATIC_KEY_NAME = "conformance-caller"
MASK_ONLY_TOKEN = "sk-mask-only"


def configure_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "PRIVACY_AUTH_INTERNAL_KEYS_JSON",
        f'{{"{STATIC_TOKEN}":{{"name":"{STATIC_KEY_NAME}","scopes":["*"]}}}}',
    )
    monkeypatch.setenv(
        "PRIVACY_AUTH_EXTERNAL_KEYS_JSON",
        f'{{"{MASK_ONLY_TOKEN}":{{"name":"mask-only","scopes":["privacy:mask"]}}}}',
    )
    monkeypatch.setenv("PRIVACY_AUTH_USER_STORE_BACKEND", "memory")
    monkeypatch.setenv("PRIVACY_AUTH_USER_BOOTSTRAP_REMOTE_ALLOWED", "true")
    monkeypatch.setenv("PRIVACY_AUTH_USER_LOGIN_THROTTLE_PER_MIN", "1000")
    monkeypatch.setenv("PRIVACY_AUTH_ADMIN_STEPUP_REQUIRED", "false")
    monkeypatch.setenv("PRIVACY_AUTH_BCRYPT_COST", "4")
    monkeypatch.setenv("PRIVACY_AUTH_SM3_ITERATIONS", "1000")
    monkeypatch.delenv("PRIVACY_AUTH_USER_STORE_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("AUTH_TOTP_ALGORITHM", raising=False)
    monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "false")


@pytest.fixture
def auth_runtime(monkeypatch: pytest.MonkeyPatch) -> AuthRuntime:
    configure_auth_env(monkeypatch)
    runtime = build_runtime(AuthSettings.from_env())
    reset_auth_runtime_for_tests(runtime)
    try:
        yield runtime
    finally:
        reset_auth_runtime_for_tests(None)


@pytest.fixture
def api_client(auth_runtime: AuthRuntime) -> TestClient:
    app = FastAPI()
    register_envelope_exception_handlers(app)
    register_auth_exception_handler(app)

    @app.get("/v1/privacy/probe")
    async def privacy_probe(identity: Identity = Depends(get_current_identity)) -> dict[str, Any]:
        return {"subject": identity.subject, "name": identity.name, "scopes": identity.scopes}

    @app.post(
        "/v1/privacy/mask-probe",
        dependencies=[require_permission("privacy:mask")],
    )
    async def mask_probe() -> dict[str, bool]:
        return {"ok": True}

    @app.post(
        "/v1/privacy/dp-probe",
        dependencies=[require_permission("privacy:dp")],
    )
    async def dp_probe() -> dict[str, bool]:
        return {"ok": True}

    @app.api_route("/health", methods=["GET", "HEAD"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)
    app.include_router(well_known_router)
    return TestClient(app)


def bootstrap_admin(client: TestClient) -> None:
    response = client.post(
        "/v1/auth/users/bootstrap",
        json={"username": ADMIN, "password": ADMIN_PASSWORD, "display_name": "Root Admin"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["username"] == ADMIN


def login(
    client: TestClient, *, password: str = ADMIN_PASSWORD_NEW, **extra: str
) -> dict[str, Any]:
    response = client.post(
        "/v1/auth/login",
        json={"username": ADMIN, "password": password, **extra},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Login uses the Go success envelope; keep helpers returning the payload.
    return body.get("data", body)


def change_initial_password(client: TestClient) -> None:
    """Bootstrap → login → change the first-login password."""
    bootstrap_admin(client)
    first = login(client, password=ADMIN_PASSWORD)
    response = client.post(
        "/v1/auth/change-password",
        json={
            "username": ADMIN,
            "old_password": ADMIN_PASSWORD,
            "new_password": ADMIN_PASSWORD_NEW,
        },
        headers={"Authorization": f"Bearer {first['token']}"},
    )
    assert response.status_code == 200, response.text
