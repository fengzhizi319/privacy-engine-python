"""REST route-table and credential-policy parity with Go ``pkg/auth``.

对照 ``pkg/auth/user_handlers*.go`` 的 Gin 注册表，锁定 Python 侧同构路由；
同时验证未认证 / 无效凭证 / 公开路径 / 健康探针 / scope 判定的 HTTP 行为。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from engine.auth.routes import router as auth_router
from engine.auth.routes import well_known_router

from .conftest import MASK_ONLY_TOKEN, STATIC_KEY_NAME, STATIC_TOKEN

# Go ``RegisterUserRoutes`` + ``RegisterTokenExchangeRoutes`` 的唯一路由集合。
GO_AUTH_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/v1/auth/login"),
        ("POST", "/v1/auth/register"),
        ("POST", "/v1/auth/users/register"),
        ("POST", "/v1/auth/users/bootstrap"),
        ("POST", "/v1/auth/logout"),
        ("POST", "/v1/auth/change-password"),
        ("GET", "/v1/auth/permissions"),
        ("POST", "/v1/auth/users"),
        ("GET", "/v1/auth/users"),
        ("GET", "/v1/auth/users/:username"),
        ("PUT", "/v1/auth/users/:username/permissions"),
        ("PUT", "/v1/auth/users/:username/status"),
        ("PUT", "/v1/auth/users/:username/certificates"),
        ("DELETE", "/v1/auth/users/:username/certificates"),
        ("POST", "/v1/auth/users/:username/reset-password"),
        ("DELETE", "/v1/auth/users/:username"),
        ("POST", "/v1/auth/users/:username/keys"),
        ("GET", "/v1/auth/users/:username/keys"),
        ("DELETE", "/v1/auth/users/:username/keys/:key_id"),
        ("POST", "/v1/auth/mfa/setup"),
        ("POST", "/v1/auth/mfa/setup/cancel"),
        ("POST", "/v1/auth/mfa/enable"),
        ("POST", "/v1/auth/mfa/disable"),
        ("GET", "/v1/auth/mfa/status"),
        ("POST", "/v1/auth/mfa/recovery-codes/regenerate"),
        ("POST", "/v1/auth/service-accounts"),
        ("GET", "/v1/auth/service-accounts"),
        ("POST", "/v1/auth/service-accounts/activate"),
        ("GET", "/v1/auth/service-accounts/:username"),
        ("POST", "/v1/auth/service-accounts/:username/freeze"),
        ("POST", "/v1/auth/service-accounts/:username/unfreeze"),
        ("PUT", "/v1/auth/service-accounts/:username/quota"),
        ("POST", "/v1/auth/service-accounts/:username/keys"),
        ("GET", "/v1/auth/service-accounts/:username/keys"),
        ("DELETE", "/v1/auth/service-accounts/:username/keys/:key_id"),
        ("POST", "/v1/auth/service-accounts/:username/keys/:key_id/rotate"),
        ("POST", "/v1/auth/token-exchange"),
        ("POST", "/v1/auth/tokens/revoke"),
        ("GET", "/v1/.well-known/service-capabilities/public"),
        ("POST", "/v1/.well-known/service-capabilities/negotiate"),
    }
)


def _go_style_path(path: str) -> str:
    return path.replace("{", ":").replace("}", "")


def test_auth_route_table_matches_go() -> None:
    actual: set[tuple[str, str]] = set()
    for included in (auth_router, well_known_router):
        for route in included.routes:
            methods = getattr(route, "methods", None)
            path = getattr(route, "path", "")
            if not methods or not path.startswith("/v1/"):
                continue
            for method in methods:
                if method in {"HEAD", "OPTIONS"}:
                    continue
                actual.add((method, _go_style_path(path)))

    assert actual == set(GO_AUTH_ROUTES)


def test_missing_credentials_returns_go_envelope(api_client: TestClient) -> None:
    response = api_client.get("/v1/privacy/probe")
    assert response.status_code == 401
    body = response.json()
    assert body["code"] == "UNAUTHENTICATED"
    assert body["message"] == "Unauthorized: missing credentials"
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Trace-ID"]


@pytest.mark.parametrize("header", ["Bearer invalid-token", "bare-token-without-scheme"])
def test_invalid_credentials_returns_401(api_client: TestClient, header: str) -> None:
    response = api_client.get("/v1/privacy/probe", headers={"Authorization": header})
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHENTICATED"


def test_health_probe_is_exempt_for_get_and_head(api_client: TestClient) -> None:
    for method in ("GET", "HEAD"):
        response = api_client.request(method, "/health")
        assert response.status_code == 200


def test_public_auth_path_bypasses_token_requirement(api_client: TestClient) -> None:
    response = api_client.post("/v1/auth/login", json={"username": "nobody"})
    # 没有 Bearer 也进入业务处理器，由载荷校验返回 400 而非 401。
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_ARGUMENT"

    # 但显式携带无效凭证仍必须 401（Go：公开路径不等于忽略坏凭证）。
    response = api_client.post(
        "/v1/auth/login",
        json={"username": "nobody", "password": "x"},
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


def test_scope_enforcement_uses_registered_permission(api_client: TestClient) -> None:
    headers = {"Authorization": f"Bearer {MASK_ONLY_TOKEN}"}
    assert api_client.post("/v1/privacy/mask-probe", headers=headers).status_code == 200
    response = api_client.post("/v1/privacy/dp-probe", headers=headers)
    assert response.status_code == 403

    internal = {"Authorization": f"Bearer {STATIC_TOKEN}"}
    assert api_client.post("/v1/privacy/dp-probe", headers=internal).status_code == 200


def _token_exchange_payload() -> dict[str, Any]:
    return {
        "grant_type": "client_credentials",
        "client_id": STATIC_KEY_NAME,
        "audience": "privacy-engine",
    }


def test_token_exchange_accepts_json_and_form(api_client: TestClient) -> None:
    headers = {"Authorization": f"Bearer {STATIC_TOKEN}"}

    json_response = api_client.post(
        "/v1/auth/token-exchange", json=_token_exchange_payload(), headers=headers
    )
    assert json_response.status_code == 200, json_response.text
    json_token = json_response.json()["access_token"]
    assert json_token.startswith("psk_")

    # Go ``c.ShouldBind`` also accepts application/x-www-form-urlencoded (RFC 6749).
    form_response = api_client.post(
        "/v1/auth/token-exchange", data=_token_exchange_payload(), headers=headers
    )
    assert form_response.status_code == 200, form_response.text
    assert form_response.json()["access_token"].startswith("psk_")


def test_token_exchange_missing_required_field_uses_invalid_request(
    api_client: TestClient,
) -> None:
    headers = {"Authorization": f"Bearer {STATIC_TOKEN}"}
    payload = _token_exchange_payload()
    payload.pop("audience")
    response = api_client.post("/v1/auth/token-exchange", json=payload, headers=headers)
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


def test_token_revocation_stays_json_only(api_client: TestClient) -> None:
    headers = {"Authorization": f"Bearer {STATIC_TOKEN}"}
    response = api_client.post(
        "/v1/auth/tokens/revoke", data={"token": "psk_deadbeef"}, headers=headers
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"
