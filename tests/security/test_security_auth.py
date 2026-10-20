"""Tests for REST authentication and authorization.

验证开启认证后：内部 API Key 可访问全部接口，外部 API Key 受 scope 限制，
缺失/无效凭证返回 401，越权返回 403，健康检查默认保持开放。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine.main import app

client = TestClient(app)

INTERNAL_TOKEN = "sk-internal-test"
EXTERNAL_TOKEN = "sk-external-test"


@pytest.fixture(autouse=True)
def _disable_rate_limit(monkeypatch):
    """Ensure rate limiting does not interfere with auth tests."""
    monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "false")
    yield


@pytest.fixture
def auth_enabled(monkeypatch):
    """Enable auth with one internal and one external key."""
    monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "PRIVACY_AUTH_INTERNAL_KEYS_JSON",
        f'{{"{INTERNAL_TOKEN}":{{"name":"secretpad","scopes":["*"]}}}}',
    )
    monkeypatch.setenv(
        "PRIVACY_AUTH_EXTERNAL_KEYS_JSON",
        f'{{"{EXTERNAL_TOKEN}":{{"name":"portal","scopes":["privacy:mask"]}}}}',
    )
    yield


def test_no_auth_by_default():
    """默认未启用认证时，业务接口可直接访问。"""
    response = client.post("/v1/privacy/mask", json={"field_name": "mobile", "value": "13812345678"})
    assert response.status_code == 200


def test_auth_missing_token(auth_enabled):
    """启用认证后，不带凭证访问业务接口返回 401。"""
    response = client.post("/v1/privacy/mask", json={"field_name": "mobile", "value": "13812345678"})
    assert response.status_code == 401


def test_auth_invalid_token(auth_enabled):
    """无效 token 返回 401。"""
    response = client.post(
        "/v1/privacy/mask",
        json={"field_name": "mobile", "value": "13812345678"},
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


def test_internal_token_full_access(auth_enabled):
    """内部服务 token 拥有通配权限，可调用所有接口。"""
    headers = {"Authorization": f"Bearer {INTERNAL_TOKEN}"}

    response = client.post(
        "/v1/privacy/mask", json={"field_name": "mobile", "value": "13812345678"}, headers=headers
    )
    assert response.status_code == 200

    response = client.post(
        "/v1/privacy/dp/count",
        json={"values": [1.0, 0.0, 1.0], "params": {"epsilon": 1.0}},
        headers=headers,
    )
    assert response.status_code == 200


def test_external_token_allowed_scope(auth_enabled):
    """外部服务 token 在授权范围内可访问脱敏接口。"""
    response = client.post(
        "/v1/privacy/mask",
        json={"field_name": "mobile", "value": "13812345678"},
        headers={"Authorization": f"Bearer {EXTERNAL_TOKEN}"},
    )
    assert response.status_code == 200


def test_external_token_forbidden_dp(auth_enabled):
    """外部服务 token 越权访问差分隐私接口返回 403。"""
    response = client.post(
        "/v1/privacy/dp/count",
        json={"values": [1.0, 0.0, 1.0], "params": {"epsilon": 1.0}},
        headers={"Authorization": f"Bearer {EXTERNAL_TOKEN}"},
    )
    assert response.status_code == 403


def test_health_exempt_by_default(auth_enabled):
    """默认配置下健康检查不受认证限制。"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_requires_auth_when_configured(auth_enabled, monkeypatch):
    """当 PRIVACY_HEALTH_NO_AUTH=false 时，健康检查也需要凭证。"""
    monkeypatch.setenv("PRIVACY_HEALTH_NO_AUTH", "false")
    response = client.get("/health")
    assert response.status_code == 401

    response = client.get("/health", headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"})
    assert response.status_code == 200


def test_security_headers_present():
    """验证 REST 响应中包含通用的安全响应头。"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"

