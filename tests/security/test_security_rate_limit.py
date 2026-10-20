"""Tests for REST rate limiting.

验证开启限速后：超速调用返回 429，健康检查默认不限速。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine.main import app

client = TestClient(app)


@pytest.fixture
def rate_limit_enabled(monkeypatch):
    """Enable rate limiting with a very tight limit on the mask endpoint."""
    monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("PRIVACY_RATE_LIMIT_DEFAULT_RPS", "100")
    monkeypatch.setenv("PRIVACY_RATE_LIMIT_DEFAULT_BURST", "100")
    monkeypatch.setenv(
        "PRIVACY_RATE_LIMIT_PER_ENDPOINT_JSON",
        '{"/v1/privacy/mask":{"rps":1,"burst":1}}',
    )
    yield


def test_rate_limit_blocks_excess(rate_limit_enabled):
    """超出限流阈值后，REST 返回 429。"""
    # First request should succeed.
    response = client.post("/v1/privacy/mask", json={"field_name": "mobile", "value": "13812345678"})
    assert response.status_code == 200

    # Second request within the same window should be throttled.
    response = client.post("/v1/privacy/mask", json={"field_name": "mobile", "value": "13912345678"})
    assert response.status_code == 429


def test_rate_limit_health_exempt(rate_limit_enabled):
    """健康检查默认不受限速影响。"""
    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200


def test_rate_limit_other_endpoints_use_default(rate_limit_enabled):
    """未单独配置限流的接口使用默认阈值，连续少量请求不被拦截。"""
    for i in range(3):
        response = client.post(
            "/v1/privacy/hash",
            json={"value": f"value-{i}", "salt": "salt"},
        )
        assert response.status_code == 200
