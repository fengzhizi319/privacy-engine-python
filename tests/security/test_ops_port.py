"""Ops port allowlist / TLS settings and dedicated metrics-key admission tests."""

from __future__ import annotations

import asyncio
import json
import ssl

import pytest

from engine.ops_port import (
    OPS_ALLOWED_PATHS,
    OpsAllowlistMiddleware,
    default_ops_port,
    is_loopback_host,
    load_ops_settings,
    ops_uvicorn_ssl_kwargs,
)
from engine.security.auth import ApiKeyAuthAsgiMiddleware
from engine.security.config import SecuritySettings, metrics_auth_mode
from engine.security.tls import uvicorn_ssl_kwargs


async def _ok_app(scope, receive, send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def _call(app, path: str, headers: list[tuple[bytes, bytes]] | None = None, scope_type: str = "http"):
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": scope_type, "method": "GET", "path": path, "headers": headers or []}
    await app(scope, receive, send)
    return sent


def _status(sent: list[dict]) -> int:
    for message in sent:
        if message["type"] == "http.response.start":
            return int(message["status"])
    raise AssertionError("no response.start message")


def _body(sent: list[dict]) -> bytes:
    return b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )


@pytest.fixture
def _clean_env(monkeypatch):
    for name in (
        "PRIVACY_OPS_HOST",
        "PRIVACY_OPS_PORT",
        "PRIVACY_OPS_TLS_STACK",
        "PRIVACY_OPS_TLS_CERT_FILE",
        "PRIVACY_OPS_TLS_KEY_FILE",
        "PRIVACY_OPS_TLS_CA_FILE",
        "PRIVACY_OPS_TLS_KEY_PASSWORD",
        "PRIVACY_METRICS_API_KEY",
        "PRIVACY_METRICS_API_KEY_PREVIOUS",
        "PRIVACY_METRICS_ALLOW_MTLS",
        "PRIVACY_AUTH_ENABLED",
        "APP_ENV",
        "PRIVACY_ENV",
        "SERVICE_HUB_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch


def test_default_ops_port_is_business_plus_1000():
    assert default_ops_port(8079) == 9079
    assert default_ops_port(0) == 0
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("10.0.0.1")


@pytest.mark.parametrize(
    "path",
    ["/health", "/healthz", "/livez", "/readyz", "/metrics", "/metrics/",
     "/ops/diagnostics", "/debug/pprof/", "/debug/pprof", "/debug/pprof/heap"],
)
def test_ops_allowlist_passes_probe_metrics_and_diagnostics(path):
    sent = asyncio.run(_call(OpsAllowlistMiddleware(_ok_app), path))
    assert _status(sent) == 200
    assert _body(sent) == b"ok"


def test_ops_allowlist_denies_business_paths_with_go_404_body():
    sent = asyncio.run(_call(OpsAllowlistMiddleware(_ok_app), "/v1/privacy/mask"))
    assert _status(sent) == 404
    payload = json.loads(_body(sent))
    assert payload == {
        "code": "NOT_FOUND",
        "message": "ops port only serves probe, metrics and diagnostics paths",
    }


def test_ops_allowlist_passes_non_http_scopes():
    calls: list[str] = []

    async def recorder(scope, receive, send) -> None:
        calls.append(scope["type"])

    asyncio.run(_call(OpsAllowlistMiddleware(recorder), "/v1/privacy/mask", scope_type="lifespan"))
    assert calls == ["lifespan"]


def test_ops_settings_default_inherits_business_listener(_clean_env):
    settings = SecuritySettings()
    ops = load_ops_settings("0.0.0.0", 8079, settings)
    assert (ops.host, ops.port, ops.tls_stack) == ("0.0.0.0", 9079, "")
    assert ops_uvicorn_ssl_kwargs(ops, settings) == uvicorn_ssl_kwargs(settings) == {}


def test_ops_settings_random_business_port_forces_random_ops_port(_clean_env):
    ops = load_ops_settings("127.0.0.1", 0, SecuritySettings())
    assert ops.port == 0


def test_ops_settings_rejects_tlcp_on_non_loopback(_clean_env):
    _clean_env.setenv("PRIVACY_OPS_TLS_STACK", "both")
    with pytest.raises(ValueError, match="cannot be terminated by the Python engine"):
        load_ops_settings("0.0.0.0", 8079, SecuritySettings())


def test_ops_settings_allows_tlcp_on_loopback_for_terminator(_clean_env):
    _clean_env.setenv("PRIVACY_OPS_TLS_STACK", "tlcp")
    ops = load_ops_settings("127.0.0.1", 8079, SecuritySettings())
    assert ops.terminator_fronted
    assert ops_uvicorn_ssl_kwargs(ops, SecuritySettings()) == {}


def test_ops_settings_tls_requires_full_mtls_material(_clean_env):
    _clean_env.setenv("PRIVACY_OPS_TLS_STACK", "tls")
    with pytest.raises(ValueError, match="requires PRIVACY_OPS_TLS_CERT_FILE"):
        load_ops_settings("0.0.0.0", 8079, SecuritySettings())


def test_ops_settings_tls_builds_mtls_kwargs(_clean_env, tmp_path):
    cert = tmp_path / "ops.crt"
    key = tmp_path / "ops.key"
    ca = tmp_path / "ops-ca.crt"
    for path in (cert, key, ca):
        path.write_text("placeholder", encoding="utf-8")
    _clean_env.setenv("PRIVACY_OPS_TLS_STACK", "tls")
    _clean_env.setenv("PRIVACY_OPS_TLS_CERT_FILE", str(cert))
    _clean_env.setenv("PRIVACY_OPS_TLS_KEY_FILE", str(key))
    _clean_env.setenv("PRIVACY_OPS_TLS_CA_FILE", str(ca))

    ops = load_ops_settings("0.0.0.0", 8079, SecuritySettings())
    kwargs = ops_uvicorn_ssl_kwargs(ops, SecuritySettings())
    assert kwargs["ssl_certfile"] == str(cert)
    assert kwargs["ssl_keyfile"] == str(key)
    assert kwargs["ssl_ca_certs"] == str(ca)
    assert kwargs["ssl_cert_reqs"] == ssl.CERT_REQUIRED


def test_ops_settings_inherit_business_tls(_clean_env, tmp_path):
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    ca = tmp_path / "ca.crt"
    for path in (cert, key, ca):
        path.write_text("placeholder", encoding="utf-8")
    settings = SecuritySettings(
        tls_enabled=True,
        tls_cert_file=cert,
        tls_key_file=key,
        tls_ca_file=ca,
        tls_client_auth="require",
    )
    ops = load_ops_settings("0.0.0.0", 8079, settings)
    assert ops_uvicorn_ssl_kwargs(ops, settings) == uvicorn_ssl_kwargs(settings)


def _asgi_headers(token: str | None) -> list[tuple[bytes, bytes]]:
    if token is None:
        return []
    return [(b"authorization", f"Bearer {token}".encode("utf-8"))]


def test_metrics_key_admits_current_and_previous_keys(_clean_env):
    _clean_env.setenv("PRIVACY_METRICS_API_KEY", "metrics-key")
    _clean_env.setenv("PRIVACY_METRICS_API_KEY_PREVIOUS", "metrics-old")
    middleware = ApiKeyAuthAsgiMiddleware(_ok_app)

    assert _status(asyncio.run(_call(middleware, "/metrics", _asgi_headers("metrics-key")))) == 200
    assert _status(asyncio.run(_call(middleware, "/metrics", _asgi_headers("metrics-old")))) == 200
    assert _status(asyncio.run(_call(middleware, "/metrics", _asgi_headers("wrong")))) == 401
    assert _status(asyncio.run(_call(middleware, "/metrics"))) == 401


def test_metrics_anonymous_only_in_dev_posture(_clean_env):
    middleware = ApiKeyAuthAsgiMiddleware(_ok_app)
    assert _status(asyncio.run(_call(middleware, "/metrics"))) == 200

    _clean_env.setenv("APP_ENV", "production")
    assert _status(asyncio.run(_call(middleware, "/metrics"))) == 401


def test_metrics_auth_mode_matches_go_strategies(_clean_env):
    dev = metrics_auth_mode(SecuritySettings())
    assert dev == ("dev_anonymous", ["dev_anonymous"])

    keyed = metrics_auth_mode(SecuritySettings(metrics_api_key="metrics-key"))
    assert keyed == ("bearer_key", ["bearer_key"])

    scoped = metrics_auth_mode(
        SecuritySettings(auth_enabled=True, metrics_api_key="metrics-key")
    )
    assert scoped == ("bearer_key", ["bearer_key", "scope"])

    _clean_env.setenv("APP_ENV", "production")
    prod = metrics_auth_mode(SecuritySettings())
    assert prod == ("scope", [])


def test_ops_default_allowlist_is_go_parity():
    assert OPS_ALLOWED_PATHS == (
        "/health",
        "/healthz",
        "/livez",
        "/readyz",
        "/metrics",
        "/metrics/",
        "/ops/diagnostics",
        "/debug/pprof/",
    )
