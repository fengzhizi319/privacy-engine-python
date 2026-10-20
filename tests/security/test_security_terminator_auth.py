"""Integration of verified terminator identity with REST, gRPC and /metrics auth."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request

from engine.auth.errors import AuthError
from engine.auth.runtime import reset_auth_runtime_for_tests
from engine.security.auth import (
    ApiKeyAuthAsgiMiddleware,
    _extract_identity_from_grpc_context,
    _grpc_terminator_peer,
    get_current_identity,
)
from engine.security.config import get_security_settings
from engine.security.identity import permission_for_grpc_method
from engine.security.terminator import reset_key_cache
from engine.security.whitelist import reset_whitelist_manager

from .test_security_terminator import KEY, identity_headers

CN = "grpc-client"
STATIC_TOKEN = "sk-terminator-rest"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    reset_auth_runtime_for_tests(None)
    reset_whitelist_manager()
    reset_key_cache()
    monkeypatch.setenv("PRIVACY_TERMINATOR_IDENTITY_ENABLED", "true")
    monkeypatch.setenv("PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO", "sm3")
    monkeypatch.setenv("PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY", KEY.decode())
    monkeypatch.setenv("PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS", "60")
    yield
    reset_auth_runtime_for_tests(None)
    reset_whitelist_manager()
    reset_key_cache()


def _fresh_headers(**overrides: Any) -> dict[str, list[str]]:
    overrides.setdefault("timestamp", int(time.time()))
    return identity_headers(**overrides)


def _request(
    path: str,
    headers: dict[str, list[str]],
    *,
    method: str = "GET",
    client: tuple[str, int] | None = ("127.0.0.1", 41234),
) -> Request:
    raw_headers: list[tuple[bytes, bytes]] = []
    for name, values in headers.items():
        for value in values:
            raw_headers.append((name.lower().encode("latin-1"), value.encode("latin-1")))
    scope: dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": raw_headers,
        "server": ("engine", 18079),
    }
    if client is not None:
        scope["client"] = client
    return Request(scope)


def _asgi_scope(
    path: str,
    headers: dict[str, list[str]],
    *,
    client: tuple[str, int] | None = ("127.0.0.1", 41234),
) -> dict[str, Any]:
    scope = _request(path, headers, client=client).scope
    return dict(scope)


def _run_middleware(scope: dict[str, Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def app(inner_scope: dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> dict[str, Any]:
        return {"type": "http.request"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(ApiKeyAuthAsgiMiddleware(app)(scope, receive, send))
    return sent


def _status(messages: list[dict[str, Any]]) -> int:
    start = next(msg for msg in messages if msg["type"] == "http.response.start")
    return int(start["status"])


class _FakeGRPCContext:
    def __init__(self, metadata: list[tuple[str, str]], peer: str):
        self._metadata = metadata
        self._peer = peer

    def invocation_metadata(self) -> list[tuple[str, str]]:
        return self._metadata

    def peer(self) -> str:
        return self._peer

    def auth_context(self) -> dict[str, Any]:
        return {}


def test_rest_populates_verified_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "PRIVACY_AUTH_INTERNAL_KEYS_JSON",
        f'{{"{STATIC_TOKEN}":{{"name":"rest-caller","scopes":["*"]}}}}',
    )
    headers = _fresh_headers(method="GET", path="/v1/privacy/probe")
    headers["authorization"] = [f"Bearer {STATIC_TOKEN}"]

    request = _request("/v1/privacy/probe", headers)
    identity = asyncio.run(get_current_identity(request))

    assert identity.name == "rest-caller"
    peer = request.state.terminator_peer
    assert peer is not None
    assert peer.common_name == "app-lz-client"


def test_rest_rejects_forged_terminator_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "PRIVACY_AUTH_INTERNAL_KEYS_JSON",
        f'{{"{STATIC_TOKEN}":{{"name":"rest-caller","scopes":["*"]}}}}',
    )
    headers = _fresh_headers(method="GET", path="/v1/privacy/probe", key=b"forged-key-0123456789")
    headers["authorization"] = [f"Bearer {STATIC_TOKEN}"]

    with pytest.raises(AuthError) as excinfo:
        asyncio.run(get_current_identity(_request("/v1/privacy/probe", headers)))
    assert excinfo.value.status_code == 401
    assert excinfo.value.code == "UNAUTHENTICATED"


def test_rest_rejects_terminator_headers_from_non_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "PRIVACY_AUTH_INTERNAL_KEYS_JSON",
        f'{{"{STATIC_TOKEN}":{{"name":"rest-caller","scopes":["*"]}}}}',
    )
    headers = _fresh_headers(method="GET", path="/v1/privacy/probe")
    headers["authorization"] = [f"Bearer {STATIC_TOKEN}"]

    with pytest.raises(AuthError) as excinfo:
        asyncio.run(
            get_current_identity(
                _request("/v1/privacy/probe", headers, client=("10.0.0.9", 50000))
            )
        )
    assert excinfo.value.status_code == 401


def _write_whitelist(tmp_path: Path, allowed_scopes: list[str]) -> Path:
    scopes = ", ".join(f'"{item}"' for item in allowed_scopes)
    path = tmp_path / "whitelist.yaml"
    path.write_text(
        "version: \"1.0\"\n"
        "clients:\n"
        f"  - cn: \"{CN}\"\n"
        f"    allowed_scopes: [{scopes}]\n",
        encoding="utf-8",
    )
    return path


def test_grpc_terminator_identity_becomes_mtls_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    method = "/privacy.local.PrivacyService/Mask"
    monkeypatch.setenv(
        "PRIVACY_AUTH_MTLS_WHITELIST_FILE", str(_write_whitelist(tmp_path, [method]))
    )
    monkeypatch.setenv("PRIVACY_AUTH_INTERNAL_MTLS_ENABLED", "true")
    settings = get_security_settings()
    headers = _fresh_headers(method="POST", path=method, cn=CN)
    context = _FakeGRPCContext(
        [(name, values[0]) for name, values in headers.items()],
        "ipv4:127.0.0.1:41234",
    )

    peer = _grpc_terminator_peer(settings, context, method)
    assert peer is not None

    identity = _extract_identity_from_grpc_context(settings, context, method, peer)
    assert identity is not None
    assert identity.name == CN
    assert identity.scopes == [permission_for_grpc_method(method)]


def test_grpc_terminator_identity_respects_method_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "PRIVACY_AUTH_MTLS_WHITELIST_FILE",
        str(_write_whitelist(tmp_path, ["/privacy.local.PrivacyService/Mask"])),
    )
    monkeypatch.setenv("PRIVACY_AUTH_INTERNAL_MTLS_ENABLED", "true")
    settings = get_security_settings()

    method = "/privacy.local.PrivacyService/DPCount"
    headers = _fresh_headers(method="POST", path=method, cn=CN)
    context = _FakeGRPCContext(
        [(name, values[0]) for name, values in headers.items()],
        "ipv4:127.0.0.1:41234",
    )
    peer = _grpc_terminator_peer(settings, context, method)
    assert peer is not None

    with pytest.raises(AuthError) as excinfo:
        _extract_identity_from_grpc_context(settings, context, method, peer)
    assert excinfo.value.code == "PERMISSION_DENIED"


def test_grpc_terminator_identity_requires_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PRIVACY_AUTH_MTLS_WHITELIST_FILE", str(_write_whitelist(tmp_path, ["*"])))
    monkeypatch.setenv("PRIVACY_AUTH_INTERNAL_MTLS_ENABLED", "false")
    settings = get_security_settings()

    method = "/privacy.local.PrivacyService/Mask"
    headers = _fresh_headers(method="POST", path=method, cn=CN)
    context = _FakeGRPCContext(
        [(name, values[0]) for name, values in headers.items()],
        "ipv4:127.0.0.1:41234",
    )
    peer = _grpc_terminator_peer(settings, context, method)
    assert peer is not None
    assert _extract_identity_from_grpc_context(settings, context, method, peer) is None


def _metrics_env(monkeypatch: pytest.MonkeyPatch, cns: str = "monitor") -> None:
    monkeypatch.setenv("PRIVACY_METRICS_ALLOW_MTLS", "true")
    monkeypatch.setenv("PRIVACY_METRICS_ALLOWED_CLIENT_CNS", cns)
    monkeypatch.delenv("PRIVACY_METRICS_API_KEY", raising=False)
    monkeypatch.delenv("PRIVACY_METRICS_API_KEY_PREVIOUS", raising=False)
    monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "false")


def test_metrics_mtls_admission_accepts_verified_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    _metrics_env(monkeypatch)
    headers = _fresh_headers(method="GET", path="/metrics", cn="monitor")
    assert _status(_run_middleware(_asgi_scope("/metrics", headers))) == 200


def test_metrics_mtls_admission_rejects_unlisted_cn(monkeypatch: pytest.MonkeyPatch) -> None:
    _metrics_env(monkeypatch)
    headers = _fresh_headers(method="GET", path="/metrics", cn="intruder")
    assert _status(_run_middleware(_asgi_scope("/metrics", headers))) == 401


def test_metrics_mtls_admission_rejects_forged_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metrics_env(monkeypatch)
    headers = _fresh_headers(
        method="GET", path="/metrics", cn="monitor", key=b"forged-key-0123456789"
    )
    assert _status(_run_middleware(_asgi_scope("/metrics", headers))) == 401


def test_metrics_without_credentials_is_denied_when_mtls_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metrics_env(monkeypatch)
    assert _status(_run_middleware(_asgi_scope("/metrics", {}))) == 401
