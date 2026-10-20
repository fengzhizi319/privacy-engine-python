"""Tests for REST/gRPC TLS and mTLS.

使用动态生成的自签名证书，验证：
- REST 仅服务端 TLS 下，受信 CA 可建立连接，不受信 CA 失败。
- gRPC TLS/mTLS 下，受信 CA/客户端证书可建立连接，不受信 CA/缺失客户端证书失败。
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from concurrent import futures
from typing import TYPE_CHECKING, Any

import grpc
import httpx
import pytest

pytestmark = pytest.mark.integration
import uvicorn

from engine import privacy_pb2, privacy_pb2_grpc
from engine.grpc_server import PrivacyServicer
from engine.main import app
from engine.security.config import get_security_settings
from engine.security.tls import grpc_server_credentials, uvicorn_ssl_kwargs
from tests.security_certs import generate_test_certs

if TYPE_CHECKING:
    from pathlib import Path


def _free_port() -> int:
    """Return an ephemeral TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def certs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate an ephemeral certificate chain once for the whole module."""
    return generate_test_certs(tmp_path_factory.mktemp("test-certs"))


# ---------------------------------------------------------------------------
# REST TLS helpers and tests
# ---------------------------------------------------------------------------


class _RestServer:
    """Tiny wrapper around a Uvicorn server running in a daemon thread."""

    def __init__(self, port: int, ssl_kwargs: dict[str, Any]):
        self._port = port
        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                **ssl_kwargs,
            )
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        # Wait until /health responds.
        last_error = ""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with httpx.Client(verify=str(self._ssl_ca)) as client:
                    resp = client.get(f"https://127.0.0.1:{self._port}/health")
                    if resp.status_code == 200:
                        return
                    last_error = f"unexpected status {resp.status_code}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.05)
        raise RuntimeError(f"REST server did not start in time: {last_error}")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


@contextlib.contextmanager
def _rest_tls_server(certs: dict[str, Path], client_auth: str = "none"):
    """Context manager that starts the REST server with TLS enabled."""
    os.environ["PRIVACY_TLS_ENABLED"] = "true"
    os.environ["PRIVACY_TLS_CERT_FILE"] = str(certs["server_cert"])
    os.environ["PRIVACY_TLS_KEY_FILE"] = str(certs["server_key"])
    os.environ["PRIVACY_TLS_CLIENT_AUTH"] = client_auth
    if client_auth in ("optional", "require"):
        os.environ["PRIVACY_TLS_CA_FILE"] = str(certs["ca_cert"])
    else:
        os.environ.pop("PRIVACY_TLS_CA_FILE", None)

    port = _free_port()
    ssl_kwargs = uvicorn_ssl_kwargs(get_security_settings())
    server = _RestServer(port, ssl_kwargs)
    server._ssl_ca = certs["ca_cert"]  # type: ignore[attr-defined]
    try:
        server.start()
        yield port
    finally:
        server.stop()
        os.environ.pop("PRIVACY_TLS_ENABLED", None)
        os.environ.pop("PRIVACY_TLS_CERT_FILE", None)
        os.environ.pop("PRIVACY_TLS_KEY_FILE", None)
        os.environ.pop("PRIVACY_TLS_CLIENT_AUTH", None)
        os.environ.pop("PRIVACY_TLS_CA_FILE", None)


def test_rest_tls_trusted_ca(certs: dict[str, Path]):
    """使用受信 CA 可成功访问 HTTPS 健康检查。"""
    with _rest_tls_server(certs) as port, httpx.Client(verify=str(certs["ca_cert"])) as client:
        resp = client.get(f"https://127.0.0.1:{port}/health")
        assert resp.status_code == 200


def test_rest_tls_untrusted_ca_fails(certs: dict[str, Path]):
    """使用不受信 CA 访问 HTTPS 时握手失败。"""
    with (
        _rest_tls_server(certs) as port,
        httpx.Client(verify=str(certs["bad_ca_cert"])) as client,
        pytest.raises(httpx.ConnectError),
    ):
        client.get(f"https://127.0.0.1:{port}/health")


# ---------------------------------------------------------------------------
# gRPC TLS helpers and tests
# ---------------------------------------------------------------------------


def _start_grpc_server(port: int, certs: dict[str, Path], client_auth: str = "none") -> grpc.Server:
    """Start a gRPC server with the requested TLS/client-auth configuration."""
    os.environ["PRIVACY_TLS_ENABLED"] = "true"
    os.environ["PRIVACY_TLS_CERT_FILE"] = str(certs["server_cert"])
    os.environ["PRIVACY_TLS_KEY_FILE"] = str(certs["server_key"])
    os.environ["PRIVACY_TLS_CLIENT_AUTH"] = client_auth
    if client_auth in ("optional", "require"):
        os.environ["PRIVACY_TLS_CA_FILE"] = str(certs["ca_cert"])
    else:
        os.environ.pop("PRIVACY_TLS_CA_FILE", None)

    settings = get_security_settings()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    privacy_pb2_grpc.add_PrivacyServiceServicer_to_server(PrivacyServicer(), server)
    creds = grpc_server_credentials(settings)
    server.add_secure_port(f"127.0.0.1:{port}", creds)
    server.start()
    return server


@contextlib.contextmanager
def _grpc_tls_server(certs: dict[str, Path], client_auth: str = "none"):
    """Context manager that starts the gRPC server with TLS enabled."""
    port = _free_port()
    server = _start_grpc_server(port, certs, client_auth)
    try:
        yield port
    finally:
        server.stop(0)
        os.environ.pop("PRIVACY_TLS_ENABLED", None)
        os.environ.pop("PRIVACY_TLS_CERT_FILE", None)
        os.environ.pop("PRIVACY_TLS_KEY_FILE", None)
        os.environ.pop("PRIVACY_TLS_CLIENT_AUTH", None)
        os.environ.pop("PRIVACY_TLS_CA_FILE", None)


def _grpc_channel(
    port: int,
    ca_cert: Path | None = None,
    client_cert: Path | None = None,
    client_key: Path | None = None,
) -> grpc.Channel:
    """Build a gRPC secure channel with optional mTLS client credentials."""
    ca = ca_cert.read_bytes() if ca_cert else None
    key = client_key.read_bytes() if client_key else None
    cert = client_cert.read_bytes() if client_cert else None
    creds = grpc.ssl_channel_credentials(
        root_certificates=ca,
        private_key=key,
        certificate_chain=cert,
    )
    channel = grpc.secure_channel(f"127.0.0.1:{port}", creds)
    grpc.channel_ready_future(channel).result(timeout=5)
    return channel


def test_grpc_tls_trusted_ca(certs: dict[str, Path]):
    """使用受信 CA 可成功建立 gRPCs 连接并调用 Health。"""
    with _grpc_tls_server(certs) as port, _grpc_channel(port, ca_cert=certs["ca_cert"]) as channel:
        stub = privacy_pb2_grpc.PrivacyServiceStub(channel)
        resp = stub.Health(privacy_pb2.HealthRequest())
        assert resp.status == "ok"


def test_grpc_tls_untrusted_ca_fails(certs: dict[str, Path]):
    """使用不受信 CA 时 gRPCs 连接失败。"""
    with (
        _grpc_tls_server(certs) as port,
        pytest.raises(grpc.FutureTimeoutError),
        _grpc_channel(port, ca_cert=certs["bad_ca_cert"]),
    ):
        pass


def test_grpc_mtls_require_client_cert(certs: dict[str, Path]):
    """gRPC mTLS require 模式下必须提供受信客户端证书。"""
    with _grpc_tls_server(certs, client_auth="require") as port:
        # No client cert -> handshake fails.
        with pytest.raises(grpc.FutureTimeoutError), _grpc_channel(port, ca_cert=certs["ca_cert"]) as channel:
            pass

        # Trusted client cert -> succeeds.
        with _grpc_channel(
            port,
            ca_cert=certs["ca_cert"],
            client_cert=certs["client_cert"],
            client_key=certs["client_key"],
        ) as channel:
            stub = privacy_pb2_grpc.PrivacyServiceStub(channel)
            resp = stub.Health(privacy_pb2.HealthRequest())
            assert resp.status == "ok"
