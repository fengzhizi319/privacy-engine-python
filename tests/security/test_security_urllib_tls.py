"""Tests for urllib.request TLS and mTLS interactions against PrivShield REST service.

验证使用 Python 标准库 urllib.request 与 ssl.SSLContext 发起 TLS/mTLS 请求：
- 单向 TLS：使用受信 CA 成功建立 HTTPS 连接并读取 JSON 响应；
- 单向 TLS：使用不受信 CA 时握手失败（抛出 URLError / SSLError）；
- 单向 TLS：使用 ssl._create_unverified_context() 绕过自签名证书校验；
- 双向 mTLS：服务端 require 客户端证书模式下，提供客户端证书链成功请求，缺失客户端证书则握手失败；
- TLS 加密通道 POST：使用 urllib.request.Request 发送 JSON 载荷调用脱敏接口。
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import uvicorn

from engine.main import app
from engine.security.config import get_security_settings
from engine.security.tls import uvicorn_ssl_kwargs
from tests.security_certs import generate_test_certs

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


def _free_port() -> int:
    """Return an ephemeral TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def certs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate an ephemeral certificate chain once for the whole module."""
    return generate_test_certs(tmp_path_factory.mktemp("urllib-test-certs"))


class _RestTlsServer:
    """Tiny wrapper around a Uvicorn server running in a daemon thread."""

    def __init__(
        self,
        port: int,
        ssl_kwargs: dict[str, Any],
        ca_cert: Path,
        client_cert: Path | None = None,
        client_key: Path | None = None,
    ):
        self._port = port
        self._ca_cert = ca_cert
        self._client_cert = client_cert
        self._client_key = client_key
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
        last_error = ""
        deadline = time.monotonic() + 10
        ssl_ctx = ssl.create_default_context(cafile=str(self._ca_cert))
        if self._client_cert and self._client_key:
            ssl_ctx.load_cert_chain(
                certfile=str(self._client_cert),
                keyfile=str(self._client_key),
            )
        while time.monotonic() < deadline:
            try:
                with httpx.Client(verify=ssl_ctx) as client:
                    resp = client.get(f"https://127.0.0.1:{self._port}/health")
                    if resp.status_code == 200:
                        return
                    last_error = f"unexpected status {resp.status_code}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.05)
        raise RuntimeError(f"REST TLS server did not start in time: {last_error}")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


@contextlib.contextmanager
def _rest_tls_server(certs: dict[str, Path], client_auth: str = "none"):
    """Context manager that starts the REST server with TLS/mTLS enabled."""
    os.environ["PRIVACY_TLS_ENABLED"] = "true"
    os.environ["PRIVACY_TLS_CERT_FILE"] = str(certs["server_cert"])
    os.environ["PRIVACY_TLS_KEY_FILE"] = str(certs["server_key"])
    os.environ["PRIVACY_TLS_CLIENT_AUTH"] = client_auth
    client_cert: Path | None = None
    client_key: Path | None = None
    if client_auth in ("optional", "require"):
        os.environ["PRIVACY_TLS_CA_FILE"] = str(certs["ca_cert"])
        if client_auth == "require":
            client_cert = certs["client_cert"]
            client_key = certs["client_key"]
    else:
        os.environ.pop("PRIVACY_TLS_CA_FILE", None)

    port = _free_port()
    ssl_kwargs = uvicorn_ssl_kwargs(get_security_settings())
    server = _RestTlsServer(
        port, ssl_kwargs, certs["ca_cert"], client_cert=client_cert, client_key=client_key
    )
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


def test_urllib_tls_with_trusted_ca(certs: dict[str, Path]):
    """【单向 TLS】使用 urllib.request.Request + 受信 CA 的 SSLContext 成功请求健康探针。"""
    with _rest_tls_server(certs) as port:
        # 1. 构造 HTTPS 请求对象
        req = urllib.request.Request(f"https://127.0.0.1:{port}/health", method="GET")
        # 2. 构造加载自签名受信 CA 的 SSLContext
        ssl_ctx = ssl.create_default_context(cafile=str(certs["ca_cert"]))

        # 3. 发送请求并验证响应
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") == "ok"


def test_urllib_tls_untrusted_ca_fails(certs: dict[str, Path]):
    """【单向 TLS 负面测试】使用不受信 CA 证书时 TLS 握手校验失败。"""
    with _rest_tls_server(certs) as port:
        req = urllib.request.Request(f"https://127.0.0.1:{port}/health", method="GET")
        # 加载错误的/不受信的 CA
        ssl_ctx = ssl.create_default_context(cafile=str(certs["bad_ca_cert"]))

        with pytest.raises((urllib.error.URLError, ssl.SSLError)):
            urllib.request.urlopen(req, context=ssl_ctx, timeout=3.0)


def test_urllib_tls_unverified_context(certs: dict[str, Path]):
    """【单向 TLS】使用 ssl._create_unverified_context() 绕过证书校验访问。"""
    with _rest_tls_server(certs) as port:
        req = urllib.request.Request(f"https://127.0.0.1:{port}/health", method="GET")
        ssl_ctx = ssl._create_unverified_context()

        with urllib.request.urlopen(req, context=ssl_ctx, timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") == "ok"


def test_urllib_mtls_require_client_cert(certs: dict[str, Path]):
    """【双向 mTLS】服务端 require 客户端证书时：
    1. 缺少客户端证书请求失败；
    2. 提供受信客户端证书与私钥请求成功。
    """
    with _rest_tls_server(certs, client_auth="require") as port:
        req = urllib.request.Request(f"https://127.0.0.1:{port}/health", method="GET")

        # 1. 未配置客户端证书 -> 握手失败
        no_client_cert_ctx = ssl.create_default_context(cafile=str(certs["ca_cert"]))
        with pytest.raises((urllib.error.URLError, ssl.SSLError, ConnectionResetError)):
            urllib.request.urlopen(req, context=no_client_cert_ctx, timeout=3.0)

        # 2. 配置了合法的客户端证书与私钥 -> 握手成功
        mtls_ctx = ssl.create_default_context(cafile=str(certs["ca_cert"]))
        mtls_ctx.load_cert_chain(
            certfile=str(certs["client_cert"]),
            keyfile=str(certs["client_key"]),
        )
        with urllib.request.urlopen(req, context=mtls_ctx, timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") == "ok"


def test_urllib_tls_post_masking_interaction(certs: dict[str, Path]):
    """【TLS 数据交互】在 TLS 加密通道下使用 urllib.request POST JSON 数据调用脱敏接口。"""
    with _rest_tls_server(certs) as port:
        url = f"https://127.0.0.1:{port}/v1/privacy/mask"
        payload = json.dumps({"field_name": "mobile", "value": "13812345678"}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        ssl_ctx = ssl.create_default_context(cafile=str(certs["ca_cert"]))

        with urllib.request.urlopen(req, context=ssl_ctx, timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("result") == "138****5678"
