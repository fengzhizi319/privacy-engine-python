"""网关→后端 TLS 回源配置单元测试。

覆盖 balancer 模块的 backend_tls_enabled / backend_tls_verify /
backend_channel_credentials 辅助函数与 BackendNode.grpc_stub 的
secure/insecure channel 选择逻辑。
"""

from __future__ import annotations

import grpc
import pytest

from engine.gateway import balancer


@pytest.fixture()
def _clean_env(monkeypatch):
    for var in (
        "PRIVACY_GATEWAY_BACKEND_TLS_ENABLED",
        "PRIVACY_GATEWAY_BACKEND_TLS_CA",
        "PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_CERT",
        "PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


class TestBackendTlsVerify:
    def test_disabled_returns_default_verify(self, _clean_env):
        assert balancer.backend_tls_enabled() is False
        assert balancer.backend_tls_verify() is True
        assert balancer.backend_channel_credentials() is None

    def test_enabled_without_ca_raises(self, _clean_env, monkeypatch):
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_ENABLED", "true")
        with pytest.raises(RuntimeError, match="PRIVACY_GATEWAY_BACKEND_TLS_CA"):
            balancer.backend_tls_verify()

    def test_enabled_with_missing_ca_file_raises(self, _clean_env, monkeypatch, tmp_path):
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_ENABLED", "1")
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_CA", str(tmp_path / "nope.pem"))
        with pytest.raises(RuntimeError, match="不存在"):
            balancer.backend_tls_verify()

    def test_enabled_with_valid_ca(self, _clean_env, monkeypatch, tmp_path):
        ca = tmp_path / "ca.pem"
        ca.write_bytes(b"fake-ca-bytes")
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_ENABLED", "yes")
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_CA", str(ca))
        assert balancer.backend_tls_verify() == str(ca)
        creds = balancer.backend_channel_credentials()
        assert isinstance(creds, grpc.ChannelCredentials)

    def test_mtls_requires_cert_and_key_pair(self, _clean_env, monkeypatch, tmp_path):
        ca = tmp_path / "ca.pem"
        ca.write_bytes(b"fake-ca-bytes")
        cert = tmp_path / "client.pem"
        cert.write_bytes(b"fake-cert")
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_ENABLED", "true")
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_CA", str(ca))
        # 只给证书不给私钥 → 必须报错
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_CERT", str(cert))
        with pytest.raises(RuntimeError, match="CLIENT_KEY"):
            balancer.backend_channel_credentials()


class TestGrpcStubChannelSelection:
    def test_insecure_by_default(self, _clean_env, monkeypatch):
        from unittest.mock import MagicMock

        calls: dict = {}

        def _fake_insecure(addr, options=None):
            calls["insecure"] = (addr, options)
            return MagicMock()

        monkeypatch.setattr(balancer.grpc.aio, "insecure_channel", _fake_insecure)
        node = balancer.BackendNode(http_url="http://127.0.0.1:8079", grpc_address="127.0.0.1:50051")
        _ = node.grpc_stub
        assert "insecure" in calls

    def test_secure_channel_when_tls_enabled(self, _clean_env, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        ca = tmp_path / "ca.pem"
        ca.write_bytes(b"fake-ca-bytes")
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_ENABLED", "true")
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_CA", str(ca))

        calls: dict = {}

        def _fake_secure(addr, credentials, options=None):
            calls["secure"] = (addr, credentials, options)
            return MagicMock()

        monkeypatch.setattr(balancer.grpc.aio, "secure_channel", _fake_secure)
        node = balancer.BackendNode(http_url="https://backend:8079", grpc_address="backend:50051")
        _ = node.grpc_stub
        assert "secure" in calls
        assert isinstance(calls["secure"][1], grpc.ChannelCredentials)

    def test_missing_ca_fails_fast_on_stub_access(self, _clean_env, monkeypatch):
        monkeypatch.setenv("PRIVACY_GATEWAY_BACKEND_TLS_ENABLED", "true")
        node = balancer.BackendNode(http_url="https://backend:8079", grpc_address="backend:50051")
        with pytest.raises(RuntimeError, match="PRIVACY_GATEWAY_BACKEND_TLS_CA"):
            _ = node.grpc_stub
