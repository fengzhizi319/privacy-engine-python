"""安全 TLS/配置校验与 tracing noop 路径补充测试 / TLS, Config & Tracing Unit Tests.

中文说明：
补齐覆盖率门禁所需的少量分支：
- security/tls.py：uvicorn_ssl_kwargs 关闭/完整分支、grpc_server_credentials 关闭时报错
- security/config.py：TLS 一致性校验失败、_load_json_env 非法 JSON / 非对象
- observability/tracing.py：opentelemetry 缺失时的 noop tracer 与 start_span 路径

English Description:
Covers remaining branches in tls.py, config.py validators, and the no-op tracing
path used when opentelemetry is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.observability import tracing
from engine.security import tls as tls_mod
from engine.security import config as config_mod
from engine.security.config import SecuritySettings, _load_json_env


class TestUvicornSslKwargs:
    def test_disabled_returns_empty(self):
        assert tls_mod.uvicorn_ssl_kwargs(SecuritySettings()) == {}

    def test_enabled_full(self):
        settings = SecuritySettings(
            tls_enabled=True,
            tls_cert_file=Path("server.crt"),
            tls_key_file=Path("server.key"),
            tls_ca_file=Path("ca.crt"),
            tls_key_password="secret",
            tls_client_auth="require",
        )
        kwargs = tls_mod.uvicorn_ssl_kwargs(settings)
        assert kwargs["ssl_certfile"] == "server.crt"
        assert kwargs["ssl_keyfile"] == "server.key"
        assert kwargs["ssl_ca_certs"] == "ca.crt"
        assert kwargs["ssl_keyfile_password"] == "secret"


class TestGrpcServerCredentials:
    def test_disabled_raises(self):
        with pytest.raises(RuntimeError, match="TLS disabled"):
            tls_mod.grpc_server_credentials(SecuritySettings())


class TestConfigValidators:
    def test_tls_enabled_requires_cert_key(self):
        with pytest.raises(ValueError, match="CERT_FILE"):
            SecuritySettings(tls_enabled=True)

    def test_client_auth_requires_ca(self):
        with pytest.raises(ValueError, match="CA_FILE"):
            SecuritySettings(tls_client_auth="require")


class TestLoadJsonEnv:
    def test_invalid_json_falls_back_to_default(self, monkeypatch):
        """非法 JSON 不再抛出（import 期容错），回退默认值。"""
        monkeypatch.setenv("X_JSON", "{not-valid")
        assert _load_json_env("X_JSON", {"a": 1}) == {"a": 1}

    def test_non_object_falls_back_to_default(self, monkeypatch):
        """JSON 非对象时回退默认值。"""
        monkeypatch.setenv("X_JSON", "[1, 2]")
        assert _load_json_env("X_JSON", {"b": 2}) == {"b": 2}

    def test_valid_object_parsed(self, monkeypatch):
        monkeypatch.setenv("X_JSON", '{"k": "v"}')
        assert _load_json_env("X_JSON", {}) == {"k": "v"}


class TestLoadFloatEnv:
    def test_invalid_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("X_FLOAT", "not-a-number")
        assert config_mod._load_float_env("X_FLOAT", 10.0) == 10.0

    def test_valid_float_parsed(self, monkeypatch):
        monkeypatch.setenv("X_FLOAT", "2.5")
        assert config_mod._load_float_env("X_FLOAT", 10.0) == 2.5


class TestLoadStrListEnv:
    def test_empty_defaults_to_empty_list(self, monkeypatch):
        monkeypatch.delenv("X_LIST", raising=False)
        assert config_mod._load_str_list_env("X_LIST") == []

    def test_json_array(self, monkeypatch):
        monkeypatch.setenv("X_LIST", '["a", "b"]')
        assert config_mod._load_str_list_env("X_LIST") == ["a", "b"]

    def test_comma_separated(self, monkeypatch):
        monkeypatch.setenv("X_LIST", "a, b ,,c")
        assert config_mod._load_str_list_env("X_LIST") == ["a", "b", "c"]

    def test_invalid_json_array_falls_back(self, monkeypatch):
        monkeypatch.setenv("X_LIST", "[not-json")
        assert config_mod._load_str_list_env("X_LIST") == []

    def test_non_string_array_falls_back(self, monkeypatch):
        monkeypatch.setenv("X_LIST", "[1, 2]")
        assert config_mod._load_str_list_env("X_LIST") == []


class TestGetSecuritySettingsResilience:
    def test_invalid_keys_json_env_does_not_crash(self, monkeypatch):
        """import 期执行的 get_security_settings() 遇到非法 JSON 必须容错。"""
        monkeypatch.setenv("PRIVACY_AUTH_INTERNAL_KEYS_JSON", "{broken")
        monkeypatch.setenv("PRIVACY_RATE_LIMIT_DEFAULT_RPS", "abc")
        settings = config_mod.get_security_settings()
        assert settings.internal_keys == {}
        assert settings.rate_limit_default_rps == 10.0

    def test_mtls_defaults_fail_closed(self, monkeypatch):
        """mTLS 认证默认关闭且白名单为空。"""
        monkeypatch.delenv("PRIVACY_AUTH_INTERNAL_MTLS_ENABLED", raising=False)
        monkeypatch.delenv("PRIVACY_AUTH_MTLS_ALLOWED_CNS", raising=False)
        settings = config_mod.get_security_settings()
        assert settings.auth_internal_mtls_enabled is False
        assert settings.auth_mtls_allowed_cns == []

    def test_mtls_allowed_cns_comma_separated(self, monkeypatch):
        monkeypatch.setenv("PRIVACY_AUTH_MTLS_ALLOWED_CNS", "svc-a,svc-b")
        settings = config_mod.get_security_settings()
        assert settings.auth_mtls_allowed_cns == ["svc-a", "svc-b"]


class TestTracingNoOp:
    def test_get_tracer_initializes_noop(self):
        old = tracing._tracer
        tracing._tracer = None
        try:
            tracer = tracing.get_tracer()
            assert tracer is not None
        finally:
            tracing._tracer = old

    def test_start_span_noop(self, monkeypatch):
        monkeypatch.setattr(tracing, "_HAS_OTEL", False)
        old = tracing._tracer
        tracing._tracer = None
        try:
            with tracing.start_span("op", attributes={"k": "v"}) as span:
                assert span is None
        finally:
            tracing._tracer = old

    def test_noop_tracer_start_span_returns_none(self, monkeypatch):
        monkeypatch.setattr(tracing, "_HAS_OTEL", False)
        tracer = tracing._noop_tracer()
        assert tracer.start_span("x") is None

    def test_init_tracing_noop_without_otel(self):
        tracer = tracing.init_tracing()
        assert tracer is not None
