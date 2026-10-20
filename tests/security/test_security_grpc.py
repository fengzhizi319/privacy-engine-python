"""安全模块 gRPC 拦截器与依赖项单元测试 / Security gRPC Interceptor Unit Tests.

中文说明：
覆盖 security 子包中依赖 gRPC 上下文的逻辑：
- auth：_extract_identity_from_grpc_context（mTLS / 健康豁免 / Bearer Token 三条路径）、
  AuthInterceptor 的 _check 与 intercept_service（四种流式组合 + 空 handler）、
  get_identity_from_grpc_context、require_rest_path_permission
- ratelimit：RateLimitInterceptor 的 _check 与 intercept_service、rate_limit_for_path 依赖项

使用伪造的 ServicerContext / RpcMethodHandler，无需真实网络连接。

English Description:
Unit tests for gRPC-context-dependent security logic using fake contexts/handlers.
"""

from __future__ import annotations

import asyncio

import grpc
import pytest

from engine.security import auth as auth_mod
from engine.security import ratelimit as rl
from engine.security import whitelist as wl_mod
from engine.security.auth import AuthInterceptor
from engine.security.config import KeyConfig, SecuritySettings, get_security_settings
from engine.security.identity import ANONYMOUS_IDENTITY
from engine.security.ratelimit import RateLimitInterceptor


class AbortError(Exception):
    """Raised by fake context.abort to mimic real gRPC abort propagation."""


class FakeContext:
    """Minimal grpc.ServicerContext double."""

    def __init__(self, auth_context=None, metadata=None):
        self._auth = auth_context if auth_context is not None else {}
        self._meta = metadata if metadata is not None else []
        self.abort_code = None
        self.abort_details = None

    def auth_context(self):
        return self._auth

    def invocation_metadata(self):
        return self._meta

    def abort(self, code, details):
        self.abort_code = code
        self.abort_details = details
        raise AbortError(f"{code}:{details}")


class FakeHandler:
    """Minimal grpc.RpcMethodHandler double with configurable streaming flags."""

    def __init__(self, request_streaming=False, response_streaming=False):
        self.request_streaming = request_streaming
        self.response_streaming = response_streaming
        self.request_deserializer = None
        self.response_serializer = None

    def unary_unary(self, request, context):
        return "uu"

    def unary_stream(self, request, context):
        return "us"

    def stream_unary(self, request, context):
        return "su"

    def stream_stream(self, request, context):
        return "ss"


class FakeCallDetails:
    def __init__(self, method):
        self.method = method
        self.invocation_metadata = []


MASK_METHOD = "/privacy.local.PrivacyService/Mask"
HEALTH_METHOD = "/privacy.local.PrivacyService/Health"


# ---------------------------------------------------------------------------
# _extract_identity_from_grpc_context
# ---------------------------------------------------------------------------

class TestExtractIdentityFromGrpcContext:
    def test_mtls_path(self, monkeypatch):
        # _authenticate_mtls now uses the singleton WhitelistManager,
        # so configure it via env vars and reset the singleton.
        monkeypatch.setenv("PRIVACY_AUTH_MTLS_ALLOWED_CNS", "internal-client")
        monkeypatch.delenv("PRIVACY_AUTH_MTLS_WHITELIST_FILE", raising=False)
        wl_mod.reset_whitelist_manager()
        try:
            settings = SecuritySettings(
                auth_enabled=True,
                auth_internal_mtls_enabled=True,
                auth_mtls_allowed_cns=["internal-client"],
            )
            ctx = FakeContext(
                auth_context={
                    "transport_security_type": [b"ssl"],
                    "x509_common_name": [b"internal-client"],
                }
            )
            ident = auth_mod._extract_identity_from_grpc_context(settings, ctx, MASK_METHOD)
            assert ident is not None
            assert ident.name == "internal-client"
            assert ident.scopes == ["*"]
        finally:
            wl_mod.reset_whitelist_manager()

    def test_mtls_cn_not_whitelisted_falls_through(self):
        """mTLS 证书 CN 未命中白名单时不得授予身份（回退到后续凭证检查）。"""
        settings = SecuritySettings(
            auth_enabled=True,
            auth_internal_mtls_enabled=True,
            auth_mtls_allowed_cns=["allowed-svc"],
        )
        ctx = FakeContext(
            auth_context={
                "transport_security_type": [b"ssl"],
                "x509_common_name": [b"rogue-svc"],
            },
            metadata=[],
        )
        ident = auth_mod._extract_identity_from_grpc_context(settings, ctx, MASK_METHOD)
        assert ident is None

    def test_health_exempt(self):
        settings = SecuritySettings(auth_enabled=True, health_no_auth=True)
        ctx = FakeContext(auth_context={}, metadata=[])
        ident = auth_mod._extract_identity_from_grpc_context(settings, ctx, HEALTH_METHOD)
        assert ident is not None
        assert ident.name == "health-probe"

    def test_bearer_token_path(self):
        settings = SecuritySettings(
            auth_enabled=True,
            internal_keys={"in-key": KeyConfig(name="svc", scopes=["*"])},
        )
        ctx = FakeContext(auth_context={}, metadata=[("authorization", "Bearer in-key")])
        ident = auth_mod._extract_identity_from_grpc_context(settings, ctx, MASK_METHOD)
        assert ident is not None
        assert ident.name == "svc"

    def test_no_credentials(self):
        settings = SecuritySettings(auth_enabled=True)
        ctx = FakeContext(auth_context={}, metadata=[])
        ident = auth_mod._extract_identity_from_grpc_context(settings, ctx, MASK_METHOD)
        assert ident is None


# ---------------------------------------------------------------------------
# AuthInterceptor
# ---------------------------------------------------------------------------

class TestAuthInterceptor:
    def test_auth_disabled_passthrough(self):
        interceptor = AuthInterceptor(SecuritySettings(auth_enabled=False))
        handler = FakeHandler()
        new_handler = interceptor.intercept_service(
            lambda cd: handler, FakeCallDetails(MASK_METHOD)
        )
        ctx = FakeContext()
        assert new_handler.unary_unary("req", ctx) == "uu"
        assert ctx.abort_code is None

    def test_missing_credentials_aborts(self):
        interceptor = AuthInterceptor(SecuritySettings(auth_enabled=True))
        handler = FakeHandler()
        new_handler = interceptor.intercept_service(
            lambda cd: handler, FakeCallDetails(MASK_METHOD)
        )
        ctx = FakeContext(auth_context={}, metadata=[])
        with pytest.raises(AbortError):
            new_handler.unary_unary("req", ctx)
        assert ctx.abort_code == grpc.StatusCode.UNAUTHENTICATED

    def test_valid_internal_key_allowed(self):
        settings = SecuritySettings(
            auth_enabled=True,
            internal_keys={"in-key": KeyConfig(name="svc", scopes=["*"])},
        )
        interceptor = AuthInterceptor(settings)
        handler = FakeHandler()
        new_handler = interceptor.intercept_service(
            lambda cd: handler, FakeCallDetails(MASK_METHOD)
        )
        ctx = FakeContext(auth_context={}, metadata=[("authorization", "Bearer in-key")])
        assert new_handler.unary_unary("req", ctx) == "uu"
        assert ctx.abort_code is None

    def test_insufficient_scope_aborts(self):
        settings = SecuritySettings(
            auth_enabled=True,
            external_keys={"ex-key": KeyConfig(name="portal", scopes=["privacy:dp"])},
        )
        interceptor = AuthInterceptor(settings)
        handler = FakeHandler()
        new_handler = interceptor.intercept_service(
            lambda cd: handler, FakeCallDetails(MASK_METHOD)
        )
        ctx = FakeContext(auth_context={}, metadata=[("authorization", "Bearer ex-key")])
        with pytest.raises(AbortError):
            new_handler.unary_unary("req", ctx)
        assert ctx.abort_code == grpc.StatusCode.PERMISSION_DENIED

    def test_none_handler_passthrough(self):
        interceptor = AuthInterceptor(SecuritySettings(auth_enabled=True))
        assert interceptor.intercept_service(lambda cd: None, FakeCallDetails(MASK_METHOD)) is None

    @pytest.mark.parametrize(
        "req_stream,resp_stream,attr,expected",
        [
            (False, False, "unary_unary", "uu"),
            (False, True, "unary_stream", "us"),
            (True, False, "stream_unary", "su"),
            (True, True, "stream_stream", "ss"),
        ],
    )
    def test_all_streaming_combinations(self, req_stream, resp_stream, attr, expected):
        interceptor = AuthInterceptor(SecuritySettings(auth_enabled=False))
        handler = FakeHandler(request_streaming=req_stream, response_streaming=resp_stream)
        new_handler = interceptor.intercept_service(
            lambda cd: handler, FakeCallDetails(MASK_METHOD)
        )
        fn = getattr(new_handler, attr)
        assert fn("req", FakeContext()) == expected


# ---------------------------------------------------------------------------
# get_identity_from_grpc_context
# ---------------------------------------------------------------------------

class TestGetIdentityFromGrpcContext:
    def test_auth_disabled_returns_anonymous(self, monkeypatch):
        monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "false")
        ident = auth_mod.get_identity_from_grpc_context(FakeContext(), MASK_METHOD)
        assert ident is ANONYMOUS_IDENTITY

    def test_auth_enabled_no_creds_aborts(self, monkeypatch):
        monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "true")
        ctx = FakeContext(auth_context={}, metadata=[])
        with pytest.raises(AbortError):
            auth_mod.get_identity_from_grpc_context(ctx, MASK_METHOD)
        assert ctx.abort_code == grpc.StatusCode.UNAUTHENTICATED

    def test_auth_enabled_valid_key(self, monkeypatch):
        monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "true")
        monkeypatch.setenv(
            "PRIVACY_AUTH_INTERNAL_KEYS_JSON", '{"in-key":{"name":"svc","scopes":["*"]}}'
        )
        ctx = FakeContext(auth_context={}, metadata=[("authorization", "Bearer in-key")])
        ident = auth_mod.get_identity_from_grpc_context(ctx, MASK_METHOD)
        assert ident.name == "svc"


# ---------------------------------------------------------------------------
# require_rest_path_permission
# ---------------------------------------------------------------------------

class TestRequireRestPathPermission:
    def test_returns_dependency(self):
        dep = auth_mod.require_rest_path_permission("/v1/privacy/mask")
        assert hasattr(dep, "dependency")


# ---------------------------------------------------------------------------
# RateLimitInterceptor
# ---------------------------------------------------------------------------

class TestRateLimitInterceptor:
    def test_disabled_no_abort(self):
        interceptor = RateLimitInterceptor(SecuritySettings(rate_limit_enabled=False))
        ctx = FakeContext()
        interceptor._check(ctx, MASK_METHOD)
        assert ctx.abort_code is None

    def test_health_exempt(self, monkeypatch):
        monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "false")
        interceptor = RateLimitInterceptor(
            SecuritySettings(rate_limit_enabled=True, health_no_rate_limit=True)
        )
        ctx = FakeContext()
        interceptor._check(ctx, HEALTH_METHOD)
        assert ctx.abort_code is None

    def test_exceeds_limit_aborts(self, monkeypatch):
        monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "false")
        monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("PRIVACY_RATE_LIMIT_DEFAULT_RPS", "1")
        monkeypatch.setenv("PRIVACY_RATE_LIMIT_DEFAULT_BURST", "1")
        rl.reset_limiter()
        try:
            interceptor = RateLimitInterceptor(get_security_settings())
            interceptor._check(FakeContext(), MASK_METHOD)  # 第一次放行
            ctx2 = FakeContext()
            with pytest.raises(AbortError):
                interceptor._check(ctx2, MASK_METHOD)  # 第二次超限
            assert ctx2.abort_code == grpc.StatusCode.RESOURCE_EXHAUSTED
        finally:
            rl.reset_limiter()

    def test_none_handler_passthrough(self):
        interceptor = RateLimitInterceptor(SecuritySettings(rate_limit_enabled=True))
        assert interceptor.intercept_service(lambda cd: None, FakeCallDetails(MASK_METHOD)) is None

    def test_intercept_service_wraps_handler(self, monkeypatch):
        monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "false")
        monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "false")
        interceptor = RateLimitInterceptor(get_security_settings())
        handler = FakeHandler()
        new_handler = interceptor.intercept_service(
            lambda cd: handler, FakeCallDetails(MASK_METHOD)
        )
        assert new_handler.unary_unary("req", FakeContext()) == "uu"


# ---------------------------------------------------------------------------
# rate_limit_for_path dependency
# ---------------------------------------------------------------------------

class FakeURL:
    def __init__(self, path):
        self.path = path


class FakeState:
    pass


class FakeRequest:
    def __init__(self, path, identity=None):
        self.url = FakeURL(path)
        self.state = FakeState()
        if identity is not None:
            self.state.identity = identity


class TestRateLimitForPath:
    def test_disabled_noop(self, monkeypatch):
        monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "false")
        dep = rl.rate_limit_for_path("/v1/privacy/mask")
        asyncio.run(dep.dependency(FakeRequest("/v1/privacy/mask")))

    def test_health_exempt(self, monkeypatch):
        monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("PRIVACY_HEALTH_NO_RATE_LIMIT", "true")
        dep = rl.rate_limit_for_path("/health")
        asyncio.run(dep.dependency(FakeRequest("/health")))

    def test_exceeds_raises_429(self, monkeypatch):
        from fastapi import HTTPException

        monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("PRIVACY_RATE_LIMIT_DEFAULT_RPS", "1")
        monkeypatch.setenv("PRIVACY_RATE_LIMIT_DEFAULT_BURST", "1")
        monkeypatch.setenv("PRIVACY_HEALTH_NO_RATE_LIMIT", "true")
        rl.reset_limiter()
        try:
            dep = rl.rate_limit_for_path("/v1/privacy/mask")
            asyncio.run(dep.dependency(FakeRequest("/v1/privacy/mask")))  # 放行
            with pytest.raises(HTTPException) as exc:
                asyncio.run(dep.dependency(FakeRequest("/v1/privacy/mask")))  # 超限
            assert exc.value.status_code == 429
        finally:
            rl.reset_limiter()
