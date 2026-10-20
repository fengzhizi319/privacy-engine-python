"""安全模块单元测试：身份映射、认证辅助函数与限流器 / Security Unit Tests.

中文说明：
覆盖 security 子包中易于单元测试的纯逻辑：
- identity：权限映射（REST 路径 / gRPC 方法）、Identity.has_permission、健康探针判定
- auth：Bearer Token 提取、API Key 认证、mTLS 客户端证书身份提取
- ratelimit：Limiter 的端点限流项构造与放行判定、单例获取/重置

English Description:
Unit tests for security pure logic: identity permission mappings, auth helpers
(bearer token, API key, mTLS), and the rate-limit Limiter.
"""

from __future__ import annotations

import pytest

from engine.security import auth as auth_mod
from engine.security import ratelimit as rl
from engine.security import whitelist as wl_mod
from engine.security.config import KeyConfig, RateLimitConfig, SecuritySettings
from engine.security.identity import (
    ANONYMOUS_IDENTITY,
    DEV_MODE_IDENTITY,
    Identity,
    is_health_path_or_method,
    permission_for_grpc_method,
    permission_for_rest_path,
)


class TestIdentityPermission:
    def test_wildcard_grants_all(self):
        ident = Identity("internal", "svc", ["*"])
        assert ident.has_permission("privacy:mask")
        assert ident.has_permission("anything")

    def test_exact_scope(self):
        ident = Identity("external", "portal", ["privacy:mask"])
        assert ident.has_permission("privacy:mask")
        assert not ident.has_permission("privacy:dp")

    def test_anonymous_is_health_only(self):
        # Go AnonymousIdentity: 未认证匿名只保留健康探针权限。
        assert ANONYMOUS_IDENTITY.has_permission("health:read")
        assert not ANONYMOUS_IDENTITY.has_permission("privacy:dp")

    def test_dev_mode_identity_is_wildcard(self):
        # 认证整体关闭时 FastAPI 依赖仍需放行全部路由。
        assert DEV_MODE_IDENTITY.has_permission("privacy:dp")


class TestRestPathPermission:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/health", "health:read"),
            ("/livez", "health:read"),
            ("/readyz", "health:read"),
            ("/v1/privacy/mask", "privacy:mask"),
            ("/v1/privacy/mask_record", "privacy:mask"),
            ("/v1/privacy/mask/batch", "privacy:mask"),
            ("/v1/privacy/mask/dataframe", "privacy:mask"),
            ("/v1/privacy/hash", "privacy:hash"),
            ("/v1/privacy/dp/count", "privacy:dp"),
            ("/v1/privacy/k_anonymize/record", "privacy:kano"),
            ("/v1/privacy/qol/obfuscate", "privacy:qol"),
            ("/v1/privacy/qol/obfuscate/batch", "privacy:qol"),
            ("/v1/privacy/budget", "privacy:budget"),
            ("/v1/privacy/profile/recommend", "privacy:profile"),
            ("/v1/privacy/classify/field", "classification:read"),
            ("/v1/medical/process", "medical:process"),
            ("/v1/pipeline/process_records", "pipeline:process"),
            ("/v1/pipeline/process_csv", "pipeline:process"),
            ("/v1/ops/diagnostics", "ops:diagnostics"),
            ("/some/unknown/route", "*"),
        ],
    )
    def test_mapping(self, path, expected):
        assert permission_for_rest_path(path) == expected

    def test_trailing_slash_normalized(self):
        assert permission_for_rest_path("/health/") == "health:read"


class TestGrpcMethodPermission:
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("/privacy.local.PrivacyService/Mask", "privacy:mask"),
            ("/privacy.local.PrivacyService/DPCount", "privacy:dp"),
            ("/privacy.local.PrivacyService/ClassifyTable", "classification:read"),
            ("/privacy.local.PrivacyService/Health", "health:read"),
            ("/privacy.local.PrivacyService/RecommendParams", "privacy:profile"),
            ("/privacy.local.PrivacyService/Unknown", "*"),
            ("Mask", "privacy:mask"),
        ],
    )
    def test_mapping(self, method, expected):
        assert permission_for_grpc_method(method) == expected


class TestHealthDetection:
    def test_rest_health_paths(self):
        assert is_health_path_or_method("/health")
        assert is_health_path_or_method("/livez")
        assert is_health_path_or_method("/readyz")

    def test_grpc_health_method(self):
        assert is_health_path_or_method("/privacy.local.PrivacyService/Health")

    def test_non_health(self):
        assert not is_health_path_or_method("/v1/privacy/mask")


class TestBearerToken:
    def test_valid(self):
        assert auth_mod._extract_bearer_token("Bearer abc123") == "abc123"

    def test_case_insensitive_scheme(self):
        assert auth_mod._extract_bearer_token("bearer abc123") == "abc123"

    def test_none(self):
        assert auth_mod._extract_bearer_token(None) is None

    def test_malformed(self):
        assert auth_mod._extract_bearer_token("Token abc") is None
        assert auth_mod._extract_bearer_token("Bearer") is None


class TestAuthenticateApiKey:
    def _settings(self):
        return SecuritySettings(
            auth_enabled=True,
            internal_keys={"in-key": KeyConfig(name="internal-svc", scopes=["*"])},
            external_keys={"ex-key": KeyConfig(name="portal", scopes=["privacy:mask"])},
        )

    def test_internal_key(self):
        ident = auth_mod._authenticate_api_key(self._settings(), "in-key")
        assert ident is not None
        assert ident.service_type == "internal"
        assert ident.name == "internal-svc"

    def test_external_key(self):
        ident = auth_mod._authenticate_api_key(self._settings(), "ex-key")
        assert ident is not None
        assert ident.service_type == "external"
        assert ident.scopes == ["privacy:mask"]

    def test_unknown_key(self):
        assert auth_mod._authenticate_api_key(self._settings(), "nope") is None


class TestAuthenticateMtls:
    def test_valid_mtls(self, monkeypatch):
        # _authenticate_mtls now always uses the singleton WhitelistManager,
        # so we must configure it via env vars and reset the singleton.
        monkeypatch.setenv("PRIVACY_AUTH_MTLS_ALLOWED_CNS", "internal-client")
        monkeypatch.delenv("PRIVACY_AUTH_MTLS_WHITELIST_FILE", raising=False)
        wl_mod.reset_whitelist_manager()
        try:
            settings = SecuritySettings(
                auth_internal_mtls_enabled=True,
                auth_mtls_allowed_cns=["internal-client"],
            )
            ctx = {
                "transport_security_type": [b"ssl"],
                "x509_common_name": [b"internal-client"],
            }
            ident = auth_mod._authenticate_mtls(settings, ctx)
            assert ident is not None
            assert ident.service_type == "internal"
            assert ident.name == "internal-client"
            assert ident.scopes == ["*"]
        finally:
            wl_mod.reset_whitelist_manager()

    def test_default_disabled(self):
        """mTLS 认证默认关闭：仅凭 CA 校验通过的证书不能获得身份。"""
        settings = SecuritySettings()
        assert settings.auth_internal_mtls_enabled is False
        ctx = {"transport_security_type": [b"ssl"], "x509_common_name": [b"internal-client"]}
        assert auth_mod._authenticate_mtls(settings, ctx) is None

    def test_enabled_but_empty_whitelist_rejects(self):
        """启用 mTLS 但未配置 CN 白名单时拒绝所有证书（fail-closed）。"""
        settings = SecuritySettings(auth_internal_mtls_enabled=True)
        ctx = {"transport_security_type": [b"ssl"], "x509_common_name": [b"any-cn"]}
        assert auth_mod._authenticate_mtls(settings, ctx) is None

    def test_cn_not_in_whitelist_rejected(self):
        """CN 未命中白名单的证书被拒绝。"""
        settings = SecuritySettings(
            auth_internal_mtls_enabled=True,
            auth_mtls_allowed_cns=["allowed-svc"],
        )
        ctx = {"transport_security_type": [b"ssl"], "x509_common_name": [b"rogue-svc"]}
        assert auth_mod._authenticate_mtls(settings, ctx) is None

    def test_disabled(self):
        settings = SecuritySettings(
            auth_internal_mtls_enabled=False,
            auth_mtls_allowed_cns=["cn"],
        )
        ctx = {"transport_security_type": [b"ssl"], "x509_common_name": [b"cn"]}
        assert auth_mod._authenticate_mtls(settings, ctx) is None

    def test_non_ssl(self):
        settings = SecuritySettings(
            auth_internal_mtls_enabled=True,
            auth_mtls_allowed_cns=["cn"],
        )
        ctx = {"transport_security_type": [b"insecure"], "x509_common_name": [b"cn"]}
        assert auth_mod._authenticate_mtls(settings, ctx) is None

    def test_no_common_name(self):
        settings = SecuritySettings(
            auth_internal_mtls_enabled=True,
            auth_mtls_allowed_cns=["cn"],
        )
        ctx = {"transport_security_type": [b"ssl"]}
        assert auth_mod._authenticate_mtls(settings, ctx) is None


class TestLimiter:
    def _settings(self, **overrides):
        return SecuritySettings(rate_limit_enabled=True, **overrides)

    def test_is_allowed_within_limit(self):
        rl.reset_limiter()
        limiter = rl.Limiter(self._settings(rate_limit_default_rps=10, rate_limit_default_burst=20))
        ident = Identity("external", "u1", [])
        assert limiter.is_allowed(ident, "/v1/privacy/mask") is True

    def test_exceeds_limit(self):
        rl.reset_limiter()
        # burst=2 表示窗口内最多 2 次
        limiter = rl.Limiter(self._settings(rate_limit_default_rps=1, rate_limit_default_burst=2))
        ident = Identity("external", "u2", [])
        results = [limiter.is_allowed(ident, "/x") for _ in range(5)]
        assert results[:2] == [True, True]
        assert False in results[2:]

    def test_per_endpoint_override(self):
        rl.reset_limiter()
        settings = self._settings(
            rate_limit_default_rps=1,
            rate_limit_default_burst=1,
            rate_limit_per_endpoint={"/tight": RateLimitConfig(rps=1, burst=1)},
        )
        limiter = rl.Limiter(settings)
        item = limiter._limit_for_endpoint("/tight")
        assert item.amount == 1

    def test_get_limiter_singleton_and_reset(self):
        rl.reset_limiter()
        a = rl.get_limiter()
        b = rl.get_limiter()
        assert a is b
        rl.reset_limiter()
        c = rl.get_limiter()
        assert c is not a

    def test_fractional_burst_rounded_up_not_zero(self):
        """0 < burst < 1 时配额向上取整为 1，而非截断为 0 导致全量拒绝。"""
        rl.reset_limiter()
        limiter = rl.Limiter(self._settings(rate_limit_default_rps=1, rate_limit_default_burst=0.5))
        item = limiter._limit_for_endpoint("/v1/privacy/mask")
        assert item.amount == 1
        ident = Identity("external", "u3", [])
        assert limiter.is_allowed(ident, "/v1/privacy/mask") is True

    def test_storage_failure_degrades_to_allow(self):
        """存储后端（如 Redis）不可用时限流降级为放行，不应导致全站 500。"""
        rl.reset_limiter()
        limiter = rl.Limiter(self._settings())

        def _boom(*args, **kwargs):
            raise ConnectionError("redis is down")

        limiter._limiter.hit = _boom  # type: ignore[method-assign]
        ident = Identity("external", "u4", [])
        assert limiter.is_allowed(ident, "/v1/privacy/mask") is True
