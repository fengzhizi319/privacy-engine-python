"""Security configuration loaded from environment variables.

安全相关配置统一由此模块解析。所有开关默认关闭，保证本地开发与既有测试
不受影响；生产环境通过环境变量显式启用。

Security settings are centralized here. All toggles default to off so local dev and
existing tests keep working; production opts in via environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path  # noqa: TC003 - needed at runtime for Pydantic model annotations
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# 确保 .env 文件在任何模块级 settings 实例化之前已加载到 os.environ。
# 这保证了无论 config.py 的导入顺序如何（即使先于 server.py 的 load_env_file()），
# get_security_settings() 都能读到正确的环境变量而非全部默认值。
from ..env_loader import load_env_file as _load_env_file
_load_env_file()

logger = logging.getLogger(__name__)


class KeyConfig(BaseModel):
    """Static API key mapping entry.

    Attributes:
        name: Human-readable service/account name used in logs and rate-limit keys.
        scopes: List of permissions granted to this key. Use ["*"] for full access.
    """

    name: str
    scopes: list[str] = Field(default_factory=list)


class RateLimitConfig(BaseModel):
    """Per-endpoint rate limit override.

    Attributes:
        rps: Sustained requests per second.
        burst: Maximum burst allowed before throttling.
    """

    rps: float
    burst: float


class SecuritySettings(BaseModel):
    """Centralized security settings parsed from environment variables.

    The model uses Pydantic v2 BaseModel without introducing an extra dependency on
    pydantic-settings. Environment variables are read once at import time.
    """

    # ---------------------------- TLS ---------------------------------
    tls_enabled: bool = Field(default=False)
    tls_cert_file: Path | None = Field(default=None)
    tls_key_file: Path | None = Field(default=None)
    tls_ca_file: Path | None = Field(default=None)
    tls_client_auth: Literal["none", "optional", "require"] = Field(default="none")
    tls_key_password: str | None = Field(default=None)

    # ---------------------------- Auth --------------------------------
    auth_enabled: bool = Field(default=False)
    # mTLS 客户端证书认证默认关闭：任何通过 CA 校验的证书仅代表"持有合法证书"，
    # 不代表"被授权访问本服务"，必须显式启用并配置 CN 白名单才会被授予内部身份。
    auth_internal_mtls_enabled: bool = Field(default=False)
    # mTLS 客户端证书 CN 白名单；仅当 CN 命中白名单时才授予内部身份。
    # 白名单为空时拒绝所有 mTLS 证书（fail-closed）。
    # 注意：当 auth_mtls_whitelist_file 设置时，本字段被忽略。
    auth_mtls_allowed_cns: list[str] = Field(default_factory=list)
    # mTLS CN 白名单 YAML 配置文件路径。设置后启用 per-CN scope 控制与热重载。
    auth_mtls_whitelist_file: Path | None = Field(default=None)
    internal_keys: dict[str, KeyConfig] = Field(default_factory=dict)
    external_keys: dict[str, KeyConfig] = Field(default_factory=dict)

    # -------------------------- Rate Limit ----------------------------
    rate_limit_enabled: bool = Field(default=False)
    rate_limit_default_rps: float = Field(default=10.0)
    rate_limit_default_burst: float = Field(default=20.0)
    rate_limit_per_endpoint: dict[str, RateLimitConfig] = Field(default_factory=dict)
    rate_limit_redis_url: str | None = Field(default=None)

    # --------------------------- Health -------------------------------
    health_no_auth: bool = Field(default=True)
    health_no_rate_limit: bool = Field(default=True)

    # ------------------------ Metrics auth ----------------------------
    # Go 同名口径：专属轻量指标 Key 与轮转宽限 Key（PRIVACY_METRICS_API_KEY /
    # _PREVIOUS）。配置后匿名抓取被禁用，Key 仍走常量时间比对。
    metrics_api_key: str | None = Field(default=None)
    metrics_api_key_previous: str | None = Field(default=None)
    # mTLS 指标准入：Python 进程看不到客户端证书，必须由 Go 终结器把已验证的
    # 对端身份以签名头注入回环链路。未启用终结器身份校验时开启本项没有可用的
    # 证书来源，显式拒绝启动，避免「看起来启用了、实际全部放行」的静默降级。
    metrics_allow_mtls: bool = Field(default=False)
    metrics_allowed_client_cns: list[str] = Field(default_factory=list)

    # ------------------- Terminator identity headers ------------------
    # Go TLCP 终结器在回环链路注入的客户端证书身份。默认关闭；显式开启后，
    # REST 侧用于 PoP / 审计 / 服务账号激活，gRPC 侧可作为 mTLS 身份来源。
    terminator_identity_enabled: bool = Field(default=False)
    terminator_identity_hmac_algo: Literal["sm3", "sha256"] = Field(default="sm3")
    terminator_identity_hmac_key: str | None = Field(default=None)
    terminator_identity_hmac_key_file: Path | None = Field(default=None)
    terminator_identity_max_skew_seconds: int = Field(default=60)

    def effective_metrics_keys(self) -> list[str]:
        """Return the de-duplicated dedicated metrics keys (main first)."""
        keys: list[str] = []
        for raw in (self.metrics_api_key, self.metrics_api_key_previous):
            key = (raw or "").strip()
            if key and key not in keys:
                keys.append(key)
        return keys

    @model_validator(mode="after")
    def _check_tls_consistency(self) -> SecuritySettings:
        """Validate that TLS settings are mutually consistent."""
        if self.tls_enabled and (not self.tls_cert_file or not self.tls_key_file):
            raise ValueError(
                "PRIVACY_TLS_CERT_FILE and PRIVACY_TLS_KEY_FILE are required when TLS is enabled."
            )
        if self.tls_client_auth in ("optional", "require") and not self.tls_ca_file:
            raise ValueError(
                "PRIVACY_TLS_CA_FILE is required when tls_client_auth is optional or require."
            )
        if self.terminator_identity_enabled:
            # 显式开启即校验密钥材料：配置错误必须在启动期暴露，而不是首个
            # 请求到来时才表现为一脸茫然的 401。
            from .terminator import TerminatorIdentityError, build_verifier

            try:
                build_verifier(self)
            except TerminatorIdentityError as exc:
                raise ValueError(str(exc)) from exc
        if self.metrics_allow_mtls and not self.terminator_identity_enabled:
            raise ValueError(
                "PRIVACY_METRICS_ALLOW_MTLS requires verified terminator identity: the Python "
                "ASGI runtime does not expose per-connection client certificates. Enable "
                "PRIVACY_TERMINATOR_IDENTITY_ENABLED (Go TLCP terminator topology), or use "
                "PRIVACY_METRICS_API_KEY / the ops:diagnostics scope instead."
            )
        return self


# ---------------------------------------------------------------------------
# 环境变量类型化读取器 / Typed Environment Variable Loaders
# ---------------------------------------------------------------------------
#
# 本模块使用两类读取方式，根据目标类型选择：
# This module uses two categories of loaders, chosen by target type:
#
# 1. os.environ.get() — 原始字符串读取，适用于无需类型转换的字段（路径、名称等）。
#    Raw string reader, for fields that need no type conversion (paths, names, etc.).
#
# 2. _load_*_env() 系列 — 带类型转换 + 容错降级的安全读取器：
#    Type-safe loaders with conversion + graceful fallback:
#    - _load_bool_env()   : 布尔解析，支持 true/1/yes/on，大小写不敏感
#                            Bool parsing: true/1/yes/on, case-insensitive
#    - _load_float_env()  : 浮点解析，非法值回退默认值 + 警告日志
#                            Float parsing: bad values fall back to default + warning
#    - _load_json_env()   : JSON 对象解析，非法 JSON 回退默认值（不抛异常）
#                            JSON object parsing: invalid JSON falls back (no exception)
#    - _load_str_list_env(): 字符串列表解析，兼容 JSON 数组和逗号分隔
#                            String list parsing: supports JSON array and comma-separated
#
# 设计原则：本模块在 import 期执行，任何解析错误必须静默降级，绝不能导致包崩溃。
# Design principle: This module runs at import time; any parse error must degrade
# silently and never crash the entire package.
# ---------------------------------------------------------------------------


def _load_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable (true/1/yes/on, case-insensitive).

    布尔类型环境变量解析器。支持 true/1/yes/on（大小写不敏感）。
    空值或未设置时返回 default。
    """
    value = os.environ.get(name, "")
    # 空值回退默认值；否则检查是否在真值集合中
    # Fall back to default if empty; otherwise check against truthy set
    return value.lower() in {"true", "1", "yes", "on"} if value else default


def _load_json_env(name: str, default: dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON object from an environment variable.

    JSON 对象类型环境变量解析器。将环境变量值解析为 Python dict。
    解析失败时回退默认值并告警，绝不抛出——本模块在 import 期执行，
    非法环境变量不应导致整个包崩溃。
    """
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Environment variable %s contains invalid JSON (%s); falling back to default.",
            name,
            exc,
        )
        return default
    if not isinstance(parsed, dict):
        logger.warning(
            "Environment variable %s must be a JSON object; falling back to default.", name
        )
        return default
    return parsed


def _load_float_env(name: str, default: float) -> float:
    """Parse a float environment variable, falling back to default on bad input.

    浮点数类型环境变量解析器。将环境变量值转换为 float。
    非法值（如 "abc"）回退到 default 并记录警告日志。
    """
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "Environment variable %s=%r is not a valid float; falling back to %s.",
            name,
            value,
            default,
        )
        return default


def _load_int_env(name: str, default: int) -> int:
    """Parse an integer environment variable, falling back to default on bad input."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Environment variable %s=%r is not a valid integer; falling back to %s.",
            name,
            value,
            default,
        )
        return default


def _load_str_list_env(name: str) -> list[str]:
    """Parse a string-list environment variable (JSON array or comma-separated).

    字符串列表类型环境变量解析器。兼容两种格式：
    - JSON 数组: '["a","b"]'
    - 逗号分隔: 'a,b'
    解析失败时回退空列表（fail-closed）并告警。
    """
    value = os.environ.get(name, "").strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Environment variable %s contains invalid JSON (%s); falling back to empty list.",
                name,
                exc,
            )
            return []
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            logger.warning(
                "Environment variable %s must be a JSON array of strings; falling back to empty list.",
                name,
            )
            return []
        return [item.strip() for item in parsed if item.strip()]
    return [part.strip() for part in value.split(",") if part.strip()]


def get_security_settings() -> SecuritySettings:
    """Parse and return SecuritySettings from the current environment.

    每次调用都重新解析环境变量，支持测试和运行时动态修改配置。
    This function re-parses environment variables on every call so that tests and
    runtime reconfiguration can change behaviour by updating ``os.environ``.

    读取策略 / Reading strategy:
    - 需要类型转换的字段（bool/float/JSON/list）→ 使用 _load_*_env() 安全读取器
      Fields requiring type conversion → use _load_*_env() safe loaders
    - 纯字符串字段（路径、名称）→ 直接使用 os.environ.get()，空值转 None
      Pure string fields (paths, names) → use os.environ.get() directly, empty → None
    """
    return SecuritySettings(
        # ── TLS 配置 / TLS Configuration ──
        # 布尔类型 → _load_bool_env() 解析 true/false
        # Bool type → _load_bool_env() parses true/false
        tls_enabled=_load_bool_env("PRIVACY_TLS_ENABLED"),
        # 字符串路径 → os.environ.get() 直接读取，空值转 None
        # String path → os.environ.get() raw read, empty → None
        tls_cert_file=os.environ.get("PRIVACY_TLS_CERT_FILE") or None,
        tls_key_file=os.environ.get("PRIVACY_TLS_KEY_FILE") or None,
        tls_ca_file=os.environ.get("PRIVACY_TLS_CA_FILE") or None,
        # 枚举字符串 → os.environ.get() + 默认值 "none"
        # Enum string → os.environ.get() with default "none"
        tls_client_auth=(
            os.environ.get("PRIVACY_TLS_CLIENT_AUTH", "none") or "none"  # type: ignore[arg-type]
        ),
        tls_key_password=os.environ.get("PRIVACY_TLS_KEY_PASSWORD") or None,
        # ── API Key 认证 / API Key Authentication ──
        auth_enabled=_load_bool_env("PRIVACY_AUTH_ENABLED"),
        # mTLS 认证默认关闭（fail-closed）：仅当显式启用且 CN 命中白名单时才授权。
        # mTLS auth defaults off (fail-closed): only authorize when explicitly enabled
        # and CN matches whitelist.
        auth_internal_mtls_enabled=_load_bool_env(
            "PRIVACY_AUTH_INTERNAL_MTLS_ENABLED", default=False
        ),
        # CN 白名单：JSON 数组或逗号分隔 → _load_str_list_env() 兼容两种格式
        # CN whitelist: JSON array or comma-separated → _load_str_list_env() handles both
        # 注意：当 PRIVACY_AUTH_MTLS_WHITELIST_FILE 设置时，本字段被忽略。
        # Note: ignored when PRIVACY_AUTH_MTLS_WHITELIST_FILE is set.
        auth_mtls_allowed_cns=_load_str_list_env("PRIVACY_AUTH_MTLS_ALLOWED_CNS"),
        # 字符串路径 → os.environ.get() 直接读取
        # String path → os.environ.get() raw read
        auth_mtls_whitelist_file=os.environ.get("PRIVACY_AUTH_MTLS_WHITELIST_FILE") or None,
        # JSON 对象 → _load_json_env() 安全解析，非法 JSON 回退空 dict
        # JSON object → _load_json_env() safe parse, invalid JSON falls back to {}
        internal_keys=_load_json_env("PRIVACY_AUTH_INTERNAL_KEYS_JSON", {}),
        external_keys=_load_json_env("PRIVACY_AUTH_EXTERNAL_KEYS_JSON", {}),
        # ── 速率限制 / Rate Limiting ──
        rate_limit_enabled=_load_bool_env("PRIVACY_RATE_LIMIT_ENABLED"),
        # 浮点数 → _load_float_env() 安全转换，非法值回退默认
        # Float → _load_float_env() safe conversion, bad values fall back to default
        rate_limit_default_rps=_load_float_env("PRIVACY_RATE_LIMIT_DEFAULT_RPS", 10.0),
        rate_limit_default_burst=_load_float_env("PRIVACY_RATE_LIMIT_DEFAULT_BURST", 20.0),
        rate_limit_per_endpoint=_load_json_env(
            "PRIVACY_RATE_LIMIT_PER_ENDPOINT_JSON", {}
        ),
        # 字符串 URL → os.environ.get() 直接读取
        # String URL → os.environ.get() raw read
        rate_limit_redis_url=os.environ.get("PRIVACY_RATE_LIMIT_REDIS_URL") or None,
        # ── 健康检查 / Health Check ──
        # 布尔类型，默认 True（健康检查端点免认证/免限速，K8s 探针依赖）
        # Bool type, default True (health endpoints are auth-free/rate-limit-free for K8s)
        health_no_auth=_load_bool_env("PRIVACY_HEALTH_NO_AUTH", default=True),
        health_no_rate_limit=_load_bool_env(
            "PRIVACY_HEALTH_NO_RATE_LIMIT", default=True
        ),
        # ── 指标专属准入 / Dedicated metrics admission ──
        metrics_api_key=os.environ.get("PRIVACY_METRICS_API_KEY") or None,
        metrics_api_key_previous=os.environ.get("PRIVACY_METRICS_API_KEY_PREVIOUS") or None,
        metrics_allow_mtls=_load_bool_env("PRIVACY_METRICS_ALLOW_MTLS"),
        metrics_allowed_client_cns=_load_str_list_env(
            "PRIVACY_METRICS_ALLOWED_CLIENT_CNS"
        ),
        # ── 终结器身份校验 / Terminator identity ──
        terminator_identity_enabled=_load_bool_env(
            "PRIVACY_TERMINATOR_IDENTITY_ENABLED", default=False
        ),
        terminator_identity_hmac_algo=(
            os.environ.get("PRIVACY_TERMINATOR_IDENTITY_HMAC_ALGO", "sm3").strip().lower()
            or "sm3"
        ),  # type: ignore[arg-type]
        terminator_identity_hmac_key=os.environ.get(
            "PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY"
        )
        or None,
        terminator_identity_hmac_key_file=os.environ.get(
            "PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE"
        )
        or None,
        terminator_identity_max_skew_seconds=_load_int_env(
            "PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS", 60
        ),
    )


def metrics_auth_mode(settings: SecuritySettings) -> tuple[str, list[str]]:
    """Return the effective ``/metrics`` posture (Go ``MetricsAuthConfig`` parity).

    Modes: ``dev_anonymous`` / ``bearer_key`` / ``mtls`` / ``scope``.
    ``mtls`` admission requires verified terminator identity headers; the ASGI
    runtime cannot see client certificates directly.
    """
    # 延迟导入：engine.auth 运行时依赖较重，且经包 __init__ 间接导入本模块。
    from ..auth.config import is_production_labeled

    keys = settings.effective_metrics_keys()
    if (
        not is_production_labeled()
        and not settings.auth_enabled
        and not keys
        and not settings.metrics_allow_mtls
    ):
        return "dev_anonymous", ["dev_anonymous"]
    if keys:
        strategies = ["bearer_key"]
        if settings.metrics_allow_mtls:
            strategies.append("mtls")
        if settings.auth_enabled:
            strategies.append("scope")
        return "bearer_key", strategies
    if settings.metrics_allow_mtls:
        strategies = ["mtls"]
        if settings.auth_enabled:
            strategies.append("scope")
        return "mtls", strategies
    return "scope", (["scope"] if settings.auth_enabled else [])


# Module-level convenience alias used by the rest of the package.
settings = get_security_settings()
