"""独立运维口（业务口 + 1000）：仅放行探针、指标与诊断路径。

Go 对照：``pkg/opsport``。运维口与业务口共享同一份 FastAPI 应用与安全栈，
差别只有监听地址与路径白名单：

- ``/health`` ``/healthz`` ``/livez`` ``/readyz``：探针，业务口本就免 Key；
- ``/metrics`` ``/ops/diagnostics`` ``/debug/pprof/``：运维数据，鉴权规则与业务口
  完全一致（``/metrics`` 走 ``ApiKeyAuthAsgiMiddleware``，诊断路由要
  ``ops:diagnostics`` scope），不存在「运维口上少一道门」；
- 白名单之外一律 404 JSON，且请求不进入主路由（运维口不是业务数据的第二入口）。

协议栈口径（``PRIVACY_OPS_TLS_STACK``）：

- 未设置 / 空：完整继承业务口 ``PRIVACY_TLS_*``（开发默认明文，业务口启用 TLS
  时同口径启用）；
- ``tls``：运维口独立 mTLS，必须同时提供 cert/key/CA，缺失拒绝启动；
- ``tlcp`` / ``both``：Python 进程内不能终结国密，仅允许绑定环回地址并由 Phase 8
  Go 终结器对外提供 TLCP；监听本身为回环明文（终结器之后的受信段）。
"""

from __future__ import annotations

import ipaddress
import os
import ssl
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .observability.logging_config import get_logger
from .security.tls import uvicorn_ssl_kwargs

if TYPE_CHECKING:
    from .security.config import SecuritySettings

logger = get_logger(__name__)

# Go ``opsport.PortOffset``：看到端口就知道是不是运维口。
OPS_PORT_OFFSET = 1000

# Go ``opsport.DefaultAllowedPaths``：精确匹配；以 "/" 结尾的条目按前缀匹配。
# ``/metrics/`` 是 Python 侧 FastAPI ``app.mount("/metrics", ...)`` 的规范形态：
# Starlette 会把 ``/metrics`` 307 重定向到 ``/metrics/``，抓取器若被白名单挡住
# 重定向目标，运维口上的指标就会永远抓不到。
OPS_ALLOWED_PATHS: tuple[str, ...] = (
    "/health",
    "/healthz",
    "/livez",
    "/readyz",
    "/metrics",
    "/metrics/",
    "/ops/diagnostics",
    "/debug/pprof/",
)

_OPS_NOT_FOUND_BODY = (
    b'{"code":"NOT_FOUND","message":"ops port only serves probe, metrics and '
    b'diagnostics paths"}\n'
)

_TLS_STACKS = {"", "tlcp", "both", "tls"}


def default_ops_port(business_port: int) -> int:
    """由业务口推导运维口端口；业务口为随机端口（0）时运维口同样随机。"""
    return 0 if business_port == 0 else business_port + OPS_PORT_OFFSET


def is_loopback_host(host: str) -> bool:
    """Return True for loopback literals (127.0.0.0/8, ::1, localhost)."""
    candidate = (host or "").strip().strip("[]").lower()
    if candidate in {"localhost", "ip6-localhost", "ip6-loopback"}:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class OpsSettings:
    """Resolved ops listener settings (port, bind address and TLS material)."""

    host: str
    port: int
    tls_stack: str
    tls_cert_file: str | None
    tls_key_file: str | None
    tls_ca_file: str | None
    tls_key_password: str | None

    @property
    def terminator_fronted(self) -> bool:
        """True when a Go TLCP terminator is expected to front this listener."""
        return self.tls_stack in {"tlcp", "both"}


def _env_str(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _require_readable(label: str, path: str) -> None:
    if not os.path.isfile(path):
        raise ValueError(f"{label} is not a readable file: {path}")


def load_ops_settings(
    rest_host: str,
    rest_port: int,
    security: SecuritySettings,
    *,
    ops_host: str | None = None,
    ops_port: int | None = None,
) -> OpsSettings:
    """Load and validate the ops listener from ``PRIVACY_OPS_*``.

    CLI overrides take precedence over environment values, matching the rest of
    the server entry points.

    Raises:
        ValueError: invalid stack, missing TLS material, or a TLCP/both bind that
            is not loopback (Python cannot terminate national-crypto TLS).
    """
    host = ops_host or _env_str("PRIVACY_OPS_HOST") or rest_host

    raw_port = _env_str("PRIVACY_OPS_PORT")
    try:
        if ops_port is not None:
            port = int(ops_port)
        elif raw_port is not None:
            port = int(raw_port)
        else:
            port = default_ops_port(rest_port)
    except ValueError as exc:
        raise ValueError(f"PRIVACY_OPS_PORT must be an integer, got {raw_port!r}") from exc
    if rest_port == 0:
        port = 0
    if not 0 <= port <= 65535:
        raise ValueError(f"PRIVACY_OPS_PORT out of range: {port}")

    stack = (_env_str("PRIVACY_OPS_TLS_STACK") or "").lower()
    if stack not in _TLS_STACKS:
        raise ValueError(
            f"PRIVACY_OPS_TLS_STACK must be one of tlcp|both|tls or empty (inherit), got {stack!r}"
        )

    # 显式 OPS_* 优先；未配置时回退业务口材料，沿用 Go 的 ops→business 回退顺序。
    cert_file = _env_str("PRIVACY_OPS_TLS_CERT_FILE") or (
        str(security.tls_cert_file) if security.tls_cert_file else None
    )
    key_file = _env_str("PRIVACY_OPS_TLS_KEY_FILE") or (
        str(security.tls_key_file) if security.tls_key_file else None
    )
    ca_file = _env_str("PRIVACY_OPS_TLS_CA_FILE") or (
        str(security.tls_ca_file) if security.tls_ca_file else None
    )
    key_password = _env_str("PRIVACY_OPS_TLS_KEY_PASSWORD") or security.tls_key_password

    if stack in {"tlcp", "both"} and not is_loopback_host(host):
        raise ValueError(
            f"PRIVACY_OPS_TLS_STACK={stack} cannot be terminated by the Python engine "
            "(no native TLCP); bind PRIVACY_OPS_HOST to a loopback address behind the "
            "Go TLCP terminator, or set PRIVACY_OPS_TLS_STACK=tls"
        )
    if stack == "tls":
        if not (cert_file and key_file and ca_file):
            raise ValueError(
                "PRIVACY_OPS_TLS_STACK=tls requires PRIVACY_OPS_TLS_CERT_FILE, "
                "PRIVACY_OPS_TLS_KEY_FILE and PRIVACY_OPS_TLS_CA_FILE (mTLS)"
            )
        _require_readable("PRIVACY_OPS_TLS_CERT_FILE", cert_file)
        _require_readable("PRIVACY_OPS_TLS_KEY_FILE", key_file)
        _require_readable("PRIVACY_OPS_TLS_CA_FILE", ca_file)

    return OpsSettings(
        host=host,
        port=port,
        tls_stack=stack,
        tls_cert_file=cert_file,
        tls_key_file=key_file,
        tls_ca_file=ca_file,
        tls_key_password=key_password,
    )


def ops_uvicorn_ssl_kwargs(
    ops: OpsSettings, security: SecuritySettings
) -> dict[str, Any]:
    """Build uvicorn TLS kwargs for the ops listener.

    Empty stack inherits the business listener verbatim; ``tlcp``/``both`` stay
    cleartext because the listener is loopback-only behind the Go terminator.
    """
    if ops.tls_stack == "":
        return uvicorn_ssl_kwargs(security)
    if ops.terminator_fronted:
        return {}
    kwargs: dict[str, Any] = {
        "ssl_keyfile": ops.tls_key_file,
        "ssl_certfile": ops.tls_cert_file,
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
        "ssl_version": ssl.PROTOCOL_TLS_SERVER,
        "ssl_ca_certs": ops.tls_ca_file,
    }
    if ops.tls_key_password is not None:
        kwargs["ssl_keyfile_password"] = ops.tls_key_password
    return kwargs


class OpsAllowlistMiddleware:
    """Pure-ASGI allowlist wrapping the business app for the ops listener.

    Path matching mirrors Go ``opsport.Allowlist``: exact entries, plus prefix
    matching for entries ending in ``/`` (the trimmed form is also exact). A
    denied request never reaches the wrapped application.
    """

    def __init__(
        self, app: Any, allowed_paths: tuple[str, ...] | list[str] | None = None
    ) -> None:
        self.app = app
        paths = OPS_ALLOWED_PATHS if allowed_paths is None else tuple(allowed_paths)
        self._exact: set[str] = set()
        self._prefixes: list[str] = []
        for entry in paths:
            if entry.endswith("/"):
                self._prefixes.append(entry)
                self._exact.add(entry.rstrip("/"))
            else:
                self._exact.add(entry)

    def _is_allowed(self, path: str) -> bool:
        if path in self._exact:
            return True
        return any(path.startswith(prefix) for prefix in self._prefixes)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if self._is_allowed(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(_OPS_NOT_FOUND_BODY)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _OPS_NOT_FOUND_BODY})


def create_ops_asgi_app(app: Any) -> OpsAllowlistMiddleware:
    """Wrap the business ASGI app with the ops path allowlist."""
    return OpsAllowlistMiddleware(app)
