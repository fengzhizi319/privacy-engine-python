"""Gateway runtime settings (Go ``internal/config.LoadGateway`` parity).

环境变量前缀与默认值逐项对齐 Go 网关：

- ``PRIVACY_GATEWAY_HOST`` / ``_PORT``（默认 127.0.0.1:8000）
- ``PRIVACY_GATEWAY_GRPC_HOST`` / ``_GRPC_PORT``（默认 127.0.0.1:50000）
- ``PRIVACY_GATEWAY_OPS_PORT``（默认业务口 + 1000）
- ``PRIVACY_GATEWAY_BACKENDS`` / ``_GRPC_BACKENDS``（默认 127.0.0.1:8079 / :50051）
- ``PRIVACY_GATEWAY_STRATEGY``（默认 p2c）
- ``PRIVACY_GATEWAY_AUTH_ENABLED`` / ``_BACKEND_API_KEY``
- ``PRIVACY_GATEWAY_ALLOWED_CIDRS`` / ``_TRUSTED_PROXIES``
- ``PRIVACY_GATEWAY_REQUIRE_TLS`` / ``_REQUIRE_FORWARDED_PROTO``
- ``PRIVACY_GATEWAY_METRICS_API_KEY`` / ``_ADMIN_API_KEY``
- ``PRIVACY_GATEWAY_TLS_STACK`` / ``_TLS_CERT_FILE`` / ``_TLS_KEY_FILE`` / ``_TLS_CA_FILE``
- ``PRIVACY_GATEWAY_INBOUND_KEYS_FILE``

Python 无法原生终止 TLCP：``TLS_STACK=tlcp`` / ``both`` 时由 Go 终结器侧车
对外提供国密 mTLS，Python 网关只监听环回明文。非环回绑定 + TLCP 档直接拒启。
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field

_DEFAULT_BACKENDS = "127.0.0.1:8079"
_DEFAULT_GRPC_BACKENDS = "127.0.0.1:50051"
_TLS_STACKS = {"tlcp", "both", "tls"}
_PRODUCTION_LABELS = {"production", "prod"}


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name, "")
    if not raw:
        return default
    return raw.lower() in {"true", "1", "yes", "on"}


def _env_list(name: str) -> tuple[str, ...]:
    """Parse comma-separated or JSON-array list env values."""
    raw = _env_str(name, "")
    if not raw:
        return ()
    if raw.startswith("["):
        import json

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ()
        if not isinstance(parsed, list):
            return ()
        return tuple(item.strip() for item in parsed if isinstance(item, str) and item.strip())
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def is_loopback_host(host: str) -> bool:
    """Mirror Go ``pkgconfig.IsLoopbackHost`` (empty host means all interfaces)."""
    value = (host or "").strip()
    if not value:
        return False
    if value == "localhost":
        return True
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if value.count(":") == 1 and "." in value:
        value = value.rsplit(":", 1)[0]
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _split_addresses(raw: str, default: str) -> tuple[str, ...]:
    value = raw.strip() or default
    addresses = tuple(part.strip() for part in value.split(",") if part.strip())
    return addresses or tuple(part.strip() for part in default.split(","))


def _with_scheme(address: str, scheme: str) -> str:
    if "://" in address:
        return address.rstrip("/")
    return f"{scheme}://{address.rstrip('/')}"


def _production_labeled() -> bool:
    for name in ("APP_ENV", "PRIVACY_GATEWAY_ENV"):
        if _env_str(name).lower() in _PRODUCTION_LABELS:
            return True
    return False


@dataclass(frozen=True)
class GatewaySettings:
    """Immutable gateway configuration snapshot."""

    rest_host: str = "127.0.0.1"
    rest_port: int = 8000
    grpc_host: str = "127.0.0.1"
    grpc_port: int = 50000
    ops_port: int = 9000
    rest_backends: tuple[str, ...] = ()
    grpc_backends: tuple[str, ...] = ()
    strategy: str = "p2c"
    auth_enabled: bool = False
    backend_api_key: str = ""
    inbound_keys_file: str = ""
    allowed_cidrs: tuple[str, ...] = ()
    trusted_proxies: tuple[str, ...] = ()
    max_body_bytes: int = 32 * 1024 * 1024
    require_tls: bool = False
    require_forwarded_proto: bool = False
    metrics_api_key: str = ""
    admin_api_key: str = ""
    tls_stack: str = "tlcp"
    tls_cert_file: str = ""
    tls_key_file: str = ""
    tls_ca_file: str = ""
    ops_tls_stack: str = "both"
    ops_tls_cert_file: str = ""
    ops_tls_key_file: str = ""
    ops_tls_ca_file: str = ""
    health_check_interval: float = 5.0
    production: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def tls_enabled(self) -> bool:
        """Business-port standard TLS termination (Python cannot speak TLCP)."""
        return self.tls_stack == "tls" and bool(self.tls_cert_file and self.tls_key_file)

    @property
    def tls_require_client_auth(self) -> bool:
        return self.tls_enabled and bool(self.tls_ca_file)

    @property
    def ops_tls_enabled(self) -> bool:
        return self.ops_tls_stack == "tls" and bool(
            self.ops_tls_cert_file and self.ops_tls_key_file
        )

    @property
    def ops_tls_require_client_auth(self) -> bool:
        return self.ops_tls_enabled and bool(self.ops_tls_ca_file)

    @property
    def backend_scheme(self) -> str:
        return "https" if _env_bool("PRIVACY_GATEWAY_BACKEND_TLS_ENABLED") else "http"

    def rest_address(self) -> str:
        return f"{self.rest_host}:{self.rest_port}"

    def grpc_address(self) -> str:
        return f"{self.grpc_host}:{self.grpc_port}"

    def ops_address(self) -> str:
        return f"{self.rest_host}:{self.ops_port}"

    @classmethod
    def from_env(cls) -> GatewaySettings:
        """Load settings from the environment and enforce startup invariants."""
        rest_host = _env_str("PRIVACY_GATEWAY_HOST", "127.0.0.1")
        rest_port = _env_int("PRIVACY_GATEWAY_PORT", 8000)
        grpc_host = _env_str("PRIVACY_GATEWAY_GRPC_HOST", "127.0.0.1")
        grpc_port = _env_int("PRIVACY_GATEWAY_GRPC_PORT", 50000)
        ops_port = _env_int(
            "PRIVACY_GATEWAY_OPS_PORT",
            0 if rest_port == 0 else rest_port + 1000,
        )
        if rest_port == 0:
            ops_port = 0

        stack = _env_str("PRIVACY_GATEWAY_TLS_STACK", "tlcp").lower()
        if stack not in _TLS_STACKS:
            raise ValueError(
                f"PRIVACY_GATEWAY_TLS_STACK must be one of tlcp|both|tls, got {stack!r}"
            )
        ops_stack = _env_str("PRIVACY_GATEWAY_OPS_TLS_STACK", "both").lower()
        if ops_stack not in _TLS_STACKS:
            raise ValueError(
                f"PRIVACY_GATEWAY_OPS_TLS_STACK must be one of tlcp|both|tls, got {ops_stack!r}"
            )

        cert_file = _env_str("PRIVACY_GATEWAY_TLS_CERT_FILE")
        key_file = _env_str("PRIVACY_GATEWAY_TLS_KEY_FILE")
        ca_file = _env_str("PRIVACY_GATEWAY_TLS_CA_FILE")
        ops_cert_file = _env_str("PRIVACY_GATEWAY_OPS_TLS_CERT_FILE", cert_file)
        ops_key_file = _env_str("PRIVACY_GATEWAY_OPS_TLS_KEY_FILE", key_file)
        ops_ca_file = _env_str("PRIVACY_GATEWAY_OPS_TLS_CA_FILE", ca_file)
        if bool(cert_file) != bool(key_file):
            raise ValueError(
                "PRIVACY_GATEWAY_TLS_CERT_FILE and PRIVACY_GATEWAY_TLS_KEY_FILE must be set together"
            )
        if bool(ops_cert_file) != bool(ops_key_file):
            raise ValueError(
                "PRIVACY_GATEWAY_OPS_TLS_CERT_FILE and PRIVACY_GATEWAY_OPS_TLS_KEY_FILE "
                "must be set together"
            )
        if stack == "tls" and not (cert_file and key_file):
            raise ValueError(
                "PRIVACY_GATEWAY_TLS_STACK=tls requires PRIVACY_GATEWAY_TLS_CERT_FILE and "
                "PRIVACY_GATEWAY_TLS_KEY_FILE (mTLS, no cleartext listener)"
            )
        if stack == "tls" and not ca_file:
            raise ValueError(
                "PRIVACY_GATEWAY_TLS_STACK=tls is mTLS; PRIVACY_GATEWAY_TLS_CA_FILE is required"
            )
        if ops_stack == "tls" and not (ops_cert_file and ops_key_file and ops_ca_file):
            raise ValueError(
                "PRIVACY_GATEWAY_OPS_TLS_STACK=tls requires PRIVACY_GATEWAY_OPS_TLS_CERT_FILE, "
                "PRIVACY_GATEWAY_OPS_TLS_KEY_FILE and PRIVACY_GATEWAY_OPS_TLS_CA_FILE (mTLS)"
            )
        if stack in {"tlcp", "both"} and not (rest_host == "127.0.0.1" or is_loopback_host(rest_host)):
            raise ValueError(
                f"PRIVACY_GATEWAY_TLS_STACK={stack} cannot be terminated by the Python gateway "
                "(no native TLCP); bind a loopback address behind the Go terminator sidecar "
                "or set PRIVACY_GATEWAY_TLS_STACK=tls"
            )
        if ops_stack in {"tlcp", "both"} and not is_loopback_host(rest_host):
            raise ValueError(
                f"PRIVACY_GATEWAY_OPS_TLS_STACK={ops_stack} cannot be terminated by the Python "
                "gateway (no native TLCP); set PRIVACY_GATEWAY_OPS_TLS_STACK=tls for a "
                "non-loopback ops bind (no cleartext listener)"
            )

        backend_scheme = "https" if _env_bool("PRIVACY_GATEWAY_BACKEND_TLS_ENABLED") else "http"
        rest_backends = tuple(
            _with_scheme(address, backend_scheme)
            for address in _split_addresses(
                _env_str("PRIVACY_GATEWAY_BACKENDS", _DEFAULT_BACKENDS),
                _DEFAULT_BACKENDS,
            )
        )
        grpc_backends = _split_addresses(
            _env_str("PRIVACY_GATEWAY_GRPC_BACKENDS", _DEFAULT_GRPC_BACKENDS),
            _DEFAULT_GRPC_BACKENDS,
        )

        metrics_api_key = _env_str("PRIVACY_GATEWAY_METRICS_API_KEY")
        admin_api_key = _env_str("PRIVACY_GATEWAY_ADMIN_API_KEY", metrics_api_key)
        inbound_keys_file = _env_str("PRIVACY_GATEWAY_INBOUND_KEYS_FILE")
        production = _production_labeled()
        if production and not inbound_keys_file:
            raise ValueError(
                "production gateway requires PRIVACY_GATEWAY_INBOUND_KEYS_FILE to bind original callers"
            )

        settings = cls(
            rest_host=rest_host,
            rest_port=rest_port,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            ops_port=ops_port,
            rest_backends=rest_backends,
            grpc_backends=grpc_backends,
            strategy=_env_str("PRIVACY_GATEWAY_STRATEGY", "p2c").lower(),
            auth_enabled=_env_bool("PRIVACY_GATEWAY_AUTH_ENABLED"),
            backend_api_key=_env_str("PRIVACY_GATEWAY_BACKEND_API_KEY"),
            inbound_keys_file=inbound_keys_file,
            allowed_cidrs=_env_list("PRIVACY_GATEWAY_ALLOWED_CIDRS"),
            trusted_proxies=_env_list("PRIVACY_GATEWAY_TRUSTED_PROXIES"),
            max_body_bytes=_env_int("PRIVACY_GATEWAY_MAX_BODY_BYTES", 32 * 1024 * 1024),
            require_tls=_env_bool("PRIVACY_GATEWAY_REQUIRE_TLS"),
            require_forwarded_proto=_env_bool("PRIVACY_GATEWAY_REQUIRE_FORWARDED_PROTO"),
            metrics_api_key=metrics_api_key,
            admin_api_key=admin_api_key,
            tls_stack=stack,
            tls_cert_file=cert_file,
            tls_key_file=key_file,
            tls_ca_file=ca_file,
            ops_tls_stack=ops_stack,
            ops_tls_cert_file=ops_cert_file,
            ops_tls_key_file=ops_key_file,
            ops_tls_ca_file=ops_ca_file,
            health_check_interval=float(
                _env_str("PRIVACY_GATEWAY_HEALTH_CHECK_INTERVAL", "5.0") or "5.0"
            ),
            production=production,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Fail-closed startup invariants (Go ``Runtime.Validate`` gateway branch)."""
        for host in (self.rest_host, self.grpc_host):
            if is_loopback_host(host):
                continue
            if not self.auth_enabled and not self.backend_api_key:
                raise ValueError(
                    "non-loopback gateway bind requires PRIVACY_GATEWAY_AUTH_ENABLED=true "
                    "(per-request credential admission) or PRIVACY_GATEWAY_BACKEND_API_KEY "
                    "(fallback outbound credential); both are empty"
                )
        if not is_loopback_host(self.rest_host):
            if not self.metrics_api_key and not self.admin_api_key:
                raise ValueError(
                    "PRIVACY_GATEWAY_METRICS_API_KEY (or PRIVACY_GATEWAY_ADMIN_API_KEY) is "
                    f"required when the gateway binds a non-loopback address ({self.rest_host}): "
                    "/metrics and /gateway/backends must not be anonymously readable"
                )
        for path in (
            self.tls_cert_file,
            self.tls_key_file,
            self.tls_ca_file,
            self.ops_tls_cert_file,
            self.ops_tls_key_file,
            self.ops_tls_ca_file,
            self.inbound_keys_file,
        ):
            if path and not os.path.isfile(path):
                raise ValueError(f"gateway credential material not accessible: {path}")
        if self.production and self.inbound_keys_file and _is_repo_checkout_path(
            self.inbound_keys_file
        ):
            raise ValueError(
                "production gateway inbound keyfile must be outside the repository checkout"
            )


def _is_repo_checkout_path(path: str) -> bool:
    """Best-effort mirror of Go ``pkgconfig.IsRepoCheckoutPath``."""
    absolute = os.path.abspath(path)
    markers = (os.path.join(".git"), os.path.join("go.work"), os.path.join("AGENTS.md"))
    current = os.path.dirname(absolute)
    while True:
        if any(os.path.exists(os.path.join(current, marker)) for marker in markers):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent
