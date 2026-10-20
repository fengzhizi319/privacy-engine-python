"""mTLS CN whitelist management with hot-reload support.

提供基于 YAML 配置文件的 mTLS 客户端证书 CN 白名单管理，支持：
- 每个 CN 独立 scope 控制（最小权限原则）
- 基于文件 mtime 的热重载（请求驱动，被动检查）
- 线程安全的两阶段提交
- 向后兼容环境变量静态列表

Provides YAML-based mTLS client certificate CN whitelist management with:
- Per-CN scope control (least privilege principle)
- Hot-reload via file mtime detection (request-driven, passive check)
- Thread-safe two-phase commit
- Backward compatibility with environment variable static list
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


def normalize_fingerprint_list(values: list[str]) -> list[str]:
    """归一化证书 SHA-256 指纹：去冒号/空格、小写、去重、剔除非法项。

    与 Go ``pkg/tlsutil.whitelist.go::normalizeFingerprintList`` 同口径。
    """
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in values:
        cleaned = "".join(ch for ch in (raw or "").strip().lower() if ch not in {":", " "})
        if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def match_scope_pattern(pattern: str, value: str) -> bool:
    """Go ``matchScopePattern``：``*`` / 精确 / ``/Service/*`` 前缀。"""
    if pattern == "*":
        return True
    if len(pattern) > 2 and pattern.endswith("/*"):
        return value.startswith(pattern[:-1])
    return pattern == value


class CNEntry(BaseModel):
    """A single CN whitelist entry with metadata.

    同时兼容两种 YAML 形态：

    - Go 标准 ``clients:`` 条目：``allowed_scopes`` 为 gRPC 方法全名/通配模式，
      ``cert_fingerprints`` 执行证书钉扎（``scopes`` 未声明时为空）；
    - 历史 ``entries:`` 条目：``scopes`` 为 RBAC 权限串。

    Attributes:
        cn: Common Name from the client certificate subject.
        scopes: Legacy RBAC permission strings granted to this CN.
        allowed_scopes: Go-style gRPC method scope patterns. ``None`` 表示本条
            不走方法级匹配（历史条目）；空列表表示拒绝所有方法（fail-closed）。
        cert_fingerprints: Pinned leaf certificate SHA-256 fingerprints. Empty
            means no pinning (CN matching only).
        role: Optional role label kept for parity with the Go schema.
        description: Human-readable description for audit logs.
        enabled: Whether this entry is active.
    """

    cn: str
    scopes: list[str] = Field(default_factory=lambda: ["*"])
    allowed_scopes: list[str] | None = None
    cert_fingerprints: list[str] = Field(default_factory=list)
    role: str = ""
    description: str = ""
    enabled: bool = True

    @model_validator(mode="after")
    def _normalize(self) -> CNEntry:
        self.cert_fingerprints = normalize_fingerprint_list(self.cert_fingerprints)
        if self.allowed_scopes is not None:
            self.allowed_scopes = [str(item).strip() for item in self.allowed_scopes if str(item).strip()]
        return self

    @property
    def uses_method_scopes(self) -> bool:
        return self.allowed_scopes is not None

    def permits_method(self, full_method: str) -> bool:
        """方法级允许列表匹配；非方法型条目恒真（由 RBAC scope 裁决）。"""
        if self.allowed_scopes is None:
            return True
        return any(match_scope_pattern(pattern, full_method) for pattern in self.allowed_scopes)


class WhitelistConfig(BaseModel):
    """Root schema for the mTLS CN whitelist YAML configuration.

    Accepts the Go standard ``clients:`` key and the legacy Python ``entries:``
    key (superset). Duplicate CNs are resolved with ``clients`` taking priority.

    Attributes:
        version: Configuration format version for forward compatibility.
        clients: Go-style client entries (``allowed_scopes`` method matching).
        entries: Legacy Python entries (RBAC ``scopes``).
        default_scopes: Default scopes for CNs not explicitly listed.
    """

    version: str = "1.0"
    clients: list[CNEntry] = Field(default_factory=list)
    entries: list[CNEntry] = Field(default_factory=list)
    default_scopes: list[str] = Field(default_factory=list)
    # NOTE: default_scopes is parsed from the YAML but NOT currently wired into
    # the auth decision path (auth.py always rejects unlisted CNs). It is retained
    # for forward compatibility so the schema does not need a version bump when
    # the feature is eventually implemented. Do NOT rely on it for fail-open
    # semantics — all CNs not explicitly listed are always denied.

    @model_validator(mode="before")
    @classmethod
    def _seed_client_entries(cls, raw: Any) -> Any:
        """Go ``clients:`` 条目未写 ``scopes`` 时不得继承历史默认的 ``["*"]``。

        方法级条目的授权完全由 ``allowed_scopes`` 决定；默认 ``["*"]`` 会让
        一个只允许 ``/PrivacyService/Mask`` 的客户端在 Python 侧拿到全权身份。
        缺少 ``allowed_scopes`` 时置空（拒绝所有方法，fail-closed）。
        """
        if not isinstance(raw, dict):
            return raw
        clients = raw.get("clients")
        if not isinstance(clients, list):
            return raw
        patched: list[Any] = []
        for item in clients:
            if isinstance(item, dict):
                item = dict(item)
                item.setdefault("scopes", [])
                item.setdefault("allowed_scopes", [])
            patched.append(item)
        updated = dict(raw)
        updated["clients"] = patched
        return updated


class WhitelistManager:
    """Thread-safe mTLS CN whitelist manager with hot-reload support.

    The manager loads the whitelist from a YAML file on first access and
    reloads it when the file changes (detected via mtime). Reloading uses
    a two-phase commit: parse into a temporary buffer first, then atomically
    swap the cache on success.

    When no config file is provided, the manager falls back to a static
    list of CNs (all granted ["*"] scope) for backward compatibility with
    the ``PRIVACY_AUTH_MTLS_ALLOWED_CNS`` environment variable.

    Args:
        config_path: Path to the YAML whitelist configuration file.
                    If None, falls back to static list mode.
        static_cns: List of CNs for backward compatibility. Only used
                   when ``config_path`` is None.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        static_cns: list[str] | None = None,
    ) -> None:
        self._config_path = config_path
        self._static_cns = static_cns or []
        self._lock = threading.RLock()
        self._cache: dict[str, CNEntry] = {}
        self._default_scopes: list[str] = []
        self._last_mtime: float = 0.0
        self._last_load_time: float = 0.0
        self._load_error: str | None = None

        # Initial load
        self._load()

    def _load(self) -> None:
        """Load or reload the whitelist configuration.

        Uses two-phase commit: parse into temporary buffer, then atomically
        swap on success. On failure, retains the old cache and logs the error.
        """
        if self._config_path is None:
            # Static list mode: build cache from static_cns
            with self._lock:
                self._cache = {
                    cn: CNEntry(cn=cn, scopes=["*"], description="Static env var entry")
                    for cn in self._static_cns
                }
                self._default_scopes = []
                self._last_load_time = time.time()
                self._load_error = None
            return

        path = Path(self._config_path)
        if not path.exists():
            with self._lock:
                self._load_error = f"Whitelist config file not found: {path}"
                logger.warning(self._load_error)
            return

        try:
            import yaml

            content = path.read_text(encoding="utf-8")
            raw = yaml.safe_load(content)
            if not isinstance(raw, dict):
                raise ValueError(f"Whitelist config must be a YAML object, got {type(raw).__name__}")

            # Parse into WhitelistConfig for validation
            config = WhitelistConfig.model_validate(raw)

            # Build new cache (temporary buffer). Legacy entries first, then the
            # Go standard ``clients`` block wins on duplicate CNs.
            new_cache: dict[str, CNEntry] = {}
            for entry in [*config.entries, *config.clients]:
                if entry.enabled:
                    new_cache[entry.cn] = entry
                else:
                    logger.debug(
                        "Skipping disabled CN whitelist entry",
                        extra={"cn": entry.cn},
                    )

            # Two-phase commit: atomically swap.
            with self._lock:
                if not new_cache and self._cache:
                    # 与 Go 同口径：解析结果为空而现有配置非空时保留旧配置，
                    # 防止「误清空文件 / 覆写为空」导致全部客户端被静默放行。
                    self._load_error = "whitelist reload produced zero entries; keeping previous configuration"
                    self._last_mtime = path.stat().st_mtime
                    logger.error(self._load_error, extra={"path": str(path)})
                    return
                self._cache = new_cache
                self._default_scopes = config.default_scopes
                self._last_mtime = path.stat().st_mtime
                self._last_load_time = time.time()
                self._load_error = None

            logger.info(
                "mTLS CN whitelist loaded",
                extra={
                    "path": str(path),
                    "entry_count": len(new_cache),
                    "default_scopes": config.default_scopes,
                },
            )

        except Exception as exc:
            with self._lock:
                self._load_error = str(exc)
            logger.warning(
                "Failed to load mTLS CN whitelist, retaining previous config",
                extra={"path": str(path), "error": str(exc)},
            )

    def _check_reload(self) -> None:
        """Check if the config file has changed and trigger reload if needed.

        Called on every whitelist lookup to implement request-driven hot-reload.
        Uses mtime comparison for lightweight change detection.
        """
        if self._config_path is None:
            return

        path = Path(self._config_path)
        if not path.exists():
            return

        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return

        if current_mtime > self._last_mtime:
            logger.info(
                "mTLS CN whitelist config file changed, reloading",
                extra={"path": str(path)},
            )
            self._load()

    def get_entry(self, cn: str) -> CNEntry | None:
        """Look up a CN in the whitelist.

        Triggers a hot-reload check before lookup. Returns None if the CN
        is not in the whitelist or is disabled.

        Args:
            cn: Common Name from the client certificate.

        Returns:
            CNEntry if found and enabled, None otherwise.
        """
        self._check_reload()
        with self._lock:
            return self._cache.get(cn)

    def get_scopes(self, cn: str) -> list[str] | None:
        """Get the scopes for a CN.

        Convenience method that returns just the scopes list, or None if
        the CN is not whitelisted.

        Args:
            cn: Common Name from the client certificate.

        Returns:
            List of scope strings if CN is whitelisted, None otherwise.
        """
        entry = self.get_entry(cn)
        if entry is not None:
            return entry.scopes
        return None

    def authorize(
        self, cn: str, cert_sha256: str = "", full_method: str = ""
    ) -> tuple[CNEntry | None, str]:
        """Authorize a certificate identity against the whitelist.

        Mirrors the Go gRPC whitelist interceptor: CN must exist, pinned
        ``cert_fingerprints`` must match (constant-time), and Go-style
        ``allowed_scopes`` must cover ``full_method`` when provided.

        Returns ``(entry, "")`` on success, ``(None, reason)`` otherwise.
        """
        entry = self.get_entry(cn)
        if entry is None:
            return None, "cn_not_allowed"
        if entry.cert_fingerprints:
            presented = (cert_sha256 or "").strip().lower()
            if not presented:
                return None, "cert_fingerprint_missing"
            matched = False
            for pinned in entry.cert_fingerprints:
                if hmac.compare_digest(pinned.encode("utf-8"), presented.encode("utf-8")):
                    matched = True
            if not matched:
                return None, "cert_fingerprint_mismatch"
        if full_method and not entry.permits_method(full_method):
            return None, "method_not_allowed"
        return entry, ""

    def is_allowed(self, cn: str) -> bool:
        """Check if a CN is in the whitelist and enabled.

        Args:
            cn: Common Name from the client certificate.

        Returns:
            True if CN is whitelisted and enabled, False otherwise.
        """
        return self.get_entry(cn) is not None

    @property
    def default_scopes(self) -> list[str]:
        """Default scopes for unknown CNs (empty = fail-closed)."""
        with self._lock:
            return self._default_scopes.copy()

    @property
    def all_entries(self) -> list[CNEntry]:
        """Return a snapshot of all active whitelist entries."""
        self._check_reload()
        with self._lock:
            return list(self._cache.values())

    @property
    def last_load_time(self) -> float:
        """Timestamp of the last successful load."""
        with self._lock:
            return self._last_load_time

    @property
    def last_error(self) -> str | None:
        """Error message from the last failed load, or None."""
        with self._lock:
            return self._load_error

    def reload(self) -> None:
        """Force a reload of the whitelist configuration.

        Useful for testing or when an operator wants to explicitly trigger
        a reload without waiting for the next request.
        """
        self._load()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: WhitelistManager | None = None
_manager_lock = threading.Lock()


def get_whitelist_manager() -> WhitelistManager:
    """Get or create the module-level WhitelistManager singleton.

    The manager is initialized from environment variables on first call:
    - ``PRIVACY_AUTH_MTLS_WHITELIST_FILE``: Path to YAML config file.
    - ``PRIVACY_AUTH_MTLS_ALLOWED_CNS``: Fallback static CN list (comma-separated
      or JSON array) when no config file is specified.

    Returns:
        The module-level WhitelistManager instance.
    """
    global _manager
    if _manager is not None:
        return _manager

    with _manager_lock:
        if _manager is not None:
            return _manager

        import os

        from .config import _load_str_list_env

        whitelist_file = os.environ.get("PRIVACY_AUTH_MTLS_WHITELIST_FILE", "").strip()
        if whitelist_file:
            config_path = Path(whitelist_file)
            static_cns = None
        else:
            config_path = None
            static_cns = _load_str_list_env("PRIVACY_AUTH_MTLS_ALLOWED_CNS")

        _manager = WhitelistManager(config_path=config_path, static_cns=static_cns)
        return _manager


def reset_whitelist_manager() -> None:
    """Reset the module-level singleton. For testing only."""
    global _manager
    with _manager_lock:
        _manager = None
