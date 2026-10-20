"""终结器身份头校验 / Verified terminator identity.

Python 引擎不自行终结 TLCP，也不直接暴露非回环地址：mTLS 由 Go 终结器完成，
回环链路是明文。终结器把客户端证书身份以 **HMAC 签名的应用层头** 注入，
本模块在 Python 侧做防御性二次校验：

- 只接受环回对端（终结器与引擎同 Pod / 同机）；
- 时间戳偏差不超过 ``PRIVACY_TERMINATOR_IDENTITY_MAX_SKEW_SECONDS``；
- HMAC-SM3 / HMAC-SHA256 常量时间比对签名；
- ``X-PrivShield-Client-SPKI-DER`` 解码后必须与已签名的 SPKI 摘要一致。

签名规范串与 Go 终结器 ``terminator/identity.go`` 完全一致::

    privshield-identity-v1\\n
    <METHOD>\\n
    <PATH>\\n
    <CN>\\n
    <CERT_SHA256>\\n
    <SPKI_SHA256>\\n
    <TIMESTAMP>

gRPC 侧 METHOD 固定为 ``POST``、PATH 为完整方法名。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from .sm3 import new_sm3

if TYPE_CHECKING:
    from .config import SecuritySettings

IDENTITY_CANONICAL_VERSION = "privshield-identity-v1"

HEADER_CLIENT_CN = "x-privshield-client-cn"
HEADER_CLIENT_CERT_SHA256 = "x-privshield-client-cert-sha256"
HEADER_CLIENT_SPKI = "x-privshield-client-spki"
HEADER_CLIENT_SPKI_DER = "x-privshield-client-spki-der"
HEADER_IDENTITY_TIMESTAMP = "x-privshield-identity-timestamp"
HEADER_IDENTITY_SIG = "x-privshield-identity-sig"

IDENTITY_HEADERS: tuple[str, ...] = (
    HEADER_CLIENT_CN,
    HEADER_CLIENT_CERT_SHA256,
    HEADER_CLIENT_SPKI,
    HEADER_CLIENT_SPKI_DER,
    HEADER_IDENTITY_TIMESTAMP,
    HEADER_IDENTITY_SIG,
)

# 签名头必须成组出现；CN/CERT/SPKI/TS/SIG 是签名覆盖的最小集合。
_REQUIRED_HEADERS: tuple[str, ...] = (
    HEADER_CLIENT_CN,
    HEADER_CLIENT_CERT_SHA256,
    HEADER_CLIENT_SPKI,
    HEADER_IDENTITY_TIMESTAMP,
    HEADER_IDENTITY_SIG,
)

_HEX_DIGITS = frozenset("0123456789abcdef")


class TerminatorIdentityError(Exception):
    """身份头缺失、伪造、超时或来源非环回。"""


@dataclass(frozen=True)
class TerminatorPeer:
    """已校验的终结器对端证书视图。

    ``spki_der`` 仅在终结器注入了 raw DER 时非空；服务账号激活需要它，
    因为绑定值与 Go 侧一样存的是公钥本身而不只是摘要。
    """

    common_name: str
    cert_sha256: str
    spki_sha256: str
    spki_der: bytes
    timestamp: int

    @property
    def spki_der_base64(self) -> str:
        """激活接口使用的 ``actual_spki``（标准 base64 SPKI DER）。"""
        if not self.spki_der:
            return ""
        return base64.b64encode(self.spki_der).decode("ascii")


def is_loopback_host(host: str | None) -> bool:
    """True 当且仅当 ``host`` 是 IPv4 / IPv6 环回或 localhost。"""
    if not host:
        return False
    value = host.strip().strip("[]")
    if not value:
        return False
    if value.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)


def client_host_from_peer(peer: str | None) -> str | None:
    """从 gRPC ``context.peer()`` 字符串提取主机（``ipv4:`` / ``ipv6:``）。

    非 TCP 传输（如 ``unix:``）返回 ``None``，调用方按非环回处理（fail-closed）。
    """
    if not peer:
        return None
    if peer.startswith("ipv4:"):
        host = peer[len("ipv4:") :]
        return host.rsplit(":", 1)[0] if ":" in host else host
    if peer.startswith("ipv6:"):
        host = peer[len("ipv6:") :]
        if host.startswith("["):
            closing = host.find("]")
            if closing > 0:
                return host[1:closing]
        return host.rsplit(":", 1)[0] if ":" in host else host
    return None


def client_host_from_scope(scope: Mapping[str, Any]) -> str | None:
    """从 ASGI scope 的 ``client`` 元组提取主机。"""
    client = scope.get("client")
    if not client:
        return None
    try:
        return str(client[0])
    except (TypeError, IndexError):
        return None


def _canonical(
    method: str, path: str, cn: str, cert_sha256: str, spki_sha256: str, timestamp: int
) -> str:
    return (
        f"{IDENTITY_CANONICAL_VERSION}\n"
        f"{method.upper()}\n"
        f"{path}\n"
        f"{cn}\n"
        f"{cert_sha256}\n"
        f"{spki_sha256}\n"
        f"{timestamp}"
    )


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(char in _HEX_DIGITS for char in value)


class TerminatorIdentityVerifier:
    """校验终结器注入的身份头并返回对端证书视图。"""

    def __init__(
        self,
        key: bytes,
        algo: str,
        max_skew_seconds: int,
        *,
        now: Callable[[], float] = time.time,
    ) -> None:
        normalized = (algo or "").strip().lower()
        if normalized not in {"sm3", "sha256"}:
            raise TerminatorIdentityError(
                f"unsupported identity HMAC algorithm {algo!r} (want sm3 or sha256)"
            )
        if len(key) < 16:
            raise TerminatorIdentityError(
                f"identity HMAC key must be at least 16 bytes (got {len(key)})"
            )
        if max_skew_seconds <= 0:
            raise TerminatorIdentityError("identity max skew must be > 0")
        self._key = bytes(key)
        self._algo = normalized
        self._max_skew = int(max_skew_seconds)
        self._now = now

    @property
    def algo(self) -> str:
        return self._algo

    def _sign(self, canonical: str) -> str:
        digestmod = hashlib.sha256 if self._algo == "sha256" else new_sm3
        return hmac.new(self._key, canonical.encode("utf-8"), digestmod=digestmod).hexdigest()

    def verify(
        self,
        *,
        method: str,
        path: str,
        values: Mapping[str, list[str]],
        client_host: str | None,
    ) -> TerminatorPeer | None:
        """校验身份头。

        没有任何身份头时返回 ``None``（请求未出示证书 / 不是经终结器的 mTLS 调用）。
        一旦出现任一身份头，就必须完整且校验通过，否则抛
        :class:`TerminatorIdentityError` —— 绝不在校验失败后静默降级为「无证书」。
        """
        present = [name for name in IDENTITY_HEADERS if values.get(name)]
        if not present:
            return None
        if not is_loopback_host(client_host):
            raise TerminatorIdentityError(
                "terminator identity headers are only accepted from loopback"
            )

        def single(name: str) -> str:
            entries = values.get(name) or []
            if len(entries) != 1:
                raise TerminatorIdentityError(f"identity header {name} must appear exactly once")
            value = (entries[0] or "").strip()
            if not value:
                raise TerminatorIdentityError(f"identity header {name} is empty")
            return value

        for name in _REQUIRED_HEADERS:
            single(name)

        cn = single(HEADER_CLIENT_CN)
        cert_fp = single(HEADER_CLIENT_CERT_SHA256).lower()
        spki_fp = single(HEADER_CLIENT_SPKI).lower()
        timestamp_raw = single(HEADER_IDENTITY_TIMESTAMP)
        signature = single(HEADER_IDENTITY_SIG).lower()

        if not _is_lower_hex(cert_fp, 64):
            raise TerminatorIdentityError("identity cert fingerprint is malformed")
        if not _is_lower_hex(spki_fp, 64):
            raise TerminatorIdentityError("identity SPKI fingerprint is malformed")
        if not _is_lower_hex(signature, 64):
            raise TerminatorIdentityError("identity signature is malformed")
        try:
            timestamp = int(timestamp_raw, 10)
        except ValueError as exc:
            raise TerminatorIdentityError("identity timestamp is malformed") from exc
        if timestamp <= 0:
            raise TerminatorIdentityError("identity timestamp is malformed")
        if abs(self._now() - timestamp) > self._max_skew:
            raise TerminatorIdentityError("identity header timestamp is outside the allowed skew")

        canonical = _canonical(method, path, cn, cert_fp, spki_fp, timestamp)
        expected = self._sign(canonical)
        if not hmac.compare_digest(expected, signature):
            raise TerminatorIdentityError("identity signature verification failed")

        spki_der = b""
        der_entries = values.get(HEADER_CLIENT_SPKI_DER) or []
        if der_entries:
            if len(der_entries) != 1:
                raise TerminatorIdentityError(
                    f"identity header {HEADER_CLIENT_SPKI_DER} must appear exactly once"
                )
            encoded = (der_entries[0] or "").strip()
            if encoded:
                try:
                    spki_der = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise TerminatorIdentityError("identity SPKI DER is not valid base64") from exc
                if not spki_der:
                    raise TerminatorIdentityError("identity SPKI DER is empty")
                if hashlib.sha256(spki_der).hexdigest() != spki_fp:
                    raise TerminatorIdentityError(
                        "identity SPKI DER does not match the signed SPKI fingerprint"
                    )

        return TerminatorPeer(
            common_name=cn,
            cert_sha256=cert_fp,
            spki_sha256=spki_fp,
            spki_der=spki_der,
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Key material
# ---------------------------------------------------------------------------

_key_lock = threading.Lock()
_key_cache: dict[tuple[str, int, int], bytes] = {}


def _read_key_file(path: Path) -> bytes:
    try:
        stat = path.stat()
    except OSError as exc:
        raise TerminatorIdentityError(
            f"read terminator identity key file {path}: {exc}"
        ) from exc
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _key_lock:
        cached = _key_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TerminatorIdentityError(
            f"read terminator identity key file {path}: {exc}"
        ) from exc
    key = raw.strip()
    if not key:
        raise TerminatorIdentityError(f"terminator identity key file {path} is empty")
    with _key_lock:
        _key_cache[cache_key] = key
    return key


def load_identity_key(settings: SecuritySettings) -> bytes:
    """按 Go 终结器同口径读取 HMAC Key：文件优先，其次环境变量。"""
    path = settings.terminator_identity_hmac_key_file
    if path:
        return _read_key_file(Path(path))
    value = (settings.terminator_identity_hmac_key or "").strip()
    if not value:
        raise TerminatorIdentityError(
            "terminator identity verification is enabled but no HMAC key is configured "
            "(set PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY_FILE or "
            "PRIVACY_TERMINATOR_IDENTITY_HMAC_KEY, or disable with "
            "PRIVACY_TERMINATOR_IDENTITY_ENABLED=false)"
        )
    return value.encode("utf-8")


def build_verifier(settings: SecuritySettings) -> TerminatorIdentityVerifier | None:
    """按安全配置构建校验器；未启用时返回 ``None``。"""
    if not settings.terminator_identity_enabled:
        return None
    return TerminatorIdentityVerifier(
        load_identity_key(settings),
        settings.terminator_identity_hmac_algo,
        settings.terminator_identity_max_skew_seconds,
    )


def reset_key_cache() -> None:
    """清空 Key 文件缓存。仅测试使用。"""
    with _key_lock:
        _key_cache.clear()
