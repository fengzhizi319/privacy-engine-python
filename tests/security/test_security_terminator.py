"""Tests for the Go TLCP terminator identity-header verifier.

覆盖与 ``services/privacy-engine-python/terminator/identity.go`` 的跨语言契约：

- HMAC-SM3 / HMAC-SHA256 规范串与签名；
- 时间戳偏差、环回来源、重复/缺失头；
- ``X-PrivShield-Client-SPKI-DER`` 与已签名 SPKI 摘要的绑定；
- Key 文件优先于环境变量、启动期配置校验。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from engine.security.config import SecuritySettings
from engine.security.sm3 import new_sm3
from engine.security.terminator import (
    HEADER_CLIENT_CN,
    HEADER_CLIENT_CERT_SHA256,
    HEADER_CLIENT_SPKI,
    HEADER_CLIENT_SPKI_DER,
    HEADER_IDENTITY_SIG,
    HEADER_IDENTITY_TIMESTAMP,
    TerminatorIdentityError,
    TerminatorIdentityVerifier,
    build_verifier,
    client_host_from_peer,
    client_host_from_scope,
    is_loopback_host,
    reset_key_cache,
)

KEY = b"terminator-test-key-0123456789"
NOW = 1_800_000_000
CN = "app-lz-client"
CERT_FP = hashlib.sha256(b"leaf-cert-der").hexdigest()

_SPKI_DER = (
    ec.generate_private_key(ec.SECP256R1())
    .public_key()
    .public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
)
SPKI_FP = hashlib.sha256(_SPKI_DER).hexdigest()


@pytest.fixture(autouse=True)
def _clear_key_cache():
    reset_key_cache()
    yield
    reset_key_cache()


def _canonical(
    method: str, path: str, cn: str, cert_fp: str, spki_fp: str, timestamp: int
) -> str:
    return (
        f"privshield-identity-v1\n{method.upper()}\n{path}\n"
        f"{cn}\n{cert_fp}\n{spki_fp}\n{timestamp}"
    )


def _sign(
    canonical: str, *, key: bytes = KEY, algo: str = "sm3"
) -> str:
    digestmod = hashlib.sha256 if algo == "sha256" else new_sm3
    return hmac.new(key, canonical.encode("utf-8"), digestmod=digestmod).hexdigest()


def identity_headers(
    *,
    method: str = "GET",
    path: str = "/v1/privacy/mask",
    key: bytes = KEY,
    algo: str = "sm3",
    cn: str = CN,
    cert_fp: str = CERT_FP,
    spki_fp: str = SPKI_FP,
    timestamp: int = NOW,
    spki_der: bytes | None = _SPKI_DER,
) -> dict[str, list[str]]:
    signature = _sign(
        _canonical(method, path, cn, cert_fp, spki_fp, timestamp), key=key, algo=algo
    )
    values = {
        HEADER_CLIENT_CN: [cn],
        HEADER_CLIENT_CERT_SHA256: [cert_fp],
        HEADER_CLIENT_SPKI: [spki_fp],
        HEADER_IDENTITY_TIMESTAMP: [str(timestamp)],
        HEADER_IDENTITY_SIG: [signature],
    }
    if spki_der is not None:
        values[HEADER_CLIENT_SPKI_DER] = [base64.b64encode(spki_der).decode("ascii")]
    return values


def _verifier(algo: str = "sm3", skew: int = 60) -> TerminatorIdentityVerifier:
    return TerminatorIdentityVerifier(KEY, algo, skew, now=lambda: NOW)


@pytest.mark.parametrize("algo", ["sm3", "sha256"])
def test_valid_signature_is_accepted(algo: str) -> None:
    peer = _verifier(algo).verify(
        method="GET",
        path="/v1/privacy/mask",
        values=identity_headers(algo=algo),
        client_host="127.0.0.1",
    )
    assert peer is not None
    assert peer.common_name == CN
    assert peer.cert_sha256 == CERT_FP
    assert peer.spki_sha256 == SPKI_FP
    assert peer.spki_der == _SPKI_DER
    assert peer.spki_der_base64 == base64.b64encode(_SPKI_DER).decode("ascii")


def test_absent_headers_mean_no_peer() -> None:
    assert (
        _verifier().verify(
            method="GET",
            path="/v1/privacy/mask",
            values={},
            client_host="127.0.0.1",
        )
        is None
    )


def test_partial_header_group_is_rejected() -> None:
    values = identity_headers()
    values.pop(HEADER_IDENTITY_SIG)
    with pytest.raises(TerminatorIdentityError, match="exactly once"):
        _verifier().verify(
            method="GET", path="/v1/privacy/mask", values=values, client_host="127.0.0.1"
        )


def test_duplicate_header_is_rejected() -> None:
    values = identity_headers()
    values[HEADER_CLIENT_CN] = [CN, CN]
    with pytest.raises(TerminatorIdentityError, match="exactly once"):
        _verifier().verify(
            method="GET", path="/v1/privacy/mask", values=values, client_host="127.0.0.1"
        )


def test_non_loopback_source_is_rejected() -> None:
    with pytest.raises(TerminatorIdentityError, match="loopback"):
        _verifier().verify(
            method="GET",
            path="/v1/privacy/mask",
            values=identity_headers(),
            client_host="10.1.2.3",
        )


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"])
def test_loopback_variants_are_accepted(host: str) -> None:
    peer = _verifier().verify(
        method="GET",
        path="/v1/privacy/mask",
        values=identity_headers(),
        client_host=host,
    )
    assert peer is not None


def test_wrong_signature_is_rejected() -> None:
    values = identity_headers(key=b"a-different-key-0123456789")
    with pytest.raises(TerminatorIdentityError, match="signature"):
        _verifier().verify(
            method="GET", path="/v1/privacy/mask", values=values, client_host="127.0.0.1"
        )


def test_signature_is_bound_to_method_and_path() -> None:
    values = identity_headers(method="GET", path="/v1/privacy/mask")
    with pytest.raises(TerminatorIdentityError, match="signature"):
        _verifier().verify(
            method="POST", path="/v1/privacy/mask", values=values, client_host="127.0.0.1"
        )
    with pytest.raises(TerminatorIdentityError, match="signature"):
        _verifier().verify(
            method="GET",
            path="/v1/privacy/dp/count",
            values=values,
            client_host="127.0.0.1",
        )


@pytest.mark.parametrize("timestamp", [NOW - 61, NOW + 61])
def test_timestamp_outside_skew_is_rejected(timestamp: int) -> None:
    with pytest.raises(TerminatorIdentityError, match="skew"):
        _verifier(skew=60).verify(
            method="GET",
            path="/v1/privacy/mask",
            values=identity_headers(timestamp=timestamp),
            client_host="127.0.0.1",
        )


def test_malformed_hex_fields_are_rejected() -> None:
    values = identity_headers(cert_fp="not-hex")
    values[HEADER_IDENTITY_SIG] = ["0" * 64]
    with pytest.raises(TerminatorIdentityError, match="cert fingerprint"):
        _verifier().verify(
            method="GET", path="/v1/privacy/mask", values=values, client_host="127.0.0.1"
        )


def test_der_not_matching_signed_spki_is_rejected() -> None:
    other_der = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    values = identity_headers(spki_der=other_der)
    with pytest.raises(TerminatorIdentityError, match="SPKI DER"):
        _verifier().verify(
            method="GET", path="/v1/privacy/mask", values=values, client_host="127.0.0.1"
        )


def test_der_is_optional_for_control_plane_requests() -> None:
    peer = _verifier().verify(
        method="GET",
        path="/health",
        values=identity_headers(method="GET", path="/health", spki_der=None),
        client_host="127.0.0.1",
    )
    assert peer is not None
    assert peer.spki_der == b""
    assert peer.spki_der_base64 == ""


def test_client_host_parsing() -> None:
    assert client_host_from_peer("ipv4:127.0.0.1:51234") == "127.0.0.1"
    assert client_host_from_peer("ipv6:[::1]:51234") == "::1"
    assert client_host_from_peer("unix:/tmp/engine.sock") is None
    assert client_host_from_peer(None) is None
    assert client_host_from_scope({"client": ("127.0.0.1", 1234)}) == "127.0.0.1"
    assert client_host_from_scope({}) is None
    assert is_loopback_host("127.0.0.2")
    assert not is_loopback_host("192.168.1.10")
    assert not is_loopback_host("")


def test_key_file_takes_precedence_over_env(tmp_path: Path) -> None:
    file_key = b"file-key-0123456789abcdef"
    env_key = b"env-key-0123456789abcdefg"
    key_file = tmp_path / "identity.key"
    key_file.write_bytes(file_key + b"\n")

    settings = SecuritySettings(
        terminator_identity_enabled=True,
        terminator_identity_hmac_algo="sm3",
        terminator_identity_hmac_key=env_key.decode(),
        terminator_identity_hmac_key_file=key_file,
        terminator_identity_max_skew_seconds=60,
    )
    verifier = build_verifier(settings)
    assert verifier is not None
    timestamp = int(time.time())

    peer = verifier.verify(
        method="GET",
        path="/v1/privacy/mask",
        values=identity_headers(key=file_key, timestamp=timestamp),
        client_host="127.0.0.1",
    )
    assert peer is not None

    with pytest.raises(TerminatorIdentityError, match="signature"):
        verifier.verify(
            method="GET",
            path="/v1/privacy/mask",
            values=identity_headers(key=env_key, timestamp=timestamp),
            client_host="127.0.0.1",
        )


def test_build_verifier_requires_key_material() -> None:
    # 启动期模型校验先于任何请求失败（fail-fast），而不是首个请求 401。
    with pytest.raises(ValueError, match="no HMAC key"):
        SecuritySettings(terminator_identity_enabled=True)


def test_build_verifier_disabled_returns_none() -> None:
    assert build_verifier(SecuritySettings()) is None


def test_settings_validation_fails_fast() -> None:
    with pytest.raises(ValueError, match="at least 16 bytes"):
        SecuritySettings(
            terminator_identity_enabled=True,
            terminator_identity_hmac_key="short",
        )
    with pytest.raises(ValueError, match="PRIVACY_METRICS_ALLOW_MTLS"):
        SecuritySettings(metrics_allow_mtls=True)
    # Literal 字段在 Pydantic 层拒绝未知算法（ValidationError 即 ValueError）。
    with pytest.raises(ValueError):
        SecuritySettings(
            terminator_identity_enabled=True,
            terminator_identity_hmac_key=KEY.decode(),
            terminator_identity_hmac_algo="md5",
        )


def test_verifier_rejects_short_keys_and_bad_skew() -> None:
    with pytest.raises(TerminatorIdentityError, match="at least 16 bytes"):
        TerminatorIdentityVerifier(b"short", "sm3", 60)
    with pytest.raises(TerminatorIdentityError, match="max skew"):
        TerminatorIdentityVerifier(KEY, "sm3", 0)
    with pytest.raises(TerminatorIdentityError, match="unsupported"):
        TerminatorIdentityVerifier(KEY, "md5", 60)


def test_verifier_uses_wall_clock_by_default() -> None:
    verifier = TerminatorIdentityVerifier(KEY, "sm3", 60)
    values = identity_headers(timestamp=int(time.time()))
    assert (
        verifier.verify(
            method="GET", path="/v1/privacy/mask", values=values, client_host="127.0.0.1"
        )
        is not None
    )
