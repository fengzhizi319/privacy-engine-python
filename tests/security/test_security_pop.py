"""Tests for the PoP binding state machine (Go ``pkg/auth/pop_decision.go``)."""

from __future__ import annotations

import base64
import hashlib

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from engine.auth.errors import AuthError
from engine.security.auth import _enforce_credential_policy
from engine.security.identity import Identity
from engine.security.pop import PoPOutcome, decide_pop
from engine.security.terminator import TerminatorPeer

_SPKI_DER = (
    ec.generate_private_key(ec.SECP256R1())
    .public_key()
    .public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
)
CERT_FP = hashlib.sha256(b"leaf-der").hexdigest()
SPKI_FP = hashlib.sha256(_SPKI_DER).hexdigest()


def _peer(*, cn: str = "client-a", spki_der: bytes = _SPKI_DER) -> TerminatorPeer:
    return TerminatorPeer(
        common_name=cn,
        cert_sha256=CERT_FP,
        spki_sha256=SPKI_FP,
        spki_der=spki_der,
        timestamp=1_800_000_000,
    )


def _identity(**overrides) -> Identity:
    values = {"service_type": "internal", "name": "svc:client", "scopes": ["*"]}
    values.update(overrides)
    return Identity(**values)


def test_unbound_non_workload_is_allowed_without_peer() -> None:
    outcome = decide_pop(_identity(), None, health_exempt=False)
    assert outcome.allowed


def test_unbound_workload_is_restricted() -> None:
    outcome = decide_pop(
        _identity(principal_type="workload"), None, health_exempt=False
    )
    assert not outcome.allowed
    assert outcome.decision == "unbound_restricted"
    assert outcome.code == "UNBOUND_KEY_RESTRICTED"
    assert outcome.status == 403


def test_health_exempts_bound_credentials_without_peer() -> None:
    identity = _identity(bound_public_key_der=base64.b64encode(_SPKI_DER).decode())
    assert decide_pop(identity, None, health_exempt=True).allowed


def test_bound_credential_requires_peer_certificate() -> None:
    identity = _identity(bound_cn="client-a")
    outcome = decide_pop(identity, None, health_exempt=False)
    assert not outcome.allowed
    assert outcome.decision == "cert_required"
    assert outcome.code == "POP_TLS_CERT_REQUIRED"
    assert outcome.status == 401


def test_bound_cn_mismatch() -> None:
    identity = _identity(bound_cn="someone-else")
    outcome = decide_pop(identity, _peer(), health_exempt=False)
    assert not outcome.allowed
    assert outcome.code == "CLIENT_CN_MISMATCH"


def test_bound_cn_is_case_insensitive() -> None:
    identity = _identity(bound_cn="CLIENT-A")
    assert decide_pop(identity, _peer(), health_exempt=False).allowed


def test_bound_fingerprint_match_and_mismatch() -> None:
    matching = _identity(bound_cert_fingerprint=f"sha256:{CERT_FP}")
    assert decide_pop(matching, _peer(), health_exempt=False).allowed

    mismatch = _identity(bound_cert_fingerprint="sha256:" + "0" * 64)
    outcome = decide_pop(mismatch, _peer(), health_exempt=False)
    assert not outcome.allowed
    assert outcome.code == "CERT_FINGERPRINT_PIN_MISMATCH"


@pytest.mark.parametrize(
    "binding",
    [
        "md5:" + CERT_FP,
        "sha256:not-hex",
        CERT_FP,
        "sm3:" + CERT_FP,
    ],
)
def test_unevaluable_fingerprint_binding_is_malformed(binding: str) -> None:
    outcome = decide_pop(
        _identity(bound_cert_fingerprint=binding), _peer(), health_exempt=False
    )
    assert not outcome.allowed
    assert outcome.code == "CERT_BINDING_MALFORMED"


def test_bound_public_key_match_and_mismatch() -> None:
    binding = base64.b64encode(_SPKI_DER).decode("ascii")
    assert decide_pop(
        _identity(bound_public_key_der=binding), _peer(), health_exempt=False
    ).allowed

    other_der = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    other = base64.b64encode(other_der).decode("ascii")
    outcome = decide_pop(
        _identity(bound_public_key_der=other), _peer(), health_exempt=False
    )
    assert not outcome.allowed
    assert outcome.code == "POP_PUBLIC_KEY_MISMATCH"


def test_malformed_public_key_binding() -> None:
    outcome = decide_pop(
        _identity(bound_public_key_der="md5:" + CERT_FP),
        _peer(),
        health_exempt=False,
    )
    assert not outcome.allowed
    assert outcome.code == "CERT_BINDING_MALFORMED"


def test_public_key_binding_without_der_in_peer_is_mismatch() -> None:
    binding = base64.b64encode(_SPKI_DER).decode("ascii")
    outcome = decide_pop(
        _identity(bound_public_key_der=binding),
        _peer(spki_der=b""),
        health_exempt=False,
    )
    assert not outcome.allowed
    assert outcome.code == "POP_PUBLIC_KEY_MISMATCH"


def test_none_identity_is_allowed() -> None:
    assert decide_pop(None, None, health_exempt=False).allowed


def test_credential_policy_uses_pop_for_business_paths() -> None:
    bound = _identity(bound_cn="client-a")
    with pytest.raises(AuthError) as excinfo:
        _enforce_credential_policy(
            bound, method="POST", path="/v1/privacy/mask", contract_header=""
        )
    assert excinfo.value.code == "POP_TLS_CERT_REQUIRED"
    assert excinfo.value.status_code == 401

    _enforce_credential_policy(
        bound,
        method="POST",
        path="/v1/privacy/mask",
        contract_header="",
        peer=_peer(),
    )


def test_credential_policy_exempts_health_and_public_paths() -> None:
    bound = _identity(bound_cn="client-a")
    _enforce_credential_policy(
        bound, method="GET", path="/health", contract_header=""
    )
    _enforce_credential_policy(
        bound, method="POST", path="/v1/auth/login", contract_header=""
    )


def test_pop_outcome_shape_matches_go_decisions() -> None:
    outcome = PoPOutcome.deny("cn_mismatch", 403, "CLIENT_CN_MISMATCH", "x")
    assert not outcome.allowed
    assert outcome.decision == "cn_mismatch"
    assert outcome.status == 403
