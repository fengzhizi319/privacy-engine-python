"""MFA lifecycle parity: enrollment, rebinding, recovery codes and sandboxes."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from engine.auth.totp import generate_code

from .conftest import ADMIN, change_initial_password, login


@pytest.fixture
def totp_clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Deterministic clock local to ``engine.auth.totp`` (no global time patch)."""
    clock = SimpleNamespace(now=time.time())
    fake_time = SimpleNamespace(time=lambda: clock.now, sleep=time.sleep)
    monkeypatch.setattr("engine.auth.totp.time", fake_time)
    return clock


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _start_session(client: TestClient) -> str:
    change_initial_password(client)
    return login(client)["token"]


def _setup(client: TestClient, token: str, **payload: str) -> dict[str, Any]:
    response = client.post("/v1/auth/mfa/setup", json=payload, headers=_bearer(token))
    assert response.status_code == 200, response.text
    return response.json()


def _enable(client: TestClient, token: str, secret: str, setup_id: str) -> dict[str, Any]:
    response = client.post(
        "/v1/auth/mfa/enable",
        json={"code": generate_code(secret), "setup_id": setup_id},
        headers=_bearer(token),
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_unenrolled_session_is_blocked_by_mfa_sandbox(
    api_client: TestClient, totp_clock: SimpleNamespace
) -> None:
    token = _start_session(api_client)

    response = api_client.get("/v1/privacy/probe", headers=_bearer(token))
    assert response.status_code == 403
    assert response.json()["code"] == "MFA_ENROLLMENT_REQUIRED"
    assert response.json()["detail"] == {"must_enroll_mfa": True}

    setup = _setup(api_client, token)
    assert setup["algorithm"] == "SHA1"
    assert setup["secret"]
    assert setup["otpauth"].startswith("otpauth://totp/")

    enabled = _enable(api_client, token, setup["secret"], setup["setup_id"])
    assert enabled["mfa_enabled"] is True
    assert len(enabled["recovery_codes"]) == 8
    assert enabled["token"]

    # ``/mfa/enable`` issues an MFA-authenticated session; business access now passes.
    response = api_client.get("/v1/privacy/probe", headers=_bearer(enabled["token"]))
    assert response.status_code == 200, response.text

    status = api_client.get("/v1/auth/mfa/status", headers=_bearer(enabled["token"]))
    assert status.status_code == 200
    assert status.json()["mfa_enabled"] is True
    assert status.json()["recovery_codes_remaining"] == 8


def test_login_requires_totp_after_enrollment(
    api_client: TestClient, totp_clock: SimpleNamespace
) -> None:
    token = _start_session(api_client)
    setup = _setup(api_client, token)
    _enable(api_client, token, setup["secret"], setup["setup_id"])

    missing = api_client.post(
        "/v1/auth/login", json={"username": ADMIN, "password": "Adm1n-Passw0rd-2!"}
    )
    assert missing.status_code == 401
    assert missing.json()["code"] == "MFA_REQUIRED"

    totp_clock.now += 31
    session = login(api_client, totp_code=generate_code(setup["secret"]))
    assert session["must_enroll_mfa"] is False
    response = api_client.get("/v1/privacy/probe", headers=_bearer(session["token"]))
    assert response.status_code == 200, response.text


def test_rebinding_requires_proof_of_possession(
    api_client: TestClient, totp_clock: SimpleNamespace
) -> None:
    token = _start_session(api_client)
    setup = _setup(api_client, token)
    enabled = _enable(api_client, token, setup["secret"], setup["setup_id"])
    session_token = enabled["token"]

    no_proof = api_client.post("/v1/auth/mfa/setup", json={}, headers=_bearer(session_token))
    assert no_proof.status_code == 403
    assert no_proof.json()["code"] == "MFA_REBIND_PROOF_REQUIRED"

    totp_clock.now += 31
    old_code = generate_code(setup["secret"])
    rebind = _setup(api_client, session_token, code=old_code)
    assert rebind["rebinding"] is True
    assert rebind["secret"] != setup["secret"]

    # New binding takes over only after /mfa/enable verifies a code from the new device.
    totp_clock.now += 31
    completed = _enable(api_client, session_token, rebind["secret"], rebind["setup_id"])
    assert completed["mfa_enabled"] is True
    assert len(completed["recovery_codes"]) == 8

    totp_clock.now += 31
    stale = api_client.post(
        "/v1/auth/login",
        json={
            "username": ADMIN,
            "password": "Adm1n-Passw0rd-2!",
            "totp_code": generate_code(setup["secret"]),
        },
    )
    assert stale.status_code == 401
    assert stale.json()["code"] == "INVALID_MFA_CODE"

    totp_clock.now += 31
    fresh = login(api_client, totp_code=generate_code(rebind["secret"]))
    assert fresh["must_enroll_mfa"] is False


def test_recovery_code_forces_rebinding_before_business_access(
    api_client: TestClient, totp_clock: SimpleNamespace
) -> None:
    token = _start_session(api_client)
    setup = _setup(api_client, token)
    enabled = _enable(api_client, token, setup["secret"], setup["setup_id"])
    recovery_code = enabled["recovery_codes"][0]

    recovered = api_client.post(
        "/v1/auth/login",
        json={
            "username": ADMIN,
            "password": "Adm1n-Passw0rd-2!",
            "recovery_code": recovery_code,
        },
    )
    assert recovered.status_code == 200, recovered.text
    recovered_body = recovered.json()["data"]
    assert recovered_body["recovery_code_used"] is True
    recovery_token = recovered_body["token"]

    blocked = api_client.get("/v1/privacy/probe", headers=_bearer(recovery_token))
    assert blocked.status_code == 403
    assert blocked.json()["code"] == "MFA_ENROLLMENT_REQUIRED"

    # Recovery code核销后旧绑定与剩余恢复码一并作废：可免旧因素重新绑定。
    new_setup = _setup(api_client, recovery_token)
    totp_clock.now += 31
    new_enabled = _enable(api_client, recovery_token, new_setup["secret"], new_setup["setup_id"])
    assert len(new_enabled["recovery_codes"]) == 8

    response = api_client.get("/v1/privacy/probe", headers=_bearer(new_enabled["token"]))
    assert response.status_code == 200


def test_mfa_setup_cancel_requires_session(api_client: TestClient) -> None:
    response = api_client.post(
        "/v1/auth/mfa/setup/cancel",
        json={"setup_id": "whatever"},
        headers={"Authorization": "Bearer sk-auth-conformance"},
    )
    assert response.status_code == 401


def test_key_count_counts_all_keys(api_client: TestClient, auth_runtime: Any) -> None:
    """Go ``UserSummary.KeyCount`` 为 ``len(u.APIKeys)``，包含停用/过期 Key。"""
    change_initial_password(api_client)
    user = auth_runtime.store.get_user(ADMIN)
    from engine.auth.models import APIKeyRecord

    user.api_keys["expired-key"] = APIKeyRecord(
        key_id="expired-key",
        name="expired",
        token_digest="deadbeef",
        token_prefix="psk_old",
        scopes=["privacy:mask"],
        active=False,
        created_at="2020-01-01T00:00:00Z",
        expires_at="2020-01-02T00:00:00Z",
    )
    summary = user.to_summary()
    assert summary["key_count"] == 1
