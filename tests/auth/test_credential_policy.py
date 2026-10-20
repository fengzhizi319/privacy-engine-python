"""Credential policy (audience / contract / drain-only) and sandbox parity.

Go 的 ``CheckCredentialPolicy`` 与密码/MFA 沙箱是 REST、gRPC、挂载子应用共用的
准入裁决；这里同时覆盖 HTTP 端到端与直接调用，避免只验证到路由层。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from engine.auth.errors import AuthError
from engine.auth.service import AuthService
from engine.security.auth import _enforce_credential_policy
from engine.security.identity import Identity

from .conftest import ADMIN, change_initial_password, login

SERVICE_AUDIENCE = "privacy-engine"


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _user_key(
    auth_runtime: Any,
    *,
    name: str,
    scopes: list[str],
    audiences: list[str] | None = None,
    contract: dict[str, Any] | None = None,
) -> str:
    record, token = auth_runtime.store.issue_api_key(
        ADMIN, name, scopes, audiences or [], 600, "", contract
    )
    assert record.key_id
    return token


def _policy(identity: Identity, *, method: str, path: str, contract_header: str = "") -> None:
    _enforce_credential_policy(
        identity, method=method, path=path, contract_header=contract_header
    )


def test_audience_mismatch_is_denied() -> None:
    identity = Identity(
        "internal",
        "svc:caller",
        ["privacy:mask"],
        allowed_audiences=["other-service"],
    )
    with pytest.raises(AuthError) as excinfo:
        _policy(identity, method="POST", path="/v1/privacy/mask-probe")
    assert excinfo.value.code == "AUDIENCE_MISMATCH"

    wildcard = Identity(
        "internal", "svc:caller", ["privacy:mask"], allowed_audiences=["*"]
    )
    _policy(wildcard, method="POST", path="/v1/privacy/mask-probe")

    issuer_bound = Identity(
        "internal", "svc:caller", ["privacy:mask"], audience=SERVICE_AUDIENCE
    )
    _policy(issuer_bound, method="POST", path="/v1/privacy/mask-probe")


def test_contract_version_and_allow_list_are_enforced() -> None:
    identity = Identity(
        "internal",
        "svc:caller",
        ["privacy:mask", "privacy:dp"],
        contract_version="v1",
        contract_allowed_paths=["POST /v1/privacy/mask-probe"],
    )
    _policy(identity, method="POST", path="/v1/privacy/mask-probe")

    with pytest.raises(AuthError) as excinfo:
        _policy(identity, method="POST", path="/v1/privacy/dp-probe")
    assert excinfo.value.code == "CONTRACT_VIOLATION"
    assert "allow-list" in excinfo.value.message

    with pytest.raises(AuthError) as excinfo:
        _policy(
            identity,
            method="POST",
            path="/v1/privacy/mask-probe",
            contract_header="v2",
        )
    assert excinfo.value.code == "CONTRACT_VIOLATION"
    assert "version mismatch" in excinfo.value.message

    # 未登记契约却自报版本也必须拒绝。
    no_contract = Identity("internal", "svc:caller", ["privacy:mask"])
    with pytest.raises(AuthError) as excinfo:
        _policy(
            no_contract,
            method="POST",
            path="/v1/privacy/mask-probe",
            contract_header="v1",
        )
    assert excinfo.value.code == "CONTRACT_VIOLATION"


def test_drain_only_credential_allows_only_health() -> None:
    identity = Identity(
        "internal", "svc:draining", ["*"], binding_status="drain_only"
    )
    with pytest.raises(AuthError) as excinfo:
        _policy(identity, method="POST", path="/v1/privacy/mask-probe")
    assert excinfo.value.code == "CREDENTIAL_DRAIN_ONLY"

    _policy(identity, method="GET", path="/health")
    _policy(identity, method="POST", path="/v1/auth/login")


def test_http_contract_and_audience_enforcement(
    api_client: TestClient, auth_runtime: Any
) -> None:
    change_initial_password(api_client)

    contract_token = _user_key(
        auth_runtime,
        name="contract",
        scopes=["privacy:mask", "privacy:dp"],
        contract={
            "version": "v1",
            "allowed_paths": ["POST /v1/privacy/mask-probe"],
        },
    )
    headers = _bearer(contract_token)
    assert api_client.post("/v1/privacy/mask-probe", headers=headers).status_code == 200

    denied = api_client.post("/v1/privacy/dp-probe", headers=headers)
    assert denied.status_code == 403
    assert denied.json()["code"] == "CONTRACT_VIOLATION"

    versioned = api_client.post(
        "/v1/privacy/mask-probe",
        headers={**headers, "X-PrivShield-Contract-Version": "v2"},
    )
    assert versioned.status_code == 403
    assert "version mismatch" in versioned.json()["message"]

    audience_token = _user_key(
        auth_runtime,
        name="wrong-audience",
        scopes=["privacy:mask"],
        audiences=["other-service"],
    )
    audience_denied = api_client.post(
        "/v1/privacy/mask-probe", headers=_bearer(audience_token)
    )
    assert audience_denied.status_code == 403
    assert audience_denied.json()["code"] == "AUDIENCE_MISMATCH"


def test_password_sandbox_precedes_mfa_sandbox(
    api_client: TestClient, auth_runtime: Any
) -> None:
    change_initial_password(api_client)
    service: AuthService = auth_runtime.service

    # 重置口令 → 首登沙箱；业务路径阻断，豁免路径放行。
    auth_runtime.store.reset_password(ADMIN, new_password="Adm1n-Passw0rd-3!")
    session = login(api_client, password="Adm1n-Passw0rd-3!")
    identity = service.resolve_token(session["token"])
    assert identity is not None

    with pytest.raises(AuthError) as excinfo:
        service.enforce_sandboxes(identity, "/v1/privacy/mask")
    assert excinfo.value.code == "PASSWORD_CHANGE_REQUIRED"

    service.enforce_sandboxes(identity, "/v1/auth/change-password")
    service.enforce_sandboxes(identity, f"/v1/auth/users/{ADMIN}")

    # API Key 身份不做 MFA 强制（仅会话受 MFA 沙箱约束）；先解除首登标记。
    auth_runtime.store.users[ADMIN].must_change_password = False
    api_key_identity = Identity(
        "internal", "rootadmin:key", ["*"], subject=ADMIN, session_auth=False
    )
    service.enforce_sandboxes(api_key_identity, "/v1/privacy/mask")


def test_mfa_sandbox_distinguishes_enrollment_and_session(
    api_client: TestClient, auth_runtime: Any
) -> None:
    change_initial_password(api_client)
    session = login(api_client)
    identity = auth_runtime.service.resolve_token(session["token"])
    assert identity is not None

    with pytest.raises(AuthError) as excinfo:
        auth_runtime.service.enforce_sandboxes(identity, "/v1/privacy/mask")
    assert excinfo.value.code == "MFA_ENROLLMENT_REQUIRED"

    # 已绑定但会话未过 TOTP → MFA_SESSION_REQUIRED。
    from engine.auth.store import hash_token
    from engine.auth.totp import generate_code

    secret, _, _ = auth_runtime.store.setup_totp(ADMIN)
    auth_runtime.store.enable_totp(ADMIN, generate_code(secret), hash_token(secret))

    with pytest.raises(AuthError) as excinfo:
        auth_runtime.service.enforce_sandboxes(identity, "/v1/privacy/mask")
    assert excinfo.value.code == "MFA_SESSION_REQUIRED"
