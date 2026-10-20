"""Proof-of-Possession 判定 / PoP decision.

与 Go ``pkg/auth/pop_decision.go`` 同一套绑定状态机：

1. 健康探针与公开认证路径豁免（否则绑定凭据会打死探针）；
2. 凭据声明任一绑定（CN / 证书指纹 / 公钥）→ 必须出示证书且全部匹配；
3. 未声明绑定的 workload 主体一律拒绝（机器凭据必须绑定）；
4. 其余主体（登录会话、测试注入身份）放行。

Python 侧的对端证书视图来自 Go 终结器注入并签名的身份头，因此绑定核验
没有「算法协商」维度：`sha256:` 指纹直接与签名头中的证书摘要比对，
公钥绑定则比对原始 SPKI DER。
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass

from ..auth.bound_pubkey import parse_bound_public_key_der
from .identity import Identity
from .sm3 import new_sm3
from .terminator import TerminatorPeer

_DIGEST_PREFIX = re.compile(r"^(sha256|sm3):(.+)$", re.IGNORECASE)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PoPOutcome:
    """PoP 判定结果及对应的 REST/gRPC 拒绝形态。"""

    allowed: bool
    code: str = ""
    message: str = ""
    status: int = 0
    decision: str = "allow"

    @classmethod
    def allow(cls) -> PoPOutcome:
        return cls(allowed=True)

    @classmethod
    def deny(cls, decision: str, status: int, code: str, message: str) -> PoPOutcome:
        return cls(allowed=False, code=code, message=message, status=status, decision=decision)


def _normalized_digest(raw: str) -> tuple[str, str] | None:
    match = _DIGEST_PREFIX.match((raw or "").strip())
    if match is None:
        return None
    algo = match.group(1).lower()
    value = "".join(ch for ch in match.group(2).lower() if ch not in {":", " "})
    if not _HEX64.match(value):
        return None
    return algo, value


def _match_bound_cn(bound: str, peer: TerminatorPeer) -> bool:
    if not bound:
        return True
    target = bound.strip().lower()
    return bool(target) and peer.common_name.strip().lower() == target


def _match_bound_fingerprint(bound: str, peer: TerminatorPeer) -> tuple[bool, bool]:
    """Return ``(matched, malformed)``.

    Only ``sha256:`` can be evaluated from the signed terminator view: the
    loopback headers carry the leaf SHA-256 digest, not the certificate DER.
    Python 侧当前没有任何路径写入 ``bound_cert_fingerprint``；无法评估的
    绑定按「配置写坏」拒绝，绝不静默跳过。
    """
    parsed = _normalized_digest(bound)
    if parsed is None:
        return False, True
    algo, want = parsed
    if algo != "sha256":
        return False, True
    return hmac.compare_digest(want, peer.cert_sha256.strip().lower()), False


def _match_bound_public_key(bound: str, peer: TerminatorPeer) -> tuple[bool, bool]:
    try:
        want = parse_bound_public_key_der(bound)
    except ValueError:
        return False, True
    if not peer.spki_der:
        return False, False
    if len(want) != len(peer.spki_der):
        return False, False
    return hmac.compare_digest(want, peer.spki_der), False


def _deny_cert_required() -> PoPOutcome:
    return PoPOutcome.deny(
        "cert_required",
        401,
        "POP_TLS_CERT_REQUIRED",
        "Unauthorized: mTLS client certificate required for this credential",
    )


def decide_pop(
    identity: Identity | None,
    peer: TerminatorPeer | None,
    *,
    health_exempt: bool,
) -> PoPOutcome:
    """执行绑定判定（Go ``DecidePoP`` 的 Python 同构实现）。"""
    if identity is None:
        return PoPOutcome.allow()
    if health_exempt:
        return PoPOutcome.allow()

    if identity.has_credential_bindings():
        if peer is None:
            return _deny_cert_required()
        if not _match_bound_cn(identity.bound_cn, peer):
            return PoPOutcome.deny(
                "cn_mismatch",
                403,
                "CLIENT_CN_MISMATCH",
                "Forbidden: client certificate identity does not match bound credential",
            )
        if identity.bound_cert_fingerprint:
            matched, malformed = _match_bound_fingerprint(
                identity.bound_cert_fingerprint, peer
            )
            if malformed:
                return PoPOutcome.deny(
                    "binding_malformed",
                    403,
                    "CERT_BINDING_MALFORMED",
                    "Forbidden: bound certificate constraint is malformed",
                )
            if not matched:
                return PoPOutcome.deny(
                    "fingerprint_mismatch",
                    403,
                    "CERT_FINGERPRINT_PIN_MISMATCH",
                    "Forbidden: client certificate fingerprint is not pinned",
                )
        if identity.bound_public_key_der:
            matched, malformed = _match_bound_public_key(
                identity.bound_public_key_der, peer
            )
            if malformed:
                return PoPOutcome.deny(
                    "binding_malformed",
                    403,
                    "CERT_BINDING_MALFORMED",
                    "Forbidden: bound certificate constraint is malformed",
                )
            if not matched:
                return PoPOutcome.deny(
                    "public_key_mismatch",
                    403,
                    "POP_PUBLIC_KEY_MISMATCH",
                    "Forbidden: client public key does not match bound credential",
                )
        return PoPOutcome.allow()

    if identity.principal_type == "workload":
        return PoPOutcome.deny(
            "unbound_restricted",
            403,
            "UNBOUND_KEY_RESTRICTED",
            "Forbidden: unbound workload API key restricted on sensitive business endpoints",
        )
    return PoPOutcome.allow()


def sm3_hex(data: bytes) -> str:
    """Small helper for tests and future SM3 bindings."""
    return new_sm3(data).hexdigest()
