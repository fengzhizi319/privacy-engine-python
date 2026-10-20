"""Inbound credential keyfile for the gateway (Go ``InboundPolicy`` parity).

网关是凭据的**转交者**而非 scope 校验者：这里只回答「这把凭据是否为已
登记的调用方、是否仍有效、受众/契约是否允许它访问本网关」。权限（scope）
裁决仍由被代理的引擎按自身配置完成。

Keyfile 形态与 Go ``pkg/auth/KeyStore`` 一致（JSON version "1"）：

```json
{
  "version": "1",
  "credentials": [
    {
      "token": "psk_...",                  // 与 token_digest 二选一
      "token_digest": "sm3:<64hex>",
      "name": "svc-batch",
      "subject": "svc:batch",
      "scopes": ["privacy:mask"],
      "allowed_audiences": ["privacy-gateway"],
      "expires_at": "2027-01-01T00:00:00Z",
      "binding": {"pubkey": "<base64 SPKI DER>", "cn": "...",
                  "cert_fingerprint": "sm3:<hex>", "status": "active"},
      "contract": {"version": "v1", "allowed_paths": ["POST /v1/privacy/*"]}
    }
  ]
}
```

热重载：按 mtime 惰性检测；新文件解析失败时保留上一份有效快照（与 Go
``KeyStore`` 一致），坏文件不会把已生效的凭据清空。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from engine.observability.logging_config import get_logger
from engine.security.sm3 import new_sm3

logger = get_logger(__name__)

_SERVICE_NAME = "privacy-gateway"
_ALLOWED_TOP_LEVEL = {"version", "_comment", "credentials", "trusted_issuers"}
_ALLOWED_CREDENTIAL = {
    "token",
    "token_digest",
    "name",
    "subject",
    "scopes",
    "expires_at",
    "tenant_id",
    "allowed_audiences",
    "principal_type",
    "client_id",
    "mfa_authenticated",
    "_note",
    "binding",
    "contract",
    "quota",
}


@dataclass(frozen=True)
class InboundCredential:
    name: str
    subject: str = ""
    token: str = ""
    token_digest: str = ""
    scopes: tuple[str, ...] = ()
    allowed_audiences: tuple[str, ...] = ()
    expires_at: datetime | None = None
    binding_pubkey: str = ""
    binding_cn: str = ""
    binding_cert_fingerprint: str = ""
    binding_status: str = ""
    contract_version: str = ""
    contract_allowed_paths: tuple[str, ...] = field(default_factory=tuple)

    def matches_audience(self, audience: str) -> bool:
        target = audience.strip()
        if not target:
            return False
        return any(aud == "*" or aud == target for aud in self.allowed_audiences)


@dataclass(frozen=True)
class InboundDecision:
    ok: bool
    status: int = 0
    code: str = ""
    message: str = ""
    credential: InboundCredential | None = None


def _as_str_list(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be an array of strings")
    return tuple(item.strip() for item in value if item.strip())


def _parse_rfc3339(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _parse_credential(raw: object, index: int) -> InboundCredential:
    if not isinstance(raw, dict):
        raise ValueError(f"credential[{index}] must be an object")
    unknown = set(raw) - _ALLOWED_CREDENTIAL
    if unknown:
        raise ValueError(f"credential[{index}] has unknown fields: {sorted(unknown)}")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError(f"credential[{index}]: name is required")
    token = str(raw.get("token", "")).strip()
    token_digest = str(raw.get("token_digest", "")).strip()
    if bool(token) == bool(token_digest):
        raise ValueError(
            f"credential[{index}] {name!r}: exactly one of token or token_digest is required"
        )
    if token_digest:
        lowered = token_digest.lower()
        if not (lowered.startswith("sm3:") or lowered.startswith("sha256:")):
            raise ValueError(
                f"credential[{index}] {name!r}: token_digest must be 'sm3:<hex>' or 'sha256:<hex>'"
            )
        digest = lowered.split(":", 1)[1]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"credential[{index}] {name!r}: token_digest must be 64 hex chars")
    scopes = _as_str_list(raw.get("scopes"), "scopes")
    if not scopes:
        raise ValueError(f"credential[{index}] {name!r}: scopes must not be empty")
    audiences = _as_str_list(raw.get("allowed_audiences"), "allowed_audiences")
    if not audiences:
        raise ValueError(
            f"credential[{index}] {name!r}: allowed_audiences must not be empty "
            "(which service accepts this credential?)"
        )
    binding = raw.get("binding")
    if not isinstance(binding, dict):
        raise ValueError(
            f"credential[{index}] {name!r}: binding is required (machine credentials "
            "must be bound to the caller certificate public key)"
        )
    pubkey = str(binding.get("pubkey", "")).strip()
    if not pubkey:
        raise ValueError(f"credential[{index}] {name!r}: binding.pubkey is required")
    contract = raw.get("contract")
    contract_version = ""
    allowed_paths: tuple[str, ...] = ()
    if contract is not None:
        if not isinstance(contract, dict):
            raise ValueError(f"credential[{index}] {name!r}: contract must be an object")
        unknown = set(contract) - {"version", "allowed_paths"}
        if unknown:
            raise ValueError(
                f"credential[{index}] {name!r}: contract has unknown fields: {sorted(unknown)}"
            )
        contract_version = str(contract.get("version", "")).strip()
        allowed_paths = _as_str_list(contract.get("allowed_paths"), "contract.allowed_paths")

    expires_at = None
    raw_expiry = str(raw.get("expires_at", "")).strip()
    if raw_expiry:
        expires_at = _parse_rfc3339(raw_expiry)

    return InboundCredential(
        name=name,
        subject=str(raw.get("subject", "")).strip(),
        token=token,
        token_digest=token_digest.lower(),
        scopes=scopes,
        allowed_audiences=audiences,
        expires_at=expires_at,
        binding_pubkey=pubkey,
        binding_cn=str(binding.get("cn", "")).strip(),
        binding_cert_fingerprint=str(binding.get("cert_fingerprint", "")).strip().lower(),
        binding_status=str(binding.get("status", "")).strip(),
        contract_version=contract_version,
        contract_allowed_paths=allowed_paths,
    )


def parse_keyfile(raw: str) -> tuple[InboundCredential, ...]:
    """Parse and strictly validate a keyfile; any bad entry rejects the whole file."""
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"parse credential file JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("credential file must be a JSON object")
    unknown = set(document) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise ValueError(f"credential file has unknown fields: {sorted(unknown)}")
    if document.get("version") != "1":
        raise ValueError(
            f"unsupported credential file version {document.get('version')!r} (want \"1\")"
        )
    credentials_raw = document.get("credentials")
    if not isinstance(credentials_raw, list) or not credentials_raw:
        raise ValueError("credential file contains no credentials")
    credentials = tuple(
        _parse_credential(item, index) for index, item in enumerate(credentials_raw)
    )
    tokens = [c.token or c.token_digest for c in credentials]
    if len(set(tokens)) != len(tokens):
        raise ValueError("credential file contains duplicate token entries")
    return credentials


def _hash_token(kind: str, token: str) -> str:
    data = token.encode("utf-8")
    if kind == "sm3":
        return new_sm3(data).hexdigest()
    return hashlib.sha256(data).hexdigest()


def _canonical_path(path: str) -> str:
    path = (path or "").split("?", 1)[0].strip()
    if not path.startswith("/"):
        path = "/" + path
    # Collapse duplicate slashes and dot segments (best-effort mirror of Go
    # CanonicalizeRequestPath; reject nothing here — auth still fails closed).
    trailing = path.endswith("/")
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    canonical = "/" + "/".join(parts)
    if trailing and canonical != "/":
        canonical += "/"
    return canonical


def contract_path_allowed(pattern_list: tuple[str, ...], method: str, path: str) -> bool:
    """Mirror Go ``IsContractPathAllowed`` (method prefix + trailing ``*`` wildcard)."""
    if not pattern_list:
        return True
    clean = _canonical_path(path)
    method = (method or "").strip().upper()
    for raw in pattern_list:
        pattern = raw.strip()
        if not pattern:
            continue
        want_method = ""
        if " " in pattern:
            head, tail = pattern.split(" ", 1)
            want_method = head.strip().upper()
            pattern = tail.strip()
        if want_method and want_method not in ("*", method):
            continue
        if pattern in ("*", clean):
            return True
        if pattern.endswith("*"):
            base = pattern[:-1]
            if base.endswith("/"):
                if clean.startswith(base):
                    return True
            elif base.endswith("."):
                if clean == base or clean.startswith(base):
                    return True
            elif clean == base or clean.startswith(base + "/") or clean.startswith(base + "."):
                return True
    return False


class InboundKeyStore:
    """Hot-reloading keyfile store used by REST and gRPC admission."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._credentials: tuple[InboundCredential, ...] = ()
        self._mtime_ns = -1
        self._failed_mtime_ns = -1
        self._reload(force=True)

    def _reload(self, force: bool = False) -> None:
        try:
            stat = os.stat(self.path)
        except OSError as exc:
            raise ValueError(f"read inbound keyfile {self.path}: {exc}") from exc
        if not force and stat.st_mtime_ns == self._mtime_ns:
            return
        if stat.st_mtime_ns == self._failed_mtime_ns:
            return
        try:
            with open(self.path, encoding="utf-8") as handle:
                credentials = parse_keyfile(handle.read())
        except (OSError, ValueError) as exc:
            self._failed_mtime_ns = stat.st_mtime_ns
            if not self._credentials:
                raise
            logger.error(
                "Inbound keyfile reload failed; keeping last valid snapshot",
                extra={"path": self.path, "error": str(exc)},
            )
            return
        self._credentials = credentials
        self._mtime_ns = stat.st_mtime_ns
        self._failed_mtime_ns = -1
        logger.info(
            "Inbound keyfile loaded",
            extra={"path": self.path, "credentials": len(credentials)},
        )

    def _snapshot(self) -> tuple[InboundCredential, ...]:
        with self._lock:
            self._reload()
            return self._credentials

    def verify(
        self,
        authorization: str | None,
        method: str,
        path: str,
        contract_header: str = "",
        *,
        peer_spki_der: bytes | None = None,
        peer_cn: str = "",
        peer_cert_fingerprint: str = "",
        health_exempt: bool = False,
    ) -> InboundDecision:
        """Authenticate the caller and apply audience/contract/drain-only policy."""
        raw = (authorization or "").strip()
        scheme, _, token = raw.partition(" ")
        if not raw or scheme.lower() != "bearer" or not token.strip():
            return InboundDecision(
                ok=False, status=401, code="UNAUTHENTICATED",
                message="invalid or missing gateway credential",
            )
        token = token.strip()
        matched: InboundCredential | None = None
        for credential in self._snapshot():
            if credential.token:
                candidate = credential.token
                if hmac.compare_digest(candidate.encode(), token.encode()):
                    matched = credential
            else:
                kind, _, digest = credential.token_digest.partition(":")
                if hmac.compare_digest(_hash_token(kind, token), digest):
                    matched = credential
        if matched is None:
            return InboundDecision(
                ok=False, status=401, code="UNAUTHENTICATED",
                message="invalid gateway credential",
            )
        now = datetime.now(timezone.utc)
        if matched.expires_at is not None and now >= matched.expires_at:
            return InboundDecision(
                ok=False, status=401, code="UNAUTHENTICATED",
                message="invalid gateway credential",
            )
        if not matched.matches_audience(_SERVICE_NAME):
            return InboundDecision(
                ok=False, status=403, code="AUDIENCE_MISMATCH",
                message="Forbidden: credential audience mismatch, cannot access service "
                + _SERVICE_NAME,
            )
        if contract_header.strip() and contract_header.strip() != matched.contract_version:
            return InboundDecision(
                ok=False, status=403, code="CONTRACT_VIOLATION",
                message="Forbidden: contract version mismatch",
            )
        if matched.binding_status == "drain_only" and not health_exempt:
            return InboundDecision(
                ok=False, status=403, code="CREDENTIAL_DRAIN_ONLY",
                message="Forbidden: credential in drain-only mode; only health probes are permitted",
            )
        if not contract_path_allowed(
            matched.contract_allowed_paths, method, path
        ) and not health_exempt:
            return InboundDecision(
                ok=False, status=403, code="CONTRACT_VIOLATION",
                message="Forbidden: request path is not in the credential's contract allow-list",
            )
        if peer_spki_der is not None and matched.binding_pubkey:
            expected = _decode_spki(matched.binding_pubkey)
            if not hmac.compare_digest(expected, peer_spki_der):
                return InboundDecision(
                    ok=False, status=403, code="POP_REQUIRED",
                    message="Forbidden: caller certificate does not match the credential binding",
                )
        if peer_cn and matched.binding_cn and not hmac.compare_digest(
            peer_cn.encode(), matched.binding_cn.encode()
        ):
            return InboundDecision(
                ok=False, status=403, code="POP_REQUIRED",
                message="Forbidden: caller certificate CN does not match the credential binding",
            )
        if (
            peer_cert_fingerprint
            and matched.binding_cert_fingerprint
            and not hmac.compare_digest(
                peer_cert_fingerprint.lower().encode(),
                matched.binding_cert_fingerprint.encode(),
            )
        ):
            return InboundDecision(
                ok=False, status=403, code="POP_REQUIRED",
                message="Forbidden: caller certificate fingerprint does not match the binding",
            )
        return InboundDecision(ok=True, credential=matched)


def _decode_spki(value: str) -> bytes:
    text = value.strip()
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)
    except ValueError:
        return b""


def load_inbound_store(path: str) -> InboundKeyStore | None:
    """Create a store when configured; empty path means no inbound policy."""
    return InboundKeyStore(path) if path.strip() else None
