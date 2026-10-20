"""Per-key call contract snapshots (mirrors Go ``pkg/auth/key_contract.go``)."""

from __future__ import annotations

import posixpath
import re
from typing import Any

from .errors import AuthError

_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MAX_RULES = 128


def _invalid(message: str) -> AuthError:
    return AuthError(400, "INVALID_CONTRACT", f"invalid key contract: {message}")


def parse_contract(raw: Any) -> dict[str, Any] | None:
    """Validate and normalize an inbound ``contract`` object.

    ``None``/absent preserves existing centrally registered contracts; a present
    object must carry a non-empty identifier version and 1..128 ``METHOD /path``
    rules. Ambiguous, duplicate or traversal-bearing paths are rejected.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _invalid("contract must be an object")
    version = raw.get("version", "")
    if not isinstance(version, str) or not version.strip() or not _VERSION_RE.fullmatch(version):
        raise _invalid("contract.version must be a nonempty identifier")
    rules = raw.get("allowed_paths", [])
    if not isinstance(rules, list) or not 1 <= len(rules) <= _MAX_RULES:
        raise _invalid(f"contract requires 1..{_MAX_RULES} method/path rules")
    normalized: list[str] = []
    seen: set[str] = set()
    for rule in rules:
        if not isinstance(rule, str):
            raise _invalid("contract rule requires METHOD /path")
        parts = rule.split(" ")
        if len(parts) != 2 or parts[0] not in _METHODS:
            raise _invalid("contract rule requires METHOD /path")
        path = parts[1]
        plain = path[:-2] if path.endswith("/*") else path
        if (
            not plain.startswith("/")
            or plain.startswith("//")
            or "//" in plain
            or any(char in plain for char in "*?#%\\ \t\r\n")
            or posixpath.normpath(plain) != plain
        ):
            raise _invalid(f'invalid or duplicate contract path "{rule}"')
        if rule in seen:
            raise _invalid(f'invalid or duplicate contract path "{rule}"')
        seen.add(rule)
        normalized.append(rule)
    return {"version": version, "allowed_paths": normalized}


def contract_allows(allowed_paths: list[str], method: str, path: str) -> bool:
    """Return True when ``method path`` matches the contract allow-list."""
    if not allowed_paths:
        return True
    try:
        clean = posixpath.normpath(path or "/")
    except (TypeError, ValueError):
        clean = path or "/"
    if not clean.startswith("/"):
        clean = "/" + clean
    method = (method or "").strip().upper()
    for pattern in allowed_paths:
        pattern = pattern.strip()
        if not pattern:
            continue
        want_method = ""
        if " " in pattern:
            want_method, pattern = pattern.split(" ", 1)
            want_method = want_method.strip().upper()
            pattern = pattern.strip()
        if want_method and want_method != "*" and want_method != method:
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


def contract_denial() -> AuthError:
    return AuthError(
        403,
        "CONTRACT_VIOLATION",
        "Forbidden: request path is not in the credential's contract allow-list",
    )
