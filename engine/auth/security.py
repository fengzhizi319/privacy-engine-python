"""Small SM3 helpers shared by the auth kernel (no engine.privacy import to avoid cycles)."""

from __future__ import annotations

import hashlib


def sm3_hex(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.new("sm3", data).hexdigest()
