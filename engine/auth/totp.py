"""RFC 6238 TOTP with SHA1 (default) and HMAC-SM3 extensions.

与 Go ``pkg/auth/mfa.go`` 对齐：6 位、30 秒、±1 步窗口、严格单调时间步防重放。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import struct
import time
import urllib.parse

TOTP_PERIOD = 30
TOTP_DIGITS = 6
TOTP_ALGO_SHA1 = "SHA1"
TOTP_ALGO_SM3 = "SM3"
SUPPORTED_TOTP_ALGORITHMS = (TOTP_ALGO_SM3, TOTP_ALGO_SHA1)


def default_totp_algorithm() -> str:
    """读取 ``AUTH_TOTP_ALGORITHM``；非法值回落到 SHA1（与 Go 一致）。"""
    raw = os.environ.get("AUTH_TOTP_ALGORITHM", "").strip().upper()
    if raw == TOTP_ALGO_SHA1:
        return TOTP_ALGO_SHA1
    if raw == TOTP_ALGO_SM3:
        return TOTP_ALGO_SM3
    return TOTP_ALGO_SHA1


def _digestmod(algo: str):
    if algo == TOTP_ALGO_SHA1:
        return hashlib.sha1
    if algo == TOTP_ALGO_SM3:
        return lambda data=b"": hashlib.new("sm3", data)
    raise ValueError(f"unsupported TOTP algorithm {algo!r}")


def generate_secret() -> str:
    """32 字节 CSPRNG 密钥（Base32 无填充，与 Go GenerateTOTPSecret 一致）。"""
    return base64.b32encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def _decode_secret(secret: str) -> bytes:
    normalized = secret.strip().upper()
    if not normalized:
        raise ValueError("auth: mfa secret cannot be empty")
    padding = "=" * ((8 - len(normalized) % 8) % 8)
    return base64.b32decode(normalized + padding)


def generate_code(secret: str, at: float | None = None, algo: str | None = None) -> str:
    algorithm = (algo or default_totp_algorithm()).upper()
    key = _decode_secret(secret)
    counter = int((time.time() if at is None else at) // TOTP_PERIOD)
    message = struct.pack(">Q", counter)
    digest = hmac.new(key, message, _digestmod(algorithm)).digest()
    offset = digest[-1] & 0x0F
    binary = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return f"{binary % 1_000_000:06d}"


def validate_code(
    secret: str,
    code: str,
    at: float | None = None,
    last_used_step: int = 0,
    algo: str | None = None,
) -> tuple[bool, int]:
    """校验 ±1 窗口并拒绝已消费/倒退的时间步，返回 (valid, step)。"""
    candidate = code.strip()
    if len(candidate) != TOTP_DIGITS:
        return False, 0
    algorithm = (algo or default_totp_algorithm()).strip().upper()
    if algorithm not in SUPPORTED_TOTP_ALGORITHMS:
        return False, 0
    now = time.time() if at is None else at
    current_step = int(now // TOTP_PERIOD)
    for step in (current_step - 1, current_step, current_step + 1):
        if step <= last_used_step:
            continue
        expected = generate_code(secret, step * TOTP_PERIOD, algorithm)
        if hmac.compare_digest(candidate, expected):
            return True, step
    return False, 0


def generate_uri(username: str, issuer: str, secret: str, algo: str | None = None) -> str:
    algorithm = (algo or default_totp_algorithm()).upper()
    label = f"{issuer or 'PrivShield'}:{username}"
    query = urllib.parse.urlencode(
        {
            "secret": secret.strip().upper(),
            "issuer": issuer or "PrivShield",
            "algorithm": algorithm,
            "digits": str(TOTP_DIGITS),
            "period": str(TOTP_PERIOD),
        }
    )
    return f"otpauth://totp/{urllib.parse.quote(label, safe='')}?{query}"
