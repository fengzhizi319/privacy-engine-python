"""Password hashing and strength policy.

主校验口径与 Go ``pkg/auth/password_sm3.go`` 一致：
``sm3:v2:<salt-hex>:<iterations>:<digest-hex>``，digest = SM3^N(salt‖password)。
bcrypt(cost=12) 与 SM3 哈希并行存储，用于口令历史检查。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Final

import bcrypt

SM3_V1_PREFIX: Final = "sm3:v1:"
SM3_V2_PREFIX: Final = "sm3:v2:"
SM3_SALT_SIZE: Final = 16
DEFAULT_SM3_ITERATIONS: Final = 1_000_000
DEFAULT_BCRYPT_COST: Final = 12

MIN_PASSWORD_LENGTH: Final = 8
MAX_PASSWORD_LENGTH: Final = 72
PASSWORD_HISTORY_DEPTH: Final = 3
MAX_PASSWORD_AGE_DAYS: Final = 90

WEAK_PASSWORDS: Final = frozenset(
    {
        "password", "12345678", "123456789", "1234567890", "qwerty123456", "admin123456",
        "iloveyou1234", "abc12345678", "password1234", "letmein12345", "welcome12345",
        "monkey123456", "dragon123456", "master123456", "123456789012", "0987654321ab",
        "password12345", "adminadmin12", "rootadmin123", "administrator", "guest123456",
        "operator123", "default12345", "pass12345678", "system123456", "security123",
        "privshield123", "database123", "oracle123456", "service1234", "changeme1234",
        "trustno11234", "superman1234", "sunshine1234", "princess1234", "football1234",
        "baseball1234", "shadow123456", "michael12345", "jordan123456", "harley123456",
        "ranger123456", "freedom12345", "computer1234", "starwars1234", "trustme12345",
        "hunter212345", "buster123456", "silver123456", "ginger123456",
    }
)

_KEYBOARD_WALKS: Final = (
    "qwertyuiop", "asdfghjkl", "zxcvbnm",
    "poiuytrewq", "lkjhgfdsa", "mnbvcxz",
    "1qaz2wsx3edc", "cde3xsw2zaq1",
)


def _sm3(data: bytes) -> bytes:
    return hashlib.new("sm3", data).digest()


def iterate_sm3(salt: bytes, password: str, iterations: int) -> bytes:
    """SM3 迭代拉伸：h0 = SM3(salt‖password)，h_i = SM3(h_{i-1})。"""
    digest = _sm3(salt + password.encode("utf-8"))
    for _ in range(1, iterations):
        digest = _sm3(digest)
    return digest


def hash_password_sm3(password: str, *, iterations: int = DEFAULT_SM3_ITERATIONS) -> str:
    salt = secrets.token_bytes(SM3_SALT_SIZE)
    digest = iterate_sm3(salt, password, iterations)
    return f"{SM3_V2_PREFIX}{salt.hex()}:{iterations}:{digest.hex()}"


def verify_password_sm3(encoded: str, password: str) -> bool:
    """常量时间校验 SM3 v2 编码；v1 与损坏编码一律拒绝。"""
    if not encoded.startswith(SM3_V2_PREFIX):
        return False
    rest = encoded[len(SM3_V2_PREFIX):]
    parts = rest.split(":")
    if len(parts) != 3:
        return False
    salt_hex, iters_raw, digest_hex = parts
    try:
        salt = bytes.fromhex(salt_hex)
        iterations = int(iters_raw)
    except ValueError:
        return False
    if len(salt) != SM3_SALT_SIZE or iterations <= 0:
        return False
    expected = iterate_sm3(salt, password, iterations).hex()
    return hmac.compare_digest(expected, digest_hex)


def hash_password_bcrypt(password: str, *, cost: int = DEFAULT_BCRYPT_COST) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=cost)).decode("ascii")


def verify_password_bcrypt(encoded: str, password: str) -> bool:
    if not encoded:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), encoded.encode("ascii"))
    except ValueError:
        return False


def _has_repeated_chars(value: str, minimum: int = 4) -> bool:
    lowered = value.lower()
    count = 1
    for previous, current in zip(lowered, lowered[1:]):
        count = count + 1 if current == previous else 1
        if count >= minimum:
            return True
    return False


def _has_sequential_chars(value: str, minimum: int = 4) -> bool:
    lowered = value.lower()
    asc = desc = 1
    for previous, current in zip(lowered, lowered[1:]):
        alnum = current.isascii() and current.isalnum()
        prev_alnum = previous.isascii() and previous.isalnum()
        asc = asc + 1 if alnum and prev_alnum and ord(current) == ord(previous) + 1 else 1
        desc = desc + 1 if alnum and prev_alnum and ord(current) == ord(previous) - 1 else 1
        if asc >= minimum or desc >= minimum:
            return True
    return False


def _has_keyboard_walk(value: str, minimum: int = 4) -> bool:
    lowered = value.lower()
    for walk in _KEYBOARD_WALKS:
        for index in range(len(walk) - minimum + 1):
            if walk[index:index + minimum] in lowered:
                return True
    return False


def validate_password_strength(password: str, username: str) -> None:
    """等保三级 G-04 口令复杂度校验；失败抛出 ValueError(message)。"""
    from .errors import invalid_argument

    encoded_len = len(password.encode("utf-8"))
    if encoded_len < MIN_PASSWORD_LENGTH:
        raise invalid_argument("password must be at least 8 characters")
    if encoded_len > MAX_PASSWORD_LENGTH:
        raise invalid_argument("password must not exceed 72 bytes (bcrypt input limit)")

    classes = sum(
        (
            any("a" <= char <= "z" for char in password),
            any("A" <= char <= "Z" for char in password),
            any("0" <= char <= "9" for char in password),
            any(not ("a" <= char <= "z" or "A" <= char <= "Z" or "0" <= char <= "9") for char in password),
        )
    )
    if classes < 3:
        raise invalid_argument(
            "password must contain at least 3 classes of characters "
            "(uppercase, lowercase, digits, special characters)"
        )

    name = username.strip().lower()
    lowered = password.lower()
    if name and (name in lowered or name[::-1] in lowered):
        raise invalid_argument("password must not contain the username (in either order or case)")
    if any(lowered == weak or lowered.startswith(weak) for weak in WEAK_PASSWORDS):
        raise invalid_argument("password appears in the common weak-password dictionary")
    if _has_repeated_chars(password):
        raise invalid_argument("password must not contain 4 or more repeated characters")
    if _has_sequential_chars(password):
        raise invalid_argument("password must not contain sequential characters (e.g. 1234, abcd)")
    if _has_keyboard_walk(password):
        raise invalid_argument("password must not contain keyboard walk patterns (e.g. qwerty)")
