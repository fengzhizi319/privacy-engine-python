"""SM3 国密散列算法 / SM3 cryptographic hash.

优先使用运行环境 OpenSSL 提供的 SM3 实现（``hashlib.new("sm3")``）；
缺少该算法时回退到本模块的纯 Python 实现，保证跨环境输出一致。
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

_MASK = 0xFFFFFFFF

_IV = (
    0x7380166F,
    0x4914B2B9,
    0x172442D7,
    0xDA8A0600,
    0xA96F30BC,
    0x163138AA,
    0xE38DEE4D,
    0xB0FB0E4E,
)

_T = (0x79CC4519,) * 16 + (0x7A879D8A,) * 48


def _rotl(value: int, count: int) -> int:
    count &= 31
    if count == 0:
        return value & _MASK
    return ((value << count) | (value >> (32 - count))) & _MASK


def _p0(value: int) -> int:
    return value ^ _rotl(value, 9) ^ _rotl(value, 17)


def _p1(value: int) -> int:
    return value ^ _rotl(value, 15) ^ _rotl(value, 23)


def sm3_digest(data: bytes) -> bytes:
    """Return the 32-byte SM3 digest of ``data``."""
    buffer = bytearray(data)
    bit_length = len(buffer) * 8
    buffer.append(0x80)
    while len(buffer) % 64 != 56:
        buffer.append(0)
    buffer.extend(bit_length.to_bytes(8, "big"))

    state = list(_IV)
    for offset in range(0, len(buffer), 64):
        words = list(struct.unpack(">16I", bytes(buffer[offset : offset + 64]))) + [0] * 52
        for index in range(16, 68):
            words[index] = (
                _p1(words[index - 16] ^ words[index - 9] ^ _rotl(words[index - 3], 15))
                ^ _rotl(words[index - 13], 7)
                ^ words[index - 6]
            )
        expanded = [words[index] ^ words[index + 4] for index in range(64)]

        a, b, c, d, e, f, g, h = state
        for index in range(64):
            a12 = _rotl(a, 12)
            ss1 = _rotl((a12 + e + _rotl(_T[index], index)) & _MASK, 7)
            ss2 = ss1 ^ a12
            if index < 16:
                ff = a ^ b ^ c
                gg = e ^ f ^ g
            else:
                ff = (a & b) | (a & c) | (b & c)
                gg = (e & f) | ((~e & _MASK) & g)
            tt1 = (ff + d + ss2 + expanded[index]) & _MASK
            tt2 = (gg + h + ss1 + words[index]) & _MASK
            d, c, b, a = c, _rotl(b, 9), a, tt1
            h, g, f, e = g, _rotl(f, 19), e, _p0(tt2)

        state = [old ^ new for old, new in zip(state, (a, b, c, d, e, f, g, h))]

    return struct.pack(">8I", *state)


class SM3Hash:
    """Minimal hashlib-compatible SM3 object used when OpenSSL lacks SM3."""

    name = "sm3"
    digest_size = 32
    block_size = 64

    def __init__(self, data: bytes = b"") -> None:
        self._buffer = bytearray(data)

    def update(self, data: Any) -> None:
        self._buffer.extend(bytes(data))

    def digest(self) -> bytes:
        return sm3_digest(bytes(self._buffer))

    def hexdigest(self) -> str:
        return self.digest().hex()

    def copy(self) -> SM3Hash:
        return SM3Hash(bytes(self._buffer))


def new_sm3(data: bytes = b"") -> Any:
    """Return an SM3 hash object, preferring the OpenSSL implementation."""
    try:
        return hashlib.new("sm3", data)
    except ValueError:
        return SM3Hash(data)
