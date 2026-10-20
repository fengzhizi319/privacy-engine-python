"""Raw SPKI DER public-key binding helpers mirroring Go ``pkg/auth/bound_pubkey.go``.

绑定值保存的是**公钥本身**（base64 SPKI DER），不是摘要：比对退化为一次
定长字节比较，没有算法维度需要协商。Python 侧用 ``cryptography`` 解析
RSA/ECDSA/Ed25519，SM2 则按 OID + DER 结构校验（OpenSSL 默认不认 SM2）。
"""

from __future__ import annotations

import base64
import hashlib
import re

from cryptography.hazmat.primitives.serialization import load_der_public_key

SM2_OID = bytes.fromhex("06082a811ccf5501822d")  # 1.2.156.10197.1.301
_PEM_NOISE = re.compile(r"-----BEGIN PUBLIC KEY-----|-----END PUBLIC KEY-----|[\r\n ]")


def _read_tlv(data: bytes, offset: int) -> tuple[int, bytes, int] | None:
    if offset + 2 > len(data):
        return None
    tag = data[offset]
    length = data[offset + 1]
    cursor = offset + 2
    if length & 0x80:
        octets = length & 0x7F
        if octets == 0 or cursor + octets > len(data):
            return None
        length = int.from_bytes(data[cursor : cursor + octets], "big")
        cursor += octets
    end = cursor + length
    if end > len(data):
        return None
    return tag, data[cursor:end], end


def _looks_like_spki(der: bytes) -> bool:
    outer = _read_tlv(der, 0)
    if outer is None or outer[0] != 0x30 or outer[2] != len(der):
        return False
    algorithm = _read_tlv(outer[1], 0)
    if algorithm is None or algorithm[0] != 0x30:
        return False
    bit_string = _read_tlv(outer[1], algorithm[2])
    return bit_string is not None and bit_string[0] == 0x03 and bit_string[2] == len(outer[1])


def parse_bound_public_key_der(bound: str) -> bytes:
    """接受带/不带 padding 的标准或 URL-safe base64，返回原始 SPKI DER。"""
    value = _PEM_NOISE.sub("", bound or "").strip()
    if not value:
        raise ValueError("bound public key is empty")
    candidates = [value, value + "=" * (-len(value) % 4)]
    for candidate in candidates:
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                der = decoder(candidate, validate=True)
            except Exception:  # noqa: BLE001 - try the next encoding
                continue
            if not der or not _looks_like_spki(der):
                continue
            if SM2_OID in der:
                return der
            try:
                load_der_public_key(der)
            except Exception:  # noqa: BLE001
                continue
            return der
    raise ValueError("bound public key is not a valid base64-encoded SPKI DER public key")


def format_bound_public_key_der(der: bytes) -> str:
    return base64.b64encode(der).decode("ascii")


def public_key_digest(spki: str) -> str:
    """视图层摘要：解析失败返回 ``invalid``，与 Go ``publicKeyDigest`` 同口径。"""
    if not spki:
        return ""
    try:
        der = parse_bound_public_key_der(spki)
    except ValueError:
        return "invalid"
    return "sha256:" + hashlib.sha256(der).hexdigest()
