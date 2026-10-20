"""One-time MFA recovery codes.

码形态 XXXX-XXXX-XXXX（12 位，31 字符表，剔除易混淆 0/O/1/I/L），
落盘只存 SM3 摘要与 4 位前缀，明文仅签发响应出现一次。
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .security import sm3_hex

RECOVERY_CODE_COUNT = 8
RECOVERY_CODE_GROUP_LEN = 4
RECOVERY_CODE_GROUPS = 3
RECOVERY_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"


@dataclass
class RecoveryCodeEntry:
    digest: str
    prefix: str
    created_at: str
    used_at: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {"prefix": self.prefix, "created_at": self.created_at}
        if self.used_at:
            payload["used_at"] = self.used_at
        return payload


@dataclass
class RecoveryCodeSet:
    entries: list[RecoveryCodeEntry] = field(default_factory=list)

    def remaining(self) -> int:
        return sum(1 for entry in self.entries if not entry.used_at)


def normalize_recovery_code(raw: str) -> str:
    return "".join(char.upper() for char in raw if char not in "- \t")


def is_valid_recovery_code(normalized: str) -> bool:
    return len(normalized) == RECOVERY_CODE_GROUP_LEN * RECOVERY_CODE_GROUPS and all(
        char in RECOVERY_ALPHABET for char in normalized
    )


def hash_recovery_code(normalized: str) -> str:
    return sm3_hex(normalized)


def generate_recovery_codes() -> tuple[list[str], list[RecoveryCodeEntry]]:
    total = RECOVERY_CODE_GROUP_LEN * RECOVERY_CODE_GROUPS
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    codes: list[str] = []
    entries: list[RecoveryCodeEntry] = []
    seen: set[str] = set()
    while len(codes) < RECOVERY_CODE_COUNT:
        normalized = "".join(
            RECOVERY_ALPHABET[byte % len(RECOVERY_ALPHABET)]
            for byte in secrets.token_bytes(total)
        )
        if normalized in seen:
            continue
        seen.add(normalized)
        groups = [
            normalized[index:index + RECOVERY_CODE_GROUP_LEN]
            for index in range(0, total, RECOVERY_CODE_GROUP_LEN)
        ]
        codes.append("-".join(groups))
        entries.append(
            RecoveryCodeEntry(
                digest=hash_recovery_code(normalized),
                prefix=normalized[:RECOVERY_CODE_GROUP_LEN],
                created_at=now,
            )
        )
    return codes, entries
