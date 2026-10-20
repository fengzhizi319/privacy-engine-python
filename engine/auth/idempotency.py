"""Process-local idempotency store mirroring Go ``MemoryIdempotencyStore``.

工作负载密钥签发与轮换会一次性下发明文凭据，重放同一 ``X-Idempotency-Key``
必须返回同一次操作的结果（且不得再次泄露 token）。本实现按「租户 + 主体 +
路径 + 请求体摘要」四元组绑定，冲突或不一致一律 fail-closed。
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

from .errors import AuthError

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMMITTED = "committed"

DEFAULT_CAPACITY = 4096
DEFAULT_TTL_SECONDS = 24 * 3600


def request_hash(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body or b"").hexdigest()


@dataclass
class IdempotencyRecord:
    tenant_id: str
    key: str
    subject: str
    path: str
    req_hash: str
    status: str = STATUS_IN_PROGRESS
    response_code: int = 0
    response_body: bytes = b""
    created_at: float = field(default_factory=time.monotonic)
    expires_at: float = 0.0


class MemoryIdempotencyStore:
    """Thread-safe, capacity-bounded idempotency cache with 24h retention."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._capacity = max(1, capacity)
        self._ttl = max(1, ttl_seconds)
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], IdempotencyRecord] = {}

    def begin(
        self,
        tenant_id: str,
        key: str,
        subject: str,
        path: str,
        req_hash: str,
    ) -> tuple[IdempotencyRecord, bool]:
        """Reserve ``key`` or return the committed record.

        Returns ``(record, replayed)``. Raises :class:`AuthError` on conflicts,
        concurrent in-progress operations, or capacity exhaustion.
        """
        namespace = (tenant_id or "default", key)
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            existing = self._entries.get(namespace)
            if existing is not None:
                if (
                    existing.subject != (subject or "")
                    or existing.path != path
                    or existing.req_hash != req_hash
                ):
                    raise AuthError(
                        409,
                        "IDEMPOTENCY_KEY_MISMATCH",
                        "idempotency key conflict with mismatched request parameters",
                    )
                if existing.status == STATUS_IN_PROGRESS:
                    raise AuthError(
                        409,
                        "IDEMPOTENCY_IN_PROGRESS",
                        "operation with this idempotency key is currently in progress",
                    )
                if not existing.response_body:
                    raise AuthError(
                        409,
                        "IDEMPOTENCY_RESPONSE_UNAVAILABLE",
                        "operation already completed; cached response is unavailable; "
                        "inspect the resource instead of repeating the operation",
                    )
                return existing, True
            if len(self._entries) >= self._capacity:
                raise AuthError(
                    503,
                    "IDEMPOTENCY_CAPACITY_EXCEEDED",
                    "idempotency store is at capacity; retry later",
                    headers={"Retry-After": "1"},
                )
            record = IdempotencyRecord(
                tenant_id=tenant_id or "default",
                key=key,
                subject=subject or "",
                path=path,
                req_hash=req_hash,
                expires_at=now + self._ttl,
            )
            self._entries[namespace] = record
            return record, False

    def commit(self, record: IdempotencyRecord | None, status_code: int, body: bytes) -> None:
        if record is None:
            return
        with self._lock:
            record.status = STATUS_COMMITTED
            record.response_code = status_code
            record.response_body = body
            record.expires_at = time.monotonic() + self._ttl

    def abort(self, record: IdempotencyRecord | None) -> None:
        if record is None:
            return
        with self._lock:
            namespace = (record.tenant_id, record.key)
            existing = self._entries.get(namespace)
            if existing is record:
                self._entries.pop(namespace, None)

    def _prune_locked(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if now >= entry.expires_at]
        for key in expired:
            self._entries.pop(key, None)


_SHARED = MemoryIdempotencyStore()


def get_idempotency_store() -> MemoryIdempotencyStore:
    return _SHARED


def reset_idempotency_store_for_tests() -> None:
    global _SHARED
    _SHARED = MemoryIdempotencyStore()
