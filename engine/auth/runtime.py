"""Process-wide auth runtime bootstrap.

用户库后端必须显式配置（``PRIVACY_AUTH_USER_STORE_BACKEND``），与 Go 侧
``OpenUserStoreFromEnv`` 同姿态：未配置即表示 IAM 未启用；配置为未知或
当前解释器不支持的档位则**拒绝启动**，绝不静默降级到内存库。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..observability.logging_config import get_logger
from .config import AuthSettings
from .persistence import SqliteRepository
from .service import AuthService
from .store import UserStore

logger = get_logger(__name__)


@dataclass
class AuthRuntime:
    settings: AuthSettings
    store: UserStore
    service: AuthService
    repository: object | None = None


_runtime: AuthRuntime | None = None
_runtime_lock = threading.Lock()


def build_runtime(settings: AuthSettings | None = None) -> AuthRuntime:
    """按配置装配内存 / SQLite 用户库；PostgreSQL 因缺少驱动显式失败。"""
    settings = settings or AuthSettings.from_env()
    if not settings.enabled:
        raise RuntimeError(
            "auth runtime requested without PRIVACY_AUTH_USER_STORE_BACKEND configured"
        )
    repository: object | None = None
    if settings.backend == "sqlite":
        repository = SqliteRepository(
            settings.sqlite_file(),
            encryption_key=settings.encryption_key,
        )
    elif settings.backend == "postgres":
        raise RuntimeError(
            "PRIVACY_AUTH_USER_STORE_BACKEND=postgres is not available in this build: "
            "the psycopg driver is not installed; use sqlite or install the postgres extra"
        )
    elif settings.backend != "memory":
        raise RuntimeError(f"unsupported auth user store backend: {settings.backend!r}")

    store = UserStore(settings, repository)
    service = AuthService(settings, store)
    logger.info(
        "auth runtime initialized",
        extra={
            "backend": settings.backend,
            "users": store.count(),
            "step_up_required": settings.admin_stepup_required,
        },
    )
    return AuthRuntime(settings=settings, store=store, service=service, repository=repository)


def get_auth_runtime() -> AuthRuntime | None:
    """返回进程级 IAM 运行时；未显式配置用户库后端时返回 ``None``。"""
    global _runtime
    settings = AuthSettings.from_env()
    if not settings.enabled:
        return None
    with _runtime_lock:
        if _runtime is None:
            _runtime = build_runtime(settings)
        return _runtime


def reset_auth_runtime_for_tests(runtime: AuthRuntime | None = None) -> None:
    """测试辅助：替换或清空进程级运行时。"""
    global _runtime
    with _runtime_lock:
        if _runtime is not None and _runtime.repository is not None:
            close = getattr(_runtime.repository, "close", None)
            if callable(close):
                close()
        _runtime = runtime
