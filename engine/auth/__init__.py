"""User / MFA / credential kernel mirroring Go ``pkg/auth``.

The REST router is intentionally not re-exported here: it depends on the shared
security layer, which in turn resolves identities through this package's
runtime. Import :mod:`engine.auth.routes` explicitly from the app assembly.
"""

from .config import AuthSettings, is_production_labeled
from .errors import AuthError
from .models import APIKeyRecord, EgressQuota, User
from .runtime import AuthRuntime, build_runtime, get_auth_runtime, reset_auth_runtime_for_tests
from .service import AuthService
from .store import UserStore

__all__ = [
    "APIKeyRecord",
    "AuthError",
    "AuthRuntime",
    "AuthService",
    "AuthSettings",
    "EgressQuota",
    "User",
    "UserStore",
    "build_runtime",
    "get_auth_runtime",
    "is_production_labeled",
    "reset_auth_runtime_for_tests",
]
