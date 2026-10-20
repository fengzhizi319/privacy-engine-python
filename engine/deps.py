"""共享依赖模块。

集中管理 REST 应用的跨路由共享对象：
    - 配置常量（profile 路径、命名空间、上传大小上限）；
    - ``PrivacyService`` 单例与按命名空间的服务缓存；
    - 通用安全依赖列表 ``SECURITY_DEPS``（认证 + 限速）；
    - 统一异常映射 ``handle_request_exception``。

把这些对象从 ``main.py`` 抽离到独立模块，使各 ``routers/*`` 子路由可按域
导入所需依赖，``main.py`` 仅负责应用装配（中间件、metrics、挂载路由）。
"""

import os

from fastapi import Depends, HTTPException

from .observability.logging_config import get_logger
from .privacy.budget import PrivacyBudgetExhausted
from .security.auth import get_current_identity
from .security.ratelimit import rate_limit_dependency
from .service import PrivacyService

# 从环境变量读取配置文件路径与命名空间，便于在不同部署环境间切换
PROFILE_PATH = os.environ.get("PRIVACY_PROFILE", "privacy-profile.yaml")
NAMESPACE = os.environ.get("PRIVACY_NAMESPACE", "default")

# 上传文件大小上限（字节），默认 50MB（与 Go ParseMultipartForm 同口径）；
# 超限返回 413，避免大文件耗尽内存（DoS 防护）。
MAX_UPLOAD_BYTES = int(os.environ.get("PRIVACY_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

# 模块级单例服务，供各路由处理函数复用
service = PrivacyService(profile_path=PROFILE_PATH, namespace=NAMESPACE)

# Per-namespace PrivacyService cache to avoid re-initializing on every recommend request
_service_cache: dict[str, PrivacyService] = {NAMESPACE: service}

# Module-level logger for unexpected server errors
logger = get_logger(__name__)

# 通用安全依赖：认证 + 限速。健康检查端点单独声明，不使用本依赖列表。
SECURITY_DEPS = [Depends(get_current_identity), Depends(rate_limit_dependency)]


def handle_request_exception(exc: Exception) -> None:
    """将隐私计算异常映射到合适的 HTTP 状态码，避免服务器错误误报为 400。

    Known exception types return a fixed detail string to avoid leaking internal
    implementation details (file paths, SQL errors, etc.) to callers.
    """
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, PrivacyBudgetExhausted):
        raise HTTPException(status_code=429, detail="Privacy budget exhausted") from exc
    if isinstance(exc, ValueError):
        logger.debug("ValueError in request handler", extra={"error": str(exc)})
        raise HTTPException(status_code=400, detail="Invalid request parameters") from exc
    logger.exception("unexpected_request_error")
    raise HTTPException(status_code=500, detail="Internal server error") from exc
