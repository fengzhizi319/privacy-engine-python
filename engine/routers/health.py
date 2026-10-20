"""健康检查与探针路由（health / livez / readyz）。"""

import os

from fastapi import APIRouter, Depends, HTTPException

from ..deps import NAMESPACE, service
from ..security.auth import get_current_identity
from ..security.config import get_security_settings, metrics_auth_mode
from ..security.ratelimit import rate_limit_dependency

router = APIRouter()

# 健康检查类端点单独声明认证 + 限速依赖（不使用含权限校验的 SECURITY_DEPS）。
_HEALTH_DEPS = [Depends(get_current_identity), Depends(rate_limit_dependency)]


@router.get("/health", dependencies=_HEALTH_DEPS)
@router.post("/health", dependencies=_HEALTH_DEPS)
@router.get("/v1/privacy/health", dependencies=_HEALTH_DEPS)
@router.post("/v1/privacy/health", dependencies=_HEALTH_DEPS)
def health():
    """健康检查接口。

    Returns:
        包含状态与当前命名空间的 JSON 字典。
    """
    return {"status": "ok", "engine": "python", "namespace": NAMESPACE}


@router.get("/livez", dependencies=_HEALTH_DEPS)
def livez():
    """存活探针接口。"""
    return {"status": "ok"}


@router.get("/readyz", dependencies=_HEALTH_DEPS)
def readyz():
    """就绪探针接口。

    验证关键依赖（配置与隐私预算持久化存储）是否可访问。
    """
    if not service.resolver:
        raise HTTPException(status_code=503, detail="Service not ready")

    db_path = os.environ.get("PRIVACY_BUDGET_DB")
    if db_path:
        import sqlite3
        try:
            conn = sqlite3.connect(db_path, timeout=2.0)
            conn.execute("SELECT 1")
            conn.close()
        except sqlite3.Error as e:
            # Use a fixed message to avoid leaking internal file paths to
            # unauthenticated callers (health endpoints are auth-exempt by default).
            import logging
            logging.getLogger(__name__).warning(
                "readyz database check failed", extra={"error": str(e)}
            )
            raise HTTPException(
                status_code=503, detail="Database connectivity check failed"
            ) from e

    # 如实报告实际生效的传输栈：Python 进程内只可能终止标准 TLS 或明文，
    # 不回显 PRIVACY_TLS_STACK 配置值（Go 默认 tlcp，Python 不实现国密）。
    settings = get_security_settings()
    mode, strategies = metrics_auth_mode(settings)
    return {
        "status": "ok",
        "tls_stack": "tls" if settings.tls_enabled else "none",
        "metrics_auth_mode": mode,
        "metrics_auth_strategies": strategies,
    }


@router.get("/readyz/llm", dependencies=_HEALTH_DEPS)
def readyz_llm():
    """LLM 分类器就绪探针接口。"""
    try:
        from ..dynclassification.llm_adapter import LlmAdapter

        adapter = LlmAdapter()
        if adapter.is_available:
            return {"status": "ready", "llm": "available"}
    except Exception:
        pass
    return {"status": "ready", "llm": "unavailable"}

