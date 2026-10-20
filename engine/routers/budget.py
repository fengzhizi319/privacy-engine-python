"""隐私预算查询路由（budget）。"""

from fastapi import APIRouter

from ..deps import SECURITY_DEPS, service
from ..security.auth import require_permission

router = APIRouter()


@router.get("/v1/privacy/budget", dependencies=[*SECURITY_DEPS, require_permission("privacy:budget")])
def budget():
    """查询剩余隐私预算。

    Returns:
        当前命名空间下 epsilon 与 delta 的剩余量字典。
    """
    return service.budget_remaining()


@router.post("/v1/privacy/budget/reset", dependencies=[*SECURITY_DEPS, require_permission("privacy:budget")])
def reset_budget():
    """重置当前命名空间隐私预算。"""
    return service.budget_reset()


@router.delete("/v1/privacy/budget", dependencies=[*SECURITY_DEPS, require_permission("privacy:budget")])
def delete_budget():
    """重置当前命名空间隐私预算（DELETE 别名）。"""
    return service.budget_reset()
