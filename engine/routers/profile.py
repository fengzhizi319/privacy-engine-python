"""隐私参数推荐路由（profile/recommend）。"""

from fastapi import APIRouter

from ..deps import PROFILE_PATH, SECURITY_DEPS, _service_cache
from ..schemas import RecommendRequest
from ..security.auth import require_permission
from ..service import PrivacyService

router = APIRouter()


def _recommend(namespace: str, values=None, rows=None, qi_cols=None):
    rec_service = _service_cache.get(namespace)
    if rec_service is None:
        rec_service = PrivacyService(profile_path=PROFILE_PATH, namespace=namespace)
        _service_cache[namespace] = rec_service
    recommended = rec_service.recommend_and_save_params(values, rows, qi_cols)
    return {
        "status": "success",
        "namespace": namespace,
        "recommended_params": recommended,
    }


@router.post(
    "/v1/privacy/profile/recommend",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:profile")],
)
def recommend_profile(req: RecommendRequest):
    """根据输入数据自动推荐差分隐私等隐私处理参数并保存。"""
    return _recommend(req.namespace, req.values, req.rows, req.qi_cols)


@router.get(
    "/v1/privacy/profile/recommend",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:profile")],
)
def recommend_profile_get(namespace: str = "default"):
    """GET 形态（Go 对齐）：无请求体，仅按命名空间返回推荐参数。"""
    return _recommend(namespace or "default")

