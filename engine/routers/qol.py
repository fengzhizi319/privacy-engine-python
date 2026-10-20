"""查询混淆路由（qol/obfuscate|batch）。"""

from fastapi import APIRouter

from ..deps import SECURITY_DEPS, handle_request_exception, service
from ..schemas import QolBatchRequest, QolRequest
from ..security.auth import require_permission

router = APIRouter()


def _num_decoys(num_decoys: int, num_dummies: int) -> int:
    """Go 别名优先级：num_decoys > 0 时优先，否则回退 num_dummies（服务层收口非正数）。"""
    return num_decoys if num_decoys > 0 else num_dummies


def _domain(value: str) -> str:
    """Go REST 口径：空串默认 medical，其余仅 medical 走医疗池，否则 generic。"""
    return "medical" if value in ("", "medical") else "generic"


@router.post("/v1/privacy/qol/obfuscate", dependencies=[*SECURITY_DEPS, require_permission("privacy:qol")])
def qol_obfuscate(req: QolRequest):
    """查询混淆接口。

    Args:
        req: QolRequest 请求体。

    Returns:
        {"result": <混淆后的查询列表>}
    """
    try:
        queries = service.obfuscate_query(
            req.query,
            _num_decoys(req.num_decoys, req.num_dummies),
            _domain(req.domain),
            medical_pool=req.medical_pool,
            generic_pool=req.generic_pool,
            seed=req.seed,
        )
        return {
            "queries": queries,
            "real_index": queries.index(req.query),
            "result": queries,
        }
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/qol/obfuscate/batch", dependencies=[*SECURITY_DEPS, require_permission("privacy:qol")])
def qol_obfuscate_batch(req: QolBatchRequest):
    """批量查询混淆接口。"""
    try:
        return {"results": service.obfuscate_query_batch(
            req.queries,
            _num_decoys(req.num_decoys, req.num_dummies),
            _domain(req.domain),
            medical_pool=req.medical_pool,
            generic_pool=req.generic_pool,
            seed=req.seed,
        )}
    except Exception as e:
        handle_request_exception(e)
