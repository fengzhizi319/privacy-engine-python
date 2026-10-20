"""数据脱敏与哈希路由（mask / mask/record / mask/batch / mask/dataframe / hash）。

路径、请求别名与响应键均与 Go ``internal/rest/routes.go`` 对齐；Python 历史
字段（``field_name``/``context``/``result``）作为向后兼容的超集保留。
"""

from fastapi import APIRouter, HTTPException

from ..deps import SECURITY_DEPS, handle_request_exception, service
from ..privacy.masking import guess_field_type
from ..schemas import (
    HashRequest,
    MaskBatchRequest,
    MaskDataFrameRequest,
    MaskRecordRequest,
    MaskRequest,
)
from ..security.auth import require_permission

router = APIRouter()

# 与 Go handler 的 inferMaskType 结果保持一致：未识别字段按 default 拒绝出域。
_GO_MASK_TYPES = {
    "id_card",
    "phone",
    "bank_card",
    "name",
    "email",
    "address",
    "officer_id",
}


def _mask_one(field_name: str, value: str, field_type: str, salt: str) -> str:
    """按显式类型或字段名推断执行单值脱敏（与 Go maskHandler 对齐）。"""
    if field_type:
        return service.mask_by_type(field_type, value, salt)
    inferred = guess_field_type(field_name)
    if inferred in _GO_MASK_TYPES:
        return service.mask_by_type(inferred, value, salt)
    if inferred == "default":
        # Go 口径：不认识就不出域（400），绝不返回原文
        raise HTTPException(status_code=400, detail="INVALID_MASK_TYPE")
    return service.mask_by_type(inferred, value, salt)


@router.post("/v1/privacy/mask", dependencies=[*SECURITY_DEPS, require_permission("privacy:mask")])
def mask(req: MaskRequest):
    """对单个字段值进行脱敏。

    请求兼容 Go 契约 ``{field|field_name, value, type, salt}``；
    响应同时提供 Go 的 ``field``/``masked`` 与历史的 ``result`` 键。
    """
    try:
        field_name = req.field or req.field_name
        if not field_name and not req.type:
            raise HTTPException(status_code=400, detail="field or field_name is required")
        result = _mask_one(field_name, req.value, req.type, req.salt)
        return {"field": field_name, "masked": result, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/mask/record", dependencies=[*SECURITY_DEPS, require_permission("privacy:mask")])
@router.post("/v1/privacy/mask_record", dependencies=[*SECURITY_DEPS, require_permission("privacy:mask")])
def mask_record(req: MaskRecordRequest):
    """对整条记录进行脱敏（``/mask/record`` 与 ``/mask_record`` 双别名）。"""
    try:
        return {"result": service.mask_record(req.record, req.context)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/mask/batch", dependencies=[*SECURITY_DEPS, require_permission("privacy:mask")])
@router.post("/v1/privacy/mask_batch", dependencies=[*SECURITY_DEPS, require_permission("privacy:mask")])
def mask_batch(req: MaskBatchRequest):
    """批量脱敏。

    - ``values``（可配 ``field_names``）优先，与 Go 一致；
    - 否则使用 ``records`` 逐条按字段名自动脱敏。
    """
    try:
        if req.values:
            results: list[str] = []
            for index, value in enumerate(req.values):
                if index < len(req.field_names):
                    try:
                        results.append(_mask_one(req.field_names[index], value, "", ""))
                    except Exception:
                        # Go 批量口径：单个值失败以 *** 占位，不中断整批
                        results.append("***")
                else:
                    try:
                        results.append(service.mask_by_type("default", value))
                    except ValueError:
                        results.append("***")
            return {"results": results, "result": results}
        if req.records:
            records = [dict(service.mask_record(record, req.context)) for record in req.records]
            return {"results": records, "result": records}
        raise HTTPException(status_code=400, detail="records or values required")
    except HTTPException:
        raise
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/mask/dataframe", dependencies=[*SECURITY_DEPS, require_permission("privacy:mask")])
@router.post("/v1/privacy/mask_dataframe", dependencies=[*SECURITY_DEPS, require_permission("privacy:mask")])
def mask_dataframe(req: MaskDataFrameRequest):
    """DataFrame 脱敏接口。

    接受 records 列表，返回脱敏后的 records 列表。
    调用方可将 pandas/SecretFlow DataFrame 转为 records 后调用。
    """
    try:
        import pandas as pd

        df = pd.DataFrame(req.data)
        result = service.mask_dataframe(df, columns=req.columns, context=req.context)
        records = result.to_dict(orient="records")
        return {"data": records, "result": records}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/hash", dependencies=[*SECURITY_DEPS, require_permission("privacy:hash")])
def hash_value(req: HashRequest):
    """对单个值进行 HMAC-SM3 哈希（Go 响应键 ``hash``）。

    Returns:
        {"hash": <16 位 base64 摘要>, "result": <同值>}
    """
    try:
        result = service.hash(req.value, req.salt)
        return {"hash": result, "result": result}
    except Exception as e:
        handle_request_exception(e)
