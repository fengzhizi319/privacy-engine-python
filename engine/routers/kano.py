"""K-匿名路由（k_anonymize[.record|.table|.dataframe]）。

响应同时提供 Go 契约字段（records/result/k/group_count/...）与历史 Python
字段，保证两端 REST 契约一致且不破坏既有调用方。
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from ..deps import SECURITY_DEPS, handle_request_exception, service
from ..schemas import (
    KAnonDataFrameRequest,
    KAnonLDiversityRequest,
    KAnonRequest,
    KAnonTableRequest,
    LDiversityCheckRequest,
)
from ..security.auth import require_permission

router = APIRouter()


def _resolve_dataset(req: KAnonRequest) -> tuple[list[dict[str, Any]], list[str], int]:
    """解析 Go/Python 双别名并返回（records, qi_cols, k）。"""
    if req.records:
        records = list(req.records)
    elif req.record is not None:
        records = [req.record]
    else:
        records = []
    qi_cols = req.qi_fields or req.qi_cols
    # 与 Go clampK 对齐：k < 1 统一收口为 2
    k = req.k if req.k >= 1 else 2
    if not records or not qi_cols:
        raise HTTPException(
            status_code=400, detail="records and qi_fields/qi_cols required"
        )
    return records, qi_cols, k


def _dataset_response(records: list[dict[str, Any]], qi_cols: list[str], k: int) -> dict[str, Any]:
    """Go KAnonymize 契约：records/result/k/group_count。"""
    details = service.k_anonymize_table(
        records, qi_cols, k, return_details=True
    )
    generalized = list(details.value)
    return {
        "records": generalized,
        "result": generalized,
        "k": details.k,
        "group_count": details.equivalence_classes_count,
    }


@router.post(
    "/v1/privacy/k_anonymize",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:kano")],
)
def k_anonymize(req: KAnonRequest):
    """数据集级 K-匿名（Go 对齐：Mondrian 等价类泛化）。"""
    try:
        records, qi_cols, k = _resolve_dataset(req)
        return _dataset_response(records, qi_cols, k)
    except Exception as e:
        handle_request_exception(e)


@router.post(
    "/v1/privacy/k_anonymize/record",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:kano")],
)
def k_anonymize_record(req: KAnonRequest):
    """Go 容器级 K-匿名别名：``record`` 包装为单行数据集处理。

    Go 的 ``/k_anonymize/record`` 与 ``/k_anonymize`` 共用同一 handler，
    单条记录落入单一等价组（保持原值），响应 ``result`` 恒为记录列表。
    """
    try:
        records, qi_cols, k = _resolve_dataset(req)
        return _dataset_response(records, qi_cols, k)
    except Exception as e:
        handle_request_exception(e)


@router.post(
    "/v1/privacy/k_anonymize/table",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:kano")],
)
@router.post(
    "/v1/privacy/k_anonymize_table",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:kano")],
)
def k_anonymize_table(req: KAnonTableRequest):
    """整张表 K-匿名泛化接口（Mondrian）。"""
    try:
        records = req.records if req.records else req.rows
        qi_cols = req.qi_cols or req.qi_fields
        # 与 Go 表级 handler 对齐：k < 2 统一收口为 2
        k = req.k if req.k >= 2 else 2
        if not records or not qi_cols:
            raise HTTPException(
                status_code=400, detail="records/rows and qi_cols required"
            )
        details = service.k_anonymize_table(
            list(records), qi_cols, k, req.max_depth, return_details=True
        )
        generalized = list(details.value)
        return {
            "records": generalized,
            "result": generalized,
            "k": details.k,
            "qi_cols": qi_cols,
            "group_count": details.equivalence_classes_count,
            "equivalence_classes_count": details.equivalence_classes_count,
        }
    except Exception as e:
        handle_request_exception(e)


@router.post(
    "/v1/privacy/k_anonymize/dataframe",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:kano")],
)
def k_anonymize_dataframe(req: KAnonDataFrameRequest):
    """DataFrame K-匿名泛化接口。"""
    try:
        records = req.records if req.records else req.data
        qi_cols = req.qi_cols or req.qi_fields
        k = req.k if req.k >= 2 else 2
        if not records or not qi_cols:
            raise HTTPException(
                status_code=400, detail="records/data and qi_cols required"
            )
        import pandas as pd

        df = pd.DataFrame(list(records))
        details = service.k_anonymize_dataframe(
            df, qi_cols, k, req.max_depth, return_details=True
        )
        output = details.value
        result = output.to_dict(orient="records") if hasattr(output, "to_dict") else output
        return {
            "result": result,
            "data": result,
            "k": details.k,
            "qi_cols": qi_cols,
            "rows_out": len(result),
        }
    except Exception as e:
        handle_request_exception(e)


@router.post(
    "/v1/privacy/l_diversity/check",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:kano")],
)
def check_l_diversity(req: LDiversityCheckRequest):
    """Distinct L-多样性事后检验接口。"""
    try:
        records = req.records
        qi_cols = req.qi_fields or req.qi_cols
        if not records or not qi_cols:
            raise HTTPException(
                status_code=400, detail="records and qi_fields/qi_cols required"
            )
        if not req.sensitive_field:
            raise HTTPException(
                status_code=400, detail="sensitive_field required"
            )
        res = service.check_distinct_l_diversity(
            records, qi_cols, req.sensitive_field, req.l
        )
        return {
            "is_compliant": res.is_compliant,
            "l": res.l,
            "min_diversity": res.min_diversity,
            "violations": res.violations,
            "group_count": res.group_count,
            "group_stats": [
                {
                    "group_index": g.group_index,
                    "record_count": g.record_count,
                    "distinct_count": g.distinct_count,
                    "sensitive_values": g.sensitive_values,
                    "is_compliant": g.is_compliant,
                }
                for g in res.group_stats
            ],
        }
    except Exception as e:
        handle_request_exception(e)


@router.post(
    "/v1/privacy/k_anonymize/ldiversity",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:kano")],
)
@router.post(
    "/v1/privacy/k_anonymize_ldiversity",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:kano")],
)
def k_anonymize_with_ldiversity(req: KAnonLDiversityRequest):
    """带 L-多样性约束的 K-匿名泛化接口（Mondrian）。"""
    try:
        records = req.records if req.records is not None else req.rows
        qi_cols = req.qi_fields or req.qi_cols
        k = req.k if req.k >= 2 else 2
        l = req.l if req.l >= 1 else 1
        if not records or not qi_cols:
            raise HTTPException(
                status_code=400, detail="records/rows and qi_cols required"
            )
        if not req.sensitive_field:
            raise HTTPException(
                status_code=400, detail="sensitive_field required"
            )
        details = service.k_anonymize_ldiversity(
            records,
            qi_cols,
            req.sensitive_field,
            k=k,
            l=l,
            max_depth=req.max_depth,
            return_details=True,
        )
        generalized = list(details.value)
        return {
            "records": generalized,
            "result": generalized,
            "k": details.k,
            "l": l,
            "sensitive_field": req.sensitive_field,
            "qi_cols": qi_cols,
            "group_count": details.equivalence_classes_count,
            "equivalence_classes_count": details.equivalence_classes_count,
        }
    except Exception as e:
        handle_request_exception(e)
