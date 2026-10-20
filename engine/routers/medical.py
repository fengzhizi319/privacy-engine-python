"""医疗敏感数据处理路由模块 / Medical Data Processing Router Module.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from engine.deps import SECURITY_DEPS, service
from engine.security.auth import require_permission

router = APIRouter(prefix="/v1/medical", tags=["medical"])


# 输入规模上限（资源耗尽防护）：单请求记录数、单记录字段数、单字段值长度。
# 脱敏管线含 NER 推理（百毫秒~秒级/字段）与复杂句法正则，无界输入可被用于 DoS。
_MAX_RECORDS = 500
_MAX_FIELDS_PER_RECORD = 100
_MAX_FIELD_VALUE_LENGTH = 100_000


class MedicalProcessRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., max_length=_MAX_RECORDS, description="医疗与身份数据记录列表")

    @field_validator("records")
    @classmethod
    def _cap_payload_size(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """限制单记录字段数与字段值长度（防超大 payload 造成的 CPU/内存耗尽）。"""
        for rec in records:
            if len(rec) > _MAX_FIELDS_PER_RECORD:
                raise ValueError(f"单条记录字段数超过上限 {_MAX_FIELDS_PER_RECORD}")
            for key, value in rec.items():
                if isinstance(value, str) and len(value) > _MAX_FIELD_VALUE_LENGTH:
                    raise ValueError(f"字段 {key!r} 值长度超过上限 {_MAX_FIELD_VALUE_LENGTH}")
        return records


class MedicalProcessResponse(BaseModel):
    classification_report: list[dict[str, Any]] = Field(..., description="分类分级报告")
    sanitized_data: list[dict[str, str]] = Field(..., description="脱敏清洗后的合规数据")
    summary: dict[str, Any] = Field(..., description="处理元数据与统计")


class MedicalSanitizeRequest(BaseModel):
    """与 Go `/v1/medical/sanitize` 对齐的单记录请求。"""

    record: dict[str, Any] = Field(..., max_length=_MAX_FIELDS_PER_RECORD)
    domain: str = Field(..., max_length=100)


class MedicalSanitizeBatchRequest(BaseModel):
    """与 Go `/v1/medical/sanitize/batch` 对齐的批量请求。"""

    records: list[dict[str, Any]] = Field(..., max_length=_MAX_RECORDS)
    domain: str = Field(..., max_length=100)


def _sanitize_or_400(records: list[dict[str, Any]], domain: str):
    try:
        return service.sanitize_medical_batch(records, domain)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Unknown or unsupported datasource"
        ) from exc


@router.post(
    "/sanitize",
    dependencies=[*SECURITY_DEPS, require_permission("medical:process")],
)
def sanitize_medical(req: MedicalSanitizeRequest):
    """单记录医疗/身份数据脱敏（Go 契约）。"""
    results = _sanitize_or_400([req.record], req.domain)
    return {"result": results[0]}


@router.post(
    "/sanitize/batch",
    dependencies=[*SECURITY_DEPS, require_permission("medical:process")],
)
def sanitize_medical_batch(req: MedicalSanitizeBatchRequest):
    """批量医疗/身份数据脱敏（Go 契约）。"""
    return {"results": _sanitize_or_400(req.records, req.domain)}


@router.post(
    "/process",
    response_model=MedicalProcessResponse,
    dependencies=[*SECURITY_DEPS, require_permission("medical:process")],
)
def process_medical(req: MedicalProcessRequest, response: Response) -> dict[str, Any]:
    """对提交的医疗数据集执行 3-Layer 分类分级与 L4/L5 敏感数据抹平脱敏。"""
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Thu, 31 Dec 2026 23:59:59 GMT"
    response.headers["Link"] = '</v1/agent/process>; rel="successor-version"'
    response.headers["X-PrivShield-Canonical-Path"] = "/v1/agent/process"

    result = service.process_medical_data(req.records)

    import hashlib
    import json

    raw_bytes = json.dumps(req.records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    input_hash = hashlib.sha256(raw_bytes).hexdigest()

    sanitized = result.get("sanitized_data", [])
    sanitized_bytes = json.dumps(sanitized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    output_hash = hashlib.sha256(sanitized_bytes).hexdigest()

    summary = dict(result.get("summary", {}))
    summary["input_hash"] = input_hash
    summary["output_hash"] = output_hash
    result["summary"] = summary

    return result
