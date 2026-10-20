import hashlib
import json
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from engine.deps import SECURITY_DEPS, service
from engine.security.auth import require_any_permission

router = APIRouter(prefix="/v1/agent", tags=["agent"])

_MAX_RECORDS = 500
_MAX_FIELDS_PER_RECORD = 100
_MAX_FIELD_VALUE_LENGTH = 100_000


class AgentProcessRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., max_length=_MAX_RECORDS, description="待评估与脱敏的数据记录列表")
    api_code: str = Field(default="", description="canonical API code, e.g. api1_yibao")
    datasource_id: str = Field(default="", description="canonical datasource ID, e.g. ds_yibao")

    @field_validator("records")
    @classmethod
    def _cap_payload_size(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for rec in records:
            if len(rec) > _MAX_FIELDS_PER_RECORD:
                raise ValueError(f"单条记录字段数超过上限 {_MAX_FIELDS_PER_RECORD}")
            for key, value in rec.items():
                if isinstance(value, str) and len(value) > _MAX_FIELD_VALUE_LENGTH:
                    raise ValueError(f"字段 {key!r} 值长度超过上限 {_MAX_FIELD_VALUE_LENGTH}")
        return records


class AgentProcessResponse(BaseModel):
    classification_report: list[dict[str, Any]] = Field(..., description="分类分级报告")
    sanitized_data: list[dict[str, str]] = Field(..., description="脱敏清洗后的合规数据")
    summary: dict[str, Any] = Field(..., description="处理元数据与统计")


@router.post(
    "/process",
    response_model=AgentProcessResponse,
    dependencies=[*SECURITY_DEPS, require_any_permission("agent:process", "medical:process")],
)
def process_agent(req: AgentProcessRequest) -> dict[str, Any]:
    """对提交的数据集执行 3-Layer 敏感特征识别、分类分级与隐私脱敏治理。"""
    result = service.process_medical_data(req.records)

    raw_bytes = json.dumps(req.records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    input_hash = hashlib.sha256(raw_bytes).hexdigest()

    sanitized = result.get("sanitized_data", [])
    sanitized_bytes = json.dumps(sanitized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    output_hash = hashlib.sha256(sanitized_bytes).hexdigest()

    summary = dict(result.get("summary", {}))
    summary["input_hash"] = input_hash
    summary["output_hash"] = output_hash
    if req.api_code:
        summary["api_code"] = req.api_code
    if req.datasource_id:
        summary["datasource_id"] = req.datasource_id
    result["summary"] = summary

    return result
