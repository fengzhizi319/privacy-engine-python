"""流水线 REST 路由 / Pipeline REST Router.

提供 /v1/pipeline REST API 端点。
"""

from __future__ import annotations

import csv
import io
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, File, UploadFile, Query, status
from pydantic import BaseModel, Field

from engine.deps import SECURITY_DEPS
from engine.security.auth import require_permission
from engine.observability.logging_config import get_logger

from .models import PipelineResult
from .service import PipelineService

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/pipeline", tags=["Pipeline"])
_service = PipelineService()
_MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


class ProcessRecordsRequest(BaseModel):
    """process_records 请求模型。"""

    records: list[dict[str, Any]] = Field(description="记录数组")
    standard: Optional[str] = Field(default="jrt0197", description="分类标准")
    mask_l4: bool = Field(default=True, description="是否掩码 L4 数据")
    mask_l5: bool = Field(default=True, description="是否掩码 L5 数据")


@router.post(
    "/process_records",
    response_model=PipelineResult,
    dependencies=[*SECURITY_DEPS, require_permission("pipeline:process")],
)
async def process_records(req: ProcessRecordsRequest) -> PipelineResult:
    """对 JSON 记录数组执行分类分级与脱敏流水线。"""
    try:
        return _service.process_records(
            records=req.records,
            standard=req.standard,
            mask_l4=req.mask_l4,
            mask_l5=req.mask_l5,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception:
        logger.error("pipeline_process_records_failed", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Pipeline processing encounter internal error")


@router.post(
    "/process_csv",
    response_model=PipelineResult,
    dependencies=[*SECURITY_DEPS, require_permission("pipeline:process")],
)
async def process_csv(
    file: UploadFile = File(..., description="上传的 CSV 文件"),
    standard: Optional[str] = Query(default="jrt0197", description="分类标准"),
    mask_l4: bool = Query(default=True, description="是否掩码 L4 数据"),
    mask_l5: bool = Query(default=True, description="是否掩码 L5 数据"),
) -> PipelineResult:
    """上传 CSV 文件，执行分类分级与脱敏流水线。"""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV files are supported")

    try:
        # 分块读取并累计校验大小：在读取过程中即时检测超限，
        # 避免先全量读入内存再校验导致超大文件耗尽内存（DoS 防护）；超限返回 413。
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_CSV_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"CSV file size exceeds limit of {_MAX_CSV_SIZE_BYTES // (1024*1024)}MB",
                )
            chunks.append(chunk)
        content = b"".join(chunks)
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        records = [dict(row) for row in reader]

        return _service.process_records(
            records=records,
            standard=standard,
            mask_l4=mask_l4,
            mask_l5=mask_l5,
        )
    except HTTPException:
        raise
    except Exception:
        logger.error("pipeline_process_csv_failed", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process uploaded CSV file")
