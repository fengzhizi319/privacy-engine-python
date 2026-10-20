"""医疗数据分类分级与脱敏流水线 / Medical Data Classification & Masking Pipeline.

提供端到端的医疗数据处理流水线：
1. 调用 DynClassificationService 进行分类分级
2. 调用 masking 原语对 L4/L5 级数据脱敏
3. 输出分级报告 + 脱敏后数据

Public API:
    PipelineService  — 流水线编排器
    PipelineResult   — 统一输出模型
"""

from .classifier import classify_records
from .masker import mask_records
from .models import (
    ClassificationSummary,
    FieldClassificationDetail,
    MaskingDetail,
    PipelineResult,
    RecordClassificationDetail,
)
from .router import router
from .service import PipelineService

__all__ = [
    "PipelineService",
    "PipelineResult",
    "ClassificationSummary",
    "FieldClassificationDetail",
    "RecordClassificationDetail",
    "MaskingDetail",
    "classify_records",
    "mask_records",
    "router",
]
