"""流水线数据模型 / Pipeline Data Models.

定义流水线处理的请求/响应 Pydantic 模型。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class FieldClassificationDetail(BaseModel):
    """单字段分级明细。"""

    field_name: str = Field(description="字段名")
    field_value: str = Field(description="字段原始值")
    sensitivity_level: str = Field(description="敏感度等级 (L1-L5)")
    category: Optional[str] = Field(default=None, description="分类类别")
    confidence: float = Field(default=1.0, description="置信度")
    engine_layer: str = Field(default="L1_RULE", description="引擎层级")
    reasoning: Optional[str] = Field(default=None, description="推理说明")


class RecordClassificationDetail(BaseModel):
    """单记录分级明细。"""

    record_index: int = Field(description="记录索引")
    final_level: str = Field(description="记录最终等级")
    field_details: list[FieldClassificationDetail] = Field(
        default_factory=list, description="字段级分级明细"
    )


class ClassificationSummary(BaseModel):
    """分级汇总统计。"""

    total_records: int = Field(description="总记录数")
    level_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="各级别分布，如 {'L1': 5, 'L2': 8, 'L3': 3, 'L4': 3, 'L5': 1}",
    )
    high_risk_fields: list[str] = Field(
        default_factory=list, description="L4/L5 字段名列表"
    )
    standard_id: str = Field(default="jrt0197", description="使用的分类标准")
    duration_ms: float = Field(default=0.0, description="处理耗时(毫秒)")


class MaskingDetail(BaseModel):
    """脱敏操作明细。"""

    record_index: int = Field(description="记录索引")
    field_name: str = Field(description="字段名")
    original_level: str = Field(description="原始敏感度等级")
    masking_type: str = Field(description="脱敏类型 (ID_CARD, NAME, ADDRESS 等)")
    original_value: str = Field(description="原始值")
    masked_value: str = Field(description="脱敏后值")


class PipelineResult(BaseModel):
    """流水线统一输出模型。"""

    # 分级数据
    classification_summary: ClassificationSummary = Field(description="分级汇总")
    record_details: list[RecordClassificationDetail] = Field(
        default_factory=list, description="记录级分级明细"
    )

    # 脱敏数据
    masked_records: list[dict[str, Any]] = Field(
        default_factory=list, description="脱敏后记录列表"
    )
    masking_details: list[MaskingDetail] = Field(
        default_factory=list, description="脱敏操作明细"
    )
