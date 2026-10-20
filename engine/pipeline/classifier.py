"""分类分级封装器 / Classification Wrapper.

封装 DynClassificationService 调用，将分类结果转换为流水线内部模型。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

from ..dynclassification import DynClassificationService
from .models import FieldClassificationDetail, RecordClassificationDetail

# 单例管理
_service: Optional[DynClassificationService] = None
_service_lock = threading.Lock()


def _get_dyn_service(rules_dir: str | Path | None = None) -> DynClassificationService:
    """获取 DynClassificationService 单例。"""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                rd = rules_dir or os.environ.get("PRIVACY_DYNCLASSIFICATION_RULES_DIR", "rules")
                _service = DynClassificationService(rules_dir=rd)
    return _service


def classify_records(
    records: list[dict[str, Any]],
    standard: str = "jrt0197",
    rules_dir: str | Path | None = None,
) -> list[RecordClassificationDetail]:
    """对多条记录进行分类分级。

    Args:
        records: 记录字典列表。
        standard: 分类标准。
        rules_dir: 规则目录。

    Returns:
        记录级分级明细列表。
    """
    svc = _get_dyn_service(rules_dir)
    svc.check_and_reload()

    results: list[RecordClassificationDetail] = []

    for idx, record in enumerate(records):
        resp = svc.classify_record(record=record, record_index=idx, standard=standard)

        field_details: list[FieldClassificationDetail] = []
        if resp.record_result:
            for fname, fres in resp.record_result.field_results.items():
                # 提取分类类别
                category = None
                if fres.tags:
                    for tag in fres.tags:
                        if tag.category:
                            category = tag.category
                            break

                field_details.append(
                    FieldClassificationDetail(
                        field_name=fname,
                        field_value=str(fres.field_value) if fres.field_value is not None else "",
                        sensitivity_level=fres.final_level or "L1",
                        category=category,
                        confidence=fres.confidence,
                        engine_layer=fres.engine_layer or "L1_RULE",
                        reasoning=fres.reasoning,
                    )
                )

        # 记录最终等级
        final_level = "L1"
        if resp.record_result and resp.record_result.final_level:
            final_level = resp.record_result.final_level
        elif field_details:
            # 取最高级别
            level_order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
            final_level = max(
                (fd.sensitivity_level for fd in field_details),
                key=lambda x: level_order.get(x, 0),
            )

        results.append(
            RecordClassificationDetail(
                record_index=idx,
                final_level=final_level,
                field_details=field_details,
            )
        )

    return results
