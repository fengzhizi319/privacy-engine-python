"""流水线单元与集成测试 / Pipeline Unit and Integration Tests.

覆盖 PipelineService、classify_records、mask_records 以及 REST 路由端点。
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.main import app
from engine.pipeline import (
    PipelineResult,
    PipelineService,
    classify_records,
    mask_records,
)


@pytest.fixture
def sample_records() -> list[dict[str, str]]:
    """提供测试用样本医疗记录。"""
    return [
        {
            "name": "张伟",
            "id_card_no": "110101199003072381",
            "gender": "男",
            "age": "34",
            "diagnosis_name": "获得性免疫缺陷综合征(HIV)",
            "present_illness": "患者因反复发热就诊，检出HIV抗体阳性",
            "registered_address": "北京市东城区天安门广场1号",
        },
        {
            "name": "李娜",
            "id_card_no": "310101199508151247",
            "gender": "女",
            "age": "29",
            "diagnosis_name": "高血压病",
            "present_illness": "头晕1周，血压150/95mmHg",
            "registered_address": "上海市黄浦区南京东路100号",
        },
    ]


def test_classify_records(sample_records: list[dict[str, str]]) -> None:
    """测试 classify_records 识别风险等级。"""
    results = classify_records(sample_records, standard="jrt0197")
    assert len(results) == 2
    assert results[0].record_index == 0
    assert results[1].record_index == 1
    # 包含了敏感信息，最终等级至少为高风险 (L3/L4/L5 或 C3/C4/C5)
    assert results[0].final_level in ("L3", "L4", "L5", "C3", "C4", "C5")


def test_mask_records(sample_records: list[dict[str, str]]) -> None:
    """测试 mask_records 脱敏与 L4/L5 敏感词剥离。"""
    details = classify_records(sample_records, standard="jrt0197")
    masked_recs, mask_details = mask_records(sample_records, details, mask_l4=True, mask_l5=True)
    assert len(masked_recs) == 2
    
    # 验证 PII 掩码
    assert masked_recs[0]["name"] != "张伟"
    assert "110101" in masked_recs[0]["id_card_no"]
    assert "********" in masked_recs[0]["id_card_no"]
    
    # 验证 L5 HIV 强剥离
    assert "HIV" not in masked_recs[0]["diagnosis_name"]
    assert "[L5-IMMUNODEFICIENCY-SENSITIVE-MASKED]" in masked_recs[0]["diagnosis_name"]


def test_pipeline_service_process_records(sample_records: list[dict[str, str]]) -> None:
    """测试 PipelineService.process_records 端到端处理。"""
    service = PipelineService()
    res: PipelineResult = service.process_records(sample_records)
    
    assert res.classification_summary.total_records == 2
    assert len(res.record_details) == 2
    assert len(res.masked_records) == 2
    assert res.classification_summary.duration_ms >= 0.0


def test_pipeline_service_process_csv(tmp_path: Path, sample_records: list[dict[str, str]]) -> None:
    """测试 PipelineService.process_csv 从文件读取并处理。"""
    import csv

    csv_file = tmp_path / "test_data.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_records[0].keys()))
        writer.writeheader()
        writer.writerows(sample_records)

    service = PipelineService()
    res = service.process_csv(csv_file)
    assert res.classification_summary.total_records == 2
    assert len(res.masked_records) == 2


def test_rest_pipeline_process_records_endpoint(sample_records: list[dict[str, str]]) -> None:
    """测试 REST 端点 /v1/pipeline/process_records。"""
    client = TestClient(app)
    response = client.post(
        "/v1/pipeline/process_records",
        json={"records": sample_records, "standard": "jrt0197", "mask_l4": True, "mask_l5": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["classification_summary"]["total_records"] == 2
    assert len(data["masked_records"]) == 2


def test_rest_pipeline_process_csv_endpoint(sample_records: list[dict[str, str]]) -> None:
    """测试 REST 端点 /v1/pipeline/process_csv 文件上传。"""
    import csv

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(sample_records[0].keys()))
    writer.writeheader()
    writer.writerows(sample_records)
    csv_bytes = output.getvalue().encode("utf-8")

    client = TestClient(app)
    response = client.post(
        "/v1/pipeline/process_csv",
        files={"file": ("data.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["classification_summary"]["total_records"] == 2
