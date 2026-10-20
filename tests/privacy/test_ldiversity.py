"""L-多样性（L-Diversity）与 L-多样性感知 Mondrian 切分测试套件。

对齐 Go sdk/kano/ldiversity_partition_test.go 与 CheckDistinctLDiversity 行为。
"""

import pytest
from fastapi.testclient import TestClient

from engine.main import app
from engine.privacy.kano import check_distinct_l_diversity
from engine.privacy.kano_table import (
    anonymize_with_l_diversity,
    distinct_field_count,
    mondrian_with_l_diversity,
)


def test_distinct_field_count_ignores_empty():
    records = [
        {"val": "A"},
        {"val": "B"},
        {"val": ""},
        {"other": "C"},
    ]
    # 空值与缺失字段不应计入
    assert distinct_field_count(records, "val") == 2


def test_check_distinct_l_diversity_compliant():
    records = [
        {"zip": "1000", "age": "20", "disease": "Flu"},
        {"zip": "1000", "age": "20", "disease": "Fever"},
        {"zip": "2000", "age": "30", "disease": "Cancer"},
        {"zip": "2000", "age": "30", "disease": "Diabetes"},
    ]
    res = check_distinct_l_diversity(records, ["zip", "age"], "disease", 2)
    assert res.is_compliant is True
    assert res.l == 2
    assert res.violations == 0
    assert res.min_diversity == 2
    assert res.group_count == 2
    for stat in res.group_stats:
        assert stat.is_compliant is True
        assert stat.distinct_count >= 2


def test_check_distinct_l_diversity_non_compliant():
    records = [
        {"zip": "1000", "age": "20", "disease": "Flu"},
        {"zip": "1000", "age": "20", "disease": "Flu"},  # 同质性攻击
        {"zip": "2000", "age": "30", "disease": "Cancer"},
        {"zip": "2000", "age": "30", "disease": "Diabetes"},
    ]
    res = check_distinct_l_diversity(records, ["zip", "age"], "disease", 2)
    assert res.is_compliant is False
    assert res.l == 2
    assert res.violations == 1
    assert res.min_diversity == 1
    assert res.group_count == 2
    non_comp = [s for s in res.group_stats if not s.is_compliant]
    assert len(non_comp) == 1
    assert non_comp[0].distinct_count == 1
    assert non_comp[0].sensitive_values == {"Flu": 2}


def test_check_distinct_l_diversity_empty():
    res = check_distinct_l_diversity([], ["zip"], "disease", 2)
    assert res.is_compliant is True
    assert res.min_diversity == 0
    assert res.group_count == 0


def test_anonymize_with_l_diversity_rejects_split():
    # 4条记录，age两两相同（10或11），disease在age=10完全同质（都是Flu）
    # 若按age切分，[10, 10]一侧disease distinct计数只有1，跌破l=2
    records = [
        {"age": "10", "disease": "Flu"},
        {"age": "10", "disease": "Flu"},
        {"age": "11", "disease": "Cold"},
        {"age": "11", "disease": "Fever"},
    ]
    # L-多样性感知切分：应拒绝切分，保持为1个等价类
    res = anonymize_with_l_diversity(records, ["age"], "disease", k=2, l=2)
    assert len(res) == 4
    # 交叉验证输出结果是否满足l=2
    audit = check_distinct_l_diversity(res, ["age"], "disease", 2)
    assert audit.is_compliant is True
    assert audit.group_count == 1


def test_anonymize_with_l_diversity_allows_split():
    records = [
        {"age": "10", "disease": "Flu"},
        {"age": "10", "disease": "Cold"},
        {"age": "11", "disease": "Fever"},
        {"age": "11", "disease": "Cough"},
    ]
    res = anonymize_with_l_diversity(records, ["age"], "disease", k=2, l=2)
    assert len(res) == 4
    audit = check_distinct_l_diversity(res, ["age"], "disease", 2)
    assert audit.is_compliant is True
    assert audit.group_count == 2


def test_mondrian_with_l_diversity():
    rows = [
        {"age": 10, "disease": "Flu"},
        {"age": 10, "disease": "Flu"},
        {"age": 11, "disease": "Cold"},
        {"age": 11, "disease": "Fever"},
    ]
    details = mondrian_with_l_diversity(rows, ["age"], "disease", k=2, l=2, return_details=True)
    assert details.equivalence_classes_count == 1
    assert details.k == 2


def test_api_ldiversity_check_and_anonymize():
    client = TestClient(app)

    # 1. 检查 API
    records = [
        {"zip": "1000", "age": "20", "disease": "Flu"},
        {"zip": "1000", "age": "20", "disease": "Flu"},
        {"zip": "2000", "age": "30", "disease": "Cancer"},
        {"zip": "2000", "age": "30", "disease": "Diabetes"},
    ]
    r = client.post(
        "/v1/privacy/l_diversity/check",
        json={
            "records": records,
            "qi_fields": ["zip", "age"],
            "sensitive_field": "disease",
            "l": 2,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["is_compliant"] is False
    assert data["violations"] == 1
    assert len(data["group_stats"]) == 2

    # 2. 泛化 API
    r2 = client.post(
        "/v1/privacy/k_anonymize/ldiversity",
        json={
            "records": [
                {"age": "10", "disease": "Flu"},
                {"age": "10", "disease": "Flu"},
                {"age": "11", "disease": "Cold"},
                {"age": "11", "disease": "Fever"},
            ],
            "qi_cols": ["age"],
            "sensitive_field": "disease",
            "k": 2,
            "l": 2,
        },
    )
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["group_count"] == 1
    assert data2["l"] == 2
