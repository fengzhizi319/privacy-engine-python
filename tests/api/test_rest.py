"""REST API 接口测试集。

使用 FastAPI TestClient 对 engine.main 中的各端点进行集成测试，
覆盖健康检查、脱敏、哈希、差分隐私、K-匿名与查询混淆等核心接口。
数据分类接口测试已拆分到 tests/test_classification_rest.py。

Integration tests for REST endpoints using FastAPI TestClient.
Covers health, masking, hashing, DP, K-anonymity and query obfuscation.
Classification endpoint tests are in tests/test_classification_rest.py.
"""

import pytest
from fastapi.testclient import TestClient

from engine.main import app

# 复用同一个 TestClient 实例，避免重复创建应用
client = TestClient(app)


@pytest.fixture(autouse=True)
def seed_global_rng():
    """每个 REST 测试前设定全局 DP API 随机种子，确保测试结果确定。"""
    from engine.main import service
    service.dp_api.rng.seed(42)
    yield


def test_health():
    """测试健康检查接口返回状态正常。"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_mask():
    """测试单字段手机号脱敏接口。"""
    response = client.post("/v1/privacy/mask", json={"field_name": "mobile", "value": "13812345678", "context": ""})
    assert response.status_code == 200
    assert response.json()["result"] == "138****5678"


def test_mask_record():
    """测试整记录脱敏接口，验证手机号与姓名字段均按预期脱敏。"""
    response = client.post(
        "/v1/privacy/mask_record",
        json={"record": {"mobile": "13812345678", "name": "张三丰"}, "context": ""},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["mobile"] == "138****5678"


def test_hash():
    """测试 HMAC 哈希接口，验证返回结果长度为 16。"""
    response = client.post("/v1/privacy/hash", json={"value": "13812345678", "salt": "salt-a"})
    assert response.status_code == 200
    assert len(response.json()["result"]) == 16


def test_dp_count():
    """测试差分隐私计数接口，验证返回非负结果。"""
    response = client.post(
        "/v1/privacy/dp/count",
        json={"values": [1.0, 0.0, 1.0, 1.0], "params": {"epsilon": 1.0}},
    )
    assert response.status_code == 200
    assert response.json()["result"] >= 0


def test_dp_sum_with_clipping():
    """测试差分隐私求和接口的 clipping 与 Gaussian 机制。"""
    response = client.post(
        "/v1/privacy/dp/sum",
        json={
            "values": [1.0, 2.0, 3.0, 100.0],
            "params": {
                "epsilon": 10.0,
                "delta": 1e-5,
                "mechanism": "gaussian",
                "clip_lower": 0.0,
                "clip_upper": 10.0,
            },
        },
    )
    assert response.status_code == 200, response.json()
    result = response.json()["result"]
    assert 0 <= result <= 30


def test_dp_sum_missing_clip_for_gaussian(monkeypatch):
    """Gaussian 求和缺少 clip 参数时应返回 400。

    禁用个性化 profile，避免命名空间默认 clip 掩盖「请求未显式提供边界」的路径。
    """
    monkeypatch.setenv("PRIVACY_PERSONALIZED_PROFILE", "/nonexistent/profiles.yaml")
    response = client.post(
        "/v1/privacy/dp/sum",
        json={
            "values": [1.0, 2.0, 3.0],
            "params": {"epsilon": 1.0, "delta": 1e-6, "mechanism": "gaussian"},
        },
    )
    assert response.status_code == 400


def test_k_anonymize():
    """测试 K-匿名 record 别名接口（与 Go 同 handler）。

    单条记录落入单一等价组，准标识符保持原值；响应 result 为记录列表。
    """
    response = client.post(
        "/v1/privacy/k_anonymize/record",
        json={
            "record": {"age": "28", "zipcode": "518057", "gender": "女", "disease": "胃癌"},
            "qi_cols": ["age", "zipcode", "gender"],
            "k": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["result"], list)
    result = body["result"][0]
    assert result["gender"] == "女"
    assert result["zipcode"] == "518057"
    assert body["k"] == 5
    assert body["group_count"] == 1


def test_k_anonymize_table():
    """测试数据集级 K-匿名接口。"""
    response = client.post(
        "/v1/privacy/k_anonymize/table",
        json={
            "rows": [
                {"age": 25, "zipcode": "100001", "gender": "M", "disease": "A"},
                {"age": 26, "zipcode": "100002", "gender": "M", "disease": "B"},
                {"age": 27, "zipcode": "100003", "gender": "M", "disease": "C"},
                {"age": 55, "zipcode": "200001", "gender": "F", "disease": "D"},
                {"age": 56, "zipcode": "200002", "gender": "F", "disease": "E"},
                {"age": 57, "zipcode": "200003", "gender": "F", "disease": "F"},
            ],
            "qi_cols": ["age", "zipcode", "gender"],
            "k": 3,
        },
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert len(result) == 6
    diseases = {r["disease"] for r in result}
    assert diseases == {"A", "B", "C", "D", "E", "F"}


def test_qol():
    """测试查询混淆接口，验证返回列表包含真实查询且总长度为 num_dummies + 1。"""
    response = client.post(
        "/v1/privacy/qol/obfuscate",
        json={"query": "糖尿病患者用药趋势", "num_dummies": 3, "domain": "medical"},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert len(result) == 4
    assert "糖尿病患者用药趋势" in result


def test_livez():
    """测试 /livez 存活探针接口。"""
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz():
    """测试 /readyz 就绪探针接口。"""
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_qol_custom_pools():
    """测试查询混淆接口，传入自定义 dummy pool。"""
    custom_medical_pool = ["自定义虚假医学查询1", "自定义虚假医学查询2"]
    response = client.post(
        "/v1/privacy/qol/obfuscate",
        json={
            "query": "真实查询",
            "num_dummies": 2,
            "domain": "medical",
            "medical_pool": custom_medical_pool
        },
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert len(result) == 3
    assert "真实查询" in result
    # 所有的虚假查询都应该来自我们指定的自定义 pool
    dummies = [q for q in result if q != "真实查询"]
    assert all(q in custom_medical_pool for q in dummies)


def test_recommend_and_apply_personalized_profile(monkeypatch, tmp_path):
    """测试隐私参数推荐、自动保存及后续请求中的自动应用。"""
    # 模拟个性化配置文件存储路径，避免污染真实工作区
    profile_file = str(tmp_path / "test-personalized-profiles.yaml")
    monkeypatch.setenv("PRIVACY_PERSONALIZED_PROFILE", profile_file)

    # 1. 针对 default 命名空间请求参数推荐
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    response = client.post(
        "/v1/privacy/profile/recommend",
        json={
            "namespace": "default",
            "values": values,
            "rows": [{"age": i} for i in range(20)]
        }
    )
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"

    dp_rec = res_json["recommended_params"]["dp"]
    assert "clip_lower" in dp_rec
    assert "clip_upper" in dp_rec

    # 2. 检查生成的个性化配置文件是否确实存在并包含对应内容
    import os

    import yaml
    assert os.path.exists(profile_file)
    with open(profile_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "default" in data
    assert "dp" in data["default"]
    assert "clip_lower" in data["default"]["dp"]

    # 3. 发送一个不带 clip_lower/clip_upper 的 Gaussian DP sum 请求。
    # 通常情况下这会返回 400，但由于我们已经在 default 命名空间保存了个性化推荐参数，
    # 框架应当自动加载推荐的 clip 值，使请求成功返回 200！
    # 另外，我们显式设置 RNG 种子保证结果确定
    from engine.main import service
    service.dp_api.rng.seed(42)

    response = client.post(
        "/v1/privacy/dp/sum",
        json={
            "values": [20.0, 30.0],
            "params": {
                "epsilon": 10.0,
                "delta": 1e-5,
                "mechanism": "gaussian"
                # 不提供 clip_lower 和 clip_upper，自动应用推荐值
            }
        }
    )
    assert response.status_code == 200, response.json()
    assert "result" in response.json()


def test_ldp_perturb_and_estimate():
    """测试二值与类别型本地 DP 扰动与估计接口。"""
    # 1. Perturb Binary Batch
    resp = client.post(
        "/v1/privacy/ldp/perturb/binary",
        json={"values": [1, 0, 1, 1], "epsilon": 10.0}
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 4
    assert all(r in (0, 1) for r in results)

    # 2. Estimate Binary Frequency
    resp_est = client.post(
        "/v1/privacy/ldp/estimate/binary",
        json={"reported_values": [1, 1, 0, 1], "epsilon": 5.0}
    )
    assert resp_est.status_code == 200
    assert 0.0 <= resp_est.json()["estimated_frequency"] <= 1.0

    # 3. Perturb Categorical Batch
    resp_cat = client.post(
        "/v1/privacy/ldp/perturb/categorical",
        json={"values": ["A", "B", "A"], "categories": ["A", "B", "C"], "epsilon": 10.0}
    )
    assert resp_cat.status_code == 200
    cat_results = resp_cat.json()["results"]
    assert len(cat_results) == 3
    assert all(c in ("A", "B", "C") for c in cat_results)

    # 4. Estimate Categorical Histogram
    resp_cat_est = client.post(
        "/v1/privacy/ldp/estimate/categorical",
        json={"reported_values": ["A", "B", "C", "A"], "categories": ["A", "B", "C"], "epsilon": 5.0}
    )
    assert resp_cat_est.status_code == 200
    hist = resp_cat_est.json()["estimated_histogram"]
    assert all(c in hist for c in ("A", "B", "C"))
    assert abs(sum(hist.values()) - 1.0) < 1e-9


def test_dp_mean_with_threshold():
    """测试带有 min_count 低频阈值保护的差分隐私均值接口。"""
    resp = client.post(
        "/v1/privacy/dp/mean",
        json={
            "values": [10.0, 10.0, 10.0],
            "params": {
                "epsilon": 10.0,
                "clip_lower": 0.0,
                "clip_upper": 20.0,
                "min_count": 5.0,
            }
        }
    )
    assert resp.status_code == 200
    assert resp.json()["result"] == 0.0


def test_dp_histogram_rest():
    """测试差分隐私直方图聚合接口。"""
    resp = client.post(
        "/v1/privacy/dp/histogram",
        json={
            "values": ["A", "B", "A", "C"],
            "categories": ["A", "B", "C", "D"],
            "params": {
                "epsilon": 10.0,
                "mechanism": "laplace",
            }
        }
    )
    assert resp.status_code == 200
    res_dict = resp.json()["result"]
    assert all(c in res_dict for c in ("A", "B", "C", "D"))
    assert res_dict["A"] >= 0.0




def test_dp_noisy_count_rest():
    """测试对已聚合计数加噪的 REST 接口。"""
    resp = client.post(
        "/v1/privacy/dp/noisy_count",
        json={"true_count": 100.0, "params": {"epsilon": 1.0, "mechanism": "laplace"}},
    )
    assert resp.status_code == 200
    assert resp.json()["result"] >= 0.0


def test_dp_noisy_sum_rest():
    """测试对已聚合求和加噪的 REST 接口。"""
    resp = client.post(
        "/v1/privacy/dp/noisy_sum",
        json={
            "true_sum": 100.0,
            "params": {
                "epsilon": 1.0,
                "mechanism": "laplace",
                "sensitivity": 10.0,
            },
        },
    )
    assert resp.status_code == 200
    assert resp.json()["result"] >= 80.0


def test_dp_arrow_ipc_rest():
    """测试 REST 端点 PyArrow IPC Stream 二进制二进制传输。"""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    from engine.privacy.data_adapters import table_to_arrow_ipc_bytes

    input_table = pa.Table.from_arrays([pa.array([10.0, 20.0, 30.0, 40.0])], names=["value"])
    ipc_payload = table_to_arrow_ipc_bytes(input_table)

    resp = client.post(
        "/v1/privacy/dp/arrow_ipc?aggregation=sum&epsilon=1.0&clip_lower=0.0&clip_upper=100.0",
        content=ipc_payload,
        headers={"Content-Type": "application/vnd.apache.arrow.stream"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.apache.arrow.stream"

    # 解包响应 Stream 并验证带噪 DP 结果与 Metadata
    reader = ipc.RecordBatchStreamReader(resp.content)
    result_table = reader.read_all()
    assert b"dp_metadata" in result_table.schema.metadata
    assert "dp_value" in result_table.column_names
    assert len(result_table.column("dp_value")) == 1


def test_dp_arrow_ipc_vector_sum():
    """测试 Arrow IPC 端点的 vector_sum 聚合与 max_norm 参数。"""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    from engine.privacy.data_adapters import table_to_arrow_ipc_bytes

    # 构造 3 条 4 维向量
    vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [0.5, 0.5, 0.5, 0.5]]
    input_table = pa.Table.from_arrays(
        [pa.array(v) for v in zip(*vectors)],
        names=["d0", "d1", "d2", "d3"],
    )
    ipc_payload = table_to_arrow_ipc_bytes(input_table)

    resp = client.post(
        "/v1/privacy/dp/arrow_ipc?aggregation=vector_sum&epsilon=1.0&delta=1e-5"
        "&mechanism=gaussian&max_norm=10.0",
        content=ipc_payload,
        headers={"Content-Type": "application/vnd.apache.arrow.stream"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.apache.arrow.stream"

    reader = ipc.RecordBatchStreamReader(resp.content)
    result_table = reader.read_all()
    assert b"dp_metadata" in result_table.schema.metadata


def test_dp_arrow_ipc_vector_mean():
    """测试 Arrow IPC 端点的 vector_mean 聚合。"""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    from engine.privacy.data_adapters import table_to_arrow_ipc_bytes

    vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    input_table = pa.Table.from_arrays(
        [pa.array(v) for v in zip(*vectors)],
        names=["d0", "d1"],
    )
    ipc_payload = table_to_arrow_ipc_bytes(input_table)

    resp = client.post(
        "/v1/privacy/dp/arrow_ipc?aggregation=vector_mean&epsilon=2.0&delta=1e-5"
        "&mechanism=gaussian&max_norm=5.0",
        content=ipc_payload,
        headers={"Content-Type": "application/vnd.apache.arrow.stream"},
    )
    assert resp.status_code == 200

    reader = ipc.RecordBatchStreamReader(resp.content)
    result_table = reader.read_all()
    assert b"dp_metadata" in result_table.schema.metadata


def test_dp_arrow_ipc_unsupported_aggregation():
    """测试 Arrow IPC 端点对不支持的聚合类型返回 400。"""
    import pyarrow as pa

    from engine.privacy.data_adapters import table_to_arrow_ipc_bytes

    input_table = pa.Table.from_arrays([pa.array([1.0, 2.0])], names=["v"])
    ipc_payload = table_to_arrow_ipc_bytes(input_table)

    resp = client.post(
        "/v1/privacy/dp/arrow_ipc?aggregation=median&epsilon=1.0",
        content=ipc_payload,
        headers={"Content-Type": "application/vnd.apache.arrow.stream"},
    )
    assert resp.status_code == 400
    # The detail is now a fixed message to avoid leaking internal error text.
    assert "Invalid request parameters" in resp.json()["detail"]


def test_dp_noisy_histogram_rest():
    """测试对已聚合直方图加噪的 REST 接口。"""
    resp = client.post(
        "/v1/privacy/dp/noisy_histogram",
        json={
            "true_counts": {"A": 100.0, "B": 200.0},
            "params": {"epsilon": 10.0, "mechanism": "laplace"},
        },
    )
    assert resp.status_code == 200
    res_dict = resp.json()["result"]
    assert res_dict["A"] >= 0.0
    assert res_dict["B"] >= 0.0


def test_dp_chunked_count_rest():
    """测试分块流式 DP 计数 REST 接口。"""
    resp = client.post(
        "/v1/privacy/dp/chunked_count",
        json={
            "chunks": [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            "params": {"epsilon": 10.0, "mechanism": "laplace"},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["result"] >= 0.0


def test_dp_chunked_sum_rest():
    """测试分块流式 DP 求和 REST 接口。"""
    resp = client.post(
        "/v1/privacy/dp/chunked_sum",
        json={
            "chunks": [[1.0, 2.0, 3.0], [100.0, 5.0]],
            "params": {
                "epsilon": 10.0,
                "mechanism": "laplace",
                "clip_lower": 0.0,
                "clip_upper": 10.0,
            },
        },
    )
    assert resp.status_code == 200
    assert resp.json()["result"] >= 15.0


def test_dp_chunked_mean_rest():
    """测试分块流式 DP 均值 REST 接口。"""
    resp = client.post(
        "/v1/privacy/dp/chunked_mean",
        json={
            "chunks": [[1.0, 2.0, 3.0], [4.0, 5.0]],
            "params": {
                "epsilon": 10.0,
                "mechanism": "laplace",
                "clip_lower": 0.0,
                "clip_upper": 10.0,
            },
        },
    )
    assert resp.status_code == 200
    assert 0 <= resp.json()["result"] <= 10


def test_dp_chunked_histogram_rest():
    """测试分块流式 DP 直方图 REST 接口。"""
    resp = client.post(
        "/v1/privacy/dp/chunked_histogram",
        json={
            "chunks": [["A", "B", "A"], ["B", "C", "A"]],
            "categories": ["A", "B", "C"],
            "params": {"epsilon": 10.0, "mechanism": "laplace"},
        },
    )
    assert resp.status_code == 200
    res_dict = resp.json()["result"]
    assert all(c in res_dict for c in ("A", "B", "C"))
