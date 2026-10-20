"""处理原语 gRPC 接口测试。

直接调用 PrivacyServicer 方法验证 DP、K-匿名等接口的参数解析与响应序列化。
"""

from engine import privacy_pb2
from engine.grpc_server import PrivacyServicer


def test_grpc_dp_count():
    """验证 gRPC 差分隐私计数返回非负结果。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPRequest(
        values=[1.0, 0.0, 1.0, 1.0],
        epsilon=10.0,
        mechanism="laplace",
    )
    response = servicer.DPCount(request, None)
    assert response.result >= 0


def test_grpc_dp_sum_with_clip():
    """验证 gRPC 差分隐私求和 clipping 与 Gaussian 机制。"""
    servicer = PrivacyServicer()
    servicer.service.dp_api.rng.seed(42)
    request = privacy_pb2.DPRequest(
        values=[1.0, 2.0, 3.0, 100.0],
        epsilon=10.0,
        delta=1e-5,
        mechanism="gaussian",
        clip_lower=0.0,
        clip_upper=10.0,
    )
    response = servicer.DPSum(request, None)
    assert 10 <= response.result <= 25


def test_grpc_k_anonymize_table():
    """验证 gRPC 整张表 K-匿名接口。"""
    servicer = PrivacyServicer()
    rows = [
        privacy_pb2.RecordEntry(
            fields={"age": "25", "zipcode": "100001", "gender": "M", "disease": "A"}
        ),
        privacy_pb2.RecordEntry(
            fields={"age": "26", "zipcode": "100002", "gender": "M", "disease": "B"}
        ),
        privacy_pb2.RecordEntry(
            fields={"age": "55", "zipcode": "200001", "gender": "F", "disease": "C"}
        ),
        privacy_pb2.RecordEntry(
            fields={"age": "56", "zipcode": "200002", "gender": "F", "disease": "D"}
        ),
    ]
    request = privacy_pb2.KAnonymizeTableRequest(
        rows=rows,
        qi_cols=["age", "zipcode", "gender"],
        k=2,
    )
    response = servicer.KAnonymizeTable(request, None)
    assert len(response.rows) == 4
    diseases = {r.fields["disease"] for r in response.rows}
    assert diseases == {"A", "B", "C", "D"}


def test_grpc_recommend_params():
    """验证 gRPC 隐私参数推荐接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.RecommendRequest(
        namespace="grpc-recommend-ns",
        values=[10.0, 20.0, 30.0, 40.0],
        rows=[
            privacy_pb2.RecordEntry(fields={"age": "20"}),
            privacy_pb2.RecordEntry(fields={"age": "30"}),
        ]
    )
    response = servicer.RecommendParams(request, None)
    assert response.status == "success"
    assert response.namespace == "grpc-recommend-ns"

    import json
    rec_dict = json.loads(response.recommended_params_json)
    assert "dp" in rec_dict
    assert "k_anonymity" in rec_dict
    assert rec_dict["dp"]["clip_lower"] < rec_dict["dp"]["clip_upper"]


def test_grpc_ldp_operations():
    """验证 gRPC 本地 DP 扰动与估计接口。"""
    servicer = PrivacyServicer()

    # 1. Perturb Binary Batch
    req_bin = privacy_pb2.PerturbBinaryBatchRequest(
        values=[1, 0, 1, 1],
        epsilon=10.0
    )
    resp_bin = servicer.PerturbBinaryBatch(req_bin, None)
    assert len(resp_bin.results) == 4
    assert all(r in (0, 1) for r in resp_bin.results)

    # 2. Estimate Binary Frequency
    req_est_bin = privacy_pb2.EstimateBinaryFrequencyRequest(
        reported_values=[1, 1, 0, 1],
        epsilon=5.0
    )
    resp_est_bin = servicer.EstimateBinaryFrequency(req_est_bin, None)
    assert 0.0 <= resp_est_bin.estimated_frequency <= 1.0

    # 3. Perturb Categorical Batch
    req_cat = privacy_pb2.PerturbCategoricalBatchRequest(
        values=["A", "B", "A"],
        categories=["A", "B", "C"],
        epsilon=10.0
    )
    resp_cat = servicer.PerturbCategoricalBatch(req_cat, None)
    assert len(resp_cat.results) == 3
    assert all(c in ("A", "B", "C") for c in resp_cat.results)

    # 4. Estimate Categorical Histogram
    req_est_cat = privacy_pb2.EstimateCategoricalHistogramRequest(
        reported_values=["A", "B", "C", "A"],
        categories=["A", "B", "C"],
        epsilon=5.0
    )
    resp_est_cat = servicer.EstimateCategoricalHistogram(req_est_cat, None)
    hist = resp_est_cat.estimated_histogram
    assert all(c in hist for c in ("A", "B", "C"))
    assert abs(sum(hist.values()) - 1.0) < 1e-9


def test_grpc_dp_histogram():
    """验证 gRPC 差分隐私直方图接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPHistogramRequest(
        values=["A", "B", "A", "C"],
        categories=["A", "B", "C", "D"],
        epsilon=10.0,
        mechanism="laplace",
    )
    response = servicer.DPHistogram(request, None)
    assert len(response.result) == 4
    assert all(c in response.result for c in ("A", "B", "C", "D"))
    assert response.result["A"] >= 0.0




def test_grpc_dp_noisy_count():
    """验证 gRPC 对已聚合计数加噪接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPNoisyCountRequest(
        true_count=100.0, epsilon=10.0, mechanism="laplace"
    )
    response = servicer.DPNoisyCount(request, None)
    assert response.result >= 0.0


def test_grpc_dp_noisy_sum():
    """验证 gRPC 对已聚合求和加噪接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPNoisySumRequest(
        true_sum=100.0,
        epsilon=10.0,
        mechanism="laplace",
        sensitivity=10.0,
    )
    response = servicer.DPNoisySum(request, None)
    assert response.result >= 80.0


def test_grpc_dp_noisy_mean():
    """验证 gRPC 对已聚合 sum/count 加噪得到均值接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPNoisyMeanRequest(
        true_sum=150.0,
        true_count=10.0,
        epsilon=10.0,
        mechanism="laplace",
        clip_lower=0.0,
        clip_upper=100.0,
    )
    response = servicer.DPNoisyMean(request, None)
    assert 0.0 <= response.result <= 100.0


def test_grpc_dp_noisy_histogram():
    """验证 gRPC 对已聚合直方图加噪接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPNoisyHistogramRequest(
        true_counts={"A": 100.0, "B": 200.0},
        epsilon=10.0,
        mechanism="laplace",
    )
    response = servicer.DPNoisyHistogram(request, None)
    assert "A" in response.result
    assert "B" in response.result
    assert response.result["A"] >= 0.0


def test_grpc_dp_chunked_count():
    """验证 gRPC 分块流式 DP 计数接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPChunkedCountRequest(
        chunks=[
            privacy_pb2.DoubleChunk(values=[1.0, 0.0, 1.0]),
            privacy_pb2.DoubleChunk(values=[0.0, 1.0]),
        ],
        epsilon=10.0,
        mechanism="laplace",
    )
    response = servicer.DPChunkedCount(request, None)
    assert response.result >= 0.0


def test_grpc_dp_chunked_sum():
    """验证 gRPC 分块流式 DP 求和接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPChunkedSumRequest(
        chunks=[
            privacy_pb2.DoubleChunk(values=[1.0, 2.0, 3.0]),
            privacy_pb2.DoubleChunk(values=[100.0, 5.0]),
        ],
        epsilon=10.0,
        mechanism="laplace",
        clip_lower=0.0,
        clip_upper=10.0,
    )
    response = servicer.DPChunkedSum(request, None)
    assert response.result >= 15.0


def test_grpc_dp_chunked_mean():
    """验证 gRPC 分块流式 DP 均值接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPChunkedMeanRequest(
        chunks=[
            privacy_pb2.DoubleChunk(values=[1.0, 2.0, 3.0]),
            privacy_pb2.DoubleChunk(values=[4.0, 5.0]),
        ],
        epsilon=10.0,
        mechanism="laplace",
        clip_lower=0.0,
        clip_upper=10.0,
    )
    response = servicer.DPChunkedMean(request, None)
    assert 0.0 <= response.result <= 10.0


def test_grpc_dp_chunked_histogram():
    """验证 gRPC 分块流式 DP 直方图接口。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DPChunkedHistogramRequest(
        chunks=[
            privacy_pb2.StringChunk(values=["A", "B", "A"]),
            privacy_pb2.StringChunk(values=["B", "C", "A"]),
        ],
        categories=["A", "B", "C"],
        epsilon=10.0,
        mechanism="laplace",
    )
    response = servicer.DPChunkedHistogram(request, None)
    assert all(c in response.result for c in ("A", "B", "C"))
