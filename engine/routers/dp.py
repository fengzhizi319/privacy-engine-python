"""差分隐私路由（dp/* 聚合、加噪、分块流式、向量、groupby 与 arrow_ipc）。"""


from typing import Any

from fastapi import APIRouter, Request

from ..deps import SECURITY_DEPS, handle_request_exception, service
from ..schemas import (
    DPAdaptiveClipRequest,
    DPAggregateRequest,
    DPChunkedCountRequest,
    DPChunkedHistogramRequest,
    DPChunkedMeanRequest,
    DPChunkedSumRequest,
    DPGroupByRequest,
    DPHistogramRequest,
    DPNoisyCountRequest,
    DPNoisyHistogramRequest,
    DPNoisyMeanRequest,
    DPNoisySumRequest,
    DPRequest,
    DPVectorSumRequest,
)
from ..security.auth import require_permission

router = APIRouter()


@router.post("/v1/privacy/dp/count", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_count(req: DPRequest):
    """差分隐私计数接口。

    当参数校验或预算不足时，捕获异常并返回 400 Bad Request。

    Args:
        req: DPRequest 请求体。

    Returns:
        {"result": <带噪声计数值>}

    Raises:
        HTTPException: 参数非法或预算耗尽时返回 400。
    """
    try:
        return {"result": service.dp_count(req.values, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/sum", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_sum(req: DPRequest):
    """差分隐私求和接口。

    Args:
        req: DPRequest 请求体。

    Returns:
        {"result": <带噪声求和值>}

    Raises:
        HTTPException: 参数非法或预算耗尽时返回 400。
    """
    try:
        return {"result": service.dp_sum(req.values, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/mean", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_mean(req: DPRequest):
    """差分隐私均值接口。

    Args:
        req: DPRequest 请求体。

    Returns:
        {"result": <带噪声均值>}

    Raises:
        HTTPException: 参数非法或预算耗尽时返回 400。
    """
    try:
        return {"result": service.dp_mean(req.values, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/histogram", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_histogram(req: DPHistogramRequest):
    """差分隐私直方图聚合接口。"""
    try:
        return {"result": service.dp_histogram(req.values, req.categories, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/noisy_count", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_noisy_count(req: DPNoisyCountRequest):
    """对已聚合计数注入 DP 噪声。"""
    try:
        return {"result": service.dp_noisy_count(req.true_count, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/noisy_sum", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_noisy_sum(req: DPNoisySumRequest):
    """对已聚合求和注入 DP 噪声。"""
    try:
        return {"result": service.dp_noisy_sum(req.true_sum, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/noisy_mean", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_noisy_mean(req: DPNoisyMeanRequest):
    """对已聚合 sum/count 注入 DP 噪声后得到均值。"""
    try:
        return {"result": service.dp_noisy_mean(req.true_sum, req.true_count, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/noisy_histogram", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_noisy_histogram(req: DPNoisyHistogramRequest):
    """对已聚合直方图计数注入 DP 噪声。"""
    try:
        return {"result": service.dp_noisy_histogram(req.true_counts, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/aggregate", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_aggregate(req: DPAggregateRequest):
    """表格级原位 DP 聚合接口。"""
    try:
        import pandas as pd

        df = pd.DataFrame(req.rows)
        return {"result": service.dp_aggregate(df, req.specs, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/vector_sum", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_vector_sum(req: DPVectorSumRequest):
    """高维向量 / 梯度 $L_2$ 范数截断加噪接口。"""
    try:
        import numpy as np

        vectors = np.array(req.vectors)
        res = service.dp_vector_sum(vectors, req.params)
        if hasattr(res, "value"):
            return {"result": list(res.value), "details": str(res)}
        return {"result": list(res)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/vector_mean", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_vector_mean(req: DPVectorSumRequest):
    """高维向量 DP 均值：L2 范数截断 + 各向同性加噪 + noisy_count 归一化。"""
    try:
        import numpy as np

        vectors = np.array(req.vectors)
        res = service.dp_vector_mean(vectors, req.params)
        if hasattr(res, "value"):
            return {"result": list(res.value), "details": str(res)}
        return {"result": list(res)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/adaptive_clip", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_adaptive_clip(req: DPAdaptiveClipRequest):
    """差分隐私自适应二分搜索估计上下界接口。"""
    try:
        lower, upper = service.dp_adaptive_clip(req.values, req.params)
        return {"clip_lower": lower, "clip_upper": upper}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/groupby", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_groupby(req: DPGroupByRequest):
    """Tau-Thresholding 差分隐私 SQL Group-By 接口。"""
    try:
        import pandas as pd

        df = pd.DataFrame(req.rows)
        return {"result": service.dp_groupby(df, req.group_col, req.target_col, req.agg, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/chunked_count", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_chunked_count(req: DPChunkedCountRequest):
    """分块流式差分隐私计数。"""
    try:
        return {"result": service.dp_chunked_count(req.chunks, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/chunked_sum", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_chunked_sum(req: DPChunkedSumRequest):
    """分块流式差分隐私求和。"""
    try:
        return {"result": service.dp_chunked_sum(req.chunks, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/chunked_mean", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_chunked_mean(req: DPChunkedMeanRequest):
    """分块流式差分隐私均值。"""
    try:
        return {"result": service.dp_chunked_mean(req.chunks, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/dp/chunked_histogram", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def dp_chunked_histogram(req: DPChunkedHistogramRequest):
    """分块流式差分隐私直方图计数。"""
    try:
        return {"result": service.dp_chunked_histogram(req.chunks, req.categories, req.params)}
    except Exception as e:
        handle_request_exception(e)


@router.post(
    "/v1/privacy/dp/arrow_ipc",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")],
)
async def dp_arrow_ipc(
    request: Request,
    aggregation: str = "count",
    epsilon: float = 1.0,
    delta: float = 0.0,
    mechanism: str = "laplace",
    clip_lower: float | None = None,
    clip_upper: float | None = None,
    max_norm: float | None = None,
    column: str | None = None,
):
    """高效二进制 REST 端点：接收 application/vnd.apache.arrow.stream 字节载荷并返回带 DP Metadata 的 Arrow IPC Stream。

    跳过 JSON 序列化/反序列化开销，直接通过零拷贝 PyArrow 二进制 Stream 进行交互。

    Query Parameters:
        aggregation: 聚合类型，支持 count / sum / mean / vector_sum / vector_mean。
        epsilon: 隐私预算 epsilon。
        delta: 隐私预算 delta。
        mechanism: 噪声机制（"laplace" 或 "gaussian"）。
        clip_lower: 数值截断下界（sum / mean 必填）。
        clip_upper: 数值截断上界（sum / mean 必填；vector_sum / vector_mean 用作 max_norm 回退）。
        max_norm: 向量 L2 范数截断阈值（vector_sum / vector_mean 专用，优先于 clip_upper）。
        column: Arrow Table 中的目标列名（可选，默认取第一列）。
    """
    try:
        from fastapi.responses import Response

        from ..privacy.data_adapters import parse_arrow_ipc_bytes, table_to_arrow_ipc_bytes
        from ..privacy.dp import AggregationType, DPResult, compute_confidence_interval

        # Step 1: Read raw Arrow IPC Stream bytes from request body
        body_bytes = await request.body()
        arr = parse_arrow_ipc_bytes(body_bytes, column=column)

        # Step 2: Dispatch to the corresponding DPApi method based on aggregation type
        dp_res: Any = None
        if aggregation == AggregationType.COUNT:
            dp_res = service.dp_api.count(
                arr, epsilon=epsilon, delta=delta, mechanism=mechanism,
                return_details=True,
            )
        elif aggregation == AggregationType.SUM:
            dp_res = service.dp_api.sum(
                arr, epsilon=epsilon, delta=delta, mechanism=mechanism,
                clip_lower=clip_lower, clip_upper=clip_upper,
                return_details=True,
            )
        elif aggregation == AggregationType.MEAN:
            dp_res = service.dp_api.mean(
                arr, epsilon=epsilon, delta=delta, mechanism=mechanism,
                clip_lower=clip_lower, clip_upper=clip_upper,
                return_details=True,
            )
        elif aggregation == AggregationType.VECTOR_SUM:
            # max_norm explicitly provided takes priority; fall back to clip_upper for backward compatibility
            resolved_norm = max_norm if max_norm is not None else (clip_upper or 1.0)
            dp_res = service.dp_api.vector_sum(
                arr, max_norm=resolved_norm,
                epsilon=epsilon, delta=delta, mechanism=mechanism,
                return_details=True,
            )
        elif aggregation == AggregationType.VECTOR_MEAN:
            resolved_norm = max_norm if max_norm is not None else (clip_upper or 1.0)
            dp_res = service.dp_api.vector_mean(
                arr, max_norm=resolved_norm,
                epsilon=epsilon, delta=delta, mechanism=mechanism,
                return_details=True,
            )
        else:
            raise ValueError(
                f"Unsupported aggregation '{aggregation}'. "
                "Supported: count, sum, mean, vector_sum, vector_mean"
            )

        from ..privacy.dp import calibrate_analytic_gaussian

        # Step 3: Ensure dp_res is a DPResult (all branches above return DPResult via return_details=True)
        if isinstance(dp_res, (int, float)):
            # Scalar fallback: compute a valid noise scale from the aggregation parameters
            # rather than reporting a zero-scale (non-DP) CI.
            if aggregation == AggregationType.COUNT:
                sensitivity = 1.0
            elif aggregation in (AggregationType.SUM, AggregationType.MEAN):
                if clip_lower is None or clip_upper is None:
                    raise ValueError(f"clip_lower and clip_upper are required for {aggregation}")
                sensitivity = clip_upper - clip_lower
            elif aggregation in (AggregationType.VECTOR_SUM, AggregationType.VECTOR_MEAN):
                sensitivity = resolved_norm
            else:
                sensitivity = 1.0

            noise_scale = (
                sensitivity / epsilon
                if mechanism == "laplace"
                else calibrate_analytic_gaussian(epsilon, delta, sensitivity)
            )
            dp_res = DPResult(
                value=dp_res,
                noise_mechanism=mechanism,
                noise_scale=noise_scale,
                epsilon_spent=epsilon,
                delta_spent=delta,
                confidence_interval=compute_confidence_interval(
                    float(dp_res), noise_scale, mechanism, 0.95
                ),
            )

        # Step 4: Export DPResult to Arrow Table with embedded DP metadata, then serialize to IPC bytes
        table = dp_res.to_arrow()
        ipc_bytes = table_to_arrow_ipc_bytes(table)
        return Response(content=ipc_bytes, media_type="application/vnd.apache.arrow.stream")
    except Exception as e:
        handle_request_exception(e)
