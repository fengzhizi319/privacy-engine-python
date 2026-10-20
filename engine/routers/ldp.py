"""本地差分隐私路由（ldp/randomized_response|orr|perturb/*|estimate/*）。

路径与响应键与 Go ``internal/rest/routes.go`` 对齐：
- ``/ldp/randomized_response``、``/ldp/orr`` 响应 ``{result}``；
- ``/ldp/perturb/*`` 响应 ``{result}``（并存历史键 ``results``）；
- ``/ldp/estimate/binary`` 响应 ``{frequency}``（并存 ``estimated_frequency``）；
- ``/ldp/estimate/categorical`` 响应 ``{histogram}``（并存 ``estimated_histogram``）。
"""

from fastapi import APIRouter, HTTPException

from ..deps import SECURITY_DEPS, handle_request_exception, service
from ..schemas import (
    LdpEstimateBinaryRequest,
    LdpEstimateCategoricalRequest,
    LdpOrrRequest,
    LdpPerturbBinaryRequest,
    LdpPerturbCategoricalRequest,
    LdpRandomizedResponseRequest,
)
from ..security.auth import require_permission

router = APIRouter()


@router.post(
    "/v1/privacy/ldp/randomized_response",
    dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")],
)
def ldp_randomized_response(req: LdpRandomizedResponseRequest):
    """单值二值随机响应（Go ``RandomizedResponse`` 对齐）。"""
    try:
        return {"result": service.randomized_response(req.value, req.epsilon)}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/ldp/orr", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def ldp_orr(req: LdpOrrRequest):
    """单值 O-RR 响应（Go ``ORRResponse`` 对齐）。"""
    try:
        if req.value >= req.domain_size:
            raise HTTPException(status_code=400, detail="类别不属于声明域")
        result = service.orr_response(req.value, req.epsilon, req.domain_size)
        if result < 0:
            raise HTTPException(status_code=400, detail="类别不属于声明域")
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/ldp/perturb/binary", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def ldp_perturb_binary(req: LdpPerturbBinaryRequest):
    """二值本地 DP 扰动接口。"""
    try:
        results = service.perturb_binary_batch(req.values, req.epsilon)
        return {"result": results, "results": results}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/ldp/perturb/categorical", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def ldp_perturb_categorical(req: LdpPerturbCategoricalRequest):
    """类别型本地 DP 扰动接口。"""
    try:
        results = service.perturb_categorical_batch(req.values, req.categories, req.epsilon)
        return {"result": results, "results": results}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/ldp/estimate/binary", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def ldp_estimate_binary(req: LdpEstimateBinaryRequest):
    """二值本地 DP 估计接口。"""
    try:
        frequency = service.estimate_binary_frequency(req.reported_values, req.epsilon)
        return {"frequency": frequency, "estimated_frequency": frequency}
    except Exception as e:
        handle_request_exception(e)


@router.post("/v1/privacy/ldp/estimate/categorical", dependencies=[*SECURITY_DEPS, require_permission("privacy:dp")])
def ldp_estimate_categorical(req: LdpEstimateCategoricalRequest):
    """类别型本地 DP 估计接口。"""
    try:
        histogram = service.estimate_categorical_histogram(
            req.reported_values, req.categories, req.epsilon
        )
        return {"histogram": histogram, "estimated_histogram": histogram}
    except Exception as e:
        handle_request_exception(e)
