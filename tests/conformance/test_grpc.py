"""Replay the shared gRPC conformance fixtures against the Python engine."""

from __future__ import annotations

from typing import Any

import pytest
from google.protobuf import json_format

from engine import privacy_pb2
from engine.grpc_server import PrivacyServicer

from ._compare import assert_subset

_REQUEST_TYPES = {
    "Mask": privacy_pb2.MaskRequest,
    "MaskRecord": privacy_pb2.MaskRecordRequest,
    "Hash": privacy_pb2.HashRequest,
}


def test_grpc_conformance(grpc_conformance_case: dict[str, Any]) -> None:
    case = grpc_conformance_case
    rpc = case["request"]["rpc"]
    request_type = _REQUEST_TYPES.get(rpc)
    assert request_type is not None, f"unsupported rpc {rpc!r}"

    request = json_format.ParseDict(case["request"]["message"], request_type())
    servicer = PrivacyServicer()

    if case["kind"] == "error":
        with pytest.raises(Exception) as excinfo:
            getattr(servicer, rpc)(request, None)
        assert case["expect"].get("error_contains", "") in str(excinfo.value)
        return

    response = getattr(servicer, rpc)(request, None)
    got = json_format.MessageToDict(response, preserving_proto_field_name=True)
    assert_subset(got, case["expect"]["json"], case["id"], case.get("tolerance", 0))
