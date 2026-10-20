"""动态分类分级 gRPC 方法、环境变量与 Prometheus 指标单元测试。"""

import os
from unittest.mock import MagicMock

import pytest

from engine import privacy_pb2
from engine.dynclassification.profile_loader import ProfileLoader
from engine.grpc_server import PrivacyServicer
from engine.observability.metrics import (
    DYNCLASSIFICATION_ENGINE_LOAD_DURATION,
    DYNCLASSIFICATION_OPERATOR_CALLS_TOTAL,
    DYNCLASSIFICATION_RULE_HITS_TOTAL,
)


def test_dynclassification_grpc_method():
    """测试 gRPC DynClassify 服务方法能力。"""
    servicer = PrivacyServicer()
    request = privacy_pb2.DynClassificationRequest(
        field_name="user_id_card",
        field_value="110101199003072375",
        domain="general-pii",
    )
    context = MagicMock()
    response = servicer.DynClassify(request, context)

    assert response.max_level == "L3"
    assert len(response.tags) > 0
    assert response.tags[0].category == "PERSONAL_BASIC"
    assert response.engine_layer == "L1_RULE"


def test_dynclassification_env_vars_configuration(tmp_path, monkeypatch):
    """测试 PRIVACY_DYNCLASSIFICATION_* 环境变量读入。"""
    monkeypatch.setenv("PRIVACY_DYNCLASSIFICATION_RULES_DIR", str(tmp_path))
    monkeypatch.setenv("PRIVACY_DYNCLASSIFICATION_HOT_RELOAD", "false")
    monkeypatch.setenv("PRIVACY_DYNCLASSIFICATION_RELOAD_INTERVAL", "120")

    loader = ProfileLoader()
    assert loader.rules_dir == tmp_path
    assert loader.hot_reload_enabled is False
    assert loader.reload_interval_seconds == 120.0
