"""Tests for observability: logging, metrics, request_id propagation.

验证结构化日志、Prometheus metrics、request_id 透传与认证拒绝事件计数。
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from engine.main import app
from engine.observability.logging_config import configure_logging

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_logging(monkeypatch):
    """Ensure logging is reconfigurable across tests."""
    # Force configure_logging to re-run by resetting its guard.
    import engine.observability.logging_config as lc

    lc._logging_configured = False
    yield
    lc._logging_configured = False


def test_request_id_propagation():
    """请求携带的 x-request-id 应在响应中返回。"""
    rid = "test-req-123"
    response = client.get("/health", headers={"x-request-id": rid})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == rid


def test_request_id_generated_in_go_format():
    """未携带 request id 时应生成 req-<时间戳>-<nanos>-<8hex>。"""
    response = client.get("/health")
    rid = response.headers["x-request-id"]
    assert rid.startswith("req-")
    assert len(rid) <= 128


@pytest.mark.parametrize(
    "hostile",
    ["bad id", "<script>alert(1)</script>", "rid/../etc/passwd", "a" * 129],
)
def test_hostile_request_id_is_replaced_not_echoed(hostile):
    """非法入站 request id 必须回退到自生成值，不能原样回显。"""
    response = client.get("/health", headers={"x-request-id": hostile})
    rid = response.headers["x-request-id"]
    assert rid != hostile
    assert rid.startswith("req-")


def test_request_id_sanitizer_matches_go():
    from engine.observability.middleware import _sanitize_request_id

    assert (
        _sanitize_request_id("3f2504e0-4f89-11d3-9a0c-0305e82c3301")
        == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    )
    assert _sanitize_request_id("a.b_c-d~e") == "a.b_c-d~e"
    assert _sanitize_request_id("a" * 128) == "a" * 128
    assert _sanitize_request_id("a" * 129) == ""
    assert _sanitize_request_id("rid\twith\ttab") == ""
    assert _sanitize_request_id("rid\nLine2") == ""
    assert _sanitize_request_id("rid-中文") == ""
    assert _sanitize_request_id("") == ""


def test_rest_server_span_matches_go_conventions():
    """REST span 名 = METHOD FullPath，属性与 Go TraceMiddleware 对齐。"""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import engine.observability.tracing as tracing

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = tracing._tracer
    tracing._tracer = provider.get_tracer("test")
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    try:
        response = client.get("/health", headers={"traceparent": traceparent})
        assert response.status_code == 200
    finally:
        tracing._tracer = previous

    span = next(s for s in exporter.get_finished_spans() if s.name == "GET /health")
    attributes = dict(span.attributes)
    assert attributes["span.kind"] == "server"
    assert attributes["http.request.method"] == "GET"
    assert attributes["http.route"] == "/health"
    assert span.parent is not None
    assert span.parent.span_id == int("00f067aa0ba902b7", 16)


def test_json_logging_formatter():
    """PRIVACY_LOG_FORMAT=json 应配置 JsonFormatter。"""
    from pythonjsonlogger.jsonlogger import JsonFormatter

    configure_logging(json_format=True)
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)


def test_text_logging_formatter():
    """默认文本格式应配置标准 Formatter。"""
    configure_logging(json_format=False)
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, logging.Formatter)
    assert not isinstance(handler.formatter, type(__import__("pythonjsonlogger").jsonlogger.JsonFormatter))


def test_metrics_endpoint_exists():
    """/metrics 端点应返回 Prometheus 格式数据。"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "# HELP" in response.text


def test_metrics_requires_auth_when_enabled(monkeypatch):
    """PRIVACY_AUTH_ENABLED=true 时 /metrics 需要合法 API Key。"""
    monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "true")
    monkeypatch.setenv("PRIVACY_AUTH_INTERNAL_KEYS_JSON", '{"sk-metrics":{"name":"mon","scopes":["*"]}}')
    monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "false")

    # 无凭证 → 401
    resp = client.get("/metrics")
    assert resp.status_code == 401

    # 错误凭证 → 401
    resp = client.get("/metrics", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401

    # 合法凭证 → 200
    resp = client.get("/metrics", headers={"Authorization": "Bearer sk-metrics"})
    assert resp.status_code == 200
    assert "# HELP" in resp.text


def test_request_metrics_recorded():
    """调用业务接口后 /metrics 应包含对应请求指标。"""
    response = client.post(
        "/v1/privacy/mask",
        json={"field_name": "mobile", "value": "13812345678"},
    )
    assert response.status_code == 200

    metrics = client.get("/metrics").text
    assert (
        'privshield_requests_total{endpoint="/v1/privacy/mask",protocol="http",status="200"}'
        in metrics
    )
    assert (
        'privshield_request_duration_seconds_count{endpoint="/v1/privacy/mask",protocol="http"}'
        in metrics
    )


def test_dp_metrics_recorded():
    """调用 DP 接口后应记录 privacy_dp_queries_total。"""
    response = client.post(
        "/v1/privacy/dp/count",
        json={"values": [1.0, 0.0, 1.0], "params": {"epsilon": 1.0}},
    )
    assert response.status_code == 200

    metrics = client.get("/metrics").text
    assert 'privacy_dp_queries_total{aggregation="count",mechanism="laplace"}' in metrics


def test_budget_metrics_recorded():
    """调用 budget 接口后应记录 privacy_budget_remaining。"""
    response = client.get("/v1/privacy/budget")
    assert response.status_code == 200

    metrics = client.get("/metrics").text
    assert 'privacy_budget_remaining{budget_type="epsilon",namespace="default"}' in metrics
    assert 'privacy_budget_remaining{budget_type="delta",namespace="default"}' in metrics


def test_auth_denial_metric_recorded(monkeypatch):
    """认证失败时应增加 privacy_auth_denials_total。"""
    monkeypatch.setenv("PRIVACY_AUTH_ENABLED", "true")
    monkeypatch.setenv("PRIVACY_AUTH_INTERNAL_KEYS_JSON", '{"sk":{"name":"x","scopes":["*"]}}')
    monkeypatch.setenv("PRIVACY_RATE_LIMIT_ENABLED", "false")

    response = client.post("/v1/privacy/mask", json={"field_name": "mobile", "value": "13812345678"})
    assert response.status_code == 401

    # 认证开启时抓取 /metrics 也需要携带合法凭证
    metrics = client.get("/metrics", headers={"Authorization": "Bearer sk"}).text
    assert 'privacy_auth_denials_total{reason="unauthenticated"}' in metrics


def test_unmatched_route_uses_bounded_label():
    """404 路径不得产生无界 label：统一归入 path="unmatched"。"""
    import engine.observability.metrics as metrics_module

    probe_path = "/definitely-not-a-route-7f3a9c"
    counter = metrics_module.REQUESTS_TOTAL.labels(
        protocol="http", endpoint="unmatched", status="404"
    )
    before = counter._value.get()

    response = client.get(probe_path)
    assert response.status_code == 404

    after = counter._value.get()
    assert after == before + 1

    # 原始随机路径不得出现在任何指标 label 中
    metrics = client.get("/metrics").text
    assert probe_path not in metrics


def test_traffic_metrics_recorded():
    """调用业务接口后 /metrics 应包含请求与响应流量指标。"""
    response = client.post(
        "/v1/privacy/mask",
        json={"field_name": "mobile", "value": "13812345678"},
    )
    assert response.status_code == 200

    metrics = client.get("/metrics").text
    assert (
        'privshield_traffic_bytes_total{direction="request",endpoint="/v1/privacy/mask",protocol="http"}'
        in metrics
    )
    assert (
        'privshield_traffic_bytes_total{direction="response",endpoint="/v1/privacy/mask",protocol="http"}'
        in metrics
    )


def test_traffic_request_size_positive():
    """请求体越大，request traffic 指标累计值应越大。"""
    import engine.observability.metrics as metrics_module

    counter = metrics_module.TRAFFIC_BYTES_TOTAL.labels(
        protocol="http", endpoint="/v1/privacy/mask", direction="request"
    )
    before = counter._value.get()

    client.post(
        "/v1/privacy/mask",
        json={"field_name": "mobile", "value": "13812345678"},
    )

    after = counter._value.get()
    assert after > before
