"""Prometheus metrics definitions and ASGI app factory.

集中定义所有 Prometheus 指标，并提供挂载到 FastAPI 的 ASGI app。
"""

from __future__ import annotations

from typing import Any

from contextlib import contextmanager
import time

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)
from prometheus_client import (
    make_asgi_app as _make_asgi_app,
)


@contextmanager
def observe_duration(histogram, **labels):
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.labels(**labels).observe(time.perf_counter() - start)

# Shared RED metrics — names/labels aligned with Go pkg/observability.REDMetrics.
RED_REQUESTS_TOTAL = Counter(
    "privshield_requests_total",
    "Total requests processed.",
    ["protocol", "endpoint", "status"],
)

RED_REQUEST_DURATION = Histogram(
    "privshield_request_duration_seconds",
    "Request latency histogram in seconds.",
    ["protocol", "endpoint"],
    # Go prometheus.DefBuckets (Python's default adds .075/.75/7.5).
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Backwards-compatible module aliases used by middleware/tests.
REQUESTS_TOTAL = RED_REQUESTS_TOTAL
REQUEST_DURATION = RED_REQUEST_DURATION

# Differential privacy query counter.
DP_QUERIES_TOTAL = Counter(
    "privacy_dp_queries_total",
    "Total number of differential privacy queries.",
    ["mechanism", "aggregation"],
)

# Remaining privacy budget per namespace.
BUDGET_REMAINING = Gauge(
    "privacy_budget_remaining",
    "Remaining privacy budget (epsilon or delta) per namespace.",
    ["namespace", "budget_type"],
)

# Cumulative consumed budget — Go privshield_budget_consumed_total parity.
BUDGET_CONSUMED_TOTAL = Counter(
    "privshield_budget_consumed_total",
    "Cumulative differential privacy budget consumed.",
    ["namespace", "mechanism"],
)

# Classification results counter (Python-native family).
CLASSIFICATION_TOTAL = Counter(
    "privacy_classification_total",
    "Total number of classification results by final level and layer.",
    ["final_level", "layer"],
)

# Classification funnel hits — Go privshield_classification_total parity.
CLASSIFICATION_GO_TOTAL = Counter(
    "privshield_classification_total",
    "Classification funnel hits by engine/level/domain.",
    ["engine", "level", "domain"],
)

# NER inference latency — Go privshield_ner_inference_seconds parity.
NER_INFERENCE_SECONDS = Histogram(
    "privshield_ner_inference_seconds",
    "NER inference latency by device and batch size.",
    ["device", "batch_size"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Inbound identifier alias hits — Go privshield_api_alias_requests_total parity.
API_ALIAS_REQUESTS_TOTAL = Counter(
    "privshield_api_alias_requests_total",
    "Total requests processed via legacy/alias identifier mapping.",
    ["alias", "canonical", "target", "module"],
)

# Identifier normalization failures — Go privshield_datasource_normalize_errors_total parity.
DATASOURCE_NORMALIZE_ERRORS_TOTAL = Counter(
    "privshield_datasource_normalize_errors_total",
    "Total identifier normalization errors and fail-closed rejections.",
    ["reason", "module"],
)

# Authentication/authorization/rate-limit denials (Python-native family).
AUTH_DENIALS_TOTAL = Counter(
    "privacy_auth_denials_total",
    "Total number of authentication/authorization/rate-limit denials.",
    ["reason"],
)

# Go pkg/auth security counters (parity names).
AUTH_FAILURES_TOTAL = Counter(
    "privshield_auth_failures_total",
    "Total authentication failures by bounded reason.",
    ["reason"],
)

AUTH_FORBIDDEN_TOTAL = Counter(
    "privshield_auth_forbidden_total",
    "Total authorization failures (insufficient scope or policy denial).",
)

AUTH_ALLOWED_TOTAL = Counter(
    "privshield_auth_allowed_total",
    "Total permissions granted to workload principals.",
    ["subject", "permission"],
)

# Request/response traffic in bytes (Python extra, aligned labels).
TRAFFIC_BYTES_TOTAL = Counter(
    "privshield_traffic_bytes_total",
    "Total request/response traffic in bytes.",
    ["protocol", "endpoint", "direction"],
)

# Masking operations counter.
MASKING_OPERATIONS_TOTAL = Counter(
    "privacy_masking_operations_total",
    "Total number of masking operations.",
    ["operation"],
)

# K-anonymity operations counter.
KANO_OPERATIONS_TOTAL = Counter(
    "privacy_kano_operations_total",
    "Total number of K-anonymity operations.",
    ["operation"],
)

# Query obfuscation operations counter.
QOL_OPERATIONS_TOTAL = Counter(
    "privacy_qol_operations_total",
    "Total number of query obfuscation operations.",
    ["domain"],
)

# Classification async jobs counter.
CLASSIFICATION_JOBS_TOTAL = Counter(
    "privacy_classification_jobs_total",
    "Total number of classification async jobs by status.",
    ["status"],
)

# Classification async job duration histogram.
CLASSIFICATION_JOBS_DURATION = Histogram(
    "privacy_classification_jobs_duration_seconds",
    "Classification async job execution latency in seconds.",
    ["status"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# Classification rule engine hit counter (Layer-1).
CLASSIFICATION_RULE_HITS_TOTAL = Counter(
    "privacy_classification_rule_hits_total",
    "Total number of Layer-1 rule engine hits by rule_id.",
    ["rule_id"],
)

# Classification NER engine invocations (Layer-2).
CLASSIFICATION_NER_TOTAL = Counter(
    "privacy_classification_ner_total",
    "Total number of Small-NER engine invocations.",
    ["status"],
)

# Classification LLM engine invocations (Layer-3).
CLASSIFICATION_LLM_TOTAL = Counter(
    "privacy_classification_llm_total",
    "Total number of LLM classifier invocations.",
    ["status"],
)

# Classification field/record/table operation duration histogram.
CLASSIFICATION_DURATION = Histogram(
    "privacy_classification_duration_seconds",
    "Classification operation latency in seconds.",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Classification composite rule hit counter.
CLASSIFICATION_COMPOSITE_HITS_TOTAL = Counter(
    "privacy_classification_composite_hits_total",
    "Total number of composite rule hits by rule_id.",
    ["rule_id"],
)

# NER engine inference duration histogram (Layer-2).
CLASSIFICATION_NER_DURATION = Histogram(
    "privacy_classification_ner_duration_seconds",
    "Small-NER engine inference latency in seconds.",
    ["engine"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# LLM engine inference duration histogram (Layer-3).
CLASSIFICATION_LLM_DURATION = Histogram(
    "privacy_classification_llm_duration_seconds",
    "LLM classifier inference latency in seconds.",
    ["engine"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# Classification LLM tokens consumed counter (Layer-3).
CLASSIFICATION_LLM_TOKENS_TOTAL = Counter(
    "privacy_classification_llm_tokens_total",
    "Total LLM tokens consumed by prompt/completion type and engine.",
    ["type", "engine"],
)

# Dynamic classification metrics (动态分类分级可观测性)
DYNCLASSIFICATION_RULE_HITS_TOTAL = Counter(
    "classification_rule_hits_total",
    "Total number of dynamic classification rule hits.",
    ["rule_id", "domain", "standard"],
)

DYNCLASSIFICATION_OPERATOR_CALLS_TOTAL = Counter(
    "classification_operator_calls_total",
    "Total number of dynamic classification operator calls.",
    ["operator", "result"],
)

DYNCLASSIFICATION_ENGINE_LOAD_DURATION = Histogram(
    "classification_engine_load_duration_seconds",
    "Dynamic classification engine load duration in seconds.",
    ["domain", "standard"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

DYNCLASSIFICATION_PROFILE_CACHE_SIZE = Gauge(
    "classification_profile_cache_size",
    "Number of cached dynamic classification engine instances.",
)

DYNCLASSIFICATION_OPERATOR_ERRORS_TOTAL = Counter(
    "classification_operator_errors_total",
    "Total number of dynamic classification operator errors.",
    ["operator", "rule_id"],
)

DYNCLASSIFICATION_OVERRIDE_SUPPRESSED_TOTAL = Counter(
    "classification_override_suppressed_total",
    "Total number of normal tags suppressed by override downgrade rules.",
    ["domain", "suppressed_rule_id"],
)

# Profile parameter resolution counter.
PROFILE_RESOLVE_TOTAL = Counter(
    "privacy_profile_resolve_total",
    "Total number of parameter resolution operations.",
    ["primitive", "status"],
)

# Data adapter extraction counter.
DATA_EXTRACTION_TOTAL = Counter(
    "privacy_data_extraction_total",
    "Total number of data extraction operations by source format.",
    ["format", "status"],
)

# ---------------------------------------------------------------------------
# Gateway metrics (P0: 网关可观测性)
# ---------------------------------------------------------------------------

# Gateway proxy request counter (Go: privshield_gateway_requests_total{node_id,status}).
GATEWAY_REQUESTS_TOTAL = Counter(
    "privshield_gateway_requests_total",
    "Total requests forwarded by the gateway.",
    ["node_id", "status"],
)

# Per-backend in-flight gauge (Go parity).
GATEWAY_BACKEND_IN_FLIGHT = Gauge(
    "privshield_gateway_backend_in_flight",
    "Current in-flight requests per backend node.",
    ["node_id", "backend_addr"],
)

# Per-backend EWMA latency gauge (Go parity).
GATEWAY_BACKEND_EWMA_LATENCY = Gauge(
    "privshield_gateway_backend_ewma_latency_seconds",
    "Exponentially weighted moving average latency per backend node.",
    ["node_id"],
)

# Gateway proxy latency histogram (Python extra).
GATEWAY_LATENCY = Histogram(
    "privacy_gateway_latency_seconds",
    "Gateway proxy latency in seconds (time spent forwarding to backend).",
    ["protocol"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# Gateway healthy backend nodes gauge (Python extra).
GATEWAY_HEALTHY_NODES = Gauge(
    "privacy_gateway_healthy_nodes",
    "Current number of healthy backend nodes in the gateway pool.",
)

# Gateway retry counter (Python extra).
GATEWAY_RETRIES_TOTAL = Counter(
    "privacy_gateway_retries_total",
    "Total number of gateway retry attempts.",
    ["protocol", "reason"],
)

# Gateway circuit breaker state per backend node (Go parity).
GATEWAY_CIRCUIT_BREAKER_STATE = Gauge(
    "privshield_gateway_circuit_breaker_state",
    "Circuit breaker state per node (0=closed, 1=half-open, 2=open).",
    ["node_id", "state"],
)

# Gateway node isolated/drained flag (Python extra).
GATEWAY_NODE_ADMIN_STATE = Gauge(
    "privacy_gateway_node_admin_state",
    "Administrative node state (0=active, 1=isolated, 2=drained).",
    ["node"],
)

# ---------------------------------------------------------------------------
# Module-level duration histograms (P1: 全局延迟指标)
# ---------------------------------------------------------------------------

# Masking operation duration histogram.
MASKING_DURATION = Histogram(
    "privacy_masking_duration_seconds",
    "Masking operation latency in seconds.",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# K-anonymity operation duration histogram.
KANO_DURATION = Histogram(
    "privacy_kano_duration_seconds",
    "K-anonymity operation latency in seconds.",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Differential privacy query duration histogram.
DP_DURATION = Histogram(
    "privacy_dp_duration_seconds",
    "Differential privacy query latency in seconds.",
    ["aggregation", "mechanism"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Query obfuscation duration histogram.
QOL_DURATION = Histogram(
    "privacy_qol_duration_seconds",
    "Query obfuscation operation latency in seconds.",
    ["domain"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Security auth/rate-limit operation duration histogram.
AUTH_DURATION = Histogram(
    "privacy_auth_duration_seconds",
    "Authentication and authorization check latency in seconds.",
    ["result"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
)


def make_asgi_app() -> Any:
    """Return the Prometheus metrics ASGI application to mount on FastAPI."""
    return _make_asgi_app()
