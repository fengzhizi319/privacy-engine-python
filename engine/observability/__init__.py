"""Observability layer for engine.

提供结构化日志、Prometheus metrics、可选 OpenTelemetry tracing 以及 REST/gRPC 中间件。
Provides structured logging, Prometheus metrics, optional OpenTelemetry tracing,
and REST/gRPC middleware.
"""

from .context import RequestContext, get_request_context, set_request_context
from .logging_config import configure_logging, get_logger
from .metrics import (
    AUTH_DENIALS_TOTAL,
    AUTH_DURATION,
    BUDGET_REMAINING,
    CLASSIFICATION_COMPOSITE_HITS_TOTAL,
    CLASSIFICATION_DURATION,
    CLASSIFICATION_JOBS_DURATION,
    CLASSIFICATION_JOBS_TOTAL,
    CLASSIFICATION_RULE_HITS_TOTAL,
    CLASSIFICATION_TOTAL,
    DP_DURATION,
    DP_QUERIES_TOTAL,
    KANO_DURATION,
    MASKING_DURATION,
    QOL_DURATION,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    make_asgi_app,
    observe_duration,
)
from .tracing import get_tracer, init_tracing, start_span

__all__ = [
    "AUTH_DENIALS_TOTAL",
    "AUTH_DURATION",
    "BUDGET_REMAINING",
    "CLASSIFICATION_COMPOSITE_HITS_TOTAL",
    "CLASSIFICATION_DURATION",
    "CLASSIFICATION_JOBS_DURATION",
    "CLASSIFICATION_JOBS_TOTAL",
    "CLASSIFICATION_RULE_HITS_TOTAL",
    "CLASSIFICATION_TOTAL",
    "DP_DURATION",
    "DP_QUERIES_TOTAL",
    "KANO_DURATION",
    "MASKING_DURATION",
    "QOL_DURATION",
    "REQUESTS_TOTAL",
    "REQUEST_DURATION",
    "RequestContext",
    "configure_logging",
    "get_logger",
    "get_request_context",
    "get_tracer",
    "init_tracing",
    "make_asgi_app",
    "observe_duration",
    "set_request_context",
    "start_span",
]
