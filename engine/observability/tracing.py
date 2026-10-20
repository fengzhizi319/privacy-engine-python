"""Optional OpenTelemetry tracing.

OpenTelemetry 作为可选依赖；未安装或环境变量未设置时使用 NoOp tracer，零开销。
OpenTelemetry is an optional dependency. When not installed or not configured,
a NoOp tracer is used so there is no runtime overhead.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any

# Optional import: if opentelemetry-api/sdk are absent, tracing is a no-op.
try:  # pragma: no cover
    from opentelemetry import propagate, trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTEL = True
except ImportError:  # pragma: no cover
    _HAS_OTEL = False
    propagate = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]

# The OTLP exporter is a separate distribution; span creation and W3C context
# propagation still work without it (spans are simply not exported).
try:  # pragma: no cover
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    _HAS_OTLP = True
except ImportError:  # pragma: no cover
    _HAS_OTLP = False
    OTLPSpanExporter = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from collections.abc import Iterator

_logger = logging.getLogger(__name__)

# Module-level tracer instance. Initialized lazily by init_tracing().
_tracer: Any = None


def init_tracing(
    endpoint: str | None = None,
    service_name: str = "PrivShield",
) -> Any:
    """Initialize OpenTelemetry tracing if endpoint is provided and library exists.

    Args:
        endpoint: OTLP HTTP endpoint, e.g. ``http://jaeger:4318/v1/traces``.
                  Falls back to ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.
        service_name: Service name attached to every span.

    Returns:
        The configured tracer, or a no-op tracer.
    """
    global _tracer

    if not _HAS_OTEL:
        _tracer = _noop_tracer()
        return _tracer

    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        _tracer = trace.get_tracer(__name__)
        return _tracer
    if not _HAS_OTLP:
        _logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp "
            "is not installed; spans will be created but not exported"
        )
        _tracer = trace.get_tracer(__name__)
        return _tracer

    provider = TracerProvider(
        resource=Resource({"service.name": service_name or "PrivShield"})
    )
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = provider.get_tracer(service_name)
    return _tracer


def _noop_tracer() -> Any:
    """Return a no-op tracer when opentelemetry is unavailable."""
    if _HAS_OTEL:
        return trace.get_tracer(__name__)
    # Minimal duck-typing fallback for environments without opentelemetry.
    class _NoOpTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
            yield None

        def start_span(self, *args: Any, **kwargs: Any) -> Any:
            return None

    return _NoOpTracer()


def get_tracer() -> Any:
    """Return the configured tracer, initializing a no-op tracer if necessary."""
    global _tracer
    if _tracer is None:
        _tracer = _noop_tracer()
    return _tracer


def shutdown_tracing() -> None:
    """Flush and shut down the OpenTelemetry tracer provider.

    Ensures all buffered spans are exported before process exit.
    Safe to call even if tracing was never initialized.
    刷新并关闭 OpenTelemetry TracerProvider，确保所有缓冲 span 在进程退出前导出。
    即使追踪从未初始化也可安全调用。
    """
    if not _HAS_OTEL:
        return
    provider = trace.get_tracer_provider() if trace else None
    if provider and hasattr(provider, "shutdown"):
        provider.shutdown()


def context_from_carrier(carrier: dict[str, str] | None) -> Any:
    """Extract a W3C ``traceparent`` context from inbound headers/metadata.

    Returns ``None`` when tracing is unavailable or the carrier is malformed;
    callers then start a new root span instead of failing the request.
    """
    if not _HAS_OTEL or not carrier:
        return None
    try:
        return propagate.extract(carrier)
    except Exception:  # noqa: BLE001 - malformed traceparent must not break serving
        return None


def inject_context(carrier: dict[str, str]) -> dict[str, str]:
    """Inject the current W3C trace context into an outbound carrier (best effort)."""
    if _HAS_OTEL:
        try:
            propagate.inject(carrier)
        except Exception:  # noqa: BLE001 - propagation is best effort
            pass
    return carrier


@contextlib.contextmanager
def start_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    context: Any = None,
) -> Iterator[Any]:
    """Context manager to start a span with optional parent context and attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, context=context) as span:
        if span and attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span
