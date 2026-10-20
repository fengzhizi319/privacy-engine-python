"""Structured logging configuration.

支持 JSON / 文本两种格式，并自动注入 request_id、identity_name、method、path 等上下文字段。
Supports JSON or text formats and automatically injects request context fields.
"""

from __future__ import annotations

import logging
import sys

from .context import get_request_context

# Guard to make configure_logging idempotent across multiple entrypoints.
_logging_configured = False


class _ContextFilter(logging.Filter):
    """Inject request-context fields into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = get_request_context()
        record.request_id = ctx.request_id if ctx else ""
        record.identity_name = ctx.identity_name if ctx else ""
        record.method = ctx.method if ctx else ""
        record.path = ctx.path if ctx else ""
        return True


def _make_json_formatter() -> logging.Formatter:
    """Build a JSON formatter with the desired field set."""
    try:
        from pythonjsonlogger import jsonlogger
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PRIVACY_LOG_FORMAT=json requires python-json-logger. "
            "Install with: pip install python-json-logger"
        ) from exc

    # The format string lists every field we want in the JSON output.
    fmt = (
        "%(asctime)s %(levelname)s %(name)s %(message)s "
        "%(request_id)s %(identity_name)s %(method)s %(path)s "
        "%(lineno)d %(funcName)s"
    )
    formatter = jsonlogger.JsonFormatter(
        fmt,
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
        },
    )
    return formatter  # type: ignore[return-value]


def configure_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    service_name: str = "PrivShield",
) -> None:
    """Configure root logging.

    Args:
        log_level: One of DEBUG/INFO/WARNING/ERROR/CRITICAL.
        json_format: When True, emit JSON; otherwise plain text.
        service_name: Service name added as a static field (reserved for future use).
    """
    global _logging_configured
    if _logging_configured:
        return

    root = logging.getLogger()
    # Remove any existing handlers to avoid duplicate logs when reconfigured.
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())

    if json_format:
        handler.setFormatter(_make_json_formatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s "
                "request_id=%(request_id)s identity=%(identity_name)s"
            )
        )

    root.addHandler(handler)
    root.setLevel(log_level.upper())

    # Reduce noise from some third-party libraries.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name.

    All loggers share the root handler configured by ``configure_logging``.
    """
    return logging.getLogger(name)
