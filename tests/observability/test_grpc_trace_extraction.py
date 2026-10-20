"""Tests for gRPC trace ID extraction (engine/grpc_server.py _extract_trace_id).

Verifies that the Python gRPC server correctly extracts distributed trace IDs
from incoming gRPC metadata, ensuring HTTP → gRPC cross-protocol trace context
propagation is not broken at the Python engine boundary.

Design reference: docs/archive/unified_design.md §专项方案 2 Step 2.2
"""

import pytest
from unittest.mock import MagicMock

from engine.grpc_server import _extract_trace_id


def _mock_context(metadata: list[tuple[str, str]]) -> MagicMock:
    """Create a mock grpc.ServicerContext with the given invocation_metadata."""
    ctx = MagicMock()
    ctx.invocation_metadata.return_value = metadata
    return ctx


class TestExtractTraceId:
    """_extract_trace_id behaviour."""

    def test_extracts_x_request_id(self):
        """Should return the value of x-request-id when present."""
        ctx = _mock_context([("x-request-id", "req-12345-abc")])
        assert _extract_trace_id(ctx) == "req-12345-abc"

    def test_extracts_x_trace_id(self):
        """Should return the value of x-trace-id when x-request-id is absent."""
        ctx = _mock_context([("x-trace-id", "trace-67890-def")])
        assert _extract_trace_id(ctx) == "trace-67890-def"

    def test_returns_first_matching_key(self):
        """Should return the first matching trace key encountered in metadata order."""
        ctx = _mock_context([
            ("x-trace-id", "trace-first"),
            ("x-request-id", "req-second"),
        ])
        # Function returns first match (x-trace-id appears first)
        assert _extract_trace_id(ctx) == "trace-first"

    def test_fallback_when_no_metadata(self):
        """Should generate a fallback ID when no trace metadata is present."""
        ctx = _mock_context([])
        result = _extract_trace_id(ctx)
        assert result.startswith("req-")
        assert len(result) > 4

    def test_ignores_unrelated_metadata(self):
        """Should ignore non-trace metadata keys."""
        ctx = _mock_context([
            ("api-key", "secret-key-123"),
            ("content-type", "application/grpc"),
            ("x-request-id", "req-filtered"),
        ])
        assert _extract_trace_id(ctx) == "req-filtered"

    def test_with_bff_go_withtrace_format(self):
        """Should handle the dual-header format sent by bff-go WithTrace().

        bff-go WithTrace() appends both x-request-id and x-trace-id with the
        same value. The function should return that value.
        """
        trace_id = "req-1787554500-abc12345"
        ctx = _mock_context([
            ("x-request-id", trace_id),
            ("x-trace-id", trace_id),
        ])
        assert _extract_trace_id(ctx) == trace_id
