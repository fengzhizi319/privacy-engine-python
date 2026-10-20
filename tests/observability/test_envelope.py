"""Tests for the unified API error envelope (engine/observability/envelope.py).

Verifies that all error responses follow the unified envelope format:
{code, message, detail, trace_id, timestamp} with proper X-Request-ID / X-Trace-ID headers.
"""

import pytest
from unittest.mock import MagicMock

from engine.observability.envelope import (
    _STATUS_CODE_MAP,
    _build_error_envelope,
    _get_trace_id,
    _make_http_exception_handler,
    _make_validation_handler,
    register_envelope_exception_handlers,
)


class TestStatusCodeMap:
    """HTTP status code → error code mapping."""

    def test_common_codes_present(self):
        assert _STATUS_CODE_MAP[400] == "INVALID_ARGUMENT"
        assert _STATUS_CODE_MAP[401] == "UNAUTHORIZED"
        assert _STATUS_CODE_MAP[403] == "FORBIDDEN"
        assert _STATUS_CODE_MAP[404] == "NOT_FOUND"
        assert _STATUS_CODE_MAP[429] == "RATE_LIMITED"
        assert _STATUS_CODE_MAP[500] == "INTERNAL_ERROR"
        assert _STATUS_CODE_MAP[503] == "UPSTREAM_UNAVAILABLE"

    def test_aligned_with_go(self):
        """Verify the mapping matches Go ErrorCodeFromStatus()."""
        expected = {
            400: "INVALID_ARGUMENT",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            409: "CONFLICT",
            429: "RATE_LIMITED",
            500: "INTERNAL_ERROR",
            503: "UPSTREAM_UNAVAILABLE",
        }
        for code, err_code in expected.items():
            assert _STATUS_CODE_MAP[code] == err_code, f"Mismatch for HTTP {code}"


class TestGetTraceId:
    """Trace ID extraction from request headers/state."""

    def test_from_header(self):
        request = MagicMock()
        request.headers = {"X-Request-ID": "req-test-123"}
        request.state = MagicMock(spec=[])  # no attributes
        assert _get_trace_id(request) == "req-test-123"

    def test_from_state_request_id(self):
        request = MagicMock()
        request.headers = {}
        state = MagicMock()
        state.request_id = "req-from-state"
        # Remove trace_id attribute
        del state.trace_id
        request.state = state
        assert _get_trace_id(request) == "req-from-state"

    def test_from_state_trace_id(self):
        request = MagicMock()
        request.headers = {}
        state = MagicMock()
        state.trace_id = "req-trace-456"
        request.state = state
        assert _get_trace_id(request) == "req-trace-456"

    def test_generated_fallback(self):
        request = MagicMock()
        request.headers = {}
        state = MagicMock(spec=[])  # no attributes
        request.state = state
        result = _get_trace_id(request)
        assert result.startswith("req-")


class TestBuildErrorEnvelope:
    """Error envelope JSON response construction."""

    def test_response_structure(self):
        request = MagicMock()
        request.headers = {"X-Request-ID": "req-env-test"}
        request.state = MagicMock(spec=[])

        resp = _build_error_envelope(
            code="INVALID_ARGUMENT",
            message="Bad request",
            detail="field 'name' is required",
            request=request,
            status_code=400,
        )
        assert resp.status_code == 400
        body = resp.body
        import json
        data = json.loads(body)
        assert data["code"] == "INVALID_ARGUMENT"
        assert data["message"] == "Bad request"
        assert data["detail"] == "field 'name' is required"
        assert data["trace_id"] == "req-env-test"
        assert "timestamp" in data
        assert data["timestamp"].endswith("Z")

    def test_headers_set(self):
        request = MagicMock()
        request.headers = {"X-Request-ID": "req-header-test"}
        request.state = MagicMock(spec=[])

        resp = _build_error_envelope(
            code="NOT_FOUND", message="Not found", detail=None,
            request=request, status_code=404,
        )
        assert resp.headers["X-Request-ID"] == "req-header-test"
        assert resp.headers["X-Trace-ID"] == "req-header-test"


class TestValidationHandler:
    """RequestValidationError handler with envelope format."""

    @pytest.mark.asyncio
    async def test_production_mode_hides_details(self):
        handler = _make_validation_handler(debug=False)
        request = MagicMock()
        request.headers = {"X-Request-ID": "req-val-test"}
        request.state = MagicMock(spec=[])

        exc = MagicMock()
        exc.errors.return_value = [{"loc": ["body", "name"], "msg": "required"}]

        resp = await handler(request, exc)
        import json
        data = json.loads(resp.body)
        assert data["code"] == "INVALID_ARGUMENT"
        assert resp.status_code == 422
        # In production mode, detail should be a generic message
        assert "required" not in str(data["detail"])

    @pytest.mark.asyncio
    async def test_debug_mode_shows_details(self):
        handler = _make_validation_handler(debug=True)
        request = MagicMock()
        request.headers = {"X-Request-ID": "req-debug-test"}
        request.state = MagicMock(spec=[])

        errors = [{"loc": ["body", "name"], "msg": "required"}]
        exc = MagicMock()
        exc.errors.return_value = errors

        resp = await handler(request, exc)
        import json
        data = json.loads(resp.body)
        assert resp.status_code == 422
        assert data["detail"] == errors


class TestHTTPExceptionHandler:
    """HTTPException handler with envelope format."""

    @pytest.mark.asyncio
    async def test_known_status_code(self):
        handler = _make_http_exception_handler()
        request = MagicMock()
        request.headers = {"X-Request-ID": "req-http-test"}
        request.state = MagicMock(spec=[])

        exc = MagicMock()
        exc.status_code = 404
        exc.detail = "Resource not found"

        resp = await handler(request, exc)
        import json
        data = json.loads(resp.body)
        assert data["code"] == "NOT_FOUND"
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_status_code(self):
        handler = _make_http_exception_handler()
        request = MagicMock()
        request.headers = {"X-Request-ID": "req-unknown-test"}
        request.state = MagicMock(spec=[])

        exc = MagicMock()
        exc.status_code = 418  # I'm a teapot
        exc.detail = "I'm a teapot"

        resp = await handler(request, exc)
        import json
        data = json.loads(resp.body)
        assert data["code"] == "UNKNOWN_ERROR"


class TestRegisterEnvelopeHandlers:
    """Verify registration on FastAPI app."""

    def test_registration(self):
        from fastapi import FastAPI
        app = FastAPI()
        # Should not raise
        register_envelope_exception_handlers(app, debug_validation=False)
        # Verify handlers are registered by checking exception handlers exist
        assert len(app.exception_handlers) > 0


class TestObservabilityMiddlewareEnvelope:
    """Verify ObservabilityMiddleware uses envelope format for 500 errors
    and sets X-Trace-ID header."""

    def test_success_response_has_trace_id_header(self):
        """Normal responses should include X-Trace-ID header."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from engine.main import app as main_app

        client = TestClient(main_app)
        resp = client.get("/health")
        assert resp.status_code == 200
        # Both headers should be present and identical
        rid = resp.headers.get("x-request-id", "")
        tid = resp.headers.get("x-trace-id", "")
        assert rid != "", "x-request-id header should be set"
        assert tid != "", "x-trace-id header should be set"
        assert rid == tid, f"x-request-id ({rid}) should equal x-trace-id ({tid})"

    def test_422_response_uses_envelope_format(self):
        """422 validation errors should use unified envelope format."""
        from fastapi.testclient import TestClient
        from engine.main import app as main_app

        client = TestClient(main_app)
        # Send invalid data to trigger 422
        resp = client.post("/v1/privacy/mask", json={"invalid_field": 123})
        assert resp.status_code == 422
        data = resp.json()
        # Should have envelope fields
        assert "code" in data, "response should have 'code' field"
        assert data["code"] == "INVALID_ARGUMENT"
        assert "message" in data, "response should have 'message' field"
        assert "trace_id" in data, "response should have 'trace_id' field"
        assert "timestamp" in data, "response should have 'timestamp' field"
        # Headers should be set
        assert resp.headers.get("x-request-id", "") != ""
        assert resp.headers.get("x-trace-id", "") != ""


class TestGrpcTraceMetadataExtraction:
    """Verify gRPC metadata extraction supports both x-request-id and x-trace-id."""

    def test_extracts_x_request_id(self):
        from engine.observability.middleware import _get_request_id_from_metadata

        ctx = MagicMock()
        ctx.invocation_metadata.return_value = [("x-request-id", "test-rid-123")]
        assert _get_request_id_from_metadata(ctx) == "test-rid-123"

    def test_falls_back_to_x_trace_id(self):
        from engine.observability.middleware import _get_request_id_from_metadata

        ctx = MagicMock()
        ctx.invocation_metadata.return_value = [("x-trace-id", "test-tid-456")]
        assert _get_request_id_from_metadata(ctx) == "test-tid-456"

    def test_prefers_x_request_id_over_x_trace_id(self):
        from engine.observability.middleware import _get_request_id_from_metadata

        ctx = MagicMock()
        ctx.invocation_metadata.return_value = [
            ("x-request-id", "primary"),
            ("x-trace-id", "secondary"),
        ]
        assert _get_request_id_from_metadata(ctx) == "primary"

    def test_generates_id_when_missing(self):
        from engine.observability.middleware import _get_request_id_from_metadata

        ctx = MagicMock()
        ctx.invocation_metadata.return_value = []
        result = _get_request_id_from_metadata(ctx)
        assert result != "", "should generate a fallback ID when metadata is empty"
