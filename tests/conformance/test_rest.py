"""Replay the shared REST conformance fixtures against the Python engine."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from engine.main import app

from ._compare import assert_subset

client = TestClient(app)


def test_rest_conformance(rest_conformance_case: dict[str, Any]) -> None:
    case = rest_conformance_case
    request = case["request"]
    response = client.request(
        request["method"],
        request["path"],
        json=request.get("json"),
    )

    expected = case["expect"]
    assert response.status_code == expected.get("status", 200), response.text

    if case["kind"] == "error":
        assert expected.get("error_contains", "") in response.text
        return

    assert_subset(response.json(), expected["json"], case["id"], case.get("tolerance", 0))
