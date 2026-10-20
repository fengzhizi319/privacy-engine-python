"""Shared conformance fixtures for the Python engine.

The fixtures live at the repository root in ``contracts/conformance`` and are
consumed by both the Go reference tests and this suite. Never duplicate or
rewrite expectations here: Go is the source of truth for shared behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "contracts" / "conformance").is_dir():
            return parent
    raise RuntimeError("contracts/conformance not found; run from the repository checkout")


CONFORMANCE_DIR = repo_root() / "contracts" / "conformance"


def load_cases(suite: str = "primitives") -> list[dict[str, Any]]:
    cases = []
    for path in sorted((CONFORMANCE_DIR / suite).glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            case = json.load(fh)
        case["_path"] = str(path)
        cases.append(case)
    return cases


CASES = load_cases()


@pytest.fixture(params=CASES, ids=[c["id"] for c in CASES])
def conformance_case(request: pytest.FixtureRequest) -> dict[str, Any]:
    return request.param


REST_CASES = load_cases("rest")
GRPC_CASES = load_cases("grpc")


@pytest.fixture(params=REST_CASES, ids=[c["id"] for c in REST_CASES])
def rest_conformance_case(request: pytest.FixtureRequest) -> dict[str, Any]:
    return request.param


@pytest.fixture(params=GRPC_CASES, ids=[c["id"] for c in GRPC_CASES])
def grpc_conformance_case(request: pytest.FixtureRequest) -> dict[str, Any]:
    return request.param
