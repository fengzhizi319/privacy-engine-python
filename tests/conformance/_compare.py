"""Subset comparison shared by REST/gRPC conformance replay.

Python intentionally exposes extra backward-compatible response keys. Every key
the Go reference emits must match; additional Python keys are allowed.
"""

from __future__ import annotations

import math
from typing import Any


def assert_subset(got: Any, want: Any, path: str = "", tolerance: float = 0) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{path}: got {type(got).__name__}, want object"
        for key, value in want.items():
            assert key in got, f"{path}.{key}: missing key (got {sorted(got)})"
            assert_subset(got[key], value, f"{path}.{key}", tolerance)
        return
    if isinstance(want, list):
        assert isinstance(got, list), f"{path}: got {type(got).__name__}, want array"
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for index, item in enumerate(want):
            assert_subset(got[index], item, f"{path}[{index}]", tolerance)
        return
    if isinstance(want, bool):
        assert got is want, f"{path}: got {got!r}, want {want!r}"
        return
    if isinstance(want, (int, float)):
        assert isinstance(got, (int, float)) and not isinstance(got, bool), (
            f"{path}: got {type(got).__name__}, want number"
        )
        if tolerance > 0:
            assert math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tolerance), (
                f"{path}: got {got}, want {want} (tolerance {tolerance})"
            )
        else:
            assert float(got) == float(want), f"{path}: got {got}, want {want}"
        return
    assert got == want, f"{path}: got {got!r}, want {want!r}"
