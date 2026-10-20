"""Replay the shared conformance fixtures against the Python engine."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from engine.medical_pipeline import rules as medical_rules
from engine.privacy import dp, kano, masking, qol
from engine.privacy.dp import LocalDPApi
from engine.privacy.kano_table import k_anonymize_table


def _args(case: dict[str, Any]) -> dict[str, Any]:
    return case["request"].get("args", {})


def _classify_icd10(code: str) -> dict[str, Any]:
    classified = medical_rules.classify_icd10_code(code)
    if classified is None:
        return {"level": "", "category": "", "hit": False}
    level, category = classified
    return {"level": level, "category": category, "hit": True}


def run_primitive(op: str, args: dict[str, Any]) -> Any:
    if op == "mask_phone":
        return masking.mask_mobile(args["value"])
    if op == "mask_id_card":
        return masking.mask_id_card(args["value"])
    if op == "mask_bank_card":
        return masking.mask_bank_card(args["value"])
    if op == "mask_name":
        return masking.mask_name(args["value"])
    if op == "mask_email":
        return masking.mask_email(args["value"])
    if op == "mask_address":
        return masking.mask_address(args["value"])
    if op == "mask_default":
        return masking.mask_default(args["value"], args.get("prefix", 3), args.get("suffix", 3))
    if op == "mask_value":
        return masking.mask_value(args["field_name"], args["value"])
    if op == "mask_record":
        return masking.mask_record(args["record"])
    if op == "guess_field_type":
        return masking.guess_field_type(args["field_name"])
    if op == "hash_hmac":
        return masking.hash_value(args["value"], args["salt"])
    if op == "fpe_encrypt_numeric":
        return masking.fpe_encrypt_numeric(args["value"], args["secret_key"])
    if op == "truncate":
        return masking.truncate(args["value"], args["keep_prefix"])
    if op == "random_date_offset":
        return masking.random_date_offset(args["date"], args["offset_days"])
    if op == "age_hierarchy":
        return kano.age_hierarchy(args["value"], args["level"])
    if op == "zipcode_hierarchy":
        return kano.zipcode_hierarchy(args["value"], args["level"])
    if op == "gender_hierarchy":
        return kano.gender_hierarchy(args["value"], args["level"])
    if op == "salary_hierarchy":
        return kano.salary_hierarchy(args["value"], args["level"])
    if op == "education_hierarchy":
        return kano.education_hierarchy(args["value"], args["level"])
    if op == "icd10_classify":
        return _classify_icd10(args["code"])
    if op == "icd10_redact":
        return medical_rules.redact_icd10_code(args["code"])
    if op == "truncate_date_to_month":
        return medical_rules.truncate_date_to_month(args["date"])
    if op == "clip_value":
        return dp.clip_value(args["value"], args["bound"])
    if op == "clip_value_range":
        return dp.clip_value_range(args["value"], args["lower"], args["upper"])
    if op == "clip_l2_norm":
        return dp.clip_l2_norm(args["vector"], args["max_norm"])
    if op == "kano_mondrian":
        result = k_anonymize_table(
            args["rows"],
            args["qi_cols"],
            k=args["k"],
            max_depth=args.get("max_depth", 10),
            return_details=True,
        )
        return {
            "records": result.value,
            "k": result.k,
            "qi_cols": result.qi_cols,
            "equivalence_classes_count": result.equivalence_classes_count,
        }
    if op == "qol_inject":
        result = qol.obfuscate_query(
            args["query"],
            num_dummies=args["num_decoys"],
            domain=args["domain"],
            return_details=True,
        )
        return {"queries": result.queries, "real_index": result.real_query_index}
    if op == "ldp_randomized_response":
        return bool(LocalDPApi().perturb_binary(1 if args["value"] else 0, args["epsilon"]))
    if op == "dp_noisy_count":
        return float(dp.DPApi().noisy_count(args["count"], args["epsilon"]))
    raise AssertionError(f"unsupported op {op!r}")


def _compare(path: str, got: Any, want: Any, tolerance: float) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{path}: got {type(got).__name__}, want object"
        assert set(got) == set(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for key in want:
            _compare(f"{path}.{key}", got[key], want[key], tolerance)
        return
    if isinstance(want, list):
        assert isinstance(got, list), f"{path}: got {type(got).__name__}, want array"
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for index, item in enumerate(want):
            _compare(f"{path}[{index}]", got[index], item, tolerance)
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


def _record_key(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _check_unordered(got: dict[str, Any], expected: dict[str, Any]) -> None:
    assert len(got["records"]) == len(expected["records"]), "record count mismatch"
    assert sorted(_record_key(r) for r in got["records"]) == sorted(
        _record_key(r) for r in expected["records"]
    ), "record multiset mismatch"
    assert got["k"] == expected["k"]
    assert got["qi_cols"] == expected["qi_cols"]
    assert got["equivalence_classes_count"] == expected["equivalence_classes_count"]


def _check_invariants(op: str, args: dict[str, Any], got: Any, invariants: dict[str, Any]) -> None:
    if op == "qol_inject":
        queries = got["queries"]
        real_index = got["real_index"]
        if "total_count" in invariants:
            assert len(queries) == invariants["total_count"]
        if invariants.get("real_count"):
            assert queries.count(args["query"]) == 1
        if invariants.get("real_index_in_range"):
            assert 0 <= real_index < len(queries)
    elif op == "ldp_randomized_response":
        if invariants.get("result_is_bool"):
            assert isinstance(got, bool)
    elif op == "dp_noisy_count":
        if invariants.get("result_is_finite"):
            assert math.isfinite(got)
    else:
        raise AssertionError(f"no invariant handler for {op!r}")


def test_conformance_case(conformance_case: dict[str, Any]) -> None:
    case = conformance_case
    op = case["request"]["op"]
    args = _args(case)
    kind = case["kind"]
    tolerance = case.get("tolerance", 0)

    if kind == "error":
        with pytest.raises(Exception) as excinfo:
            run_primitive(op, args)
        expected = case["expect"].get("error_contains", "")
        if expected:
            assert expected in str(excinfo.value), f"error {excinfo.value!r} lacks {expected!r}"
        return

    got = run_primitive(op, args)
    if kind == "exact":
        _compare(case["id"], got, case["expect"]["result"], tolerance)
    elif kind == "exact_unordered":
        _check_unordered(got, case["expect"])
    elif kind == "invariant":
        _check_invariants(op, args, got, case["expect"]["invariants"])
    else:
        raise AssertionError(f"unsupported kind {kind!r}")
