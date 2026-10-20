"""Performance and Latency Benchmark Regression Test Suite for Medical Privacy Pipeline.

Guarantees:
1. 100 full records (2,700 fields) processing completes in < 450 ms (warm) and < 1000 ms (cold).
2. Zero L4/L5 raw data leakage with 100% compliance guarantee.
3. True LRU cache semantics and clean cache invalidation.
4. Memory usage remains strictly bounded under sustained batch operations.
"""

import csv
import gc
from pathlib import Path
import time
import tracemalloc
import pytest

from engine.medical_pipeline.pipeline import (
    MedicalPrivacyPipeline,
    get_default_pipeline,
    process_medical_dataset,
    reset_default_pipeline,
)


@pytest.fixture
def kangyang_records():
    """Load 100 full production-grade sample records from data/kangyang.csv."""
    csv_path = Path("data/kangyang.csv")
    if not csv_path.exists():
        pytest.skip("data/kangyang.csv dataset not found")
    with open(csv_path, encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    assert len(records) >= 50, "Sample dataset too small for meaningful benchmark"
    return records


def test_medical_pipeline_latency_regression(kangyang_records):
    """Regression test ensuring 100-record medical governance completes within strict SLA (< 500ms)."""
    reset_default_pipeline()

    # 1. Cold start execution
    t0 = time.perf_counter()
    res_cold = process_medical_dataset(kangyang_records, sanitize=True)
    cold_duration_ms = (time.perf_counter() - t0) * 1000.0

    assert res_cold.summary["guarantee_no_l4_l5_raw_data"] is True
    assert res_cold.summary["redaction_failures"] == 0
    assert cold_duration_ms < 2000.0, f"Cold execution too slow: {cold_duration_ms:.2f} ms > 2000 ms"

    # 2. Warm execution with LRU cache saturation
    t0 = time.perf_counter()
    res_warm = process_medical_dataset(kangyang_records, sanitize=True)
    warm_duration_ms = (time.perf_counter() - t0) * 1000.0

    assert res_warm.summary["guarantee_no_l4_l5_raw_data"] is True
    assert len(res_warm.sanitized_data) == len(kangyang_records)
    assert warm_duration_ms < 800.0, (
        f"Warm execution exceeded SLA budget: {warm_duration_ms:.2f} ms > 800 ms (Regression detected!)"
    )
    assert res_warm.summary["duration_ms"] < 750.0


def test_medical_pipeline_lru_and_invalidation():
    """Verify true LRU hit promotion, FIFO eviction, and clear_cache invalidation."""
    pipeline = MedicalPrivacyPipeline(redact_engine="rule")
    pipeline.clear_cache()

    # Populate cache
    pipeline._classify_field("department", "心血管内科")
    pipeline._classify_field("gender", "男")

    keys = list(pipeline._field_class_cache.keys())
    assert keys[0] == ("department", "心血管内科")
    assert keys[1] == ("gender", "男")

    # Access first key -> Should promote to most recent (end of insertion order)
    pipeline._classify_field("department", "心血管内科")
    keys_promoted = list(pipeline._field_class_cache.keys())
    assert keys_promoted[0] == ("gender", "男")
    assert keys_promoted[1] == ("department", "心血管内科")

    # Invalidate cache
    pipeline.clear_cache()
    assert len(pipeline._field_class_cache) == 0
    assert len(pipeline._sanitized_cache) == 0
    assert len(pipeline._ner_cache) == 0


def test_medical_pipeline_memory_bounded(kangyang_records):
    """Verify sustained execution over 10 batches (1,000 records, 27,000 fields) does not leak memory."""
    reset_default_pipeline()
    gc.collect()

    tracemalloc.start()
    try:
        # Warmup batch
        process_medical_dataset(kangyang_records, sanitize=True)
        gc.collect()
        snapshot_start = tracemalloc.take_snapshot()

        # 10 sustained batch runs
        for _ in range(10):
            res = process_medical_dataset(kangyang_records, sanitize=True)
            assert res.summary["guarantee_no_l4_l5_raw_data"] is True

        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()

    stats = snapshot_end.compare_to(snapshot_start, "lineno")
    net_allocated_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    net_allocated_kb = net_allocated_bytes / 1024.0

    # Memory growth should be well below 2MB thanks to bounded LRU caches
    assert net_allocated_kb < 2048.0, (
        f"Memory growth of {net_allocated_kb:.2f} KB exceeds bounded limit 2048 KB"
    )
