"""Memory Regression and Leak Prevention Test Suite for PrivShield Engine.

Verifies that:
1. High-frequency privacy primitives (masking, DP, Kano, hashing) maintain constant memory.
2. ConfigurableRuleEngine LRU evaluation cache saturates cleanly without memory leak.
3. BudgetAccountant transactions do not retain unbounded object graphs.
4. Garbage collection correctly reclaims transient batch processing buffers.
"""

import gc
import logging
import tracemalloc
import pytest

from engine.service import PrivacyService
from engine.privacy.dp import DPApi
from engine.privacy.kano import anonymize_record, BUILTIN_HIERARCHIES
from engine.privacy.budget import default_registry
from engine.dynclassification.engine import ConfigurableRuleEngine
from engine.dynclassification.models import DomainTaxonomy, SensitivityLevelDef
from engine.dynclassification.rule_schema import RuleProfile, RuleDef, MatcherDef


@pytest.fixture(autouse=True)
def clean_gc():
    """Ensure clean memory state and suppress logging buffer growth during leak checks."""
    logging.disable(logging.CRITICAL)
    gc.collect()
    yield
    gc.collect()
    logging.disable(logging.NOTSET)


def test_masking_memory_bounded():
    """Verify that masking 5,000 records exhibits bounded memory usage without leaks."""
    service = PrivacyService()
    tracemalloc.start()
    try:
        # Warmup
        for i in range(100):
            service.mask_record({"mobile": "13800138000", "id_card": "110101199003072345", "name": f"User{i}"})

        gc.collect()
        snapshot_warmup = tracemalloc.take_snapshot()

        # Sustained execution: 5,000 record masking operations
        for i in range(5000):
            service.mask_record({
                "mobile": f"138{i:08d}",
                "id_card": "110101199003072345",
                "name": f"TestUser{i % 100}",
                "amount": f"{float(i)}",
            })

        gc.collect()
        snapshot_final = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()

    stats = snapshot_final.compare_to(snapshot_warmup, "lineno")
    total_growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # Memory growth should be negligible (< 1MB) for 5000 stateless operations
    assert total_growth < 1024 * 1024, f"Unexpected memory growth in masking: {total_growth} bytes"


def test_rule_engine_lru_cache_bounded_memory():
    """Verify that ConfigurableRuleEngine LRU evaluation cache adheres to max size without unbounded growth."""
    taxonomy = DomainTaxonomy(
        domain="perf-domain",
        standard_id="STD_PERF",
        levels={
            "S1": SensitivityLevelDef(id="S1", name="公开", rank=1),
            "S2": SensitivityLevelDef(id="S2", name="内部", rank=2),
            "S3": SensitivityLevelDef(id="S3", name="敏感", rank=3),
            "S4": SensitivityLevelDef(id="S4", name="极敏", rank=4),
        },
        default_level="S1",
    )
    profile = RuleProfile(
        domain="perf-domain",
        rules=[
            RuleDef(
                id="rule-mobile",
                level="S3",
                category="PII",
                matchers=[
                    MatcherDef(target="field_name", operator="regex", params={"pattern": ".*mobile.*"}),
                ],
            ),
            RuleDef(
                id="rule-idcard",
                level="S4",
                category="PII",
                matchers=[
                    MatcherDef(target="field_name", operator="regex", params={"pattern": ".*id_card.*"}),
                ],
            ),
        ],
    )

    engine = ConfigurableRuleEngine(
        taxonomy=taxonomy,
        profiles=[profile],
        cache_max_size=256,
    )
    tracemalloc.start()
    try:
        # Saturate cache with 1000 distinct field evaluations
        for i in range(1000):
            engine.evaluate(field_name=f"test_field_name_{i}", value=f"sample_{i}")

        gc.collect()
        snapshot_saturated = tracemalloc.take_snapshot()

        # Run another 5000 evaluations with cache evictions
        for i in range(1000, 6000):
            engine.evaluate(field_name=f"test_field_name_{i}", value=f"sample_{i}")

        gc.collect()
        snapshot_final = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()

    # Ensure cache length does not exceed configured capacity
    assert len(engine._eval_cache) <= 256

    stats = snapshot_final.compare_to(snapshot_saturated, "lineno")
    growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # Growth after saturation must be very tight (< 512KB)
    assert growth < 512 * 1024, f"Rule engine cache memory leak: {growth} bytes"


def test_differential_privacy_memory_stability():
    """Verify DPApi does not leak memory over 5,000 noise injections."""
    dp_api = DPApi()
    dp_api.budget.epsilon_total = 100000.0
    dp_api.budget.delta_total = 100.0
    tracemalloc.start()
    try:
        # Warmup
        for _ in range(100):
            dp_api.count([1.0, 0.0, 1.0], epsilon=0.01)

        gc.collect()
        snapshot_warmup = tracemalloc.take_snapshot()

        values = [float(x % 10) for x in range(50)]
        for i in range(5000):
            dp_api.count(values, epsilon=0.001)
            dp_api.sum(values, epsilon=0.001, clip_lower=0.0, clip_upper=10.0, mechanism="laplace")
            dp_api.mean(values, epsilon=0.001, clip_lower=0.0, clip_upper=10.0)

        gc.collect()
        snapshot_final = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()

    stats = snapshot_final.compare_to(snapshot_warmup, "lineno")
    growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    assert growth < 512 * 1024, f"DP memory leak detected: {growth} bytes"


def test_budget_accountant_memory_stability():
    """Verify in-memory BudgetAccountant ledger maintains stable memory under repeated transactions."""
    accountants = [
        default_registry.get_or_create(namespace=f"perf-tenant-{ns_id}", epsilon_total=10000.0)
        for ns_id in range(20)
    ]
    tracemalloc.start()
    try:
        # Warmup
        for acct in accountants:
            acct.spend(epsilon=0.01)

        gc.collect()
        snapshot_warmup = tracemalloc.take_snapshot()

        # Perform 5,000 budget consumption calls across namespaces
        for i in range(5000):
            acct = accountants[i % len(accountants)]
            acct.spend(epsilon=0.001)

        gc.collect()
        snapshot_final = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()

    stats = snapshot_final.compare_to(snapshot_warmup, "lineno")
    growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    assert growth < 512 * 1024, f"Budget accountant memory growth: {growth} bytes"


def test_k_anonymity_memory_stability():
    """Verify K-Anonymity heuristic routines release transient data structures."""
    tracemalloc.start()
    try:
        # Warmup
        for _ in range(50):
            anonymize_record(
                record={"age": "28", "zipcode": "100084", "salary": "25000"},
                qi_cols=["age", "zipcode"],
                hierarchies=BUILTIN_HIERARCHIES,
                k=3,
            )

        gc.collect()
        snapshot_warmup = tracemalloc.take_snapshot()

        for i in range(2000):
            anonymize_record(
                record={"age": str(20 + (i % 60)), "zipcode": f"{100000 + (i % 900)}", "disease": "flu"},
                qi_cols=["age", "zipcode"],
                hierarchies=BUILTIN_HIERARCHIES,
                k=5,
            )

        gc.collect()
        snapshot_final = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()

    stats = snapshot_final.compare_to(snapshot_warmup, "lineno")
    growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    assert growth < 512 * 1024, f"K-anonymity memory growth: {growth} bytes"
