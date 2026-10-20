"""Locust Load Testing Suite for PrivShield Engine.

Simulates enterprise privacy sidecar workloads including:
1. High-frequency PII masking (single field & batch records)
2. Differential privacy noise injection (count, sum, mean)
3. Dynamic data classification (Layer-1 rule evaluation)
4. K-anonymity record generalization
5. System health and Prometheus metrics telemetry

Usage:
  # Web UI mode (http://localhost:8089):
  locust -f tests/perf/locustfile.py --host=http://127.0.0.1:8079

  # Headless load test (100 users, spawn rate 10/s, run for 30s):
  locust -f tests/perf/locustfile.py --headless -u 100 -r 10 --run-time 30s --host=http://127.0.0.1:8079
"""

import random

try:
    from locust import HttpUser, task, between, tag
except ImportError:
    # Graceful fallback when locust is not installed in the execution environment
    class HttpUser:
        def __init__(self, *args, **kwargs):
            self.client = None

    def task(weight=1):
        def decorator(f):
            f.locust_task_weight = weight
            return f
        return decorator

    def between(a, b):
        return lambda: random.uniform(a, b)

    def tag(*tags):
        def decorator(f):
            f.locust_tags = tags
            return f
        return decorator


class PrivShieldLoadTestUser(HttpUser):
    """Simulates high-throughput client traffic hitting PrivShield privacy sidecar."""

    wait_time = between(0.01, 0.1)  # 10ms ~ 100ms request intervals

    SAMPLE_MOBILES = [
        "13800138000",
        "13912345678",
        "18611112222",
        "15099887766",
        "19988776655",
    ]

    SAMPLE_IDS = [
        "110101199003072345",
        "310104198512154567",
        "440106199208201234",
        "510104198801013456",
    ]

    SAMPLE_NAMES = ["张三", "李四", "王五", "赵六", "钱七", "孙八"]

    SAMPLE_FIELDS = [
        ("mobile", "13800138000"),
        ("id_card", "110101199003072345"),
        ("name", "张三"),
        ("bank_card", "6222021234567890123"),
        ("email", "test.user@example.com"),
        ("address", "北京市海淀区中关村南大街1号"),
    ]

    @tag("health")
    @task(5)
    def test_health(self):
        """Health check endpoint probe."""
        self.client.get("/health", name="/health")

    @tag("metrics")
    @task(2)
    def test_metrics(self):
        """Prometheus metrics endpoint scrape."""
        self.client.get("/metrics", name="/metrics")

    @tag("masking")
    @task(30)
    def test_mask_field(self):
        """Single field PII masking."""
        field_name, value = random.choice(self.SAMPLE_FIELDS)
        payload = {
            "field_name": field_name,
            "value": value,
            "context": "",
        }
        self.client.post("/v1/privacy/mask", json=payload, name="/v1/privacy/mask")

    @tag("masking")
    @task(25)
    def test_mask_record(self):
        """Multi-field record batch masking."""
        payload = {
            "record": {
                "mobile": random.choice(self.SAMPLE_MOBILES),
                "name": random.choice(self.SAMPLE_NAMES),
                "id_card": random.choice(self.SAMPLE_IDS),
                "age": str(random.randint(18, 80)),
                "diagnosis": "高血压",
            },
            "context": "medical_outpatient",
        }
        self.client.post("/v1/privacy/mask_record", json=payload, name="/v1/privacy/mask_record")

    @tag("hash")
    @task(10)
    def test_hmac_hash(self):
        """HMAC salted hashing."""
        payload = {
            "value": random.choice(self.SAMPLE_MOBILES),
            "salt": "perf-test-salt",
        }
        self.client.post("/v1/privacy/hash", json=payload, name="/v1/privacy/hash")

    @tag("dp")
    @task(15)
    def test_dp_count(self):
        """Differential privacy count."""
        values = [random.choice([0.0, 1.0]) for _ in range(20)]
        payload = {
            "values": values,
            "params": {"epsilon": 0.5},
        }
        self.client.post("/v1/privacy/dp/count", json=payload, name="/v1/privacy/dp/count")

    @tag("dp")
    @task(10)
    def test_dp_sum(self):
        """Differential privacy sum with Laplace mechanism."""
        values = [random.uniform(10.0, 500.0) for _ in range(20)]
        payload = {
            "values": values,
            "params": {
                "epsilon": 1.0,
                "mechanism": "laplace",
                "clip_lower": 0.0,
                "clip_upper": 500.0,
            },
        }
        self.client.post("/v1/privacy/dp/sum", json=payload, name="/v1/privacy/dp/sum")

    @tag("kano")
    @task(5)
    def test_kano_anonymize(self):
        """K-anonymity heuristic record generalization."""
        payload = {
            "record": {
                "age": str(random.randint(20, 70)),
                "zipcode": f"{random.randint(100000, 999999)}",
                "disease": "Diabetes",
            },
            "k": 3,
            "qi_cols": ["age", "zipcode"],
        }
        self.client.post("/v1/privacy/k_anonymize/record", json=payload, name="/v1/privacy/k_anonymize/record")

    @tag("classification")
    @task(15)
    def test_classification_evaluate(self):
        """Layer-1 rule engine dynamic classification."""
        payload = {
            "field_name": "patient_id_card",
            "value": random.choice(self.SAMPLE_IDS),
            "domain": "general-pii",
        }
        self.client.post("/v1/dynclassification/eval", json=payload, name="/v1/dynclassification/eval")
