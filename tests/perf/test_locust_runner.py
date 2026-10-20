"""Unit & Integration verification for Locust load test tasks.

Runs all tasks defined in PrivShieldLoadTestUser against FastAPI TestClient
to ensure load testing contracts never break in CI.
"""

import pytest
from fastapi.testclient import TestClient
from engine.main import app
from tests.perf.locustfile import PrivShieldLoadTestUser


class MockLocustClient:
    """Adapts FastAPI TestClient to mimic Locust's self.client."""

    def __init__(self, test_client: TestClient):
        self._client = test_client

    def get(self, url, name=None):
        resp = self._client.get(url)
        assert resp.status_code in (200, 404), f"GET {url} failed with {resp.status_code}"
        return resp

    def post(self, url, json=None, name=None):
        resp = self._client.post(url, json=json)
        assert resp.status_code == 200, f"POST {url} failed with {resp.status_code}: {resp.text}"
        return resp


@pytest.fixture
def mock_locust_user():
    """Create an instance of PrivShieldLoadTestUser with mocked HttpUser client."""
    client = TestClient(app)
    user = PrivShieldLoadTestUser(environment=None)
    user.client = MockLocustClient(client)
    return user


def test_locust_health_task(mock_locust_user):
    mock_locust_user.test_health()


def test_locust_metrics_task(mock_locust_user):
    mock_locust_user.test_metrics()


def test_locust_mask_field_task(mock_locust_user):
    mock_locust_user.test_mask_field()


def test_locust_mask_record_task(mock_locust_user):
    mock_locust_user.test_mask_record()


def test_locust_hmac_hash_task(mock_locust_user):
    mock_locust_user.test_hmac_hash()


def test_locust_dp_count_task(mock_locust_user):
    mock_locust_user.test_dp_count()


def test_locust_dp_sum_task(mock_locust_user):
    mock_locust_user.test_dp_sum()


def test_locust_kano_anonymize_task(mock_locust_user):
    mock_locust_user.test_kano_anonymize()


def test_locust_classification_evaluate_task(mock_locust_user):
    mock_locust_user.test_classification_evaluate()
