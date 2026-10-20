"""Tests for Redis-backed distributed budget accounting and fallback mechanism."""

import sys
from unittest.mock import MagicMock
import pytest

from engine.privacy.budget import (
    BudgetRegistry,
    PrivacyBudgetExhaustedError,
)


def test_budget_redis_init_and_spend_success(monkeypatch):
    """Test Redis budget accounting spend and remaining when Redis is available."""
    registry = BudgetRegistry()

    # Mock redis module in sys.modules
    mock_redis_module = MagicMock()
    mock_redis_client = MagicMock()
    mock_redis_module.Redis.from_url.return_value = mock_redis_client

    # Mock eval to return [1, new_eps, new_del, eps_tot, del_tot]
    mock_redis_client.eval.return_value = [1, 2.5, 0.0, 10.0, 0.0001]
    mock_redis_client.hmget.return_value = ["2.5", "0.0", "1700000000", "10.0", "0.0001"]

    monkeypatch.setitem(sys.modules, "redis", mock_redis_module)
    monkeypatch.setenv("PRIVACY_BUDGET_BACKEND", "redis")
    monkeypatch.setenv("PRIVACY_BUDGET_REDIS_URL", "redis://localhost:6379/0")

    acct = registry.get_or_create("redis_test_ns", epsilon_total=10.0, delta_total=0.0001)

    # Test spend
    acct.spend(2.5, 0.0)
    assert acct.epsilon_spent == 2.5
    mock_redis_client.eval.assert_called_once()

    # Test remaining
    rem = acct.remaining()
    assert rem["epsilon"] == pytest.approx(7.5)


def test_budget_redis_exhaustion(monkeypatch):
    """Test Redis budget exhaustion raises PrivacyBudgetExhaustedError."""
    registry = BudgetRegistry()

    mock_redis_module = MagicMock()
    mock_redis_client = MagicMock()
    mock_redis_module.Redis.from_url.return_value = mock_redis_client
    # Mock eval to return status = -1 (exhausted)
    mock_redis_client.eval.return_value = [-1, 10.0, 0.0, 10.0, 0.0001]

    monkeypatch.setitem(sys.modules, "redis", mock_redis_module)
    monkeypatch.setenv("PRIVACY_BUDGET_BACKEND", "redis")

    acct = registry.get_or_create("redis_exhaust_ns", epsilon_total=10.0)

    with pytest.raises(PrivacyBudgetExhaustedError):
        acct.spend(5.0)
