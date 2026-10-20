"""通用领域策略与脱敏回调注册表单元测试。"""

from engine.dynclassification import (
    DomainStrategyRegistry,
    default_domain_registry,
)


def test_domain_strategy_registry_basic() -> None:
    registry = DomainStrategyRegistry()

    def dummy_sanitizer(field_name: str, text: str, level: str, mode: str = "redact") -> str:
        return "[CLEANED]"

    registry.register_sanitizer("finance", dummy_sanitizer)
    retrieved = registry.get_sanitizer("FINANCE")
    assert retrieved is not None
    assert retrieved("card_no", "123456", "L4") == "[CLEANED]"

    assert registry.unregister_sanitizer("finance") is True
    assert registry.get_sanitizer("finance") is None


def test_default_domain_registry_global_singleton() -> None:
    def test_sanitizer(field_name: str, text: str, level: str, mode: str = "redact") -> str:
        return "MASKED"

    default_domain_registry.register_sanitizer("hr", test_sanitizer)
    assert default_domain_registry.get_sanitizer("HR") is not None
    default_domain_registry.unregister_sanitizer("hr")
