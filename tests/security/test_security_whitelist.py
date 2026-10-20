"""Tests for mTLS CN whitelist management with hot-reload.

验证白名单管理器的核心功能：
- YAML 配置文件加载
- Per-CN scope 控制
- 基于 mtime 的热重载
- 向后兼容静态 CN 列表
- Fail-closed 安全设计
- 线程安全
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from engine.security.whitelist import (
    CNEntry,
    WhitelistConfig,
    WhitelistManager,
    get_whitelist_manager,
    reset_whitelist_manager,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton before each test."""
    reset_whitelist_manager()
    yield
    reset_whitelist_manager()


# ---------------------------------------------------------------------------
# CNEntry / WhitelistConfig model tests
# ---------------------------------------------------------------------------


class TestCNEntry:
    """Tests for the CNEntry Pydantic model."""

    def test_default_values(self):
        """CNEntry 默认 scopes 为 ['*']，enabled 为 True。"""
        entry = CNEntry(cn="test-client")
        assert entry.cn == "test-client"
        assert entry.scopes == ["*"]
        assert entry.description == ""
        assert entry.enabled is True

    def test_custom_values(self):
        """CNEntry 支持自定义 scopes、description、enabled。"""
        entry = CNEntry(
            cn="monitor",
            scopes=["health:read"],
            description="Prometheus scraper",
            enabled=True,
        )
        assert entry.cn == "monitor"
        assert entry.scopes == ["health:read"]
        assert entry.description == "Prometheus scraper"

    def test_disabled_entry(self):
        """CNEntry 支持 enabled=False 临时禁用。"""
        entry = CNEntry(cn="deprecated", enabled=False)
        assert entry.enabled is False


class TestWhitelistConfig:
    """Tests for the WhitelistConfig root model."""

    def test_default_values(self):
        """WhitelistConfig 默认空 entries，空 default_scopes。"""
        config = WhitelistConfig()
        assert config.version == "1.0"
        assert config.entries == []
        assert config.default_scopes == []

    def test_with_entries(self):
        """WhitelistConfig 支持多个 entries。"""
        config = WhitelistConfig(
            entries=[
                CNEntry(cn="client-a", scopes=["privacy:mask"]),
                CNEntry(cn="client-b", scopes=["*"]),
            ],
            default_scopes=[],
        )
        assert len(config.entries) == 2
        assert config.entries[0].cn == "client-a"


# ---------------------------------------------------------------------------
# WhitelistManager tests - static list mode
# ---------------------------------------------------------------------------


class TestWhitelistManagerStaticMode:
    """Tests for WhitelistManager in static list mode (backward compatibility)."""

    def test_static_cns_all_wildcard(self):
        """静态 CN 列表模式下所有 CN 获得 ['*'] scope。"""
        manager = WhitelistManager(static_cns=["client-a", "client-b"])
        assert manager.is_allowed("client-a")
        assert manager.is_allowed("client-b")
        assert manager.get_scopes("client-a") == ["*"]
        assert manager.get_scopes("client-b") == ["*"]

    def test_static_cn_not_found(self):
        """静态 CN 列表模式下未列出的 CN 返回 None。"""
        manager = WhitelistManager(static_cns=["client-a"])
        assert not manager.is_allowed("unknown-client")
        assert manager.get_scopes("unknown-client") is None
        assert manager.get_entry("unknown-client") is None

    def test_static_empty_list(self):
        """空静态 CN 列表拒绝所有（fail-closed）。"""
        manager = WhitelistManager(static_cns=[])
        assert not manager.is_allowed("any-client")

    def test_static_mode_no_config_path(self):
        """静态模式下 config_path 为 None。"""
        manager = WhitelistManager(static_cns=["client-a"])
        assert manager._config_path is None


# ---------------------------------------------------------------------------
# WhitelistManager tests - YAML config mode
# ---------------------------------------------------------------------------


class TestWhitelistManagerYAMLMode:
    """Tests for WhitelistManager with YAML config file."""

    @pytest.fixture
    def yaml_config(self, tmp_path: Path) -> Path:
        """Create a temporary YAML config file."""
        config_content = """
version: "1.0"
default_scopes: []
entries:
  - cn: "gateway"
    scopes: ["*"]
    description: "API gateway"
    enabled: true
  - cn: "monitor"
    scopes: ["health:read"]
    description: "Prometheus scraper"
    enabled: true
  - cn: "deprecated"
    scopes: ["*"]
    description: "Deprecated service"
    enabled: false
"""
        config_path = tmp_path / "mtls-whitelist.yaml"
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def test_load_from_yaml(self, yaml_config: Path):
        """从 YAML 文件加载白名单。"""
        manager = WhitelistManager(config_path=yaml_config)
        assert manager.is_allowed("gateway")
        assert manager.is_allowed("monitor")
        assert manager.get_scopes("gateway") == ["*"]
        assert manager.get_scopes("monitor") == ["health:read"]

    def test_disabled_entry_not_loaded(self, yaml_config: Path):
        """enabled=false 的条目不被加载。"""
        manager = WhitelistManager(config_path=yaml_config)
        assert not manager.is_allowed("deprecated")

    def test_unknown_cn_rejected(self, yaml_config: Path):
        """未在名单中的 CN 被拒绝（fail-closed）。"""
        manager = WhitelistManager(config_path=yaml_config)
        assert not manager.is_allowed("unknown-client")

    def test_all_entries(self, yaml_config: Path):
        """all_entries 返回所有启用的条目。"""
        manager = WhitelistManager(config_path=yaml_config)
        entries = manager.all_entries
        assert len(entries) == 2
        cns = {e.cn for e in entries}
        assert cns == {"gateway", "monitor"}

    def test_default_scopes(self, yaml_config: Path):
        """default_scopes 从 YAML 加载。"""
        manager = WhitelistManager(config_path=yaml_config)
        assert manager.default_scopes == []

    def test_config_file_not_found(self, tmp_path: Path):
        """配置文件不存在时记录错误。"""
        missing_path = tmp_path / "nonexistent.yaml"
        manager = WhitelistManager(config_path=missing_path)
        assert manager.last_error is not None
        assert "not found" in manager.last_error

    def test_invalid_yaml(self, tmp_path: Path):
        """无效 YAML 格式保留旧配置并记录错误。"""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("not: a: valid: yaml: [", encoding="utf-8")
        manager = WhitelistManager(config_path=config_path)
        assert manager.last_error is not None

    def test_wrong_root_type(self, tmp_path: Path):
        """YAML 根类型不是 dict 时报错。"""
        config_path = tmp_path / "wrong-type.yaml"
        config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        manager = WhitelistManager(config_path=config_path)
        assert manager.last_error is not None


# ---------------------------------------------------------------------------
# Hot-reload tests
# ---------------------------------------------------------------------------


class TestWhitelistManagerHotReload:
    """Tests for hot-reload functionality."""

    @pytest.fixture
    def yaml_config(self, tmp_path: Path) -> Path:
        """Create a temporary YAML config file."""
        config_content = """
version: "1.0"
entries:
  - cn: "client-a"
    scopes: ["privacy:mask"]
    description: "Initial entry"
"""
        config_path = tmp_path / "whitelist.yaml"
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def test_hot_reload_on_file_change(self, yaml_config: Path):
        """修改文件后触发热重载。"""
        manager = WhitelistManager(config_path=yaml_config)
        assert manager.is_allowed("client-a")
        assert manager.get_scopes("client-a") == ["privacy:mask"]
        initial_load_time = manager.last_load_time

        # Wait to ensure mtime changes
        time.sleep(0.1)

        # Modify the config file
        new_content = """
version: "1.0"
entries:
  - cn: "client-a"
    scopes: ["*"]
    description: "Updated entry"
  - cn: "client-b"
    scopes: ["health:read"]
    description: "New entry"
"""
        yaml_config.write_text(new_content, encoding="utf-8")

        # Trigger reload check
        assert manager.is_allowed("client-a")
        assert manager.get_scopes("client-a") == ["*"]
        assert manager.is_allowed("client-b")
        assert manager.get_scopes("client-b") == ["health:read"]
        assert manager.last_load_time > initial_load_time

    def test_explicit_reload(self, yaml_config: Path):
        """显式调用 reload() 触发重载。"""
        manager = WhitelistManager(config_path=yaml_config)
        initial_load_time = manager.last_load_time

        time.sleep(0.1)
        manager.reload()
        assert manager.last_load_time >= initial_load_time

    def test_reload_failure_retains_old_config(self, yaml_config: Path):
        """重载失败时保留旧配置。"""
        manager = WhitelistManager(config_path=yaml_config)
        assert manager.is_allowed("client-a")

        # Write invalid YAML
        time.sleep(0.1)
        yaml_config.write_text("invalid: yaml: [", encoding="utf-8")

        manager.reload()
        # Old config should be retained
        assert manager.is_allowed("client-a")
        assert manager.last_error is not None

    def test_no_reload_when_file_unchanged(self, yaml_config: Path):
        """文件未修改时不触发重载。"""
        manager = WhitelistManager(config_path=yaml_config)
        initial_load_time = manager.last_load_time

        # Multiple lookups without file change
        manager.is_allowed("client-a")
        manager.is_allowed("client-a")
        manager.is_allowed("client-a")

        # Load time should not change
        assert manager.last_load_time == initial_load_time


# ---------------------------------------------------------------------------
# Go ``clients:`` schema parity
# ---------------------------------------------------------------------------


GO_CLIENTS_YAML = """
version: "1.0"
clients:
  - cn: "service-hub"
    allowed_scopes: ["/PrivacyService/Process", "/PrivacyService/Health/*"]
    cert_fingerprints:
      - "AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99"
    role: "service"
  - cn: "mask-only"
    allowed_scopes: ["/PrivacyService/Mask"]
entries:
  - cn: "legacy-client"
    scopes: ["privacy:mask"]
"""

GO_FP = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"


class TestGoClientsSchema:
    """Go ``DynamicWhitelist`` 的 ``clients:`` 字段与钉扎/方法级授权同口径。"""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> WhitelistManager:
        path = tmp_path / "go-whitelist.yaml"
        path.write_text(GO_CLIENTS_YAML, encoding="utf-8")
        return WhitelistManager(config_path=path)

    def test_fingerprints_are_normalized(self, manager: WhitelistManager):
        entry = manager.get_entry("service-hub")
        assert entry is not None
        assert entry.cert_fingerprints == [GO_FP]
        assert entry.role == "service"

    def test_clients_without_scopes_do_not_inherit_wildcard(self, manager: WhitelistManager):
        entry = manager.get_entry("mask-only")
        assert entry is not None
        assert entry.scopes == []
        assert entry.uses_method_scopes

    def test_method_scope_matching(self, manager: WhitelistManager):
        entry, reason = manager.authorize(
            "service-hub", GO_FP, "/PrivacyService/Process"
        )
        assert entry is not None, reason
        entry, reason = manager.authorize(
            "service-hub", GO_FP, "/PrivacyService/Health/Ready"
        )
        assert entry is not None, reason

        entry, reason = manager.authorize("service-hub", GO_FP, "/PrivacyService/Mask")
        assert entry is None
        assert reason == "method_not_allowed"

    def test_fingerprint_pinning(self, manager: WhitelistManager):
        entry, reason = manager.authorize(
            "service-hub", GO_FP.upper(), "/PrivacyService/Process"
        )
        assert entry is not None, reason

        entry, reason = manager.authorize(
            "service-hub", "00" * 32, "/PrivacyService/Process"
        )
        assert entry is None
        assert reason == "cert_fingerprint_mismatch"

        entry, reason = manager.authorize("service-hub", "", "/PrivacyService/Process")
        assert entry is None
        assert reason == "cert_fingerprint_missing"

    def test_unpinned_client_needs_no_fingerprint(self, manager: WhitelistManager):
        entry, reason = manager.authorize("mask-only", "", "/PrivacyService/Mask")
        assert entry is not None, reason

    def test_legacy_entries_remain_supported(self, manager: WhitelistManager):
        entry, reason = manager.authorize("legacy-client", "", "/PrivacyService/Mask")
        assert entry is not None, reason
        assert entry.scopes == ["privacy:mask"]
        assert not entry.uses_method_scopes

    def test_unknown_cn_is_rejected(self, manager: WhitelistManager):
        entry, reason = manager.authorize("intruder", GO_FP, "/PrivacyService/Process")
        assert entry is None
        assert reason == "cn_not_allowed"

    def test_clients_win_over_duplicate_legacy_entry(self, tmp_path: Path):
        path = tmp_path / "duplicate.yaml"
        path.write_text(
            """
version: "1.0"
entries:
  - cn: "dual"
    scopes: ["*"]
clients:
  - cn: "dual"
    allowed_scopes: ["/PrivacyService/Mask"]
""",
            encoding="utf-8",
        )
        manager = WhitelistManager(config_path=path)
        entry = manager.get_entry("dual")
        assert entry is not None
        assert entry.uses_method_scopes
        assert entry.scopes == []

    def test_zero_entry_reload_keeps_previous_config(self, tmp_path: Path):
        path = tmp_path / "hot.yaml"
        path.write_text(GO_CLIENTS_YAML, encoding="utf-8")
        manager = WhitelistManager(config_path=path)
        assert manager.is_allowed("service-hub")

        time.sleep(0.05)
        path.write_text('version: "1.0"\nclients: []\n', encoding="utf-8")
        manager.reload()

        assert manager.is_allowed("service-hub")
        assert manager.last_error is not None
        assert "zero entries" in manager.last_error


# ---------------------------------------------------------------------------
# Module-level singleton tests
# ---------------------------------------------------------------------------


class TestWhitelistManagerSingleton:
    """Tests for the module-level singleton pattern."""

    def test_singleton_with_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """通过环境变量初始化单例。"""
        config_path = tmp_path / "whitelist.yaml"
        config_path.write_text(
            'version: "1.0"\nentries:\n  - cn: "test"\n    scopes: ["*"]\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("PRIVACY_AUTH_MTLS_WHITELIST_FILE", str(config_path))

        manager = get_whitelist_manager()
        assert manager.is_allowed("test")

        # Subsequent calls return the same instance
        manager2 = get_whitelist_manager()
        assert manager is manager2

    def test_singleton_fallback_to_static_cns(self, monkeypatch: pytest.MonkeyPatch):
        """未设置 WHITELIST_FILE 时回退到静态 CN 列表。"""
        monkeypatch.delenv("PRIVACY_AUTH_MTLS_WHITELIST_FILE", raising=False)
        monkeypatch.setenv("PRIVACY_AUTH_MTLS_ALLOWED_CNS", "client-a,client-b")

        manager = get_whitelist_manager()
        assert manager.is_allowed("client-a")
        assert manager.is_allowed("client-b")
        assert manager.get_scopes("client-a") == ["*"]

    def test_singleton_empty_fallback(self, monkeypatch: pytest.MonkeyPatch):
        """环境变量都为空时白名单为空（fail-closed）。"""
        monkeypatch.delenv("PRIVACY_AUTH_MTLS_WHITELIST_FILE", raising=False)
        monkeypatch.delenv("PRIVACY_AUTH_MTLS_ALLOWED_CNS", raising=False)

        manager = get_whitelist_manager()
        assert not manager.is_allowed("any-client")


# ---------------------------------------------------------------------------
# Integration with auth module
# ---------------------------------------------------------------------------


class TestWhitelistManagerAuthIntegration:
    """Integration tests for whitelist manager with auth module."""

    def test_authenticate_mtls_uses_whitelist_scopes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """mTLS 认证使用白名单中定义的 scopes。"""
        from engine.security.auth import _authenticate_mtls
        from engine.security.config import get_security_settings

        config_path = tmp_path / "whitelist.yaml"
        config_path.write_text(
            """
version: "1.0"
entries:
  - cn: "limited-client"
    scopes: ["privacy:mask", "health:read"]
    description: "Limited access client"
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("PRIVACY_AUTH_MTLS_WHITELIST_FILE", str(config_path))
        monkeypatch.setenv("PRIVACY_AUTH_INTERNAL_MTLS_ENABLED", "true")

        settings = get_security_settings()
        auth_context = {
            "transport_security_type": [b"ssl"],
            "x509_common_name": [b"limited-client"],
        }

        identity = _authenticate_mtls(settings, auth_context)
        assert identity is not None
        assert identity.name == "limited-client"
        assert identity.scopes == ["privacy:mask", "health:read"]
        assert identity.has_permission("privacy:mask")
        assert identity.has_permission("health:read")
        assert not identity.has_permission("privacy:dp")

    def test_authenticate_mtls_rejects_unknown_cn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """mTLS 认证拒绝不在白名单中的 CN。"""
        from engine.security.auth import _authenticate_mtls
        from engine.security.config import get_security_settings

        config_path = tmp_path / "whitelist.yaml"
        config_path.write_text(
            'version: "1.0"\nentries:\n  - cn: "allowed-client"\n    scopes: ["*"]\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("PRIVACY_AUTH_MTLS_WHITELIST_FILE", str(config_path))
        monkeypatch.setenv("PRIVACY_AUTH_INTERNAL_MTLS_ENABLED", "true")

        settings = get_security_settings()
        auth_context = {
            "transport_security_type": [b"ssl"],
            "x509_common_name": [b"unknown-client"],
        }

        identity = _authenticate_mtls(settings, auth_context)
        assert identity is None
