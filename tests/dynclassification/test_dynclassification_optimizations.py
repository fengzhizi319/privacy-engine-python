"""动态分类分级优化功能测试 (校验器、热加载监控、REST 路由与影子模式)."""

from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from engine.dynclassification import DynClassificationService, ProfileLoader
from engine.dynclassification.validator import validate_rules_dir, _suggest_similar_operator
from engine.main import app

client = TestClient(app)


def test_fuzzy_operator_suggestion():
    """测试拼写错误时的模糊推荐"""
    suggestion = _suggest_similar_operator("regexx")
    assert "regex" in suggestion


def test_validator_rules_dir(tmp_path):
    """测试离线规则校验工具 validate_rules_dir"""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    tax_dir = rules_dir / "taxonomies"
    dom_dir = rules_dir / "domains"
    tax_dir.mkdir()
    dom_dir.mkdir()

    (tax_dir / "default.yaml").write_text("domain: default\nstandard_id: INTERNAL\ndefault_level: L1", encoding="utf-8")
    # 建立一条使用了错误算子的规则
    (dom_dir / "bad_domain.yaml").write_text("""
domain: bad_domain
rules:
  - id: BAD_RULE
    category: PII
    level: L3
    matchers:
      - target: field_value
        operator: reggex_typo
""", encoding="utf-8")

    res = validate_rules_dir(rules_dir)
    assert res.is_valid is False
    assert any("reggex_typo" in err for err in res.errors)


def test_check_and_reload_mtime_monitoring(tmp_path):
    """测试 ProfileLoader 文件的 mtime 变更自动感应热重载"""
    loader = ProfileLoader(rules_dir=tmp_path)
    (tmp_path / "taxonomies").mkdir()
    yaml_file = tmp_path / "taxonomies" / "test.yaml"
    yaml_file.write_text("domain: test\nstandard_id: test\ndefault_level: L1", encoding="utf-8")

    # 首次检查触发缓存建立
    assert loader.check_and_reload() is True
    # 再次检查未修改返回 False
    assert loader.check_and_reload() is False

    # 修改文件内容与时间戳
    time.sleep(0.01)
    yaml_file.write_text("domain: test\nstandard_id: test\ndefault_level: L2", encoding="utf-8")
    # 再次检查应该自动检测到变动并触发 reload
    assert loader.check_and_reload() is True


def test_rest_router_endpoints(tmp_path):
    """测试 FastAPI 挂载的 REST 端点完整通告"""
    # 1. 动态评估 REST API
    res = client.post("/v1/dynclassification/eval", json={"fieldName": "identity", "value": "110101199003072375", "domain": "general-pii"})
    assert res.status_code == 200
    data = res.json()
    assert "fieldResult" in data

    # 2. 列出标准 REST API
    res_std = client.get("/v1/dynclassification/standards")
    assert res_std.status_code == 200

    # 3. 列出算子 REST API
    res_ops = client.get("/v1/dynclassification/operators")
    assert res_ops.status_code == 200
    assert "regex" in res_ops.json()["operators"]

    # 4. 热重载 REST API
    res_reload = client.post("/v1/dynclassification/profiles/reload")
    assert res_reload.status_code == 200

    # 5. 规则校验 REST API
    res_val = client.post("/v1/dynclassification/validate")
    assert res_val.status_code == 200
