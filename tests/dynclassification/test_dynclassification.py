"""动态分类分级模块单元测试 / Dynamic Classification Module Tests.

覆盖：
- OperatorRegistry 注册/获取/并发安全
- ConfigurableRuleEngine 规则求值
- CompositeRuleEngine 记录级升级
- DomainTaxonomy 等级比较
- DynClassificationService 全流程 YAML 配置加载
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from engine.dynclassification import (
    CompositeRuleEngine,
    ConfigurableRuleEngine,
    DomainTaxonomy,
    DynClassificationService,
    OperatorRegistry,
    SecurityTag,
)
from engine.dynclassification.rule_schema import (
    CompositeRuleDef,
    MatcherDef,
    RuleDef,
    RuleProfile,
)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]
RULES_DIR = ROOT_DIR / "rules"


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def default_taxonomy() -> DomainTaxonomy:
    """构造一个最小 L1~L5 分类体系。"""
    return DomainTaxonomy(
        domain="test",
        standard_id="TEST",
        levels={
            "L1": {"id": "L1", "name": "公开", "rank": 1},
            "L2": {"id": "L2", "name": "内部", "rank": 2},
            "L3": {"id": "L3", "name": "敏感", "rank": 3},
            "L4": {"id": "L4", "name": "高敏感", "rank": 4},
            "L5": {"id": "L5", "name": "极敏感", "rank": 5},
        },
        default_level="L3",
    )


@pytest.fixture()
def simple_profile() -> RuleProfile:
    """构造一个包含手机号和身份证规则的 Profile。"""
    return RuleProfile(
        domain="test-pii",
        rules=[
            RuleDef(
                id="RULE_PHONE",
                name="手机号",
                category="PERSONAL_BASIC",
                level="L3",
                priority=100,
                matchers=[
                    MatcherDef(
                        target="field_value",
                        operator="regex",
                        params={"pattern": r"^1[3-9]\d{9}$"},
                    )
                ],
            ),
            RuleDef(
                id="RULE_IDCARD",
                name="身份证",
                category="PERSONAL_BASIC",
                level="L3",
                priority=100,
                matchers=[
                    MatcherDef(
                        target="field_value",
                        operator="id_card_checksum",
                        params={},
                    )
                ],
            ),
            RuleDef(
                id="RULE_GENOMIC",
                name="基因组字段名",
                category="GENOMIC",
                level="L5",
                priority=200,
                matchers=[
                    MatcherDef(
                        target="field_name",
                        operator="keyword_contains",
                        params={"keywords": ["brca1", "brca2", "tp53"], "use_word_boundaries": False},
                    )
                ],
            ),
        ],
        composite_rules=[
            CompositeRuleDef(
                id="COMP_COMBO",
                name="三字段组合",
                field_patterns=["^name$", "id_card|idcard", "mobile|phone"],
                min_matches=3,
                target_level="L5",
                category="COMPOSITE_COMBO",
            )
        ],
    )


@pytest.fixture()
def service() -> DynClassificationService:
    """构造使用项目 rules/ 目录的服务实例。"""
    return DynClassificationService(rules_dir=RULES_DIR)


# ===========================================================================
# OperatorRegistry Tests
# ===========================================================================


class TestOperatorRegistry:
    """算子注册表测试。"""

    def test_builtin_operators_registered(self):
        """内置算子在模块导入后应自动注册。"""
        # 触发 operators 模块加载
        from engine.dynclassification import operators  # noqa: F401

        expected = [
            "regex",
            "keyword_contains",
            "prefix_match",
            "suffix_match",
            "id_card_checksum",
            "medical_card_checksum",
            "icd10_range",
            "luhn_checksum",
            "length_range",
            "exact_match",
        ]
        registered = OperatorRegistry.list_operators()
        for op in expected:
            assert op in registered, f"算子 '{op}' 未注册"

    def test_register_and_get(self):
        """动态注册算子后可正常获取。"""
        def custom_op(value: Any, params: dict[str, Any]) -> bool:
            return str(value) == params.get("expected", "")

        OperatorRegistry.register_func("test_custom_op", custom_op)
        try:
            assert OperatorRegistry.has("test_custom_op")
            op = OperatorRegistry.get("test_custom_op")
            assert op("hello", {"expected": "hello"}) is True
            assert op("world", {"expected": "hello"}) is False
        finally:
            # 清理：从注册表移除（直接操作内部字典）
            OperatorRegistry._operators.pop("test_custom_op", None)

    def test_get_unknown_raises(self):
        """获取未注册算子应抛出 KeyError。"""
        with pytest.raises(KeyError, match="未找到"):
            OperatorRegistry.get("nonexistent_operator_xyz")

    def test_concurrent_registration(self):
        """多线程并发注册不应丢失算子。"""
        errors: list[Exception] = []

        def register_many(prefix: str, count: int):
            try:
                for i in range(count):
                    name = f"{prefix}_{i}"
                    OperatorRegistry.register_func(
                        name, lambda v, p: True
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_many, args=(f"conc_t{t}", 50))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # 验证全部注册成功
        for t in range(4):
            for i in range(50):
                assert OperatorRegistry.has(f"conc_t{t}_{i}")

        # 清理
        for t in range(4):
            for i in range(50):
                OperatorRegistry._operators.pop(f"conc_t{t}_{i}", None)


# ===========================================================================
# ConfigurableRuleEngine Tests
# ===========================================================================


class TestConfigurableRuleEngine:
    """通用规则引擎测试。"""

    def test_phone_regex_match(self, default_taxonomy, simple_profile):
        """手机号正则匹配。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
        )
        tags, _ = engine.evaluate("phone", "13800138000")
        print( tags)
        assert len(tags) >= 1
        assert any(t.rule_id == "RULE_PHONE" for t in tags)

    def test_phone_regex_no_match(self, default_taxonomy, simple_profile):
        """非手机号不应命中。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
        )
        tags, _ = engine.evaluate("phone", "12345")
        print(tags)
        assert not any(t.rule_id == "RULE_PHONE" for t in tags)

    def test_id_card_checksum(self, default_taxonomy, simple_profile):
        """身份证校验码匹配。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
        )
        # 合法身份证号（校验码正确）
        tags, _ = engine.evaluate("id_card", "110101199001011237")
        assert any(t.rule_id == "RULE_IDCARD" for t in tags)

    def test_field_name_keyword(self, default_taxonomy, simple_profile):
        """字段名关键词匹配（基因组）。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
        )
        tags, _ = engine.evaluate("patient_brca1_result", "阳性")
        assert any(t.level == "L5" and t.rule_id == "RULE_GENOMIC" for t in tags)

    def test_engine_lru_cache(self, default_taxonomy, simple_profile):
        """规则引擎评估 LRU 缓存测试。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
        )
        engine.clear_cache()
        # 第一次评估：缓存未命中
        t1, _ = engine.evaluate("phone", "13800138000")
        info1 = engine.cache_info()
        assert info1["hits"] == 0
        assert info1["misses"] == 1

        # 第二次评估相同字段与值：缓存命中
        t2, _ = engine.evaluate("phone", "13800138000")
        info2 = engine.cache_info()
        assert info2["hits"] == 1
        assert info2["misses"] == 1
        assert t1 == t2

    def test_engine_lru_cache_eviction(self, default_taxonomy, simple_profile):
        """测试真正 LRU 逐出（淘汰最久未访问条目，而非全清空）。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
            cache_max_size=2,  # 设置缓存容量上限为 2
        )
        engine.clear_cache()

        # 放入 key1, key2
        engine.evaluate("k1", "v1")
        engine.evaluate("k2", "v2")
        assert engine.cache_info()["size"] == 2

        # 再次访问 k1，提升 k1 为最新使用 (k2 变为最久未使用)
        engine.evaluate("k1", "v1")

        # 插入 k3，触发逐出 (应该逐出 k2)
        engine.evaluate("k3", "v3")
        assert engine.cache_info()["size"] == 2

        # k1 应该依然在缓存中 (hit)
        h_before = engine.cache_info()["hits"]
        engine.evaluate("k1", "v1")
        assert engine.cache_info()["hits"] == h_before + 1

        # k2 应该已被逐出 (miss)
        m_before = engine.cache_info()["misses"]
        engine.evaluate("k2", "v2")
        assert engine.cache_info()["misses"] == m_before + 1

    def test_engine_cache_context_bypass(self, default_taxonomy, simple_profile):
        """测试当存在 context 时跳过缓存。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
        )
        engine.clear_cache()

        # 带有 context 时评估
        t1, _ = engine.evaluate("phone", "13800138000", context={"tenant": "t1"})
        assert engine.cache_info()["hits"] == 0
        assert engine.cache_info()["misses"] == 0  # 带有 context 不会计入缓存统计

    def test_engine_cache_thread_safety(self, default_taxonomy, simple_profile):
        """多线程并发评估缓存安全性与结果一致性测试。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
            cache_max_size=50,
        )
        engine.clear_cache()

        # 串行先获取基准预期结果
        expected_results = {
            f"field_{i}": engine.evaluate(f"field_{i}", "13800138000")
            for i in range(10)
        }
        engine.clear_cache()

        thread_results = [[] for _ in range(5)]
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(100):
                    key = f"field_{i % 10}"
                    res = engine.evaluate(key, "13800138000")
                    thread_results[thread_id].append((key, res))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        info = engine.cache_info()
        assert info["hits"] + info["misses"] == 500
        assert info["hits"] > 400  # 验证高并发高频重复 key 时命中率 > 80%

        # 验证所有线程获得的结果与串行基准结果 100% 完全一致
        for thread_res in thread_results:
            for key, res in thread_res:
                assert res == expected_results[key]

    def test_engine_cache_invalid_env(self, default_taxonomy, simple_profile, monkeypatch):
        """测试非法的 PRIVACY_ENGINE_CACHE_MAX_SIZE 环境变量自动降级回退。"""
        monkeypatch.setenv("PRIVACY_ENGINE_CACHE_MAX_SIZE", "invalid_number")
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
        )
        assert engine.cache_info()["max_size"] == 4096

    def test_priority_ordering(self, default_taxonomy, simple_profile):
        """规则按 priority 降序排列。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
        )
        priorities = [r.priority for r in engine.rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_evaluate_batch(self, default_taxonomy, simple_profile):
        """批量评估多字段。"""
        engine = ConfigurableRuleEngine(
            taxonomy=default_taxonomy,
            profiles=[simple_profile],
            domain="test",
        )
        results = engine.evaluate_batch([
            ("phone", "13800138000"),
            ("brca1_status", "positive"),
            ("age", "30"),
        ])
        phone_tags, _ = results["phone"]
        brca1_tags, _ = results["brca1_status"]
        age_tags, _ = results["age"]
        assert len(phone_tags) >= 1
        assert len(brca1_tags) >= 1
        assert len(age_tags) == 0


# ===========================================================================
# CompositeRuleEngine Tests
# ===========================================================================


class TestCompositeRuleEngine:
    """复合规则引擎测试。"""

    def test_composite_upgrade(self, simple_profile):
        """三字段组合应触发升级。"""
        engine = CompositeRuleEngine(
            rules=simple_profile.composite_rules,
            domain="test",
        )
        record = {"name": "张三", "id_card": "110101199001011237", "phone": "13800138000"}
        tags = engine.evaluate(record)
        assert len(tags) == 1
        assert tags[0].level == "L5"
        assert tags[0].source_engine == "COMPOSITE"

    def test_composite_no_match(self, simple_profile):
        """字段不足时不应触发。"""
        engine = CompositeRuleEngine(
            rules=simple_profile.composite_rules,
            domain="test",
        )
        record = {"name": "张三", "age": "30"}
        tags = engine.evaluate(record)
        assert len(tags) == 0

    def test_apply_to_record_level(self, default_taxonomy, simple_profile):
        """复合标签应正确升级记录等级。"""
        engine = CompositeRuleEngine(
            rules=simple_profile.composite_rules,
            domain="test",
        )
        composite_tags = [
            SecurityTag(level="L5", category="COMPOSITE_COMBO")
        ]
        result = engine.apply_to_record_level("L3", composite_tags, default_taxonomy)
        assert result == "L5"

    def test_apply_no_tags_keeps_level(self, default_taxonomy, simple_profile):
        """无复合标签时等级不变。"""
        engine = CompositeRuleEngine(rules=[], domain="test")
        result = engine.apply_to_record_level("L3", [], default_taxonomy)
        assert result == "L3"


# ===========================================================================
# DomainTaxonomy Tests
# ===========================================================================


class TestDomainTaxonomy:
    """分类体系模型测试。"""

    def test_max_level_basic(self, default_taxonomy):
        """max_level 应返回 rank 最高的等级。"""
        assert default_taxonomy.max_level("L1", "L3", "L5") == "L5"
        assert default_taxonomy.max_level("L2", "L4") == "L4"
        assert default_taxonomy.max_level("L3") == "L3"

    def test_max_level_empty(self, default_taxonomy):
        """无输入时返回 default_level。"""
        assert default_taxonomy.max_level() == "L3"

    def test_max_level_invalid_ids(self, default_taxonomy):
        """无效等级 ID 应被忽略。"""
        assert default_taxonomy.max_level("INVALID", "L2") == "L2"
        assert default_taxonomy.max_level("INVALID", "ALSO_INVALID") == "L3"

    def test_get_level_rank(self, default_taxonomy):
        """get_level_rank 应返回正确权重。"""
        assert default_taxonomy.get_level_rank("L5") == 5
        assert default_taxonomy.get_level_rank("L1") == 1
        assert default_taxonomy.get_level_rank("UNKNOWN") == 0


# ===========================================================================
# DynClassificationService Integration Tests
# ===========================================================================


@pytest.mark.skipif(
    not RULES_DIR.exists(),
    reason="rules/ 目录不存在，跳过集成测试",
)
class TestDynClassificationService:
    """服务全流程集成测试（依赖 rules/ YAML 配置）。"""

    def test_classify_phone(self, service):
        """手机号字段分类。"""
        resp = service.classify_field("phone_number", "13800138000")
        assert resp.field_result is not None
        assert resp.field_result.final_level == "L3"
        assert len(resp.field_result.tags) >= 1

    def test_classify_genomic(self, service):
        """基因组字段分类应为 L5。"""
        resp = service.classify_field("brca1", "阳性")
        assert resp.field_result is not None
        assert resp.field_result.final_level == "L5"

    def test_classify_icd10_hiv(self, service):
        """ICD-10 HIV 编码应升级为 L4。"""
        resp = service.classify_field("diagnosis_code", "B20")
        assert resp.field_result is not None
        assert resp.field_result.final_level == "L4"

    def test_classify_icd10_general(self, service):
        """ICD-10 普通编码应为 L3。"""
        resp = service.classify_field("diagnosis_code", "J00")
        assert resp.field_result is not None
        assert resp.field_result.final_level == "L3"

    def test_classify_with_finance_standard(self, service):
        """金融标准下银行卡号应为 C4。"""
        resp = service.classify_field(
            "bank_card", "6222021234567890123", standard="jrt0197"
        )
        assert resp.field_result is not None
        assert resp.field_result.final_level == "C4"

    def test_classify_record_composite(self, service):
        """记录级分类：三字段组合触发复合规则升级。"""
        record = {
            "name": "张三",
            "idcard": "110101199001011237",
            "phone": "13800138000",
        }
        resp = service.classify_record(record)
        assert resp.record_result is not None
        assert resp.record_result.final_level == "L5"

    def test_classify_table(self, service):
        """表级分类。"""
        rows = [
            {"name": "张三", "phone": "13800138000"},
            {"name": "李四", "brca1": "阳性"},
        ]
        resp = service.classify_table(["name", "phone", "brca1"], rows)
        assert resp.table_result is not None
        assert resp.table_result.final_level == "L5"
        assert len(resp.table_result.record_results) == 2

    def test_list_standards(self, service):
        """列出可用标准。"""
        standards = service.list_standards()
        assert "sc_health_db51" in standards
        assert "jrt0197" in standards

    def test_list_domains(self, service):
        """列出可用领域包。"""
        domains = service.list_domains()
        assert "general-pii" in domains
        assert "medical" in domains
        assert "finance" in domains

    def test_list_operators(self, service):
        """列出已注册算子。"""
        operators = service.list_operators()
        assert "regex" in operators
        assert "id_card_checksum" in operators
        assert "icd10_range" in operators

    def test_audit_info(self, service):
        """审计信息应正确填充。"""
        resp = service.classify_field("phone", "13800138000")
        audit = resp.audit_info
        assert audit.domain != ""
        assert audit.rules_evaluated > 0
        assert audit.duration_ms >= 0

    def test_dry_run(self, service):
        """Dry-Run 预演应返回命中分布统计。"""
        sample = [
            {"name": "张三", "id_card": "110101199001011237", "phone": "13800138000"},
            {"email": "test@example.com", "age": "30"},
        ]
        result = service.dry_run(sample)
        assert result["summary"]["total_records"] == 2
        assert result["summary"]["total_fields"] == 5
        assert result["summary"]["total_hits"] >= 1
        assert 0 < result["summary"]["hit_rate"] <= 1.0
        assert isinstance(result["level_distribution"], dict)
        assert isinstance(result["category_distribution"], dict)
        assert isinstance(result["hit_details"], list)

    def test_dry_run_empty(self, service):
        """空样本 Dry-Run 应返回零命中。"""
        result = service.dry_run([])
        assert result["summary"]["total_records"] == 0
        assert result["summary"]["total_hits"] == 0
        assert result["summary"]["hit_rate"] == 0.0
