"""Tests for the main PrivacyService orchestrator.

覆盖 PrivacyService 的初始化、各隐私原语委托调用、参数解析、
预算管理及异常处理等核心路径。
"""

import os
import tempfile

import pytest
import yaml

from engine.privacy.budget import BudgetAccountant, BudgetRegistry
from engine.privacy.dp import DPApi, LocalDPApi
from engine.privacy.profile import ParameterResolver
from engine.service import PrivacyService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def service():
    """创建一个使用独立 BudgetRegistry 的 PrivacyService 实例，避免测试间干扰。"""
    registry = BudgetRegistry()
    return PrivacyService(namespace="test_ns", registry=registry)


@pytest.fixture()
def profile_path():
    """创建临时 YAML profile 文件并在测试结束后清理。"""
    profile_content = {
        "primitives": {
            "dp": {
                "epsilon": 2.0,
                "mechanism": "laplace",
            },
            "k_anonymity": {
                "k": 3,
            },
            "qol": {
                "num_dummies": 5,
            },
        }
    }
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.safe_dump(profile_content, f)
        path = f.name
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# 初始化测试
# ---------------------------------------------------------------------------


class TestInitialization:
    """PrivacyService 初始化相关测试。"""

    def test_default_initialization(self):
        """默认初始化：验证各属性类型正确。"""
        svc = PrivacyService()
        assert isinstance(svc.resolver, ParameterResolver)
        assert isinstance(svc.registry, BudgetRegistry)
        assert isinstance(svc.dp_api, DPApi)
        assert isinstance(svc.local_dp_api, LocalDPApi)
        assert svc.namespace == "default"

    def test_custom_namespace(self):
        """自定义命名空间初始化。"""
        registry = BudgetRegistry()
        svc = PrivacyService(namespace="custom_ns", registry=registry)
        assert svc.namespace == "custom_ns"
        assert svc.registry is registry

    def test_initialization_with_profile(self, profile_path):
        """通过 YAML profile 初始化：验证 resolver 正确加载配置。"""
        registry = BudgetRegistry()
        svc = PrivacyService(profile_path=profile_path, registry=registry)
        assert svc.resolver is not None
        # 验证 profile 中的 dp epsilon 被正确加载
        params = svc.resolver.resolve("dp", None)
        assert params["epsilon"] == 2.0

    def test_initialization_with_budget_params(self):
        """指定 epsilon_total / delta_total 初始化。"""
        registry = BudgetRegistry()
        svc = PrivacyService(
            namespace="budget_ns",
            registry=registry,
            epsilon_total=5.0,
            delta_total=1e-3,
        )
        accountant = registry.get("budget_ns")
        assert accountant is not None
        assert accountant.epsilon_total == 5.0
        assert accountant.delta_total == 1e-3

    def test_dp_api_shares_registry(self):
        """DPApi 与 PrivacyService 共享同一个 BudgetRegistry。"""
        registry = BudgetRegistry()
        svc = PrivacyService(namespace="shared_ns", registry=registry)
        assert svc.dp_api.registry is registry


# ---------------------------------------------------------------------------
# 脱敏（Masking）委托测试
# ---------------------------------------------------------------------------


class TestMasking:
    """PrivacyService 脱敏方法委托测试。"""

    def test_mask_single_value(self, service):
        """单值脱敏：手机号字段。"""
        result = service.mask("mobile", "13812345678")
        # 脱敏后不应等于原始值
        assert result != "13812345678"
        assert isinstance(result, str)

    def test_mask_email_field(self, service):
        """单值脱敏：邮箱字段保留域名。"""
        result = service.mask("email", "john.doe@example.com")
        assert result != "john.doe@example.com"
        assert "example.com" in result

    def test_mask_record(self, service):
        """整条记录脱敏。"""
        record = {"name": "John Doe", "email": "john.doe@example.com"}
        masked = service.mask_record(record)
        assert "name" in masked
        assert "email" in masked
        assert masked["name"] != "John Doe"
        assert masked["email"] != "john.doe@example.com"
        assert "example.com" in masked["email"]

    def test_mask_batch(self, service):
        """批量脱敏。"""
        field_names = ["mobile", "email"]
        values = ["13812345678", "test@test.com"]
        results = service.mask_batch(field_names, values)
        assert len(results) == 2
        assert results[0] != "13812345678"
        assert results[1] != "test@test.com"

    def test_hash(self, service):
        """HMAC 哈希：结果确定性。"""
        h1 = service.hash("hello", "salt123")
        h2 = service.hash("hello", "salt123")
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 16  # 16 字符 base64 摘要

    def test_hash_different_salt(self, service):
        """HMAC 哈希：不同盐值产生不同结果。"""
        h1 = service.hash("hello", "salt_a")
        h2 = service.hash("hello", "salt_b")
        assert h1 != h2

    def test_truncate(self, service):
        """截断：保留指定前缀长度。"""
        result = service.truncate("13812345678", 3)
        assert result.startswith("138")
        assert len(result) < len("13812345678")


# ---------------------------------------------------------------------------
# 差分隐私（DP）委托测试
# ---------------------------------------------------------------------------


class TestDifferentialPrivacy:
    """PrivacyService 差分隐私方法委托测试。"""

    def test_dp_count(self, service):
        """DP 计数：结果在真实值附近。"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = service.dp_count(values, {"epsilon": 1.0})
        assert isinstance(result, float)
        # 带噪声的计数应在合理范围内
        assert 0 <= result <= 20

    def test_dp_sum(self, service):
        """DP 求和：需要 clip 参数。"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = service.dp_sum(
            values, {"epsilon": 1.0, "clip_lower": 0.0, "clip_upper": 10.0}
        )
        assert isinstance(result, float)

    def test_dp_mean(self, service):
        """DP 均值。"""
        values = [10.0, 20.0, 30.0, 40.0, 50.0] * 10
        result = service.dp_mean(
            values, {"epsilon": 2.0, "clip_lower": 0.0, "clip_upper": 100.0}
        )
        assert isinstance(result, float)

    def test_dp_histogram(self, service):
        """DP 直方图。"""
        values = ["A", "B", "A", "C", "B", "A"]
        categories = ["A", "B", "C"]
        result = service.dp_histogram(values, categories, {"epsilon": 1.0})
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B", "C"}

    def test_dp_noisy_count(self, service):
        """对已聚合计数注入噪声。"""
        result = service.dp_noisy_count(100.0, {"epsilon": 1.0})
        assert isinstance(result, float)
        assert 50 <= result <= 150  # 合理范围

    def test_dp_noisy_sum_with_sensitivity(self, service):
        """对已聚合求和注入噪声（直接指定 sensitivity）。"""
        result = service.dp_noisy_sum(500.0, {"epsilon": 1.0, "sensitivity": 10.0})
        assert isinstance(result, float)

    def test_dp_noisy_sum_with_clip_bounds(self, service):
        """对已聚合求和注入噪声（通过 clip 边界推导 sensitivity）。"""
        result = service.dp_noisy_sum(
            500.0, {"epsilon": 1.0, "clip_lower": 0.0, "clip_upper": 10.0}
        )
        assert isinstance(result, float)

    def test_dp_noisy_sum_missing_params_raises(self, service):
        """dp_noisy_sum 缺少 sensitivity 和 clip 参数时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="sensitivity"):
            service.dp_noisy_sum(500.0, {"epsilon": 1.0})

    def test_dp_noisy_mean(self, service):
        """对已聚合的 sum/count 注入噪声得到均值。"""
        result = service.dp_noisy_mean(
            500.0, 50.0, {"epsilon": 1.0, "sensitivity": 10.0}
        )
        assert isinstance(result, float)

    def test_dp_noisy_mean_missing_params_raises(self, service):
        """dp_noisy_mean 缺少参数时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="sensitivity"):
            service.dp_noisy_mean(500.0, 50.0, {"epsilon": 1.0})

    def test_dp_noisy_histogram(self, service):
        """对已聚合直方图注入噪声。"""
        true_counts = {"A": 100, "B": 50, "C": 30}
        result = service.dp_noisy_histogram(true_counts, {"epsilon": 1.0})
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B", "C"}

    def test_dp_chunked_count(self, service):
        """分块流式 DP 计数。"""
        chunks = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
        result = service.dp_chunked_count(chunks, {"epsilon": 1.0})
        assert isinstance(result, float)
        assert 0 <= result <= 20

    def test_dp_chunked_sum(self, service):
        """分块流式 DP 求和。"""
        chunks = [[1.0, 2.0], [3.0, 4.0], [5.0]]
        result = service.dp_chunked_sum(
            chunks, {"epsilon": 1.0, "clip_lower": 0.0, "clip_upper": 10.0}
        )
        assert isinstance(result, float)

    def test_dp_chunked_mean(self, service):
        """分块流式 DP 均值。"""
        chunks = [[10.0, 20.0], [30.0, 40.0], [50.0] * 10]
        result = service.dp_chunked_mean(
            chunks, {"epsilon": 2.0, "clip_lower": 0.0, "clip_upper": 100.0}
        )
        assert isinstance(result, float)

    def test_dp_chunked_histogram(self, service):
        """分块流式 DP 直方图。"""
        chunks = [["A", "B"], ["A", "C"], ["B", "C", "A"]]
        categories = ["A", "B", "C"]
        result = service.dp_chunked_histogram(chunks, categories, {"epsilon": 1.0})
        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B", "C"}

    def test_dp_format_kwargs_extraction(self, service):
        """_dp_format_kwargs 正确提取 column/party 参数。"""
        params = {"column": "age", "party": "alice", "epsilon": 1.0}
        kwargs = service._dp_format_kwargs(params)
        assert kwargs == {"column": "age", "party": "alice"}

    def test_dp_format_kwargs_none(self, service):
        """_dp_format_kwargs 对 None 输入返回空字典。"""
        assert service._dp_format_kwargs(None) == {}

    def test_dp_format_kwargs_empty(self, service):
        """_dp_format_kwargs 对无相关键的输入返回空字典。"""
        assert service._dp_format_kwargs({"epsilon": 1.0}) == {}


# ---------------------------------------------------------------------------
# 本地差分隐私（Local DP）测试
# ---------------------------------------------------------------------------


class TestLocalDP:
    """PrivacyService 本地差分隐私方法测试。"""

    def test_perturb_binary_batch(self, service):
        """批量二值扰动：输出长度一致且值为 0/1。"""
        values = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        result = service.perturb_binary_batch(values, epsilon=1.0)
        assert len(result) == len(values)
        assert all(v in (0, 1) for v in result)

    def test_perturb_categorical_batch(self, service):
        """批量类别型扰动：输出长度一致且值在类别集内。"""
        values = ["A", "B", "C", "A", "B"]
        categories = ["A", "B", "C"]
        result = service.perturb_categorical_batch(values, categories, epsilon=1.0)
        assert len(result) == len(values)
        assert all(v in categories for v in result)

    def test_estimate_binary_frequency(self, service):
        """二值频率估计：结果在 [0, 1] 范围内。"""
        reported = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10
        freq = service.estimate_binary_frequency(reported, epsilon=1.0)
        assert isinstance(freq, float)
        # 估计值应在合理范围内（可能因纠偏略超出 [0,1]，但不应偏离太远）
        assert -0.5 <= freq <= 1.5

    def test_estimate_categorical_histogram(self, service):
        """类别型直方图估计：返回各类别频率。"""
        reported = ["A", "B", "C", "A", "A", "B"] * 10
        categories = ["A", "B", "C"]
        hist = service.estimate_categorical_histogram(reported, categories, epsilon=1.0)
        assert isinstance(hist, dict)
        assert set(hist.keys()) == set(categories)


# ---------------------------------------------------------------------------
# K-匿名（K-Anonymity）测试
# ---------------------------------------------------------------------------


class TestKAnonymity:
    """PrivacyService K-匿名方法测试。"""

    def test_k_anonymize_record(self, service):
        """单条记录 K-匿名泛化。"""
        record = {"age": "25", "zipcode": "10001", "gender": "M"}
        result = service.k_anonymize_record(record, ["age", "zipcode"], k=5)
        assert isinstance(result, dict)
        # 泛化后年龄应变为区间形式
        assert result["age"] != "25" or result["age"] == "*"
        # 非 QI 列保持不变
        assert result["gender"] == "M"

    def test_k_anonymize_record_custom_hierarchies(self, service):
        """自定义泛化层次结构应被正确合并。"""

        def custom_age(value: str, level: int) -> str:
            return "CUSTOM"

        record = {"age": "30", "zipcode": "20001"}
        result = service.k_anonymize_record(
            record, ["age"], k=5, hierarchies={"age": custom_age}
        )
        # 自定义层次应生效
        assert result["age"] == "CUSTOM"

    def test_k_anonymize_table(self, service):
        """整张表 K-匿名泛化。"""
        rows = [
            {"age": "25", "zipcode": "10001"},
            {"age": "26", "zipcode": "10002"},
            {"age": "27", "zipcode": "10003"},
            {"age": "28", "zipcode": "10004"},
            {"age": "29", "zipcode": "10005"},
        ]
        result = service.k_anonymize_table(rows, ["age", "zipcode"], k=2)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_k_anonymize_table_respects_profile_k(self, profile_path):
        """profile 中配置的 k 值应被正确解析。"""
        registry = BudgetRegistry()
        svc = PrivacyService(
            profile_path=profile_path, namespace="kano_ns", registry=registry
        )
        rows = [
            {"age": "25", "zipcode": "10001"},
            {"age": "26", "zipcode": "10002"},
            {"age": "27", "zipcode": "10003"},
        ]
        # profile 中 k=3，请求 k=2 → 请求参数优先
        result = svc.k_anonymize_table(rows, ["age", "zipcode"], k=2)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 查询混淆（QOL）测试
# ---------------------------------------------------------------------------


class TestQueryObfuscation:
    """PrivacyService 查询混淆方法测试。"""

    def test_obfuscate_query(self, service):
        """单条查询混淆：返回 num_dummies+1 条查询。"""
        result = service.obfuscate_query("高血压如何治疗", num_dummies=3, seed=42)
        assert isinstance(result, list)
        assert len(result) == 4  # 1 真实 + 3 dummy
        assert "高血压如何治疗" in result

    def test_obfuscate_query_generic_domain(self, service):
        """通用领域查询混淆。"""
        result = service.obfuscate_query(
            "天气预报", num_dummies=2, domain="generic", seed=42
        )
        assert isinstance(result, list)
        assert len(result) == 3
        assert "天气预报" in result

    def test_obfuscate_query_custom_pool(self, service):
        """自定义 dummy 池。"""
        custom_pool = ["自定义查询A", "自定义查询B", "自定义查询C"]
        result = service.obfuscate_query(
            "真实查询", num_dummies=2, medical_pool=custom_pool, seed=42
        )
        assert isinstance(result, list)
        assert len(result) == 3

    def test_obfuscate_query_batch(self, service):
        """批量查询混淆。"""
        queries = ["高血压如何治疗", "糖尿病饮食"]
        results = service.obfuscate_query_batch(queries, num_dummies=2, seed=42)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, list)
            assert len(r) == 3  # 1 真实 + 2 dummy

    def test_obfuscate_query_deterministic_with_seed(self, service):
        """相同 seed 产生相同结果。"""
        r1 = service.obfuscate_query("测试查询", num_dummies=3, seed=123)
        r2 = service.obfuscate_query("测试查询", num_dummies=3, seed=123)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 预算管理测试
# ---------------------------------------------------------------------------


class TestBudgetManagement:
    """PrivacyService 隐私预算管理测试。"""

    def test_budget_remaining_initial(self, service):
        """初始状态剩余预算等于总预算。"""
        remaining = service.budget_remaining()
        assert "epsilon" in remaining
        assert "delta" in remaining
        assert remaining["epsilon"] > 0

    def test_budget_decreases_after_dp_query(self, service):
        """DP 查询后预算减少。"""
        before = service.budget_remaining()
        service.dp_count([1, 2, 3], {"epsilon": 1.0})
        after = service.budget_remaining()
        assert after["epsilon"] < before["epsilon"]


# ---------------------------------------------------------------------------
# 参数推荐测试
# ---------------------------------------------------------------------------


class TestRecommendParams:
    """PrivacyService 参数自动推荐测试。"""

    def test_recommend_dp_params(self, service, tmp_path, monkeypatch):
        """根据数值数据推荐 DP 参数。"""
        # 使用临时文件避免污染项目配置
        personalized_file = tmp_path / "personalized-profiles.yaml"
        monkeypatch.setenv("PRIVACY_PERSONALIZED_PROFILE", str(personalized_file))

        values = [float(i) for i in range(100)]
        result = service.recommend_and_save_params(values=values)
        assert "dp" in result
        dp_params = result["dp"]
        assert dp_params["epsilon"] == 1.0
        assert "clip_lower" in dp_params
        assert "clip_upper" in dp_params
        assert dp_params["clip_lower"] < dp_params["clip_upper"]

    def test_recommend_kano_params(self, service, tmp_path, monkeypatch):
        """根据表格数据推荐 K-Anonymity 参数。"""
        personalized_file = tmp_path / "personalized-profiles.yaml"
        monkeypatch.setenv("PRIVACY_PERSONALIZED_PROFILE", str(personalized_file))

        rows = [{"age": str(i), "zipcode": f"100{i:02d}"} for i in range(50)]
        result = service.recommend_and_save_params(rows=rows, qi_cols=["age", "zipcode"])
        assert "k_anonymity" in result
        kano_params = result["k_anonymity"]
        assert 2 <= kano_params["k"] <= 10

    def test_recommend_no_data_returns_empty(self, service):
        """无数据时返回空推荐。"""
        result = service.recommend_and_save_params()
        assert result == {}


# ---------------------------------------------------------------------------
# 异常处理测试
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """PrivacyService 异常处理测试。"""

    def test_dp_invalid_epsilon_raises(self, service):
        """epsilon <= 0 时应抛出 ValueError。"""
        with pytest.raises(ValueError):
            service.dp_count([1, 2, 3], {"epsilon": -1.0})

    def test_k_anonymity_invalid_k_raises(self, service):
        """k < 2 时应抛出 ValueError。"""
        with pytest.raises(ValueError):
            service.k_anonymize_record({"age": "25"}, ["age"], k=1)

    def test_hash_empty_salt_raises(self, service):
        """空盐值应抛出 ValueError。"""
        with pytest.raises(ValueError):
            service.hash("hello", "")
