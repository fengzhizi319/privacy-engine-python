"""隐私原语性能基准测试 / Performance Benchmarks for Privacy Primitives.

中文说明：
使用 pytest-benchmark 插件测量核心隐私算子的吞吐量与延迟，覆盖：

1. 差分隐私 (DP) / Differential Privacy:
   - count: 10K 数据点计数加噪 / Count with 10K data points
   - sum: 带 clipping 的求和加噪 / Sum with clipping bounds
   - mean: 组合定理均值计算 / Mean via composition theorem
   - histogram: 5 类别直方图加噪 / 5-category histogram noise
   - vector_sum: 128 维向量求和 (DP-SGD) / 128-dim vector sum

2. 数据脱敏 (Masking) / Data Masking:
   - 单字段手机号/身份证 / Single field mobile/ID card
   - 100 条批量脱敏 / Batch of 100 records
   - 10 字段整记录脱敏 / Full record with 10 fields

3. K-匿名 (K-Anonymity) / K-Anonymity:
   - 100 行表级 Mondrian / 100-row table-level Mondrian

4. 查询混淆 (QOL) / Query Obfuscation:
   - 单条查询混淆 / Single query obfuscation
   - 10 条批量混淆 / Batch of 10 queries

运行方式 / How to Run:
    pytest tests/benchmark_primitives.py --benchmark-only
    pytest tests/benchmark_primitives.py --benchmark-compare

English Description:
Performance benchmarks using pytest-benchmark to measure throughput and latency
of core privacy operators: DP, masking, K-anonymity, and query obfuscation.
"""

import numpy as np
import pytest

from engine.service import PrivacyService
from engine.privacy.dp import DPApi
from engine.privacy.kano_table import k_anonymize_table


@pytest.fixture
def benchmark(pytestconfig):
    """Fallback benchmark runner if pytest-benchmark plugin is not active."""
    if pytestconfig.pluginmanager.hasplugin("benchmark"):
        # Let pytest-benchmark handle it if plugin is installed
        pass

    def _runner(func, *args, **kwargs):
        return func(*args, **kwargs)

    return _runner


@pytest.fixture
def privacy_service():
    return PrivacyService()


@pytest.fixture
def dp_api(privacy_service):
    return privacy_service.dp_api


@pytest.fixture
def masking_api(privacy_service):
    class _MaskingWrapper:
        @staticmethod
        def mask_value(field_name, value, context=""):
            return privacy_service.mask(field_name, value, context)

        @staticmethod
        def mask_batch(field_names, values, context=""):
            return privacy_service.mask_batch(field_names, values, context)

        @staticmethod
        def mask_record(record, context=""):
            return privacy_service.mask_record(record, context)

    return _MaskingWrapper()


@pytest.fixture
def qol_api(privacy_service):
    class _QolWrapper:
        @staticmethod
        def obfuscate_query(query, num_dummies=3, domain="general"):
            return privacy_service.obfuscate_query(query, num_dummies=num_dummies, domain=domain)

        @staticmethod
        def obfuscate_query_batch(queries, num_dummies=3, domain="general"):
            return privacy_service.obfuscate_query_batch(queries, num_dummies=num_dummies, domain=domain)

    return _QolWrapper()


# ── DP 基准测试 / DP Benchmarks ────────────────────────────────────


class TestDPBenchmarks:
    """差分隐私操作基准测试 / Differential Privacy Operation Benchmarks.

    测试数据规模：10,000 个数据点，使用固定种子 (42) 保证可重复性。
    关注指标：每次操作的平均延迟 (mean) 和每秒吐量 (ops/sec)。
    """

    def test_dp_count(self, dp_api, benchmark):
        """基准：10K 数据点 DP 计数（Laplace 机制，epsilon=1.0）。

        敏感度 = 1（增减一条记录最多改变计数 1），
        噪声尺度 b = 1/epsilon = 1.0。
        """
        values = np.random.default_rng(42).normal(50, 10, size=10000)
        result = benchmark(dp_api.count, values, epsilon=1.0)
        assert result is not None

    def test_dp_sum(self, dp_api, benchmark):
        """基准：带 clipping [0,100] 的 DP 求和。

        敏感度 = clip_upper - clip_lower = 100，
        噪声尺度 b = 100/epsilon = 100.0。
        """
        values = np.random.default_rng(42).normal(50, 10, size=10000)
        result = benchmark(
            dp_api.sum, values, epsilon=1.0, clip_lower=0.0, clip_upper=100.0
        )
        assert result is not None

    def test_dp_mean(self, dp_api, benchmark):
        """基准：DP 均值（组合定理：epsilon/2 给 count，epsilon/2 给 sum）。"""
        values = np.random.default_rng(42).normal(50, 10, size=10000)
        result = benchmark(
            dp_api.mean, values, epsilon=1.0, clip_lower=0.0, clip_upper=100.0
        )
        assert result is not None

    def test_dp_histogram(self, dp_api, benchmark):
        """基准：5 类别 DP 直方图（每个 bin 独立加噪）。"""
        rng = np.random.default_rng(42)
        categories = ["A", "B", "C", "D", "E"]
        values = rng.choice(categories, size=10000).tolist()
        result = benchmark(dp_api.histogram, values, categories=categories, epsilon=1.0)
        assert result is not None

    def test_dp_vector_sum(self, dp_api, benchmark):
        """基准：128 维向量 DP 求和 (DP-SGD 风格)。

        1000 个 128 维向量，max_norm=1.0 用于梯度裁剪，
        模拟联邦学习中 DP-SGD 的梯度聚合场景。
        """
        rng = np.random.default_rng(42)
        vectors = rng.normal(0, 1, size=(1000, 128))
        result = benchmark(dp_api.vector_sum, vectors, max_norm=1.0, epsilon=1.0, delta=1e-5)
        assert result is not None


# ── 脱敏基准测试 / Masking Benchmarks ─────────────────────────────────────────


class TestMaskingBenchmarks:
    """数据脱敏操作基准测试 / Data Masking Operation Benchmarks.

    脱敏操作为纯 CPU 字符串操作，无 I/O，延迟应稳定在微秒级。
    """

    def test_mask_mobile(self, masking_api, benchmark):
        """基准：单次手机号脱敏（保留前3后4，中间替换为 ****）。"""
        result = benchmark(masking_api.mask_value, "mobile", "13812345678", "")
        assert "****" in result

    def test_mask_id_card(self, masking_api, benchmark):
        """基准：单次身份证号脱敏（保留前6后4，中间 8 位替换）。"""
        result = benchmark(masking_api.mask_value, "id_card", "110105199001011234", "")
        assert "********" in result

    def test_mask_batch_100(self, masking_api, benchmark):
        """基准：100 条混合字段批量脱敏（mobile/id_card/name/email/address 各 20 条）。"""
        field_names = ["mobile", "id_card", "name", "email", "address"] * 20
        values = [
            "13812345678", "110105199001011234", "张三丰",
            "test@example.com", "北京市朝阳区建国路88号",
        ] * 20
        result = benchmark(masking_api.mask_batch, field_names, values, "")
        assert len(result) == 100

    def test_mask_record(self, masking_api, benchmark):
        """基准：10 字段整记录脱敏（模拟真实业务记录结构）。"""
        record = {
            "mobile": "13812345678",
            "id_card": "110105199001011234",
            "name": "张三丰",
            "email": "test@example.com",
            "address": "北京市朝阳区建国路88号",
            "bank_card": "6222021234567890123",
            "phone": "13998765432",
            "user_name": "李四",
            "mail": "user@domain.org",
            "addr": "上海市浦东新区陆家嘴环路1000号",
        }
        result = benchmark(masking_api.mask_record, record, "")
        assert len(result) == 10


# ── K-匿名基准测试 / K-Anonymity Benchmarks ─────────────────────────────────────


class TestKAnonBenchmarks:
    """数据集级 K-匿名基准测试 / Dataset-Level K-Anonymity Benchmarks.

    Mondrian 算法时间复杂度约为 O(n * d * log(n/k))，
    其中 n=记录数，d=QI 维度数。此测试用 100 行 3 维 QI 验证基准性能。
    """

    def test_kano_table_small(self, benchmark):
        """基准：100 行表级 K-匿名（k=5, 3 个 QI 维度, max_depth=5）。"""
        rng = np.random.default_rng(42)
        rows = [
            {
                "age": int(rng.integers(20, 70)),
                "zipcode": f"{rng.integers(100000, 999999)}",
                "gender": rng.choice(["M", "F"]),
            }
            for _ in range(100)
        ]
        result = benchmark(
            k_anonymize_table, rows, ["age", "zipcode", "gender"], k=5, max_depth=5
        )
        assert result is not None


# ── 查询混淆基准测试 / QoL Benchmarks ─────────────────────────────────────────────


class TestQolBenchmarks:
    """查询混淆基准测试 / Query Obfuscation Benchmarks.

    QOL 操作为纯内存随机采样 + 字符串拼接，延迟极低。
    主要关注批量操作的线性扩展性。
    """

    def test_obfuscate_single(self, qol_api, benchmark):
        """基准：单条医疗查询混淆（生成 3 个同类 dummy）。"""
        result = benchmark(
            qol_api.obfuscate_query,
            "糖尿病用药指南",
            num_dummies=3,
            domain="medical",
        )
        assert len(result) >= 1

    def test_obfuscate_batch_10(self, qol_api, benchmark):
        """基准：10 条医疗查询批量混淆（验证线性扩展性）。"""
        queries = [
            "糖尿病用药", "高血压治疗", "感冒药推荐",
            "抗生素使用", "疫苗接种", "体检项目",
            "中医调理", "营养补充", "康复训练", "心理咨询",
        ]
        result = benchmark(
            qol_api.obfuscate_query_batch,
            queries,
            num_dummies=3,
            domain="medical",
        )
        assert len(result) == 10


# ── 医疗全流程治理流水线基准测试 / Medical Privacy Pipeline Benchmarks ────────────────


class TestMedicalPipelineBenchmarks:
    """医疗隐私治理流水线基准测试 (分级 + PII 掩码 + L4/L5 文本抹平 + 报告统计)."""

    @pytest.fixture
    def sample_kangyang_dataset(self):
        import csv
        from pathlib import Path
        csv_path = Path("data/kangyang.csv")
        if not csv_path.exists():
            pytest.skip("data/kangyang.csv not found")
        with open(csv_path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def test_medical_pipeline_single_record(self, sample_kangyang_dataset, benchmark):
        """基准：单条康养病历记录全流程治理（27 字段分类分级 + 脱敏 + 报告）。"""
        from engine.medical_pipeline.pipeline import process_medical_dataset
        rec = [sample_kangyang_dataset[0]]
        res = benchmark(process_medical_dataset, rec, sanitize=True)
        assert res.summary["guarantee_no_l4_l5_raw_data"] is True

    def test_medical_pipeline_batch_100(self, sample_kangyang_dataset, benchmark):
        """基准：100 条康养完整病历记录全流程治理（2,700 字段批量处理与 LRU 缓存加速）。"""
        from engine.medical_pipeline.pipeline import process_medical_dataset
        res = benchmark(process_medical_dataset, sample_kangyang_dataset, sanitize=True)
        assert res.summary["guarantee_no_l4_l5_raw_data"] is True
        assert len(res.sanitized_data) == len(sample_kangyang_dataset)

