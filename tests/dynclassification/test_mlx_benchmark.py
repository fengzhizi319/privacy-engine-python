"""MLX Metal GPU vs CPU 性能对比基准测试 / Metal GPU vs CPU Performance Benchmark.

对比 NER 和 LLM 在 Apple Silicon Metal GPU 与 CPU 上的推理性能：
- NER (BERT): 不同长度文本的推理延迟对比
- LLM (Qwen2): 文本生成吞吐量与首 token 延迟对比
- 矩阵运算基准: 不同规模 matmul 的 GPU/CPU 加速比

运行方式：
    # 完整基准测试（含真实模型，约 2-5 分钟）
    PYTHONPATH=. pytest tests/dynclassification/test_mlx_benchmark.py -v -s

    # 仅基础矩阵运算基准（无需模型，秒级完成）
    PYTHONPATH=. pytest tests/dynclassification/test_mlx_benchmark.py -v -s -k "TestMatmulBenchmark"

    # 仅 NER 基准
    PYTHONPATH=. pytest tests/dynclassification/test_mlx_benchmark.py -v -s -k "TestNerBenchmark"

    # 仅 LLM 基准
    PYTHONPATH=. pytest tests/dynclassification/test_mlx_benchmark.py -v -s -k "TestLlmBenchmark"
"""

from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path

import pytest

# MLX 仅在 macOS 上可用
pytestmark = [
    pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="MLX Metal benchmark only runs on macOS",
    ),
    pytest.mark.benchmark,
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLX_NER_MODEL_DIR = PROJECT_ROOT / ".models" / "raner_cmeee-mlx"
MLX_LLM_MODEL_DIR = PROJECT_ROOT / ".models" / "Qwen2-VL-2B-Instruct-mlx"


# =========================================================================== #
# Fixtures
# =========================================================================== #


@pytest.fixture(scope="module")
def mlx():
    """导入 mlx 并返回模块。"""
    try:
        import mlx.core as mx
        return mx
    except ImportError:
        pytest.skip("mlx not installed")


@pytest.fixture(scope="module")
def ner_engine():
    """加载 MLX NER 引擎（模块级复用）。"""
    if not MLX_NER_MODEL_DIR.exists():
        pytest.skip(f"MLX NER model not found: {MLX_NER_MODEL_DIR}")
    from engine.dynclassification.mlx_ner_engine import MLXSmallNerEngine

    engine = MLXSmallNerEngine(model_dir=str(MLX_NER_MODEL_DIR))
    engine._lazy_init()
    return engine


@pytest.fixture(scope="module")
def llm_classifier():
    """加载 MLX LLM 分类器（模块级复用）。"""
    if not MLX_LLM_MODEL_DIR.exists():
        pytest.skip(f"MLX LLM model not found: {MLX_LLM_MODEL_DIR}")
    from engine.dynclassification.mlx_llm_engine import MLXLlmClassifier

    classifier = MLXLlmClassifier(model_dir=str(MLX_LLM_MODEL_DIR))
    classifier._lazy_init()
    return classifier


# =========================================================================== #
# 辅助函数
# =========================================================================== #


def _benchmark_fn(fn, warmup: int = 2, repeats: int = 5) -> dict:
    """对函数进行基准测试，返回统计结果。

    Args:
        fn: 待测函数（无参数）。
        warmup: 预热次数。
        repeats: 正式计时次数。

    Returns:
        {"mean": 平均耗时, "median": 中位数, "min": 最小, "max": 最大, "stdev": 标准差, "all": 全部}
    """
    # 预热
    for _ in range(warmup):
        fn()

    # 正式计时
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    return {
        "mean": statistics.mean(timings),
        "median": statistics.median(timings),
        "min": min(timings),
        "max": max(timings),
        "stdev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "all": timings,
    }


def _print_comparison(title: str, gpu_result: dict, cpu_result: dict) -> None:
    """打印 GPU vs CPU 性能对比表。"""
    speedup = cpu_result["mean"] / gpu_result["mean"] if gpu_result["mean"] > 0 else float("inf")
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'指标':<12} {'Metal GPU':>14} {'CPU':>14} {'加速比':>10}")
    print(f"  {'-' * 52}")
    print(f"  {'平均 (ms)':<12} {gpu_result['mean']*1000:>12.2f} {cpu_result['mean']*1000:>12.2f} {speedup:>9.2f}x")
    print(f"  {'中位数 (ms)':<12} {gpu_result['median']*1000:>12.2f} {cpu_result['median']*1000:>12.2f}")
    print(f"  {'最小 (ms)':<12} {gpu_result['min']*1000:>12.2f} {cpu_result['min']*1000:>12.2f}")
    print(f"  {'最大 (ms)':<12} {gpu_result['max']*1000:>12.2f} {cpu_result['max']*1000:>12.2f}")
    print(f"  {'标准差 (ms)':<12} {gpu_result['stdev']*1000:>12.2f} {cpu_result['stdev']*1000:>12.2f}")
    print(f"{'=' * 70}\n")


# =========================================================================== #
# 矩阵运算基准：GPU vs CPU 加速比
# =========================================================================== #


class TestMatmulBenchmark:
    """MLX 矩阵运算 Metal GPU vs CPU 基准对比。"""

    def test_matmul_small(self, mlx):
        """小矩阵 (256x256) GPU vs CPU 对比。"""
        mx = mlx
        a = mx.random.normal((256, 256))
        b = mx.random.normal((256, 256))

        def run():
            c = a @ b
            mx.eval(c)

        # GPU
        mx.set_default_device(mx.gpu)
        gpu_result = _benchmark_fn(run, warmup=3, repeats=10)

        # CPU
        mx.set_default_device(mx.cpu)
        cpu_result = _benchmark_fn(run, warmup=3, repeats=10)

        # 恢复 GPU
        mx.set_default_device(mx.gpu)
        _print_comparison("Matmul 256×256", gpu_result, cpu_result)

    def test_matmul_medium(self, mlx):
        """中矩阵 (1024x1024) GPU vs CPU 对比。"""
        mx = mlx
        a = mx.random.normal((1024, 1024))
        b = mx.random.normal((1024, 1024))

        def run():
            c = a @ b
            mx.eval(c)

        mx.set_default_device(mx.gpu)
        gpu_result = _benchmark_fn(run, warmup=3, repeats=10)

        mx.set_default_device(mx.cpu)
        cpu_result = _benchmark_fn(run, warmup=3, repeats=10)

        mx.set_default_device(mx.gpu)
        speedup = cpu_result["mean"] / gpu_result["mean"]
        _print_comparison("Matmul 1024×1024", gpu_result, cpu_result)
        # 中等规模矩阵 GPU 应有加速
        assert speedup > 0.5, f"GPU 不应比 CPU 慢超过 2x (实际 {speedup:.2f}x)"

    def test_matmul_large(self, mlx):
        """大矩阵 (4096x4096) GPU vs CPU 对比 — GPU 应显著快于 CPU。"""
        mx = mlx
        a = mx.random.normal((4096, 4096))
        b = mx.random.normal((4096, 4096))

        def run():
            c = a @ b
            mx.eval(c)

        mx.set_default_device(mx.gpu)
        gpu_result = _benchmark_fn(run, warmup=2, repeats=5)

        mx.set_default_device(mx.cpu)
        cpu_result = _benchmark_fn(run, warmup=2, repeats=5)

        mx.set_default_device(mx.gpu)
        speedup = cpu_result["mean"] / gpu_result["mean"]
        _print_comparison("Matmul 4096×4096", gpu_result, cpu_result)
        # 大规模矩阵 GPU 加速应明显
        print(f"  >>> GPU 加速比: {speedup:.2f}x")

    def test_batch_matmul(self, mlx):
        """批量矩阵乘法 (12x512x512) 模拟注意力头计算。"""
        mx = mlx
        # 模拟 12 个注意力头，seq_len=512
        q = mx.random.normal((12, 512, 64))
        k = mx.random.normal((12, 512, 64))

        def run():
            scores = q @ k.transpose(0, 2, 1)
            mx.eval(scores)

        mx.set_default_device(mx.gpu)
        gpu_result = _benchmark_fn(run, warmup=3, repeats=10)

        mx.set_default_device(mx.cpu)
        cpu_result = _benchmark_fn(run, warmup=3, repeats=10)

        mx.set_default_device(mx.gpu)
        _print_comparison("Batch Matmul (12 heads × 512 × 64)", gpu_result, cpu_result)


# =========================================================================== #
# NER 推理基准：Metal GPU vs CPU
# =========================================================================== #


class TestNerBenchmark:
    """MLX NER (BERT) 推理 Metal GPU vs CPU 性能对比。"""

    # 不同长度的测试文本
    TEST_TEXTS = {
        "短文本 (15字)": "患者诊断为糖尿病",
        "中文本 (50字)": "患者张三，男，55岁，因反复胸闷气短3天入院，既往有高血压病史10年，长期服用氨氯地平降压治疗",
        "长文本 (150字)": (
            "患者李四，女，68岁，因间断头晕伴恶心呕吐2天就诊。"
            "患者2天前无明显诱因出现头晕，伴视物旋转，伴恶心呕吐，无耳鸣及听力下降。"
            "既往有2型糖尿病病史15年，目前口服二甲双胍及格列美脲降糖治疗，空腹血糖控制在7mmol/L左右。"
            "有冠心病病史5年，长期口服阿司匹林、阿托伐他汀治疗。"
        ),
    }

    def test_ner_gpu_vs_cpu_short(self, mlx, ner_engine):
        """NER 短文本推理 GPU vs CPU 对比。"""
        self._run_ner_comparison(mlx, ner_engine, "短文本 (15字)")

    def test_ner_gpu_vs_cpu_medium(self, mlx, ner_engine):
        """NER 中等文本推理 GPU vs CPU 对比。"""
        self._run_ner_comparison(mlx, ner_engine, "中文本 (50字)")

    def test_ner_gpu_vs_cpu_long(self, mlx, ner_engine):
        """NER 长文本推理 GPU vs CPU 对比。"""
        self._run_ner_comparison(mlx, ner_engine, "长文本 (150字)")

    def test_ner_throughput(self, mlx, ner_engine):
        """NER 吞吐量测试：连续处理 10 条文本的总耗时对比。"""
        mx = mlx
        texts = [
            "患者诊断为急性心肌梗死",
            "给予阿司匹林和氯吡格雷抗血小板治疗",
            "血常规检查白细胞计数偏高",
            "患者有青霉素过敏史",
            "血压150/95mmHg",
            "空腹血糖8.2mmol/L",
            "心电图示ST段抬高",
            "诊断为2型糖尿病",
            "行冠脉造影及支架植入术",
            "术后给予低分子肝素抗凝",
        ]

        def run_batch():
            for t in texts:
                ner_engine.extract(t)

        # GPU
        mx.set_default_device(mx.gpu)
        gpu_result = _benchmark_fn(run_batch, warmup=1, repeats=3)

        # CPU
        mx.set_default_device(mx.cpu)
        cpu_result = _benchmark_fn(run_batch, warmup=1, repeats=3)

        mx.set_default_device(mx.gpu)
        speedup = cpu_result["mean"] / gpu_result["mean"]
        _print_comparison("NER 批量吞吐 (10条文本)", gpu_result, cpu_result)
        print(f"  >>> GPU 吞吐量: {10/gpu_result['mean']:.1f} texts/s")
        print(f"  >>> CPU 吞吐量: {10/cpu_result['mean']:.1f} texts/s")
        print(f"  >>> 加速比: {speedup:.2f}x")

    def _run_ner_comparison(self, mx, engine, text_key: str):
        """执行单个 NER GPU vs CPU 对比。"""
        text = self.TEST_TEXTS[text_key]

        def run():
            engine.extract(text)

        # GPU
        mx.set_default_device(mx.gpu)
        gpu_result = _benchmark_fn(run, warmup=2, repeats=5)

        # CPU
        mx.set_default_device(mx.cpu)
        cpu_result = _benchmark_fn(run, warmup=2, repeats=5)

        # 恢复 GPU
        mx.set_default_device(mx.gpu)
        speedup = cpu_result["mean"] / gpu_result["mean"]
        _print_comparison(f"NER 推理 - {text_key}", gpu_result, cpu_result)
        print(f"  >>> 加速比: {speedup:.2f}x")

        # 验证两种设备产出一致
        mx.set_default_device(mx.gpu)
        gpu_entities = engine.extract(text)
        mx.set_default_device(mx.cpu)
        cpu_entities = engine.extract(text)
        mx.set_default_device(mx.gpu)

        # 实体数量应一致
        assert len(gpu_entities) == len(cpu_entities), (
            f"GPU/CPU 产出实体数不一致: GPU={len(gpu_entities)}, CPU={len(cpu_entities)}"
        )


# =========================================================================== #
# LLM 推理基准：Metal GPU vs CPU
# =========================================================================== #


class TestLlmBenchmark:
    """MLX LLM (Qwen2) 推理 Metal GPU vs CPU 性能对比。"""

    TEST_TEXTS = {
        "短文本": "身份证号：510101199001011234",
        "中文本": "患者张三，身份证号510101199001011234，诊断为2型糖尿病，目前口服二甲双胍治疗",
    }

    def test_llm_gpu_vs_cpu_short(self, mlx, llm_classifier):
        """LLM 短文本分类 GPU vs CPU 对比。"""
        self._run_llm_comparison(mlx, llm_classifier, "短文本")

    def test_llm_gpu_vs_cpu_medium(self, mlx, llm_classifier):
        """LLM 中等文本分类 GPU vs CPU 对比。"""
        self._run_llm_comparison(mlx, llm_classifier, "中文本")

    def test_llm_token_generation_speed(self, mlx, llm_classifier):
        """LLM token 生成速度对比（tokens/s）。"""
        mx = mlx
        from engine.dynclassification.base import SensitivityLevel

        text = "患者有精神分裂症病史，长期服用奥氮平治疗"

        def run():
            llm_classifier.classify(text, SensitivityLevel.L3, 0.5)

        # GPU 计时
        mx.set_default_device(mx.gpu)
        gpu_result = _benchmark_fn(run, warmup=1, repeats=3)

        # CPU 计时
        mx.set_default_device(mx.cpu)
        cpu_result = _benchmark_fn(run, warmup=1, repeats=3)

        mx.set_default_device(mx.gpu)
        speedup = cpu_result["mean"] / gpu_result["mean"]
        _print_comparison("LLM 完整分类 (含生成)", gpu_result, cpu_result)

        # 估算 token 生成速度（假设平均生成 ~50 tokens）
        est_tokens = 50
        gpu_tps = est_tokens / gpu_result["mean"]
        cpu_tps = est_tokens / cpu_result["mean"]
        print(f"  >>> 估算 GPU 生成速度: ~{gpu_tps:.1f} tokens/s")
        print(f"  >>> 估算 CPU 生成速度: ~{cpu_tps:.1f} tokens/s")
        print(f"  >>> 加速比: {speedup:.2f}x")

    def test_llm_prefill_vs_decode(self, mlx, llm_classifier):
        """LLM Prefill（首 token 延迟）vs Decode（逐 token）分阶段对比。"""
        mx = mlx

        # 准备 prompt
        system_prompt = llm_classifier._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请评估以下文本数据的敏感数据等级：\n身份证号510101199001011234"},
        ]
        prompt_text = llm_classifier._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = llm_classifier._tokenizer.encode(prompt_text)

        def run_prefill():
            """仅测量 prefill 阶段（处理完整 prompt）。"""
            input_ids = mx.array([prompt_ids])
            logits, _ = llm_classifier._forward_step(input_ids, cache=None, offset=0)
            mx.eval(logits)

        def run_decode_step():
            """测量单步 decode（单 token 前向传播）。"""
            input_ids = mx.array([prompt_ids])
            logits, cache = llm_classifier._forward_step(input_ids, cache=None, offset=0)
            mx.eval(logits)
            # 单步 decode
            next_token = int(mx.argmax(logits[-1, :]).item())
            input_ids = mx.array([[next_token]])
            logits2, _ = llm_classifier._forward_step(input_ids, cache=cache, offset=len(prompt_ids))
            mx.eval(logits2)

        # GPU
        mx.set_default_device(mx.gpu)
        gpu_prefill = _benchmark_fn(run_prefill, warmup=2, repeats=5)
        gpu_decode = _benchmark_fn(run_decode_step, warmup=2, repeats=5)

        # CPU
        mx.set_default_device(mx.cpu)
        cpu_prefill = _benchmark_fn(run_prefill, warmup=2, repeats=5)
        cpu_decode = _benchmark_fn(run_decode_step, warmup=2, repeats=5)

        mx.set_default_device(mx.gpu)

        _print_comparison("LLM Prefill (首 token 延迟)", gpu_prefill, cpu_prefill)
        _print_comparison("LLM Prefill + 1 Decode Step", gpu_decode, cpu_decode)

        # 计算纯 decode 延迟
        gpu_decode_only = gpu_decode["mean"] - gpu_prefill["mean"]
        cpu_decode_only = cpu_decode["mean"] - cpu_prefill["mean"]
        print(f"  >>> 纯 Decode 单步延迟 (GPU): {gpu_decode_only*1000:.2f} ms")
        print(f"  >>> 纯 Decode 单步延迟 (CPU): {cpu_decode_only*1000:.2f} ms")
        if gpu_decode_only > 0:
            print(f"  >>> Decode 加速比: {cpu_decode_only/gpu_decode_only:.2f}x")

    def _run_llm_comparison(self, mx, classifier, text_key: str):
        """执行单个 LLM GPU vs CPU 对比。"""
        from engine.dynclassification.base import SensitivityLevel

        text = self.TEST_TEXTS[text_key]

        def run():
            classifier.classify(text, SensitivityLevel.L3, 0.5)

        # GPU
        mx.set_default_device(mx.gpu)
        gpu_result = _benchmark_fn(run, warmup=1, repeats=3)

        # CPU
        mx.set_default_device(mx.cpu)
        cpu_result = _benchmark_fn(run, warmup=1, repeats=3)

        # 恢复 GPU
        mx.set_default_device(mx.gpu)
        speedup = cpu_result["mean"] / gpu_result["mean"]
        _print_comparison(f"LLM 分类 - {text_key}", gpu_result, cpu_result)
        print(f"  >>> 加速比: {speedup:.2f}x")


# =========================================================================== #
# 综合对比报告
# =========================================================================== #


class TestBenchmarkSummary:
    """生成综合性能对比报告。"""

    def test_full_benchmark_report(self, mlx, ner_engine, llm_classifier):
        """生成完整 NER + LLM GPU vs CPU 性能报告。"""
        mx = mlx
        from engine.dynclassification.base import SensitivityLevel

        print("\n" + "=" * 70)
        print("  Apple MLX Metal GPU vs CPU 性能对比报告")
        print(f"  设备: {platform.machine()} / macOS {platform.mac_ver()[0]}")
        print(f"  MLX 版本: {mx.__version__}")
        print("=" * 70)

        # --- NER 基准 ---
        ner_text = "患者张三，男，55岁，因反复胸闷气短3天入院，既往有高血压病史10年"

        def ner_run():
            ner_engine.extract(ner_text)

        mx.set_default_device(mx.gpu)
        ner_gpu = _benchmark_fn(ner_run, warmup=2, repeats=5)
        mx.set_default_device(mx.cpu)
        ner_cpu = _benchmark_fn(ner_run, warmup=2, repeats=5)

        # --- LLM 基准 ---
        llm_text = "身份证号：510101199001011234"

        def llm_run():
            llm_classifier.classify(llm_text, SensitivityLevel.L3, 0.5)

        mx.set_default_device(mx.gpu)
        llm_gpu = _benchmark_fn(llm_run, warmup=1, repeats=3)
        mx.set_default_device(mx.cpu)
        llm_cpu = _benchmark_fn(llm_run, warmup=1, repeats=3)

        mx.set_default_device(mx.gpu)

        # 输出汇总表
        ner_speedup = ner_cpu["mean"] / ner_gpu["mean"]
        llm_speedup = llm_cpu["mean"] / llm_gpu["mean"]

        print(f"\n{'─' * 70}")
        print(f"  {'模型':<20} {'GPU (ms)':>12} {'CPU (ms)':>12} {'加速比':>10}")
        print(f"{'─' * 70}")
        print(f"  {'NER (BERT)':<20} {ner_gpu['mean']*1000:>10.1f} {ner_cpu['mean']*1000:>10.1f} {ner_speedup:>9.2f}x")
        print(f"  {'LLM (Qwen2-2B)':<20} {llm_gpu['mean']*1000:>10.1f} {llm_cpu['mean']*1000:>10.1f} {llm_speedup:>9.2f}x")
        print(f"{'─' * 70}")
        print(f"\n  结论:")
        if ner_speedup > 1.0:
            print(f"  • NER: Metal GPU 比 CPU 快 {ner_speedup:.2f}x")
        else:
            print(f"  • NER: CPU 比 Metal GPU 快 {1/ner_speedup:.2f}x（小模型 GPU 开销可能大于收益）")
        if llm_speedup > 1.0:
            print(f"  • LLM: Metal GPU 比 CPU 快 {llm_speedup:.2f}x")
        else:
            print(f"  • LLM: CPU 比 Metal GPU 快 {1/llm_speedup:.2f}x")
        print()

        # 基本断言：推理应成功完成
        assert ner_gpu["mean"] > 0
        assert ner_cpu["mean"] > 0
        assert llm_gpu["mean"] > 0
        assert llm_cpu["mean"] > 0
