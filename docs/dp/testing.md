# Python 差分隐私测试指南 (Testing)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 测试体系结构

| 测试模块 | 覆盖功能 |
|---|---|
| `tests/privacy/test_dp.py` | 基础 Laplace、Gaussian 加噪、裁剪与置信区间测试 |
| `tests/privacy/test_dp_extended.py` | 多维向量加噪、自适应裁剪、单遍 Welford 均值计算 |
| `tests/privacy/test_ldp.py` | Warner 随机响应、O-RR 离散扰动、无偏直方图与频率重建 |
| `tests/privacy/test_budget.py` | 预算账本、并发线程安全、Redis/SQLite 驱动、RDP 累加器 |

---

## 2. 自动化执行

```bash
# 运行 DP 与预算所有单测
.venv/bin/python -m pytest tests/privacy/test_dp*.py tests/privacy/test_ldp*.py tests/privacy/test_budget*.py -v
```

---

## 3. 统计学性质验证 (Hypothesis & Kolmogorov-Smirnov)
在 `test_dp.py` 中，采用经验分布检验（KS-Test）验证加噪输出的统计学无偏性：
- 抽取 $N=10,000$ 个扰动样本，计算样本均值与方差，确保其符合理论期望：
  $$\mathbb{E}[\tilde{X}] \approx X_{\text{raw}}, \quad \text{Var}[\tilde{X}] \approx \frac{2 \Delta^2}{\epsilon^2}$$
- 双侧 KS 检验 $p$-value 必须 $> 0.01$，排除伪随机数发生器偏差。
