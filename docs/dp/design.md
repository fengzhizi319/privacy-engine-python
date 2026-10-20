# Python 差分隐私（DP & LDP）与预算管理设计文档 (Design)

> **组件归属**：`services/privacy-engine-python`
> **核心源码**：[`engine/privacy/dp.py`](../../engine/privacy/dp.py) · [`engine/privacy/dp_jit.py`](../../engine/privacy/dp_jit.py) · [`engine/privacy/budget.py`](../../engine/privacy/budget.py)
> **REST 路由**：[`engine/routers/dp.py`](../../engine/routers/dp.py) · [`engine/routers/budget.py`](../../engine/routers/budget.py)

---

## 1. 架构总览

PrivShield 差分隐私子系统覆盖**中心化差分隐私（CDP）**、**本地化差分隐私（LDP）**与**细粒度隐私预算记账体系（Budget Accountant）**。

```mermaid
flowchart TD
    subgraph Client ["客户端输入"]
        Q["统计查询请求 (Count/Sum/Mean)"]
        L["本地客户端数据 (Binary/Categorical)"]
    end

    subgraph Service ["服务与调度"]
        BA["BudgetAccountant (原子配额检查与扣减)"]
        DP["DPApi (中央差分隐私加噪)"]
        LDP["LocalDPApi (本地扰动与无偏估计)"]
    end

    subgraph Storage ["预算存储驱动"]
        M["内存状态 (线程安全 RLock)"]
        R["Redis (分布式集中记账)"]
        S["SQLite (单机持久化)"]
    end

    Q --> BA
    BA -->|扣减成功| DP
    BA -.->|超额拦截 (429/400)| ERR["PrivacyBudgetExhaustedError"]
    DP -->|Laplace / Gaussian / Welford| RES["DPResult (带噪结果 + 95% 置信区间)"]
    L --> LDP
    LDP -->|随机响应 / 投影| EST["无偏聚合频率分布"]
    BA <--> M
    BA <--> R
    BA <--> S
```

---

## 2. 核心算法原理与实现

### 2.1 中央差分隐私（CDP）机制
1. **Laplace 机制**：
   对于数值型函数 $f$，添加服从 $\text{Lap}(\Delta f / \epsilon)$ 分布的拉普拉斯噪声：
   $$Y = f(D) + \text{Lap}\left(\frac{\Delta f}{\epsilon}\right)$$
   适用于具有明确 $L_1$ 敏感度边界的计数（$\Delta=1$）和求和（$\Delta=B$）查询。
2. **高斯机制（Gaussian Mechanism）**：
   添加服从 $\mathcal{N}(0, \sigma^2)$ 分布的高斯噪声，满足 $(\epsilon, \delta)$-DP：
   $$\sigma \ge \frac{\Delta_2 f \sqrt{2 \ln(1.25/\delta)}}{\epsilon}$$
   针对复杂多维向量（`vector_sum`、`vector_mean`），配合自适应或固定 $L_2$ 范数裁剪（`clip_l2_norm`）。
3. **单遍均值计算（Single-Pass Welford Mean）**：
   为了防止多次遍历全量数据产生二次泄露，采用单遍 Welford 算法流式同步维护样本均值与方差，使用敏感度约束 $B$ 限制每个个体的最大贡献。

### 2.2 本地差分隐私（LDP）机制
1. **二值随机响应（Warner's Randomized Response）**：
   $$P(\tilde{x}=1 \mid x=1) = \frac{e^\epsilon}{1+e^\epsilon}, \quad P(\tilde{x}=0 \mid x=1) = \frac{1}{1+e^\epsilon}$$
   保证无法反推单用户的真实二值偏好。
2. **多分类优化随机响应（Optimized Randomized Response, O-RR）**：
   针对基数为 $k$ 的离散类别，根据扰动概率矩阵投影并反演无偏直方图估算：
   $$\hat{p}_v = \frac{\tilde{p}_v - q}{p - q}$$

### 2.3 隐私预算控制（Budget Accountant）
1. **组合定理**：
   - **串行组合（Sequential Composition）**：在同一数据集上的多次独立查询，其总隐私损失为各次查询预算之和：$\sum \epsilon_i \le \epsilon_{\text{total}}$。
   - **Rényi 差分隐私（RDP）**：内置 `RDPAccountant`，对亚高斯加噪操作实现更紧致的紧致组合阶数累加。
2. **原子预算扣减**：
   支持基于 `threading.RLock`、Redis 单命令事务或 SQLite 严格事务的原子借记操作。若当前剩余预算不足，抛出 `PrivacyBudgetExhaustedError`，强力阻断侧信道攻击。
