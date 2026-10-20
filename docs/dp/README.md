# 差分隐私（DP / LDP）模块

> 代码：`engine/privacy/dp.py`、`dp_jit.py`、`budget.py` · REST：`/v1/privacy/dp/*`、`/v1/privacy/ldp/*`、`/v1/privacy/budget` · 测试：`tests/privacy/test_dp*.py`、`test_budget*.py`

## 定位

中央差分隐私（Laplace / Gaussian）与本地差分隐私（Randomized Response / O-RR）原语，配套进程内/Redis/SQLite 预算记账。`DPApi` 与 `LocalDPApi` 无状态（随机源可注入），预算状态集中在 `BudgetAccountant`。

## 代码地图

| 文件/符号 | 职责 |
|---|---|
| `Mechanism` | `laplace` / `gaussian` |
| `AggregationType` | `count` / `sum` / `mean` / `histogram` / `vector_sum` / `vector_mean` |
| `DPApi` | 中央 DP：`count`、`sum`、`mean`、`histogram`、`batch_count`、`batch_sum`、`noisy_*`、`chunked_*`、`dp_aggregate`、`dp_groupby`、`vector_sum`、`vector_mean`、`adaptive_clip`、`create_accumulator` / `finalize_dp` |
| `LocalDPApi` | 本地 DP：`randomized_response`、`orr_response`、`perturb_binary(_batch)`、`perturb_categorical(_batch)`、`estimate_binary_frequency`、`estimate_categorical_histogram` |
| `Accumulator` | 可序列化增量累加器，用于流式/分布式聚合 |
| `DPResult` | 带噪结果 + 敏感度 + 噪声尺度 + 置信区间 + Arrow 导出 |
| `SecureRandom` | 基于 `secrets` 的随机源；显式 `seed` 时退化为可复现 PRNG（仅测试） |
| `calibrate_analytic_gaussian()` | 解析 Gaussian 机制 sigma 校准 |
| `clip_value` / `clip_l2_norm` / `_bound_contributions` | 敏感度控制；`user_ids` + `max_contributions` 实现 User-Level DP |
| `dp_jit.py` | 批量 Laplace/Gaussian、裁剪与向量 L2 裁剪（NumPy 向量化，非 Numba） |
| `budget.py` | `BudgetAccountant`、`BudgetRegistry`（`default_registry`）、`RDPAccountant`、`BudgetAuditLogger`、`PrivacyBudgetExhaustedError` |

## REST 接口

| 路径 | 说明 |
|---|---|
| `/v1/privacy/dp/count`、`sum`、`mean`、`histogram` | 标准 DP 查询（扣预算） |
| `/v1/privacy/dp/noisy_count`、`noisy_sum`、`noisy_mean`、`noisy_histogram` | 仅加噪、不扣预算 |
| `/v1/privacy/dp/aggregate`、`groupby` | 多聚合 / 分组聚合 |
| `/v1/privacy/dp/vector_sum`、`vector_mean` | 向量聚合 + L2 裁剪 |
| `/v1/privacy/dp/adaptive_clip` | 自适应裁剪边界 |
| `/v1/privacy/dp/chunked_count`、`chunked_sum`、`chunked_mean`、`chunked_histogram` | 分块流式 |
| `/v1/privacy/dp/arrow_ipc` | 直接接收 Arrow IPC 字节流 |
| `/v1/privacy/ldp/randomized_response`、`orr` | 本地扰动 |
| `/v1/privacy/ldp/perturb/binary`、`perturb/categorical` | 批量本地扰动 |
| `/v1/privacy/ldp/estimate/binary`、`estimate/categorical` | 无偏频率/直方图估计 |
| GET/POST/DELETE `/v1/privacy/budget` | 查询 / 手动扣减 / 删除命名空间预算 |
| POST `/v1/privacy/budget/reset` | 重置预算 |

全部路由要求 `privacy:dp` 或 `privacy:budget` scope（见 [auth.md](../auth.md)）。`DPApi.batch_count` / `batch_sum` 目前只有 Python API，没有对应 REST 路由。

## 预算语义

```python
from engine.privacy.budget import default_registry, PrivacyBudgetExhaustedError
acct = default_registry.get_or_create("tenant-a", epsilon_total=10.0, delta_total=1e-4)
```

- **不能直接构造** `BudgetAccountant(...)`，`__new__` 会抛 `TypeError`；必须经 `default_registry.get_or_create()` / `get_budget()`。
- 默认总预算 `epsilon=10.0`、`delta=1e-4`；`window_seconds`（或 `PRIVACY_BUDGET_WINDOW_SECONDS`）到期后已消耗预算清零。
- 超预算抛 `PrivacyBudgetExhaustedError`；非法金额（非正）抛 `ValueError`。
- 后端优先级：Redis（`PRIVACY_BUDGET_BACKEND=redis` 或 `PRIVACY_BUDGET_REDIS_URL`）→ SQLite（`PRIVACY_BUDGET_DB`）→ 进程内存。Redis/SQLite 不可用时告警并回退，跨副本一致性需自行评估。
- `BudgetAuditLogger` 为每次扣减写 HMAC 签名的 JSONL 审计；密钥取 `PRIVACY_AUDIT_KEY`，落盘路径取 `PRIVACY_BUDGET_AUDIT_LOG`。未配置密钥会告警，生产必须显式配置。
- RDP 记账（`RDPAccountant.record_gaussian/record_rdp`）用于 Gaussian 组合分析；启动 `DPApi` 时通过 `rdp_accountant` 参数注入。

## 行为要点

- **后处理选项**：`clip_non_negative`（默认 True）、`round_int`、`discrete`（整数格上的离散 Laplace）；`return_details=True` 时附加解析高斯置信区间。
- **随机源**：默认 `SecureRandom`（OS 熵源）；生产禁止传 `seed`，该参数仅用于测试复现。
- **LDP epsilon 归一**：`epsilon <= 0` 或非有限值按 0 处理（扰动最均匀），与 Go `ldp` 语义对齐。
- **LDP 与中央 DP 共用 scope**：`privacy:dp`。
- **批量路径**：`dp_jit.py` 使用 NumPy 向量化；`chunked_*` 以块为单位累计，峰值内存受块大小约束，但调用方仍需限制单块大小。

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_BUDGET_BACKEND` | 空 | `redis` 启用分布式预算 |
| `PRIVACY_BUDGET_REDIS_URL` | 空 | 设置后即使未声明 backend 也尝试 Redis |
| `PRIVACY_BUDGET_DB` | 空 | SQLite 路径；空则进程内存 |
| `PRIVACY_BUDGET_WINDOW_SECONDS` | 无窗口 | 预算窗口长度 |
| `PRIVACY_BUDGET_AUDIT_LOG` | `/tmp/budget_audit.log` | 预算审计落盘路径；生产必须改为持久卷 |
| `PRIVACY_AUDIT_KEY` | 空 | 审计 HMAC 密钥（生产必配） |

## 测试

```bash
pytest tests/privacy/test_dp.py tests/privacy/test_dp_property.py \
       tests/privacy/test_dp_distributions.py tests/privacy/test_dp_security_fixes.py -q
pytest tests/privacy/test_budget.py tests/privacy/test_budget_concurrency.py tests/privacy/test_budget_redis.py -q
```

## 与 Go 的差异与边界

| 项 | Go 引擎 | Python 引擎 |
|---|---|---|
| 随机数生成 | 单遍采样 + 原子预算 | `secrets` / NumPy；`NoisePool` 已实现但未接线 |
| 分块并行 | 无锁分块多核 | NumPy 向量化，GIL 外释放；无多核分块调度 |
| 预算共享 | PG 等持久后端 | Redis / SQLite / 内存 |
| 审计链 | 统一存证 | 本地 HMAC JSONL，不进入 audit-log 服务 |
| JIT | — | `dp_jit.py` 为 NumPy 向量化，不依赖 Numba |

相关文档：[api.md](../api.md) · [architecture.md](../architecture.md) · [high_concurrency/README.md](../high_concurrency/README.md) · [算法优化.md](../算法优化.md)
