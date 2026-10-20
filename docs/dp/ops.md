# Python 差分隐私（DP / LDP）运维与调优指南 (Ops)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 运行时配置与环境变量

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `PRIVACY_DP_DEFAULT_EPSILON` | float | `1.0` | 默认全局单次查询 $\epsilon$ 上限 |
| `PRIVACY_DP_DEFAULT_DELTA` | float | `1e-5` | 默认全局单次查询 $\delta$ 上限 |
| `PRIVACY_BUDGET_BACKEND` | string | `memory` | 预算持久化后端：`memory` / `redis` / `sqlite` |
| `PRIVACY_BUDGET_REDIS_URL` | string | `""` | Redis 记账实例连接串 |
| `PRIVACY_BUDGET_SQLITE_PATH` | string | `budget.db` | SQLite 存储路径 |

---

## 2. 监控指标 (Prometheus Metrics)

| 指标名称 | 类型 | 标签 | 含义与调优建议 |
|---|---|---|---|
| `privacy_dp_operations_total` | Counter | `operation`, `mechanism` | 加噪操作总次数 |
| `privacy_dp_budget_consumed` | Counter | `namespace`, `metric="epsilon" / "delta"` | 累积消耗的隐私预算 |
| `privacy_dp_budget_exhaustions_total` | Counter | `namespace` | 预算耗尽拦截次数（激增表示存在恶意爬取或配额过低） |
| `privacy_dp_duration_seconds` | Histogram | `operation` | 加噪计算耗时分布 |

---

## 3. 运维实战与故障诊断

### Q1: 频繁报 `BUDGET_EXHAUSTED` (429)
- **原因分析**：调用方在短时间内对同一 `namespace` 发起高频聚合查询，总累计消耗突破初始设定的 `epsilon_total`。
- **应急处理**：
  1. 通过 `GET /v1/privacy/budget?namespace=<ns>` 查看消耗历史。
  2. 若为合法扩容业务，可通过管理员权限调用 `POST /v1/privacy/budget/reset` 或重新配置增量配额。
  3. 推进调用方合并查询或使用低消耗机制（如增大批处理颗粒度）。
