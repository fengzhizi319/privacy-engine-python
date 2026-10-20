# Python 差分隐私（DP / LDP）API 参考手册 (API Reference)

> **组件归属**：`services/privacy-engine-python`
> **路由实现**：[`engine/routers/dp.py`](../../engine/routers/dp.py) · [`engine/routers/budget.py`](../../engine/routers/budget.py)
> **权限要求**：`privacy:dp`、`privacy:budget`

---

## 1. 核心端点列表

| 方法 | 路径 | 权限 Scope | 说明 |
|---|---|---|---|
| `POST` | `/v1/privacy/dp/count` | `privacy:dp` | 带噪计数（扣除预算） |
| `POST` | `/v1/privacy/dp/sum` | `privacy:dp` | 带噪求和（数值裁剪 + 扣除预算） |
| `POST` | `/v1/privacy/dp/mean` | `privacy:dp` | 单遍带噪均值估计（扣除预算） |
| `POST` | `/v1/privacy/dp/noisy_count` | `privacy:dp` | 仅加噪计数（不扣除预算） |
| `POST` | `/v1/privacy/dp/noisy_sum` | `privacy:dp` | 仅加噪求和（不扣除预算） |
| `POST` | `/v1/privacy/dp/aggregate` | `privacy:dp` | 复合批量聚合查询 |
| `POST` | `/v1/privacy/dp/vector_sum` | `privacy:dp` | 多维向量聚合（$L_2$ 范数裁剪） |
| `POST` | `/v1/privacy/ldp/randomized_response` | `privacy:dp` | 二值随机响应扰动 |
| `POST` | `/v1/privacy/ldp/orr` | `privacy:dp` | 离散多分类 O-RR 本地扰动 |
| `POST` | `/v1/privacy/ldp/estimate/binary` | `privacy:dp` | 从扰动样本中无偏重构真实均值 |
| `GET` | `/v1/privacy/budget` | `privacy:budget` | 查询租户/命名空间当前可用隐私预算 |
| `POST` | `/v1/privacy/budget` | `privacy:budget` | 手动预扣/结算预算 |
| `POST` | `/v1/privacy/budget/reset` | `privacy:budget` | 重置指定租户的预算池 |

---

## 2. 接口参数与响应范例

### 2.1 POST `/v1/privacy/dp/sum`
```json
// 请求体
{
  "values": [12.5, 34.0, 56.2, 78.1, 99.0],
  "epsilon": 0.5,
  "lower_bound": 0.0,
  "upper_bound": 100.0,
  "mechanism": "laplace",
  "namespace": "finance_reports"
}

// 响应 (200 OK)
{
  "value": 281.42857142857144,
  "raw_value": 279.8,
  "epsilon": 0.5,
  "delta": 0.0,
  "mechanism": "laplace",
  "sensitivity": 100.0,
  "noise_scale": 200.0,
  "confidence_interval_95": [
    -318.14,
    880.99
  ]
}
```

### 2.2 POST `/v1/privacy/ldp/randomized_response`
```json
// 请求体
{
  "value": true,
  "epsilon": 1.0
}

// 响应 (200 OK)
{
  "value": true,
  "perturbed_value": true,
  "epsilon": 1.0,
  "flip_probability": 0.2689414213699951
}
```

---

## 3. 错误信封标准

当预算耗尽时，系统立即拒绝并返回统一信封：
```json
{
  "code": "BUDGET_EXHAUSTED",
  "message": "隐私预算已耗尽",
  "detail": "Requested epsilon 1.0 exceeds remaining budget 0.4 for namespace 'finance_reports'",
  "trace_id": "req-9876543210"
}
```
