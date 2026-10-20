# Python 差分隐私代码范例 (Examples)

> **组件归属**：`services/privacy-engine-python`

---

## 1. Python SDK 核心调用范例

```python
from engine.privacy.dp import dp_engine
from engine.privacy.budget import default_registry

# 1. 初始化租户预算 (总 epsilon=10.0)
accountant = default_registry.get_or_create("hospital_stat", epsilon_total=10.0, delta_total=1e-5)

# 2. 扣除预算并执行带噪求和
data = [35.2, 42.1, 28.5, 56.0, 61.2]
epsilon = 0.5
accountant.consume(epsilon=epsilon)

res = dp_engine.sum(
    values=data,
    epsilon=epsilon,
    lower_bound=0.0,
    upper_bound=100.0,
    mechanism="laplace",
)

print(f"真实总和: {res.raw_value}, 加噪总和: {res.value}, 噪声标度: {res.noise_scale}")
print(f"剩余预算: {accountant.remaining_epsilon}")
```

---

## 2. cURL REST API 调用范例

### 2.1 请求带噪计数
```bash
curl -X POST http://127.0.0.1:8079/v1/privacy/dp/count \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{
    "raw_count": 1420,
    "epsilon": 0.2,
    "mechanism": "laplace",
    "namespace": "user_daily_active"
  }'
```

### 2.2 本地二值随机响应
```bash
curl -X POST http://127.0.0.1:8079/v1/privacy/ldp/randomized_response \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{
    "value": true,
    "epsilon": 0.8
  }'
```
