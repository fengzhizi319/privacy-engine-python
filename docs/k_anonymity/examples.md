# Python K-匿名与 L-多样性代码范例 (Examples)

> **组件归属**：`services/privacy-engine-python`

---

## 1. Python SDK 本地调用范例

```python
from engine.privacy.kano_table import k_anonymize_table, mondrian_with_l_diversity
from engine.privacy.kano import check_distinct_l_diversity

# 原始数据样本
dataset = [
    {"age": 22, "zip": "10001", "disease": "流感"},
    {"age": 23, "zip": "10002", "disease": "流感"},
    {"age": 25, "zip": "10001", "disease": "发热"},
    {"age": 29, "zip": "10003", "disease": "肺炎"},
    {"age": 31, "zip": "10004", "disease": "胃炎"},
    {"age": 35, "zip": "10005", "disease": "肠炎"},
]

# 1. 执行带 L-多样性约束的 K-匿名泛化 (k=2, l=2)
result = mondrian_with_l_diversity(
    rows=dataset,
    qi_cols=["age", "zip"],
    sensitive_field="disease",
    k=2,
    l=2,
    return_details=True,
)

print(f"生成的等价类数量: {result.equivalence_classes_count}")
for row in result.value:
    print(row)

# 2. 事后合规审计
audit = check_distinct_l_diversity(
    records=result.value,
    qi_fields=["age", "zip"],
    sensitive_field="disease",
    l=2,
)
print(f"L-多样性合规状态: {audit.is_compliant}, 最小相异度: {audit.min_diversity}")
```

---

## 2. cURL REST API 调用范例

### 2.1 提交带 L-多样性约束的泛化任务
```bash
curl -X POST http://127.0.0.1:8079/v1/privacy/k_anonymize/ldiversity \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{
    "records": [
      {"age": "10", "disease": "Flu"},
      {"age": "10", "disease": "Flu"},
      {"age": "11", "disease": "Cold"},
      {"age": "11", "disease": "Fever"}
    ],
    "qi_cols": ["age"],
    "sensitive_field": "disease",
    "k": 2,
    "l": 2
  }'
```

### 2.2 审计数据集的 L-多样性
```bash
curl -X POST http://127.0.0.1:8079/v1/privacy/l_diversity/check \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{
    "records": [
      {"zip": "1000", "age": "20", "disease": "Flu"},
      {"zip": "1000", "age": "20", "disease": "Flu"},
      {"zip": "2000", "age": "30", "disease": "Cancer"},
      {"zip": "2000", "age": "30", "disease": "Diabetes"}
    ],
    "qi_fields": ["zip", "age"],
    "sensitive_field": "disease",
    "l": 2
  }'
```
