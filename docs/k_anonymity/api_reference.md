# Python K-匿名与 L-多样性 API 参考手册 (API Reference)

> **组件归属**：`services/privacy-engine-python`
> **路由实现**：[`engine/routers/kano.py`](../../engine/routers/kano.py)
> **请求模型**：[`engine/schemas.py`](../../engine/schemas.py)
> **权限要求**：`privacy:kano`

---

## 1. 端点概览

| HTTP 方法 | 路径 | 权限 Scope | 描述 |
|---|---|---|---|
| `POST` | `/v1/privacy/k_anonymize` | `privacy:kano` | 数据集级 K-匿名泛化（Mondrian 算法，Go 契约） |
| `POST` | `/v1/privacy/k_anonymize/record` | `privacy:kano` | 容器级单记录 K-匿名泛化别名 |
| `POST` | `/v1/privacy/k_anonymize/table` | `privacy:kano` | 表级 Mondrian K-匿名泛化 |
| `POST` | `/v1/privacy/k_anonymize_table` | `privacy:kano` | 表级 Mondrian K-匿名泛化（同义别名） |
| `POST` | `/v1/privacy/k_anonymize/dataframe` | `privacy:kano` | DataFrame 格式 K-匿名泛化 |
| `POST` | `/v1/privacy/l_diversity/check` | `privacy:kano` | Distinct L-多样性事后合规审计 |
| `POST` | `/v1/privacy/k_anonymize/ldiversity` | `privacy:kano` | 带 L-多样性约束的 Mondrian K-匿名泛化 |
| `POST` | `/v1/privacy/k_anonymize_ldiversity` | `privacy:kano` | 带 L-多样性约束的泛化（同义别名） |

---

## 2. 接口详细说明

### 2.1 POST `/v1/privacy/k_anonymize`
数据集级 K-匿名泛化。

#### 请求体 (JSON)
```json
{
  "records": [
    {"age": "25", "zipcode": "100001", "name": "张三"},
    {"age": "26", "zipcode": "100002", "name": "李四"},
    {"age": "25", "zipcode": "100001", "name": "王五"},
    {"age": "30", "zipcode": "100003", "name": "赵六"}
  ],
  "qi_fields": ["age", "zipcode"],
  "k": 2
}
```
- `records` (或 `record`): 输入记录列表。
- `qi_fields` (或 `qi_cols`): 准标识符字段名称数组。
- `k`: 目标 K-匿名阈值（$k < 1$ 时自动收口为 2）。

#### 成功响应 (200 OK)
```json
{
  "records": [
    {"age": "25", "zipcode": "100001", "name": "张三"},
    {"age": "25", "zipcode": "100001", "name": "王五"},
    {"age": "[20-40]", "zipcode": "100***", "name": "李四"},
    {"age": "[20-40]", "zipcode": "100***", "name": "赵六"}
  ],
  "result": [ ... ],
  "k": 2,
  "group_count": 2
}
```

---

### 2.2 POST `/v1/privacy/l_diversity/check`
校验数据集在给定准标识符和敏感属性下是否满足 Distinct L-多样性约束。

#### 请求体 (JSON)
```json
{
  "records": [
    {"zip": "1000", "age": "20", "disease": "Flu"},
    {"zip": "1000", "age": "20", "disease": "Flu"},
    {"zip": "2000", "age": "30", "disease": "Cancer"},
    {"zip": "2000", "age": "30", "disease": "Diabetes"}
  ],
  "qi_fields": ["zip", "age"],
  "sensitive_field": "disease",
  "l": 2
}
```
- `records`: 待审计的记录集。
- `qi_fields` (或 `qi_cols`): 准标识符字段列表。
- `sensitive_field`: 敏感属性列名。
- `l`: 要求的相异取值数下限，默认 2。

#### 成功响应 (200 OK)
```json
{
  "is_compliant": false,
  "l": 2,
  "min_diversity": 1,
  "violations": 1,
  "group_count": 2,
  "group_stats": [
    {
      "group_index": 1,
      "record_count": 2,
      "distinct_count": 1,
      "sensitive_values": {
        "Flu": 2
      },
      "is_compliant": false
    },
    {
      "group_index": 2,
      "record_count": 2,
      "distinct_count": 2,
      "sensitive_values": {
        "Cancer": 1,
        "Diabetes": 1
      },
      "is_compliant": true
    }
  ]
}
```

---

### 2.3 POST `/v1/privacy/k_anonymize/ldiversity`
在 Mondrian 多维切分时强制施加 L-多样性护栏。

#### 请求体 (JSON)
```json
{
  "records": [
    {"age": "10", "disease": "Flu"},
    {"age": "10", "disease": "Flu"},
    {"age": "11", "disease": "Cold"},
    {"age": "11", "disease": "Fever"}
  ],
  "qi_cols": ["age"],
  "sensitive_field": "disease",
  "k": 2,
  "l": 2,
  "max_depth": 10
}
```

#### 成功响应 (200 OK)
```json
{
  "records": [
    {"age": "[10-11]", "disease": "Flu"},
    {"age": "[10-11]", "disease": "Flu"},
    {"age": "[10-11]", "disease": "Cold"},
    {"age": "[10-11]", "disease": "Fever"}
  ],
  "result": [ ... ],
  "k": 2,
  "l": 2,
  "sensitive_field": "disease",
  "qi_cols": ["age"],
  "group_count": 1,
  "equivalence_classes_count": 1
}
```
> [!NOTE]
> 由于若按 age 切分会导致 age=10 的子集内 disease 只有 1 种取值（跌破 $l=2$），算法自动拒绝该切分并保留为一个包含 4 条记录的等价类，泛化 age 为 `[10-11]`，成功抵御同质性攻击。

---

## 3. 错误码标准信封

| HTTP 状态码 | 业务错误代码 (`code`) | 场景说明 |
|---|---|---|
| 400 | `INVALID_ARGUMENT` | 参数缺失（如缺少 records、qi_cols 或 sensitive_field） |
| 401 | `UNAUTHORIZED` | 凭据缺失或 Token 签名无效 |
| 403 | `FORBIDDEN` | 凭据缺少 `privacy:kano` 操作权限 |
| 422 | `UNPROCESSABLE_ENTITY` | Pydantic 校验失败（如类型不匹配、超出条数限制） |
| 500 | `KANONYMIZE_FAILED` | 引擎内部递归或内存分配错误 |
