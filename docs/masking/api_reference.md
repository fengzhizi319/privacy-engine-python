# Python 数据脱敏 API 参考手册 (API Reference)

> **组件归属**：`services/privacy-engine-python`
> **路由实现**：[`engine/routers/masking.py`](../../engine/routers/masking.py)
> **权限要求**：`privacy:mask`

---

## 1. 端点概览

| HTTP 方法 | 路径 | 权限 Scope | 说明 |
|---|---|---|---|
| `POST` | `/v1/privacy/mask` | `privacy:mask` | 单值自适应脱敏 |
| `POST` | `/v1/privacy/mask/record` | `privacy:mask` | 单条记录（Dict）批量字段脱敏 |
| `POST` | `/v1/privacy/mask_record` | `privacy:mask` | 单条记录脱敏（同义别名） |
| `POST` | `/v1/privacy/mask/batch` | `privacy:mask` | 批量字段与值列表脱敏 |
| `POST` | `/v1/privacy/mask_batch` | `privacy:mask` | 批量字段脱敏（同义别名） |
| `POST` | `/v1/privacy/mask/dataframe` | `privacy:mask` | 批量表/数据集脱敏 |
| `POST` | `/v1/privacy/hash` | `privacy:mask` | HMAC-SM3 / HMAC-SHA256 哈希去标识化 |

---

## 2. 请求与响应格式

### 2.1 POST `/v1/privacy/mask`
```json
// 请求
{
  "field_name": "mobile_phone",
  "value": "13912345678",
  "context": "export"
}

// 响应 (200 OK)
{
  "result": "139****5678",
  "field_name": "mobile_phone",
  "field_type": "mobile"
}
```

### 2.2 POST `/v1/privacy/mask/record`
```json
// 请求
{
  "record": {
    "user_name": "张三丰",
    "id_card": "110101199001011234",
    "email": "sanfeng@wudang.org",
    "remark": "无过敏史"
  }
}

// 响应 (200 OK)
{
  "result": {
    "user_name": "张*丰",
    "id_card": "110101********1234",
    "email": "s*****g@wudang.org",
    "remark": "无过敏史"
  }
}
```

### 2.3 POST `/v1/privacy/hash`
```json
// 请求
{
  "value": "secret_user_id_1001",
  "salt": "enterprise_secret_salt_2026"
}

// 响应 (200 OK)
{
  "result": "Gk7v9zLmNp2xQ1wA"
}
```
