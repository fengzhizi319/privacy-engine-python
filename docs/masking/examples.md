# Python 数据脱敏代码范例 (Examples)

> **组件归属**：`services/privacy-engine-python`

---

## 1. Python SDK 调用

```python
from engine.privacy.masking import mask_record, hash_value

record = {
    "name": "张无忌",
    "mobile": "13800138000",
    "id_card": "110101199503072345",
    "salary": "35000",
}

masked = mask_record(record)
print("脱敏后记录:", masked)

h = hash_value("user_1001", salt="enterprise_secret_salt_2026")
print("HMAC-SM3 哈希:", h)
```

---

## 2. cURL REST API 调用

```bash
curl -X POST http://127.0.0.1:8079/v1/privacy/mask/record \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{
    "record": {
      "user_name": "李四",
      "mobile": "13911112222",
      "card_no": "6222021000123456"
    }
  }'
```
