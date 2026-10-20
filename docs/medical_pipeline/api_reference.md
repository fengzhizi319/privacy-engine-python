# Python 医疗数据治理流水线 API 参考手册 (API Reference)

> **组件归属**：`services/privacy-engine-python`
> **路由实现**：[`engine/routers/medical.py`](../../engine/routers/medical.py)
> **权限要求**：`medical:process` 或 `agent:process`

---

## 1. 端点概览

| HTTP 方法 | 路径 | 权限 Scope | 描述 |
|---|---|---|---|
| `POST` | `/v1/medical/sanitize` | `medical:process` | 单条医疗记录脱敏与抹平（Go 契约） |
| `POST` | `/v1/medical/sanitize/batch` | `medical:process` | 批量医疗记录脱敏与抹平 |
| `POST` | `/v1/agent/process` | `agent:process` | 标准全流程医疗处理（脱敏数据 + 审计报告） |

---

## 2. 请求与响应示例

### POST `/v1/agent/process`
```json
// 请求
{
  "records": [
    {
      "name": "李强",
      "id_card": "510101198505051234",
      "phone": "13800138000",
      "diagnosis": "确诊人类免疫缺陷病毒(HIV)感染伴肺部感染"
    }
  ],
  "domain": "yibao",
  "sanitize": true
}

// 响应 (200 OK)
{
  "sanitized_data": [
    {
      "name": "李*",
      "id_card": "510101********1234",
      "phone": "138****8000",
      "diagnosis": "确诊[IMMUNODEFICIENCY]感染伴肺部感染"
    }
  ],
  "classification_report": [
    {
      "record_index": 0,
      "max_level": "L5",
      "contains_pii": true,
      "field_details": {
        "diagnosis": {
          "level": "L5",
          "category": "极高敏感疾病",
          "rule_id": "rule_l5_hiv",
          "confidence": 0.99
        }
      }
    }
  ],
  "summary": {
    "total_records": 1,
    "l5_count": 1,
    "l4_count": 0,
    "pii_fields_masked": 3,
    "fail_safe_triggered": 0,
    "guarantee_no_l4_l5_raw_data": true
  }
}
```
