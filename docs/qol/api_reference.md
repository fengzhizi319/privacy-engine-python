# Python 查询混淆 API 参考手册 (API Reference)

> **组件归属**：`services/privacy-engine-python`
> **路由实现**：[`engine/routers/qol.py`](../../engine/routers/qol.py)
> **权限要求**：`privacy:qol`

---

## 1. 端点概览

| HTTP 方法 | 路径 | 权限 Scope | 描述 |
|---|---|---|---|
| `POST` | `/v1/privacy/qol/obfuscate` | `privacy:qol` | 单条查询混淆 |
| `POST` | `/v1/privacy/qol/obfuscate/batch` | `privacy:qol` | 批量查询混淆 |

---

## 2. 请求与响应格式

### POST `/v1/privacy/qol/obfuscate`
```json
// 请求
{
  "query": "高血压患者日常用药注意事项",
  "num_decoys": 4,
  "domain": "medical"
}

// 响应 (200 OK)
{
  "queries": [
    "糖尿病患者日常用药注意事项",
    "哮喘患者日常用药注意事项",
    "高血压患者日常用药注意事项",
    "冠心病患者日常用药注意事项",
    "慢性胃炎患者日常用药注意事项"
  ],
  "real_index": 2,
  "result": [ ... ],
  "domain": "medical",
  "strategy": "slot_filling"
}
```
- `num_decoys`（或 `num_dummies`）：混入的诱饵数量。
- `real_index`：真实查询在返回列表中的 0 基索引位置。
