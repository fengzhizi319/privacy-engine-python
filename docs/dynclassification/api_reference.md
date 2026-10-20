# Python 动态分类分级 API 参考手册 (API Reference)

> **组件归属**：`services/privacy-engine-python`
> **路由实现**：[`engine/routers/dynclassification.py`](../../engine/routers/dynclassification.py)
> **权限要求**：`dynclassification:read`、`dynclassification:write`

---

## 1. 端点概览

| HTTP 方法 | 路径 | 权限 Scope | 描述 |
|---|---|---|---|
| `POST` | `/v1/dynclassification/classify` | `dynclassification:read` | 单字段分类分级评估 |
| `POST` | `/v1/dynclassification/classify/batch` | `dynclassification:read` | 批量字段分类评估 |
| `POST` | `/v1/dynclassification/eval` | `dynclassification:read` | 记录评估（支持指定标准与领域） |
| `POST` | `/v1/dynclassification/eval_record` | `dynclassification:read` | 记录评估同义别名 |
| `POST` | `/v1/dynclassification/eval_table` | `dynclassification:read` | 整表评估（字段聚合密级） |
| `POST` | `/v1/dynclassification/dry_run` | `dynclassification:read` | 规则干跑测试（暴露中间匹配明细） |
| `GET` | `/v1/dynclassification/standards` | `dynclassification:read` | 查询当前加载的所有合规标准元数据 |
| `GET` | `/v1/dynclassification/domains` | `dynclassification:read` | 查询当前加载的所有行业领域 |
| `GET` | `/v1/dynclassification/operators` | `dynclassification:read` | 查询已注册的可用算子列表 |
| `POST` | `/v1/dynclassification/profiles/reload` | `dynclassification:write` | 清空规则缓存并重新加载 |
| `POST` | `/v1/dynclassification/validate` | `dynclassification:read` | 验证指定规则目录结构与语法 |

---

## 2. 请求与响应示例

### 2.1 POST `/v1/dynclassification/eval`
```json
// 请求
{
  "record": {
    "patient_name": "王小明",
    "id_card": "510101199801012345",
    "diagnosis": "急性心肌梗死"
  },
  "standard": "sc_health_db51",
  "domain": "medical"
}

// 响应 (200 OK)
{
  "record_level": "L4",
  "domain": "medical",
  "standard": "sc_health_db51",
  "fields": {
    "patient_name": {
      "level": "L3",
      "category": "个人基本信息",
      "rule_id": "rule_medical_name",
      "confidence": 0.95,
      "layer": "rule"
    },
    "id_card": {
      "level": "L4",
      "category": "个人重要身份凭据",
      "rule_id": "rule_id_card_strict",
      "confidence": 1.0,
      "layer": "rule"
    },
    "diagnosis": {
      "level": "L4",
      "category": "严重慢性疾病与急性重症",
      "rule_id": "rule_critical_diagnosis",
      "confidence": 0.9,
      "layer": "rule"
    }
  }
}
```
