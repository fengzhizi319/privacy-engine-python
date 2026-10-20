# Python 数据脱敏运维与调优指南 (Ops)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 性能指标与监控

| 指标名称 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `privacy_masking_operations_total` | Counter | `operation`, `field_type` | 脱敏操作调用频次 |
| `privacy_masking_duration_seconds` | Histogram | `operation` | 脱敏计算耗时（P99 应 $< 5\text{ms}$） |

---

## 2. 性能瓶颈与调优建议

1. **表格脱敏推荐 PyArrow 路径**：
   在向 `/v1/privacy/mask/dataframe` 或内部 `mask_dataframe` 提交数据时，若输入数据为 pandas 或 PyArrow Table，底层会自动调用 C 绑定的字符串向量化替换，吞吐量比 Python 原生遍历高 15~20 倍。
2. **避免重复创建大字典**：
   在流式数据流水线中，推荐使用 `chunked_mask_records(records, chunk_size=1000)` 生成器，避免将整个百万行数据集一次性在内存中展开为 Python 字典列表导致 OOM。
