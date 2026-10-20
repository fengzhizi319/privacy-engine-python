# Python 查询混淆运维与监控指南 (Ops)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 运行时配置与词典

内置诱饵词典位于 `engine/privacy/qol.py`，支持动态加载扩展词库：
- `MEDICAL_DUMMY`：涵盖常见内科、外科、慢病、用药指南问答。
- `GENERIC_DUMMY`：日常政务、生活服务、通用知识库问答。

---

## 2. 监控指标

| 指标 | 说明 |
|---|---|
| `privacy_qol_operations_total` | 混淆调用总次数 |
| `privacy_qol_duration_seconds` | 混淆执行耗时 |
