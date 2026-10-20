# Python 医疗数据治理运维与调优指南 (Ops)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 内存与缓存配置

- **LRU / FIFO 抹平词缓存**：`_sanitized_cache` 与 `_ner_cache` 默认容量 2,048，高频常见病历短语命中时耗时 $< 0.05\text{ms}$。
- **并发推理防护**：对于 NER 实体识别，采用 Double-Checked Locking 机制，先查缓存，未命中时在锁外执行推理，避免由于 GIL 锁导致整个 Python 进程停顿。

---

## 2. 告警阈值

- **门禁触发告警 (`Fail-Safe Triggered`)**：若 `fail_safe_triggered` 计数 $>0$，说明正则抹平或 NER 存在未捕获的漏网高敏词，应触发 P2 级告警，并将未清洗词录入词库补充。
