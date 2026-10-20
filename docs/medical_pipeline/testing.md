# Python 医疗数据治理测试指南 (Testing)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 测试用例矩阵

- `tests/test_medical_pipeline.py`：测试四川医保、康养照护样本数据的全流程脱敏与分类分级。
- `tests/perf/test_medical_pipeline_perf.py`：大批量（500+）医疗文本下的吞吐量、内存边界与延迟回归。

---

## 2. 自动化执行

```bash
.venv/bin/python -m pytest tests/test_medical_pipeline.py -v
```
