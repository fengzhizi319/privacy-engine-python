# Python 动态分类分级测试指南 (Testing)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 测试套件结构

- `tests/dynclassification/test_funnel.py`：三层漏斗串联、层级穿透、冲突仲裁、最高密级保守原则。
- `tests/dynclassification/test_standards_switching.py`：四川省 DB51、金融 JR/T 0197、广东省标准动态切换验证。
- `tests/dynclassification/test_operators.py`：14 个内置算子正向与负向边界测试。
- `tests/dynclassification/test_bundle.py`：`privshield-policy-bundle/v1` 清单 SHA-256 完整性验真与符号链接防护。

---

## 2. 自动化执行

```bash
.venv/bin/python -m pytest tests/dynclassification/ -v
```
