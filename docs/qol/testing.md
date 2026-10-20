# Python 查询混淆测试指南 (Testing)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 测试套件构成

- `tests/privacy/test_qol.py`：测试诱饵生成数量、`real_index` 定位、无放回抽样与置乱随机性。
- `tests/privacy/test_qol_semantic.py`：测试语义槽位替换、医学实体库匹配与混合抽样策略。

---

## 2. 自动化执行

```bash
.venv/bin/python -m pytest tests/privacy/test_qol*.py -v
```
