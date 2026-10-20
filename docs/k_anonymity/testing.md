# Python K-匿名与 L-多样性测试指南 (Testing)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 测试套件结构

| 测试文件 | 职责范围 |
|---|---|
| `tests/privacy/test_kano_table.py` | Mondrian 表级分区、区间/集合泛化、退化边界、Arrow 导出测试 |
| `tests/privacy/test_kano_table_internal.py` | 内部中位数切分点算法、跨度评估、异常边界 |
| `tests/privacy/test_ldiversity.py` | L-多样性合规审计、防同质性切分拒绝与放行测试、REST 路由集成测试 |
| `tests/conformance/test_rest.py` | 跨语言 Go / Python REST 契约一致性测试 |

---

## 2. 自动化执行

```bash
# 1. 运行全部 K-匿名与 L-多样性单测
.venv/bin/python -m pytest tests/privacy/test_kano* tests/privacy/test_ldiversity.py -v

# 2. 运行跨语言契约测试 (Conformance)
make conformance
```

---

## 3. 核心断言设计

1. **等价类大小不变性**：
   $$\forall \text{group} \in \text{EquivalenceClasses}, \quad |\text{group}| \ge k$$
2. **L-多样性相异度约束**：
   $$\forall \text{group} \in \text{EquivalenceClasses}, \quad |\{ \text{record}[\text{sensitive}] \mid \text{record} \in \text{group} \}| \ge l$$
3. **确定性回归**：在固定随机种子下，两次相同输入的输出字符串必须 100% 相同。
