# K-匿名（K-Anonymity）模块

> 代码：`engine/privacy/kano.py`、`kano_table.py` · REST：`/v1/privacy/k_anonymize*` · 测试：`tests/privacy/test_kano_table.py`、`test_kano_table_internal.py`

## 定位

面向准标识符（QI）的泛化原语，提供两条路径：

1. **记录级**：按字段类型选择泛化层级，逐条记录替换 QI 值（单机无分组保证）。
2. **数据集级**：Mondrian 多维递归分区，保证每个等价类至少 `k` 条记录，输出区间/集合泛化结果。

## 代码地图

| 文件/符号 | 职责 |
|---|---|
| `QIType` | 内置 QI 类型：`age`、`zipcode`、`gender`、`salary`、`education` |
| `GeneralizationStrategy` | `interval` / `set` / `suppression` / `prefix` |
| `BUILTIN_HIERARCHIES` | 类型 → 泛化函数映射 |
| `age_hierarchy` / `zipcode_hierarchy` / `gender_hierarchy` / `salary_hierarchy` / `education_hierarchy` | 内置层次实现；`adaptive_age_hierarchy` 可按 k 自动选层 |
| `choose_level(k, max_level)` | 启发式层级选择（约 `k // 5`，至少 1，不超过 `max_level`） |
| `anonymize_record()` | 单记录泛化；可传自定义 `hierarchies` |
| `anonymize_records_batch()` | 批量泛化；默认 60s 超时保护，超过 10k 行时启用 |
| `kano_table.k_anonymize_table()` | Mondrian 数据集级泛化；支持 pandas 向量化路径；`return_details=True` 返回 `KAnonymityResult` |
| `kano_table.k_anonymize_dataframe()` | pandas DataFrame 入口 |
| `kano.check_distinct_l_diversity()` | Distinct L-多样性事后检验，统计各等价类敏感属性相异基数 |
| `kano_table.mondrian_with_l_diversity()` | 带 L-多样性约束的感知切分 Mondrian 算法 |
| `kano_table.anonymize_with_l_diversity()` | 记录集级带 L-多样性约束泛化 |

## REST 接口

| 路径 | 说明 |
|---|---|
| POST `/v1/privacy/k_anonymize` | 数据集级 K-匿名（Go 契约：`records`/`result`/`k`/`group_count`） |
| POST `/v1/privacy/k_anonymize/record` | 单条记录泛化 |
| POST `/v1/privacy/k_anonymize/table`、`/v1/privacy/k_anonymize_table` | 表级 Mondrian |
| POST `/v1/privacy/k_anonymize/dataframe` | DataFrame 输入 |
| POST `/v1/privacy/l_diversity/check` | Distinct L-多样性事后审计接口 |
| POST `/v1/privacy/k_anonymize/ldiversity` | 带 L-多样性约束的 Mondrian K-匿名泛化接口 |

请求同时接受 Go 风格别名：`records`/`record`、`qi_fields`/`qi_cols`、`rows`/`data`。`k < 1` 统一收口为 2（与 Go `clampK` 对齐）。所需 scope 为 `privacy:kano`。

## 行为要点

- **Mondrian 分区**：每轮选择跨度最大的 QI 维度，在合法中位数处切分，左右子集均须 ≥ k；达到 `max_depth` 或无法合法切分时停止并对叶子泛化。
- **小于 k 的表**：不报错，整表作为一个等价类泛化（与 Go `kano.Anonymize` 对齐）。
- **数值 QI**：输出区间字符串；分类 QI：输出取值集合；无法泛化时按策略抑制。
- **记录级不是隐私保证**：`anonymize_record` 只按 k 选择层级，不验证数据集中同组记录数；需要可证明的 k-匿名时必须使用表级 Mondrian。
- **L-多样性支持**：支持 Distinct L-多样性检验（`check_distinct_l_diversity`）与感知切分的 Mondrian（`mondrian_with_l_diversity`），当切分会导致任意一侧敏感属性 distinct 数跌破 l 时拒绝切分，杜绝同质性攻击。
- **超时保护是协作式的**：批量路径按时间检查中止，不是硬性线程中断；超大输入仍可能占用线程较久。
- **PyArrow 导出**：`KAnonymityResult.to_arrow()` 在 Schema metadata 写入 `k_anonymity_metadata`（k、QI 列表、等价类数）；`KAnonymityRecordResult.to_arrow()` 写入 `k_anonymity_record_metadata`。

## 测试

```bash
pytest tests/privacy/test_kano_table.py tests/privacy/test_kano_table_internal.py -q
```

## 与 Go 的差异与边界

| 项 | Go 引擎 | Python 引擎 |
|---|---|---|
| Mondrian KD-tree | 原生实现 + 递归护栏 | pandas/纯 Python 递归实现 |
| L-Diversity | 有（distinct 校验） | **无** |
| 多核 | 分块并行 | 单线程（pandas 向量化） |
| 表级隐私证明 | 有 | 有 k-匿名，无 l/t 约束 |

相关文档：[api.md](../api.md) · [architecture.md](../architecture.md) · [算法优化.md](../算法优化.md)
