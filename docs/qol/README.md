# 查询混淆（QoL）模块

> 代码：`engine/privacy/qol.py` · REST：`/v1/privacy/qol/obfuscate*` · 测试：`tests/privacy/test_qol.py`、`test_qol_semantic.py`

## 定位

在真实查询中混入语义相近的虚假查询（dummy），使观察者无法仅凭查询流量判断用户真实意图。适用于检索/问诊入口的流量级隐私保护，不改变查询结果本身。

## 代码地图

| 文件/符号 | 职责 |
|---|---|
| `ObfuscationDomain` | `medical` / `generic`，决定 dummy 池与实体词库 |
| `ObfuscationStrategy` | `slot_filling`（实体槽位替换）、`length_similarity`（长度相近抽样）、`hybrid`（两者混合） |
| `MEDICAL_DUMMY` / `GENERIC_DUMMY` | 各 20 条内置 dummy 查询；可通过请求参数扩展 |
| `DISEASES` / `ENTITIES` | 槽位替换使用的疾病与实体词库 |
| `obfuscate_query()` | 单查询混淆；返回列表或 `QoLResult` |
| `obfuscate_query_batch()` | 逐查询循环调用单查询实现 |
| `QoLResult` | `queries`、`real_query_index`、`domain`、`num_dummies`；`to_arrow()` 写入 `qol_metadata` |

## REST 接口

| 路径 | 说明 |
|---|---|
| POST `/v1/privacy/qol/obfuscate` | 单查询；返回 `{queries, real_index, result}` |
| POST `/v1/privacy/qol/obfuscate/batch` | 批量；返回 `{results: [[...], ...]}` |

请求字段兼容 Go 别名：`num_decoys` 优先于 `num_dummies`；`domain` 为空串或 `medical` 走医疗池，其它值统一走通用池。所需 scope 为 `privacy:qol`。

## 行为要点

- **策略选择**：先尝试语义槽位替换；生成数量不足时按长度相近从 dummy 池无放回抽样补齐；两种来源混合时内部标记为 `HYBRID`。
- **随机插入**：真实查询被随机插入混淆列表的随机位置，调用方必须按 `real_index` / `real_query_index` 取回真实查询。
- **随机源**：`seed=None`（默认）使用 `secrets.SystemRandom()`，密码学安全；显式 `seed` 使用可复现 PRNG，只能用于测试。
- **数量约束**：`num_dummies` 必须 ≥ 1。候选池足够时无放回抽样；不足时循环补齐并可能产生重复；池为空时实际生成数可能少于请求值（`QoLResult.num_dummies` 记录实际值）。
- **不是差分隐私**：QoL 只提供启发式流量混淆，无形式化隐私预算；不能替代 DP/LDP。
- **REST 的 `real_index`**：由 `queries.index(query)` 计算，若 dummy 与真实查询文本完全相同可能返回错误位置；需要严格定位时使用库 API 的 `QoLResult.real_query_index`。

## 测试

```bash
pytest tests/privacy/test_qol.py tests/privacy/test_qol_semantic.py -q
```

## 与 Go 的差异与边界

| 项 | Go 引擎 | Python 引擎 |
|---|---|---|
| dummy 注入 + 洗牌 | 有 | 有，并额外提供语义槽位替换 |
| 批量并行 | 分块并发 | 串行循环 |
| 形式化隐私保证 | 无 | 无 |

相关文档：[api.md](../api.md) · [architecture.md](../architecture.md)
