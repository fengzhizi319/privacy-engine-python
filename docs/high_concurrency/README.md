# 高并发组件（High Concurrency）

> 代码：`engine/privacy/high_concurrency.py`、`engine/privacy/dp_jit.py` · 测试：`tests/test_high_concurrency.py`、`tests/test_all_features_high_concurrency.py`

## 定位

为 10K+ QPS 场景准备的并发优化组件集合：噪声池、批量预算扣减、线程安全 LRU、异步动态批处理、并发节流。**目前只有 `HighConcurrencyLRUCache` 接入生产路径**（分类结果缓存）；其余组件已实现并有单测，但未接线到 REST/gRPC 请求路径。

## 代码地图

| 符号 | 设计目标 | 当前接线 |
|---|---|---|
| `HighConcurrencyLRUCache` | `OrderedDict` + `threading.Lock` 的容量受限 LRU，带 hits/misses/hit_rate 统计 | ✅ `DynClassificationService._classification_cache`（容量 `PRIVACY_CLASSIFICATION_CACHE_SIZE`，默认 10000） |
| `NoisePool` | 预生成 Laplace/Gaussian 噪声，O(1) 切片取用，耗尽自动 refill；声称 ~80x 采样加速 | ❌ 未使用；`DPApi` 仍实时采样 |
| `BatchedBudgetSpend` | 1ms 窗口合并多个 `spend` 为一次预算锁操作，降低锁竞争 | ❌ 未使用；`BudgetAccountant.spend` 仍逐次加锁 |
| `AsyncDynamicBatcher` | 2ms 窗口把并发小请求合并为一次批处理（适合 NER/LLM 批推理） | ❌ 未使用 |
| `ConcurrencyThrottle` | 异步信号量背压，限制密集任务并发 | ❌ 未使用；LLM 另有进程级信号量（`PRIVACY_LLM_MAX_CONCURRENCY`） |
| `_SpendFuture` | `BatchedBudgetSpend` 的等待结果封装 | ❌ 仅内部使用 |
| `dp_jit.*` | NumPy 向量化 Laplace/Gaussian、裁剪、L2 裁剪 | ❌ 未被 `DPApi` 调用；`DPApi` 自身使用 NumPy 采样 |

> 注意：`BatchedBudgetSpend` 的模块 docstring 写作 `BatchedBudgetAccountant`，实际类名为 `BatchedBudgetSpend`。

## 使用方式（库级）

```python
from engine.privacy.high_concurrency import HighConcurrencyLRUCache

cache = HighConcurrencyLRUCache[str, dict](capacity=4096)
cache.put("k", {"v": 1})
cache.get("k")          # 命中并提升为最近使用
cache.stats             # {"capacity", "size", "hits", "misses", "hit_rate"}
```

`NoisePool` / `BatchedBudgetSpend` / `AsyncDynamicBatcher` 的构造与启动方式见各自 docstring；接入前必须评估：

- **NoisePool 复用噪声**：从预生成池取噪声会破坏每次查询独立采样的语义吗？Laplace/Gaussian 样本本身独立同分布，池化在统计上可行，但池耗尽/refill 的随机源与并发安全需要单独验证，且不能与显式 `seed` 的确定性测试混用。
- **BatchedBudgetSpend 延迟**：1ms 合并窗口会给每次 DP 查询增加固定延迟；超预算的返回语义变为 Future，调用方必须处理异常传播。
- **AsyncDynamicBatcher**：批处理回调失败会向整批 Future 传播异常；需要保证同批请求可合并（同 domain/standard、同模型）。
- **ConcurrencyThrottle**：限流是进程内信号量，多 worker 不共享。

## 性能模型

- REST 同步端点运行在 AnyIO 线程池；gRPC 使用 `PRIVACY_GRPC_MAX_WORKERS`（默认 64）线程池。
- NumPy/PyArrow 计算会释放 GIL，但规则求值、正则、逐元素脱敏仍是 Python 字节码。
- 真正的横向扩展靠多进程（`engine.launcher`，`PRIVACY_WORKERS`）；进程间缓存、预算内存后端、限流状态均不共享。
- `HighConcurrencyLRUCache` 是单锁结构，热点 key 下会串行化；容量应按工作集而非请求量设置。

## 测试

```bash
pytest tests/test_high_concurrency.py tests/test_all_features_high_concurrency.py -q
```

## 与 Go 的差异与边界

| 项 | Go 引擎 | Python 引擎 |
|---|---|---|
| 并行模型 | goroutine + 分块多核 | AnyIO/gRPC 线程池 + 多进程 |
| 噪声池 | 原生复用 | 已实现未接线 |
| 批量预算 | 原子核算 | 已实现未接线 |
| 缓存 | 分片/无锁结构 | 单锁 LRU |
| 指标聚合 | 进程内统一 | 多 worker 需外部聚合 |

相关文档：[算法优化.md](../算法优化.md) · [dp/README.md](../dp/README.md) · [architecture.md](../architecture.md) · [config.md](../config.md)
