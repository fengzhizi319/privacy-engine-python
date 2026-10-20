# 数盾 PrivShield · Python 隐私计算引擎 — 测试规范与用例清单

> **组件归属**：`services/privacy-engine-python`
> **覆盖范围**：`tests/` 下 74 个被收集文件、**1278 个用例**（`pytest --collect-only` 实测，2026-10-19）；`tests/scripts`、`tests/deployment` 由 `collect_ignore` 排除
> **配套文档**：[api.md](api.md) · [architecture.md](architecture.md) · [config.md](config.md) · [ops.md](ops.md) · [prometheus.md](prometheus.md)
> **事实源**：`pyproject.toml`（pytest/coverage/ruff/mypy）、`tests/conftest.py`、`tests/conformance/conftest.py`、`contracts/conformance/`
> **口径**：文档与代码不一致时以代码为准；新增/删除用例后请同步更新本文计数。

---

## 1. 运行方式

### 1.1 入口

| 入口 | 命令 | 说明 |
|---|---|---|
| 模块全量（推荐） | `make test` | `pytest tests/ -q --tb=short`，含 integration/slow 标记用例 |
| 快速单测 | `make test-unit` | `-m "not integration and not slow"` |
| 仅集成 | `make test-integration` | `-m integration` |
| 覆盖率 | `make test-cov` | `--cov=engine --cov-report=term-missing`，排除 integration/slow |
| 跨语言契约 | `make conformance` | `pytest tests/conformance`，使用本仓库 `contracts/conformance/` |
| 双端契约回归 | 上游 `bash scripts/dev/run-conformance.sh` | 在上游 monorepo 执行，串联 Go/Python/Java 契约套件 |
| 样式 / 类型 | `make check` | `ruff check engine/ tests/` + `mypy` + 终结器 `go vet` |

环境准备：

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

`dev` extras 固定包含：`pytest>=8`、`pytest-asyncio`、`pytest-cov`、`pytest-benchmark`、`hypothesis`、`scipy`、`locust`、`ruff`、`mypy`、`grpcio-tools==1.82.1`。

### 1.2 pytest 配置（`pyproject.toml`）

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = ["integration", "slow", "real_models", "benchmark"]
addopts = "-q --tb=short"
```

仓库根目录的 CI 不额外注入 Python 专用环境变量；测试所需的开关由各用例通过 `monkeypatch`/`os.environ` 显式设置，不依赖宿主 shell。

### 1.3 覆盖率门禁

```toml
[tool.coverage.run]
source = ["engine"]
omit = [
  "engine/privacy_pb2*", "engine/privacy_pb2_grpc*", "*/tests/*",
  "engine/server.py", "engine/gateway/server.py",
  "engine/privacy/download_model.py", "engine/privacy/download_ner_model.py",
]
[tool.coverage.report]
fail_under = 80
```

- **`fail_under = 80`**：低于 80% 时 `pytest-cov` 返回非零；
- 进程入口与模型下载脚本被显式排除，不把无法单测的 `main()` 装配算作缺口；
- `test-cov` 默认跳过 integration/slow，覆盖率数字是「快速单测口径」，不是全量口径。发布前若需要全量数据，显式运行：

```bash
.venv/bin/python -m pytest tests --cov=engine --cov-report=term-missing
```

---

## 2. 目录与职责

| 目录 | 收集用例 | 职责 |
|---|---:|:---|
| `tests/api/` | 95 | REST/gRPC 端点契约、错误信封、文件上传、运维诊断 |
| `tests/privacy/` | 291 | 脱敏、DP、LDP、K-匿名、QOL、预算、数据适配器、高并发优化 |
| `tests/dynclassification/` | 372 | 三层漏斗、规则/操作符、画像生成、MLX/vLLM/LLM 适配、图像分类、真实模型冒烟 |
| `tests/security/` | 128 | API Key、gRPC 鉴权、mTLS/TLS、CN 白名单、限流 |
| `tests/observability/` | 59 | 错误信封、gRPC tracing、TLS tracing、指标/日志 |
| `tests/gateway/` | 57 | 负载均衡、熔断、HTTP/gRPC 代理、回源 TLS、网关服务装配 |
| `tests/conformance/` | 56 | 跨语言共享契约（50 primitives + 3 REST + 3 gRPC） |
| `tests/perf/` | 17 | Locust 任务契约、医疗流水线性能、内存泄漏回归 |
| `tests/*.py`（根级） | 194 | 服务层、医疗流水线、命名规范、高并发全家桶、启动器、审计整改 |
| `tests/scripts/` | **0（排除）** | 旧 Monorepo 根 `scripts/` 验证；由 `collect_ignore` 排除 |
| `tests/deployment/` | **0（排除）** | 旧 Helm/K8s 资产验证；由 `collect_ignore` 排除 |

> `tests/conftest.py` 中的 `collect_ignore = ["scripts", "deployment"]` 是有意为之：这两个目录验证的是仓库根 `scripts/` 与 `deploy/` 的历史资产，不属于本服务的功能一致性基线。文件保留作为迁移参考，**不要**把它们计入覆盖率或"全绿"结论。

### 2.1 关键用例文件

- `tests/api/test_rest.py`、`test_grpc.py`：走真实 FastAPI `TestClient` 与 `PrivacyServicer`，断言 5 字段信封与状态码；
- `tests/api/test_ops.py`：`/v1/ops/diagnostics` 字段、`refresh=true` 的 `ops:admin` 校验；
- `tests/privacy/test_budget*.py`：SQLite/Redis 记账、并发原子性、重启恢复；
- `tests/privacy/test_dp_property.py`：Hypothesis 属性测试（无偏性/边界）；
- `tests/privacy/test_qol_semantic.py`：医疗诱饵的语义/结构不变量；
- `tests/dynclassification/test_funnel.py`：置信度衰减、仲裁、覆盖抑制；
- `tests/dynclassification/test_real_models.py`：真实 NER/Qwen 模型冒烟（默认跳过）；
- `tests/gateway/test_gateway.py`：起本地上游的集成测试（`integration` 标记）；
- `tests/security/test_security_tls.py`、`test_security_urllib_tls.py`：真实 TLS listener（`integration` 标记）。

---

## 3. 夹具与隔离纪律

### 3.1 全局预算重置

`tests/conftest.py` 提供 `autouse=True` 的 `reset_all_budgets`：每个用例前清空 `default_registry`，并同步 `engine.main.service` 持有的 `dp_api.budget` 引用。

差分隐私预算是有状态全局资产；新增涉及预算的用例**不要**自己清理注册表，依赖该夹具即可。若绕开它（例如直接构造独立 `BudgetAccountant`），必须在用例内用 `try/finally` 清理。

### 3.2 scipy/torch 兼容补丁

部分 LLM 用例用 `MagicMock` 替换 `torch`，会触发 scipy 的 `is_torch_array()` 类型检查异常。`conftest.py` 在导入期包装该函数，保证收集与执行阶段安全。这是测试环境补丁，不改变生产路径。

### 3.3 进程内优先

- REST：`fastapi.testclient.TestClient`；
- gRPC：多数用例直接调用 `PrivacyServicer` 方法；需要验证拦截器/端口时使用真实 `grpc.server`（`integration` 标记）；
- 网关：上游用本地 FastAPI/HTTP server 或 mock，不依赖真实隐私引擎；
- 文件与 DICOM：写入 `tmp_path`，不读取仓库中的二进制夹具。

### 3.4 可选依赖门控

| 条件 | 行为 |
|---|---|
| `pandas` / `pyarrow` / `scipy.sparse` 缺失 | `pytest.importorskip(...)`，用例 Skip 而非失败 |
| `.models/` 不存在 | `test_real_models.py` 整文件 Skip（`skipif`） |
| `slow` / `real_models` | 需显式运行；`make test-unit` 与 `test-cov` 均排除 |
| 平台无 `SO_REUSEPORT` | `test_launcher.py` 对应用例 Skip |
| `helm` 未安装 | `tests/deployment` 内 Helm 用例 Skip（且该目录默认不收集） |

**门控原则**：Skip 必须写明原因，且不能让关键安全断言静默降级。若一个用例在缺依赖时 Skip、在有依赖时才验证，新增门控时同步登记到本节。

---

## 4. 跨语言一致性契约（Conformance）

### 4.1 事实源与目录

```text
contracts/conformance/
  manifest.yaml          # suite 注册表（primitives/rest/grpc 均 active）
  primitives/*.json      # 50 个原语用例
  rest/*.json            # 3 个 REST 回放用例
  grpc/*.json            # 3 个 gRPC 方法用例
```

Go 参考实现 `services/privacy-engine/conformance/` 与 Python `tests/conformance/` **读取同一批 JSON**。规则：

1. **Go 是共享行为的事实源**；夹具必须由 Go 行为生成，禁止手写期望值去迁就 Python；
2. 随机算法（DP/LDP/QOL）只固化不变量、边界与机制，不冻结随机样本；
3. 共享契约变更必须新增/更新夹具，**不得削弱既有夹具**来让实现变绿；
4. Python 独有能力（MLX、Arrow IPC、SQLite/Redis 预算、额外流水线）不进入共享契约，除非 Go 也暴露同样行为。

### 4.2 用例种类

| `kind` | 断言 |
|---|---|
| `exact` | 确定性结果精确相等（浮点按 `tolerance`） |
| `exact_unordered` | 结果集合按多重集比较 |
| `invariant` | 仅断言 `expect.invariants`（噪声尺度、边界、保留率等） |
| `error` | 两端都必须拒绝；`error_contains` 匹配错误文本 |

### 4.3 执行

```bash
# 本仓库跑 Python 消费方（夹具在 contracts/conformance/）
make conformance

# 或直接调用 pytest
.venv/bin/python -m pytest tests/conformance -q
```

新增契约时：

1. 在 Go 侧生成行为与夹具；
2. 在 `contracts/conformance/manifest.yaml` 注册（新 suite 时）；
3. Python 侧在 `tests/conformance/test_primitives.py` 的 `run_primitive()` 补充 `op` 分派；
4. 运行 `make conformance`；夹具同步到上游后，再在上游执行双端脚本确认两端同时绿。

> 当前 Python 契约执行是**进程内**调用：REST 用 `TestClient`，gRPC 直接调 `PrivacyServicer`。它验证业务语义，不替代真实端口上的 mTLS/限流测试（后者见 `tests/security`、`tests/gateway`）。

---

## 5. 安全负向测试基线

| 主题 | 文件 | 关键断言 |
|---|---|---|
| API Key | `tests/security/test_security_auth.py`、`test_security_unit.py` | 缺失/无效 Key → 401；scope 不足 → 403；常量时间比对路径 |
| `/metrics` | `tests/observability`、`tests/api` | 认证开启时需 `ops:diagnostics`；401/403 纯文本；`/metrics` 不计入自身 RED 指标 |
| gRPC 鉴权 | `tests/security/test_security_grpc.py` | `AuthInterceptor` 拒绝无凭据/越权；健康方法白名单 |
| mTLS / 白名单 | `tests/security/test_security_tls.py`、`test_security_whitelist.py` | 空白名单 fail-closed；CN 未命中拒绝；证书校验失败拒绝 |
| 限流 | `tests/security/test_security_rate_limit.py` | 超限 429；存储故障 fail-open 且 `reason="rate_limit_storage_error"` 计数 |
| 错误信封 | `tests/observability/test_envelope.py` | 5 字段、状态码映射、trace_id 透传；422 默认不泄露字段细节 |
| 请求体上限 | `tests/api/test_process_file.py`、`test_rest.py` | 32 MiB 全局体限制；上传 10 MiB 流式上限 |

这些用例不依赖外部 TLS 材料时使用 `tmp_path` 现场生成证书；依赖真实 listener 的用例标记 `integration`。

---

## 6. 性能与内存回归

| 文件 | 内容 | 运行建议 |
|---|---|---|
| `tests/perf/locustfile.py` + `test_locust_runner.py` | 将 Locust 任务挂到 `TestClient`，保证压测脚本契约不腐烂 | 默认收集 |
| `tests/perf/test_medical_pipeline_perf.py` | 医疗流水线吞吐/时延回归 | 默认收集，可能较慢 |
| `tests/perf/test_memory_leak.py` | `tracemalloc` 验证脱敏 5000 条、规则缓存、预算事务不无界增长 | 默认收集 |
| `tests/benchmark_primitives.py` | 原语基准脚本（非 `test_` 前缀，不被收集） | 手动执行 |
| `tests/dynclassification/test_mlx_benchmark.py` | Metal GPU vs CPU（`benchmark` 标记） | 仅在有 MLX 的 macOS |
| `tests/dynclassification/test_real_models.py` | 真实 NER/LLM 冒烟（`slow` + `real_models`） | 手动，需 `.models/` |

长稳/容量结论以仓库根 `scripts/dev/bench-all.sh`（Go 全栈）与部署侧压测为准；Python 侧这些用例是回归护栏，不产出对外性能承诺数字。

---

## 7. 代码质量门禁

```bash
make lint       # ruff check engine/ tests/
make format     # ruff format + --fix（会改文件）
make typecheck  # mypy
make check      # lint + typecheck
```

- **ruff**：`target-version=py313`、`line-length=120`，启用 `E/W/F/I/N/UP/B/S/A/C4/SIM/TCH/RUF`；测试中允许 `assert`（`S101`）与容器监听全部接口（`S104`）；
- **mypy**：`python_version=3.13`、Pydantic 插件、`check_untyped_defs=true`；生成的 `privacy_pb2*` 跳过导入分析；
- 提交前建议安装 `pre-commit`（`dev` extras 已包含）。

新增路由时，除 Python 测试外还需确认其权限映射：本服务在路由上显式声明 `require_permission(...)`；跨语言路由清单与 Go 的对应关系见 [api.md](api.md)。

---

## 8. 已知缺口与边界

| 项 | 现状 |
|---|---|
| TLCP / 国密抓取 | Python 进程只支持标准 TLS 与 mTLS；TLCP 由 Go 终结器终结，Go 侧有单测（`make terminator-check`），无 Python 抓取级用例（见 [tlcp.md](tlcp.md)） |
| 独立运维口 | 已有 `:9079`（业务口 +1000）与白名单测试 `tests/security/test_ops_port.py`；终结器拓扑端到端探针未纳入 pytest |
| 网关指标导出 | 网关独立运维口已导出 `/metrics`；抓取级断言仍以 `tests/gateway/` 单测为主 |
| 策略清单（policy bundle） | 未实现清单验真/原子发布，无对应测试 |
| IAM / MFA / RBAC 用户生命周期 | `tests/auth/` 覆盖登录、MFA、动态 Key、令牌交换与 SQLite 运行时；Redis 多实例与集群会话未覆盖 |
| Fuzz | 无 Go 式 `Fuzz*` 目标；`hypothesis` 仅在 DP 属性测试中使用 |
| L-Diversity | Python K-匿名仅覆盖 Mondrian K-匿名，无 L-Diversity 用例 |
| `tests/scripts`、`tests/deployment` | 默认不收集，不能作为质量证据引用 |

修改上述能力时，先补契约夹具与负向用例，再改实现；不要把"未收集的旧测试"当作已覆盖。

---

## 9. 新增用例检查单

1. **放对目录**：API 契约进 `tests/api`，原语进 `tests/privacy`，漏斗进 `tests/dynclassification`，安全负向进 `tests/security`；
2. **状态隔离**：依赖全局预算/注册表时使用 autouse 夹具或显式清理；修改环境变量用 `monkeypatch`；
3. **安全断言必须负向**：新端点至少覆盖未认证、越权、非法入参三类；
4. **随机算法只断言不变量**：写明噪声尺度与统计容忍度，不比较单次样本；
5. **共享行为加夹具**：能被 Go 复现的行为放入 `contracts/conformance` 并跑双端脚本；
6. **可选依赖用 `importorskip`**，并注明原因；
7. **更新本文计数**：`pytest --collect-only` 后同步 §2 与文首数字。
