# 与 Go 参考实现的对齐口径（Parity）

> **组件归属**：`services/privacy-engine-python`
> **参考实现**：`services/privacy-engine`（共享行为的事实源）
> **验证入口**：`contracts/conformance/` + `make conformance`（上游另有跨语言脚本 `scripts/dev/run-conformance.sh`）

## 1. 判定原则

1. **共享端点默认行为以 Go 为准**：同路径、同方法、同默认参数、同错误信封、同权限 scope；除非本文明确列为语言边界或 Python 超集。
2. **确定性算子逐字节一致**：掩码、哈希、FPE、银行卡等直接比对完整输出。
3. **随机算子比对不变量**：DP/LDP/QOL 不固化随机样本，只断言机制、参数、边界、条数、范围与 ε 约束。
4. **Python 独有能力是超集**：允许存在，但不得改变共享端点的默认行为。

## 2. 共享能力对齐

| 领域 | 对齐内容 | 事实源 / 文档 |
|---|---|---|
| REST / gRPC 契约 | 路由、请求/响应字段、契约别名、错误信封、分页与幂等 | [api.md](./api.md) |
| 掩码与哈希 | 字段名感知、HMAC-SM3、SM4-FPE、银行卡前 6 后 4、`PRIVACY_HASH_LEGACY_SHA256` 显式回退 | [masking/README.md](./masking/README.md) |
| 差分隐私 | Laplace/Gaussian、count/sum/mean/histogram/groupby/vector/adaptive clip、原子预算核算 | [dp/README.md](./dp/README.md) |
| LDP | randomized response / categorical / projected frequency | [dp/README.md](./dp/README.md) |
| K-匿名 | 记录级泛化 `K>=1`、Mondrian 表级 `K>=2`、QIType 与泛化策略 | [k_anonymity/README.md](./k_anonymity/README.md) |
| 查询混淆 | 默认 3 条 decoy、`domain=medical`、Fisher-Yates 槽位洗牌 | [qol/README.md](./qol/README.md) |
| 分类分级 | 规则 → Small-NER → LLM 三层漏斗、算子、画像热重载、熔断 | [dynclassification/README.md](./dynclassification/README.md) |
| 医疗流水线 | 分类 + 脱敏单次请求、五阶段展示契约、PII 拦截 | [medical_pipeline/README.md](./medical_pipeline/README.md) |
| 文件处理 | CSV（BOM/50 MiB 上限）、Excel（XLSX 驻留 / Zip bomb 上限）、JSON、DICOM | [api.md](./api.md) |
| 认证与授权 | 静态 Key、凭据受众/契约/路径策略、沙箱、scope 语义、限流、PoP | [auth.md](./auth.md) · [security/README.md](./security/README.md) |
| IAM / MFA | 登录、TOTP 强制、恢复码、换绑、动态 Key、服务账号、令牌交换 | [auth.md](./auth.md) |
| 网关 | 六种调度策略、P2C-EWMA、熔断、REST/gRPC 代理、运维口 | [gateway_balancer/README.md](./gateway_balancer/README.md) |
| 运维与指标 | `:9079` 运维口、`/metrics`、`/ops/diagnostics`、Prometheus 指标同名语义 | [ops.md](./ops.md) · [prometheus.md](./prometheus.md) |
| 传输层 | 标准 TLS + 国密 TLCP（Go 终结器终结）、mTLS、身份验签 | [tlcp.md](./tlcp.md) |

跨语言夹具当前覆盖 **50 个原语 + 3 个 REST + 3 个 gRPC** 用例，Go 与 Python 消费同一份 JSON。

## 3. Python 超集（允许保留）

| 能力 | 说明 | 入口 |
|---|---|---|
| MLX / 本地推理 | Apple Silicon 本地 LLM/VLM 路径；`PRIVACY_LLM_BACKEND=local` | `engine/dynclassification/llm_engines.py` |
| Arrow IPC | `POST /v1/privacy/dp/arrow_ipc`，二进制聚合降低序列化开销 | `engine/routers/dp.py` |
| 预算后端 | Redis / SQLite 分布式或持久化预算；Go 使用进程内 + 可选 Redis | `engine/privacy/budget.py` |
| 网关主动健康检查 | `PRIVACY_GATEWAY_HEALTH_CHECK_INTERVAL` 周期探针与节点恢复 | `engine/gateway/balancer.py` |
| 网关额外指标 | `privacy_gateway_latency_seconds` / `healthy_nodes` / `node_admin_state` 等 | [prometheus.md](./prometheus.md) |
| 额外流水线入口 | 文件、Agent 批处理等 Python 专用路由 | [api.md](./api.md) |

超集能力不得成为共享端点的默认路径；关闭方式与默认值见 [config.md](./config.md)。

## 4. 语言边界与已知差异

| 边界 | 现状 | 处理 |
|---|---|---|
| TLCP | Python 进程内不实现；由 `terminator/` 的 Go 终结器终结，Python 监听回环 | [tlcp.md](./tlcp.md) |
| pprof | 无 Go 运行时二进制 profile 格式；运维入口保留但与 Go 不同构 | [ops.md](./ops.md) |
| L-Diversity | Python 已对齐（含 SDK 校验、感知切分与 REST 路由 `/v1/privacy/l_diversity/check`、`/v1/privacy/k_anonymize/ldiversity`） | [k_anonymity/README.md](./k_anonymity/README.md) · [k_anonymity/design.md](./k_anonymity/design.md) |
| 策略清单（bundle manifest） | Python 已实现清单加载与 SHA-256 验真 (`engine/dynclassification/bundle.py`) | [policy_bundles.md](./policy_bundles.md) |
| PostgreSQL 用户库 | 当前构建缺驱动，启动期显式拒绝；支持 memory / SQLite | [auth.md](./auth.md) |
| 集群会话 / 委托令牌验签 | Python 未实现；交换票据仅进程内 | [auth.md](./auth.md) |
| LoRA 训练 | 仅推理接入；不包含训练流水线 | [llmlora/README.md](./llmlora/README.md) |
| REST 默认绑定 | Python 默认 `0.0.0.0`，Go 默认 `127.0.0.1`；终结器拓扑必须显式绑定回环 | [config.md](./config.md) |

## 5. 新增一致性用例

1. 在 `contracts/conformance/{primitives,rest,grpc}/` 增加 JSON 夹具（schema 见 `contracts/conformance/manifest.yaml`）。
2. 确定性用例用 `kind: exact`；随机用例用 `kind: invariant` / `statistical`，禁止固化随机样本。
3. 上游 Go 侧在 `services/privacy-engine/conformance/` 登记消费逻辑；本仓库在 `tests/conformance/` 补 `op` / 路由分派，并同步 `contracts/conformance/` 副本。
4. 运行本仓库门禁：

```bash
make conformance
```

## 6. 最终验证

```bash
# Python 全量测试与跨语言夹具
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest tests/conformance -q

# Go 终结器
make terminator-check

# 上游 Go 参考实现（在上游 monorepo 执行）
# cd services/privacy-engine && go test ./... && go test ./conformance/...
# bash scripts/dev/run-conformance.sh
```
