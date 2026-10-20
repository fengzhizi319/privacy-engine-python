# PrivShield Python 隐私计算引擎

本仓库是 PrivShield 隐私计算引擎的 Python 实现，与 Go 参考实现
`services/privacy-engine`（上游仓库
[`fengzhizi319/PrivShield`](https://github.com/fengzhizi319/PrivShield)）共享 REST/gRPC 契约、
隐私原语语义、医疗与动态分类规则。共享端点的默认行为、错误码与响应形状以 Go 实现为准。

仓库自带 `proto/privacy.proto`、`rules/`、`config/`、`contracts/conformance/` 夹具副本，
并 vendor 了终结器所需的 Go `pkg/` 子集，可脱离上游 monorepo 独立构建与测试。

Python 独有能力（MLX 推理、Arrow IPC、Redis/SQLite 预算后端、额外流水线路由）作为**超集**保留。
国密 TLCP 由同目录的 Go 终结器 [`terminator/`](terminator/README.md) 提供：Python 进程内不实现 TLCP，
也不直接暴露非回环地址。策略清单验真、LoRA 训练、PostgreSQL 用户库、集群会话验签等 Go 能力仍未移植。
IAM/MFA 仅在配置用户库后端后条件接线，详见 [`docs/auth.md`](docs/auth.md) 与
[`docs/security/README.md`](docs/security/README.md)。

## 快速开始

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

python -m engine.server        # REST :8079 + gRPC :50051 + ops :9079 一体进程
python -m engine.grpc_server   # 仅 gRPC
uvicorn engine.main:app --port 8079
```

生产镜像与 TLCP 终结器：

```bash
make docker-core               # 轻量运行镜像
make docker-ml                 # 追加 torch / transformers / onnxruntime
make docker-terminator         # Go TLCP/mTLS 终结器（唯一 TLS 边界）
```

## 目录结构

```text
engine/                # 可导入的运行包
  main.py              #   FastAPI 应用装配
  server.py            #   REST + gRPC + ops 一体入口
  grpc_server.py       #   gRPC 服务与拦截器
  ops_port.py          #   独立运维口（业务口 +1000，路径白名单）
  privacy/             #   Masking / DP / LDP / K-匿名 / QOL / 预算
  dynclassification/   #   规则 → NER → LLM 三层分类漏斗
  medical_pipeline/    #   医疗分类与脱敏
  pipeline/            #   文件处理流水线
  gateway/             #   P2C（按在途连接数）网关与反向代理
  security/            #   API Key / mTLS / 终结器身份验签 / 限流 / CN 白名单
  observability/       #   日志 / 指标 / Tracing / 错误信封
  auth/                #   IAM/MFA（配置用户库后端后接线）
terminator/            #   Go TLCP/mTLS 终结器（独立模块，GOWORK=off）
  third_party/pkg/     #     终结器依赖的 Go pkg/ 子集（config/tlsutil/crypto）
contracts/conformance/ #   与 Go 参考实现共享的跨语言 JSON 夹具
tests/                 #   单元、集成与跨语言一致性测试
rules/                 #   分类规则、标准与词表（运行时热加载）
config/                #   运行配置与场景 profile（不含私钥）
deploy/                #   终结器 compose / K8s sidecar 示例
proto/privacy.proto    #   gRPC 契约唯一事实源
```

## 测试与质量门禁

```bash
make test              # pytest 全量
make test-unit         # 排除 integration / slow
make lint              # ruff
make typecheck         # mypy
make check             # lint + typecheck + terminator-lint
make conformance       # 与 Go 参考实现共享的 JSON 夹具
make terminator-check  # 终结器 go vet + go test（GOWORK=off）
```

跨语言一致性夹具位于本仓库 `contracts/conformance/`（同步自上游共享契约），由 Go 与 Python
测试消费；确定性原语做精确断言，DP/LDP/QOL 等随机原语只断言机制、边界与分布不变量。
终结器不写入任何 `go.work`，始终以 `GOWORK=off` 独立构建。

## 配置

环境变量统一使用 `PRIVACY_*` 前缀；完整变量表与默认值见上游仓库的 `.env.example` 与
`docs/architecture/服务间认证.md` §12。

关键默认值：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRIVACY_REST_HOST` / `PRIVACY_REST_PORT` | `0.0.0.0` / `8079` | REST 监听 |
| `PRIVACY_GRPC_HOST` / `PRIVACY_GRPC_PORT` | `0.0.0.0` / `50051` | gRPC 监听 |
| `PRIVACY_OPS_HOST` / `PRIVACY_OPS_PORT` | 继承 REST / REST+1000 | 独立运维口（仅探针、指标与诊断） |
| `PRIVACY_TLS_STACK` | 未设置 | **引擎仅回显**；协议栈由终结器 `PRIVACY_TLS_STACK` 选择 |
| `PRIVACY_OPS_TLS_STACK` | 未设置 | 运维口档位；`tlcp`/`both` 要求绑定环回并由终结器前置 |
| `PRIVACY_DYNCLASSIFICATION_RULES_DIR` | `rules` | 分类规则目录（相对工作目录） |
| `PRIVACY_MAX_UPLOAD_BYTES` | `10485760` | 文件上传上限（10 MiB） |

引擎已拆分独立运维口（业务口 +1000），`/metrics`、`/ops/diagnostics` 与 `/debug/pprof/`
只在运维口白名单内可达；详见 [`docs/ops.md`](docs/ops.md)。
TLCP 由 Go 终结器承担，详见 [`terminator/README.md`](terminator/README.md) 与 [`docs/tlcp.md`](docs/tlcp.md)。

## 与 Go 参考实现的关系

- 契约事实源：`proto/privacy.proto`、Go `internal/rest/routes.go`、`sdk/*`。
- 共享端点默认行为不一致时，以 Go 为准；修复应同时补充跨语言夹具。
- 共享契约、Python 超集与语言边界的完整清单见 [`docs/parity.md`](docs/parity.md)。
- Hash/FPE 自 2.0.0 起切换为 SM3（HMAC-SM3 / FPE 计数器模式），与 Go 对齐；
  `PRIVACY_HASH_LEGACY_SHA256=true` 仅供一个版本的显式回退。
- 语言边界：TLCP 由 [`terminator/`](terminator/README.md) 的 Go 终结器终结，Python 进程内不实现国密；
  `/debug/pprof/*` 仅保证同名诊断入口，不伪造 Go profile 格式。
