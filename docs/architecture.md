# Python 隐私引擎 (privacy-engine-python) — 架构设计与技术全景

> **组件归属**：`services/privacy-engine-python`
> **定位**：与 Go `services/privacy-engine` 功能对齐的 Python 实现，额外承担 macOS MLX / 多格式文件 / Arrow 低开销链路等 Python 生态能力。
> **协议**：REST `:8079` + gRPC `:50051`，同一进程双协议；网关为独立进程 `:8000` / `:50000`。

---

## 1. 设计哲学

1. **契约优先**：跨语言共享行为由 `contracts/conformance/` 固定，Go 是事实源；Python 对确定算子逐字节对齐，对随机算子校验不变量。
2. **纯函数原语 + 有状态服务分层**：`engine/privacy/*` 的脱敏、DP、K-匿名、QOL 计算保持无状态；预算、缓存、连接池等状态收敛在 `engine/service.py`、`budget.py` 与 `high_concurrency.py`。
3. **默认安全关闭、生产显式开启**：TLS/Auth/RateLimit 默认全部关闭，启动横幅显式告警；开启后 fail-closed（空 mTLS 白名单拒绝全部证书、未知 gRPC 方法仅接受通配凭据）。
4. **重依赖懒加载**：torch / transformers / MLX / TensorRT / ModelScope 仅在分类漏斗真正需要时导入，核心镜像不携带重型 ML 依赖。
5. **失败可诊断**：统一错误信封、`/readyz`、`/v1/ops/diagnostics`、Prometheus 指标与结构化日志共享 request_id。

---

## 2. 总体架构与服务拓扑

```text
                        ┌────────────────────────────────────────────┐
                        │  service-hub / BFF / 外部调用方              │
                        └───────────────┬────────────────────────────┘
                                        │ HTTP / gRPC
                    ┌───────────────────▼───────────────────┐
                    │ privacy-engine-python 网关（可选）       │
                    │ engine.gateway.server                  │
                    │  REST :8000 / gRPC :50000 / Ops :9000  │
                    │  P2C-EWMA + 熔断 + 准入门禁 + 节点管理    │
                    └───────────────────┬───────────────────┘
                                        │ 回源（明文或 TLS/mTLS）
        ┌───────────────────────────────▼───────────────────────────────┐
        │ privacy-engine-python 主进程                                    │
        │ engine.server / engine.launcher                                │
        │  ┌─────────────────────┐      ┌──────────────────────────────┐ │
        │  │ FastAPI REST :8079  │      │ grpcio Server :50051         │ │
        │  │ 中间件栈 + 路由       │      │ Auth/RateLimit Interceptor   │ │
        │  └──────────┬──────────┘      └──────────────┬───────────────┘ │
        │             └──────────────┬─────────────────┘                 │
        │                     engine.service.PrivacyService              │
        │    ┌────────────┬──────────┼───────────┬──────────────┐        │
        │    ▼            ▼          ▼           ▼              ▼        │
        │  masking      DP/LDP    K-匿名/QOL   分类漏斗      医疗/业务流水线 │
        │  privacy/*    privacy/* privacy/*   dynclassification  medical_pipeline│
        │    └────────────┴──────────┴───────────┴──────────────┘        │
        │                    预算 / 画像 / 缓存 / 适配器                  │
        └───────────────────────────────────────────────────────────────┘
```

与 Go 拓扑的关键差异：

| 项目 | Go engine | Python engine |
|---|---|---|
| 运维口 | 独立 `:9079` | `:9079`（业务口 +1000，路径白名单）；网关独立 `:9000` |
| 多进程 | Go 原生并发 | 可选 `engine.launcher`（SO_REUSEPORT，`PRIVACY_WORKERS`） |
| TLCP | 原生双证书终止 | **进程内不实现**，由 Go 终结器侧车终结，见 [`tlcp.md`](./tlcp.md) |
| 策略清单 | `PRIVACY_CONFIG_MANIFEST_FILE` | **未实现**，见 [`policy_bundles.md`](./policy_bundles.md) |
| IAM/MFA | 已接线 | 条件接线：配置用户库后端后 `/v1/auth/*` 才挂载，见 [`auth.md`](./auth.md) |

---

## 3. 三层动态分类分级漏斗

```text
输入字段/记录
   │
   ▼
L1 规则引擎（ConfigurableRuleEngine + CompositeRuleEngine）
   │  AC/Regex/操作符 + 领域/标准画像 YAML + 优先级合并
   ▼
L2 Small-NER（按优先级尝试）
   │  MLX(Metal) → TensorRT → ONNX Runtime → ModelScope
   ▼
L3 LLM（可选）
   │  Qwen3.5/MLX、vLLM/OpenAI 兼容接口、本地 transformers
   ▼
置信度衰减 + 仲裁 + 降级/覆盖规则
   │
   ▼
SecurityTag[] → 最终等级（L1~L5）+ 脱敏建议
```

关键组件：

| 组件 | 文件 | 职责 |
|---|---|---|
| `ProfileLoader` | `dynclassification/profile_loader.py` | YAML 加载、缓存、mtime 热重载、引擎实例构建 |
| `ConfigurableRuleEngine` | `dynclassification/engine.py` | 规则合并、求值缓存、优先级裁决 |
| `CompositeRuleEngine` | `dynclassification/composite.py` | 复合规则（多字段联合） |
| `ClassificationFunnel` | `dynclassification/funnel.py` | L1→L2→L3 漏斗编排 |
| `NerAdapter` | `dynclassification/ner_adapter.py` | NER 引擎优先级探测与懒加载 |
| `LlmAdapter` | `dynclassification/llm_adapter.py` | 外部/本地 LLM 选择、熔断、并发闸门 |
| `StandardProfileGenerator` | `dynclassification/standard_profile_generator.py` | 由标准文档生成个性化画像 |

默认值：`PRIVACY_NER_ENABLE=false`、`PRIVACY_LLM_ENABLE=false`、`PRIVACY_LLM_ENABLE_ARBITRATION=true`、`PRIVACY_LLM_CONFIDENCE_THRESHOLD=0.75`、`PRIVACY_LLM_AUTO_ON_IMAGE=true`、规则目录默认 `rules/`。

规则/画像热重载：`PRIVACY_DYNCLASSIFICATION_HOT_RELOAD=true`（默认），`PRIVACY_DYNCLASSIFICATION_RELOAD_INTERVAL=0`（不节流）；也可调用 `POST /v1/dynclassification/profiles/reload`。

---

## 4. 隐私原语体系

| 能力 | 入口 | 关键实现 |
|---|---|---|
| 字段名感知脱敏 | `privacy/masking.py` | 手机/身份证/姓名/银行卡/邮箱/地址/医疗文本 + 默认策略；纯 Python 逐字符处理，无 ASCII fast-path |
| 键控哈希 | `masking.hash_value` | 默认 HMAC-SM3；`PRIVACY_HASH_LEGACY_SHA256=true` 回退 SHA-256 |
| 格式保留加密 | `masking.fpe_encrypt_numeric` | HMAC-SM3 计数器密钥流逐位映射（**非 NIST FF1/FF3**） |
| 日期偏移 / 列洗牌 | `masking.random_date_offset` / `shuffle_column` | 科研数据可用性保护 |
| DP 聚合 | `privacy/dp.py::DPApi` | Laplace/Gaussian、count/sum/mean/histogram、向量、GroupBy、自适应截断、分块流式、解析高斯校准 |
| LDP | `privacy/dp.py::LocalDPApi` | randomized response / ORR / 二值/类别扰动与频率估计 |
| K-匿名 | `privacy/kano.py`、`kano_table.py` | 单记录自适应层级泛化 + 表级 Mondrian（pandas 向量化优先） |
| 查询混淆 | `privacy/qol.py` | medical/generic 诱饵池 + 实体槽位替换 + Fisher-Yates 洗牌 |
| 预算核算 | `privacy/budget.py` | 单例 Registry、SQLite/Redis、窗口重置、HMAC 审计链、RDP accountant |
| 数据适配 | `privacy/data_adapters.py` | list/ndarray/Series/DataFrame/SecretFlow/sparse/Arrow IPC/polars |
| 画像推荐 | `privacy/profile.py` | YAML 画像 + 热加载 + 命名空间隔离 |

Python 未实现：L-Diversity（Go SDK 有）。`KAnonymizeRecord` 是逐行泛化，不是 Mondrian。

---

## 5. 网关与负载均衡

`engine/gateway/` 为独立进程：

- **调度策略**：默认 `p2c`（P2C-EWMA，`(inflight+1) * max(EWMA, 0.001)`）；可选 `round_robin`、`weighted_round_robin`、`random`、`weighted_random`、`least_connections`。
- **熔断**：每节点独立三态（closed/open/half-open），连续失败 5 次打开，30s 冷却，半开最多 3 个探测；代际绑定，迟到结果丢弃。
- **单次转发**：与 Go 一致，选中节点后只转发一次；传输失败 REST 502 / gRPC `UNAVAILABLE`，不做重试。
- **响应记账**：`<500` 记 success，`>=500` 记 failure 并原样透传；4xx 不惩罚节点。
- **主动健康（超集）**：`health_check_loop` 周期并发 HTTP `/health` 与 gRPC `Health`，失联节点移出候选，恢复后重置熔断器。
- **准入与凭据**：CIDR → HTTPS 强制 → 逐请求凭据 → keyfile 策略 → 选路 → 熔断；入站 `Authorization` 原样透传，匿名且准入关闭时才注入回源兜底 Key。
- **回源 TLS**：`PRIVACY_GATEWAY_BACKEND_TLS_*`，CA 缺失在构造期 fail-fast；可选 mTLS 客户端证书。
- **管理面**：`/v1/gateway/{register,deregister,isolate,drain,activate}`，`PRIVACY_GATEWAY_ADMIN_API_KEY` 未配置时返回 503（fail-closed）。

网关独立运维口（默认 `业务口 + 1000` = `:9000`）暴露 `/health`、`/readyz`、`/metrics`、`/gateway/backends`；指标统一为 `privshield_gateway_*`，配置 metrics/admin Key 后需 Bearer。

---

## 6. 安全与可观测

```text
请求 → ObservabilityMiddleware（request_id/日志/指标）
     → CORS（可选）→ GZip → SecurityHeaders
     → MaxBodySizeMiddleware（默认 32 MiB）
     → FastAPI 依赖：身份认证 → 限流 → Scope 校验
     → 路由处理（异常 → 统一信封）
```

- 认证：静态 API Key（internal/external JSON）、常量时间比较；gRPC mTLS CN 白名单 + YAML 热重载。
- 限流：`limits` 滑动窗口，内存或 Redis；存储异常 fail-open 并计数。
- IAM：用户库、登录会话、动态 API Key、TOTP 与沙箱在配置用户库后端后接入 Bearer 解析；`/v1/auth/*` 仅在此时挂载。
- 指标：跨语言共享集合统一为 `privshield_*`（RED、鉴权、别名、NER、网关），Python 独有集合保留 `privacy_*` / `classification_*`；`/metrics` 需 `ops:diagnostics`（引擎）或 metrics key（网关）。
- 日志：文本或 JSON，自动注入 request_id/identity/method/path；容器默认 JSON。
- 追踪：设置 `OTEL_EXPORTER_OTLP_ENDPOINT` 后启用 OpenTelemetry，否则 no-op。

---

## 7. 端到端数据流

```text
service-hub → POST /v1/agent/process
  → Pydantic 规模校验（500×100×100k）
  → MedicalPrivacyPipeline.process_records
      → 逐字段 _classify_field（PII 拦截 → L5/L4 词库 → 医疗规则 → 兜底 L1）
      → L4/L5 强抹平 + 残余整值擦除 + 双引擎（rule/ner）
  → classification_report + sanitized_data + summary
  → SHA-256 input/output hash
```

文件链路：

```text
process_file / pipeline.process_csv
  → 分块读取并即时限额（10 MiB）
  → CSV/JSON/XLSX/XLS 解析为 records
  → mask_dataframe / k_anonymize / classify+mask
  → records 或 PipelineResult
```

---

## 8. 源码工程拓扑

```text
services/privacy-engine-python/
├── engine/
│   ├── main.py                 # FastAPI 装配、中间件、路由挂载
│   ├── server.py               # REST + gRPC 单进程启动与优雅关闭
│   ├── launcher.py             # SO_REUSEPORT 多进程启动器
│   ├── grpc_server.py          # PrivacyService 实现、拦截器、错误映射
│   ├── service.py              # 业务门面（原语/预算/医疗/画像）
│   ├── schemas.py              # REST 请求模型
│   ├── deps.py                 # 共享单例、SECURITY_DEPS、异常映射
│   ├── env_loader.py           # .env / config/env/<profile>.env 级联加载
│   ├── naming.py               # 数据源/API 编码权威注册表
│   ├── routers/                # 按域拆分的 REST 路由
│   ├── privacy/                # 无状态原语 + 预算 + 适配器 + 并发工具
│   ├── dynclassification/      # 三层分类漏斗
│   ├── medical_pipeline/       # 医疗/业务分类与 L4/L5 脱敏
│   ├── pipeline/               # 通用分类+脱敏流水线（多格式）
│   ├── security/               # 认证、TLS、限流、身份
│   ├── observability/          # 日志、指标、信封、追踪、中间件
│   ├── auth/                   # IAM/MFA 内核（配置用户库后端后接线）
│   └── gateway/                # 独立网关进程
├── rules/{domains,standards,taxonomies}/
├── config/{env,certs}/ + gateway.yaml + privacy.yaml
├── data/                       # 示例数据
├── proto/privacy.proto
├── tests/                      # api/privacy/dynclassification/gateway/security/...
└── docs/
```

---

## 9. 已知边界与后续工作

| 边界 | 影响 | 参考 |
|---|---|---|
| TLCP 由 Go 终结器提供 | Python 进程不终结国密传输，必须前置 terminator 侧车 | [`tlcp.md`](./tlcp.md) |
| 无策略 bundle 清单 | 规则文件完整性依赖部署侧，非清单验真 | [`policy_bundles.md`](./policy_bundles.md) |
| PostgreSQL 用户库未内置驱动 | `USER_STORE_BACKEND=postgres` 会显式拒绝；当前支持 memory/sqlite | [`auth.md`](./auth.md) |
| 运维口指标口径 | 运维口与业务口共用应用与安全栈，白名单外一律 404 | [`prometheus.md`](./prometheus.md) |
| 无 L-Diversity | 与 Go SDK 存在能力差 | 本文 §4 |
