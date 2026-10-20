# Python 隐私引擎文档索引

> **组件归属**：`services/privacy-engine-python`
> **参考实现**：Go `services/privacy-engine/docs`（上游 [PrivShield](https://github.com/fengzhizi319/PrivShield)）
> **口径**：以本仓库代码为事实源；与 Go 不一致处显式标注，不用 Go 文档推断 Python 行为。

Python 引擎与 Go 共享 REST/gRPC 契约、隐私原语语义与分类规则；TLCP 由同仓 Go 终结器提供（Python 监听环回），策略清单验真、LoRA 训练、
PostgreSQL 用户库、集群会话验签等能力未移植。IAM/MFA 为条件接线。逐项差异见各文档的「与 Go 的差异」小节。

---

## 1. 按目标阅读

| 目标 | 推荐顺序 |
|---|---|
| 首次了解引擎 | [architecture.md](./architecture.md) → [api.md](./api.md) → [config.md](./config.md) |
| 本地启动与排障 | [ops.md](./ops.md) → [config.md](./config.md) → [testing.md](./testing.md) |
| 接入认证与权限 | [auth.md](./auth.md) → [security/README.md](./security/README.md) → [prometheus.md](./prometheus.md) |
| 使用隐私原语 | [masking/README.md](./masking/README.md) · [dp/README.md](./dp/README.md) · [k_anonymity/README.md](./k_anonymity/README.md) · [qol/README.md](./qol/README.md) |
| 分类与医疗流水线 | [dynclassification/README.md](./dynclassification/README.md) · [medical_pipeline/README.md](./medical_pipeline/README.md) |
| 网关与性能 | [gateway_balancer/README.md](./gateway_balancer/README.md) · [high_concurrency/README.md](./high_concurrency/README.md) · [算法优化.md](./算法优化.md) |
| 模型接入 | [llmlora/README.md](./llmlora/README.md) → [算法优化.md](./算法优化.md) |
| 合规与安全边界 | [tlcp.md](./tlcp.md) · [policy_bundles.md](./policy_bundles.md) · [standard/](./standard) |

---

## 2. 总览与运维

| 文档 | 内容 |
|---|---|
| [architecture.md](./architecture.md) | 设计哲学、服务拓扑、三层分类漏斗、隐私原语体系、源码结构、已知边界 |
| [api.md](./api.md) | REST/gRPC 契约、路由清单、错误信封、分页与幂等、与 Go 的契约别名 |
| [config.md](./config.md) | 配置加载顺序、传输/日志/CORS/安全/分类/预算/网关变量与默认值 |
| [ops.md](./ops.md) | 启动、探针、诊断、优雅关闭、多进程、常见故障处置 |
| [prometheus.md](./prometheus.md) | 指标清单与口径、`/metrics` 认证、抓取配置、Grafana 建议 |
| [testing.md](./testing.md) | 测试分层、运行方式、跨语言一致性夹具、生产门禁与已知测试边界 |
| [parity.md](./parity.md) | 共享契约、Python 超集清单、语言边界与跨语言验证入口 |
| [auth.md](./auth.md) | 线路层认证/授权、IAM 条件接线、沙箱、限流、与 Go 的差距 |
| [security/README.md](./security/README.md) | 安全模块代码地图、请求链、凭据策略、mTLS、限流、TLS 与配置 |
| [tlcp.md](./tlcp.md) | 国密 TLCP 拓扑、Go 终结器配置、身份约定与验证命令 |
| [`../terminator/README.md`](../terminator/README.md) | Go TLCP 终结器的构建、配置、白名单与 Key 轮换 |
| [policy_bundles.md](./policy_bundles.md) | 策略清单/规则目录的加载与验真差异、部署约束 |
| [算法优化.md](./算法优化.md) | 性能模型、GIL/线程/进程边界、NumPy/Arrow 优化与基准方法 |

## 3. 原语与流水线模块

| 文档 | 内容 |
|---|---|
| [masking/README.md](./masking/README.md) | 字段名感知脱敏、HMAC-SM3 哈希、保形加密、批量/DataFrame/文件入口 |
| [dp/README.md](./dp/README.md) | 中央 DP、LDP、预算核算、审计链、REST 路由与后端 |
| [k_anonymity/README.md](./k_anonymity/README.md) | 记录级泛化、表级 Mondrian、QIType/泛化策略、无 L-Diversity |
| [qol/README.md](./qol/README.md) | 查询混淆：诱饵池、槽位替换、长度相似策略与随机源 |
| [dynclassification/README.md](./dynclassification/README.md) | 规则→NER→LLM 漏斗、算子、画像热重载、仲裁与缓存风险 |
| [medical_pipeline/README.md](./medical_pipeline/README.md) | 医疗分类、L4/L5 脱敏、PII 拦截、摘要与规模上限 |
| [gateway_balancer/README.md](./gateway_balancer/README.md) | 调度策略、P2C-EWMA、熔断、主动/被动健康、单次转发、运维口与管理面 |
| [high_concurrency/README.md](./high_concurrency/README.md) | 并发工具的接线现状、LRU 缓存、未使用组件与性能模型 |
| [llmlora/README.md](./llmlora/README.md) | NER/LLM 适配器、provider 选择、熔断、提示词注入防护、无 LoRA 训练 |

## 4. 标准与规则参考

[`standard/`](./standard) 内置五份监管/标准文档，供规则生成与分类分级口径参考，内容保持原样：

- GB/T 35273-2020 个人信息安全规范
- GB/T 43697-2024 数据分类分级规则
- JR/T 0197-2020 金融数据安全分级指南
- 四川省健康医疗大数据应用指南
- 广东省健康医疗数据安全分类分级管理技术规范

规则 YAML 位于服务根目录 `rules/{domains,standards,taxonomies}/`，运行期由
`PRIVACY_DYNCLASSIFICATION_RULES_DIR` 指向，详见 [dynclassification/README.md](./dynclassification/README.md)。

## 5. 文档维护约定

- 路由、环境变量、默认值以代码为唯一事实源；改动后同步本目录对应文档与 [api.md](./api.md)/[config.md](./config.md)。
- 与 Go 对照时只写可验证差异；未实现能力明确标注「未实现/未接线」，不复制 Go 的部署承诺。
- `docs/standard/` 为标准原文，不随实现变更改写。
