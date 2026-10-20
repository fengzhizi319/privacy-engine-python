# 策略清单（Policy Bundle）与规则发布

> **组件归属**：`services/privacy-engine-python`
> **Go 对照**：上游 `services/privacy-engine/docs/policy_bundles.md`
> **状态**：✅ Python 已实现 `privshield-policy-bundle/v1` 清单加载与 SHA-256 验真（参见 [`engine/dynclassification/bundle.py`](../engine/dynclassification/bundle.py) 与 [`tests/dynclassification/test_bundle.py`](../tests/dynclassification/test_bundle.py)）；同时保留目录 + mtime 兼容加载模式。

本文先固定 Go 的清单契约，再说明 Python 当前真实的文件输入、加载语义与完整性缺口，最后给出迁移到清单模式所需的对齐项。不要把 Go 的 `PRIVACY_CONFIG_MANIFEST_FILE` 行为套用到 Python。

---

## 1. Go 清单契约（事实源）

Go 设置 `PRIVACY_CONFIG_MANIFEST_FILE=/releases/current.json` 后，以 JSON 清单作为策略、领域规则、分级词表和标准映射的唯一文件输入；未设置时保留独立路径加载。启用清单后，`PRIVACY_CONFIG_FILE` / `PRIVACY_RULES_DIR` / `PRIVACY_TAXONOMIES_DIR` / `PRIVACY_STANDARDS_DIR` 不再决定文件输入。

版本目录布局：

```text
release-001/
  manifest.json
  privacy.yaml
  domains/*.yaml
  taxonomies/*.yaml
  standards/*.yaml
```

清单样式：

```json
{
  "schema": "privshield-policy-bundle/v1",
  "release": "release-001",
  "policy": {"path": "privacy.yaml", "sha256": "<64位小写十六进制>"},
  "domains": [{"path": "domains/medical.yaml", "sha256": "<真实摘要>"}],
  "taxonomies": [{"path": "taxonomies/default.yaml", "sha256": "<真实摘要>"}],
  "standards": []
}
```

关键约束：

- 顶层字段必须齐全；空组表示**显式不加载**该类文件；未列出文件不会进入快照。
- 仅允许各组目录的直接 YAML/YML 文件；拒绝重复相对路径、绝对路径、目录跳转、嵌套目录、未知/重复 JSON 键与清单内路径的符号链接。
- 清单指针本身可为符号链接，加载时解析一次并绑定版本目录。
- 预算：单文件（含清单）4 MiB；最多 1024 个源文件；清单 + 全部文件累计 16 MiB；JSON 嵌套/节点数受限。
- 启动时清单缺失或摘要不符 → **拒绝启动**；重载失败 → 保留旧快照，**禁止回落到无清单模式**。
- 发布通过「新目录 + 原子 rename 链接 + 各副本 `POST /v1/dynclassification/profiles/reload`」完成；诊断输出 `configuration_version` / `configuration_release`。
- SHA-256 只保证清单与内容一致，**不认证发布者**；签名信任根与发布权限属于另一控制面。

> 更完整的发布流程、回滚与可观察性契约见 Go 文档原文。

---

## 2. Python 当前的真实文件输入

Python **不读取** `PRIVACY_CONFIG_MANIFEST_FILE`，也不识别 Go 的 `PRIVACY_CONFIG_FILE` / `PRIVACY_RULES_DIR` / `PRIVACY_TAXONOMIES_DIR` / `PRIVACY_STANDARDS_DIR`。

| 输入 | 环境变量 | 默认 | 消费者 | 说明 |
|---|---|---|---|---|
| 分类规则根目录 | `PRIVACY_DYNCLASSIFICATION_RULES_DIR` | `rules` | `engine/dynclassification/profile_loader.py` | 期望 `taxonomies/`、`domains/`、`standards/` 子目录 |
| 热重载开关 | `PRIVACY_DYNCLASSIFICATION_HOT_RELOAD` | `true` | 同上 | 关闭后只能手动 `/profiles/reload` |
| 热重载检查间隔 | `PRIVACY_DYNCLASSIFICATION_RELOAD_INTERVAL` | `0` | 同上 | 秒；0 表示每次请求都扫描 |
| 基础隐私参数 profile | `PRIVACY_PROFILE` | `privacy-profile.yaml` | `engine/deps.py`、`engine/privacy/profile.py` | DP/K-匿名/QOL 默认参数 |
| 个性化推荐参数 | `PRIVACY_PERSONALIZED_PROFILE` | `config/personalized-profiles.yaml` | `engine/privacy/profile.py` | 按 namespace + primitive 分层的推荐结果 |
| `config/privacy.yaml` | — | — | **无** | Go 结构示例，Python 不解析；修改它不会影响行为 |

`config/privacy.yaml` 文件头已注明「对应 Go 引擎 Config 结构」。它既不是 Python 的生效配置，也不是清单输入。

---

## 3. 规则加载与热重载语义

### 3.1 目录结构

```text
${PRIVACY_DYNCLASSIFICATION_RULES_DIR}/
├── taxonomies/<name>.yaml      # DomainTaxonomy
├── domains/<domain>.yaml       # RuleProfile
└── standards/<standard_id>.yaml# StandardDef
```

- 文件按路径组件归组（`"taxonomies" in path.parts`），不靠字符串包含判断。
- 扫描使用 `glob("**/*.yaml")`：会递归子目录，且**只认 `.yaml`**，不认 `.yml`。Go 清单模式要求直接子文件且接受 `.yaml/.yml`，两者不一致。
- 解析使用 Pydantic `model_validate`；结构错误在加载时抛出。

### 3.2 自动热重载（请求驱动）

每次分类请求前调用 `check_and_reload()`：

| 变化 | 行为 |
|---|---|
| 文件新增/删除 | 直接 `invalidate_cache()`，记录新 mtime；**下次请求懒加载** |
| 文件内容修改 | **两阶段提交**：先在临时缓存解析全部 YAML，全部成功才原子替换并清空引擎缓存 |
| 解析失败（修改路径） | 保留旧缓存与旧 mtime，记录错误并持续重试；不会误标为已加载 |
| `PRIVACY_DYNCLASSIFICATION_HOT_RELOAD=false` | 跳过自动检查 |
| 目录不存在 | 直接返回，不报错 |

两阶段提交只覆盖「已有文件被修改」路径；新增/删除走的是清空后懒加载，若新增文件语法错误，错误会延迟到**下一个业务请求**才暴露。

### 3.3 手动重载与校验端点

| 端点 | 权限 | 实际行为 |
|---|---|---|
| `POST /v1/dynclassification/profiles/reload` | `dynclassification:write` | 仅 `invalidate_cache()` + 清空漏斗/适配器缓存，**立即返回成功**；不预校验、不返回解析结果，下一次请求才重新解析 |
| `POST /v1/dynclassification/validate`（GET 同义） | `dynclassification:read` | 对磁盘候选目录做完整 Pydantic + 算子注册 + 标准引用校验；`rules_dir` 经 `_safe_path` 限制在项目基准目录内 |
| `POST /v1/dynclassification/generate_profile` | `dynclassification:write` | 从标准 Markdown 生成 3 个 YAML 写入 `rules_dir`，随后清空缓存 |

> ⚠️ `reload` 不等待校验结果。运维流程必须「先 `validate`，再 `reload`」，并在 reload 后发一次真实分类请求确认加载成功；否则语法错误会在生产流量中首次暴露。

### 3.4 无清单模式下的发布方式

Python 没有版本目录/清单契约，可行的等效流程是：

1. 在只读基准目录之外准备新规则目录（含完整 `taxonomies/`、`domains/`、`standards/`）。
2. 调用 `POST /v1/dynclassification/validate?rules_dir=<新目录>` 校验。
3. 原子切换 `PRIVACY_DYNCLASSIFICATION_RULES_DIR` 指向的目录（符号链接 + rename，或整目录替换）。
4. 调用 `POST /v1/dynclassification/profiles/reload`。
5. 打一次分类请求做冒烟，并检查 `/v1/ops/diagnostics`。

限制：

- 规则目录**只能通过重启改变**（`ProfileLoader` 在构造时解析环境变量）；运行中切换必须依赖同名路径的原子替换或 symlink 指向变更。
- 多进程 `engine.launcher` 下，每个 worker 独立扫描 mtime；reload 请求只命中其中一个进程，**其余 worker 不会同步清缓存**。全副本一致必须逐进程重启或引入广播控制面。
- 没有 `configuration_release` / `configuration_version` 诊断字段，无法证明两个实例加载了同一内容。

---

## 4. 完整性缺口（相对 Go）

| 控制 | Go 清单模式 | Python |
|---|---|---|
| SHA-256 摘要绑定 | ✅ 逐文件校验 | ❌ |
| 未列出文件隔离 | ✅ 不进入快照 | ❌ 目录内所有 `.yaml` 都会被扫描 |
| 路径/符号链接/重复键约束 | ✅ 严格拒绝 | ❌ 无清单，未做词法约束 |
| 单文件 / 总文件数 / 总字节预算 | ✅ 4 MiB / 1024 / 16 MiB | ❌ 仅上传接口有大小限制，规则目录无预算 |
| 启动失败姿态 | 清单坏 → 拒启 | 规则坏 → 启动可成功，首个请求报错 |
| 重载失败回退 | 保留旧快照，**禁止**回落无清单 | 修改路径保留旧缓存；手动 reload 后首次解析失败无旧快照 |
| 发布版本可观测 | `configuration_release` / `configuration_version` | ❌ |
| 发布者认证 | 明确交由签名/attestation 控制面 | ❌ |

### 4.1 部署侧补偿措施

在 Python 实现清单前，生产部署应至少做到：

- 规则目录以**只读**卷挂载，发布权限与业务进程分离；
- 镜像内固定规则内容，规则变更走新镜像或新版本目录，不原地编辑；
- 对规则目录做外部摘要清单与发布审计（CI 产物 + 准入校验）；
- `validate` 纳入发布门禁，并对 reload 后首个分类做探活；
- 多副本发布时按实例逐个重启/重载并核对行为。

这些措施能在流程上降低风险，但**不能替代**进程内的摘要验真。

---

## 5. 迁移到清单模式的对齐清单

若要让 Python 与 Go 的 `privshield-policy-bundle/v1` 对齐，至少需要：

1. `ProfileLoader` 支持清单路径与 schema 校验；清单模式下列表驱动加载，不再扫描目录。
2. 实现路径约束（直接子文件、拒绝重复/绝对/`..`/嵌套/清单内 symlink）、JSON 重复键拒绝、尺寸与数量预算。
3. 加载前逐文件计算 SHA-256 并与清单比对；不匹配即失败。
4. 启动语义改为 fail-fast；重载失败保留旧快照，禁止回落目录模式。
5. `/profiles/reload` 返回校验结果与配置版本；诊断输出 `configuration_release` / `configuration_version`。
6. 多 worker 广播或共享版本指针，保证全实例收敛。
7. 明确签名信任根与发布者认证（SHA-256 只解决一致性，不解决来源）。

---

## 6. 验证

```bash
# 1. 当前没有任何清单变量读取
grep -rn "PRIVACY_CONFIG_MANIFEST_FILE\|privshield-policy-bundle" engine --include='*.py'
# 预期：无输出

# 2. 确认实际规则根目录
grep -rn "PRIVACY_DYNCLASSIFICATION_RULES_DIR" engine --include='*.py'

# 3. 校验规则目录
python -m engine.dynclassification validate rules

# 4. 运行中的实例校验 + 重载（需认证时携带 Bearer Key）
curl -s -X POST 'localhost:8079/v1/dynclassification/validate?rules_dir=rules'
curl -s -X POST localhost:8079/v1/dynclassification/profiles/reload \
  -H 'authorization: Bearer <key>'

# 5. 重载后冒烟：任一分类请求
curl -s -X POST localhost:8079/v1/dynclassification/classify/field \
  -H 'authorization: Bearer <key>' -H 'content-type: application/json' \
  -d '{"field_name":"name","value":"张三"}'
```

---

## 7. 相关文档

- [架构总览](./architecture.md)
- [配置参考](./config.md)
- [运维手册](./ops.md)
- [认证与授权](./auth.md)
- [TLCP 与传输层](./tlcp.md)
- Go 清单契约：上游 `services/privacy-engine/docs/policy_bundles.md`
