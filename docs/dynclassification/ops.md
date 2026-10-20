# Python 动态分类分级运维与调优指南 (Ops)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 运行时配置

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `PRIVACY_DYNCLASSIFICATION_RULES_DIR` | string | `rules` | 分类规则根目录 |
| `PRIVACY_DYNCLASSIFICATION_HOT_RELOAD` | bool | `true` | 是否开启文件修改自动热重载 |
| `PRIVACY_DYNCLASSIFICATION_RELOAD_INTERVAL` | float | `0.0` | 检查间隔秒数，0 为每次请求检查 |
| `PRIVACY_LLM_PROVIDER` | string | `mock` | LLM 后端类型：`openai` / `vllm` / `mock` |
| `PRIVACY_LLM_API_BASE` | string | `""` | 外部大模型 API 基地址 |

---

## 2. 监控指标

| 指标名称 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `dynclassification_engine_load_duration_seconds` | Histogram | `domain` | 规则 Profile 加载耗时 |
| `dynclassification_profile_cache_size` | Gauge | — | 当前缓存的规则引擎实例总数 |
| `dynclassification_evaluations_total` | Counter | `layer`, `level` | 分类评估命中分布统计 |

---

## 3. 运维实战：规则安全发布流程

为避免不合规或损坏的 YAML 进入生产，发布规则时必须遵循以下标准变更步骤：
1. **本地校验**：
   ```bash
   curl -X POST http://127.0.0.1:8079/v1/dynclassification/validate?rules_dir=rules
   ```
2. **原子切换**：通过符号链接原子替换目标规则目录。
3. **通知重载**：
   ```bash
   curl -X POST http://127.0.0.1:8079/v1/dynclassification/profiles/reload \
     -H "Authorization: Bearer <ADMIN_TOKEN>"
   ```
4. **探活验证**：提交单条记录评估，确认返回密级与预期完全匹配。
