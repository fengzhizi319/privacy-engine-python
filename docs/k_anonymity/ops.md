# Python K-匿名与 L-多样性运维指南 (Ops)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 运行时配置与环境变量

| 变量名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `PRIVACY_PROFILE` | string | `privacy-profile.yaml` | 默认隐私参数配置文件路径 |
| `PRIVACY_PERSONALIZED_PROFILE` | string | `config/personalized-profiles.yaml` | 个性化推荐参数库 |

---

## 2. 监控指标 (Prometheus Metrics)

| 指标名称 | 类型 | 标签 (Labels) | 业务含义与阈值监控 |
|---|---|---|---|
| `privacy_kano_operations_total` | Counter | `operation="table" / "record" / "record_batch"` | K-匿名操作总调用次数 |
| `privacy_kano_duration_seconds` | Histogram | `operation` | 泛化与分区计算耗时分布，P99 应 $< 50\text{ms}$ |

### Prometheus 告警规则建议
```yaml
groups:
  - name: privshield_kano_alerts
    rules:
      - alert: KanoHighLatency
        expr: histogram_quantile(0.99, sum(rate(privacy_kano_duration_seconds_bucket[5m])) by (le)) > 0.2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "K-Anonymity P99 latency exceeded 200ms"
```

---

## 3. 故障排查与典型问题

### Q1: 大数据集批量泛化超时 (`TimeoutError`)
- **现象**：调用 `anonymize_records_batch` 超过 60 秒返回超时。
- **原因**：单次处理超过 10,000 条复杂记录或准标识符列过多。
- **解决方案**：在调用端进行数据流分块（每批次推荐 1,000 ~ 2,000 条），或改用表级 Mondrian 向量化接口 `POST /v1/privacy/k_anonymize/table`。

### Q2: 泛化后等价类数为 1
- **现象**：返回的数据全部被泛化为一个巨大的大区间。
- **原因**：$k$ 值设定过大（接近样本总量）或 $l$ 值设定过高（切分会导致子集敏感属性相异数 $< l$）。
- **核查**：先调用 `/v1/privacy/l_diversity/check` 观察原始敏感属性取值分布，评估 $l$ 设定的合理性。
