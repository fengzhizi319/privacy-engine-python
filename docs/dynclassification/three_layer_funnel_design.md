# Python 三层分类分级漏斗裁决与仲裁机制设计 (Funnel Adjudication)

> **组件归属**：`services/privacy-engine-python`
> **核心源码**：[`engine/dynclassification/funnel.py`](../../engine/dynclassification/funnel.py)

---

## 1. 漏斗分层设计原理

在真实企业数据湖中，95% 以上的数据结构具有明确的元数据字段名或标准化内容格式（如电话、卡号、身份证）。如果对所有数据均无差别调用重型大语言模型，将导致计算资源耗尽与无法承受的延迟。

因此，三层漏斗设计遵循**从快到慢、逐层过滤、加权裁决**：

| 层级 | 引擎技术 | 典型时延 | 命中比例 | 判定标准 |
|---|---|---|---|---|
| **Layer-1: 规则引擎** | 正则、AC自动机、校验和算子 | $< 0.1\text{ms}$ | 85% ~ 90% | 置信度 $\ge 0.85$ 时直接截断放行 |
| **Layer-2: Small-NER** | ONNX 嵌入轻量模型 (BiLSTM/BERT) | $2 \sim 5\text{ms}$ | 8% ~ 12% | 针对非结构化长文本提取实体 |
| **Layer-3: 外部 LLM** | Qwen-7B / 本地或远程大模型 | $200 \sim 800\text{ms}$ | $< 2\%$ | 歧义与复杂语义兜底仲裁 |

---

## 2. 冲突检测与仲裁算法

当不同层级对同一字段给出不一致分类或密级时，`ClassificationFunnel` 启动加权仲裁流程：

```python
# 核心裁决公式
final_confidence = (
    w_rule * rule_conf + 
    w_ner * ner_conf + 
    w_llm * llm_conf
)
```
- **最高密级保守原则（Fail-Closed / High-Water Mark）**：
  若不同层级的置信度接近且发生密级冲突（例如 Layer-1 判定为 L2，Layer-2 判定为 L4），在无强置信度反证时，系统默认采纳**较高密级**，防止敏感数据由于误判低密级而遭非安全泄露。
- **置信度衰减（Decay Factor）**：
  当样本取值为空、过短或非纯文本时，规则置信度按衰减系数自动折减，触发下游实体识别。
