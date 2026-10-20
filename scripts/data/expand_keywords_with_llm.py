#!/usr/bin/env python3
"""使用 LLM 自动扩展 YAML 规则文件中的 keywords 词表 / Auto-expand YAML rule keywords using LLM.

用法示例 / Usage Examples:
1. 打印扩展后的 YAML 到标准输出 (Print to stdout):
   python scripts/data/expand_keywords_with_llm.py rules/domains/general-pii.yaml

2. 直接覆盖原文件 (In-place update):
   python scripts/data/expand_keywords_with_llm.py rules/domains/general-pii.yaml -i

3. 保存到新文件 (Save to output file):
   python scripts/data/expand_keywords_with_llm.py rules/domains/finance.yaml -o rules/domains/finance_expanded.yaml

4. 使用 OpenAI/Qwen/DeepSeek API (Using API):
   python scripts/data/expand_keywords_with_llm.py rules/domains/general-pii.yaml --api-key "sk-xxx" --api-base "https://dashscope.aliyuncs.com/compatible-mode/v1" --model "qwen-plus"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List

import yaml

# 项目根路径引入
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from engine.observability.logging_config import get_logger

logger = get_logger("expand_keywords_script")


# ===========================================================================
# 内置同义词与近义词扩展词库（当 LLM 不可用时的离线回退词表）
# Built-in synonym dictionary for offline fallback when LLM is unavailable
# ===========================================================================
OFFLINE_SYNONYMS: dict[str, list[str]] = {
    # 电话/手机
    "phone": ["tel", "mobile", "cellphone", "telephone", "call", "dh", "lxfs", "联系电话", "手机号码", "联系方式"],
    "mobile": ["phone", "cell", "cellphone", "mobile_num", "手机号", "移动电话"],
    "tel": ["telephone", "phone", "landline", "固定电话", "座机"],
    "电话": ["手机", "联系电话", "联系方式", "固话", "座机", "phone", "tel", "mobile"],

    # 身份证/证件
    "idcard": ["identity", "id_card", "sfz", "passport", "身份证号", "证件号码", "身份标识"],
    "身份证": ["身份证号", "证件号", "身份标识", "idcard", "sfz", "identity"],

    # 姓名/用户
    "name": ["username", "realname", "fullname", "patient_name", "姓名", "真实姓名", "用户姓名"],
    "identity": ["idcard", "sfz", "passport", "身份证", "身份证明"],

    # 银行卡/账户
    "bankcard": ["cardno", "card_no", "bank_card", "account_no", "acc_no", "银行卡号", "支付卡号", "结算账号"],
    "cardno": ["bankcard", "card_no", "credit_card", "卡号", "卡号明细"],
    "account": ["bank_account", "account_no", "acc_no", "iban", "账户", "账号", "资产账号"],

    # 金融交易
    "transfer": ["payment", "pay", "withdraw", "deposit", "loan", "remittance", "转账", "汇款", "支付"],
    "income": ["salary", "wage", "paycheck", "tax", "fund", "revenue", "收入", "工资", "薪资", "纳税"],

    # 医疗与病种
    "hiv": ["aids", "std", "syphilis", "gonorrhea", "艾滋病", "性病", "梅毒", "淋病"],
    "gene": ["genomic", "brca", "tp53", "snp", "cnv", "chromosome", "dna", "rna", "基因", "染色体"],
    "brca1": ["brca2", "tp53", "egfr", "kras", "braf", "alk"],

    # 统计与运营
    "turnover_rate": ["device_usage", "inventory", "stat", "statistics", "count", "cnt", "total_cnt", "周转率", "使用率", "统计人次"],
    "public_report": ["annual_summary", "科普", "公开", "public", "notice", "announcement", "公示", "公告"],
}


def build_expansion_prompt(rule_id: str, rule_name: str, category: str, level: str, current_keywords: list[str]) -> str:
    """构建用于提示 LLM 扩展关键词的系统 Prompt / Construct LLM expansion prompt."""
    return f"""你是一个数据分类分级与数据安全合规专家。
请为以下敏感数据分类规则扩展关键词（keywords），以便规则引擎能更全面地匹配数据库列名、API 参数和结构化文本。

【规则信息】
- 规则 ID: {rule_id}
- 规则名称: {rule_name}
- 分类类别: {category}
- 敏感等级: {level}
- 现有关键词列表: {json.dumps(current_keywords, ensure_ascii=False)}

【扩展要求】
1. 深入挖掘与该规则业务语义相关的常用词汇（包含英文全称、常用缩写、中文词汇、数据库列名惯用名，如 phone/tel/mobile/cell/电话/手机/lxfs/dh 等）。
2. 请补充 10 到 20 个高频、高质量的词汇。
3. 必须保留并包含现有的所有关键词，不能丢失。
4. 排除过短、容易造成泛化误报的无意义词（例如单独的单字、纯数字、或过于通用的词）。
5. 必须严格以 JSON 格式输出，不得包含任何 Markdown 外壳或其他解释文本，格式如下：
{{"expanded_keywords": ["词1", "词2", "词3", ...]}}
"""


def call_llm_api(prompt: str, api_key: str, api_base: str, model: str) -> list[str] | None:
    """调用 OpenAI 兼容格式的远程 LLM API / Call OpenAI-compatible LLM API."""
    try:
        import urllib.request

        url = api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            # 提取 JSON
            json_match = re_search_json(content)
            if json_match:
                parsed = json.loads(json_match)
                return parsed.get("expanded_keywords", [])
    except Exception as e:
        logger.warning(f"调用远程 LLM API 扩展关键词失败: {e}")
    return None


def call_local_llm(prompt: str) -> list[str] | None:
    """尝试调用项目本地的 LlmAdapter / Try calling project local LlmAdapter."""
    try:
        from engine.dynclassification.llm_adapter import LlmAdapter

        adapter = LlmAdapter()
        if adapter.is_available:
            result = adapter.classify(prompt, "L3", 0.5)
            if result and isinstance(result, dict):
                return result.get("expanded_keywords", None)
    except Exception as e:
        logger.debug(f"本地 LLM不可用或初始化失败: {e}")
    return None


def fallback_expand(current_keywords: list[str]) -> list[str]:
    """离线本地规则扩展（基于内置同义词表兜底） / Offline heuristic fallback expansion."""
    expanded = set(current_keywords)
    for kw in list(current_keywords):
        kw_lower = kw.lower().strip()
        for key, syns in OFFLINE_SYNONYMS.items():
            if kw_lower == key or key in kw_lower:
                expanded.update(syns)
    return list(expanded)


def re_search_json(text: str) -> str | None:
    """正则提取字符串中的 JSON 块 / Extract JSON block from string via regex."""
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def expand_keywords_for_list(
    items: list[dict[str, Any]],
    api_key: str | None = None,
    api_base: str | None = None,
    model: str = "qwen-plus",
) -> int:
    """遍历处理规则字典列表中的 matchers 及 downgrade_rules 离线/在线扩展 keywords."""
    count = 0
    for item in items:
        rule_id = item.get("id", "UNKNOWN_RULE")
        rule_name = item.get("name", "")
        category = item.get("category", "")
        level = item.get("level", "")

        # 场景 1: 普通规则 matchers
        matchers = item.get("matchers", [])
        for m in matchers:
            params = m.get("params", {})
            if "keywords" in params and isinstance(params["keywords"], list):
                old_kws = params["keywords"]
                new_kws = None

                # 优先尝试 API 或 本地 LLM
                prompt = build_expansion_prompt(rule_id, rule_name, category, level, old_kws)
                if api_key and api_base:
                    new_kws = call_llm_api(prompt, api_key, api_base, model)
                if not new_kws:
                    new_kws = call_local_llm(prompt)
                if not new_kws:
                    new_kws = fallback_expand(old_kws)

                # 合并去重，保持已有顺序 + 追加新词
                merged = list(dict.fromkeys(old_kws + new_kws))
                params["keywords"] = merged
                count += 1

        # 场景 2: 降级规则 keywords 字段
        if "keywords" in item and isinstance(item["keywords"], list):
            old_kws = item["keywords"]
            new_kws = None
            prompt = build_expansion_prompt(rule_id, rule_name, category, level, old_kws)
            if api_key and api_base:
                new_kws = call_llm_api(prompt, api_key, api_base, model)
            if not new_kws:
                new_kws = call_local_llm(prompt)
            if not new_kws:
                new_kws = fallback_expand(old_kws)

            merged = list(dict.fromkeys(old_kws + new_kws))
            item["keywords"] = merged
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 LLM 自动扩展 YAML 规则文件中的 keywords 词表 / Auto-expand rule keywords in YAML using LLM."
    )
    parser.add_argument("yaml_file", type=str, help="目标 YAML 规则文件路径 (如 rules/domains/general-pii.yaml)")
    parser.add_argument("-i", "--in-place", action="store_true", help="是否直接修改/覆盖原文件 (In-place edit)")
    parser.add_argument("-o", "--output", type=str, default="", help="保存的输出文件路径（默认打印到终端 stdout）")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")), help="OpenAI/DashScope API Key")
    parser.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"), help="LLM API Base URL")
    parser.add_argument("--model", type=str, default="qwen-plus", help="LLM 模型名称 (默认 qwen-plus)")

    args = parser.parse_args()

    file_path = Path(args.yaml_file).resolve()
    if not file_path.exists():
        logger.error(f"指定的文件不存在: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        logger.error(f"无效的 YAML 文件内容格式: {file_path}")
        sys.exit(1)

    # 处理 rules 和 downgrade_rules
    rules_processed = 0
    if "rules" in data and isinstance(data["rules"], list):
        rules_processed += expand_keywords_for_list(data["rules"], args.api_key, args.api_base, args.model)

    if "downgrade_rules" in data and isinstance(data["downgrade_rules"], list):
        rules_processed += expand_keywords_for_list(data["downgrade_rules"], args.api_key, args.api_base, args.model)

    # 导出 YAML
    output_yaml = yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)

    if args.in_place:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(output_yaml)
        print(f"成功自动扩展并覆盖更新文件: {file_path} (扩展了 {rules_processed} 组关键词)")
    elif args.output:
        out_path = Path(args.output).resolve()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_yaml)
        print(f"成功自动扩展并写入新文件: {out_path} (扩展了 {rules_processed} 组关键词)")
    else:
        print(output_yaml)


if __name__ == "__main__":
    main()
