#!/usr/bin/env python3
"""基于 Markdown 标准文档自动提取分类分级规则并进行 Keywords 词表自动扩展的生成工具。
Extract classification rules from Markdown documents and auto-expand rule keywords into YAML profile.

本脚本完全独立，包含原生的 MD 解析能力、LLM 规则提取能力、LLM/离线关键词扩展能力。

用法示例 / Usage Examples:
1. 从 Markdown 标准文档生成规则 YAML（标准输出）:
   python scripts/data/gen_yaml_from_doc.py docs/standard/四川省健康医疗大数据应用指南.md --domain sc_health

2. 生成 YAML 并输出保存到指定路径:
   python scripts/data/gen_yaml_from_doc.py docs/standard/四川省健康医疗大数据应用指南.md --domain sc_health -o rules/domains/sc_health_auto.yaml

3. 使用大模型 API 进行高质量规则提取与词表扩充:
   python scripts/data/gen_yaml_from_doc.py docs/standard/四川省健康医疗大数据应用指南.md --domain sc_health -o rules/domains/sc_health_expanded.yaml \
     --api-key "sk-xxx" --api-base "https://dashscope.aliyuncs.com/compatible-mode/v1" --model "qwen-plus"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# 项目根目录挂载
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from engine.observability.logging_config import get_logger

logger = get_logger("gen_yaml_from_doc")


# ===========================================================================
# 独立同义词与词表扩展字典（用于离线/无 LLM 时的本地降级扩展）
# Standalone Synonym Dictionary for Offline Fallback Expansion
# ===========================================================================
OFFLINE_SYNONYM_DICT: dict[str, list[str]] = {
    # 身份识别与基本个人信息
    "idcard": ["identity", "id_card", "sfz", "passport", "身份证号", "证件号码", "身份标识", "护照号"],
    "身份证": ["身份证号", "证件号", "身份标识", "idcard", "sfz", "identity", "护照"],
    "name": ["username", "realname", "fullname", "patient_name", "姓名", "真实姓名", "用户姓名"],
    "phone": ["tel", "mobile", "cellphone", "telephone", "call", "dh", "lxfs", "联系电话", "手机号码", "联系方式"],
    "email": ["mail", "e_mail", "mailbox", "邮箱", "电子邮箱"],
    "address": ["addr", "location", "loc", "position", "居住地", "联系地址", "家庭住址", "邮编"],

    # 金融与支付
    "bankcard": ["cardno", "card_no", "bank_card", "credit_card", "account_no", "银行卡号", "支付卡号", "结算账号"],
    "account": ["bank_account", "account_no", "acc_no", "iban", "账户", "账号", "资产账号"],
    "transaction": ["transfer", "payment", "withdraw", "deposit", "loan", "remittance", "交易金额", "转账", "支付"],

    # 医疗与基因
    "disease": ["illness", "diagnosis", "symptom", "condition", "hiv", "aids", "std", "syphilis", "诊断", "疾病", "病种", "病例"],
    "gene": ["genomic", "brca", "tp53", "snp", "cnv", "chromosome", "dna", "rna", "mutation", "基因", "染色体", "基因突变"],
    "medical_record": ["his", "emr", "ehr", "prescription", "病历", "处方", "门诊记录", "住院记录"],

    # 运营与统计降级
    "turnover": ["inventory", "device_usage", "stat", "statistics", "count", "cnt", "total_cnt", "周转率", "使用率", "统计人次", "汇总"],
    "public": ["report", "annual_summary", "科普", "公开", "notice", "announcement", "公示", "公告"],
}


# ===========================================================================
# Markdown 解析与文本结构抽取 (Markdown Regex / Table Parser)
# ===========================================================================
def extract_tables_from_markdown(md_text: str) -> list[list[list[str]]]:
    """从 Markdown 文本中解析所有数据表格 / Extract all tables from Markdown text."""
    tables = []
    current_table = []

    for line in md_text.splitlines():
        line_str = line.strip()
        if line_str.startswith("|") and line_str.endswith("|"):
            cells = [c.strip() for c in line_str.split("|")[1:-1]]
            # 过滤 Markdown 表格分隔行如 |---|---|
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                continue
            current_table.append(cells)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []

    if current_table:
        tables.append(current_table)

    return tables


def clean_md_text(text: str) -> str:
    """清理 Markdown 格式标记 (如 **粗体**、`代码`、[链接](url)) / Clean Markdown formatting."""
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    return t.strip()


def parse_md_heuristically(md_text: str, domain_name: str) -> list[dict[str, Any]]:
    """通过正则与表结构启发式从 Markdown 解析分类规则 / Heuristically parse rules from MD tables."""
    extracted_rules = []
    tables = extract_tables_from_markdown(md_text)

    rule_idx = 1
    for tbl in tables:
        if len(tbl) < 2:
            continue
        headers = [clean_md_text(h).lower() for h in tbl[0]]

        # 判断表格是否包含分类分级特征列
        is_classification_table = any("级别" in h or "等级" in h or "级" in h or "level" in h for h in headers)
        if not is_classification_table:
            continue

        # 寻址列索引
        cat_col = next((i for i, h in enumerate(headers) if "类" in h or "项" in h or "category" in h), 0)
        kw_col = next((i for i, h in enumerate(headers) if "数据" in h or "元素" in h or "字段" in h or "内容" in h or "例" in h), 1)
        lvl_col = next((i for i, h in enumerate(headers) if "级" in h or "level" in h or "等" in h), -1)

        for row in tbl[1:]:
            if len(row) <= max(cat_col, kw_col, lvl_col if lvl_col != -1 else 0):
                continue

            category_raw = clean_md_text(row[cat_col])
            kw_raw = clean_md_text(row[kw_col])
            level_raw = clean_md_text(row[lvl_col]) if lvl_col != -1 else "L3"

            # 归一化敏感等级 (L1-L5 / C1-C5)
            level_match = re.search(r"([LC][1-5])", level_raw.upper())
            level = level_match.group(1) if level_match else "L3"

            # 从示例/字段文本中提取关键词列表
            raw_tokens = re.split(r"[,\s;；、/（）()]+", kw_raw)
            keywords = [clean_md_text(t) for t in raw_tokens if len(clean_md_text(t)) > 1]

            if not keywords:
                continue

            rule_id = f"RULE_{domain_name.upper()}_{rule_idx:03d}"
            rule_idx += 1

            extracted_rules.append({
                "id": rule_id,
                "name": f"{category_raw}规则",
                "category": f"{domain_name.upper()}_{rule_idx}",
                "level": level,
                "priority": 100,
                "match_logic": "OR",
                "matchers": [
                    {
                        "target": "field_name",
                        "operator": "keyword_contains",
                        "params": {
                            "keywords": keywords,
                            "use_word_boundaries": False,
                        },
                    }
                ],
            })

    return extracted_rules


# ===========================================================================
# LLM 交互模块（Markdown 规则大模型提取 & Keywords 词表大模型扩展）
# ===========================================================================
def re_search_json(text: str) -> str | None:
    """提取文本中的第一个 JSON 块 / Extract JSON string via regex."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def call_llm_api(prompt: str, api_key: str, api_base: str, model: str) -> dict[str, Any] | None:
    """通用调用 OpenAI 兼容格式远程大模型 API / Invoke remote OpenAI-compatible LLM API."""
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
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            json_str = re_search_json(content)
            if json_str:
                return json.loads(json_str)
    except Exception as e:
        logger.warning(f"调用远程 LLM API 失败: {e}")
    return None


def call_local_llm(prompt: str) -> dict[str, Any] | None:
    """尝试调用项目内置本地 LlmAdapter / Invoke project local LlmAdapter."""
    try:
        from engine.dynclassification.llm_adapter import LlmAdapter

        adapter = LlmAdapter()
        if adapter.is_available:
            result = adapter.classify(prompt, "L3", 0.5)
            if result and isinstance(result, dict):
                return result
    except Exception as e:
        logger.debug(f"本地 LLM 适配器不可用: {e}")
    return None


def extract_rules_from_md_with_llm(
    md_text: str, domain_name: str, api_key: str | None, api_base: str | None, model: str
) -> list[dict[str, Any]] | None:
    """使用大模型从 Markdown 文档整体提取规则定义 / Extract rules from MD via LLM."""
    prompt = f"""你是一个数据安全合规专家。以下是一份关于分类分级标准的 Markdown 规范文档截段：

--- Markdown 文档开始 ---
{md_text[:4000]}
--- Markdown 文档结束 ---

请解析上述文档中的敏感数据分类分级规则，并输出格式化的 JSON 数据结构。
要求：
1. 提取所有涉及的数据类别、敏感度等级（如 L1~L5 或 C1~C5）以及对应的示例字段/关键词。
2. 为每个分类构造规则定义列表 `rules`。
3. 规则包含: `id` (如 RULE_{domain_name.upper()}_001), `name`, `category`, `level`, `matchers`。
4. 必须严格以 JSON 格式输出，格式如下：
{{
  "rules": [
    {{
      "id": "RULE_{domain_name.upper()}_001",
      "name": "规则名称",
      "category": "类别标识",
      "level": "L3",
      "priority": 100,
      "match_logic": "OR",
      "matchers": [
        {{
          "target": "field_name",
          "operator": "keyword_contains",
          "params": {{
            "keywords": ["kw1", "kw2"],
            "use_word_boundaries": false
          }}
        }}
      ]
    }}
  ]
}}
"""
    result = None
    if api_key and api_base:
        result = call_llm_api(prompt, api_key, api_base, model)
    if not result:
        result = call_local_llm(prompt)

    if result and isinstance(result, dict) and "rules" in result:
        return result["rules"]
    return None


def expand_keywords_independently(
    rule_id: str,
    rule_name: str,
    category: str,
    level: str,
    current_keywords: list[str],
    api_key: str | None,
    api_base: str | None,
    model: str,
) -> list[str]:
    """独立实现 Keywords 词表扩充（包含 LLM 扩展与本地离线词表 fallback 扩展） / Independent keyword expansion."""
    # 1. 尝试 LLM 扩展
    prompt = f"""你是一个数据安全合规专家。请为敏感数据分类规则扩展关键词（keywords）。
规则信息:
- ID: {rule_id}
- 名称: {rule_name}
- 类别: {category}
- 等级: {level}
- 现有关键词: {json.dumps(current_keywords, ensure_ascii=False)}

要求：
1. 扩充 10-20 个高频中英文同义词、缩写、数据库常见列名（如 phone/tel/mobile/cell/电话/手机/lxfs/dh）。
2. 保留现有关键词。
3. 严格输出 JSON 格式: {{"expanded_keywords": ["词1", "词2", ...]}}
"""

    expanded_from_llm = None
    if api_key and api_base:
        resp_json = call_llm_api(prompt, api_key, api_base, model)
        if resp_json and "expanded_keywords" in resp_json:
            expanded_from_llm = resp_json["expanded_keywords"]

    if not expanded_from_llm:
        resp_json = call_local_llm(prompt)
        if resp_json and "expanded_keywords" in resp_json:
            expanded_from_llm = resp_json["expanded_keywords"]

    if expanded_from_llm and isinstance(expanded_from_llm, list):
        # 保持顺序去重
        return list(dict.fromkeys(current_keywords + expanded_from_llm))

    # 2. 离线本地 fallback 扩展
    expanded_set = set(current_keywords)
    for kw in list(current_keywords):
        kw_lower = kw.lower().strip()
        for dict_key, syns in OFFLINE_SYNONYM_DICT.items():
            if kw_lower == dict_key or dict_key in kw_lower:
                expanded_set.update(syns)

    return list(dict.fromkeys(current_keywords + list(expanded_set)))


def process_doc_to_yaml(
    md_file_path: Path,
    domain_name: str,
    api_key: str | None = None,
    api_base: str | None = None,
    model: str = "qwen-plus",
) -> dict[str, Any]:
    """主处理函数：从 Markdown 解析 -> 产生分类规则 -> 执行关键词自动扩展."""
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Step 1: 提取规则（优先 LLM，兜底解析器）
    logger.info(f"正在从 {md_file_path.name} 提取分类分级规则...")
    rules = extract_rules_from_md_with_llm(md_text, domain_name, api_key, api_base, model)
    if not rules:
        logger.info("LLM 规则提取跳过/失败，使用本地 Markdown 表格解析器...")
        rules = parse_md_heuristically(md_text, domain_name)

    # Step 2: 独立词表扩展 (Expand keywords for each rule)
    logger.info("正在执行 Keywords 词表语义扩展...")
    for r in rules:
        rule_id = r.get("id", "RULE_UNKNOWN")
        rule_name = r.get("name", "")
        category = r.get("category", "")
        level = r.get("level", "L3")

        matchers = r.get("matchers", [])
        for m in matchers:
            params = m.get("params", {})
            if "keywords" in params and isinstance(params["keywords"], list):
                old_kws = params["keywords"]
                new_kws = expand_keywords_independently(
                    rule_id, rule_name, category, level, old_kws, api_key, api_base, model
                )
                params["keywords"] = new_kws

    # Step 3: 构建最终标准 RuleProfile 字典
    profile_data = {
        "domain": domain_name,
        "version": "1.0.0",
        "description": f"从文档 {md_file_path.name} 提取生成的分类分级规则包",
        "rules": rules,
        "downgrade_rules": [
            {
                "id": f"{domain_name}:RULE_DOWN_OPS",
                "name": "运营统计指标降级",
                "keywords": ["stat", "statistics", "count", "cnt", "total_cnt", "total_num", "stat_num", "门诊人次", "汇总"],
                "level": "L2",
                "category": "OPERATIONAL_STAT",
                "force_suppress": True,
                "max_force_suppress_level": "L3",
                "exempt_rules": [],
            }
        ],
        "composite_rules": [
            {
                "id": f"COMP_{domain_name.upper()}_001",
                "name": "多字段组合高敏感规则",
                "field_patterns": ["idcard|identity", "mobile|phone", "bankcard|cardno"],
                "min_matches": 2,
                "target_level": "L4",
                "category": f"COMPOSITE_{domain_name.upper()}_COMBO",
                "table_name_pattern": "",
                "boost_level": "",
            }
        ],
    }

    return profile_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于 Markdown 标准文档自动生成规则 YAML 文件，并执行独立的 Keywords 词表扩充。"
    )
    parser.add_argument("md_file", type=str, help="输入 Markdown 文档路径 (如 docs/standard/四川省健康医疗大数据应用指南.md)")
    parser.add_argument("--domain", type=str, default="custom_domain", help="领域标识名称 (如 sc_health / general-pii)")
    parser.add_argument("-o", "--output", type=str, default="", help="保存的输出 YAML 文件路径 (默认标准输出 stdout)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")), help="LLM API Key")
    parser.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"), help="LLM API Base URL")
    parser.add_argument("--model", type=str, default="qwen-plus", help="使用的大模型名称 (默认 qwen-plus)")

    args = parser.parse_args()

    md_path = Path(args.md_file).resolve()
    if not md_path.exists():
        logger.error(f"输入的 Markdown 文件不存在: {md_path}")
        sys.exit(1)

    profile_data = process_doc_to_yaml(
        md_path,
        domain_name=args.domain,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
    )

    output_yaml = yaml.dump(profile_data, allow_unicode=True, sort_keys=False, default_flow_style=False)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_yaml)
        print(f"成功从 Markdown 文档解析并扩充生成规则包 YAML: {out_path}")
    else:
        print(output_yaml)


if __name__ == "__main__":
    main()
