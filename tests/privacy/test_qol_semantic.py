"""查询混淆高级语义特性单元测试 / Query Obfuscation Advanced Semantic Feature Tests.

中文说明：
本模块验证查询混淆（QOL）模块的高级语义能力，超越简单的随机替换：

1. 语义槽位填充 / Semantic Slot Filling:
   - 识别真实查询中的语义实体（如疾病名、实体名）
   - 保持句式结构不变，仅替换槽位实体
   - 示例："如何治疗高血压" → "如何治疗糖尿病"（保持"如何治疗{disease}"结构）

2. 领域感知 / Domain Awareness:
   - medical 领域：从 DISEASES 词库中选取同类疾病
   - generic 领域：从 ENTITIES 词库中选取同类实体

3. 长度过滤回退 / Length Filtering Fallback:
   - 当无法匹配语义实体时，根据真实查询长度过滤静态混淆池
   - 确保生成的 dummy 查询与真实查询在长度上接近，降低可区分性

English Description:
Tests for advanced semantic features of the query obfuscation module,
including slot filling, domain awareness, and length-based filtering fallback.
"""

from __future__ import annotations

from engine.privacy.qol import DISEASES, ENTITIES, obfuscate_query


def test_obfuscate_query_semantic_slot_filling_medical() -> None:
    """验证 medical 领域的语义槽位填充。

    真实查询"如何治疗高血压"包含 DISEASES 词库中的"高血压"，
    系统应识别"如何治疗{disease}"模式，生成的 dummy 保持相同句式，
    仅将槽位替换为其他疾病名（如糖尿病、冠心病等）。
    """
    # 真实查询包含 "高血压" (属于 DISEASES) 且模式为 "如何治疗高血压"
    real_query = "如何治疗高血压"
    result = obfuscate_query(real_query, num_dummies=3, domain="medical", seed=42)

    assert real_query in result
    assert len(result) == 4

    # 虚假查询应当保持 "如何治疗{disease}" 的句式结构
    for item in result:
        if item != real_query:
            assert item.startswith("如何治疗")
            disease_part = item.replace("如何治疗", "")
            assert disease_part in DISEASES
            assert disease_part != "高血压"


def test_obfuscate_query_semantic_slot_filling_generic() -> None:
    """验证 generic 领域的语义槽位填充。

    真实查询"公积金查询"包含 ENTITIES 词库中的"公积金"，
    系统应识别"{entity}查询"模式，生成的 dummy 保持"查询"后缀，
    仅替换前缀实体（如社保、税务等）。
    """
    # 真实查询包含 "公积金" (属于 ENTITIES) 且模式为 "公积金查询"
    real_query = "公积金查询"
    result = obfuscate_query(real_query, num_dummies=2, domain="generic", seed=42)

    assert real_query in result
    assert len(result) == 3

    for item in result:
        if item != real_query:
            assert item.endswith("查询")
            entity_part = item.replace("查询", "")
            assert entity_part in ENTITIES
            assert entity_part != "公积金"


def test_obfuscate_query_length_filtering_fallback() -> None:
    """验证无语义匹配时的长度过滤回退策略。

    当查询不包含任何已知语义实体时，系统回退到静态混淆池，
    但会根据真实查询的长度进行过滤，选择长度接近的 dummy，
    避免短查询与长查询混合导致攻击者轻易识别真实查询。
    """
    # 当无法匹配任何语义实体时，应该根据真实查询的长度过滤静态混淆池
    # 输入一个非常长且不包含已知实体的 Query
    long_query = "我想知道在本地办理跨省异地医保备案和报销需要的具体纸质材料有哪些"
    result = obfuscate_query(long_query, num_dummies=3, domain="generic", seed=42)

    assert long_query in result
    assert len(result) == 4

    # 检查生成的 dummy query 的长度，不应该比 long_query 短得太离谱
    # 原静态混淆池里有较长和较短的查询，长度分析机制应尽可能选长度接近的
    for item in result:
        if item != long_query:
            # 真实查询长度为 30 左右，过滤机制应使 dummy 长度至少在合理范围内（>10）
            assert len(item) > 10
