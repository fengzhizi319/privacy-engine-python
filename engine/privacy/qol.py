"""查询混淆（Query Obfuscation）模块 / Query Obfuscation Primitive API Implementation.

中文说明：
通过向真实查询中混入若干条虚假查询（dummy queries），降低查询日志被分析时
泄露用户真实意图的风险。当前内置医疗领域与通用领域的 dummy 查询模板。
支持语义槽位替换（Slot-Filling）与长度相近抽样两种混淆策略。
内置输入校验、结构化日志与 Prometheus 指标埋点。

English Description:
Query obfuscation primitive. Mixes a real user query with dummy queries to
mitigate inference attacks against query logs. Built-in medical and generic
domain dummy query pools with semantic slot-filling and length-similarity strategies.
Built-in input validation, structured logging, and Prometheus metrics instrumentation.

扩展能力 / Key Features:
- 枚举类型安全：ObfuscationDomain / ObfuscationStrategy 枚举避免裸字符串拼写错误。
- 语义槽位替换：基于实体词库生成近邻语义 Dummy 查询，提升混淆质量。
- 结构化日志：每次操作记录领域、虚假查询数、策略等上下文信息。
- 输入校验：统一的参数合法性检查，快速失败并给出清晰错误信息。
- 批量处理：支持批量查询混淆，保持接口一致性。
"""

# === 导入区 / Imports ===
# 启用 PEP 563 延迟注解求值，允许在类型注解中引用尚未定义的类（如自引用）
from __future__ import annotations

import random  # 伪随机数生成器，仅用于显式传入 seed 时的可复现路径
import re  # 正则表达式，用于槽位模板的子句切分与数字串掩蔽
import secrets  # 密码学安全随机源，seed=None 时的默认 RNG
from dataclasses import dataclass  # dataclass 装饰器，自动生成 __init__/__repr__ 等
from enum import Enum  # 枚举基类，用于 ObfuscationDomain/ObfuscationStrategy

# 从可观测性子包导入结构化日志工厂和 Prometheus Counter 指标实例
from ..observability.logging_config import get_logger
from ..observability.metrics import QOL_DURATION, QOL_OPERATIONS_TOTAL, observe_duration

# 创建模块级结构化日志记录器，__name__ 自动解析为 "engine.privacy.qol"
# 所有日志调用（logger.info/debug）均通过此实例发出，支持 JSON 格式输出和上下文 extra 字段
logger = get_logger(__name__)


# === 枚举定义区 / Enum Definitions ===


class ObfuscationDomain(str, Enum):
    """查询混淆领域枚举 / Query Obfuscation Domain Enum.

    继承 str 保证与字符串的向后兼容性：ObfuscationDomain.MEDICAL == "medical" 为 True。
    IDE 自动补全 + 静态类型检查，避免裸字符串拼写错误。
    """

    MEDICAL = "medical"  # 医疗领域（使用医疗类 Dummy 查询池 + 疾病实体词库）
    GENERIC = "generic"  # 通用领域（使用公共服务类 Dummy 查询池 + 实体词库）


class ObfuscationStrategy(str, Enum):
    """查询混淆策略枚举 / Query Obfuscation Strategy Enum.

    继承 str 保证与字符串的向后兼容性。
    用于标识不同的混淆方法（语义槽位替换、长度相近抽样、混合策略）。
    在结构化日志中记录实际使用的策略，便于运维分析混淆效果。
    """

    SLOT_FILLING = "slot_filling"      # 语义槽位替换：匹配查询中的实体词，替换为同类近邻实体生成 Dummy
    LENGTH_SIMILARITY = "length_similarity"  # 长度相近抽样：从 Dummy 池中选取与真实查询长度接近的条目
    HYBRID = "hybrid"                  # 混合策略：槽位替换生成部分 + 长度抽样补齐剩余


# === 输入校验函数区 / Input Validation Functions ===
# 设计原则：快速失败（Fail-Fast），在业务逻辑执行前拦截非法参数，
# 抛出带有清晰上下文的 ValueError，避免深层调用栈中产生难以定位的错误。


def _validate_query(query: str) -> None:
    """校验查询参数有效性 / Validate query parameter.

    Args:
        query: 待校验的查询字符串。

    Raises:
        ValueError: 当查询为空或不是字符串时抛出。
    """
    # 类型守卫：确保传入的是字符串而非 int/None/list 等其他类型
    if not isinstance(query, str):
        raise ValueError(f"query must be a string, got {type(query).__name__}")
    # 内容守卫：strip() 去除首尾空白后检查是否为空，防止 "   " 这样的无效输入
    if not query.strip():
        raise ValueError("query must not be empty or whitespace-only")


def _validate_num_dummies(num_dummies: int) -> None:
    """校验虚假查询数量参数有效性 / Validate num_dummies parameter.

    Args:
        num_dummies: 待校验的虚假查询数量。

    Raises:
        ValueError: 当数量不是正整数时抛出。
    """
    # 类型守卫：必须是 int 且不能是 bool（Python 中 bool 是 int 的子类，需显式排除）
    if not isinstance(num_dummies, int) or isinstance(num_dummies, bool):
        raise ValueError(f"num_dummies must be an integer, got {type(num_dummies).__name__}")
    # 范围守卫：至少生成 1 条 Dummy 查询才有混淆意义
    if num_dummies < 1:
        raise ValueError(f"num_dummies must be at least 1, got {num_dummies}")


def _validate_domain(domain: str) -> str:
    """校验并规范化领域参数 / Validate and normalize domain parameter.

    Args:
        domain: 待校验的领域字符串。

    Returns:
        规范化后的领域字符串（小写）/ Normalized domain string (lowercase).

    Raises:
        ValueError: 当领域不是支持的值时抛出。
    """
    # 规范化：转小写 + 去首尾空白，容忍用户输入大小写差异（如 "Medical"、" MEDICAL "）
    normalized = domain.lower().strip()
    # 白名单校验：仅支持 medical / generic 两种领域
    if normalized not in (ObfuscationDomain.MEDICAL.value, ObfuscationDomain.GENERIC.value):
        raise ValueError(
            f"domain must be '{ObfuscationDomain.MEDICAL.value}' or '{ObfuscationDomain.GENERIC.value}', "
            f"got '{domain}'"
        )
    return normalized  # 返回规范化后的值供后续逻辑使用


# === 结果包装类 / Result Wrapper ===


@dataclass  # 自动生成 __init__, __repr__, __eq__ 等方法，减少样板代码
class QoLResult:
    """查询混淆结果及结构化元数据包装。

    Attributes:
        queries: 包含真实查询与虚假 dummy 查询在内的混淆文本列表。
        real_query_index: 真实查询在混淆列表中的索引位置。
        domain: 应用的混淆领域（"medical" / "generic"）。
        num_dummies: 生成的 Dummy 虚假查询数量。
    """

    queries: list[str]       # 混淆后的查询列表（真实查询 + Dummy 查询混合排列）
    real_query_index: int    # 真实查询在列表中的位置索引（调用方据此提取真实查询）
    domain: str              # 混淆领域标识（对应 ObfuscationDomain 枚举值）
    num_dummies: int         # 生成的 Dummy 查询数量（不含真实查询）

    def to_arrow(self):
        """将 QoLResult 包装转换为附带查询混淆 Metadata 的 PyArrow Table。

        执行步骤：
        1. 提取 QoLResult 的混淆元数据（真实查询索引、领域、虚假查询数量）构造 JSON。
        2. 将元数据编码存入 Schema Metadata Key `b"qol_metadata"`。
        3. 构建 `queries` 列并生成 PyArrow Table 导出。
        """
        import json  # 延迟导入：仅在实际调用 to_arrow 时才加载，避免模块加载开销

        import pyarrow as pa  # 延迟导入：PyArrow 为可选依赖，未安装时不影响其他功能

        # 构建混淆元数据字典，全部转为字符串以确保 JSON 可序列化
        meta = {
            "real_query_index": str(self.real_query_index),  # 真实查询位置索引
            "domain": str(self.domain),                      # 混淆领域
            "num_dummies": str(self.num_dummies),            # Dummy 查询数量
        }
        # 将元数据序列化为 JSON 字节串，存入 Arrow Schema 的 metadata 字典中
        # key 必须为 bytes 类型（Arrow Schema metadata 规范要求）
        custom_metadata = {b"qol_metadata": json.dumps(meta).encode("utf-8")}

        # 将混淆查询列表构建为单列 Arrow Table
        arr = pa.array(self.queries)  # Python list → Arrow StringArray
        table = pa.Table.from_arrays([arr], names=["obfuscated_query"])  # 单列 Table

        # 合并已有 metadata 和自定义混淆 metadata（保留 Arrow 原生元数据不丢失）
        existing_meta = table.schema.metadata or {}  # 获取现有 schema metadata（可能为 None）
        merged_meta = {**existing_meta, **custom_metadata}  # 字典合并，自定义 key 覆盖同名 key
        # 返回替换了 metadata 的新 Table（Arrow Table 不可变，replace 返回新实例）
        return table.replace_schema_metadata(merged_meta)


# === 内置虚假查询词库 / Built-in Dummy Query Pools ===
# 设计原则：词库内容应与真实查询场景高度相关，使攻击者无法通过语义分析区分真假查询。
# 每个词库 20 条，覆盖常见查询主题，支持用户通过 medical_pool/generic_pool 参数自定义扩展。

# 医疗领域虚假查询词库（共 20 个）—— 覆盖慢性病、急性病、心理健康、康复等主题
MEDICAL_DUMMY = [
    "高血压患者的日常饮食建议",
    "糖尿病患者运动注意事项",
    "冠心病的早期症状有哪些",
    "流感疫苗接种人群建议",
    "儿童常见过敏反应处理",
    "胃溃疡患者吃什么食物好",
    "哮喘发作时的紧急处理方法",
    "慢性支气管炎的预防措施",
    "骨质疏松患者补钙与运动方案",
    "长期失眠的危害及改善建议",
    "脂肪肝患者运动处方",
    "痛风患者避免食用的食物清单",
    "过敏性鼻炎的日常防治手段",
    "颈椎病康复训练操指南",
    "偏头痛的诱发因素与缓解方式",
    "脑梗塞前兆表现及预防建议",
    "骨质疏松防摔倒安全提示",
    "带状疱疹的临床表现及治疗",
    "过敏性皮炎日常注意事项",
    "甲状腺结节患者饮食禁忌",
]

# 通用领域虚假查询词库（共 20 个）—— 覆盖公共服务、生活便民、交通出行等主题
GENERIC_DUMMY = [
    "天气预报查询",
    "附近医院挂号流程",
    "健康档案如何查询",
    "医保报销比例说明",
    "体检报告解读指南",
    "公积金提取线上办理步骤",
    "个人所得税申报操作引导",
    "社保卡丢失如何在线补办",
    "市民卡网点营业时间查询",
    "生活垃圾分类最新标准",
    "最近的公共图书馆开放时间",
    "电动自行车上牌申领流程",
    "常用快递运费价格对比",
    "附近免费公共停车场推荐",
    "燃气费线上缴费使用指南",
    "自来水水质检测结果公告",
    "本地博物馆门票预约入口",
    "公交线路首末班车时间查询",
    "居住证积分申请材料清单",
    "数字证书在线更新流程",
]

# === 语义实体词库 / Semantic Entity Dictionaries ===
# 用于槽位替换（Slot-Filling）策略：当真实查询中包含以下实体词时，
# 将其替换为同类其他实体生成近邻语义 Dummy，提升混淆质量。

# 医疗领域实体词库（疾病名称）—— 用于匹配医疗查询中的疾病关键词
DISEASES = [
    "高血压", "糖尿病", "冠心病", "流感", "胃溃疡",
    "哮喘", "脑梗塞", "痛风", "失眠", "骨质疏松", "脂肪肝",
    "肺炎", "甲状腺结节", "过敏性鼻炎", "颈椎病"
]

# 通用领域实体词库（公共服务实体）—— 用于匹配通用查询中的服务/证件关键词
ENTITIES = [
    "社保卡", "医保", "公积金", "健康档案", "体检报告",
    "居住证", "天气预报", "市民卡", "数字证书", "身份证"
]

# === 槽位模板防泄露辅助 / Slot-Filling Template Leakage Mitigation ===

# 子句切分正则：中英文标点与空白字符，用于提取包含实体的最小子句
_CLAUSE_SPLIT_RE = re.compile(r"[，,。．.;；:：!！?？、\s]+")
# 数字串掩蔽正则：防止电话/证件号等数字片段进入 Dummy 查询
_DIGITS_RUN_RE = re.compile(r"\d+")


def _extract_entity_clause(query: str, matched_term: str) -> str:
    """提取包含匹配实体的最小子句 / Extract the Minimal Clause Containing the Entity.

    按中英文标点与空白切分查询，仅返回包含实体词的第一个子句，用于构建槽位替换模板，
    避免把真实查询的其他片段（可能含 PII）带入每条 Dummy。切分后找不到时回退完整查询。
    """
    for clause in _CLAUSE_SPLIT_RE.split(query):
        if matched_term in clause:
            return clause
    return query


def obfuscate_query(
    query: str,
    num_dummies: int = 3,
    domain: str = "medical",
    medical_pool: list[str] | None = None,
    generic_pool: list[str] | None = None,
    seed: int | None = None,
    return_details: bool = False,
) -> list[str] | QoLResult:
    """对单个查询进行混淆 / Obfuscate a Single Query with Dummy Queries.

    执行步骤 / Execution Steps:
    1. 校验 query、num_dummies、domain 参数有效性。
       (Validate query, num_dummies, and domain parameters)
    2. 根据 domain 选择医疗（medical）或通用（generic）虚假查询池。
       (Select medical or generic dummy pool by domain)
    3. 优先使用语义实体槽位（Slot-Filling）匹配并生成近邻语义 Dummy 查询。
       (Prefer semantic slot-filling to generate near-neighbor dummies)
    4. 若槽位替换未生成足够 Dummy（含未匹配实体的情形），基于长度相近原则
       从 Dummy 池中无放回抽样补齐；两种来源混合时策略标记为 HYBRID。
       (Top up via length-similarity sampling without replacement when slot filling
        produced fewer than num_dummies; mixed sources are reported as HYBRID)
    5. 将真实 query 随机插入到混淆列表中的某个随机位置。
       (Insert real query at random position in obfuscated list)
    6. 记录结构化日志并返回混淆列表或 QoLResult。
       (Emit structured log and return obfuscated list or QoLResult)

    安全性说明 / Security Notes:
    - 随机源：默认（seed=None）使用密码学安全随机源 secrets.SystemRandom()，
      dummy 组合不可预测；显式传入 seed 时使用 random.Random(seed)，仅供测试复现。
    - 槽位替换的泄露缓解：模板仅保留包含实体的最小子句（按中英文标点/空白切分），
      并将子句中的数字串掩蔽为 *，缓解真实查询中的其他内容（可能含 PII）进入 Dummy。
      该措施是缓解而非根除——实体所在子句的其余文本仍会保留在每条 Dummy 中；
      高敏感场景建议传入自定义 pool（将跳过槽位替换，仅做纯池抽样）。
    - Dummy 采用无放回抽样；仅当候选池小于 num_dummies 时循环补齐（此时可能重复）。

    Args:
        query: 待混淆的真实查询 / Real query to obfuscate.
        num_dummies: 生成的虚假查询数量 / Number of dummy queries to generate.
        domain: 混淆领域 / Obfuscation domain ("medical" or "generic").
        medical_pool: 自定义医疗领域虚假查询池 / Custom medical dummy pool.
        generic_pool: 自定义通用领域虚假查询池 / Custom generic dummy pool.
        seed: 可选随机种子 / Optional random seed for reproducibility.
        return_details: 是否返回 QoLResult / Whether to return QoLResult.

    Returns:
        混淆后的查询列表或 QoLResult / Obfuscated query list or QoLResult.

    Raises:
        ValueError: 当参数无效时 / When parameters are invalid.
    """
    # Step 1: 参数校验（快速失败，拦截非法输入）
    _validate_query(query)           # 校验查询字符串非空
    _validate_num_dummies(num_dummies)  # 校验 Dummy 数量为正整数
    domain = _validate_domain(domain)   # 校验并规范化领域参数

    # Prometheus 指标埋点：每次调用按领域标签累加计数器
    QOL_OPERATIONS_TOTAL.labels(domain=domain).inc()
    with observe_duration(QOL_DURATION, domain=domain):
        # Step 2: 根据领域选择对应的 Dummy 查询池
        is_medical = domain == ObfuscationDomain.MEDICAL.value  # 判断是否为医疗领域
        pool = (medical_pool if is_medical else generic_pool)   # 优先使用用户自定义池
        if pool is None:
            pool = (MEDICAL_DUMMY if is_medical else GENERIC_DUMMY)  # 未自定义则用内置词库

        dummies: list[str] = []       # 存储生成的 Dummy 查询列表
        # 默认（seed=None）使用密码学安全随机源，避免可预测的 Dummy 组合与插入位置；
        # 显式传入 seed 时使用 random.Random(seed) 保证可复现（仅供测试/调试）
        rng = secrets.SystemRandom() if seed is None else random.Random(seed)
        strategy_used = ObfuscationStrategy.LENGTH_SIMILARITY.value  # 默认策略：长度相近抽样

        # Step 3: 语义槽位替换（Slot-Filling）—— 仅在使用内置词库时启用
        # 当用户自定义了 pool 时，实体词库可能与自定义池不匹配，因此跳过槽位替换
        if (is_medical and medical_pool is None) or (not is_medical and generic_pool is None):
            matched_term = None  # 初始化匹配到的实体词
            terms_list = DISEASES if is_medical else ENTITIES  # 根据领域选择实体词库
            placeholder = "{disease}" if is_medical else "{entity}"  # 槽位占位符

            # 遍历实体词库，查找真实查询中包含的第一个实体词
            for t in terms_list:
                if t in query:  # 子串匹配：查询中是否包含该实体词
                    matched_term = t
                    break  # 找到第一个匹配即停止（贪心策略）

            if matched_term:
                # 构建模板，先做泄露缓解（最小化真实查询内容进入 Dummy）：
                # 1. 仅保留包含匹配实体的最小子句，丢弃其他可能含 PII 的片段；
                # 2. 子句内的数字串掩蔽为 *（防止电话/证件号等进入 Dummy）。
                # 例如："我叫张三，如何治疗高血压" → "如何治疗{disease}"
                template = _DIGITS_RUN_RE.sub("*", _extract_entity_clause(query, matched_term))
                template = template.replace(matched_term, placeholder)
                # 从实体词库中排除已匹配的词，避免生成与真实查询相同的 Dummy
                choices = [t for t in terms_list if t != matched_term]
                if choices:
                    # 无放回抽取至多 num_dummies 个实体；不足时剩余数量由 Step 4 长度抽样补齐（HYBRID）
                    selected_terms = rng.sample(choices, min(num_dummies, len(choices)))
                    # 将每个实体填充到模板中生成 Dummy 查询
                    for st in selected_terms:
                        dummies.append(template.replace(placeholder, st))
                    strategy_used = ObfuscationStrategy.SLOT_FILLING.value  # 标记使用槽位替换策略

        # Step 4: 长度相近抽样补齐 —— 当槽位替换未生成足够 Dummy 时启用
        needed = num_dummies - len(dummies)  # 计算还需补齐的 Dummy 数量
        if needed > 0:
            if dummies:
                # 已有部分槽位替换结果 + 还需长度抽样补齐 → 混合策略
                strategy_used = ObfuscationStrategy.HYBRID.value
            # 从 Dummy 池中排除与真实查询或已生成 Dummy 完全相同的条目，并去重（保持顺序）
            filtered_pool = list(dict.fromkeys(p for p in pool if p != query and p not in dummies))
            if not filtered_pool:
                # 防御性处理：如果池被过滤为空，回退使用去重后的完整池
                filtered_pool = list(dict.fromkeys(pool))

            # 长度相近策略：优先选择与真实查询长度差 <= 6 的候选
            query_len = len(query)  # 计算真实查询字符长度
            close_candidates = [p for p in filtered_pool if abs(len(p) - query_len) <= 6]  # 严格阈值
            if not close_candidates:
                # 放宽阈值到 12，扩大候选范围
                close_candidates = [p for p in filtered_pool if abs(len(p) - query_len) <= 12]
            if not close_candidates:
                # 仍然无候选时使用全部池（保证始终能生成 Dummy）
                close_candidates = filtered_pool

            if close_candidates:
                # 无放回抽样补齐（避免重复 Dummy）
                dummies.extend(rng.sample(close_candidates, min(needed, len(close_candidates))))
                # 兜底：候选池小于缺口时循环复用候选补齐数量（此时不可避免出现重复）
                idx = 0
                while len(dummies) < num_dummies:
                    dummies.append(close_candidates[idx % len(close_candidates)])
                    idx += 1
            else:
                # 极端情况：自定义池为空，无法生成任何 Dummy，降级返回较少结果
                logger.warning(
                    "qol_dummy_pool_empty",
                    extra={"domain": domain, "num_dummies": num_dummies},
                )

        # Step 5: 将真实查询随机插入到 Dummy 列表中的某个位置
        pos = rng.randint(0, len(dummies))  # 生成 [0, len(dummies)] 范围内的随机插入位置
        result = list(dummies)  # 复制 Dummy 列表（避免修改原始 dummies）
        result.insert(pos, query)  # 在随机位置插入真实查询（攻击者无法通过位置推断真实查询）

        # Step 6: 结构化日志：记录混淆操作的完整上下文
        logger.info(
            "qol_obfuscate_query_completed",
            extra={
                "domain": domain,               # 混淆领域
                "num_dummies": num_dummies,     # 请求的 Dummy 数量
                "strategy": strategy_used,      # 实际使用的混淆策略
                "real_query_index": pos,        # 真实查询在结果中的位置
            },
        )

        if return_details:
            # 返回包装结果：包含混淆列表 + 元数据
            return QoLResult(
                queries=result,             # 混淆后的查询列表
                real_query_index=pos,       # 真实查询位置索引
                domain=domain,              # 混淆领域
                num_dummies=len(dummies),   # 实际生成的 Dummy 数量（池为空降级时可能少于请求值）
            )
        return result  # 直接返回混淆后的查询列表


def obfuscate_query_batch(
    queries: list[str],
    num_dummies: int = 3,
    domain: str = "medical",
    medical_pool: list[str] | None = None,
    generic_pool: list[str] | None = None,
    seed: int | None = None,
    return_details: bool = False,
) -> list[list[str]] | list[QoLResult]:
    """批量对查询进行混淆 / Batch Obfuscate Queries with Dummy Queries.

    执行步骤 / Execution Steps:
    1. 校验 queries 参数为非空列表。
       (Validate queries is a non-empty list)
    2. 遍历 `queries` 数组，对每个查询依次调用 `obfuscate_query`。
       (Iterate queries and apply obfuscate_query per query)
    3. 记录结构化日志并返回混淆结果列表。
       (Emit structured log and return obfuscated results)

    Args:
        queries: 待混淆的查询列表 / List of queries to obfuscate.
        num_dummies: 每个查询生成的虚假查询数量 / Number of dummies per query.
        domain: 混淆领域 / Obfuscation domain.
        medical_pool: 自定义医疗领域虚假查询池 / Custom medical dummy pool.
        generic_pool: 自定义通用领域虚假查询池 / Custom generic dummy pool.
        seed: 可选随机种子 / Optional random seed.
        return_details: 是否返回 QoLResult 列表 / Whether to return QoLResult list.

    Returns:
        混淆后的查询列表的列表或 QoLResult 列表 / List of obfuscated query lists or QoLResults.

    Raises:
        ValueError: 当 queries 为空时 / When queries is empty.
    """
    # 参数校验：queries 必须为非空列表
    if not queries:
        raise ValueError("queries must not be empty")  # 空列表无实际处理对象，快速失败
    if not isinstance(queries, list):
        # 类型守卫：确保传入的是列表而非字符串/元组等其他类型
        raise ValueError(f"queries must be a list, got {type(queries).__name__}")

    # 结构化日志：记录批量混淆开始（查询总数、领域、每个查询的 Dummy 数）
    logger.info(
        "qol_obfuscate_query_batch_started",
        extra={"num_queries": len(queries), "domain": domain, "num_dummies": num_dummies},
    )

    # 列表推导式：对每个查询调用 obfuscate_query 执行混淆
    # 所有查询共享相同的 num_dummies/domain/pool/seed 参数
    # 注意：显式传入相同 seed 时各查询共用同一 RNG 序列，仅当查询内容不同导致
    # 槽位匹配/候选池不同，结果才可能不同——结构相同的查询在相同 seed 下会产生
    # 完全相同的 Dummy 组合。批量场景建议不传 seed（默认使用系统安全随机源）。
    results = [
        obfuscate_query(
            query,                    # 当前待混淆的查询
            num_dummies=num_dummies,  # 每个查询生成的 Dummy 数量
            domain=domain,            # 混淆领域
            medical_pool=medical_pool,  # 自定义医疗池
            generic_pool=generic_pool,  # 自定义通用池
            seed=seed,                # 随机种子（可复现性）
            return_details=return_details,  # 是否返回 QoLResult
        )
        for query in queries  # 遍历每个查询
    ]

    # 结构化日志：记录批量混淆完成
    logger.info(
        "qol_obfuscate_query_batch_completed",
        extra={"num_queries": len(queries), "domain": domain},
    )

    if return_details:
        return [r for r in results if isinstance(r, QoLResult)]
    return [r for r in results if isinstance(r, list)]
