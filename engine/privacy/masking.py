"""数据脱敏与哈希工具模块 / Data Masking & Hashing Primitive API Implementation.

中文说明：
根据字段名自动识别敏感字段类型（手机号、身份证、姓名、银行卡等），
提供对应的掩码、截断、HMAC 哈希等数据保护措施。
支持单值、批量、DataFrame、流式分块等多种处理模式，
内置输入校验、结构化日志与 Prometheus 指标埋点。

English Description:
Field-name-aware data masking and hashing utilities. Recognizes common sensitive
field types (mobile, ID card, name, bank card, email, address) by name and applies
format-preserving masking, truncation, or HMAC hashing.
Supports scalar, batch, DataFrame, and streaming chunked processing modes with
built-in input validation, structured logging, and Prometheus metrics instrumentation.

扩展能力 / Key Features:
- 多格式输入适配：参考 DP 模块 `extract_values` 设计，
  支持 pandas DataFrame、numpy ndarray、PyArrow Table/RecordBatch、Arrow IPC 字节流、
  Polars、SecretFlow、list of dict 等多种输入格式。
- 向量化批处理：pandas DataFrame 列级 apply 加速，减少 Python 循环开销。
- 结构化日志：每次操作记录操作类型、字段数、记录数等上下文信息。
- 输入校验：统一的参数合法性检查，快速失败并给出清晰错误信息。
- 枚举类型安全：FieldType / MaskingOperation 枚举避免裸字符串拼写错误。
- 流式分块：生成器接口支持超大规模记录集的惰性脱敏。
"""

# === 导入区 / Imports ===
# 启用 PEP 563 延迟注解求值，允许在类型注解中引用尚未定义的类（如自引用）
from __future__ import annotations

import base64  # base64 编解码，用于 HMAC 摘要的可读化输出
import contextlib  # 提供 suppress 等上下文管理工具，用于优雅地忽略预期异常
import functools
import hashlib  # 提供 SHA-256（兼容开关）与 hashlib 接口
import hmac  # HMAC 消息认证码实现，用于加盐哈希
import os  # 读取可选的历史哈希兼容开关
import re  # 正则表达式，用于字段名边界感知匹配与姓名后缀剥离
import secrets  # 密码学安全随机源，用于列洗牌
from dataclasses import dataclass, field  # dataclass 装饰器，自动生成 __init__/__repr__ 等
from enum import Enum  # 枚举基类，用于 FieldType/MaskingOperation
from typing import TYPE_CHECKING, Any, cast

import numpy as np  # NumPy 数组支持，用于 ndarray 格式输入的适配转换

# 从可观测性子包导入结构化日志工厂和 Prometheus Counter 指标实例
from ..observability.logging_config import get_logger
from ..observability.metrics import MASKING_DURATION, MASKING_OPERATIONS_TOTAL, observe_duration
from ..security.sm3 import new_sm3

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# 创建模块级结构化日志记录器，__name__ 自动解析为 "engine.privacy.masking"
# 所有日志调用（logger.info/debug）均通过此实例发出，支持 JSON 格式输出和上下文 extra 字段
logger = get_logger(__name__)


# === 枚举定义区 / Enum Definitions ===


class FieldType(str, Enum):
    """敏感字段类型枚举 / Sensitive Field Type Enum.

    继承 str 保证与字符串的向后兼容性：FieldType.MOBILE == "mobile" 为 True。
    IDE 自动补全 + 静态类型检查，避免裸字符串拼写错误。
    """

    MOBILE = "mobile"        # 手机号类型（匹配 mobile/phone/手机号/联系电话 等，tel 为边界匹配）
    ID_CARD = "id_card"      # 身份证类型（匹配 id_card/idcard/身份证/identity）
    NAME = "name"            # 姓名类型（匹配 姓名/fullname 等，name 为边界匹配）
    BANK_CARD = "bank_card"  # 银行卡类型（匹配 bank/card_no/银行卡 等，card 为边界匹配）
    EMAIL = "email"          # 邮箱类型（匹配 email/mail/邮箱）
    ADDRESS = "address"      # 地址类型（匹配 addr/address/地址/住址）
    MEDICAL = "medical"      # 医疗诊断类型（匹配 diagnosis/诊断/病史/主诉/过敏史 等高敏医疗文本）
    DEFAULT = "default"      # 默认类型（未匹配到任何规则时的兆底）


class MaskingOperation(str, Enum):
    """脱敏操作类型枚举 / Masking Operation Type Enum.

    继承 str 保证与字符串的向后兼容性。
    用于 MaskingResult.operation 字段和 Prometheus 指标标签。
    """

    MASK_VALUE = "mask_value"                          # 单值脱敏操作
    MASK_VALUE_BATCH = "mask_value_batch"              # 批量字段脱敏操作
    MASK_DATAFRAME = "mask_dataframe"                  # DataFrame/表格级脱敏操作
    MASK_RECORD = "mask_record"                        # 整记录脱敏操作
    HASH_VALUE = "hash_value"                          # HMAC 哈希操作
    TRUNCATE = "truncate"                              # 字符串截断操作
    CHUNKED_MASK_RECORDS = "chunked_mask_records"      # 流式分块脱敏操作
    FPE_ENCRYPT = "fpe_encrypt"                        # 保留格式加密操作 (FPE, Format-Preserving Encryption)
    RANDOM_DATE_OFFSET = "random_date_offset"          # 日期随机统一偏移操作 (Random Date Offset)
    COLUMN_SHUFFLE = "column_shuffle"                  # 列洗牌打乱操作 (Column Shuffle)


# === 输入校验函数区 / Input Validation Functions ===
# 设计原则：快速失败（Fail-Fast），在业务逻辑执行前拦截非法参数，
# 抛出带有清晰上下文的 ValueError，避免深层调用栈中产生难以定位的错误。


def _validate_field_name(field_name: str) -> None:
    """校验字段名参数有效性 / Validate field_name parameter.

    Args:
        field_name: 待校验的字段名。

    Raises:
        ValueError: 当字段名为空或不是字符串时抛出。
    """
    # 类型守卫：确保传入的是字符串而非 int/None 等其他类型
    if not isinstance(field_name, str):
        raise ValueError(f"field_name must be a string, got {type(field_name).__name__}")
    # 内容守卫：strip() 去除首尾空白后检查是否为空，防止 "   " 这样的无效输入
    if not field_name.strip():
        raise ValueError("field_name must not be empty or whitespace-only")


def _validate_value(value: str) -> None:
    """校验待脱敏值参数有效性 / Validate value parameter.

    Args:
        value: 待校验的值。

    Raises:
        ValueError: 当值不是字符串时抛出。
    """
    # 脱敏操作仅支持字符串输入；数值/列表等需调用方预先转为 str
    if not isinstance(value, str):
        raise ValueError(f"value must be a string, got {type(value).__name__}")


def _validate_salt(salt: str) -> None:
    """校验 HMAC 盐值参数有效性 / Validate HMAC salt parameter.

    Args:
        salt: 待校验的盐值。

    Raises:
        ValueError: 当盐值为空或不是字符串时抛出。
    """
    # 类型守卫：盐值必须是字符串
    if not isinstance(salt, str):
        raise ValueError(f"salt must be a string, got {type(salt).__name__}")
    # 非空守卫：空盐值会导致 HMAC 退化为普通哈希，丧失加盐保护意义
    if not salt:
        raise ValueError("salt must not be empty for HMAC hashing")


# === 结果包装类 / Result Wrapper ===


@dataclass  # 自动生成 __init__, __repr__, __eq__ 等方法，减少样板代码
class MaskingResult:
    """数据脱敏计算结果及结构化元数据包装。

    Attributes:
        value: 脱敏后的数据结果（标量字符串、字符串列表、字典记录或 DataFrame）。
        operation: 执行的脱敏操作标识（"mask_value" / "mask_dataframe" / "hash" 等）。
        masked_fields: 被脱敏处理的字段名称列表。
        total_masked: 被脱敏记录或字符的计数量。
    """

    value: Any                                          # 脱敏后的实际数据（类型取决于调用接口）
    operation: str                                      # 操作标识，对应 MaskingOperation 枚举值
    masked_fields: list[str] = field(default_factory=list)  # 被脱敏的字段名列表（default_factory 避免可变默认值陷阱）
    total_masked: int = 1                               # 脱敏计数（单值=1，批量=记录数）

    def to_arrow(self):
        """将 MaskingResult 包装转换为附带脱敏 Metadata 的 PyArrow Table。

        执行步骤：
        1. 提取 MaskingResult 的脱敏元数据（操作类型、涉及字段列、脱敏总数）构造 JSON 结构。
        2. 将元数据存入 Schema 字典 Key `b"masking_metadata"`。
        3. 根据 `value` 的类型（字符串/列表/字典/DataFrame）构造对应的 PyArrow Table。
        4. 替换 Table Schema Metadata 后导出。
        """
        import json  # 延迟导入：仅在实际调用 to_arrow 时才加载，避免模块加载开销

        import pyarrow as pa  # 延迟导入：PyArrow 为可选依赖，未安装时不影响其他功能

        # 构建脱敏元数据字典，全部转为字符串以确保 JSON 可序列化
        meta = {
            "operation": str(self.operation),       # 操作类型（如 "mask_value:mobile"）
            "masked_fields": str(self.masked_fields),  # 被脱敏字段列表
            "total_masked": str(self.total_masked),    # 脱敏总数
        }
        # 将元数据序列化为 JSON 字节串，存入 Arrow Schema 的 metadata 字典中
        # key 必须为 bytes 类型（Arrow Schema metadata 规范要求）
        custom_metadata = {b"masking_metadata": json.dumps(meta).encode("utf-8")}

        # 根据 value 的实际类型选择对应的 Table 构建策略
        if isinstance(self.value, str):
            # 标量字符串 → 单行单列 Table
            arr = pa.array([self.value])  # 包装为长度 1 的 Arrow Array
            table = pa.Table.from_arrays([arr], names=["masked_value"])  # 构建单列 Table
        elif isinstance(self.value, list):
            if self.value and isinstance(self.value[0], dict):
                # 字典列表（记录列表）→ 多列 Table，每个 key 为一列
                table = pa.Table.from_pylist(self.value)
            else:
                # 纯值列表 → 单列 Table，每个元素转字符串
                arr = pa.array([str(x) for x in self.value])
                table = pa.Table.from_arrays([arr], names=["masked_value"])
        elif isinstance(self.value, dict):
            # 单条字典记录 → 双列 Table（field 名 + masked_value 值）
            keys = pa.array(list(self.value.keys()))       # 第一列：字段名
            vals = pa.array([str(v) for v in self.value.values()])  # 第二列：脱敏后的值
            table = pa.Table.from_arrays([keys, vals], names=["field", "masked_value"])
        else:
            # 其他类型（如 pandas DataFrame）→ 尝试专用转换，失败则回退为单值字符串
            try:
                import pandas as pd
                if isinstance(self.value, pd.DataFrame):
                    table = pa.Table.from_pandas(self.value)  # pandas → Arrow 零拷贝转换
                else:
                    arr = pa.array([str(self.value)])
                    table = pa.Table.from_arrays([arr], names=["masked_value"])
            except ImportError:
                # pandas 未安装时的兜底：强制转字符串
                arr = pa.array([str(self.value)])
                table = pa.Table.from_arrays([arr], names=["masked_value"])

        # 合并已有 metadata 和自定义脱敏 metadata（保留 Arrow 原生元数据不丢失）
        existing_meta = table.schema.metadata or {}  # 获取现有 schema metadata（可能为 None）
        merged_meta = {**existing_meta, **custom_metadata}  # 字典合并，自定义 key 覆盖同名 key
        # 返回替换了 metadata 的新 Table（Arrow Table 不可变，replace 返回新实例）
        return table.replace_schema_metadata(merged_meta)


# === 字段类型推断 / Field Type Inference ===

# 字段名关键字规则表 / Field-name keyword rule table
# 每条规则: (字段类型, 裸子串关键字, 边界感知关键字)
# - 裸子串关键字：较长、歧义低的关键字（含全部中文关键字），直接子串匹配；
# - 边界感知关键字：短且歧义高的英文关键字（tel/name/mail/card），
#   仅当出现在分词边界（字符串首尾、下划线、数字等非字母字符两侧）时才命中，
#   避免 "hotel" 误中 "tel"、"username"/"hostname"/"filename" 误中 "name"。
# 规则按优先级从高到低排列，第一个命中即返回（短路求值）。
_FIELD_TYPE_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (FieldType.MOBILE.value,
     ("mobile", "phone", "手机号", "手机", "联系电话", "电话号码", "电话"),
     ("tel",)),
    (FieldType.ID_CARD.value,
     ("id_card", "idcard", "身份证", "identity"),
     ()),
    (FieldType.EMAIL.value,
     ("email", "邮箱", "电子邮箱"),
     ("mail",)),
    (FieldType.ADDRESS.value,
     ("addr", "address", "地址", "住址"),
     ()),
    # MEDICAL 必须在 NAME 之前：diagnosis_name/past_history 等复合字段
    # 含 _name/_history 后缀，若 NAME 优先则边界匹配 "name" 会误伤。
    (FieldType.MEDICAL.value,
     ("diagnosis", "diagnosis_name", "chief_complaint", "present_illness",
      "past_history", "personal_history", "family_history", "allergic_history",
      "progress_note", "诊断", "病史", "主诉", "过敏史", "既往史", "个人史",
      "家族史", "现病史"),
     ()),
    (FieldType.NAME.value,
     # fullname/surname/nickname 等常见复合姓名词显式列入，避免被边界规则误排除
     ("姓名", "名字", "fullname", "surname", "nickname"),
     ("name",)),
    (FieldType.BANK_CARD.value,
     ("bank", "card_no", "银行卡", "卡号"),
     ("card",)),  # 边界匹配额外支持 card_number；id_card 已被上方规则先行命中
)


def _bounded_contains(lower_name: str, keyword: str) -> bool:
    """边界感知关键字匹配 / Boundary-Aware Keyword Matching.

    仅当关键字两侧为非字母边界（字符串首尾、下划线、数字、符号等）时返回 True，
    用于抑制短英文关键字的子串误伤（如 "hotel" 中的 "tel"、"username" 中的 "name"）。
    """
    return re.search(rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", lower_name) is not None


@functools.lru_cache(maxsize=2048)
def guess_field_type(field_name: str) -> str:
    """根据字段名猜测敏感字段类型 / Infer Sensitive Field Type from Field Name.

    执行步骤 / Execution Steps:
    1. 将输入字段名转为小写，进行大小写不敏感匹配。
       (Lowercase the field name for case-insensitive matching)
    2. 按优先级遍历关键字规则表：长/中文关键字做裸子串匹配；
       短歧义英文关键字（tel/name/mail/card）做边界感知匹配，
       仅在下划线/首尾等非字母边界处命中。
       (Iterate the rule table by priority: plain substring for distinctive or
        Chinese keywords; boundary-aware matching for short ambiguous English ones)
    3. 返回匹配的 FieldType 枚举值字符串，未命中任何规则时返回 default。
       (Return matched FieldType enum value string; "default" if none matches)

    Args:
        field_name: 字段名（大小写不敏感）/ Field name (case-insensitive).

    Returns:
        字段类型标识 / Field type identifier，可选值参见 FieldType 枚举。
    """
    # 统一转小写，实现大小写不敏感匹配（如 "Mobile"、"PHONE" 均可识别）
    lower = field_name.lower()
    # 规则链：按优先级从高到低依次匹配，第一个命中即返回（短路求值）
    for ftype, substring_kws, bounded_kws in _FIELD_TYPE_RULES:
        if any(kw in lower for kw in substring_kws):  # 裸子串关键字（长词/中文）
            return ftype
        if any(_bounded_contains(lower, kw) for kw in bounded_kws):  # 边界感知关键字（短词）
            return ftype
    # 兜底：未匹配任何规则，返回默认类型（使用通用前后保留策略）
    return FieldType.DEFAULT.value


# === 单类型脱敏函数区 / Per-Type Masking Functions ===
# 设计原则：每个函数只处理一种字段类型，纯函数无副作用，
# 输入输出均为 str，方便单元测试和向量化复用。


def _mask_keep_runes(value: str, prefix: int, suffix: int) -> str:
    """兜底口径：保留首 prefix 与末 suffix 个码点，中间等量星化。

    码点不足以同时容纳首尾时整体置星，避免短敏感值原样泄露。
    """
    n = len(value)
    if n <= prefix + suffix:
        return "*" * n
    return f"{value[:prefix]}{'*' * (n - prefix - suffix)}{value[-suffix:]}"


def mask_mobile(value: str) -> str:
    """中国大陆手机号脱敏 / China Mainland Mobile Number Masking.

    与 Go ``MaskPhone`` 对齐：
    - 11 位 ``1[3-9]`` 号段：保留前 3 后 4；
    - 可带 ``+86`` / ``86`` 前缀：前缀保留并补一个空格；
    - 其余输入按码点保留前 3 后 4，短值整体置星。
    """
    val = value.strip()
    n = len(val)
    if n == 11 and val[0] == "1" and "3" <= val[1] <= "9" and val[2:].isdigit():
        return f"{val[:3]}****{val[7:]}"
    match = re.fullmatch(r"(\+?86)?(1[3-9]\d)(\d{4})(\d{4})", val)
    if match:
        prefix = match.group(1) or ""
        if prefix:
            prefix += " "
        return f"{prefix}{match.group(2)}****{match.group(4)}"
    return _mask_keep_runes(val, 3, 4)


def mask_id_card(value: str) -> str:
    """中国大陆 18 位身份证号脱敏 / China 18-digit ID Card Masking.

    与 Go ``MaskIdCard`` 对齐：保留前 6 位与后 4 位，中间 8 位替换为 ********；
    非标准格式按码点保留前 4 后 4，短值整体置星，绝不原样放行泄漏。
    """
    val = value.strip()
    if len(val) == 18 and val[:17].isdigit() and (val[17].isdigit() or val[17] in "Xx"):
        return f"{val[:6]}********{val[14:]}"
    match = re.fullmatch(r"(\d{6})(\d{8})(\d{3}[\dXx])", val)
    if match:
        return f"{match.group(1)}********{match.group(3)}"
    return _mask_keep_runes(val, 4, 4)


def mask_name(value: str) -> str:
    """中文姓名脱敏 / Chinese Name Masking.

    自动剥离系统与数据集中追加的数字序号、ID与测试后缀（如 "韩雨泽_3" -> "韩雨泽"，"张三-12" -> "张三"，"李四 (3)" -> "李四"）：
    - 2字姓名 (如"张三_1" / "张三"): 保留首字 -> "张*"
    - 3字姓名 (如"韩雨泽_3" / "张三丰"): 保留首尾字 -> "韩**泽" / "张**丰"
    - 4字及以上 (如"欧阳六六_01"): 保留首尾字 -> "欧**六"
    """
    if not value:
        return value

    # 1. 彻底剥离序列号/数字序号/测试后缀与括号（例如 _3, -12, #1, (3), [2], 3 等）
    import re
    clean_val = value.strip()
    clean_val = re.sub(r"[_\-\s#(\[]*\d+[)\]]*$", "", clean_val).strip()
    clean_val = re.sub(r"\d+$", "", clean_val).strip()

    if not clean_val:
        clean_val = value

    val_len = len(clean_val)
    if val_len <= 1:
        return "*"
    if val_len == 2:
        return f"{clean_val[0]}*"
    if val_len == 3:
        return f"{clean_val[0]}**{clean_val[-1]}"

    return f"{clean_val[0]}{'*' * (val_len - 2)}{clean_val[-1]}"


def mask_bank_card(value: str) -> str:
    """银行卡号脱敏 / Bank Card Number Masking.

    与 Go ``MaskBankCard`` 对齐：保留前 6 位 BIN 与末 4 位，中间段星化；
    短于 8 位的异常输入全量星化。
    """
    val = value.strip()
    n = len(val)
    if 16 <= n <= 19 and val.isdigit():
        return f"{val[:6]}{'*' * (n - 10)}{val[-4:]}"
    match = re.fullmatch(r"(\d{6})\d+(\d{4})", val)
    if match:
        return f"{match.group(1)}{'*' * (n - 10)}{match.group(2)}"
    return _mask_keep_runes(val, 4, 4)


def mask_email(value: str) -> str:
    """电子邮箱地址脱敏 / Email Address Masking.

    执行步骤 / Execution Steps:
    1. 以 @ 分割用户名与域名。
       (Split username and domain by @)
    2. 用户名保留首尾字符，中间替换为 ***。
       (Keep first and last char of username, mask middle with ***)
    3. 域名完整保留以维持可路由性。
       (Preserve domain for routability)

    无 @ 符号的字符串使用默认脱敏策略。
    """
    # 守卫：无 @ 的字符串不是合法邮箱，回退到默认策略
    if "@" not in value:
        return mask_default(value)
    # rsplit("@", 1)：从右侧分割，最多分 1 次，确保域名中若含 @ 也不会被错误拆分
    local, domain = value.rsplit("@", 1)  # local="alice", domain="example.com"
    masked_local = (
        (local[0] + "***" if local else "***") if len(local) <= 2
        else f"{local[0]}***{local[-1]}"
    )
    # 拼接脱敏后的用户名 + @ + 完整域名（域名保留以确保可路由性）
    return f"{masked_local}@{domain}"


def mask_address(value: str) -> str:
    """地址信息脱敏 / Address Masking.

    执行步骤 / Execution Steps:
    1. 保留前 6 个字符（通常包含省/市/区信息）。
       (Keep first 6 characters, typically province/city/district)
    2. 剩余部分替换为 ****。
       (Replace remainder with ****)

    长度不超过 6 的字符串原样返回。
    """
    # 守卫：地址太短时全部保留无意义，原样返回
    if len(value) <= 6:
        return value
    # 前 6 字符通常是“北京市朝阳区”等行政区划，保留以维持数据分析可用性
    return f"{value[:6]}****"


def mask_officer_id(value: str) -> str:
    """军官证脱敏（与 Go ``MaskOfficerId`` 对齐）。

    形如 ``军`` + 2~4 位数字 + 2 位数字的证件号保留首尾（军种与序号），
    中间等量星化；其余输入按码点保留前 2 后 2。
    """
    val = value.strip()
    match = re.fullmatch(r"(军)(\d{2,4})(\d{2})", val)
    if match:
        return f"{match.group(1)}{'*' * len(match.group(2))}{match.group(3)}"
    return _mask_keep_runes(val, 2, 2)


def mask_medical(value: str) -> str:
    """医疗诊断/病史文本脱敏 / Medical Diagnosis & History Text Masking.

    医疗诊断文本包含高敏感疾病信息（如梅毒、HIV、癌症等），
    直接保留原文会泄露患者疾病隐私。本函数采用“首字保留 + 其余全星化”策略：
    - 保留首字符（用于确认非空），其余全部替换为 *。
    - 长度 <= 1 时直接返回 "*"。

    Medical diagnosis texts contain highly sensitive disease information
    (e.g. syphilis, HIV, cancer). Keeping original text leaks patient
    disease privacy. This function retains only the first character and
    replaces the rest with asterisks.
    """
    if not value:
        return value
    if len(value) <= 1:
        return "*"
    # 保留首字符，其余全部星化，彻底消除疾病关键词泄露
    return value[0] + "*" * (len(value) - 1)


def mask_default(value: str, prefix: int = 3, suffix: int = 3) -> str:
    """默认脱敏策略 / Default Masking Strategy.

    与 Go ``MaskDefault`` 对齐：保留前后指定位数，中间用 * 填充；
    长度不足以容纳前后缀时至少保留首字符，其余全部星化，防止短值泄露。
    """
    n = len(value)
    if n == 0:
        return value
    if n <= prefix + suffix:
        return f"{value[0]}{'*' * (n - 1)}"
    stars = "*" * (n - prefix - suffix)
    return f"{value[:prefix]}{stars}{value[-suffix:]}"


# === 核心公开接口区 / Core Public API ===


def mask_value(
    field_name: str,
    value: str,
    context: str = "",
    return_details: bool = False,
) -> str | MaskingResult:
    """根据字段类型对单个值进行脱敏 / Mask a Single Value by Field Type.

    执行步骤 / Execution Steps:
    1. 校验 field_name 和 value 参数有效性。
       (Validate field_name and value parameters)
    2. 调用 `guess_field_type` 识别字段敏感类型。
       (Invoke guess_field_type to classify sensitive field type)
    3. 根据分类路由调用对应的脱敏函数。
       (Route to corresponding masking function by type)
    4. 累加脱敏指标 `MASKING_OPERATIONS_TOTAL` 并记录结构化日志。
       (Increment metrics counter and emit structured log)
    5. 若 `return_details=True` 则封装导出 `MaskingResult` 结构。
       (If return_details=True, wrap in MaskingResult)

    Args:
        field_name: 字段名 / Field name for type inference.
        value: 待脱敏值 / Value to mask.
        context: 脱敏上下文标识 / Masking context identifier.
        return_details: 是否返回 MaskingResult / Whether to return MaskingResult.

    Returns:
        脱敏后的字符串或 MaskingResult / Masked string or MaskingResult.
    """
    # Step 1: 参数校验（快速失败，非法输入不会进入业务逻辑）
    _validate_field_name(field_name)  # 校验字段名非空且为字符串
    _validate_value(value)            # 校验待脱敏值为字符串类型

    # Step 2: Prometheus 指标累加（Counter +1，标签为 operation="mask_value"）
    MASKING_OPERATIONS_TOTAL.labels(operation=MaskingOperation.MASK_VALUE.value).inc()

    with observe_duration(MASKING_DURATION, operation=MaskingOperation.MASK_VALUE.value):
        # Step 3: 推断字段敏感类型（返回 FieldType 枚举值字符串）
        ft = guess_field_type(field_name)

        # Step 4: 根据字段类型路由到对应的脱敏函数（策略模式）
        if ft == FieldType.MOBILE:          # 手机号 → 前3后4
            masked_val = mask_mobile(value)
        elif ft == FieldType.ID_CARD:       # 身份证 → 前6后4
            masked_val = mask_id_card(value)
        elif ft == FieldType.NAME:          # 姓名 → 首尾保留
            masked_val = mask_name(value)
        elif ft == FieldType.BANK_CARD:     # 银行卡 → 前4后4
            masked_val = mask_bank_card(value)
        elif ft == FieldType.EMAIL:         # 邮箱 → 用户名首尾+域名保留
            masked_val = mask_email(value)
        elif ft == FieldType.ADDRESS:       # 地址 → 前6字符保留
            masked_val = mask_address(value)
        elif ft == FieldType.MEDICAL:       # 医疗诊断 → 通用前3后3（与 Go MaskByType 对齐）
            masked_val = mask_default(value, 3, 3)
        else:                               # 默认 → 前3后3通用策略
            masked_val = mask_default(value)

        # Step 5: 结构化日志（debug 级别，生产环境默认不输出，避免日志风暴）
        logger.debug(
            "mask_value_completed",
            extra={"field_name": field_name, "field_type": ft, "context": context},
        )

        # Step 6: 根据 return_details 决定返回纯字符串还是结构化包装
        if return_details:
            return MaskingResult(
                value=masked_val,  # 脱敏后的实际值
                operation=f"{MaskingOperation.MASK_VALUE.value}:{ft}",  # 操作标识含字段类型（如 "mask_value:mobile"）
                masked_fields=[field_name],  # 被脱敏的字段名列表
                total_masked=1,              # 单值脱敏计数为 1
            )
        return masked_val  # 默认返回纯字符串，保持最简调用体验


def mask_value_batch(
    field_names: list[str],
    values: list[str],
    context: str = "",
    return_details: bool = False,
) -> list[str] | MaskingResult:
    """批量对字段值进行脱敏 / Batch Mask Field Values.

    执行步骤 / Execution Steps:
    1. 校验 `field_names` 与 `values` 长度等长且非空。
       (Validate field_names and values have equal non-zero length)
    2. 顺序遍历调用 `mask_value` 完成元素级脱敏。
       (Iterate and apply mask_value per element)
    3. 记录结构化日志并累加指标。
       (Emit structured log and increment metrics)
    4. 若 `return_details=True` 封装导出 `MaskingResult`。
       (If return_details=True, wrap in MaskingResult)

    Args:
        field_names: 字段名列表 / List of field names.
        values: 待脱敏值列表 / List of values to mask.
        context: 脱敏上下文标识 / Masking context identifier.
        return_details: 是否返回 MaskingResult / Whether to return MaskingResult.

    Returns:
        脱敏后的字符串列表或 MaskingResult / List of masked strings or MaskingResult.

    Raises:
        ValueError: 当 field_names 与 values 长度不一致时 / When lengths mismatch.
    """
    # 校验：两个列表必须等长（一一对应关系）
    if len(field_names) != len(values):
        raise ValueError(
            f"field_names and values must have the same length, "
            f"got {len(field_names)} and {len(values)}"
        )
    # 校验：空列表无意义，快速失败
    if not field_names:
        raise ValueError("field_names and values must not be empty")

    # Prometheus 指标累加（整批只计 1 次，而非每个元素计 1 次）
    MASKING_OPERATIONS_TOTAL.labels(operation=MaskingOperation.MASK_VALUE_BATCH.value).inc()

    with observe_duration(MASKING_DURATION, operation=MaskingOperation.MASK_VALUE_BATCH.value):
        # 列表推导式 + zip 配对：逐元素调用 mask_value 完成脱敏
        # zip(field_names, values) 将两个列表按位置配对为 (field_name, value) 元组
        masked_list = cast(
            "list[str]",
            [mask_value(fn, val, context, return_details=False) for fn, val in zip(field_names, values)],
        )

        # 结构化日志（info 级别，记录批量操作的字段数和上下文）
        logger.info(
            "mask_value_batch_completed",
            extra={"num_fields": len(field_names), "context": context},
        )

        # 根据 return_details 决定返回纯列表还是结构化包装
        if return_details:
            return MaskingResult(
                value=masked_list,  # 脱敏后的字符串列表
                operation=MaskingOperation.MASK_VALUE_BATCH.value,  # 操作标识
                masked_fields=list(set(field_names)),  # 去重后的字段名集合（set 去重 → list 序列化）
                total_masked=len(masked_list),  # 脱敏总数 = 列表长度
            )
        return masked_list  # 默认返回纯字符串列表


# === 内部格式适配器区 / Internal Format Adapters ===
# 设计模式：责任链（Chain of Responsibility）—— 按类型依次检测，第一个命中即返回。
# 延迟导入：可选依赖（pyarrow/pandas/polars）均在使用时才 import，未安装不影响其他功能。


def _coerce_to_dict(data: Any) -> dict[str, Any]:
    """将多种单行数据格式转换为字典 / Coerce Single-Row Data to Dict.

    中文说明：
    支持 dict（直接返回）、bytes/bytearray（Arrow IPC）、PyArrow Table/RecordBatch、
    numpy ndarray、pandas Series 等格式。

    English Description:
    Supports dict (pass-through), bytes/bytearray (Arrow IPC), PyArrow Table/RecordBatch,
    numpy ndarray, pandas Series, etc.

    Args:
        data: 输入数据 / Input data.

    Returns:
        字典记录 / Dict record.
    """
    # dict: 已是目标格式，直接返回（零开销快速路径）
    if isinstance(data, dict):
        return data

    # bytes/bytearray: 解析 Arrow IPC Stream 二进制字节流 → 取第一行作为记录
    if isinstance(data, (bytes, bytearray)):
        import pyarrow.ipc as ipc  # 延迟导入：仅处理 bytes 时才需要
        reader = ipc.RecordBatchStreamReader(data)  # 创建 IPC 流式读取器
        table = reader.read_all()  # 读取所有 RecordBatch 合并为 Table
        if table.num_rows == 0:  # 空表守卫：无法提取记录
            raise ValueError("Arrow IPC table is empty, cannot extract record")
        return cast("list[dict[str, Any]]", table.to_pylist())[0]  # 转为 Python 字典列表后取第一个

    # PyArrow Table / RecordBatch: 提取第一行各列值构建字典
    try:
        import pyarrow as pa  # 延迟导入：PyArrow 为可选依赖
        if isinstance(data, pa.RecordBatch):
            data = pa.Table.from_batches([data])  # RecordBatch → Table 统一处理
        if isinstance(data, pa.Table):
            if data.num_rows == 0:  # 空表守卫
                raise ValueError("PyArrow table is empty, cannot extract record")
            # 字典推导：遍历列名，取每列第 0 行元素转为 Python 原生值
            return {col: data.column(col)[0].as_py() for col in data.column_names}
    except ImportError:
        pass  # PyArrow 未安装，跳过此分支

    # numpy ndarray: 按维度构建字典
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            # 1-D 数组：每个元素为一个字段，自动生成列名 col_0, col_1, ...
            return {f"col_{i}": str(v) for i, v in enumerate(data)}
        if data.ndim == 2 and data.shape[0] > 0:
            # 2-D 数组：取第一行，每列生成 col_0, col_1, ...
            return {f"col_{j}": str(data[0, j]) for j in range(data.shape[1])}
        raise ValueError("numpy array is empty or has unsupported dimensions")

    # pandas Series → 调用内置 to_dict() 转为字典
    try:
        import pandas as pd  # 延迟导入
        if isinstance(data, pd.Series):
            return cast("dict[str, Any]", data.to_dict())  # Series 的 index 为 key，values 为 value
    except ImportError:
        pass  # pandas 未安装，跳过

    # polars Series：鸭子类型检测（有 to_dict 但无 columns 属性 → 不是 DataFrame）
    if hasattr(data, "to_dict") and not hasattr(data, "columns"):
        with contextlib.suppress(Exception):
            return cast("dict[str, Any]", data.to_dict())

    # 兜底：无法识别的类型原样返回，由调用方决定是否报错
    return cast("dict[str, Any]", data)


def _convert_to_records(data: Any) -> list[dict[str, Any]]:
    """将多种数据格式转换为记录列表 / Convert Multiple Data Formats to Record List.

    中文说明：
    扩展 `data_adapters.to_records` 的格式支持，额外处理 numpy ndarray、
    PyArrow Table/RecordBatch、Arrow IPC 二进制字节流、Polars DataFrame。

    English Description:
    Extends `data_adapters.to_records` with additional format support including
    numpy ndarray, PyArrow Table/RecordBatch, Arrow IPC bytes, and Polars DataFrame.

    执行步骤 / Execution Steps:
    1. 检测 bytes/bytearray 输入，解析 Arrow IPC Stream 并转换为记录列表。
       (Detect bytes/bytearray input, parse Arrow IPC Stream and convert to records)
    2. 检测 PyArrow Table/RecordBatch，直接转换为记录列表。
       (Detect PyArrow Table/RecordBatch, convert to records directly)
    3. 检测 numpy ndarray，按列名或自动列名构建记录列表。
       (Detect numpy ndarray, build records with column names or auto-generated names)
    4. 检测 Polars DataFrame，转换为记录列表。
       (Detect Polars DataFrame, convert to records)
    5. 回退到 `data_adapters.to_records` 处理 pandas/SecretFlow/原生 list 格式。
       (Fallback to data_adapters.to_records for pandas/SecretFlow/native list formats)

    Args:
        data: 输入数据 / Input data (ndarray, Arrow Table, bytes, polars, etc.).

    Returns:
        记录列表 / List of record dicts.

    Raises:
        TypeError: 不支持的数据类型 / Unsupported data type.
    """
    # Step 1: Arrow IPC 二进制字节流 → 解析为 Table 再转记录列表
    if isinstance(data, (bytes, bytearray)):
        import pyarrow.ipc as ipc  # 延迟导入
        reader = ipc.RecordBatchStreamReader(data)  # 创建 IPC 流读取器
        table = reader.read_all()  # 合并所有 batch 为完整 Table
        return cast("list[dict[str, Any]]", table.to_pylist())  # Table → List[Dict]（每行一个字典）

    # Step 2: PyArrow Table / RecordBatch → 直接 to_pylist()
    try:
        import pyarrow as pa
        if isinstance(data, pa.Table):
            return cast("list[dict[str, Any]]", data.to_pylist())  # 列式存储 → 行式字典列表
        if isinstance(data, pa.RecordBatch):
            table = pa.Table.from_batches([data])  # 单 batch → Table
            return cast("list[dict[str, Any]]", table.to_pylist())
    except ImportError:
        pass  # PyArrow 未安装，跳过

    # Step 3: numpy ndarray → 按维度构建记录列表
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            # 1-D 数组：每个元素作为一条单字段记录 {"value": "..."}
            return [{"value": str(v)} for v in data]
        if data.ndim == 2:
            # 2-D 数组：每行为一条记录，列名自动生成 col_0, col_1, ...
            cols = [f"col_{i}" for i in range(data.shape[1])]  # 生成列名列表
            return [{cols[j]: str(row[j]) for j in range(data.shape[1])} for row in data]
        raise ValueError(f"numpy array must be 1-D or 2-D, got {data.ndim}-D")

    # Step 4: Polars DataFrame → 调用内置 to_dicts() 转换
    # 鸭子类型检测：同时具有 to_dicts 和 columns 属性 → Polars DataFrame
    if hasattr(data, "to_dicts") and hasattr(data, "columns"):
        with contextlib.suppress(Exception):
            return cast("list[dict[str, Any]]", data.to_dicts())  # Polars 原生转换，返回 List[Dict]

    # Step 5: 兜底回退到共享适配器（处理 pandas/SecretFlow/原生 list of dict）
    from .data_adapters import to_records  # 延迟导入避免循环依赖
    return to_records(data)  # 统一转换接口


def _mask_arrow_column(col: Any, col_name: str, context: str) -> Any:
    """对 PyArrow Table 单列执行向量化脱敏 / Vectorized Column Masking via PyArrow Compute.

    中文说明：
    利用 pyarrow.compute 的 utf8 函数在列式内存中直接完成脱敏，
    避免 to_pylist() 全量物化到 Python 对象带来的内存峰值与 GC 开销。
    对于无法纯向量化表达的操作（如 email 的 @ 分割），使用
    find_substring + utf8_slice_codeunits + binary_join_element_wise 组合实现。

    English Description:
    Applies masking directly on columnar Arrow memory using pyarrow.compute
    UTF-8 kernels, avoiding full materialization via to_pylist().
    For operations that cannot be expressed purely vectorized (e.g. email @-split),
    combines find_substring + utf8_slice_codeunits + binary_join_element_wise.

    Args:
        col: PyArrow Array 或 ChunkedArray / PyArrow Array or ChunkedArray.
        col_name: 列名，用于推断字段类型 / Column name for field type inference.
        context: 脱敏上下文标识 / Masking context identifier.

    Returns:
        脱敏后的 PyArrow Array / Masked PyArrow Array.
    """
    import pyarrow.compute as pc  # 延迟导入 PyArrow Compute 模块（向量化内核库）

    # 类型守卫：col 必须具有 .type 属性（即 Arrow Array/ChunkedArray），否则原样返回
    if not hasattr(col, "type"):
        return col
    type_str = str(col.type)  # 将 Arrow DataType 转为字符串以便关键字匹配
    # 仅处理字符串类型列（string/large_string/utf8），数值/布尔等列无需脱敏
    if "string" not in type_str and "utf8" not in type_str:
        return col

    # 根据列名推断敏感字段类型（复用 guess_field_type 的关键字匹配逻辑）
    ft = guess_field_type(col_name)

    if ft == FieldType.MOBILE:
        # 手机号脱敏：保留前3后4，中间替换为 ****；非11位原样返回
        length = pc.utf8_length(col)  # 计算每个元素的 Unicode 字符数（向量化）
        masked = pc.binary_join_element_wise(  # 逐元素拼接多个字符串片段
            pc.utf8_slice_codeunits(col, 0, 3),   # 提取前 3 个字符（区号段）
            "****",                                # 固定掩码字符串（标量自动广播）
            pc.utf8_slice_codeunits(col, 7, None), # 提取第 8 位到末尾（后 4 位）
            "",                                    # 空分隔符（join 需要至少一个分隔符参数）
        )
        # 条件选择：仅当长度恰好为 11 时应用掩码，否则保留原值（防御性处理）
        return pc.if_else(pc.equal(length, 11), masked, col)

    if ft == FieldType.ID_CARD:
        # 身份证脱敏：与标量 mask_id_card 完全对齐的分级掩码，绝不原样放行：
        # 18 位 → 前6+********+后4；>=10 位（含 15 位老证）→ 前3+*(n-7)+后4；
        # >=4 位 → 首字符+*(n-2)+尾字符；更短 → 全 * 填充。
        col_clean = pc.utf8_trim_whitespace(col)  # 对齐标量路径的 strip() 预处理
        length = pc.utf8_length(col_clean)  # 向量化计算字符长度
        masked_18 = pc.binary_join_element_wise(  # 18 位标准证掩码
            pc.utf8_slice_codeunits(col_clean, 0, 6),    # 前 6 位（地区码）
            "********",                                   # 8 个星号掩码（隐藏出生日期+顺序码）
            pc.utf8_slice_codeunits(col_clean, 14, None), # 第 15 位到末尾（后 4 位校验码段）
            "",                                           # 空分隔符
        )
        masked_ge10 = pc.binary_join_element_wise(  # >=10 位（如 15 位老证）：前3+(n-7)星+后4
            pc.utf8_slice_codeunits(col_clean, 0, 3),
            # 星号数量钳位到非负（短字符串上该分支结果会被 if_else 丢弃，但 binary_repeat 遇负数会抛错）
            pc.binary_repeat("*", pc.max_element_wise(pc.subtract(length, 7), 0)),
            pc.utf8_slice_codeunits(col_clean, -4, None),
            "",
        )
        masked_ge4 = pc.binary_join_element_wise(  # >=4 位：首字符+(n-2)星+尾字符
            pc.utf8_slice_codeunits(col_clean, 0, 1),
            pc.binary_repeat("*", pc.max_element_wise(pc.subtract(length, 2), 0)),
            pc.utf8_slice_codeunits(col_clean, -1, None),
            "",
        )
        masked_short = pc.binary_repeat("*", length)  # <4 位：全星填充
        # 分级条件选择（顺序与标量 mask_id_card 的 if/elif 链等价）
        result = pc.if_else(pc.greater_equal(length, 4), masked_ge4, masked_short)
        result = pc.if_else(pc.greater_equal(length, 10), masked_ge10, result)
        result = pc.if_else(pc.equal(length, 18), masked_18, result)
        return result

    if ft == FieldType.BANK_CARD:
        # 银行卡脱敏：保留前4后4，中间替换为 " **** **** "；长度<8原样返回
        length = pc.utf8_length(col)  # 向量化计算字符长度
        masked = pc.binary_join_element_wise(  # 逐元素拼接
            pc.utf8_slice_codeunits(col, 0, 4),    # 前 4 位（发卡行标识）
            " **** **** ",                          # 固定掩码（模拟卡号分组格式）
            pc.utf8_slice_codeunits(col, -4, None), # 后 4 位（负索引从末尾截取）
            "",                                     # 空分隔符
        )
        # 条件选择：长度 >= 8 才脱敏（过短的卡号信息量不足）
        return pc.if_else(pc.greater_equal(length, 8), masked, col)

    if ft == FieldType.NAME:
        # 姓名脱敏：与标量 mask_name 完全对齐 —— 先剥离尾部序号/测试后缀
        # （如 "韩雨泽_3"→"韩雨泽"、"张三-12"→"张三"、"王五 (3)"→"王五"），再分级掩码：
        # 1 字→"*"；2 字→首字+*；3 字→首字+**+尾字；4 字及以上→首字+*(n-2)+尾字；空串原样返回。
        col_clean = pc.utf8_trim_whitespace(col)  # 对齐标量 strip() 预处理
        # 剥离尾部序号后缀（_3 / -12 / #1 / (3) / [2] / 纯数字结尾），与标量正则一致
        cleaned = pc.replace_substring_regex(col_clean, r"[_\-\s#(\[]*\d+[)\]]*$", "")
        cleaned = pc.utf8_trim_whitespace(cleaned)
        cleaned = pc.replace_substring_regex(cleaned, r"\d+$", "")
        cleaned = pc.utf8_trim_whitespace(cleaned)
        # 剥离后为空时回退原始值（与标量 clean_val = value 一致）
        cleaned = pc.if_else(pc.equal(pc.utf8_length(cleaned), 0), col, cleaned)
        length = pc.utf8_length(cleaned)  # 向量化计算字符长度
        first_char = pc.utf8_slice_codeunits(cleaned, 0, 1)    # 提取首字符（姓氏）
        last_char = pc.utf8_slice_codeunits(cleaned, -1, None)  # 提取末字符（名字尾字）
        masked_2 = pc.binary_join_element_wise(first_char, "*", "")  # 2字：姓+*
        masked_3 = pc.binary_join_element_wise(first_char, "**", last_char, "")  # 3字：姓+**+尾
        masked_4plus = pc.binary_join_element_wise(  # 4字+：姓+(n-2)星+尾
            first_char,
            # 星号数量钳位到非负（短字符串上该分支结果会被 if_else 丢弃，但 binary_repeat 遇负数会抛错）
            pc.binary_repeat("*", pc.max_element_wise(pc.subtract(length, 2), 0)),
            last_char,
            "",
        )
        # 分级条件选择（与标量 mask_name 一致；空串/Null 保持原值）
        result = col
        result = pc.if_else(pc.equal(length, 1), "*", result)  # 1字→*
        result = pc.if_else(pc.equal(length, 2), masked_2, result)
        result = pc.if_else(pc.equal(length, 3), masked_3, result)
        result = pc.if_else(pc.greater_equal(length, 4), masked_4plus, result)
        return result

    if ft == FieldType.EMAIL:
        # 邮箱脱敏：无@原样返回; 有@时用户名首字+***+尾字+@+域名完整保留
        # 使用 regex 替换实现向量化 email 脱敏，避免 PyArrow 数组索引只接受标量的限制
        at_pos = pc.find_substring(col, "@")  # 向量化查找 @ 符号位置（未找到返回 -1）
        has_at = pc.greater_equal(at_pos, 0)  # 布尔掩码：是否包含 @ 符号
        # 提取 local 部分：split_pattern 按 @ 分割取 index 0（无@时返回原串，始终安全）
        local = pc.list_element(pc.split_pattern(col, pattern="@", max_splits=1), 0)
        local_len = pc.utf8_length(local)  # 计算用户名部分长度
        first_local = pc.utf8_slice_codeunits(local, 0, 1)  # 提取用户名首字符
        # 短 local (<=2 字符): 首字+***+@+域名（信息量不足，不保留尾字）
        domain_after_at = pc.replace_substring_regex(col, r"^[^@]*@", "")  # regex 删除 @ 及之前部分→纯域名
        masked_short = pc.binary_join_element_wise(first_local, "***", "@", domain_after_at, "")  # 拼接短格式
        # 长 local (>2 字符): regex 捕获首字(.)(.+)(尾字)@域名，中间替换为 ***
        masked_long = pc.replace_substring_regex(
            col, r"^(.)(.+)(.)@(.*)$", r"\1***\3@\4"  # \1=首字 \3=尾字 \4=域名
        )
        # 根据 local 长度选择短格式或长格式
        masked = pc.if_else(pc.less_equal(local_len, 2), masked_short, masked_long)
        # 最终条件：有@才脱敏，无@原样返回
        return pc.if_else(has_at, masked, col)

    if ft == FieldType.ADDRESS:
        # 地址脱敏：保留前6字符（省/市/区），剩余替换为 ****；长度<=6原样返回
        length = pc.utf8_length(col)  # 向量化计算字符长度
        masked = pc.binary_join_element_wise(  # 逐元素拼接
            pc.utf8_slice_codeunits(col, 0, 6),  # 前 6 个字符（行政区划信息）
            "****",                               # 固定掩码（隐藏详细地址）
            "",                                   # 空分隔符
        )
        # 条件选择：仅长度 > 6 时脱敏（短地址信息量不足，原样保留）
        return pc.if_else(pc.greater(length, 6), masked, col)

    if ft == FieldType.MEDICAL:
        # 医疗诊断/病史文本脱敏：保留首字符，其余全星化
        # 彻底消除疾病关键词（梅毒、HIV、癌等）泄露
        length = pc.utf8_length(col)
        first_char = pc.utf8_slice_codeunits(col, 0, 1)
        masked = pc.binary_join_element_wise(
            first_char,
            pc.binary_repeat("*", pc.max_element_wise(pc.subtract(length, 1), 0)),
            "",
        )
        # 空串原样返回，长度>=1 时应用掩码
        return pc.if_else(pc.greater_equal(length, 1), masked, col)

    # DEFAULT 默认脱敏：保留前3后3，中间用等量 * 填充；长度<=6原样返回
    length = pc.utf8_length(col)  # 向量化计算字符长度
    prefix = pc.utf8_slice_codeunits(col, 0, 3)   # 提取前 3 个字符
    suffix = pc.utf8_slice_codeunits(col, -3, None)  # 提取后 3 个字符（负索引）
    # 星号数量 = 总长 - 首尾各3；钳位到非负（binary_repeat 遇负数会抛错，短字符串结果由 if_else 丢弃）
    star_count = pc.max_element_wise(pc.subtract(length, 6), 0)
    stars = pc.binary_repeat("*", star_count)  # 向量化重复 * 字符 star_count 次
    masked = pc.binary_join_element_wise(prefix, stars, suffix, "")  # 拼接：前缀+星号+后缀
    # 条件选择：仅长度 > 6 时脱敏（否则中间无字符可掩码）
    return pc.if_else(pc.greater(length, 6), masked, col)


def mask_dataframe(
    df: Any,
    columns: list[str] | None = None,
    context: str = "",
    return_details: bool = False,
) -> Any | MaskingResult:
    """对 DataFrame 中的指定列进行脱敏 / Mask Specified Columns in DataFrame."""
    # Prometheus 指标埋点：每次调用 mask_dataframe 时累加计数器（按操作类型分标签）
    MASKING_OPERATIONS_TOTAL.labels(operation=MaskingOperation.MASK_DATAFRAME.value).inc()
    with observe_duration(MASKING_DURATION, operation=MaskingOperation.MASK_DATAFRAME.value):
        return _mask_dataframe_core(df, columns, context, return_details)


def _mask_dataframe_core(
    df: Any,
    columns: list[str] | None = None,
    context: str = "",
    return_details: bool = False,
) -> Any | MaskingResult:
    """对 DataFrame 中的指定列进行脱敏 / Mask Specified Columns in DataFrame.

    支持多种输入数据格式（参考 DP 模块 `extract_values` 设计）：
    - pandas DataFrame
    - SecretFlow DataFrame / HDataFrame / VDataFrame
    - numpy ndarray（1-D 或 2-D）
    - PyArrow Table / RecordBatch
    - Arrow IPC Stream 二进制字节流（bytes / bytearray）
    - Polars DataFrame
    - list of dict 记录列表

    Supports multiple input data formats (refer to DP module `extract_values` design):
    - pandas DataFrame
    - SecretFlow DataFrame / HDataFrame / VDataFrame
    - numpy ndarray (1-D or 2-D)
    - PyArrow Table / RecordBatch
    - Arrow IPC Stream binary bytes (bytes / bytearray)
    - Polars DataFrame
    - list of dict records

    执行步骤 / Execution Steps:
    1. 优先检测 pandas DataFrame，使用向量化 apply 逐列脱敏。
       (Detect pandas DataFrame first, apply vectorized column-level masking)
    2. 非 pandas 输入通过 `_convert_to_records` 统一转换为记录列表。
       (Convert non-pandas input to record list via _convert_to_records)
    3. 对记录列表中每条记录的字符串字段按字段名推断类型并脱敏。
       (Infer field type by name and mask string fields per record)
    4. 记录结构化日志并根据 `return_details` 返回结果。
       (Emit structured log and return result based on return_details)

    Args:
        df: 输入数据（支持 DataFrame/ndarray/Arrow Table/bytes/polars/list of dict）
            / Input data (supports DataFrame/ndarray/Arrow Table/bytes/polars/list of dict).
        columns: 可选，限定需要脱敏的列名列表 / Optional column name list.
        context: 脱敏上下文标识 / Masking context identifier.
        return_details: 是否返回 MaskingResult / Whether to return MaskingResult.

    Returns:
        脱敏后的 DataFrame 或 MaskingResult / Masked DataFrame or MaskingResult.
    """
    target_cols = columns or []  # 初始化目标列列表（未指定时为空，后续根据数据类型自动推断）

    # Step 1: pandas DataFrame 快速路径 — 列级向量化 apply 脱敏
    try:
        import pandas as pd  # 延迟导入：pandas 为可选依赖，未安装时跳过此分支
        if isinstance(df, pd.DataFrame):  # 类型检测：是否为 pandas DataFrame
            if columns is None:
                # 未指定列时自动选择所有字符串类型列（dtype==object 或 string dtype）
                target_cols = [
                    col for col in df.columns
                    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col])
                ]
            result_df = df.copy()  # 深拷贝原始 DataFrame，避免修改调用方数据（不可变原则）
            for col in target_cols:  # 遍历每个目标列
                if col in result_df.columns:  # 防御性检查：确保列存在
                    # 列级 apply：对每个非 NaN 值调用 mask_value 脱敏，NaN 保留原值
                    result_df[col] = result_df[col].apply(
                        lambda val, col=col: mask_value(col, str(val), context) if pd.notna(val) else val
                    )
            # 结构化日志：记录脱敏完成的上下文信息（行数、列数、列名、上下文）
            logger.info(
                "mask_dataframe_completed",
                extra={
                    "num_rows": len(result_df),       # 处理的总行数
                    "num_cols": len(target_cols),     # 脱敏的列数
                    "columns": target_cols,           # 被脱敏的列名列表
                    "context": context,               # 调用方传入的上下文标识
                },
            )
            if return_details:
                # 返回包装结果：包含脱敏后 DataFrame + 元数据
                return MaskingResult(
                    value=result_df,                                  # 脱敏后的 DataFrame
                    operation=MaskingOperation.MASK_DATAFRAME.value,  # 操作标识
                    masked_fields=target_cols,                        # 被脱敏的字段列表
                    total_masked=len(result_df),                      # 脱敏记录总数
                )
            return result_df  # 直接返回脱敏后的 DataFrame
    except ImportError:
        pass  # pandas 未安装，跳过此分支进入下一个 fast path

    # Step 1.5: PyArrow Table / RecordBatch 快速路径 — 列式计算内核脱敏，避免 to_pylist() 全量物化
    try:
        import pyarrow as pa  # 延迟导入：PyArrow 为可选依赖
        if isinstance(df, pa.RecordBatch):
            # RecordBatch 不支持直接列操作，先包装为单 batch 的 Table
            df = pa.Table.from_batches([df])
        if isinstance(df, pa.Table):  # 类型检测：是否为 PyArrow Table
            if columns is None:
                # 未指定列时自动选择所有字符串类型列（检查 schema 中的类型信息）
                target_cols = [
                    name for name in df.column_names
                    if "string" in str(df.schema.field(name).type)  # 匹配 string/large_string
                    or "utf8" in str(df.schema.field(name).type)    # 匹配 utf8 别名
                ]
            else:
                # 指定了列时过滤掉不存在的列名（防御性处理）
                target_cols = [c for c in columns if c in df.column_names]

            new_columns = []  # 存储处理后的列数据（脱敏列 + 未脱敏列）
            new_names = []    # 存储对应的列名
            for name in df.column_names:  # 遍历所有列（保持原始列顺序）
                col = df.column(name)  # 获取列数据（ChunkedArray）
                if name in target_cols:
                    # 目标列：调用向量化脱敏函数（PyArrow Compute 内核）
                    col = _mask_arrow_column(col, name, context)
                new_columns.append(col)   # 收集列数据（脱敏后或原始）
                new_names.append(name)    # 收集列名

            # 从字典构建新的 PyArrow Table（保持列名与数据的对应关系）
            result_table = pa.table(
                dict(zip(new_names, new_columns))
            )
            # 结构化日志：记录 PyArrow 引擎脱敏完成的上下文信息
            logger.info(
                "mask_dataframe_completed",
                extra={
                    "num_rows": result_table.num_rows,   # 处理的总行数
                    "num_cols": len(target_cols),        # 脱敏的列数
                    "columns": target_cols,              # 被脱敏的列名列表
                    "context": context,                  # 调用方上下文标识
                    "engine": "pyarrow_compute",         # 标记使用的计算引擎
                },
            )
            if return_details:
                # 返回包装结果：包含脱敏后 Arrow Table + 元数据
                return MaskingResult(
                    value=result_table,                              # 脱敏后的 PyArrow Table
                    operation=MaskingOperation.MASK_DATAFRAME.value, # 操作标识
                    masked_fields=target_cols,                       # 被脱敏的字段列表
                    total_masked=result_table.num_rows,              # 脱敏记录总数
                )
            return result_table  # 直接返回脱敏后的 PyArrow Table
    except ImportError:
        pass  # PyArrow 未安装，跳过此分支进入通用路径

    # Step 2: 非 pandas/PyArrow 输入 — 通过扩展格式适配器统一转换为记录列表
    records = _convert_to_records(df)  # 支持 numpy/Polars/Arrow IPC bytes/SecretFlow/list of dict
    if not records:
        # 空数据处理：返回空列表或空 MaskingResult（避免后续索引越界）
        if return_details:
            return MaskingResult(
                value=[],
                operation=MaskingOperation.MASK_DATAFRAME.value,
                masked_fields=[],
                total_masked=0,
            )
        return []

    # Step 3: 确定目标列并对每条记录执行字段级脱敏
    if columns is None:
        # 未指定列时自动选择第一条记录中所有字符串类型的字段
        target_cols = [k for k in records[0] if isinstance(records[0].get(k), str)]

    masked_records = []  # 存储所有脱敏后的记录
    for record in records:  # 遍历每条记录
        new_record = dict(record)  # 浅拷贝记录字典（避免修改原始数据）
        for col in target_cols:  # 遍历每个目标字段
            val = new_record.get(col)  # 获取字段值（可能为 None）
            if isinstance(val, str):  # 仅对字符串值执行脱敏（数值/None 保留原样）
                new_record[col] = mask_value(col, val, context)  # 根据字段名推断类型并脱敏
        masked_records.append(new_record)  # 收集脱敏后的记录

    # 结构化日志：记录通用路径脱敏完成的上下文信息
    logger.info(
        "mask_dataframe_completed",
        extra={"num_rows": len(records), "num_cols": len(target_cols), "context": context},
    )

    if return_details:
        # 返回包装结果：包含脱敏后记录列表 + 元数据
        return MaskingResult(
            value=masked_records,                                # 脱敏后的记录列表
            operation=MaskingOperation.MASK_DATAFRAME.value,     # 操作标识
            masked_fields=target_cols,                           # 被脱敏的字段列表
            total_masked=len(records),                           # 脱敏记录总数
        )
    return masked_records  # 直接返回脱敏后的记录列表


def _hash_digestmod():
    """哈希算法选择：默认 HMAC-SM3；显式兼容开关可回退 SHA-256。"""
    legacy = os.environ.get("PRIVACY_HASH_LEGACY_SHA256", "").strip().lower()
    if legacy in {"1", "true", "yes", "on"}:
        return hashlib.sha256
    return new_sm3


def hash_value(value: str, salt: str, return_details: bool = False) -> str | MaskingResult:
    """对字符串进行 HMAC-SM3 哈希 / HMAC-SM3 Keyed Hash.

    执行步骤 / Execution Steps:
    1. 校验 salt 参数非空且为字符串。
       (Validate salt parameter is non-empty string)
    2. 使用传入密钥与原始字符串构建 HMAC-SM3（国密）。
       (Construct HMAC-SM3 with key and value)
    3. 计算摘要并进行 base64 编码，截取前 16 字符导出。
       (Compute digest, base64 encode, truncate to 16 chars)
    4. 记录结构化日志并累加指标。
       (Emit structured log and increment metrics)

    安全性说明 / Security Notes:
    - 参数 salt 实为 HMAC 的**保密密钥**，而非可公开的随机盐：HMAC 抵抗字典攻击
      的能力依赖该值的机密性，调用方应按密钥管理（保密、按用途隔离、定期轮换）。
    - 输出为 base64 编码截取的前 16 个字符（96 bit 摘要）。按生日界估算，
      约 2^48 次哈希后有约 50% 概率出现碰撞；超大规模去重/关联场景需自行评估碰撞风险。
    - 默认算法为 HMAC-SM3，与 Go ``masking.HashHMAC`` 逐字节一致；
      仅当显式设置 ``PRIVACY_HASH_LEGACY_SHA256`` 时才回退历史 SHA-256 输出。

    Args:
        value: 待哈希的原始字符串 / Original string to hash.
        salt: HMAC 保密密钥（非可公开盐）/ HMAC secret key (keep confidential; must not be empty).
        return_details: 是否返回 MaskingResult / Whether to return MaskingResult.

    Returns:
        16 位 base64 编码的哈希字符串或 MaskingResult / 16-char base64 hash or MaskingResult.

    Raises:
        ValueError: 当 salt 为空时 / When salt is empty.
    """
    _validate_salt(salt)  # 校验盐值参数：必须为非空字符串，否则抛出 ValueError
    # Prometheus 指标埋点：累加 hash_value 操作计数器
    MASKING_OPERATIONS_TOTAL.labels(operation=MaskingOperation.HASH_VALUE.value).inc()
    with observe_duration(MASKING_DURATION, operation=MaskingOperation.HASH_VALUE.value):
        # HMAC-SM3：salt 为密钥，value 为消息，计算消息认证码摘要
        mac = hmac.new(salt.encode(), value.encode(), digestmod=_hash_digestmod()).digest()
        # base64 编码摘要并截取前 16 字符（平衡可读性与碰撞概率）
        hashed = base64.b64encode(mac).decode()[:16]
        # 结构化调试日志：记录原始值长度（不记录原始值本身，避免敏感信息泄漏）
        logger.debug("hash_value_completed", extra={"value_length": len(value)})
        if return_details:
            # 返回包装结果：包含哈希值 + 操作元数据
            return MaskingResult(
                value=hashed,
                operation=MaskingOperation.HASH_VALUE.value,
                masked_fields=[],
                total_masked=1,
            )
        return hashed  # 直接返回 16 位 base64 哈希字符串


def truncate(value: str, keep_prefix: int, return_details: bool = False) -> str | MaskingResult:
    """截断字符串 / Truncate String with Masking Suffix.

    执行步骤 / Execution Steps:
    1. 校验 keep_prefix 必须为非负整数。
       (Validate keep_prefix is non-negative integer)
    2. 判定输入字符串长度，若不超过 keep_prefix 则原样返回。
       (If value length <= keep_prefix, return as-is)
    3. 截取前缀字符并追加 *** 导出。
       (Truncate and append *** suffix)

    Args:
        value: 待截断字符串 / String to truncate.
        keep_prefix: 保留的前缀位数 / Number of prefix characters to keep.
        return_details: 是否返回 MaskingResult / Whether to return MaskingResult.

    Returns:
        截断后的字符串或 MaskingResult / Truncated string or MaskingResult.

    Raises:
        ValueError: 当 keep_prefix 为负数时 / When keep_prefix is negative.
    """
    if keep_prefix < 0:
        # 参数校验：保留前缀位数不能为负数
        raise ValueError(f"keep_prefix must be non-negative, got {keep_prefix}")
    # Prometheus 指标埋点：累加 truncate 操作计数器
    MASKING_OPERATIONS_TOTAL.labels(operation=MaskingOperation.TRUNCATE.value).inc()
    with observe_duration(MASKING_DURATION, operation=MaskingOperation.TRUNCATE.value):
        res = value if len(value) <= keep_prefix else value[:keep_prefix] + "***"
        if return_details:
            # 返回包装结果：包含截断后字符串 + 操作元数据
            return MaskingResult(value=res, operation=MaskingOperation.TRUNCATE.value, masked_fields=[], total_masked=1)
        return res  # 直接返回截断后的字符串


def mask_record(
    record: Any, context: str = "", return_details: bool = False
) -> dict[str, Any] | MaskingResult:
    """对整条记录中的每个字符串值进行脱敏 / Mask All String Fields in Record.

    支持多种输入数据格式（参考 DP 模块 `extract_values` 设计）：
    - dict 记录字典
    - bytes / bytearray（Arrow IPC Stream 二进制字节流，自动解析为单条记录）
    - PyArrow Table / RecordBatch（取第一行作为记录）
    - numpy ndarray（1-D 数组按 {col_0: val, ...} 构建记录；2-D 取第一行）
    - pandas Series（转为 dict）

    Supports multiple input data formats (refer to DP module `extract_values` design):
    - dict record
    - bytes / bytearray (Arrow IPC Stream bytes, auto-parsed to single record)
    - PyArrow Table / RecordBatch (first row as record)
    - numpy ndarray (1-D → {col_0: val, ...}; 2-D → first row)
    - pandas Series (convert to dict)

    执行步骤 / Execution Steps:
    1. 检测输入数据类型并转换为 dict 记录。
       (Detect input data type and convert to dict record)
    2. 校验 record 参数为非空字典。
       (Validate record is a non-empty dict)
    3. 遍历记录字典各项 Key-Value。
       (Iterate over record key-value pairs)
    4. 对字符串类型数据按 Key 名推断敏感类型并调用 `mask_value` 替换。
       (Infer field type by key name and apply mask_value for string values)
    5. 记录结构化日志并返回脱敏后记录或 MaskingResult。
       (Emit structured log and return masked record or MaskingResult)

    Args:
        record: 待脱敏记录（支持 dict/bytes/Arrow/ndarray/Series）
            / Record to mask (supports dict/bytes/Arrow/ndarray/Series).
        context: 脱敏上下文标识 / Masking context identifier.
        return_details: 是否返回 MaskingResult / Whether to return MaskingResult.

    Returns:
        脱敏后的记录字典或 MaskingResult / Masked record dict or MaskingResult.

    Raises:
        ValueError: 当 record 无法转换为字典或为空时 / When record cannot be converted to dict or is empty.
    """
    # Step 1: 将非 dict 输入转换为字典记录（支持 bytes/Arrow/ndarray/Series 等格式）
    record = _coerce_to_dict(record)
    # 类型守卫：转换后仍非字典则报错（无法进行字段级脱敏）
    if not isinstance(record, dict):
        raise ValueError(f"record must be a dict or convertible type, got {type(record).__name__}")
    # 非空守卫：空字典无字段可脱敏，快速失败
    if not record:
        raise ValueError("record must not be empty")
    # Prometheus 指标埋点：累加 mask_record 操作计数器
    MASKING_OPERATIONS_TOTAL.labels(operation=MaskingOperation.MASK_RECORD.value).inc()
    with observe_duration(MASKING_DURATION, operation=MaskingOperation.MASK_RECORD.value):
        # 字典推导式：对每个字符串值调用 mask_value 脱敏，非字符串值保留原样
        masked_rec = {
            k: mask_value(k, v, context) if isinstance(v, str) else v
            for k, v in record.items()
        }
        # 收集被脱敏的字段名列表（仅字符串类型的字段）
        masked_fields = [k for k, v in record.items() if isinstance(v, str)]
        # 结构化日志：记录脱敏字段数和上下文信息
        logger.info(
            "mask_record_completed",
            extra={"num_fields": len(masked_fields), "context": context},
        )
        if return_details:
            # 返回包装结果：包含脱敏后记录 + 元数据
            return MaskingResult(
                value=masked_rec,                                  # 脱敏后的记录字典
                operation=MaskingOperation.MASK_RECORD.value,      # 操作标识
                masked_fields=masked_fields,                       # 被脱敏的字段名列表
                total_masked=len(masked_fields),                   # 脱敏字段总数
            )
        return masked_rec  # 直接返回脱敏后的记录字典


def chunked_mask_records(
    chunks: Iterable[Any],
    columns: list[str] | None = None,
    context: str = "",
    return_details: bool = False,
) -> Iterator[list[dict[str, Any]] | MaskingResult]:
    """分块流式对记录进行脱敏 / Streaming Chunked Record Masking (Generator Interface).

    允许调用方以多个 chunk（生成器/迭代器）分批传入记录，
    每个 chunk 惰性处理并 yield 结果，避免一次性加载全部数据到内存。
    适用于超大规模记录集的脱敏场景。

    支持多种输入数据格式（参考 DP 模块 `extract_values` 设计）：
    - list of dict 记录列表
    - pandas DataFrame
    - numpy ndarray（1-D 或 2-D）
    - PyArrow Table / RecordBatch
    - Arrow IPC Stream 二进制字节流
    - Polars DataFrame
    - SecretFlow DataFrame

    Supports multiple input data formats (refer to DP module `extract_values` design):
    - list of dict records
    - pandas DataFrame
    - numpy ndarray (1-D or 2-D)
    - PyArrow Table / RecordBatch
    - Arrow IPC Stream binary bytes
    - Polars DataFrame
    - SecretFlow DataFrame

    执行步骤 / Execution Steps:
    1. 迭代每个 chunk，通过 `_convert_to_records` 统一转换为记录列表。
       (Iterate each chunk, convert to record list via _convert_to_records)
    2. 对每条记录执行字段级脱敏。
       (Apply field-level masking per record)
    3. 累计每个 chunk 的脱敏字段数与记录数。
       (Accumulate masked field count and record count per chunk)
    4. 记录结构化日志并 yield 脱敏结果。
       (Emit structured log and yield masked result)

    Args:
        chunks: 记录块的可迭代对象（支持多种数据格式）
            / Iterable of record chunks (supports multiple data formats).
        columns: 可选，限定需要脱敏的列名列表 / Optional column filter.
        context: 脱敏上下文标识 / Masking context identifier.
        return_details: 是否对每个 chunk 返回 MaskingResult / Whether to yield MaskingResult per chunk.

    Yields:
        每个 chunk 的脱敏后记录列表，或当 return_details=True 时 yield MaskingResult。
    """
    for chunk_idx, chunk in enumerate(chunks):  # 惰性迭代每个 chunk（生成器模式，不一次性加载全部数据）
        # Prometheus 指标埋点：每个 chunk 累加一次计数器
        MASKING_OPERATIONS_TOTAL.labels(operation=MaskingOperation.CHUNKED_MASK_RECORDS.value).inc()
        with observe_duration(MASKING_DURATION, operation=MaskingOperation.CHUNKED_MASK_RECORDS.value):
            # 通过扩展格式适配器将 chunk 转换为记录列表（支持多种数据格式）
            records = _convert_to_records(chunk)
            masked_chunk: list[dict[str, Any]] = []  # 当前 chunk 脱敏后的记录列表
            all_masked_fields: list[str] = []  # 累计所有被脱敏的字段名
            total_masked = 0  # 累计脱敏字段总次数（字段数 × 记录数）
            for record in records:  # 遍历当前 chunk 中的每条记录
                # 确定目标列：指定了 columns 则用之，否则自动选择所有字符串字段
                target_cols = columns or [k for k, v in record.items() if isinstance(v, str)]
                masked_rec = dict(record)  # 浅拷贝记录（避免修改原始数据）
                chunk_fields: list[str] = []  # 当前记录被脱敏的字段名
                for col in target_cols:  # 遍历每个目标字段
                    val = masked_rec.get(col)  # 获取字段值
                    if isinstance(val, str):  # 仅对字符串值执行脱敏
                        masked_rec[col] = mask_value(col, val, context)  # 根据字段名推断类型并脱敏
                        chunk_fields.append(col)  # 记录被脱敏的字段名
                masked_chunk.append(masked_rec)  # 收集脱敏后的记录
                all_masked_fields.extend(chunk_fields)  # 累计字段名
                total_masked += len(chunk_fields)  # 累计脱敏字段次数
            # 结构化调试日志：记录当前 chunk 处理完成的统计信息
            logger.debug(
                "chunked_mask_records_chunk_completed",
                extra={"chunk_idx": chunk_idx, "num_records": len(masked_chunk), "total_masked": total_masked},
            )
            if return_details:
                # yield 包装结果：包含脱敏后记录列表 + 元数据（set 去重字段名）
                yield MaskingResult(
                    value=masked_chunk,                                        # 当前 chunk 脱敏后的记录列表
                    operation=MaskingOperation.CHUNKED_MASK_RECORDS.value,     # 操作标识
                    masked_fields=list(set(all_masked_fields)),                # 去重后的被脱敏字段名
                    total_masked=total_masked,                                 # 脱敏字段总次数
                )
            else:
                yield masked_chunk  # 直接 yield 当前 chunk 的脱敏后记录列表


# === 高级加密与科研治理算子 / Advanced FPE, Date Offset & Shuffle Operators ===


def _derive_fpe_keystream(secret_key: str, value: str) -> bytes:
    """按计数器模式派生与 value 等字节长的 HMAC-SM3 密钥流。"""
    need = len(value.encode("utf-8"))
    keystream = bytearray()
    counter = 0
    key = secret_key.encode("utf-8")
    message = value.encode("utf-8")
    while len(keystream) < need:
        mac = hmac.new(key, message, digestmod=new_sm3)
        mac.update(counter.to_bytes(4, "big"))
        keystream.extend(mac.digest())
        counter += 1
    return bytes(keystream)


def fpe_encrypt_numeric(value: str, secret_key: str | None = None) -> str:
    """保留格式加密 (FPE, Format-Preserving Encryption) 算子.

    适用场景：身份证号、医保号、病历号等固定格式特征数字串。
    脱敏特性：加密后字符串与原始值保持完全一致的长度、字符集与格式形态。
    业务优势：支持数据库在不改代码/字段类型的前提下执行精准相等查询 (Exact Match Query)。

    算法与安全性说明（如实描述） / Algorithm & Security Notes:
    - 本实现并非 NIST FF1/FF3 等标准 FPE 算法，实际构造为：以计数器模式
      HMAC-SM3(secret_key, value || counter) 派生与输入等长的密钥流，再逐字符做
      模加置换（数字 mod 10、字母 mod 26、分隔符等其他字符原样保留）的确定性
      格式保持编码。与 Go ``masking.FpeEncryptNumeric`` 逐字节一致。
    - 同一密钥下输出确定，支持相等匹配；密钥保密时密文不直接暴露明文。
      但对低熵输入（如固定位数数字串），持有密钥者可穷举反推——安全性依赖
      密钥保密与输入熵，不应视为密码学意义上可证明安全的加密方案。
    - 空密钥直接返回空串：与 Go 对齐，禁止回退到进程级随机密钥或硬编码默认密钥，
      避免跨部署不可复现或可预测的输出。

    Args:
        value: 原始字符串 (如 "110101199001011234").
        secret_key: 加密密钥；None 或空串表示未提供密钥.

    Returns:
        与原始值格式形态一致的伪随机置换值；空输入或空密钥返回空串.
    """
    if not value or not isinstance(value, str):
        return value
    if not secret_key:
        return ""

    keystream = _derive_fpe_keystream(secret_key, value)

    result_chars: list[str] = []
    byte_offset = 0
    for char in value:
        byte_val = keystream[byte_offset]
        if "0" <= char <= "9":
            # 数字：取密钥流字节做 mod-10 加扰置换
            result_chars.append(str((ord(char) - ord("0") + byte_val) % 10))
        elif "A" <= char <= "Z":
            # 字母：保持大小写，取密钥流字节做 mod-26 加扰置换
            result_chars.append(chr(ord("A") + (ord(char) - ord("A") + byte_val) % 26))
        elif "a" <= char <= "z":
            result_chars.append(chr(ord("a") + (ord(char) - ord("a") + byte_val) % 26))
        else:
            # 分隔符、横杠等连字符原样保留
            result_chars.append(char)
        byte_offset += len(char.encode("utf-8"))

    return "".join(result_chars)


def random_date_offset(date_str: str, offset_days: int = 15) -> str:
    """日期统一随机序列偏移 (Random Date Offset) 算子.

    适用场景：科研数据集离线治理、病历就诊时间序列脱敏。
    脱敏特性：同一患者/同一就诊批次的所有日期按统一天数偏移（加/减 N 天）。
    业务优势：抹去绝对时间戳防组合重识别，同时 100% 完整保留事件间的时间差 (Interval Delta)。

    Args:
        date_str: 日期字符串 (支持 YYYY-MM-DD, YYYY/MM/DD, YYYY-MM-DD HH:MM:SS 等).
        offset_days: 偏移天数 (正数为往后偏移，负数为往前偏移).

    Returns:
        偏移后的日期字符串.
    """
    if not date_str or not isinstance(date_str, str):
        return date_str

    import re
    from datetime import datetime, timedelta

    # 正则提取日期部分
    match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", date_str)
    if not match:
        return date_str

    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    try:
        dt = datetime(year, month, day)
        new_dt = dt + timedelta(days=offset_days)
        
        # 原样保持连接分隔符
        sep = "-" if "-" in date_str else ("/" if "/" in date_str else ".")
        new_date_fmt = f"{new_dt.year:04d}{sep}{new_dt.month:02d}{sep}{new_dt.day:02d}"
        
        # 替换原日期部分，保留可能的时间尾巴 (HH:MM:SS)
        return date_str.replace(match.group(0), new_date_fmt)
    except Exception:
        return date_str


def shuffle_column(column_values: list[Any], seed: int | None = None) -> list[Any]:
    """科研列洗牌 (Column Shuffle) 算子.

    适用场景：科研导出数据集中的诊断码、科室、费用列洗牌。
    脱敏特性：同列内部随机打乱对应关系，破坏主键与敏感诊疗属性的行级映射。
    业务优势：保持列级别的整体统计分布 (Mean, Standard Deviation, Distribution) 不变。

    随机性说明 / Randomness Semantics:
    - 默认 seed=None：使用密码学安全随机源 secrets.SystemRandom()，每次调用产生
      不可复现的独立置换；对同长度多列分别调用也会产生不同置换，行级映射真正被破坏。
    - 显式传入 seed：使用 random.Random(seed)，结果可复现，仅供测试/科研复算。
      注意：对同长度多列传入相同 seed 会产生完全相同的置换，行级映射将被保留——
      安全敏感场景请勿固定 seed。

    Args:
        column_values: 原始列数组列表.
        seed: 随机种子；None（默认）使用系统安全随机源，显式传入则可复现.

    Returns:
        打乱洗牌后的列数组列表.
    """
    import random

    if not column_values:
        return column_values

    shuffled = list(column_values)
    if seed is None:
        secrets.SystemRandom().shuffle(shuffled)  # 密码学安全随机源，不可复现
    else:
        rng = random.Random(seed)  # 显式种子：可复现路径
        rng.shuffle(shuffled)
    return shuffled


