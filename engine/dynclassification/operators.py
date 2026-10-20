"""内置匹配算子库 / Built-in Matcher Operators.

提供通用匹配算子实现 / Provides generic matcher operator implementations, including:
- regex: 正则表达式匹配 / Regular expression matching
- keyword_contains: 关键词子串包含匹配 / Keyword substring inclusion matching
- prefix_match: 前缀匹配 / Prefix matching
- suffix_match: 后缀匹配 / Suffix matching
- id_card_checksum: 中国大陆身份证校验（GB 11643-1999） / Mainland China ID card checksum
- medical_card_checksum: 上海医保卡号校验 / Shanghai medical card checksum
- icd10_range: ICD-10 编码区间判定 / ICD-10 code range determination
- luhn_checksum: Luhn 算法（银行卡号校验） / Luhn algorithm (bank card checksum)
- length_range: 长度区间匹配 / Length range matching
- exact_match: 准确取值匹配 / Exact value matching
- ip_address: IP 地址匹配 / IP address matching
- mac_address: MAC 地址匹配 / MAC address matching
- chinese_name: 中文姓名匹配 / Chinese name matching
- email: 电子邮箱匹配 / Email address matching

所有算子均为无状态纯函数 / All operators are stateless pure functions, 签名 / signature: (value: Any, params: dict) -> bool.
本模块完全独立，不依赖旧分类引擎代码 / This module is completely independent and does not rely on old classification engine code.

===================================================================================
              算子注册与调用流程 / Operator Registration & Invocation Flow
===================================================================================

  模块加载时 (import operators)             规则评估时 (engine.evaluate)
       │                                         │
       ▼                                         ▼
  @OperatorRegistry.register("regex")      engine._evaluate_single_rule(rule, ...)
       │                                         │
       │ 装饰器注册                                │ 遍历 matchers
       ▼                                         ▼
  OperatorRegistry._operators["regex"]     OperatorRegistry.get("regex")
    = regex_matcher                              │
       │                                         │ 调用算子
       ▼                                         ▼
  _operators = {                           regex_matcher(value, params)
    "regex": regex_matcher,                    │
    "keyword_contains": ...,                   └─→ bool / OperatorResult
    "id_card_checksum": ...,
    ... (13+ 算子)
  }

  算子返回类型 / Operator return types:
    - bool: 简单命中/未命中 → normalize_result() → OperatorResult(hit=True/False)
    - OperatorResult: 携带动态等级/类别 → 引擎直接使用
    - tuple (向后兼容): (hit, level, category) → normalize_result() → OperatorResult
===================================================================================
"""

from __future__ import annotations

# re module for regex-based pattern matching operators
import re
from typing import Any

# Import the registry to register all built-in operators via decorators
from .operator_registry import OperatorRegistry, OperatorResult


# ===========================================================================
# 内部校验工具函数 / Internal Validation Utilities
# ===========================================================================

# Weight factors for China mainland 18-digit ID card checksum (GB 11643-1999).
# Formula: sum(digit[i] * weight[i] for i in 0..16) mod 11 -> check character.
_ID_CARD_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
# Check character mapping: remainder (0~10) -> expected check character.
_ID_CARD_CHARS = ["1", "0", "X", "9", "8", "7", "6", "5", "4", "3", "2"]

# Weight factors for Shanghai medical card 9-digit checksum (first 8 digits participate).
_SH_MEDICAL_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1]

# regex 算子输入长度上限（256KB）：超长输入截断评估，缓解 ReDoS 放大面。
_REGEX_MAX_INPUT_LEN = 256 * 1024


def _validate_id_card(value: str) -> bool:
    """校验中国大陆 18 位身份证号（GB 11643-1999） / Validate mainland China 18-digit ID card.

    Validation steps / 校验步骤:
    1. Check length is exactly 18 characters / 检查长度是否正好为 18 个字符。
    2. Validate format via regex (area code + birthdate + sequence + check digit) / 通过正则验证格式。
    3. Compute weighted checksum of first 17 digits, mod 11, map to check character / 计算前 17 位的加权和，对 11 取模，映射到校验字符。
    4. Compare computed check character with the 18th character / 将计算得出的校验字符与第 18 个字符进行比较。
    """
    # Step 1: Length must be exactly 18 characters.
    if len(value) != 18:
        return False
    # Step 2: Regex validates structure: 6-digit area code + 8-digit birthdate (19xx/20xx)
    # + valid month (01-12) + valid day (01-31) + 3-digit sequence + check char (digit or X).
    if not re.match(
        r"^[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]$",
        value,
    ):
        return False
    try:
        # Step 3: Compute weighted sum of first 17 digits.
        # Formula: total = sum(int(value[i]) * weight[i] for i in range(17))
        total = sum(int(value[i]) * _ID_CARD_WEIGHTS[i] for i in range(17))
        # Map remainder (mod 11) to expected check character.
        expected = _ID_CARD_CHARS[total % 11]
        # Step 4: Compare with actual 18th character (case-insensitive for 'X').
        return value[17].upper() == expected
    except (ValueError, IndexError):
        # Non-numeric characters in first 17 positions or index out of range.
        return False


def _validate_medical_card(value: str) -> bool:
    """校验上海医保卡号 9 位数字校验码 / Validate Shanghai medical card 9-digit check code.

    Algorithm / 算法:
    1. Verify input is exactly 9 digits / 验证输入是否正好为 9 位数字。
    2. Compute weighted sum of first 8 digits using _SH_MEDICAL_WEIGHTS / 使用 _SH_MEDICAL_WEIGHTS 计算前 8 位的加权和。
    3. Check digit = (10 - total % 10) % 10 (mod-10 complement) / 校验位 = (10 - total % 10) % 10。
    4. Compare computed check digit with the 9th digit / 将计算得出的校验位与第 9 个数字进行比较。
    """
    # Step 1: Must be exactly 9 numeric digits.
    if not re.match(r"^\d{9}$", value):
        return False
    # Convert each character to integer for arithmetic.
    digits = [int(c) for c in value]
    # Step 2: Weighted sum of first 8 digits.
    total = sum(digits[i] * _SH_MEDICAL_WEIGHTS[i] for i in range(8))
    # Step 3: Compute expected check digit using mod-10 complement formula.
    expected = (10 - total % 10) % 10
    # Step 4: Compare with actual 9th digit (index 8).
    return digits[8] == expected


def _normalize_icd10(code: str) -> tuple[str, int] | None:
    """解析并归一化 ICD-10 编码，支持 B20.0 与 B200 格式 / Parse and normalize ICD-10 codes, supporting B20.0 and B200 formats.

    Parses an ICD-10 code string into a canonical (letter, number) tuple
    for interval comparison. E.g. "A51.2" -> ("A", 51).
    将 ICD-10 编码字符串解析为规范的 (字母, 数字) 元组以进行区间比较。

    Returns None if the input is not a recognizable ICD-10 code.
    如果输入不是可识别的 ICD-10 编码，则返回 None。
    """
    # Normalize: uppercase, strip whitespace.
    s = str(code).upper().strip() if code else ""
    # Match pattern: 1 uppercase letter + 2 digits + optional dot + 0-2 sub-digits.
    # Examples: "A50", "B20.0", "C341" all match.
    match = re.match(r"^([A-Z])(\d{2})(?:\.?\d{0,2})?$", s)
    if not match:
        return None
    # Return (category_letter, category_number) tuple for lexicographic comparison.
    return match.group(1), int(match.group(2))


def _in_icd10_interval(code: tuple[str, int], start: str, end: str) -> bool:
    """判断 ICD-10 编码是否落在闭区间内 / Check if ICD-10 code falls within closed interval.

    Performs tuple comparison: start_norm <= code <= end_norm.
    执行元组比较：start_norm <= code <= end_norm。
    Tuple comparison is lexicographic: first by letter, then by number.
    元组比较是按字典顺序进行的：首先按字母，然后按数字。
    """
    # Normalize interval boundaries to the same tuple format.
    start_norm = _normalize_icd10(start)
    end_norm = _normalize_icd10(end)
    # If either boundary is invalid, the interval is undefined -> no match.
    if not start_norm or not end_norm:
        return False
    # Closed interval check using Python tuple comparison (letter first, then number).
    return start_norm <= code <= end_norm


def _validate_luhn(value: str, min_length: int = 13, max_length: int = 19) -> bool:
    """Luhn 算法校验（银行卡号通用） / Luhn algorithm validation (universal for bank cards).

    Luhn algorithm (ISO/IEC 7812-1) / Luhn 算法:
    1. From rightmost digit (check digit), double every second digit moving left / 从最右边的数字（校验位）开始，向左每隔一个数字乘以 2。
    2. If doubling results in > 9, subtract 9 (equivalent to sum of digits) / 如果乘积大于 9，则减去 9（相当于各位数字相加）。
    3. Sum all digits (doubled and undoubled) / 将所有数字（乘以 2 的和未乘以 2 的）相加。
    4. Valid if total mod 10 == 0 / 如果总和模 10 等于 0，则有效。
    """
    # Strip whitespace and validate basic format.
    s = value.strip()
    # Must be all digits and within acceptable card number length range.
    if not s.isdigit() or not (min_length <= len(s) <= max_length):
        return False
    digits = [int(d) for d in s]
    # Sum of odd-position digits from right (positions 1, 3, 5, ... from right).
    odd_sum = sum(digits[-1::-2])
    # Sum of even-position digits from right (positions 2, 4, 6, ...):
    # double each, then sum digits of result via divmod(2*d, 10) = (quotient + remainder).
    even_sum = sum(sum(divmod(2 * d, 10)) for d in digits[-2::-2])
    # Valid card number if total is divisible by 10.
    return (odd_sum + even_sum) % 10 == 0


# ===========================================================================
# 注册内置算子 / Register Built-in Operators
# ===========================================================================


@OperatorRegistry.register("regex")
def regex_matcher(value: Any, params: dict[str, Any]) -> bool:
    """正则表达式匹配算子 / Regex matching operator.

    Performs re.search (not fullmatch) so the pattern can match anywhere in the string.
    执行 re.search（不是 fullmatch），因此模式可以匹配字符串中的任何位置。

    ReDoS 缓解：对匹配输入值施加长度上限（256KB），超长输入截断后评估，
    缓解恶意/误配规则模式在超长输入上的灾难性回溯放大面。

    Args (params):
        pattern: str - 正则表达式模式 / Regular expression pattern
    """
    # Guard: value must be a non-empty string to perform regex matching.
    if not isinstance(value, str) or not value:
        return False
    # Guard: pattern must be provided and non-empty.
    pattern = params.get("pattern", "")
    if not pattern:
        return False
    # ReDoS 缓解：超长输入截断评估（上限 256KB）
    if len(value) > _REGEX_MAX_INPUT_LEN:
        value = value[:_REGEX_MAX_INPUT_LEN]
    try:
        # Execute regex search: returns True if pattern matches anywhere in value.
        return bool(re.search(pattern, value))
    except re.error:
        # Invalid regex pattern in configuration: treat as no match (fail-safe).
        return False


@OperatorRegistry.register("keyword_contains")
def keyword_contains_matcher(value: Any, params: dict[str, Any]) -> bool:
    """关键词子串包含匹配算子 / Keyword substring inclusion matching operator.

    将输入值与关键词均归一化（小写 + 去下划线/空格）后，检查是否包含 keywords 列表中的任一关键词。
    Normalizes both the input value and keywords (lowercase + remove underscores/spaces),
    then checks if any keyword from the list is included.

    Normalization ensures 'ID_Card' matches keyword 'idcard'.
    归一化确保 'ID_Card' 匹配关键词 'idcard'。

    Args (params):
        keywords: list[str] - 关键词列表 / Keyword list
        use_word_boundaries: bool - 是否使用单词边界（\\b）进行匹配（默认 False，纯子串匹配） / Whether to use word boundaries (\\b) (default False, pure substring matching).
            注意：启用单词边界时，对原始值（仅小写化）进行正则匹配，而非对归一化后的字符串（已去除分隔符）进行匹配。
            Note: When word boundaries are enabled, regex matching is performed on the raw value (lowercased only),
            not on the normalized string (separators removed).
    """
    keywords = params.get("keywords", [])
    use_word_boundaries = params.get("use_word_boundaries", False)

    if use_word_boundaries:
        # 单词边界模式：对原始值仅做小写化，保留分隔符以使 \b 生效
        raw_lower = str(value).lower() if value else ""
        if not raw_lower:
            return False
        for kw in keywords:
            if kw:
                # 对关键词也仅做小写化（保留原始形态以正确构建正则）
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                if re.search(pattern, raw_lower):
                    return True
        return False
    else:
        # 纯子串模式：归一化后匹配（去下划线/空格）
        norm = str(value).lower().replace("_", "").replace(" ", "") if value else ""
        if not norm:
            return False
        return any(kw.lower().replace("_", "").replace(" ", "") in norm for kw in keywords if kw)


@OperatorRegistry.register("prefix_match")
def prefix_matcher(value: Any, params: dict[str, Any]) -> bool:
    """前缀匹配算子 / Prefix matching operator.

    Checks if the value starts with any of the configured prefixes.
    检查值是否以任何配置的前缀开头。
    Useful for matching ICD codes (e.g. prefixes=["A50", "A51"]).
    对于匹配 ICD 编码很有用。
    默认大小写不敏感（case_insensitive=True），可通过参数关闭。
    Case-insensitive by default (case_insensitive=True), can be turned off via params.

    Args (params):
        prefixes: list[str] - 前缀列表 / Prefix list
        case_insensitive: bool - 是否大小写不敏感（默认 True） / Case insensitive (default True)
    """
    # Guard: must be a non-empty string.
    if not isinstance(value, str) or not value:
        return False
    prefixes = params.get("prefixes", [])
    case_insensitive = params.get("case_insensitive", True)
    if case_insensitive:
        v = value.lower()
        return any(v.startswith(p.lower()) for p in prefixes)
    return any(value.startswith(p) for p in prefixes)


@OperatorRegistry.register("suffix_match")
def suffix_matcher(value: Any, params: dict[str, Any]) -> bool:
    """后缀匹配算子 / Suffix matching operator.

    Checks if the value ends with any of the configured suffixes.
    检查值是否以任何配置的后缀结尾。
    默认大小写不敏感（case_insensitive=True），可通过参数关闭。
    Case-insensitive by default (case_insensitive=True), can be turned off via params.

    Args (params):
        suffixes: list[str] - 后缀列表 / Suffix list
        case_insensitive: bool - 是否大小写不敏感（默认 True） / Case insensitive (default True)
    """
    # Guard: must be a non-empty string.
    if not isinstance(value, str) or not value:
        return False
    suffixes = params.get("suffixes", [])
    case_insensitive = params.get("case_insensitive", True)
    if case_insensitive:
        v = value.lower()
        return any(v.endswith(s.lower()) for s in suffixes)
    return any(value.endswith(s) for s in suffixes)


@OperatorRegistry.register("id_card_checksum")
def id_card_checksum_matcher(value: Any, params: dict[str, Any]) -> bool:
    """中国大陆 18 位身份证校验算子（GB 11643-1999） / Mainland China 18-digit ID card checksum operator.

    Delegates to _validate_id_card which performs full structural + checksum validation.
    委托给执行完整结构 + 校验和验证的 _validate_id_card。

    Args (params): 无额外参数 / No extra parameters
    """
    # Convert value to string (handle None) and delegate to internal validator.
    return _validate_id_card(str(value) if value else "")


@OperatorRegistry.register("medical_card_checksum")
def medical_card_checksum_matcher(value: Any, params: dict[str, Any]) -> bool:
    """上海医保卡号校验算子 / Shanghai medical card checksum operator.

    Delegates to _validate_medical_card for 9-digit checksum verification.
    委托给 _validate_medical_card 进行 9 位校验和验证。

    Args (params): 无额外参数 / No extra parameters
    """
    # Convert value to string (handle None) and delegate to internal validator.
    return _validate_medical_card(str(value) if value else "")


@OperatorRegistry.register("icd10_range")
def icd10_range_matcher(value: Any, params: dict[str, Any]) -> OperatorResult:
    """ICD-10 编码区间判定算子 / ICD-10 code range determination operator.

    判断值是否为合法 ICD-10 编码，并检查是否落在敏感区间内。
    Determines if the value is a valid ICD-10 code and checks if it falls within sensitive intervals.
    返回 OperatorResult，携带动态等级和类别信息。
    Returns OperatorResult, carrying dynamic level and category information.

    Args (params):
        default_level: str - 默认等级（未命中敏感区间时） / Default level (when not hitting sensitive interval)
        upgrade_level: str - 升级等级（命中敏感区间时） / Upgrade level (when hitting sensitive interval)
        intervals: list[dict] - 敏感区间列表 / Sensitive interval list [{start, end, category}]
    """
    icd = _normalize_icd10(str(value) if value else "")
    if not icd:
        return OperatorResult(hit=False)

    intervals = params.get("intervals", [])
    for interval in intervals:
        start = interval.get("start", "")
        end = interval.get("end", "")
        if _in_icd10_interval(icd, start, end):
            level = params.get("upgrade_level", "L4")
            category = interval.get("category", "")
            return OperatorResult(hit=True, level=level, category=category)

    level = params.get("default_level", "L3")
    category = "MEDICAL_ICD10_GENERAL"
    return OperatorResult(hit=True, level=level, category=category)


@OperatorRegistry.register("luhn_checksum")
def luhn_checksum_matcher(value: Any, params: dict[str, Any]) -> bool:
    """Luhn 算法校验算子（银行卡号通用校验） / Luhn algorithm checksum operator (universal for bank cards).

    Validates card numbers using the Luhn algorithm (ISO/IEC 7812-1).
    使用 Luhn 算法验证卡号。
    Supports configurable length bounds for different card types.
    支持针对不同卡类型配置长度边界。

    Args (params):
        min_length: int - 最小长度（默认 13） / Minimum length (default 13)
        max_length: int - 最大长度（默认 19） / Maximum length (default 19)
    """
    # Extract configurable length bounds from params with sensible defaults.
    min_len = params.get("min_length", 13)
    max_len = params.get("max_length", 19)
    # Delegate to internal Luhn validator with length constraints.
    return _validate_luhn(str(value) if value else "", min_len, max_len)


@OperatorRegistry.register("length_range")
def length_range_matcher(value: Any, params: dict[str, Any]) -> bool:
    """字符串长度范围匹配算子 / String length range matching operator.

    Checks if the string representation of value falls within [min_length, max_length].
    检查值的字符串表示的长度是否在 [min_length, max_length] 内。
    Useful as a secondary filter (e.g. phone numbers are 11 digits).
    作为辅助过滤器很有用（例如电话号码为 11 位）。

    Args (params):
        min_length: int - 最小长度 / Minimum length
        max_length: int - 最大长度 / Maximum length
    """
    # Convert to string for length measurement.
    s = str(value) if value else ""
    # Extract bounds; default min=0, max=infinity (accept any length).
    min_len = params.get("min_length", 0)
    max_len = params.get("max_length", float("inf"))
    # Closed interval check on string length.
    return min_len <= len(s) <= max_len


@OperatorRegistry.register("exact_match")
def exact_match_matcher(value: Any, params: dict[str, Any]) -> bool:
    """精确匹配算子（归一化后完全相等） / Exact match operator (completely equal after normalization).

    Normalizes both the value and allowed values (lowercase + strip underscores/spaces)
    then checks for exact equality. Useful for enum-like fields (e.g. gender='M'/'F').
    归一化值和允许的值（小写 + 去除下划线/空格），然后检查是否完全相等。对枚举类字段很有用。

    Args (params):
        values: list[str] - 允许的值列表 / List of allowed values
    """
    # Normalize input value for comparison.
    norm = str(value).lower().replace("_", "").replace(" ", "") if value else ""
    # Retrieve the list of allowed values.
    allowed = params.get("values", [])
    # Check if normalized value matches any normalized allowed value.
    return norm in [v.lower().replace("_", "").replace(" ", "") for v in allowed]


@OperatorRegistry.register("ip_address")
def ip_address_matcher(value: Any, params: dict[str, Any]) -> bool:
    """IPv4 / IPv6 地址判定算子 / IPv4 / IPv6 address determination operator.

    Validates whether the value is a well-formed IPv4 or IPv6 address.
    验证该值是否为格式良好的 IPv4 或 IPv6 地址。
    IPv4: 4 octets (0-255) separated by dots / IPv4：由点分隔的 4 个八位字节 (0-255)。
    IPv6: 8 groups of 4 hex digits separated by colons (full form only) / IPv6：由冒号分隔的 8 组 4 位十六进制数字（仅限完整形式）。
    """
    # Guard: must be a non-empty string.
    if not isinstance(value, str) or not value:
        return False
    # IPv4 pattern: each octet is 0-255 (25[0-5] | 2[0-4]x | [01]?xx).
    ipv4_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    # IPv6 pattern: full 8-group notation (does not handle :: abbreviation).
    ipv6_pattern = r"^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"
    s = value.strip()
    # Return True if either IPv4 or IPv6 pattern matches.
    return bool(re.match(ipv4_pattern, s) or re.match(ipv6_pattern, s))


@OperatorRegistry.register("mac_address")
def mac_address_matcher(value: Any, params: dict[str, Any]) -> bool:
    """MAC 地址匹配算子 / MAC address matching operator.

    Validates standard MAC address format: 6 groups of 2 hex digits
    separated by colons or hyphens (e.g. "AA:BB:CC:DD:EE:FF").
    验证标准 MAC 地址格式：由冒号或连字符分隔的 6 组 2 位十六进制数字。
    """
    # Guard: must be a non-empty string.
    if not isinstance(value, str) or not value:
        return False
    # MAC pattern: 6 pairs of hex digits separated by ':' or '-'.
    mac_pattern = r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$"
    return bool(re.match(mac_pattern, value.strip()))


@OperatorRegistry.register("chinese_name")
def chinese_name_matcher(value: Any, params: dict[str, Any]) -> bool:
    """中文姓名匹配算子（2~4 字常见汉字姓名模式） / Chinese name matching operator (2~4 common CJK ideographs pattern).

    Matches strings consisting of 2-4 CJK Unified Ideographs (U+4E00~U+9FA5).
    匹配由 2-4 个 CJK 统一表意文字组成的字符串。
    This covers the vast majority of Chinese personal names.
    这涵盖了绝大多数中文人名。
    """
    # Guard: must be a non-empty string.
    if not isinstance(value, str) or not value:
        return False
    # Pattern: exactly 2-4 characters in the CJK Unified Ideographs block.
    name_pattern = r"^[一-龥]{2,4}$"
    return bool(re.match(name_pattern, value.strip()))


@OperatorRegistry.register("email")
def email_matcher(value: Any, params: dict[str, Any]) -> bool:
    """电子邮箱地址匹配算子 / Email address matching operator.

    检测值是否符合标准电子邮箱格式（RFC 5322 简化版）。
    Checks if the value conforms to standard email format (RFC 5322 simplified).
    邮箱属于常见 PII 类型，广泛用于个人身份识别。
    Emails are a common PII type, widely used for personal identification.

    Args (params): 无额外参数 / No extra parameters
    """
    if not isinstance(value, str) or not value:
        return False
    # RFC 5322 简化版邮箱正则：本地部分@域名部分
    email_pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(email_pattern, value.strip()))