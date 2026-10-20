"""动态分类分级 - 内置算子边界与容错测试 / Operator Edge Case Tests."""

import pytest
from typing import Any
from engine.dynclassification.operator_registry import OperatorRegistry, OperatorResult
from engine.dynclassification import operators  # noqa: F401


class TestRegexOperator:
    """regex 算子测试"""

    def test_regex_basic_match(self):
        op = OperatorRegistry.get("regex")
        assert op("13800138000", {"pattern": r"^1[3-9]\d{9}$"}) is True
        assert op("12800138000", {"pattern": r"^1[3-9]\d{9}$"}) is False

    def test_regex_invalid_inputs(self):
        op = OperatorRegistry.get("regex")
        assert op(None, {"pattern": r"\d+"}) is False
        assert op("", {"pattern": r"\d+"}) is False
        assert op(12345, {"pattern": r"\d+"}) is False  # 非字符串输入
        assert op("abc", {"pattern": ""}) is False  # 空模式

    def test_regex_malformed_pattern(self):
        op = OperatorRegistry.get("regex")
        # 非法正则表达式语法，需安全捕获不崩溃
        assert op("abc", {"pattern": "[a-z"}) is False


class TestKeywordContainsOperator:
    """keyword_contains 算子测试"""

    def test_keyword_contains_normalization(self):
        op = OperatorRegistry.get("keyword_contains")
        # 验证归一化：小写 + 去除下划线 + 去除空格
        assert op("User_Phone_Number", {"keywords": ["phone"]}) is True
        assert op("brca 1 status", {"keywords": ["brca1"]}) is True
        assert op("TP_53_GENE", {"keywords": ["tp53"]}) is True

    def test_keyword_contains_miss(self):
        op = OperatorRegistry.get("keyword_contains")
        assert op("normal_field", {"keywords": ["phone", "idcard"]}) is False
        assert op("", {"keywords": ["phone"]}) is False


class TestPrefixSuffixOperators:
    """prefix_match 和 suffix_match 算子测试"""

    def test_prefix_matcher(self):
        op = OperatorRegistry.get("prefix_match")
        assert op("BAM\x01header", {"prefixes": ["BAM\x01", "@SQ"]}) is True
        assert op("FASTQ_seq", {"prefixes": ["BAM\x01", "@SQ"]}) is False
        assert op(None, {"prefixes": ["BAM"]}) is False

    def test_suffix_matcher(self):
        op = OperatorRegistry.get("suffix_match")
        assert op("patient_data.vcf", {"suffixes": [".vcf", ".bam"]}) is True
        assert op("patient_data.txt", {"suffixes": [".vcf", ".bam"]}) is False
        assert op(123, {"suffixes": [".txt"]}) is False


class TestIDCardChecksumOperator:
    """id_card_checksum 算子测试（GB 11643-1999）"""

    def test_valid_id_cards(self):
        op = OperatorRegistry.get("id_card_checksum")
        # 合法身份证号校验码验证
        assert op("110101199003072375", {}) is True

    def test_invalid_id_cards(self):
        op = OperatorRegistry.get("id_card_checksum")
        assert op("110101199003072374", {}) is False  # 校验码算错
        assert op("12345", {}) is False  # 长度不够
        assert op("11010119900307237X1", {}) is False  # 长度超长
        assert op("AAAAAAAAAAAAAAAAAA", {}) is False  # 非数字
        assert op(None, {}) is False


class TestMedicalCardChecksumOperator:
    """medical_card_checksum 算子测试"""

    def test_medical_card_checksum(self):
        op = OperatorRegistry.get("medical_card_checksum")
        # 上海医保卡 9 位数字校验测试
        assert op("12345678", {}) is False  # 长度不足 9 位
        assert op(None, {}) is False


class TestICD10RangeOperator:
    """icd10_range 算子测试"""

    def test_icd10_hiv_sensitive_range(self):
        op = OperatorRegistry.get("icd10_range")
        params = {
            "default_level": "L3",
            "upgrade_level": "L4",
            "intervals": [
                {"start": "B20", "end": "B24", "category": "MEDICAL_ICD10_HIV"}
            ]
        }
        # B20.0 落在 B20-B24 敏感区间，应提升至 L4
        result = op("B20.0", params)
        assert isinstance(result, OperatorResult)
        assert result.hit is True
        assert result.level == "L4"
        assert result.category == "MEDICAL_ICD10_HIV"

    def test_icd10_general_code(self):
        op = OperatorRegistry.get("icd10_range")
        params = {
            "default_level": "L3",
            "upgrade_level": "L4",
            "intervals": [
                {"start": "B20", "end": "B24", "category": "MEDICAL_ICD10_HIV"}
            ]
        }
        # J00 (普通感冒) 不在敏感区间，使用默认 L3
        result = op("J00", params)
        assert result.hit is True
        assert result.level == "L3"
        assert result.category == "MEDICAL_ICD10_GENERAL"

    def test_invalid_icd10_code(self):
        op = OperatorRegistry.get("icd10_range")
        result = op("INVALID_ICD_123", {})
        assert result.hit is False
        result2 = op("", {})
        assert result2.hit is False
        result3 = op(None, {})
        assert result3.hit is False


class TestLuhnChecksumOperator:
    """luhn_checksum 算子测试"""

    def test_luhn_bankcard_valid(self):
        op = OperatorRegistry.get("luhn_checksum")
        # 标准银行卡号校验
        assert op("6222021001123456789", {"min_length": 13, "max_length": 19}) is True

    def test_luhn_bankcard_invalid(self):
        op = OperatorRegistry.get("luhn_checksum")
        assert op("6222021001123456780", {"min_length": 13, "max_length": 19}) is False  # 校验和算错
        assert op("123", {"min_length": 13, "max_length": 19}) is False  # 长度不足


class TestLengthAndExactMatchOperators:
    """length_range 与 exact_match 算子测试"""

    def test_length_range(self):
        op = OperatorRegistry.get("length_range")
        assert op("123456", {"min_length": 5, "max_length": 10}) is True
        assert op("123", {"min_length": 5, "max_length": 10}) is False

    def test_exact_match(self):
        op = OperatorRegistry.get("exact_match")
        assert op("Male", {"values": ["male", "female"]}) is True
        assert op("Unknown", {"values": ["male", "female"]}) is False


class TestNewOperators:
    """ip_address, mac_address, chinese_name 算子测试"""

    def test_ip_address(self):
        op = OperatorRegistry.get("ip_address")
        assert op("192.168.1.1", {}) is True
        assert op("2001:0db8:85a3:0000:0000:8a2e:0370:7334", {}) is True
        assert op("999.999.999.999", {}) is False
        assert op(None, {}) is False

    def test_mac_address(self):
        op = OperatorRegistry.get("mac_address")
        assert op("00:1B:44:11:3A:B7", {}) is True
        assert op("00-1B-44-11-3A-B7", {}) is True
        assert op("invalid_mac", {}) is False
        assert op("", {}) is False

    def test_chinese_name(self):
        op = OperatorRegistry.get("chinese_name")
        assert op("张三", {}) is True
        assert op("欧阳六六", {}) is True
        assert op("John", {}) is False
        assert op(123, {}) is False

    def test_email_matcher(self):
        """email 算子检测电子邮箱格式。"""
        op = OperatorRegistry.get("email")
        assert op("user@example.com", {}) is True
        assert op("test.name+tag@sub.domain.org", {}) is True
        assert op("not_an_email", {}) is False
        assert op("@missing_local.com", {}) is False
        assert op("missing_at.com", {}) is False
        assert op("", {}) is False
        assert op(123, {}) is False

    def test_prefix_matcher_case_insensitive(self):
        """prefix_match 默认大小写不敏感。"""
        op = OperatorRegistry.get("prefix_match")
        # 默认大小写不敏感
        assert op("ICD10_CODE", {"prefixes": ["icd"]}) is True
        assert op("bam_header", {"prefixes": ["BAM"]}) is True
        # 关闭大小写不敏感
        assert op("ICD10", {"prefixes": ["icd"], "case_insensitive": False}) is False
        assert op("icd10", {"prefixes": ["icd"], "case_insensitive": False}) is True

    def test_suffix_matcher_case_insensitive(self):
        """suffix_match 默认大小写不敏感。"""
        op = OperatorRegistry.get("suffix_match")
        # 默认大小写不敏感
        assert op("DATA.VCF", {"suffixes": [".vcf"]}) is True
        assert op("file.BAM", {"suffixes": [".bam"]}) is True
        # 关闭大小写不敏感
        assert op("DATA.VCF", {"suffixes": [".vcf"], "case_insensitive": False}) is False
        assert op("data.vcf", {"suffixes": [".vcf"], "case_insensitive": False}) is True
