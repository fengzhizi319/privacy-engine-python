"""
测试类与字段命名规范及别名兼容性 / Test class & field naming conventions and alias compatibility

本模块验证项目中的命名规范和向后兼容别名是否正常工作：
This module verifies that naming conventions and backward-compatible aliases work correctly:

1. 异常类命名 / Exception class naming:
   - 规范：异常类必须以 Error 或 Exception 结尾
   - Convention: Exception classes must end with Error or Exception
   - 兼容：保留旧名称作为别名 (PrivacyBudgetExhausted → PrivacyBudgetExhaustedError)
   - Compat: Keep old names as aliases

2. 首字母缩写别名 / Acronym aliases:
   - LDP (Local Differential Privacy) / 本地差分隐私
   - QOL (Query Obfuscation Layer) / 查询混淆层
   - KAnonymity (K-Anonymity) / K-匿名
   - NER (Named Entity Recognition) / 命名实体识别
   - LLM (Large Language Model) / 大语言模型

3. 属性别名 / Property aliases:
   - TableClassificationResult.columns → schema_ 的别名
   - 方便用户以直觉方式访问列名 / Convenient column name access

设计决策 / Design Decision:
    项目重构时将类名统一为 PascalCase + 完整拼写，
    但通过别名保持对旧代码的兼容性，避免破坏性变更。
    During refactoring, class names were unified to PascalCase + full spelling,
    but aliases maintain backward compatibility to avoid breaking changes.
"""

# === 异常类导入 / Exception class imports ===
# PrivacyBudgetExhausted 是旧名称（别名）/ Old name (alias)
# PrivacyBudgetExhaustedError 是规范名称 / Canonical name
from engine.privacy.budget import PrivacyBudgetExhausted, PrivacyBudgetExhaustedError

# === REST API Schema 导入 / REST API Schema imports ===
# 每个原语提供多个别名以适应不同编码风格
# Each primitive provides multiple aliases for different coding styles
from engine.schemas import (
    KAnonRequest,              # 简写 / Short form
    KAnonymityRequest,         # 完整拼写别名 / Full spelling alias
    LdpPerturbBinaryRequest,   # PascalCase 风格 / PascalCase style
    LDPPerturbBinaryRequest,   # 全大写缩写别名 / All-caps acronym alias
    QolRequest,                # PascalCase 风格 / PascalCase style
    QOLRequest,                # 全大写缩写别名 / All-caps acronym alias
    QueryObfuscationRequest,   # 完整语义别名 / Full semantic alias
)

# === 动态分类引擎导入 / Dynamic classification engine imports ===
from engine.dynclassification.base import SmallNerEngine, SmallNEREngine, LlmClassifier, LLMClassifier
from engine.dynclassification.service import DynClassificationService, DynamicClassificationService
from engine.dynclassification.models import TableClassificationResult


def test_exception_naming_convention():
    """测试异常类命名及其兼容别名 / Test exception class naming and compatible aliases.

    验证内容 / Verifies:
    - PrivacyBudgetExhaustedError 是 Exception 的子类
    - PrivacyBudgetExhaustedError is a subclass of Exception
    - PrivacyBudgetExhausted 与 PrivacyBudgetExhaustedError 是同一个对象（别名）
    - PrivacyBudgetExhausted is the same object as PrivacyBudgetExhaustedError (alias)
    - 可以正常实例化并捕获 / Can be instantiated and caught normally
    """
    # 验证继承关系 / Verify inheritance
    assert issubclass(PrivacyBudgetExhaustedError, Exception)
    # 验证别名指向同一对象 / Verify alias points to same object
    assert PrivacyBudgetExhausted is PrivacyBudgetExhaustedError

    # 验证实例化 / Verify instantiation
    err = PrivacyBudgetExhaustedError("Budget exhausted")
    assert isinstance(err, Exception)


def test_acronym_schema_aliases():
    """测试首字母缩写词 REST Schema 别名 / Test acronym REST Schema aliases.

    验证内容 / Verifies:
    - LDP 别名: LDPPerturbBinaryRequest is LdpPerturbBinaryRequest
    - QOL 别名: QOLRequest is QolRequest, QueryObfuscationRequest is QolRequest
    - K-Anonymity 别名: KAnonymityRequest is KAnonRequest
    - 别名类可以正常实例化并设置属性 / Aliased classes instantiate correctly
    """
    # 验证所有别名指向同一个类对象 / Verify all aliases point to same class object
    assert LDPPerturbBinaryRequest is LdpPerturbBinaryRequest
    assert QOLRequest is QolRequest
    assert QueryObfuscationRequest is QolRequest
    assert KAnonymityRequest is KAnonRequest

    # 验证通过别名实例化后属性正确 / Verify attributes after instantiation via alias
    req = LDPPerturbBinaryRequest(values=[1, 0, 1], epsilon=0.5)
    assert req.values == [1, 0, 1]
    assert req.epsilon == 0.5


def test_engine_acronym_aliases():
    """测试引擎接口缩写别名 / Test engine interface acronym aliases.

    验证内容 / Verifies:
    - SmallNEREngine is SmallNerEngine (NER 全大写 vs PascalCase)
    - LLMClassifier is LlmClassifier (LLM 全大写 vs PascalCase)
    - DynamicClassificationService is DynClassificationService (完整 vs 缩写)
    """
    assert SmallNEREngine is SmallNerEngine
    assert LLMClassifier is LlmClassifier
    assert DynamicClassificationService is DynClassificationService


def test_table_classification_result_columns_property():
    """测试 TableClassificationResult 的 columns 属性别名 / Test columns property alias.

    验证内容 / Verifies:
    - schema_ 属性存储原始列名列表 / schema_ stores the raw column name list
    - columns 是 schema_ 的只读别名，提供直觉的访问方式
    - columns is a read-only alias for schema_, providing intuitive access
    - 两者返回相同的内容 / Both return the same content
    """
    # 构造分类结果对象 / Construct classification result object
    res = TableClassificationResult(
        schema=["id", "name", "age"],  # 输入列名 / Input column names
        record_results=[],               # 无记录级结果 / No record-level results
        aggregated_tags=[],              # 无聚合标签 / No aggregated tags
        final_level="L1",               # 最终分类等级 / Final classification level
        confidence=1.0,                  # 置信度 / Confidence score
    )
    # 验证内部属性 / Verify internal attribute
    assert res.schema_ == ["id", "name", "age"]
    # 验证别名属性返回相同值 / Verify alias property returns same value
    assert res.columns == ["id", "name", "age"]
