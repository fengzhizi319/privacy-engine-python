"""动态分类分级模块 / Dynamic Classification Module.

提供声明式、可配置的数据分类分级引擎，支持多领域、多行业标准。 / Provides a declarative, configurable data classification engine supporting multi-domain and multi-industry standards.
核心设计思想：标准配置化、规则声明化、算子插件化、执行上下文动态化。 / Core design philosophy: configurable standards, declarative rules, pluggable operators, dynamic execution contexts.

Usage / 用法:
    from engine.dynclassification import DynClassificationService

    service = DynClassificationService(rules_dir="rules")
    result = service.classify_field("phone_number", "13800138000")
"""

# --- 导入复合规则引擎用于记录级多字段组合评估 / Import composite rule engine for record-level multi-field combination evaluation ---
from .composite import CompositeRuleEngine
# --- 导入解析声明式 RuleProfile 的核心可配置规则引擎 / Import the core configurable rule engine that interprets declarative RuleProfile ---
from .engine import ConfigurableRuleEngine
# --- 导入三层漏斗编排器 (规则 → NER → LLM) / Import the 3-layer funnel orchestrator (Rule → NER → LLM) ---
from .funnel import ClassificationFunnel, FunnelResult
# --- 导入用于从 Markdown 自动生成 YAML 配置的标准文档解析器 / Import the standard profile generator for auto-generating YAML configs from Markdown ---
from .standard_profile_generator import StandardProfileGenerator
# --- 导入用于第三层深度分类和仲裁的 LLM 适配器 / Import LLM adapter for Layer-3 deep classification and arbitration ---
from .llm_adapter import LlmAdapter
# --- 导入模块中使用的所有 Pydantic 数据模型 / Import all Pydantic data models used across the module ---
from .models import (
    AuditInfo,                     # 分类请求的审计元数据 / Audit metadata for classification requests
    CategoryDef,                   # 动态分类类别定义（替代硬编码枚举） / Dynamic category definition (replaces hardcoded enum)
    ClassificationResponse,        # 所有分类 API 的顶层响应包装器 / Top-level response wrapper for all classification APIs
    ConfidencePolicy,              # 置信度衰减和 LLM 仲裁策略 / Confidence decay and LLM arbitration policy
    DomainTaxonomy,                # 完整分类体系定义（等级 + 类别） / Full taxonomy definition (levels + categories)
    EngineLayer,                   # 引擎层级常量 (L1_RULE/L2_SMALL_NER/L3_LLM) / Engine layer constants
    FieldClassificationResult,     # 单字段分类输出 / Single field classification output
    RecordClassificationResult,    # 多字段记录分类输出 / Multi-field record classification output
    SecurityTag,                   # 每个规则命中产生的原子安全标签 / Atomic security tag produced by each rule hit
    SensitivityLevelDef,           # 动态敏感度等级定义 / Dynamic sensitivity level definition
    TableClassificationResult,     # 表/批次级分类输出 / Table/batch-level classification output
)
# --- 导入用于第二层实体提取的 NER 适配器 / Import NER adapter for Layer-2 entity extraction ---
from .ner_adapter import NerAdapter
# --- 导入用于插件式匹配器管理的算子注册表和协议 / Import operator registry and protocol for plugin-style matcher management ---
from .operator_registry import MatcherOperator, OperatorRegistry, OperatorResult, normalize_result
# --- 导入负责 YAML 加载、缓存和热重载的 profile 加载器 / Import profile loader responsible for YAML loading, caching and hot-reload ---
from .profile_loader import ProfileLoader
# --- 导入声明式规则模式模型（匹配器、规则、profile、标准） / Import declarative rule schema models (matchers, rules, profiles, standards) ---
from .rule_schema import (
    CompositeRuleDef,              # 记录级复合规则定义 / Record-level composite rule definition
    DowngradeRuleDef,              # 降级规则定义（降低敏感度） / Downgrade rule definition (lower sensitivity)
    MatcherDef,                    # 单个匹配器定义（算子 + 参数） / Single matcher definition (operator + params)
    RuleDef,                       # 单个分类规则定义 / Single classification rule definition
    RuleProfile,                   # 领域规则包（规则集合） / Domain rule pack (collection of rules)
    StandardDef,                   # 标准组合定义（跨领域） / Standard combination definition (multi-domain)
)
# --- 导入高层服务入口点 / Import the high-level service entry point ---
from .service import DynClassificationService
from .domain_registry import DomainStrategyRegistry, TextSanitizerCallback, default_domain_registry

# --- 公共 API 表面：控制通过 `from dynclassification import *` 导出的内容 / Public API surface: controls what is exported via `from dynclassification import *` ---
__all__ = [
    # Service entry point
    "DynClassificationService",
    # Domain Strategy Registry
    "DomainStrategyRegistry",
    "default_domain_registry",
    "TextSanitizerCallback",
    # Document-to-YAML generator
    "StandardProfileGenerator",
    # Rule engines
    "ConfigurableRuleEngine",
    "CompositeRuleEngine",
    # 3-layer funnel
    "ClassificationFunnel",
    "FunnelResult",
    # ML adapters (Layer-2 NER, Layer-3 LLM)
    "NerAdapter",
    "LlmAdapter",
    # Configuration loader
    "ProfileLoader",
    # Operator plugin registry
    "OperatorRegistry",
    "MatcherOperator",
    "OperatorResult",
    "normalize_result",
    # Data models
    "DomainTaxonomy",
    "SensitivityLevelDef",
    "CategoryDef",
    "SecurityTag",
    "FieldClassificationResult",
    "RecordClassificationResult",
    "TableClassificationResult",
    "AuditInfo",
    "ClassificationResponse",
    "ConfidencePolicy",
    "EngineLayer",
    # Rule schema models",
    "MatcherDef",
    "RuleDef",
    "DowngradeRuleDef",
    "CompositeRuleDef",
    "RuleProfile",
    "StandardDef",
]
