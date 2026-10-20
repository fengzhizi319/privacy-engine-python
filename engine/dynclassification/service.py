"""动态分类分级服务 / Dynamic Classification Service.

提供高层 API 入口，封装 ProfileLoader、ConfigurableRuleEngine、
CompositeRuleEngine 的调用逻辑，支持字段级、记录级和表级分类。

使用示例：
    from engine.dynclassification import DynClassificationService

    service = DynClassificationService(rules_dir="rules")

    # 字段级分类
    result = service.classify_field("phone_number", "13800138000")

    # 指定标准
    result = service.classify_field(
        "bank_card", "6222021234567890123",
        standard="jrt0197"
    )

    # 记录级分类
    record = {"name": "张三", "id_card": "110101199001011237", "phone": "13800138000"}
    result = service.classify_record(record)

===================================================================================
                  服务层执行流程 / Service Layer Execution Flow
===================================================================================

  外部请求 (REST / gRPC / 直接调用)
       │
       ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  DynClassificationService                                            │
  │                                                                      │
  │  classify_field(name, value, domain, standard, sanitize)             │
  │    │                                                                 │
  │    ├─① ProfileLoader.get_engine(domain, standard)                    │
  │    │     → 加载 YAML → 合并规则 → 构建 ConfigurableRuleEngine (带缓存) │
  │    │                                                                 │
  │    ├─② _build_funnel() (double-checked locking)                      │
  │    │     → 获取/懒初始化 NerAdapter + LlmAdapter                      │
  │    │     → 组装 ClassificationFunnel(engine, taxonomy, policy, ner, llm)│
  │    │                                                                 │
  │    ├─③ funnel.classify_field(field_name, value)                      │
  │    │     → L1 规则引擎 → L2 NER → L3 LLM 三层漏斗                     │
  │    │     → 返回 FunnelResult (tags, final_level, confidence, ...)     │
  │    │                                                                 │
  │    ├─④ 包装为 FieldClassificationResult                               │
  │    │     → 填充 sanitized_value (图像打码 / 文本脱敏回调)              │
  │    │     → 最终安全门禁: L4/L5 残留 → "[L4-L5-DATA-REMOVED]"          │
  │    │                                                                 │
  │    └─⑤ 包装为 ClassificationResponse + AuditInfo 返回                 │
  │                                                                      │
  │  classify_record(record, ...)                                         │
  │    → 对每个字段调用 classify_field()                                    │
  │    → CompositeRuleEngine.evaluate() 多字段组合升级                     │
  │    → 聚合为 RecordClassificationResult                                 │
  │                                                                      │
  │  classify_table(records, ...)                                         │
  │    → 对每条记录调用 classify_record()                                   │
  │    → 聚合为 TableClassificationResult                                  │
  │                                                                      │
  │  性能优化 / Performance:                                              │
  │    - fork-after-warmup COW: 主进程预加载模型，子进程共享只读页           │
  │    - LRU 分类结果缓存: 相同 (field_name, value, domain, standard) 命中缓存│
  │    - Funnel 实例缓存: 相同 domain:standard 复用 Funnel                 │
  ══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..observability.metrics import (
    CLASSIFICATION_DURATION,
    CLASSIFICATION_JOBS_DURATION,
    CLASSIFICATION_JOBS_TOTAL,
    CLASSIFICATION_TOTAL,
    observe_duration,
)
from .engine import ConfigurableRuleEngine
from .funnel import ClassificationFunnel
from .llm_adapter import LlmAdapter
from .models import (
    AuditInfo,
    ClassificationResponse,
    ConfidencePolicy,
    FieldClassificationResult,
    RecordClassificationResult,
    SecurityTag,
    TableClassificationResult,
)
from .ner_adapter import NerAdapter
from .operator_registry import OperatorRegistry
from .profile_loader import ProfileLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fork-after-warmup 预加载适配器注册表 / Preloaded adapter registry (COW)
# ---------------------------------------------------------------------------
# 高并发优化：launcher.py 的 --warmup 模式在主进程 fork 前加载 ML 模型，
# 并把加载好的 adapter 实例注册到此处。fork 后子进程（worker）首次需要
# NER/LLM 适配器时优先复用注册表中的实例——模型权重内存页在主进程 fork
# 时已驻留，子进程直接继承（Copy-on-Write 共享只读页），避免 N 份模型
# 内存翻倍，也省去重新读盘 + 重新构建模型的开销。
#
# 注意：注册表是 fork 前填充的，子进程各自持有独立副本，互不影响；
# 适配器内部使用锁保护初始化，fork 时锁未被持有，可安全复用。
_preloaded_adapters: dict[str, Any] = {}
_preloaded_lock = threading.Lock()


def register_preloaded_adapter(kind: str, adapter: Any) -> None:
    """注册一个 fork 前预加载的适配器实例（launcher --warmup 使用）。"""
    with _preloaded_lock:
        _preloaded_adapters[kind] = adapter


def consume_preloaded_adapter(kind: str) -> Any | None:
    """获取 fork 前预加载的适配器实例；未注册时返回 None。"""
    with _preloaded_lock:
        return _preloaded_adapters.get(kind)


class DynClassificationService:
    """动态分类分级服务。

    高层 API 入口，根据请求上下文（domain/standard）动态加载规则，
    执行字段级、记录级和表级分类。

    Attributes:
        loader: Profile 加载器。
    """

    def __init__(
        self,
        rules_dir: str | Path = "rules",
        text_sanitizer: Callable[[str, str, str], str] | None = None,
    ):
        """初始化服务。

        Args:
            rules_dir: 规则配置根目录路径。
            text_sanitizer: 可选的文本脱敏回调函数，签名为
                (field_name, text, final_level) -> sanitized_text。
                用于解耦领域特定脱敏逻辑（如医疗 L4/L5 词库替换），
                避免 dynclassification 反向依赖领域模块。
        """
        # Initialize the profile loader which handles YAML loading, caching,
        # hot-reload detection, and engine instance construction.
        self.loader = ProfileLoader(rules_dir=rules_dir)
        # Lazy-initialized NER adapter (shared across all requests).
        self._ner_adapter: NerAdapter | None = None
        self._service_lock = threading.Lock()
        # Lazy-initialized LLM adapter (shared across all requests).
        self._llm_adapter: LlmAdapter | None = None
        # Funnel instance cache: keyed by engine cache key ("domain:standard").
        self._funnel_cache: dict[str, ClassificationFunnel] = {}
        cache_size = int(os.environ.get("PRIVACY_CLASSIFICATION_CACHE_SIZE", "10000"))
        from ..privacy.high_concurrency import HighConcurrencyLRUCache
        self._classification_cache = HighConcurrencyLRUCache[tuple[str, str, str, str, bool], ClassificationResponse](
            capacity=cache_size
        )
        # 可插拔文本脱敏回调（由领域模块注入，解耦反向依赖）
        self._text_sanitizer = text_sanitizer

    def _is_image_input(self, val_str: str) -> bool:
        """判断输入是否为图像（文件路径或 Base64 Data URI）。"""
        from .image_redaction import is_image_input
        return is_image_input(val_str)

    def _compute_sanitized_value(
        self, field_name: str, val_str: str, field_result: FieldClassificationResult
    ) -> FieldClassificationResult:
        """计算 sanitized_value 并返回新的 FieldClassificationResult 副本（线程安全）。

        优先使用漏斗已计算的 sanitized_value，其次使用图像打码，最后使用可插拔文本脱敏回调。
        """
        # 优先使用漏斗已计算的 sanitized_value（如场景 C 图像自动触发 LLM）
        if getattr(field_result, "sanitized_value", None):
            return field_result

        # 创建副本以避免修改缓存对象
        new_result = field_result.model_copy(deep=True)

        if self._is_image_input(val_str):
            from .image_redaction import sanitize_image_input
            new_result.sanitized_value = sanitize_image_input(val_str)
        elif self._text_sanitizer is not None:
            # 使用领域模块注入的文本脱敏回调（解耦反向依赖）
            new_result.sanitized_value = self._text_sanitizer(
                field_name, val_str, field_result.final_level
            )
        else:
            # 向后兼容：尝试使用医疗领域规则进行文本脱敏
            new_result.sanitized_value = self._fallback_text_sanitizer(
                field_name, val_str, field_result.final_level
            )

        # 最终安全门禁：任何脱敏后仍残留 L4/L5 高敏文本的情况，整值替换为 [L4-L5-DATA-REMOVED]
        try:
            from ..medical_pipeline.rules import contains_high_risk_text
            if contains_high_risk_text(new_result.sanitized_value):
                new_result.sanitized_value = "[L4-L5-DATA-REMOVED]"
        except Exception:
            pass

        return new_result

    def _get_sanitized_value_from_funnel(
        self,
        field_name: str,
        val_str: str,
        funnel_result: Any,
        final_level: str,
    ) -> str:
        """从漏斗结果或文本脱敏回调中获取 sanitized_value。"""
        # 图像输入：调用图像打码
        if self._is_image_input(val_str):
            from .image_redaction import sanitize_image_input
            return sanitize_image_input(val_str)

        # 1. 优先使用实例上注入的可插拔文本脱敏回调
        if self._text_sanitizer is not None:
            sanitized = self._text_sanitizer(field_name, val_str, final_level)
        elif getattr(funnel_result, "sanitized_value", None):
            sanitized = funnel_result.sanitized_value
        else:
            # 2. 其次查询领域注册表 (DomainStrategyRegistry) 中注册的回调
            from .domain_registry import default_domain_registry
            reg_sanitizer = default_domain_registry.get_sanitizer("medical")
            if reg_sanitizer is not None:
                sanitized = reg_sanitizer(field_name, val_str, final_level, "redact")
            else:
                # 向后兼容：尝试使用医疗领域规则进行文本脱敏
                sanitized = self._fallback_text_sanitizer(field_name, val_str, final_level)

        # 最终安全门禁：任何脱敏后仍残留 L4/L5 高敏文本的情况，整值替换为 [L4-L5-DATA-REMOVED]
        try:
            from ..medical_pipeline.rules import contains_high_risk_text
            if contains_high_risk_text(sanitized):
                return "[L4-L5-DATA-REMOVED]"
        except Exception:
            pass

        return sanitized

    def _fallback_text_sanitizer(
        self, field_name: str, text: str, final_level: str
    ) -> str:
        """向后兼容的文本脱敏器：尝试使用医疗领域规则。

        fail-closed 语义：脱敏流程发生任何异常（医疗模块不可用、
        规则执行失败等）时返回固定失败占位符，绝不回退返回原文，
        与 DomainStrategyRegistry 路径的 fail-closed 语义对齐。
        """
        try:
            from ..medical_pipeline.rules import L4_PATTERNS, L5_PATTERNS
            from ..privacy.masking import mask_value

            # 先应用 L5 替换，再应用 L4 替换
            s_text = text
            for pat, replacement in L5_PATTERNS:
                s_text = pat.sub(replacement, s_text)
            for pat, replacement in L4_PATTERNS:
                s_text = pat.sub(replacement, s_text)

            # 对中高敏感度字段应用 PII 掩码
            if final_level in ["L3", "L4", "L5", "C4", "C5"]:
                s_text = mask_value(field_name, s_text)

            # 最终门禁检查：防止漏网高敏词
            from ..medical_pipeline.rules import contains_high_risk_text
            if contains_high_risk_text(s_text):
                return "[L4-L5-DATA-REMOVED]"

            return s_text
        except Exception:
            # fail-closed：脱敏失败绝不泄露原文（如身份证号/病历原文）
            logger.warning(
                "text_sanitization_failed",
                extra={"field_name": field_name, "final_level": final_level},
            )
            return "[REDACTION-FAILED]"

    def clear_cache(self) -> None:
        """清空分类缓存（规则重载或配置更新时调用）。"""
        self._classification_cache.clear()

    # ------------------------------------------------------------------
    # 字段级分类 / Field-level Classification
    # ------------------------------------------------------------------

    def classify_field(
        self,
        field_name: str,
        value: Any,
        domain: Optional[str] = None,
        standard: Optional[str] = None,
        sanitize: bool = False,
    ) -> ClassificationResponse:
        """对单个字段进行分类与智能抹平脱敏。"""
        start = time.perf_counter()
        status = "success"
        response: ClassificationResponse | None = None
        try:
            with observe_duration(CLASSIFICATION_DURATION, operation="field"):
                # Step 0: 高并发 LRU 缓存查找（防超长 Base64 内存暴涨）
                val_str_raw = str(value) if value is not None else ""
                val_key = val_str_raw if len(val_str_raw) <= 200 else (len(val_str_raw), val_str_raw[:100], hash(val_str_raw))
                cache_key = (domain or "", standard or "", field_name, val_key, sanitize)
                cached_resp = self._classification_cache.get(cache_key)
                if cached_resp is not None:
                    field_result = cached_resp.field_result.model_copy(deep=True)
                    if sanitize and field_result.sanitized_value is None:
                        val_str = str(value) if value is not None else ""
                        field_result = self._compute_sanitized_value(field_name, val_str, field_result)

                    duration_ms = (time.perf_counter() - start) * 1000
                    audit = AuditInfo(
                        domain=cached_resp.audit_info.domain,
                        standard_id=cached_resp.audit_info.standard_id,
                        rule_set_version=cached_resp.audit_info.rule_set_version,
                        rules_evaluated=cached_resp.audit_info.rules_evaluated,
                        rules_hit=cached_resp.audit_info.rules_hit,
                        duration_ms=round(duration_ms, 3),
                    )
                    response = ClassificationResponse(field_result=field_result, audit_info=audit)
                    if field_result.final_level:
                        CLASSIFICATION_TOTAL.labels(
                            final_level=field_result.final_level,
                            layer=field_result.engine_layer or "unknown",
                        ).inc()
                else:
                    # Step 1: Get or construct the rule engine (cached by domain:standard key).
                    engine = self.loader.get_engine(domain=domain, standard=standard)

                    # Step 2: Build and execute the 3-layer funnel.
                    funnel = self._build_funnel(engine)
                    funnel_result, suppressed_tags = funnel.classify_field(field_name, value, sanitize=sanitize)
                    tags = funnel_result.tags

                    # Step 3: Use funnel result directly (level, confidence, layer already resolved).
                    final_level = funnel_result.final_level

                    # Compute smart sanitized_value if requested
                    sanitized_val: str | None = None
                    if sanitize:
                        val_str = str(value) if value is not None else ""
                        sanitized_val = self._get_sanitized_value_from_funnel(field_name, val_str, funnel_result, final_level)

                    duration_ms = (time.perf_counter() - start) * 1000

                    # Step 5: Build the field-level classification result.
                    field_result = FieldClassificationResult(
                        field_name=field_name,
                        field_value=str(value) if value is not None else None,
                        tags=tags,
                        final_level=final_level,
                        confidence=funnel_result.confidence,
                        needs_human_review=funnel_result.needs_human_review,
                        engine_layer=funnel_result.engine_layer,
                        reasoning=funnel_result.reasoning,
                        sanitized_value=sanitized_val,
                        suppressed_tags=suppressed_tags,
                    )

                    audit = AuditInfo(
                        domain=engine.domain,
                        standard_id=engine.standard_id,
                        rule_set_version=engine.taxonomy.version,
                        rules_evaluated=engine.rule_count,
                        rules_hit=len(tags),
                        duration_ms=round(duration_ms, 3),
                    )

                    response = ClassificationResponse(field_result=field_result, audit_info=audit)
                    self._classification_cache.put(cache_key, response)
                    if field_result.final_level:
                        CLASSIFICATION_TOTAL.labels(
                            final_level=field_result.final_level,
                            layer=field_result.engine_layer or "unknown",
                        ).inc()
        except Exception:
            status = "error"
            CLASSIFICATION_JOBS_TOTAL.labels(status="error").inc()
            raise
        finally:
            CLASSIFICATION_JOBS_DURATION.labels(status=status).observe(time.perf_counter() - start)
        CLASSIFICATION_JOBS_TOTAL.labels(status="success").inc()
        return response


    # ------------------------------------------------------------------
    # 记录级分类 / Record-level Classification
    # ------------------------------------------------------------------

    def classify_record(
        self,
        record: dict[str, Any],
        record_index: int = 0,
        domain: Optional[str] = None,
        standard: Optional[str] = None,
        sanitize: bool = False,
    ) -> ClassificationResponse:
        """对单条记录（多字段）进行分类与智能抹平脱敏。"""
        start = time.perf_counter()
        status = "success"
        response: ClassificationResponse | None = None
        try:
            with observe_duration(CLASSIFICATION_DURATION, operation="record"):
                field_results: dict[str, FieldClassificationResult] = {}
                all_tags: list[SecurityTag] = []
                for field_name, value in record.items():
                    resp = self.classify_field(field_name, value, domain=domain, standard=standard, sanitize=sanitize)
                    if resp.field_result:
                        field_results[field_name] = resp.field_result
                        all_tags.extend(resp.field_result.tags)

                engine = self.loader.get_engine(domain=domain, standard=standard)
                composite_engine = self.loader.get_composite_engine(domain=domain, standard=standard)

                composite_tags = composite_engine.evaluate(record, field_results)
                all_tags.extend(composite_tags)

                # 基准记录等级：从字段裁定等级集合中解析最高等级（保持字段级降级/仲裁裁定有效性）
                field_levels = [fr.final_level for fr in field_results.values() if fr.final_level]
                base_record_level = (
                    engine.taxonomy.max_level(*field_levels)
                    if field_levels
                    else engine.taxonomy.default_level
                )
                record_level = composite_engine.apply_to_record_level(
                    base_record_level, composite_tags, engine.taxonomy
                )

                duration_ms = (time.perf_counter() - start) * 1000

                record_result = RecordClassificationResult(
                    record_index=record_index,
                    field_results=field_results,
                    aggregated_tags=all_tags,
                    final_level=record_level,
                    confidence=max((fr.confidence for fr in field_results.values()), default=0.0),
                    needs_human_review=any(fr.needs_human_review for fr in field_results.values()),
                )

                audit = AuditInfo(
                    domain=engine.domain,
                    standard_id=engine.standard_id,
                    rule_set_version=engine.taxonomy.version,
                    rules_evaluated=engine.rule_count * len(record),
                    rules_hit=len(all_tags),
                    duration_ms=round(duration_ms, 3),
                )

                response = ClassificationResponse(record_result=record_result, audit_info=audit)
                for fr in field_results.values():
                    if fr.final_level:
                        CLASSIFICATION_TOTAL.labels(
                            final_level=fr.final_level,
                            layer=fr.engine_layer or "unknown",
                        ).inc()
        except Exception:
            status = "error"
            CLASSIFICATION_JOBS_TOTAL.labels(status="error").inc()
            raise
        finally:
            CLASSIFICATION_JOBS_DURATION.labels(status=status).observe(time.perf_counter() - start)
        CLASSIFICATION_JOBS_TOTAL.labels(status="success").inc()
        return response

    # ------------------------------------------------------------------
    # 表级分类 / Table-level Classification
    # ------------------------------------------------------------------

    def classify_table(
        self,
        schema: list[str],
        rows: list[dict[str, Any]],
        domain: Optional[str] = None,
        standard: Optional[str] = None,
    ) -> ClassificationResponse:
        """对整张表进行分类。"""
        start = time.perf_counter()
        status = "success"
        response: ClassificationResponse | None = None
        try:
            with observe_duration(CLASSIFICATION_DURATION, operation="table"):
                record_results: list[RecordClassificationResult] = []
                all_tags: list[SecurityTag] = []

                for idx, row in enumerate(rows):
                    resp = self.classify_record(row, record_index=idx, domain=domain, standard=standard)
                    if resp.record_result:
                        record_results.append(resp.record_result)
                        all_tags.extend(resp.record_result.aggregated_tags)

                # Determine table-level final level (highest across all records).
                engine = self.loader.get_engine(domain=domain, standard=standard)
                rec_levels = [r.final_level for r in record_results if r.final_level]
                table_level = (
                    engine.taxonomy.max_level(*rec_levels)
                    if rec_levels
                    else engine.taxonomy.default_level
                )

                duration_ms = (time.perf_counter() - start) * 1000

                table_result = TableClassificationResult(
                    schema_=schema,
                    record_results=record_results,
                    aggregated_tags=all_tags,
                    final_level=table_level,
                    confidence=1.0 if all_tags else 0.0,
                )

                audit = AuditInfo(
                    domain=engine.domain,
                    standard_id=engine.standard_id,
                    rule_set_version=engine.taxonomy.version,
                    rules_evaluated=engine.rule_count * len(rows) * max(len(schema), 1),
                    rules_hit=len(all_tags),
                    duration_ms=round(duration_ms, 3),
                )

                response = ClassificationResponse(table_result=table_result, audit_info=audit)
                for rec in record_results:
                    for fr in rec.field_results.values():
                        if fr.final_level:
                            CLASSIFICATION_TOTAL.labels(
                                final_level=fr.final_level,
                                layer=fr.engine_layer or "unknown",
                            ).inc()
        except Exception:
            status = "error"
            CLASSIFICATION_JOBS_TOTAL.labels(status="error").inc()
            raise
        finally:
            CLASSIFICATION_JOBS_DURATION.labels(status=status).observe(time.perf_counter() - start)
        CLASSIFICATION_JOBS_TOTAL.labels(status="success").inc()
        return response

    # ------------------------------------------------------------------
    # Dry-Run 预演 / Dry-Run Preview
    # ------------------------------------------------------------------

    def dry_run(
        self,
        sample_data: list[dict[str, Any]],
        domain: Optional[str] = None,
        standard: Optional[str] = None,
    ) -> dict[str, Any]:
        """对样本数据集执行规则预演，返回命中分布与等级分布。

        用于在规则正式生效前验证其命中行为是否符合预期，
        避免误配规则造成生产数据过度脱敏或漏判。

        Args:
            sample_data: 样本记录列表，每条为 {field_name: value} 字典。
            domain: 领域标识。
            standard: 标准标识。

        Returns:
            包含 level_distribution、category_distribution、hit_details、
            summary 等统计信息的字典。
        """
        # Counter for tallying level and category distributions.
        from collections import Counter

        level_counter: Counter = Counter()       # level_id -> hit count
        category_counter: Counter = Counter()    # category_id -> hit count
        hit_details: list[dict[str, Any]] = []   # Detailed per-field hit records
        total_fields = 0                          # Total fields evaluated
        total_hits = 0                            # Fields that produced at least one tag

        # Iterate all sample records and classify each field.
        for row_idx, record in enumerate(sample_data):
            for field_name, value in record.items():
                total_fields += 1
                # Run field-level classification.
                resp = self.classify_field(
                    field_name=field_name,
                    value=value,
                    domain=domain,
                    standard=standard,
                )
                # If the field produced tags, record statistics.
                if resp.field_result and resp.field_result.tags:
                    total_hits += 1
                    level_counter[resp.field_result.final_level] += 1
                    for tag in resp.field_result.tags:
                        category_counter[tag.category] += 1
                    # Store hit detail (truncate value for safety).
                    hit_details.append({
                        "row": row_idx,
                        "field_name": field_name,
                        "value": str(value)[:100],  # Truncate to prevent log bloat
                        "level": resp.field_result.final_level,
                        "rules": [t.rule_id for t in resp.field_result.tags],
                    })

        # Fields that did NOT hit any rule are counted under the default level.
        engine = self.loader.get_engine(domain=domain, standard=standard)
        miss_count = total_fields - total_hits
        if miss_count > 0:
            level_counter[engine.taxonomy.default_level] += miss_count

        # Assemble the dry-run report.
        return {
            "summary": {
                "total_records": len(sample_data),
                "total_fields": total_fields,
                "total_hits": total_hits,
                "hit_rate": round(total_hits / max(total_fields, 1), 4),
                "domain": engine.domain,
                "standard_id": engine.standard_id,
                "rules_evaluated": engine.rule_count,
            },
            "level_distribution": dict(level_counter.most_common()),
            "category_distribution": dict(category_counter.most_common()),
            "hit_details": hit_details[:200],  # Cap at 200 entries to limit response size
        }

    # ------------------------------------------------------------------
    # 管理接口 / Management APIs
    # ------------------------------------------------------------------

    def list_standards(self) -> list[str]:
        """列出所有可用标准（扫描 standards/ 目录）。"""
        return self.loader.list_standards()

    def list_standards_detail(self) -> list[dict[str, Any]]:
        """列出所有可用标准的详细信息（含等级体系），供前端标准切换器渲染。

        每个标准返回：standard_id、description、taxonomy、domains、
        default_level、按 rank 升序排列的等级列表（levels）、
        该标准组合下的规则总数（rule_count）以及引用 taxonomy 下的
        分类总数（category_count）。
        配置损坏的标准会被跳过（仅记录警告），避免单个坏文件拖垮整个列表。

        Returns:
            标准详情字典列表（按 standard_id 排序，保证前端顺序稳定）。
        """
        details: list[dict[str, Any]] = []
        for sid in sorted(self.list_standards()):
            try:
                std_def = self.loader.load_standard(sid)
                taxonomy = self.loader.load_taxonomy(std_def.taxonomy)
            except Exception as exc:  # noqa: BLE001
                # 单个标准配置损坏不应影响其余标准的可用性。
                logger.warning(
                    "Skip broken standard '%s' in list_standards_detail: %s", sid, exc
                )
                continue
            # 统计该标准组合下所有领域包的规则总数（普通 + 降级 + 复合）。
            # Count total rules (normal + downgrade + composite) across the standard's domain packs.
            # 单个领域包损坏仅跳过计数，不影响标准本身的可用性。
            rule_count = 0
            for domain in std_def.domains:
                try:
                    profile = self.loader.load_profile(domain)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Skip broken domain '%s' while counting rules for standard '%s': %s",
                        domain, sid, exc,
                    )
                    continue
                rule_count += (
                    len(profile.rules)
                    + len(profile.downgrade_rules)
                    + len(profile.composite_rules)
                )
            details.append(
                {
                    "standard_id": sid,
                    "description": std_def.description,
                    "taxonomy": std_def.taxonomy,
                    "domains": list(std_def.domains),
                    "default_level": taxonomy.default_level,
                    "levels": [
                        {"id": lv.id, "name": lv.name, "rank": lv.rank}
                        for lv in sorted(taxonomy.levels.values(), key=lambda x: x.rank)
                    ],
                    "rule_count": rule_count,
                    # taxonomy 下定义的分类总数，供前端展示标准分类体系规模。
                    # Total categories defined in the taxonomy, for showing the standard's category scope.
                    "category_count": len(taxonomy.categories),
                }
            )
        return details

    def list_domains(self) -> list[str]:
        """列出所有可用领域包（扫描 domains/ 目录）。"""
        return self.loader.list_domains()

    def list_operators(self) -> list[str]:
        """列出所有已注册算子（从 OperatorRegistry 查询）。"""
        return OperatorRegistry.list_operators()

    def check_and_reload(self, force: bool = False) -> bool:
        """检查底层配置文件的变动。若有变动，同步清空 funnel 缓存与 loader 缓存。"""
        reloaded = self.loader.check_and_reload(force=force)
        if reloaded:
            with self._service_lock:
                self._funnel_cache.clear()
                self._classification_cache.clear()
                self._ner_adapter = None
                self._llm_adapter = None
            logger.info("DynClassificationService funnel and adapter cache cleared due to hot-reload.")
        return reloaded

    def reload(self) -> None:
        """热加载：清除缓存，下次请求时重新加载配置。

        同时重置 NER/LLM 适配器单例，确保 taxonomy 中的
        模型路径、标签映射、prompt 模板等配置变更后能生效。
        """
        self.loader.invalidate_cache()
        # 清除 funnel 与分类结果缓存（引擎重载后相关缓存也需清空）
        with self._service_lock:
            self._funnel_cache.clear()
            self._classification_cache.clear()
            self._ner_adapter = None
            self._llm_adapter = None

    def generate_profile_from_doc(self, doc_path: str | Path) -> dict[str, str]:
        """从标准 Markdown 文档自动抽取并生成 YAML 配置文件，并重新载入引擎缓存。

        Workflow:
        1. Parse the Markdown document using StandardProfileGenerator.
        2. Generate taxonomy/domain/standard YAML files into rules_dir.
        3. Invalidate cache so next request picks up the new configuration.

        Args:
            doc_path: 标准文档路径（如 'docs/standard/四川省健康医疗大数据应用指南.md'）。

        Returns:
            生成的 3 个 YAML 文件路径字典。
        """
        # Lazy import to avoid loading generator module unless needed.
        from .standard_profile_generator import StandardProfileGenerator
        parser = StandardProfileGenerator(doc_path)
        # Generate the 3 YAML files (taxonomy, domain profile, standard definition).
        generated = parser.generate_files(self.loader.rules_dir)
        # Invalidate cache to force reload on next classification request.
        self.reload()
        return {k: str(v) for k, v in generated.items()}


    # ------------------------------------------------------------------
    # 内部方法 / Internal Methods
    # ------------------------------------------------------------------

    def _resolve_final_level(self, tags: list[SecurityTag], engine: ConfigurableRuleEngine) -> str:
        """从标签列表中解析最终等级（取最高）。

        Logic: If no tags hit, return the taxonomy's default level.
        Otherwise, use taxonomy.max_level() to find the highest-rank level.
        """
        # No tags means no rules matched -> use configured default level.
        if not tags:
            return engine.taxonomy.default_level
        # Extract all level IDs from tags and find the maximum by rank.
        levels = [tag.level for tag in tags]
        return engine.taxonomy.max_level(*levels)

    def _build_funnel(self, engine: ConfigurableRuleEngine) -> ClassificationFunnel:
        """构建三层漏斗编排器（带缓存，线程安全）。

        根据引擎的 taxonomy 中配置的 confidence_policy 构建漏斗，
        并按需初始化 NER/LLM 适配器（lazy-load，全局单例）。
        Funnel 实例按 engine 的 domain:standard 键缓存，避免高频创建。

        并发保护：double-checked locking——无锁快速路径命中缓存；
        未命中时持锁构建，避免并发首请求重复构建适配器/重复加载模型。

        Args:
            engine: 当前请求对应的规则引擎实例。

        Returns:
            配置完成的三层漏斗编排器。
        """
        # 缓存键：与 ProfileLoader 的 engine 缓存键一致
        cache_key = f"{engine.domain}:{engine.standard_id}"
        cached = self._funnel_cache.get(cache_key)
        if cached is not None:
            return cached

        with self._service_lock:
            # Double-check：持锁后再次确认，防止并发重复构建
            cached = self._funnel_cache.get(cache_key)
            if cached is not None:
                return cached

            # 从 taxonomy 获取置信度策略配置
            policy = self._get_confidence_policy(engine)

            # 按需初始化 NER 适配器（全局单例，避免重复加载模型）
            ner_adapter = None
            if policy.enable_ner:
                if self._ner_adapter is None:
                    taxonomy = engine.taxonomy
                    # fork-after-warmup：优先复用主进程预加载的适配器（COW 共享模型页）
                    preloaded = consume_preloaded_adapter("ner")
                    if preloaded is not None and preloaded._model_path == taxonomy.ner_model_path:
                        self._ner_adapter = preloaded
                        logger.info("ner_adapter_reused_preloaded")
                    else:
                        self._ner_adapter = NerAdapter(
                            model_path=taxonomy.ner_model_path,
                            vocab_path=taxonomy.ner_vocab_path,
                            label_mapping=taxonomy.ner_label_mapping,
                        )
                ner_adapter = self._ner_adapter

            # 按需初始化 LLM 适配器（全局单例）
            llm_adapter = None
            if policy.enable_llm or policy.enable_llm_arbitration:
                if self._llm_adapter is None:
                    # fork-after-warmup：优先复用主进程预加载的适配器（COW 共享模型页）
                    preloaded = consume_preloaded_adapter("llm")
                    if preloaded is not None and preloaded._model_path == engine.taxonomy.llm_model_path:
                        self._llm_adapter = preloaded
                        logger.info("llm_adapter_reused_preloaded")
                    else:
                        self._llm_adapter = LlmAdapter(
                            model_path=engine.taxonomy.llm_model_path,
                            classify_prompt_template=engine.taxonomy.llm_classify_prompt_template,
                        )
                llm_adapter = self._llm_adapter

            funnel = ClassificationFunnel(
                engine=engine,
                taxonomy=engine.taxonomy,
                confidence_policy=policy,
                ner_adapter=ner_adapter,
                llm_adapter=llm_adapter,
            )
            self._funnel_cache[cache_key] = funnel
            return funnel

    def _get_confidence_policy(self, engine: ConfigurableRuleEngine) -> ConfidencePolicy:
        """从引擎的 taxonomy 中提取置信度策略配置。

        优先使用 DomainTaxonomy.confidence_policy 显式字段；
        若为 None，回退到 model_extra 中的 confidence_policy dict（向后兼容）。

        Args:
            engine: 规则引擎实例。

        Returns:
            ConfidencePolicy 实例。
        """
        taxonomy = engine.taxonomy
        # 优先使用显式字段（Pydantic 已自动校验）
        if taxonomy.confidence_policy is not None:
            return taxonomy.confidence_policy
        # 向后兼容：从 model_extra 中获取 dict 并手动构造
        if hasattr(taxonomy, "model_extra") and taxonomy.model_extra:
            policy_data = taxonomy.model_extra.get("confidence_policy")
            if policy_data and isinstance(policy_data, dict):
                return ConfidencePolicy(**policy_data)
        return ConfidencePolicy()


# Standardized class alias following full-name naming conventions
DynamicClassificationService = DynClassificationService