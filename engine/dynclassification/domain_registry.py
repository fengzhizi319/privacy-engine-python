"""领域策略与脱敏回调注册表模块 / Domain Strategy & Sanitizer Registry.

本模块实现通用动态分类引擎 (dynclassification) 与特异性领域规则 (medical / finance / hr 等) 之间的解耦。
遵循【依赖倒置原则 (DIP)】与【策略模式 (Strategy Pattern)】：
1. 核心内核 (dynclassification) 保持纯净、领域无关；
2. 领域模块 (如 medical_pipeline/rules.py) 作为 Provider 动态注册其特定规则、句法自愈逻辑与词库；
3. 支持通过 domain 参数自动调度对应领域的脱敏回调函数。

===================================================================================
              依赖倒置解耦流程 / DIP Decoupling Flow
===================================================================================

  领域模块启动时 (Provider)               分类服务运行时 (Consumer)
       │                                         │
       ▼                                         ▼
  medical_pipeline/rules.py               service._compute_sanitized_value()
    │                                         │
    │ register_sanitizer("medical", fn)       │ get_sanitizer("medical")
    ▼                                         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  DomainStrategyRegistry (全局单例)                             │
  │                                                                 │
  │  _sanitizers = {                                                │
  │    "medical":  medical_sanitize_text,                           │
  │    "finance":  finance_sanitize_text,                           │
  │    ...                                                          │
  │  }                                                              │
  └─────────────────────────────────────────────────────────────────┘

  调用链 / Call chain:
  service → funnel → L1/L2/L3 分类 → 获取 final_level
    → service._compute_sanitized_value(field_name, text, result)
      → registry.get_sanitizer(domain)(field_name, text, final_level, "redact")
      → 最终安全门禁: L4/L5 残留 → "[L4-L5-DATA-REMOVED]"
===================================================================================
"""

from __future__ import annotations

import threading
from typing import Callable, Protocol


# 领域文本脱敏回调函数签名：(field_name, text, final_level, mode) -> sanitized_text
TextSanitizerCallback = Callable[[str, str, str, str], str]


class DomainStrategyProvider(Protocol):
    """领域策略提供者协议接口 / Domain Strategy Protocol Interface."""

    domain_name: str

    def sanitize_text(
        self, field_name: str, text: str, final_level: str, mode: str = "redact"
    ) -> str:
        """执行领域特异性文本无痕抹平或标签掩码。"""
        ...


class DomainStrategyRegistry:
    """领域策略与回调注册表（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sanitizers: dict[str, TextSanitizerCallback] = {}

    def register_sanitizer(self, domain: str, sanitizer: TextSanitizerCallback) -> None:
        """注册指定领域的文本脱敏回调函数（如 domain='medical'）。"""
        if not domain or not callable(sanitizer):
            raise ValueError("Domain name must be non-empty and sanitizer must be callable")
        with self._lock:
            self._sanitizers[domain.lower().strip()] = sanitizer

    def unregister_sanitizer(self, domain: str) -> bool:
        """解绑指定领域的文本脱敏回调。"""
        with self._lock:
            return self._sanitizers.pop(domain.lower().strip(), None) is not None

    def get_sanitizer(self, domain: str) -> TextSanitizerCallback | None:
        """获取指定领域的文本脱敏回调函数。"""
        if not domain:
            return None
        with self._lock:
            return self._sanitizers.get(domain.lower().strip())

    def clear(self) -> None:
        """清空注册表。"""
        with self._lock:
            self._sanitizers.clear()


# 全局共享领域策略注册表单例
default_domain_registry = DomainStrategyRegistry()
