"""算子注册表 / Operator Registry.

提供匹配算子的统一注册、查找和管理机制。
Provides a unified registration, lookup, and management mechanism for matching operators.

所有算子必须实现 MatcherOperator 协议签名：(value, params) -> bool | OperatorResult。
All operators must implement the MatcherOperator protocol signature: (value, params) -> bool | OperatorResult.

支持两种注册方式 / Supports two registration methods:
1. 装饰器注册 / Decorator registration: @OperatorRegistry.register("operator_name")
2. 运行时动态注册 / Runtime dynamic registration: OperatorRegistry.register_func("operator_name", func)
"""

from __future__ import annotations

# threading.Lock is used to protect write paths (register/clear) from race conditions.
# 使用 threading.Lock 保护写入路径（注册/清除）免受竞争条件影响。
import threading
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Union, runtime_checkable


# ===========================================================================
# 算子统一返回类型 / Unified Operator Result
# ===========================================================================


@dataclass(slots=True)
class OperatorResult:
    """算子统一返回结果 / Unified Operator Result.

    所有算子可返回 bool（简单命中/未命中）或 OperatorResult（携带动态等级/类别）。
    All operators can return bool (simple hit/miss) or OperatorResult (carrying dynamic level/category).
    引擎通过 normalize_result() 统一处理两种返回类型。
    The engine processes both return types uniformly via normalize_result().

    Attributes:
        hit: 是否命中 / Whether it's a hit.
        level: 动态等级（None 时使用规则定义的 level） / Dynamic level (uses rule-defined level if None).
        category: 动态类别（None 时使用规则定义的 category） / Dynamic category (uses rule-defined category if None).
    """

    hit: bool
    level: str | None = None
    category: str | None = None


def normalize_result(raw: Any) -> OperatorResult:
    """将算子原始返回值归一化为 OperatorResult / Normalize raw operator return value to OperatorResult.

    支持两种输入 / Supports two inputs:
    - bool: 转为 / Convert to OperatorResult(hit=bool_val)
    - OperatorResult: 直接返回 / Return directly
    - tuple (向后兼容 / backward compatible): (hit, level, category) → OperatorResult
    """
    if isinstance(raw, OperatorResult):
        return raw
    if isinstance(raw, tuple):
        # 向后兼容旧版 icd10_range 返回 (bool, str, str) 元组
        # Backward compatibility for old icd10_range returning (bool, str, str) tuple
        hit = bool(raw[0]) if len(raw) > 0 else False
        level = raw[1] if len(raw) > 1 and raw[1] else None
        category = raw[2] if len(raw) > 2 and raw[2] else None
        return OperatorResult(hit=hit, level=level, category=category)
    return OperatorResult(hit=bool(raw))


# @runtime_checkable enables isinstance() checks against this Protocol at runtime,
# allowing validation that a callable conforms to the operator signature.
# @runtime_checkable 允许在运行时对该 Protocol 进行 isinstance() 检查，
# 从而验证可调用对象是否符合算子签名。
@runtime_checkable
class MatcherOperator(Protocol):
    """匹配算子协议 / Matcher Operator Protocol.

    所有算子必须实现此签名：接收待匹配值和参数字典，返回是否命中。
    All operators must implement this signature: receive value to match and params dict, return whether hit.
    算子必须是无状态纯函数，不持有实例变量，不产生副作用。
    Operators must be stateless pure functions, holding no instance variables and producing no side effects.

    签名 / Signature: (value: Any, params: dict[str, Any]) -> bool | OperatorResult
    """

    # Protocol method signature: any callable matching this shape is a valid operator.
    # 协议方法签名：任何符合此形状的可调用对象都是有效算子。
    # The ellipsis (...) indicates this is a structural type, not an implementation.
    # 省略号（...）表示这是一种结构类型，而不是实现。
    def __call__(self, value: Any, params: dict[str, Any]) -> Union[bool, OperatorResult]: ...


class OperatorRegistry:
    """算子注册表（类级单例） / Operator Registry (Class-level Singleton).

    管理所有已注册的匹配算子，支持装饰器注册和运行时动态注册。
    Manages all registered matching operators, supporting decorator registration and runtime dynamic registration.
    注册表使用类变量存储，所有实例共享同一份算子映射。
    The registry uses class variables for storage, and all instances share the same operator mapping.

    线程安全策略 / Thread safety strategy:
    - 写路径（register / register_func / clear）使用 Lock 保护，
      防止并发注册或热加载时的竞态条件。
    - Write paths (register / register_func / clear) are protected by Lock,
      preventing race conditions during concurrent registration or hot reloading.
    - 读路径（get / has / list_operators）无锁，
      因 CPython GIL 已保证 dict 单次读操作的原子性，
      且读操作处于规则求值热路径，加锁会引入不必要的性能开销。
    - Read paths (get / has / list_operators) are lock-free,
      because CPython GIL guarantees atomicity of single dict read operations,
      and reads are on the hot path of rule evaluation, where locking would introduce unnecessary performance overhead.
    """

    # Class-level lock: guards all write operations to _operators dict.
    # 类级锁：保护对 _operators 字典的所有写入操作。
    # Using a class variable ensures all access shares the same synchronization primitive.
    # 使用类变量确保所有访问共享相同的同步原语。
    _lock = threading.Lock()
    # Class-level operator storage: maps operator name (str) -> callable (MatcherOperator).
    # 类级算子存储：映射算子名称 (str) -> 可调用对象 (MatcherOperator)。
    # Shared across all usages since it is a class variable (singleton pattern).
    # 在所有使用中共享，因为它是类变量（单例模式）。
    _operators: dict[str, MatcherOperator] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[MatcherOperator], MatcherOperator]:
        """算子注册装饰器 / Operator registration decorator.

        Usage:
            @OperatorRegistry.register("regex")
            def regex_matcher(value, params): ...
        """
        # Return a decorator closure that captures the operator name.
        # 返回捕获算子名称的装饰器闭包。
        def decorator(func: MatcherOperator) -> MatcherOperator:
            # Acquire lock before mutating the shared _operators dict to prevent
            # race conditions when multiple threads register operators concurrently.
            # 在修改共享的 _operators 字典之前获取锁，以防止多线程并发注册算子时出现竞态条件。
            with cls._lock:
                cls._operators[name] = func
            # Return the original function unmodified so it can still be called directly.
            # 原样返回原始函数，以便仍可直接调用。
            return func
        return decorator

    @classmethod
    def register_func(cls, name: str, func: MatcherOperator) -> None:
        """运行时动态注册算子（支持插件热加载） / Runtime dynamic operator registration (supports plugin hot-reloading).

        This method enables registering operators at runtime without decorators,
        useful for plugin systems or dynamically loaded modules.
        此方法允许在没有装饰器的情况下在运行时注册算子，对于插件系统或动态加载的模块很有用。
        """
        # Acquire lock to safely insert into the shared dict.
        # 获取锁以安全地插入共享字典。
        with cls._lock:
            cls._operators[name] = func

    @classmethod
    def get(cls, name: str) -> MatcherOperator:
        """获取已注册算子（无锁读，热路径优化） / Get registered operator (lock-free read, hot path optimization).

        Lock-free read is safe because:
        1. CPython GIL guarantees atomic dict reads.
        2. This method is on the hot evaluation path; locking would add overhead.
        无锁读取是安全的，因为：
        1. CPython GIL 保证原子的字典读取。
        2. 此方法位于热求值路径上；加锁会增加开销。

        Raises:
            KeyError: 算子未注册 / Operator not registered.
        """
        try:
            # Direct dict lookup - O(1) average time complexity.
            # 直接字典查找 - O(1) 平均时间复杂度。
            return cls._operators[name]
        except KeyError:
            # Build a helpful error message listing all available operators
            # to assist debugging misconfigured rule YAML files.
            # 构建有用的错误消息，列出所有可用算子，以协助调试配置错误的规则 YAML 文件。
            available = list(cls._operators.keys())
            raise KeyError(
                f"未找到名为 '{name}' 的匹配算子。可用算子: {available}"
            )

    @classmethod
    def has(cls, name: str) -> bool:
        """检查算子是否已注册 / Check if operator is registered.

        Lock-free read: safe under GIL for single dict membership test.
        Used by the validator to check rule configs before engine execution.
        无锁读取：在 GIL 下对于单个字典成员测试是安全的。
        由验证器用于在引擎执行之前检查规则配置。
        """
        return name in cls._operators

    @classmethod
    def list_operators(cls) -> list[str]:
        """列出所有已注册算子名称 / List all registered operator names.

        Returns a snapshot list of current operator names.
        Used by management APIs and the validator's fuzzy-match suggestions.
        返回当前算子名称的快照列表。
        由管理 API 和验证器的模糊匹配建议使用。
        """
        return list(cls._operators.keys())

    @classmethod
    def clear(cls) -> None:
        """清除所有已注册算子（主要用于测试） / Clear all registered operators (primarily for testing).

        Acquires lock because this is a destructive write operation.
        Primarily used in unit tests to reset state between test cases.
        获取锁，因为这是一种破坏性的写入操作。
        主要在单元测试中用于在测试用例之间重置状态。
        """
        with cls._lock:
            cls._operators.clear()

    @classmethod
    def snapshot(cls) -> dict[str, MatcherOperator]:
        """获取当前算子注册表的快照（用于测试隔离） / Get a snapshot of current operator registry (for test isolation).

        Returns:
            当前 _operators 字典的浅拷贝 / Shallow copy of current _operators dict.
        """
        with cls._lock:
            return dict(cls._operators)

    @classmethod
    def restore(cls, snapshot: dict[str, MatcherOperator]) -> None:
        """从快照恢复算子注册表（用于测试隔离） / Restore operator registry from snapshot (for test isolation).

        Args:
            snapshot: 之前通过 snapshot() 获取的算子映射 / Operator mapping previously obtained via snapshot().
        """
        with cls._lock:
            cls._operators.clear()
            cls._operators.update(snapshot)

