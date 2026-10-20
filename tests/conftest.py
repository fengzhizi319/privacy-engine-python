"""
pytest 全局配置文件 / Pytest global configuration file

本模块提供测试套件的全局 fixture 和兼容性补丁：
This module provides global fixtures and compatibility patches for the test suite:

1. scipy/torch 兼容性补丁 / scipy/torch compatibility patch:
   - test_classification_llm.py 使用 MagicMock 模拟 torch 模块
   - test_classification_llm.py uses MagicMock to stub the torch module
   - 这会导致 scipy 内部的 is_torch_array() 检查抛出 TypeError
   - This causes scipy's internal is_torch_array() check to raise TypeError
   - 此补丁在导入时一次性修复，确保测试收集和执行阶段均安全
   - This patch fixes it once at import time, ensuring safety during collection and execution

2. 隐私预算重置 fixture / Privacy budget reset fixture:
   - 每个测试前自动清空 BudgetRegistry，防止跨测试状态泄漏
   - Automatically clears BudgetRegistry before each test to prevent cross-test state leakage
   - 同步更新 REST 全局服务实例的预算引用
   - Synchronizes the budget reference in the REST global service instance

使用方式 / Usage:
    本文件由 pytest 自动加载，无需手动导入。
    This file is auto-loaded by pytest, no manual import needed.
"""

import pytest

from engine.privacy.budget import default_registry

# tests/scripts 与 tests/deployment 验证的是旧 Monorepo 根目录的脚本与 Helm 资产，
# 那些资产现由仓库根 scripts/ 与 deploy/ 拥有，不属于本服务的功能一致性范围。
# 保留文件作为迁移参考，但不纳入本服务测试基线；阶段 9 落地本服务 deploy/ 时再评估。
collect_ignore = ["scripts", "deployment"]

# ---------------------------------------------------------------------------
# scipy/torch 兼容性补丁 / scipy/torch compatibility patch
#
# 问题背景 / Background:
#   test_classification_llm.py 中设置 sys.modules["torch"] = MagicMock()
#   来模拟 torch 依赖（避免在 CI 中安装重量级 ML 包）。
#   In test_classification_llm.py, sys.modules["torch"] = MagicMock() is set
#   to mock the torch dependency (avoiding heavy ML packages in CI).
#
#   然而 scipy._external.array_api_compat 中的 is_torch_array() 函数
#   会调用 issubclass() 检查，当 torch 是 MagicMock 时抛出 TypeError。
#   However, is_torch_array() in scipy._external.array_api_compat calls
#   issubclass() which raises TypeError when torch is a MagicMock.
#
# 解决方案 / Solution:
#   包装 is_torch_array()，捕获 TypeError 并返回 False。
#   Wrap is_torch_array() to catch TypeError and return False.
# ---------------------------------------------------------------------------
try:
    from scipy._external.array_api_compat.common import _helpers as _scipy_helpers

    # 保存原始函数引用 / Save reference to original function
    _orig_is_torch_array = _scipy_helpers.is_torch_array

    def _safe_is_torch_array(x):
        """安全的 torch 数组检测包装器 / Safe torch array detection wrapper.

        当 torch 被 MagicMock 替代时，原始 issubclass() 检查会失败，
        此时安全地返回 False（表示不是 torch 数组）。
        When torch is replaced by MagicMock, the original issubclass() check fails;
        in that case, safely return False (indicating not a torch array).
        """
        try:
            return _orig_is_torch_array(x)
        except TypeError:
            return False

    # 替换 scipy helpers 中的函数 / Replace function in scipy helpers
    _scipy_helpers.is_torch_array = _safe_is_torch_array

    # 同时补丁 _array_api 模块（它在顶层导入了 is_torch_array）
    # Also patch the _array_api module (it imports is_torch_array at top level)
    try:
        import scipy._lib._array_api as _scipy_array_api
        _scipy_array_api.is_torch_array = _safe_is_torch_array
    except Exception:
        pass  # 模块结构可能变化 / Module structure may have changed
except Exception:
    pass  # scipy 未安装或内部 API 已变更，无需补丁 / scipy not installed or internal API changed


@pytest.fixture(autouse=True)
def reset_all_budgets():
    """自动使用的 fixture：每个测试前重置隐私预算状态。

    Auto-used fixture: resets privacy budget state before each test.

    目的 / Purpose:
        差分隐私预算是全局有状态资源。如果不在测试间重置，
        前一个测试消耗的预算会影响后续测试的断言结果。
        Differential privacy budget is a global stateful resource. Without resetting
        between tests, budget consumed by earlier tests would affect later assertions.

    执行步骤 / Steps:
        1. 清空默认注册表中的所有 BudgetAccountant 实例
           Clear all BudgetAccountant instances in the default registry
        2. 同步 REST 全局单例服务的预算引用
           Synchronize the budget reference in the REST global singleton service

    Yields:
        None - fixture 仅在测试前执行清理 / Only performs cleanup before test
    """
    # 步骤 1: 清空默认注册表中的所有实例
    # Step 1: Clear all instances in the default registry
    default_registry.reset()

    # 步骤 2: REST 全局单例服务（若已导入）需重新从注册表获取预算实例，
    # 避免继续持有已从注册表移除的旧实例导致预算状态不同步。
    # Step 2: The REST global singleton service (if already imported) needs to
    # re-acquire its budget instance from the registry, avoiding stale references.
    try:
        from engine.main import service
        if hasattr(service, "dp_api"):
            service.dp_api.budget = default_registry.get_or_create(service.namespace)
    except (ImportError, AttributeError):
        pass  # 服务未导入或结构不匹配，跳过 / Service not imported or structure mismatch, skip

    yield
