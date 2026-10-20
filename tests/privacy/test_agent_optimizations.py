"""Agent 优化特性单元测试 / Agent Optimization Feature Unit Tests.

中文说明：
验证 PrivShield 的核心优化特性：
- ParameterResolver 缓存复用：相同路径的解析器实例应复用，避免重复解析 YAML。

English Description:
Tests for core optimization features of PrivShield:
- ParameterResolver caching: same path should reuse resolver instance.
"""

import unittest

from engine.privacy.profile import get_resolver


class TestAgentOptimizations(unittest.TestCase):
    """Agent 优化特性测试集 / Agent Optimization Feature Test Suite.

    中文说明：
    覆盖参数解析器缓存机制的正确性验证。

    English Description:
    Covers parameter resolver caching mechanism correctness.
    """

    def test_resolver_caching(self):
        """验证 get_resolver 对相同路径返回同一实例（缓存复用）。

        Verify that get_resolver returns the same instance for the same path,
        avoiding redundant YAML parsing overhead.
        """
        r1 = get_resolver("nonexistent-profile.yaml")
        r2 = get_resolver("nonexistent-profile.yaml")
        self.assertIs(r1, r2)
