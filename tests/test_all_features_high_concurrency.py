"""全功能高并发组件与分类分级/脱敏高并发优化测试。

测试覆盖：
- HighConcurrencyLRUCache (容量限制、LRU 淘汰、命中率统计)
- AsyncDynamicBatcher (高并发动态合并 Batch 处理)
- ConcurrencyThrottle (信号量并发限流)
- DynClassificationService (高并发分类分级 LRU 缓存与命中)
- Data Masking (guess_field_type LRU 缓存与重复请求推断性能)
"""

import asyncio
import os
import pytest

from engine.dynclassification.service import DynClassificationService
from engine.privacy.high_concurrency import (
    AsyncDynamicBatcher,
    ConcurrencyThrottle,
    HighConcurrencyLRUCache,
)
from engine.privacy.masking import guess_field_type


class TestHighConcurrencyLRUCache:
    """测试全功能高并发 LRU 缓存。"""

    def test_lru_cache_basic_and_eviction(self):
        cache = HighConcurrencyLRUCache[str, int](capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        assert cache.get("a") == 1
        assert cache.get("b") == 2

        # 写入 "d" 触发淘汰，"c" 应被淘汰 (因为 "a", "b" 最近被 get 过)
        cache.put("d", 4)
        assert cache.get("c") is None
        assert cache.get("d") == 4

        stats = cache.stats
        assert stats["capacity"] == 3
        assert stats["size"] == 3
        assert stats["hits"] > 0

    def test_lru_cache_clear(self):
        cache = HighConcurrencyLRUCache[str, str](capacity=10)
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"
        cache.clear()
        assert cache.get("k1") is None
        assert cache.stats["hits"] == 0


class TestAsyncDynamicBatcher:
    """测试异步动态批处理器。"""

    @pytest.mark.anyio
    async def test_dynamic_batching(self):
        processed_batches: list[list[int]] = []

        async def fake_batch_handler(items: list[int]) -> list[int]:
            processed_batches.append(items)
            # 乘以 2 返回
            return [x * 2 for x in items]

        batcher = AsyncDynamicBatcher(
            batch_handler=fake_batch_handler,
            max_batch_size=5,
            batch_timeout_s=0.01,
        )

        # 并发提交 10 个请求
        tasks = [batcher.submit(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert results == [i * 2 for i in range(10)]
        # 应该拆分成 2 批 (每批 5 个)
        assert len(processed_batches) == 2
        assert processed_batches[0] == [0, 1, 2, 3, 4]
        assert processed_batches[1] == [5, 6, 7, 8, 9]


class TestConcurrencyThrottle:
    """测试并发限流器。"""

    @pytest.mark.anyio
    async def test_concurrency_throttle_limit(self):
        throttle = ConcurrencyThrottle(max_concurrency=2)
        active_count = 0
        max_seen = 0

        async def worker():
            nonlocal active_count, max_seen
            async with throttle.acquire():
                active_count += 1
                max_seen = max(max_seen, active_count)
                await asyncio.sleep(0.01)
                active_count -= 1

        await asyncio.gather(*(worker() for _ in range(10)))
        assert max_seen <= 2


class TestClassificationHighConcurrencyCache:
    """测试分类分级高并发 LRU 缓存。"""

    def test_classification_service_lru_cache(self):
        service = DynClassificationService(rules_dir="rules")
        service.clear_cache()

        # 首次查询（未命中缓存）
        resp1 = service.classify_field("phone", "13800000000")
        assert resp1 is not None

        # 第二次相同查询（命中缓存）
        resp2 = service.classify_field("phone", "13800000000")
        assert resp2 is not None

        # 验证缓存指标
        stats = service._classification_cache.stats
        assert stats["hits"] == 1
        assert stats["size"] == 1

        # 清空缓存
        service.clear_cache()
        assert service._classification_cache.stats["size"] == 0

    def test_classify_table_parallel(self):
        """测试多行表级并行分类。"""
        service = DynClassificationService(rules_dir="rules")
        rows = [{"phone": f"138000000{i:02d}", "name": f"user_{i}"} for i in range(30)]
        resp = service.classify_table(schema=["phone", "name"], rows=rows)
        assert resp.table_result is not None
        assert len(resp.table_result.record_results) == 30


class TestMaskingHighConcurrency:
    """测试脱敏猜测字段高并发缓存。"""

    def test_guess_field_type_lru_cached(self):
        # 多次调用，返回一致结果，且命中 functools.lru_cache
        res1 = guess_field_type("user_mobile_phone")
        res2 = guess_field_type("user_mobile_phone")
        assert res1 == "mobile"
        assert res2 == "mobile"
        # 验证 lru_cache 命中
        info = guess_field_type.cache_info()
        assert info.hits >= 1
