"""网关负载均衡平滑加权轮询与连接跟踪扩展测试。"""

import pytest
from engine.gateway.balancer import LoadBalancer, BackendNode


class TestEnhancedLoadBalancer:
    """测试增强版 LoadBalancer 策略（Smooth Weighted Round Robin, Weighted Random 等）。"""

    @pytest.mark.anyio
    async def test_smooth_weighted_round_robin(self):
        lb = LoadBalancer(strategy="weighted_round_robin")
        lb.add_node("http://127.0.0.1:8001", "127.0.0.1:5001", weight=5)
        lb.add_node("http://127.0.0.1:8002", "127.0.0.1:5002", weight=1)

        selected_counts = {"http://127.0.0.1:8001": 0, "http://127.0.0.1:8002": 0}
        for _ in range(6):
            node = await lb.select_node()
            assert node is not None
            selected_counts[node.http_url] += 1

        # 权重 5:1 的情况下，6 次选择应该准确分发 5 次和 1 次
        assert selected_counts["http://127.0.0.1:8001"] == 5
        assert selected_counts["http://127.0.0.1:8002"] == 1

    @pytest.mark.anyio
    async def test_track_connection_context_manager(self):
        node = BackendNode("http://127.0.0.1:8001", "127.0.0.1:5001")
        assert node.active_connections == 0
        
        async with node.track_connection():
            assert node.active_connections == 1
            
        assert node.active_connections == 0
