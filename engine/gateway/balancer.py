"""负载均衡、熔断与健康检查 (Load Balancer, Circuit Breaker & Health Check).

与 Go ``pkg/circuitbreaker`` + ``pkg/gateway`` 逐语义对齐：

- **三态熔断器**：``closed`` → ``open`` → ``half-open``；连续失败阈值 5、
  冷却 30s、半开最多 3 个在途探测；结果通过 generation 绑定的 :class:`Permit`
  上报，跨状态轮次的残余结果不会误改状态。
- **P2C-EWMA**：默认策略 ``p2c``，评分 ``(in_flight + 1) * max(ewma, 0.001)``。
- **调度策略**：``p2c`` / ``round_robin`` / ``least_conn`` / ``weighted_rr`` /
  ``weighted_random``，未知策略回落 P2C。
- **选路与准入分离**：``select_node`` 只做只读的 ``can_allow`` 预筛选；
  真正的准入由调用方执行 ``breaker.acquire()``，与 Go 网关一致。

Python 额外保留主动健康探针、手动隔离/排空与平滑加权轮询别名，属于超集能力，
不改变上述 Go 语义。

回源 TLS（可选）：``PRIVACY_GATEWAY_BACKEND_TLS_ENABLED`` /
``PRIVACY_GATEWAY_BACKEND_TLS_CA`` / ``..._CLIENT_CERT`` / ``..._CLIENT_KEY``。
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import random
import threading
import time

import grpc
import httpx

from engine import privacy_pb2, privacy_pb2_grpc
from engine.observability.logging_config import get_logger
from engine.observability.metrics import (
    GATEWAY_CIRCUIT_BREAKER_STATE,
    GATEWAY_HEALTHY_NODES,
    GATEWAY_NODE_ADMIN_STATE,
)

logger = get_logger(__name__)

GRPC_MAX_MESSAGE_BYTES = 64 * 1024 * 1024

GRPC_CHANNEL_OPTIONS = [
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
]


def backend_tls_enabled() -> bool:
    """Return whether gateway→backend TLS is enabled."""
    return os.environ.get("PRIVACY_GATEWAY_BACKEND_TLS_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def backend_tls_verify() -> str | bool:
    """Return the httpx ``verify`` argument, failing fast on incomplete TLS config."""
    if not backend_tls_enabled():
        return True
    ca_path = os.environ.get("PRIVACY_GATEWAY_BACKEND_TLS_CA", "").strip()
    if not ca_path:
        raise RuntimeError(
            "PRIVACY_GATEWAY_BACKEND_TLS_ENABLED=true 但未配置 "
            "PRIVACY_GATEWAY_BACKEND_TLS_CA（后端证书校验 CA 文件路径）"
        )
    if not os.path.isfile(ca_path):
        raise RuntimeError(f"回源 TLS CA 文件不存在: {ca_path}")
    return ca_path


def backend_channel_credentials() -> grpc.ChannelCredentials | None:
    """Build gRPC channel credentials for the backend, or None for plaintext."""
    if not backend_tls_enabled():
        return None
    ca_path = backend_tls_verify()
    with open(ca_path, "rb") as f:
        root_certs = f.read()
    client_cert_path = os.environ.get("PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_CERT", "").strip()
    client_key_path = os.environ.get("PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_KEY", "").strip()
    private_key = certificate_chain = None
    if client_cert_path or client_key_path:
        if not (client_cert_path and client_key_path):
            raise RuntimeError(
                "回源 mTLS 需同时配置 PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_CERT 与 "
                "PRIVACY_GATEWAY_BACKEND_TLS_CLIENT_KEY"
            )
        for path in (client_cert_path, client_key_path):
            if not os.path.isfile(path):
                raise RuntimeError(f"回源 mTLS 客户端证书/私钥文件不存在: {path}")
        with open(client_key_path, "rb") as f:
            private_key = f.read()
        with open(client_cert_path, "rb") as f:
            certificate_chain = f.read()
    return grpc.ssl_channel_credentials(
        root_certificates=root_certs,
        private_key=private_key,
        certificate_chain=certificate_chain,
    )


# ---------------------------------------------------------------------------
# Circuit breaker (Go pkg/circuitbreaker parity)
# ---------------------------------------------------------------------------


class Permit:
    """Generation-bound admission permit.

    Exactly one of :meth:`success`, :meth:`failure`, :meth:`release` takes effect;
    later calls are ignored. Results from a previous breaker generation are
    discarded, mirroring Go ``Permit.complete``.
    """

    def __init__(self, breaker: CircuitBreaker, generation: int):
        self._breaker = breaker
        self._generation = generation
        self._done = False
        self._lock = threading.Lock()

    def success(self) -> None:
        self._complete(1)

    def failure(self) -> None:
        self._complete(-1)

    def release(self) -> None:
        self._complete(0)

    def _complete(self, result: int) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        self._breaker._complete(self._generation, result)


class CircuitBreaker:
    """Three-state circuit breaker with Go-compatible semantics.

    ``can_allow()`` is a read-only candidate filter. ``acquire()`` atomically
    transitions ``open`` → ``half-open`` after the cooldown and reserves one of
    ``half_open_max`` probe slots.
    """

    DEFAULT_THRESHOLD = 5
    DEFAULT_COOLDOWN = 30.0
    DEFAULT_HALF_OPEN_MAX = 3

    def __init__(
        self,
        failure_threshold: int = DEFAULT_THRESHOLD,
        recovery_timeout: float = DEFAULT_COOLDOWN,
        half_open_max: int = DEFAULT_HALF_OPEN_MAX,
        backoff_factor: float = 1.0,
        max_cooldown: float | None = None,
    ):
        self.failure_threshold = failure_threshold if failure_threshold > 0 else self.DEFAULT_THRESHOLD
        self.recovery_timeout = recovery_timeout if recovery_timeout > 0 else self.DEFAULT_COOLDOWN
        self.half_open_max = half_open_max if half_open_max > 0 else self.DEFAULT_HALF_OPEN_MAX
        if not math.isfinite(backoff_factor) or backoff_factor < 1.0:
            backoff_factor = 1.0
        self.backoff_factor = backoff_factor
        if max_cooldown is None and backoff_factor > 1.0:
            max_cooldown = max(self.recovery_timeout * 10, 300.0)
        self.max_cooldown = max_cooldown

        self._generation = 0
        self._state = "closed"
        self._failures = 0
        self._successes = 0
        self._in_flight_probes = 0
        self._opened_at = 0.0
        self._base_cooldown = self.recovery_timeout
        self._current_cooldown = self.recovery_timeout
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Return the current state without mutating it."""
        with self._lock:
            return self._state

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def successes(self) -> int:
        with self._lock:
            return self._successes

    @property
    def in_flight_probes(self) -> int:
        with self._lock:
            return self._in_flight_probes

    @property
    def opened_at(self) -> float:
        with self._lock:
            return self._opened_at

    @property
    def current_cooldown(self) -> float:
        with self._lock:
            return self._current_cooldown

    def can_allow(self) -> bool:
        """Read-only availability check used by candidate selection."""
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                return (time.monotonic() - self._opened_at) >= self._current_cooldown
            return (self._successes + self._in_flight_probes) < self.half_open_max

    def allow_request(self) -> bool:
        """Deprecated alias for :meth:`can_allow` (read-only)."""
        return self.can_allow()

    def acquire(self) -> Permit | None:
        """Atomically admit a request; ``None`` means the breaker denied it."""
        with self._lock:
            if self._state == "open":
                if (time.monotonic() - self._opened_at) < self._current_cooldown:
                    return None
                self._generation += 1
                self._state = "half-open"
                self._successes = 0
                self._in_flight_probes = 0
            elif self._state == "half-open":
                if (self._successes + self._in_flight_probes) >= self.half_open_max:
                    return None
            if self._state == "half-open":
                self._in_flight_probes += 1
            return Permit(self, self._generation)

    def reset(self) -> None:
        """Reset to the initial closed state (used by active health recovery)."""
        with self._lock:
            self._generation += 1
            self._state = "closed"
            self._failures = 0
            self._successes = 0
            self._in_flight_probes = 0
            self._current_cooldown = self._base_cooldown
            self._opened_at = 0.0

    def _complete(self, generation: int, result: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            if self._state == "half-open" and self._in_flight_probes > 0:
                self._in_flight_probes -= 1
            if result == 0:
                return
            if result > 0:
                if self._state == "closed":
                    self._failures = 0
                elif self._state == "half-open":
                    self._successes += 1
                    if self._successes >= self.half_open_max:
                        self._generation += 1
                        self._state = "closed"
                        self._failures = 0
                        self._in_flight_probes = 0
                        self._current_cooldown = self._base_cooldown
                return
            if self._state == "closed":
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._generation += 1
                    self._state = "open"
                    self._opened_at = time.monotonic()
                    self._current_cooldown = self._base_cooldown
            elif self._state == "half-open":
                self._generation += 1
                self._state = "open"
                self._opened_at = time.monotonic()
                self._in_flight_probes = 0
                if self.backoff_factor > 1.0:
                    next_cooldown = self._current_cooldown * self.backoff_factor
                    if self.max_cooldown is not None and next_cooldown >= self.max_cooldown:
                        self._current_cooldown = self.max_cooldown
                    else:
                        self._current_cooldown = next_cooldown


# ---------------------------------------------------------------------------
# Backend node
# ---------------------------------------------------------------------------


class BackendNode:
    """Backend node with health, in-flight, EWMA and breaker state."""

    def __init__(self, http_url: str, grpc_address: str, weight: int = 1):
        self.http_url = http_url.rstrip("/")
        self.grpc_address = grpc_address
        self.weight = max(1, weight)
        self.current_weight = 0
        self._state_lock = threading.Lock()
        self._is_healthy = True
        self._passive_unhealthy_until = 0.0
        self._active_connections = 0
        self.circuit_breaker = CircuitBreaker()
        self._ewma = 0.0
        self._ewma_lock = threading.Lock()
        self._grpc_channel: grpc.aio.Channel | None = None
        self._grpc_stub: privacy_pb2_grpc.PrivacyServiceStub | None = None
        self._stub_lock = threading.RLock()
        self._admin_state = "active"
        self._backend_auth_header = ""

    @property
    def node_id(self) -> str:
        """Stable identifier used in metrics and ops responses."""
        return self.http_url or self.grpc_address

    @property
    def rest_node_id(self) -> str:
        """Node address as seen by the REST balancer (Go ``node.Address``)."""
        return self.http_url.split("://", 1)[-1].rstrip("/")

    @property
    def grpc_node_id(self) -> str:
        """Node address as seen by the gRPC balancer (Go ``node.Address``)."""
        return self.grpc_address

    @property
    def backend_auth_header(self) -> str:
        """Fallback ``Authorization`` value injected only for anonymous calls."""
        return self._backend_auth_header

    @property
    def is_healthy(self) -> bool:
        with self._state_lock:
            return self._is_healthy

    @is_healthy.setter
    def is_healthy(self, value: bool) -> None:
        with self._state_lock:
            self._is_healthy = bool(value)

    @property
    def passive_unhealthy_until(self) -> float:
        with self._state_lock:
            return self._passive_unhealthy_until

    @passive_unhealthy_until.setter
    def passive_unhealthy_until(self, value: float) -> None:
        with self._state_lock:
            self._passive_unhealthy_until = float(value)

    @property
    def active_connections(self) -> int:
        with self._state_lock:
            return self._active_connections

    @active_connections.setter
    def active_connections(self, value: int) -> None:
        with self._state_lock:
            self._active_connections = max(0, int(value))

    @property
    def admin_state(self) -> str:
        with self._state_lock:
            return self._admin_state

    @admin_state.setter
    def admin_state(self, value: str) -> None:
        with self._state_lock:
            self._admin_state = value

    @property
    def ewma(self) -> float:
        """Read the EWMA latency in seconds."""
        with self._ewma_lock:
            return self._ewma

    def update_ewma(self, latency_seconds: float, alpha: float) -> None:
        with self._ewma_lock:
            self._ewma = alpha * latency_seconds + (1 - alpha) * self._ewma

    def set_backend_credential(self, api_key: str) -> None:
        """Set the fallback backend credential injected only for anonymous calls."""
        key = (api_key or "").strip()
        self._backend_auth_header = f"Bearer {key}" if key else ""

    def mark_unhealthy(self, cooldown_seconds: float = 5.0) -> None:
        with self._state_lock:
            self._is_healthy = False
            self._passive_unhealthy_until = time.monotonic() + cooldown_seconds

    def mark_healthy(self) -> None:
        with self._state_lock:
            self._is_healthy = True
            self._passive_unhealthy_until = 0.0

    def _build_grpc_channel(self) -> grpc.aio.Channel:
        credentials = backend_channel_credentials()
        if credentials is not None:
            return grpc.aio.secure_channel(
                self.grpc_address, credentials, options=GRPC_CHANNEL_OPTIONS
            )
        return grpc.aio.insecure_channel(self.grpc_address, options=GRPC_CHANNEL_OPTIONS)

    @property
    def grpc_channel(self) -> grpc.aio.Channel:
        """Lazily create the reusable backend channel."""
        if self._grpc_channel is None:
            with self._stub_lock:
                if self._grpc_channel is None:
                    self._grpc_channel = self._build_grpc_channel()
        return self._grpc_channel

    @property
    def grpc_stub(self) -> privacy_pb2_grpc.PrivacyServiceStub:
        """Lazily create the typed stub used by active health probes."""
        if self._grpc_stub is None:
            with self._stub_lock:
                if self._grpc_stub is None:
                    self._grpc_stub = privacy_pb2_grpc.PrivacyServiceStub(self.grpc_channel)
        return self._grpc_stub

    @contextlib.asynccontextmanager
    async def track_connection(self):
        """Track in-flight requests for least-conn and P2C scoring."""
        self.active_connections = self.active_connections + 1
        try:
            yield self
        finally:
            self.active_connections = self.active_connections - 1

    async def close(self) -> None:
        channel = self._grpc_channel
        if channel is not None:
            await channel.close()
            self._grpc_channel = None
            self._grpc_stub = None


# ---------------------------------------------------------------------------
# Load balancer
# ---------------------------------------------------------------------------

_STRATEGY_ALIASES = {
    "power_of_two_choices": "p2c",
    "least_connections": "least_conn",
    "weighted_round_robin": "weighted_rr",
    "random": "weighted_random",
    "weighted_random": "weighted_random",
    "round_robin": "round_robin",
    "p2c": "p2c",
    "least_conn": "least_conn",
    "weighted_rr": "weighted_rr",
}


class LoadBalancer:
    """Adaptive load balancer with P2C-EWMA as the default strategy."""

    def __init__(self, strategy: str = "p2c"):
        self.strategy = strategy.lower()
        self.nodes: list[BackendNode] = []
        self.rr_index = 0
        self._nodes_lock = threading.Lock()
        self.modify_lock = self._nodes_lock
        self._selection_lock = threading.Lock()

    def _normalized_strategy(self) -> str:
        return _STRATEGY_ALIASES.get(self.strategy, "p2c")

    def _get_healthy_nodes_locked(self, nodes: list[BackendNode]) -> list[BackendNode]:
        """Apply Python-only health/admin filters (Go filters only ``CanAllow``)."""
        now = time.monotonic()
        return [
            node
            for node in nodes
            if node.is_healthy
            and now >= node.passive_unhealthy_until
            and node.admin_state == "active"
        ]

    def add_node(self, http_url: str, grpc_address: str, weight: int = 1) -> None:
        with self._nodes_lock:
            clean_url = http_url.rstrip("/")
            for node in self.nodes:
                if node.http_url == clean_url and node.grpc_address == grpc_address:
                    node.mark_healthy()
                    node.weight = max(1, weight)
                    node.active_connections = 0
                    node.current_weight = 0
                    node.circuit_breaker.reset()
                    logger.info(
                        "Updated existing backend node",
                        extra={"http_url": http_url, "grpc_address": grpc_address},
                    )
                    self._update_healthy_gauge_locked()
                    return
            node = BackendNode(http_url, grpc_address, weight)
            self.nodes.append(node)
            logger.info(
                "Added backend node",
                extra={"http_url": http_url, "grpc_address": grpc_address},
            )
            self._update_healthy_gauge_locked()

    def set_backend_credential(self, api_key: str) -> None:
        with self._nodes_lock:
            for node in self.nodes:
                node.set_backend_credential(api_key)

    def remove_node(self, http_url: str, grpc_address: str) -> None:
        with self._nodes_lock:
            clean_url = http_url.rstrip("/")
            remaining = []
            removed = None
            for node in self.nodes:
                if node.http_url == clean_url and node.grpc_address == grpc_address:
                    removed = node
                else:
                    remaining.append(node)
            self.nodes = remaining
            if removed is not None:
                threading.Thread(
                    target=lambda n: asyncio.run(n.close()),
                    args=(removed,),
                    daemon=True,
                ).start()
                logger.info(
                    "Removed backend node",
                    extra={"http_url": http_url, "grpc_address": grpc_address},
                )
                self._update_healthy_gauge_locked()

    def isolate_node(self, http_url: str, grpc_address: str) -> bool:
        with self._nodes_lock:
            clean_url = http_url.rstrip("/")
            for node in self.nodes:
                if node.http_url == clean_url and node.grpc_address == grpc_address:
                    node.admin_state = "isolated"
                    node.is_healthy = False
                    GATEWAY_NODE_ADMIN_STATE.labels(node=node.node_id).set(1)
                    logger.warning(
                        "Node manually isolated",
                        extra={"http_url": http_url, "grpc_address": grpc_address},
                    )
                    self._update_healthy_gauge_locked()
                    return True
            return False

    def drain_node(self, http_url: str, grpc_address: str) -> bool:
        with self._nodes_lock:
            clean_url = http_url.rstrip("/")
            for node in self.nodes:
                if node.http_url == clean_url and node.grpc_address == grpc_address:
                    node.admin_state = "drained"
                    GATEWAY_NODE_ADMIN_STATE.labels(node=node.node_id).set(2)
                    logger.warning(
                        "Node drained (no new requests)",
                        extra={"http_url": http_url, "grpc_address": grpc_address},
                    )
                    self._update_healthy_gauge_locked()
                    return True
            return False

    def activate_node(self, http_url: str, grpc_address: str) -> bool:
        with self._nodes_lock:
            clean_url = http_url.rstrip("/")
            for node in self.nodes:
                if node.http_url == clean_url and node.grpc_address == grpc_address:
                    node.admin_state = "active"
                    node.mark_healthy()
                    GATEWAY_NODE_ADMIN_STATE.labels(node=node.node_id).set(0)
                    logger.info(
                        "Node reactivated",
                        extra={"http_url": http_url, "grpc_address": grpc_address},
                    )
                    self._update_healthy_gauge_locked()
                    return True
            return False

    def get_healthy_nodes(self) -> list[BackendNode]:
        with self._nodes_lock:
            return self._get_healthy_nodes_locked(self.nodes)

    def _select_locked(self) -> BackendNode | None:
        """Selection under ``_nodes_lock``; never reserves breaker permits."""
        healthy = self._get_healthy_nodes_locked(self.nodes)
        if not healthy:
            return None

        available = [node for node in healthy if node.circuit_breaker.can_allow()]
        if not available:
            # Go falls back to nodes[0]; the proxy's Acquire() then yields
            # CIRCUIT_OPEN / Unavailable, preserving the failure attribution.
            return healthy[0]

        strategy = self._normalized_strategy()
        if strategy == "p2c":
            if len(available) == 1:
                return available[0]
            a, b = random.sample(available, 2)
            score_a = (a.active_connections + 1) * max(a.ewma, 0.001)
            score_b = (b.active_connections + 1) * max(b.ewma, 0.001)
            return a if score_a <= score_b else b

        if strategy == "least_conn":
            return min(available, key=lambda node: node.active_connections)

        if strategy == "weighted_rr":
            total_weight = sum(node.weight for node in available)
            best_node: BackendNode | None = None
            for node in available:
                node.current_weight += node.weight
                if best_node is None or node.current_weight > best_node.current_weight:
                    best_node = node
            if best_node is not None:
                best_node.current_weight -= total_weight
            return best_node

        if strategy == "weighted_random":
            weights = [node.weight for node in available]
            return random.choices(available, weights=weights, k=1)[0]

        node = available[self.rr_index % len(available)]
        self.rr_index += 1
        return node

    async def select_node(self) -> BackendNode | None:
        """Pick a backend using the configured strategy (Go ``SelectNode`` parity)."""
        with self._selection_lock, self._nodes_lock:
            return self._select_locked()

    def health_snapshot(self) -> list[dict[str, object]]:
        """Return ops-port backend topology (Go ``/gateway/backends`` shape)."""
        with self._nodes_lock:
            return [
                {
                    "address": node.rest_node_id or node.node_id,
                    "in_flight": node.active_connections,
                    "ewma_ms": node.ewma * 1000.0,
                    "cb_state": node.circuit_breaker.state,
                }
                for node in self.nodes
            ]

    async def close_all(self) -> None:
        with self._nodes_lock:
            nodes = list(self.nodes)
        for node in nodes:
            await node.close()

    def _update_healthy_gauge_locked(self) -> None:
        count = 0
        now = time.monotonic()
        for node in self.nodes:
            if (
                node.is_healthy
                and now >= node.passive_unhealthy_until
                and node.circuit_breaker.can_allow()
                and node.admin_state == "active"
            ):
                count += 1
        GATEWAY_HEALTHY_NODES.set(count)


# ---------------------------------------------------------------------------
# Active health check loop (Python extra)
# ---------------------------------------------------------------------------


async def health_check_loop(
    balancer: LoadBalancer,
    interval: float = 5.0,
    update_gauge: bool = True,
) -> None:
    """Probe the endpoints a node exposes and reconcile breaker state.

    Python extra (Go has no active probe loop). Nodes created by the split
    REST / gRPC balancers only carry one address; the missing protocol is
    treated as "not probed" rather than "down".
    """
    logger.info("Starting background health check loop", extra={"interval_seconds": interval})
    async with httpx.AsyncClient(verify=backend_tls_verify()) as client:
        while True:
            with balancer._nodes_lock:
                nodes = balancer.nodes.copy()

            for node in nodes:
                http_ok = True
                if node.http_url:
                    http_ok = False
                    try:
                        response = await client.get(f"{node.http_url}/health", timeout=2.0)
                        if response.status_code == 200:
                            http_ok = response.json().get("status") == "ok"
                    except Exception as e:  # noqa: BLE001 - probe failures are expected
                        logger.debug(
                            "HTTP health check failed",
                            extra={"node": node.http_url, "error": str(e)},
                        )

                grpc_ok = True
                if node.grpc_address:
                    grpc_ok = False
                    try:
                        response = await node.grpc_stub.Health(privacy_pb2.HealthRequest(), timeout=2.0)
                        grpc_ok = response.status == "ok"
                    except Exception as e:  # noqa: BLE001 - probe failures are expected
                        logger.debug(
                            "gRPC health check failed",
                            extra={"node": node.grpc_address, "error": str(e)},
                        )

                was_healthy = node.is_healthy
                passive_cooldown = time.monotonic() < node.passive_unhealthy_until
                healthy_now = http_ok and grpc_ok and not passive_cooldown
                node.is_healthy = healthy_now

                if healthy_now:
                    # Active probes are authoritative for recovery; reset the
                    # breaker so a node that came back is selectable immediately.
                    node.circuit_breaker.reset()
                else:
                    permit = node.circuit_breaker.acquire()
                    if permit is not None:
                        permit.failure()

                if was_healthy != healthy_now:
                    log = logger.info if healthy_now else logger.warning
                    log(
                        "Node status changed",
                        extra={
                            "node": node.node_id,
                            "status": "healthy" if healthy_now else "unhealthy",
                            "http": "UP" if http_ok else "DOWN",
                            "grpc": "UP" if grpc_ok else "DOWN",
                            "circuit_breaker": node.circuit_breaker.state,
                        },
                    )

                cb_state = node.circuit_breaker.state
                cb_name = "half_open" if cb_state == "half-open" else cb_state
                state_value = {"closed": 0.0, "half-open": 1.0, "open": 2.0}.get(cb_state, 0.0)
                GATEWAY_CIRCUIT_BREAKER_STATE.labels(
                    node_id=node.node_id, state=cb_name
                ).set(state_value)

            if update_gauge:
                with balancer._nodes_lock:
                    balancer._update_healthy_gauge_locked()
            await asyncio.sleep(interval)
