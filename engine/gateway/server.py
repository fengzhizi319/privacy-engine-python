"""网关统一启动入口 (Go ``cmd/privshield-gateway`` parity).

从 ``PRIVACY_GATEWAY_*`` 读取配置（默认值与 Go 逐项一致），在一个事件循环内
并发托管三张网：

- **业务 REST 面**：``PRIVACY_GATEWAY_HOST:PORT``（默认 127.0.0.1:8000）
- **业务 gRPC 面**：``PRIVACY_GATEWAY_GRPC_HOST:GRPC_PORT``（默认 127.0.0.1:50000）
- **独立运维口**：业务 Host + ``PRIVACY_GATEWAY_OPS_PORT``（默认业务口 +1000），
  仅暴露 ``/health``、``/gateway/backends``、``/metrics``。

REST 与 gRPC 各自拥有独立后端池（``PRIVACY_GATEWAY_BACKENDS`` 与
``PRIVACY_GATEWAY_GRPC_BACKENDS``），选路/熔断/EWMA 行为一致。运维口
``/gateway/backends`` 只报告 REST 池（与 Go ``NewHealthCheckHandler`` 同口径）。

安全姿态（fail-closed）：

- 非环回绑定必须开启逐请求凭据准入，或至少配置回源兜底 Key；
- 非环回绑定的观测面必须配置 ``PRIVACY_GATEWAY_METRICS_API_KEY`` /
  ``PRIVACY_GATEWAY_ADMIN_API_KEY``，否则拒绝启动；
- ``TLS_STACK=tlcp|both`` 由 Phase 8 的 Go 终结器侧车对外提供国密 mTLS，
  Python 只监听环回明文；``TLS_STACK=tls`` 时 Python 自己终止标准 TLS + mTLS。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hmac
import os
import ssl
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from engine.observability.logging_config import configure_logging, get_logger

from .balancer import LoadBalancer, health_check_loop
from .grpc_proxy import start_grpc_gateway
from .http_proxy import create_http_gateway_app, error_response
from .inbound import load_inbound_store
from .settings import GatewaySettings

configure_logging(
    log_level=os.environ.get("PRIVACY_LOG_LEVEL", "INFO"),
    json_format=os.environ.get("PRIVACY_LOG_FORMAT", "text").lower() == "json",
)
logger = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def build_balancers(settings: GatewaySettings) -> tuple[LoadBalancer, LoadBalancer]:
    """Build the independent REST and gRPC backend pools."""
    rest_balancer = LoadBalancer(strategy=settings.strategy)
    for url in settings.rest_backends:
        rest_balancer.add_node(http_url=url, grpc_address="")

    grpc_balancer = LoadBalancer(strategy=settings.strategy)
    for address in settings.grpc_backends:
        grpc_balancer.add_node(http_url="", grpc_address=address)

    for balancer in (rest_balancer, grpc_balancer):
        if settings.backend_api_key:
            balancer.set_backend_credential(settings.backend_api_key)
        if not balancer.nodes:
            logger.warning(
                "No backend nodes configured; the gateway will reject forwarding requests"
            )
    return rest_balancer, grpc_balancer


def _bearer_token(header: str | None) -> str:
    value = (header or "").strip()
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _key_matches(header: str | None, expected: str) -> bool:
    if not expected:
        return False
    return hmac.compare_digest(_bearer_token(header).encode("utf-8"), expected.encode("utf-8"))


def create_ops_app(settings: GatewaySettings, rest_balancer: LoadBalancer) -> FastAPI:
    """Ops port: /health, /gateway/backends and /metrics with Bearer gating."""
    app = FastAPI(title="PrivShield Gateway Ops", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "component": "gateway"}

    @app.get("/readyz")
    async def readyz() -> Any:
        healthy = sum(1 for node in rest_balancer.nodes if node.is_healthy)
        ready = healthy > 0
        body = {
            "status": "ok" if ready else "unavailable",
            "component": "gateway",
            "strategy": settings.strategy,
            "tls_stack": settings.tls_stack,
            "metrics_auth_mode": "bearer" if settings.metrics_api_key else "anonymous",
            "rest_backends": len(rest_balancer.nodes),
            "rest_healthy": healthy,
        }
        return JSONResponse(status_code=200 if ready else 503, content=body)

    @app.get("/gateway/backends")
    async def backends(
        request: Request, authorization: str | None = Header(default=None)
    ) -> Any:
        if settings.admin_api_key and not _key_matches(authorization, settings.admin_api_key):
            return error_response(
                request,
                401,
                "UNAUTHORIZED",
                "Unauthorized: invalid gateway admin credentials",
            )
        return {"backends": rest_balancer.health_snapshot()}

    @app.get("/metrics")
    async def metrics(
        request: Request, authorization: str | None = Header(default=None)
    ) -> Any:
        if settings.metrics_api_key and not _key_matches(
            authorization, settings.metrics_api_key
        ):
            return error_response(
                request,
                401,
                "UNAUTHORIZED",
                "Unauthorized: invalid gateway metrics credentials",
            )
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _uvicorn_tls_kwargs(
    enabled: bool, cert_file: str, key_file: str, ca_file: str
) -> dict[str, Any]:
    """mTLS uvicorn kwargs: CA present means client certificates are required."""
    if not enabled:
        return {}
    return {
        "ssl_certfile": cert_file,
        "ssl_keyfile": key_file,
        "ssl_ca_certs": ca_file or None,
        "ssl_cert_reqs": ssl.CERT_REQUIRED if ca_file else ssl.CERT_NONE,
    }


async def async_main(
    rest_host: str | None = None,
    rest_port: int | None = None,
    grpc_host: str | None = None,
    grpc_port: int | None = None,
    ops_port: int | None = None,
) -> None:
    """Load settings, start REST/gRPC/ops listeners and drain on shutdown."""
    settings = GatewaySettings.from_env()
    overrides: dict[str, Any] = {}
    if rest_host is not None:
        overrides["rest_host"] = rest_host
    if rest_port is not None:
        overrides["rest_port"] = rest_port
        if ops_port is None:
            overrides["ops_port"] = 0 if rest_port == 0 else rest_port + 1000
    if grpc_host is not None:
        overrides["grpc_host"] = grpc_host
    if grpc_port is not None:
        overrides["grpc_port"] = grpc_port
    if ops_port is not None:
        overrides["ops_port"] = ops_port
    if overrides:
        settings = dataclasses.replace(settings, **overrides)
        settings.validate()

    inbound_store = load_inbound_store(settings.inbound_keys_file)
    rest_balancer, grpc_balancer = build_balancers(settings)

    grpc_server = await start_grpc_gateway(settings, grpc_balancer, inbound_store)

    rest_app = create_http_gateway_app(
        rest_balancer, settings=settings, inbound_store=inbound_store
    )
    rest_server = uvicorn.Server(
        uvicorn.Config(
            app=rest_app,
            host=settings.rest_host,
            port=settings.rest_port,
            log_level="info",
            limit_concurrency=_env_int("PRIVACY_GATEWAY_LIMIT_CONCURRENCY", 10000),
            limit_max_requests=_env_int("PRIVACY_GATEWAY_LIMIT_MAX_REQUESTS", 100000),
            timeout_keep_alive=_env_int("PRIVACY_GATEWAY_TIMEOUT_KEEP_ALIVE", 120),
            timeout_graceful_shutdown=_env_int(
                "PRIVACY_GATEWAY_TIMEOUT_GRACEFUL_SHUTDOWN", 10
            ),
            **_uvicorn_tls_kwargs(
                settings.tls_enabled,
                settings.tls_cert_file,
                settings.tls_key_file,
                settings.tls_ca_file,
            ),
        )
    )

    ops_app = create_ops_app(settings, rest_balancer)
    ops_server = uvicorn.Server(
        uvicorn.Config(
            app=ops_app,
            host=settings.rest_host,
            port=settings.ops_port,
            log_level="info",
            timeout_keep_alive=_env_int("PRIVACY_GATEWAY_TIMEOUT_KEEP_ALIVE", 120),
            timeout_graceful_shutdown=_env_int(
                "PRIVACY_GATEWAY_TIMEOUT_GRACEFUL_SHUTDOWN", 10
            ),
            **_uvicorn_tls_kwargs(
                settings.ops_tls_enabled,
                settings.ops_tls_cert_file,
                settings.ops_tls_key_file,
                settings.ops_tls_ca_file,
            ),
        )
    )

    health_tasks: list[asyncio.Task[None]] = []
    if settings.health_check_interval > 0:
        gauge_owner = rest_balancer if rest_balancer.nodes else grpc_balancer
        health_tasks = [
            asyncio.create_task(
                health_check_loop(
                    balancer,
                    settings.health_check_interval,
                    update_gauge=balancer is gauge_owner,
                )
            )
            for balancer in (rest_balancer, grpc_balancer)
            if balancer.nodes
        ]

    logger.info(
        "Gateway services launched",
        extra={
            "rest_addr": settings.rest_address(),
            "grpc_addr": settings.grpc_address(),
            "ops_addr": settings.ops_address(),
            "strategy": settings.strategy,
            "tls_stack": settings.tls_stack,
        },
    )

    try:
        await asyncio.gather(
            rest_server.serve(),
            ops_server.serve(),
            grpc_server.wait_for_termination(),
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Gateway is shutting down...")
    finally:
        for task in health_tasks:
            task.cancel()
        if health_tasks:
            await asyncio.gather(*health_tasks, return_exceptions=True)
        await grpc_server.stop(grace=1.0)
        await rest_balancer.close_all()
        await grpc_balancer.close_all()
        logger.info("Gateway services safely stopped.")


def main() -> None:
    """Synchronous entrypoint (`python -m engine.gateway.server`)."""
    parser = argparse.ArgumentParser(
        prog="engine.gateway.server",
        description="PrivShield REST + gRPC gateway / load balancer.",
    )
    parser.add_argument("--rest-host", default=None)
    parser.add_argument("--rest-port", type=int, default=None)
    parser.add_argument("--grpc-host", default=None)
    parser.add_argument("--grpc-port", type=int, default=None)
    parser.add_argument("--ops-port", type=int, default=None)
    args = parser.parse_args()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(
            async_main(
                rest_host=args.rest_host,
                rest_port=args.rest_port,
                grpc_host=args.grpc_host,
                grpc_port=args.grpc_port,
                ops_port=args.ops_port,
            )
        )


if __name__ == "__main__":
    main()
