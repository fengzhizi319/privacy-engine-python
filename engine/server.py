"""REST 与 gRPC 双协议统一启动与优雅关闭服务。

负责将 FastAPI REST 服务与 gRPC 服务分别运行，并捕获关闭信号（SIGTERM/SIGINT），
确保在途请求得到处理后安全干净地退出整个进程。

支持两种启动模式：
- 单进程模式（默认）：常规启动，适合开发调试；
- 多进程模式（通过 launcher.py）：SO_REUSEPORT 共享端口，适合高并发生产部署。

Unified launcher with graceful shutdown for REST and gRPC servers.
Captures termination signals, stops both servers with a grace period, and joins threads.

===================================================================================
              双协议服务启动与优雅关闭执行流程 / Dual-Protocol Lifecycle
===================================================================================

  python -m engine.server
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  模块加载阶段 / Module Bootstrap                                           │
  │    1. load_env_file()          → 加载 .env 环境变量                        │
  │    2. 读取 PRIVACY_REST_HOST/PORT, PRIVACY_GRPC_HOST/PORT                │
  │    3. 探测 uvloop / httptools  → 有则启用高性能事件循环与 HTTP 解析器          │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  main() 入口                                                            │
  │    1. argparse 解析 CLI 参数（优先级: CLI > 环境变量 > 默认值）               │
  │    2. get_security_settings() → 检查 TLS/Auth/RateLimit 状态             │
  │       └─ 全部关闭时打印 SECURITY WARNING 横幅                              │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  启动阶段 / Startup Phase                                                 │
  │                                                                         │
  │    ┌──────────────────────────────┐   ┌──────────────────────────────┐  │
  │    │  Step 1: 配置 REST            │   │  Step 3: 启动 gRPC            │  │
  │    │  uvicorn_ssl_kwargs()        │   │  grpc_serve(                  │  │
  │    │  uvicorn.Config(             │   │    wait_for_termination=False │  │
  │    │      limit_concurrency       │   │  )                            │  │
  │    │      limit_max_requests      │   │  → 返回 grpc_server 对象        │  │
  │    │      + uvloop/httptools      │   │    (主线程继续)                 │  │
  │    │    )                         │   └──────────────────────────────┘  │
  │    │  Step 2:                     │                                     │
  │    │  rest_thread = Thread(       │   ┌──────────────────────────────┐  │
  │    │    target=rest_server.run,   │   │  Step 4: 注册信号处理器         │  │
  │    │    daemon=True,              │   │  SIGTERM ─┐                  │  │
  │    │  )                           │   │           ├→ handle_shutdown()│  │
  │    │  rest_thread.start()         │   │  SIGINT ──┘   → set event    │  │
  │    │  → REST 在守护线程中运行        │   └──────────────────────────────┘  │
  │    └──────────────────────────────┘                                     │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  运行阶段 / Serving Phase                                                 │
  │                                                                         │
  │   主线程:  while not shutdown_event.is_set(): time.sleep(0.2)            │
  │                                                                         │
  │   REST 守护线程:  uvicorn.Server.run()  ← 处理 HTTP 请求                    │
  │   gRPC 线程池:    grpc_server       ← 处理 RPC 请求                        │
  │                                                                         │
  │   ┌──────────┐          ┌──────────────────────────┐                    │
  │   │  Client   │──HTTP──▶│  REST (uvicorn) :8079    │                    │
  │   └──────────┘          └──────────────────────────┘                    │
  │   ┌──────────┐          ┌──────────────────────────┐                    │
  │   │  Client   │──gRPC──▶│  gRPC server    :50051   │                    │
  │   └──────────┘          └──────────────────────────┘                    │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │  SIGTERM / SIGINT / Ctrl+C
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  优雅关闭阶段 / Graceful Shutdown Phase                                    │
  │                                                                         │
  │    Step 5: handle_shutdown() → shutdown_event.set()                     │
  │            主线程退出 while 循环                                           │
  │                                                                         │
  │    Step 6: 停止 gRPC                                                     │
  │      grpc_server.stop(grace=5)  → 5 秒 grace 期处理在途 RPC                │
  │      stop_event.wait(timeout=10) → 最多等 10 秒排空                        │
  │      超时 → 打印警告，强制取消剩余 RPC                                        │
  │                                                                         │
  │    Step 7: 停止 REST                                                     │
  │      rest_server.should_exit = True  → uvicorn 停止接受新连接              │
  │      rest_thread.join(timeout=10)    → 等待线程退出                        │
  │      daemon=True → 超时后随主进程强制终止                                    │
  │                                                                         │
  │    Step 8: sys.exit(0) → 进程干净退出                                      │
  └─────────────────────────────────────────────────────────────────────────┘

  线程模型 / Thread Model:
  ┌─────────────────────────────────────────────────────────────────┐
  │  主线程 (main)        : 信号监听 + 优雅关闭调度                       │
  │  守护线程 (daemon=True) : uvicorn REST 服务                        │
  │  gRPC 线程池            : max_workers=64 (可配置)                  │
  └─────────────────────────────────────────────────────────────────┘
===================================================================================
"""

import os
import signal
import sys
import threading
import time

import uvicorn

from .grpc_server import serve as grpc_serve
from .main import app
from .observability.logging_config import get_logger
from .ops_port import (
    create_ops_asgi_app,
    load_ops_settings,
    ops_uvicorn_ssl_kwargs,
)
from .security.config import get_security_settings
from .security.tls import uvicorn_ssl_kwargs

from .env_loader import load_env_file
load_env_file()

logger = get_logger(__name__)

# 从环境变量读取监听地址与端口
REST_HOST = os.environ.get("PRIVACY_REST_HOST", "0.0.0.0")
REST_PORT = int(os.environ.get("PRIVACY_REST_PORT", "8079"))
GRPC_HOST = os.environ.get("PRIVACY_GRPC_HOST", "0.0.0.0")
GRPC_PORT = int(os.environ.get("PRIVACY_GRPC_PORT", "50051"))
# 高并发优化：gRPC 线程池大小可通过环境变量配置
GRPC_MAX_WORKERS = int(os.environ.get("PRIVACY_GRPC_MAX_WORKERS", "64"))
if GRPC_MAX_WORKERS < 1:
    raise ValueError(
        f"PRIVACY_GRPC_MAX_WORKERS must be >= 1, got {GRPC_MAX_WORKERS}. "
        "Set it to the desired thread pool size for the gRPC server."
    )

# ---------------------------------------------------------------------------
# 高并发优化：uvloop + httptools 自动检测
# ---------------------------------------------------------------------------
# uvloop 基于 libuv 的高性能事件循环，配合 httptools 可显著提升 REST 吞吐。
# 若未安装则回退到 asyncio 默认事件循环 + 标准 HTTP 解析器。
_UVICORN_LOOP_KWARG: dict = {}
try:
    import uvloop  # noqa: F401
    _UVICORN_LOOP_KWARG["loop"] = "uvloop"
except ImportError:
    pass

try:
    import httptools  # noqa: F401
    _UVICORN_LOOP_KWARG["http"] = "httptools"
except ImportError:
    pass


def main():
    """主入口函数。

    解析命令行参数，配置并启动 REST 与 gRPC 服务器，注册系统关闭信号处理器实现优雅关闭。
    命令行参数优先级高于环境变量。
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="engine.server",
        description="SecretFlow Local Privacy Agent REST + gRPC server.",
    )
    parser.add_argument(
        "--rest-host",
        default=os.environ.get("PRIVACY_REST_HOST", REST_HOST),
        help=f"REST server host (default: {REST_HOST} or PRIVACY_REST_HOST).",
    )
    parser.add_argument(
        "--rest-port",
        type=int,
        default=int(os.environ.get("PRIVACY_REST_PORT", str(REST_PORT))),
        help=f"REST server port (default: {REST_PORT} or PRIVACY_REST_PORT).",
    )
    parser.add_argument(
        "--grpc-host",
        default=os.environ.get("PRIVACY_GRPC_HOST", GRPC_HOST),
        help=f"gRPC server host (default: {GRPC_HOST} or PRIVACY_GRPC_HOST).",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=int(os.environ.get("PRIVACY_GRPC_PORT", str(GRPC_PORT))),
        help=f"gRPC server port (default: {GRPC_PORT} or PRIVACY_GRPC_PORT).",
    )
    parser.add_argument(
        "--ops-host",
        default=os.environ.get("PRIVACY_OPS_HOST", REST_HOST),
        help=f"Ops listener host (default: {REST_HOST} or PRIVACY_OPS_HOST).",
    )
    parser.add_argument(
        "--ops-port",
        type=int,
        default=None,
        help="Ops listener port (default: REST port + 1000 or PRIVACY_OPS_PORT).",
    )
    args = parser.parse_args()

    # Emit a prominent security warning when all protections are disabled.
    # This helps operators notice the insecure default before exposing the service.
    _sec = get_security_settings()
    if not _sec.tls_enabled and not _sec.auth_enabled and not _sec.rate_limit_enabled:
        logger.warning(
            "="*72 + "\n"
            "  SECURITY WARNING: All security features are DISABLED.\n"
            "  TLS=off  Auth=off  RateLimit=off\n"
            "  All endpoints (including /v1/ops/diagnostics) are exposed\n"
            "  without encryption, authentication, or rate limiting.\n"
            "  For production deployments, set:\n"
            "    PRIVACY_TLS_ENABLED=true\n"
            "    PRIVACY_AUTH_ENABLED=true\n"
            "    PRIVACY_RATE_LIMIT_ENABLED=true\n"
            "  See docs/production_security/ops.md for details.\n"
            + "="*72
        )

    # 1. 配置 REST 隐式启动
    ssl_kwargs = uvicorn_ssl_kwargs(get_security_settings())
    # 高并发优化：自动使用 uvloop + httptools（若已安装）
    config = uvicorn.Config(
        app,
        host=args.rest_host,
        port=args.rest_port,
        log_level="info",
        # 高并发优化：限制最大并发连接数，防止过载 OOM
        limit_concurrency=int(os.environ.get("PRIVACY_LIMIT_CONCURRENCY", "10000")),
        # 高并发优化：worker 最大处理请求数，防止内存泄漏
        limit_max_requests=int(os.environ.get("PRIVACY_LIMIT_MAX_REQUESTS", "100000")),
        # 高并发优化：keep-alive 超时，减少空闲连接占用
        timeout_keep_alive=int(os.environ.get("PRIVACY_TIMEOUT_KEEP_ALIVE", "30")),
        # 优雅关闭超时：收到 SIGTERM 后等待在途请求完成的最大秒数，超时后强制断开。
        # Graceful shutdown timeout: max seconds to wait for in-flight requests
        # after SIGTERM before force-closing connections.
        timeout_graceful_shutdown=int(os.environ.get("PRIVACY_TIMEOUT_GRACEFUL_SHUTDOWN", "10")),
        **_UVICORN_LOOP_KWARG,
        **ssl_kwargs,
    )
    rest_server = uvicorn.Server(config)

    # 2. 在守护线程中启动 REST 服务
    # daemon=True：即使下方 join 超时 REST 线程未能退出，也不会挂住整个进程，
    # 保证 SIGTERM 后进程总能干净退出。
    rest_thread = threading.Thread(
        target=rest_server.run,
        name="uvicorn-rest-server",
        daemon=True,
    )
    rest_thread.start()

    # 2.5 独立运维口（默认 REST 端口 +1000）：只放行探针、指标与诊断路径。
    # lifespan="off"：主应用 lifespan 只由业务口执行一次，避免 tracing/IAM 重复初始化。
    # 运维口与业务口共享同一份中间件与鉴权栈，/metrics、/ops/diagnostics 的准入规则不变。
    ops_settings = load_ops_settings(
        args.rest_host,
        args.rest_port,
        _sec,
        ops_host=args.ops_host,
        ops_port=args.ops_port,
    )
    ops_config = uvicorn.Config(
        create_ops_asgi_app(app),
        host=ops_settings.host,
        port=ops_settings.port,
        log_level="info",
        lifespan="off",
        timeout_graceful_shutdown=int(os.environ.get("PRIVACY_TIMEOUT_GRACEFUL_SHUTDOWN", "10")),
        **_UVICORN_LOOP_KWARG,
        **ops_uvicorn_ssl_kwargs(ops_settings, _sec),
    )
    ops_server = uvicorn.Server(ops_config)
    ops_thread = threading.Thread(
        target=ops_server.run,
        name="uvicorn-ops-server",
        daemon=True,
    )
    ops_thread.start()

    # 3. 启动 gRPC 服务（非阻塞模式，使其返回 server 对象）
    # 高并发优化：线程池大小可通过 PRIVACY_GRPC_MAX_WORKERS 环境变量配置
    grpc_server = grpc_serve(
        host=args.grpc_host,
        port=args.grpc_port,
        max_workers=GRPC_MAX_WORKERS,
        wait_for_termination=False,
    )

    # 3.5 Startup info banner / 启动信息横幅
    # Log bind addresses and key config for operator visibility at startup.
    # 记录监听地址与关键配置，便于运维确认服务状态。
    logger.info(
        "PrivShield dual-protocol server started: "
        "REST=%s:%d | gRPC=%s:%d | OPS=%s:%d (%s) | grpc_workers=%d | tls=%s | auth=%s",
        args.rest_host, args.rest_port,
        args.grpc_host, args.grpc_port,
        ops_settings.host, ops_settings.port,
        ops_settings.tls_stack or "inherit",
        GRPC_MAX_WORKERS,
        _sec.tls_enabled, _sec.auth_enabled,
    )
    if ops_settings.terminator_fronted:
        logger.info(
            "ops port is loopback-only and expects the Go TLCP terminator to front it "
            "(PRIVACY_OPS_TLS_STACK=%s)",
            ops_settings.tls_stack,
        )

    # 4. 信号处理逻辑
    shutdown_event = threading.Event()

    def handle_shutdown(signum, frame):
        logger.warning("received signal %s — initiating graceful shutdown", signum)
        shutdown_event.set()

    # 注册信号处理器
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # 5. 主线程等待终止信号
    try:
        while not shutdown_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    # 6. 执行优雅退出
    logger.info("stopping gRPC server (5s grace period for in-flight RPCs)...")
    # stop() 返回 threading.Event：grace 期内在途 RPC 完成后触发。
    # 必须等待该事件（带超时兖底），否则进程会在在途 RPC 尚未排空时直接退出。
    stop_event = grpc_server.stop(grace=5)
    if not stop_event.wait(timeout=10):
        logger.warning("gRPC server did not drain in-flight requests within timeout; force-cancelling remaining RPCs")
    
    logger.info("stopping REST and ops servers...")
    rest_server.should_exit = True
    ops_server.should_exit = True

    # 等待 REST/运维口线程退出；daemon=True 保证即使超时进程也能退出
    rest_thread.join(timeout=10)
    if rest_thread.is_alive():
        logger.warning("REST server thread did not exit within timeout; will be force-terminated with main process")
    ops_thread.join(timeout=10)
    if ops_thread.is_alive():
        logger.warning("ops server thread did not exit within timeout; will be force-terminated with main process")
    logger.info("PrivShield dual-protocol server shut down gracefully")
    sys.exit(0)


if __name__ == "__main__":
    main()
