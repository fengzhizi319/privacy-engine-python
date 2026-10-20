"""REST API 入口模块（应用装配）。

基于 FastAPI 构建，负责应用级装配：生命周期管理、可观测性中间件、
Prometheus metrics 暴露，以及把按域拆分到 ``routers/*`` 的子路由统一挂载。

各端点的请求模型定义在 ``schemas.py``，跨路由共享的服务单例、安全依赖与
异常映射定义在 ``deps.py``；本模块仅做组装，不再内联具体端点实现。

REST API entrypoint built with FastAPI. Endpoint implementations are split into
``routers/*``; request models live in ``schemas.py`` and shared dependencies
(service singleton, security deps, exception mapping) live in ``deps.py``.
"""

# =============================================================================
# Phase 1: Environment bootstrap — load .env files before any other imports
# Phase 1: 环境引导 — 在任何其他导入之前加载 .env 配置文件
# =============================================================================

# Load environment variables from .env / config/env/<profile>.env so that
# downstream imports (e.g. security/config.py) can read settings at module level.
# 从 .env 及 config/env/<profile>.env 加载环境变量，确保下游模块（如 security/config.py）
# 在模块级导入时即可读取到配置值。
from .env_loader import load_env_file
load_env_file()

# =============================================================================
# Phase 2: Standard-library & third-party imports
# Phase 2: 标准库与第三方库导入
# =============================================================================

import os
from contextlib import asynccontextmanager

# FastAPI core framework for building REST API endpoints
# FastAPI 核心框架，用于构建 REST API 端点
from fastapi import FastAPI
# GZip middleware for compressing large HTTP responses to reduce bandwidth
# GZip 中间件，压缩大型 HTTP 响应以减少网络带宽消耗
from fastapi.middleware.gzip import GZipMiddleware
# CORS middleware for cross-origin resource sharing (opt-in via env var)
# CORS 中间件，跨域资源共享（通过环境变量可选启用）
from fastapi.middleware.cors import CORSMiddleware
# Starlette base middleware class for implementing custom ASGI middleware
# Starlette 基础中间件类，用于实现自定义 ASGI 中间件
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# =============================================================================
# Phase 3: Internal package imports — observability, security, routers
# Phase 3: 内部包导入 — 可观测性、安全、路由模块
# =============================================================================

# Structured logging configuration (text/json format, per-module log levels)
# 结构化日志配置（支持 text/json 格式、按模块设置日志级别）
from .observability.logging_config import configure_logging, get_logger
# Prometheus metrics ASGI app factory — creates a standalone metrics endpoint
# Prometheus 指标 ASGI 应用工厂 — 创建独立的指标暴露端点
from .observability.metrics import make_asgi_app
# Observability middleware: injects request_id, records access logs & metrics
# 可观测性中间件：注入 request_id、记录访问日志与 Prometheus 指标
from .observability.middleware import ObservabilityMiddleware
# OpenTelemetry distributed tracing initializer (optional, no-op if not configured)
# OpenTelemetry 分布式追踪初始化器（可选，未配置时为 no-op）
from .observability.tracing import init_tracing

# Re-export the PrivacyService singleton so that tests and callers can access
# it via ``from engine.main import service``.
# 再导出 PrivacyService 单例，使测试和外部调用方可通过
# ``from engine.main import service`` 获取。
from .deps import service  # noqa: F401
# Import exception mapper for global exception handler registration
# 导入异常映射函数，用于注册全局异常处理器
from .deps import handle_request_exception
from .deps import SECURITY_DEPS
from .security.auth import require_permission
from .auth.config import AuthSettings
from .auth.routes import (
    register_auth_exception_handler,
    router as auth_router,
    well_known_router as auth_well_known_router,
)
from .auth.runtime import get_auth_runtime
# Unified error envelope: cross-language consistent error response format
# 统一错误信封：跨语言一致的错误响应格式
from .observability.envelope import (
    register_envelope_exception_handlers,
    _build_error_envelope,
)

# Medical pipeline router (multi-step classification + privacy processing)
# 医疗流水线路由（多步分类 + 隐私处理组合端点）
from .pipeline import router as pipeline_router

# Domain-specific routers: each module exposes a ``router`` (APIRouter instance)
# 按域拆分的路由：每个模块暴露一个 ``router``（APIRouter 实例）
#   health      — /v1/health, /v1/readyz     健康检查与就绪探针
#   mask        — /v1/privacy/mask/*          字段级脱敏
#   dp          — /v1/privacy/dp/*            差分隐私（Laplace/Gaussian）
#   ldp         — /v1/privacy/ldp/*           本地差分隐私
#   kano        — /v1/privacy/k_anonymize/*   K-匿名化
#   qol         — /v1/privacy/qol/*           查询混淆
#   budget      — /v1/privacy/budget          隐私预算查询
#   profile     — /v1/privacy/profile         隐私参数 profile
#   file        — /v1/privacy/file/*          文件上传与处理
#   ops         — /v1/ops/diagnostics         运维诊断
#   medical     — /v1/medical/*               医疗数据流水线（deprecated）
#   agent_process — /v1/agent/*               通用数据评估与脱敏流水线（canonical）
#   dynclassification — /v1/classify/*        三层漏斗动态分类
from .routers import agent_process, budget, dp, dynclassification, file, health, kano, ldp, mask, medical, ops, profile, qol

# ASGI-level API Key middleware for mounted sub-apps (e.g. /metrics)
# ASGI 层 API Key 中间件，用于挂载的子应用（如 /metrics）
from .security.auth import ApiKeyAuthAsgiMiddleware
# Security settings loader — reads PRIVACY_AUTH_ENABLED etc. from environment
# 安全配置加载器 — 从环境变量读取 PRIVACY_AUTH_ENABLED 等配置
from .security.config import get_security_settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security response headers middleware.

    Injects common security headers into every HTTP response to defend against
    MIME sniffing, clickjacking, reflected XSS, protocol downgrade, referrer
    leakage, and unauthorized feature access attacks.

    安全响应头中间件。
    自动为所有 HTTP 响应注入通用安全响应头，防范 MIME 嗅探、点击劫持、反射型 XSS、
    协议降级、Referrer 泄露与未授权特性访问攻击。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Pass the request down the middleware chain and get the response
        # 将请求传递给下游中间件/路由处理函数，获取响应对象
        response: Response = await call_next(request)
        # Prevent browsers from MIME-type sniffing — forces browser to respect
        # the declared Content-Type, mitigating drive-by download attacks.
        # 禁止浏览器进行 MIME 类型嗅探 — 强制浏览器遵守声明的 Content-Type，
        # 防止“顺手牵羊”下载攻击（drive-by download）。
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Completely deny iframe embedding — prevents clickjacking by ensuring
        # the page cannot be rendered inside a <frame>/<iframe>.
        # 完全禁止 iframe 嵌入 — 通过确保页面无法在 <frame>/<iframe> 中渲染来
        # 防止点击劫持攻击。
        response.headers["X-Frame-Options"] = "DENY"
        # Enable browser built-in XSS filter — when a reflected XSS is detected,
        # block the page from rendering instead of sanitizing the script.
        # 启用浏览器内置 XSS 过滤器 — 当检测到反射型 XSS 时，阻止页面渲染而非
        # 尝试清理脚本代码（更安全）。
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # HSTS (HTTP Strict Transport Security) — instructs browsers to only
        # access the service over HTTPS, preventing protocol downgrade attacks.
        # max-age=31536000 (1 year); includeSubDomains extends to all subdomains.
        # HSTS（HTTP 严格传输安全）— 指示浏览器仅通过 HTTPS 访问服务，
        # 防止协议降级攻击。max-age=31536000（1 年）；includeSubDomains 扩展到所有子域名。
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Referrer-Policy — controls how much referrer information is sent with requests.
        # "strict-origin-when-cross-origin" sends full URL for same-origin requests
        # but only the origin (no path/query) for cross-origin requests.
        # Referrer-Policy — 控制请求携带的 Referrer 信息量。
        # "strict-origin-when-cross-origin" 同源请求发送完整 URL，
        # 跨域请求仅发送 origin（不含路径和查询参数），防止敏感 URL 参数泄露。
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Permissions-Policy — restricts access to browser features (camera, mic, etc.).
        # Deny all sensitive features by default for API services that don't need them.
        # Permissions-Policy — 限制浏览器特性（摄像头、麦克风等）的访问。
        # API 服务默认禁用所有敏感特性，缩小攻击面。
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


# Maximum request body size in bytes (default 32 MiB, aligned with Go services MaxBodySize).
# Configurable via PRIVACY_MAX_BODY_SIZE environment variable.
# 请求体最大字节数（默认 32 MiB，与 Go 服务 MaxBodySize 对齐）。
# 通过 PRIVACY_MAX_BODY_SIZE 环境变量可配置。
_MAX_BODY_SIZE = int(os.environ.get("PRIVACY_MAX_BODY_SIZE", str(32 * 1024 * 1024)))


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Request body size limit middleware — DoS protection.

    Checks the Content-Length header before the request body is read.
    Returns 413 Payload Too Large if the declared size exceeds the limit.
    This is a first line of defense; actual body reading is also bounded
    by the file upload size check in deps.py.

    请求体大小限制中间件 — DoS 防护。
    在读取请求体之前检查 Content-Length 头，超过限制时返回 413。
    这是第一道防线；实际 body 读取还受 deps.py 中文件上传大小检查约束。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > _MAX_BODY_SIZE:
                    return _build_error_envelope(
                        code="PAYLOAD_TOO_LARGE",
                        message=f"Request body exceeds {_MAX_BODY_SIZE} bytes limit",
                        detail=f"Content-Length {content_length} exceeds {_MAX_BODY_SIZE}",
                        request=request,
                        status_code=413,
                    )
            except ValueError:
                pass  # Non-integer Content-Length; let downstream handle
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Startup:  initialize structured logging and optional OpenTelemetry tracing.
    Shutdown: run cleanup logic (currently a no-op placeholder for future use).

    应用生命周期管理器。
    启动阶段：初始化结构化日志与可选的 OpenTelemetry 分布式追踪。
    关闭阶段：执行清理逻辑（当前为占位，供未来扩展如关闭连接池、刷新缓冲区等）。

    Args:
        app: FastAPI application instance / FastAPI 应用实例。
    """
    # --- Startup phase / 启动阶段 ---

    # Configure structured logging: reads format/level/service-name from env vars
    # so that log output is consistent across all modules from the very first line.
    # 配置结构化日志：从环境变量读取日志格式、级别、服务名，
    # 确保所有模块从第一行日志起就保持一致的输出格式。
    configure_logging(
        log_level=os.environ.get("PRIVACY_LOG_LEVEL", "INFO"),
        json_format=os.environ.get("PRIVACY_LOG_FORMAT", "text").lower() == "json",
        service_name=os.environ.get("PRIVACY_SERVICE_NAME", "PrivShield"),
    )

    # Initialize OpenTelemetry distributed tracing (optional).
    # If OTEL_EXPORTER_OTLP_ENDPOINT is not set, this is a no-op.
    # Service name falls back to PRIVACY_SERVICE_NAME for unified log/trace correlation.
    # 初始化 OpenTelemetry 分布式追踪（可选）。
    # 若未设置 OTEL_EXPORTER_OTLP_ENDPOINT 则为 no-op。
    # 服务名回退到 PRIVACY_SERVICE_NAME，以保持日志与追踪的统一关联。
    init_tracing(
        endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        service_name=os.environ.get(
            "OTEL_SERVICE_NAME",
            os.environ.get("PRIVACY_SERVICE_NAME", "PrivShield"),
        ),
    )

    # Build the IAM runtime when a user-store backend is explicitly configured.
    # Misconfiguration (unknown backend, missing sqlite file, pending schema,
    # production without an envelope key) must abort startup rather than fall
    # back to an ephemeral in-memory store.
    # 显式配置用户库后构建 IAM 运行时；任何配置错误都在启动期失败，绝不静默降级。
    if AuthSettings.from_env().enabled:
        get_auth_runtime()

    # --- Application running / 应用运行中 ---

    # Yield control back to the ASGI server (uvicorn); the app now serves requests.
    # 将控制权交还给 ASGI 服务器（uvicorn）；应用开始处理请求。
    try:
        yield
    finally:
        # --- Shutdown phase / 关闭阶段 ---
        # Log shutdown event for operational visibility.
        # 记录关闭事件，便于运维审计。
        _shutdown_logger = get_logger(__name__)
        _shutdown_logger.info("PrivShield Agent shutting down — flushing buffers and releasing resources")
        # Flush OpenTelemetry tracing buffers if initialized.
        # 若已初始化 OpenTelemetry 追踪，刷新缓冲区以确保所有 span 导出。
        try:
            from .observability.tracing import shutdown_tracing
            shutdown_tracing()
        except Exception:  # noqa: BLE001
            pass  # Best-effort; tracing may not be initialized / 尽力而为


# =============================================================================
# Phase 4: FastAPI application creation & middleware stack assembly
# Phase 4: FastAPI 应用创建与中间件栈装配
# =============================================================================

# Read auth setting once at import time to configure the app before it starts.
# When auth is enabled, disable /docs and /openapi.json to prevent unauthenticated
# callers from discovering the full API surface (attack surface reduction).
# 在导入时一次性读取认证配置，在应用启动前完成配置。
# 当认证启用时，禁用 /docs 和 /openapi.json，防止未认证调用方发现完整 API 表面
# （缩小攻击面）。
_auth_enabled = get_security_settings().auth_enabled
app = FastAPI(
    title="数盾 (PrivShield) 隐私治理边车",
    description="数联天下企业级数据隐私计算、多原语脱敏与三层动态分类分级治理服务 (PrivShield Privacy Governance Sidecar)",
    version="1.8.0",
    lifespan=lifespan,
    # docs_url=None disables Swagger UI; openapi_url=None disables the schema endpoint
    # docs_url=None 禁用 Swagger UI；openapi_url=None 禁用 schema 端点
    docs_url=None if _auth_enabled else "/docs",
    openapi_url=None if _auth_enabled else "/openapi.json",
)

# ---------------------------------------------------------------------------
# Global exception handler — defense-in-depth safety net
# ---------------------------------------------------------------------------
# 全局异常处理器 — 纵深防御安全网
# Some routers (budget, dynclassification, medical, ops, profile) do not wrap
# every handler in try/except. This catches any unhandled exception and maps it
# to a clean HTTP response via the same logic used by per-router handlers,
# preventing raw tracebacks from leaking to callers.
# 部分路由（budget、dynclassification、medical、ops、profile）并未在每个处理函数中
# 包裹 try/except。此全局处理器捕获所有未处理异常，通过相同逻辑映射为干净的 HTTP
# 响应，防止原始堆栈信息泄露给调用方。
app.add_exception_handler(Exception, handle_request_exception)


# ---------------------------------------------------------------------------
# 422 Validation error handler — unified error envelope format
# ---------------------------------------------------------------------------
# FastAPI 默认 422 响应包含完整字段名与校验器信息，会泄露内部 Pydantic 模型结构。
# 使用统一错误信封格式返回，仅在 PRIVACY_DEBUG_VALIDATION=true 时返回详细校验错误。
# Aligned with Go middleware.ErrorEnvelope (pkg/middleware/envelope.go).
_debug_validation = os.environ.get("PRIVACY_DEBUG_VALIDATION", "false").lower() == "true"
register_envelope_exception_handlers(app, debug_validation=_debug_validation)

# Auth-kernel errors carry the Go-style machine code; map them onto the same
# five-field envelope instead of letting the generic handler degrade them.
# IAM 内核错误保留 Go 侧机器码，统一映射为五字段信封。
register_auth_exception_handler(app)


# ---------------------------------------------------------------------------
# Middleware execution order (outermost -> innermost, i.e. request flow):
# ObservabilityMiddleware -> CORSMiddleware (opt-in) -> GZipMiddleware -> SecurityHeadersMiddleware -> router
#
# Note: Starlette add_middleware() prepends, so the LAST added runs FIRST on request.
# ---------------------------------------------------------------------------
# 中间件执行顺序（由外到内，即请求流向）：
# ObservabilityMiddleware -> CORSMiddleware（可选启用） -> GZipMiddleware -> SecurityHeadersMiddleware -> 路由处理
# 注意：Starlette add_middleware() 是前插的，因此最后添加的中间件在请求时最先执行。
# ---------------------------------------------------------------------------

# Security headers middleware — adds X-Content-Type-Options, X-Frame-Options, etc.
# Added first so it runs closest to the route handler (innermost layer).
# 安全响应头中间件 — 添加 X-Content-Type-Options、X-Frame-Options 等。
# 最先添加，使其最靠近路由处理函数（最内层）。
app.add_middleware(SecurityHeadersMiddleware)

# Request body size limit — rejects requests with Content-Length exceeding the threshold.
# DoS protection aligned with Go services' MaxBodySize middleware.
# 请求体大小限制 — 拒绝 Content-Length 超过阈值的请求，DoS 防护，与 Go 服务 MaxBodySize 对齐。
app.add_middleware(MaxBodySizeMiddleware)

# GZip response compression — reduces bandwidth for responses >= 1KB.
# minimum_size=1000 avoids compressing tiny responses where compression overhead
# would make them larger than the original.
# GZip 响应压缩 — 减少 >= 1KB 响应的网络传输开销。
# minimum_size=1000 避免压缩小型响应（压缩开销可能使压缩后反而更大）。
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware (opt-in via PRIVACY_CORS_ENABLED=true).
# Allows external frontends / SPAs to call the API from a different origin.
# Configure allowed origins via PRIVACY_CORS_ALLOW_ORIGINS (comma-separated).
# CORS 中间件（通过 PRIVACY_CORS_ENABLED=true 可选启用）。
# 允许外部前端 / SPA 从不同源调用 API。
# 通过 PRIVACY_CORS_ALLOW_ORIGINS（逗号分隔）配置允许的源。
_cors_enabled = os.environ.get("PRIVACY_CORS_ENABLED", "false").lower() == "true"
if _cors_enabled:
    _cors_origins_raw = os.environ.get("PRIVACY_CORS_ALLOW_ORIGINS", "*")
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    _cors_credentials = os.environ.get("PRIVACY_CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    # Security: CORS spec forbids Access-Control-Allow-Origin: * with credentials.
    # When wildcard origin is used, force credentials=False to prevent silent browser rejection.
    # 安全修复：CORS 规范禁止通配符 origin 与 credentials 同时使用。
    # 当允许源为 "*" 时，强制关闭 credentials，避免浏览器静默拒绝带凭证的请求。
    if "*" in _cors_origins and _cors_credentials:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "CORS misconfiguration detected: allow_origins contains '*' but "
            "allow_credentials=True. Per CORS spec, credentials are forbidden "
            "with wildcard origins. Forcing allow_credentials=False. "
            "Set PRIVACY_CORS_ALLOW_ORIGINS to specific origins to use credentials."
        )
        _cors_credentials = False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_cors_credentials,
        allow_methods=os.environ.get("PRIVACY_CORS_ALLOW_METHODS", "*"),
        allow_headers=os.environ.get("PRIVACY_CORS_ALLOW_HEADERS", "*"),
        expose_headers=os.environ.get("PRIVACY_CORS_EXPOSE_HEADERS", "x-request-id"),
        max_age=int(os.environ.get("PRIVACY_CORS_MAX_AGE", "600")),
    )

# Observability middleware (outermost layer):
#   1. Extracts/generates request_id for distributed tracing correlation
#   2. Records structured access logs (method, path, status, duration)
#   3. Updates Prometheus counters/histograms (request count, latency, traffic)
#   Note: /metrics path is excluded internally to avoid self-referencing loops.
# 可观测性中间件（最外层）：
#   1. 提取/生成 request_id，用于分布式追踪关联
#   2. 记录结构化访问日志（方法、路径、状态码、耗时）
#   3. 更新 Prometheus 计数器/直方图（请求数、延迟、流量）
#   注意：/metrics 路径在内部被排除，避免自引用循环。
app.add_middleware(ObservabilityMiddleware)

# ---------------------------------------------------------------------------
# Startup feature-flag banner — operator visibility
# ---------------------------------------------------------------------------
# 启动特性横幅 — 运维可见性
# Log key configuration flags at startup so operators can verify the security
# posture before the service begins accepting traffic.
_startup_logger = get_logger("engine.startup")
_startup_logger.info(
    "PrivShield Agent configuration: "
    "auth=%s | tls=%s | rate_limit=%s | cors=%s | ner=%s | llm=%s",
    _auth_enabled,
    get_security_settings().tls_enabled,
    get_security_settings().rate_limit_enabled,
    _cors_enabled,
    os.environ.get("PRIVACY_NER_ENABLE", "false").lower() == "true",
    os.environ.get("PRIVACY_LLM_ENABLE", "false").lower() == "true",
)

# ---------------------------------------------------------------------------
# Mounted sub-applications (ASGI mount, bypasses FastAPI dependency injection)
# 挂载的子应用（ASGI 挂载，绕过 FastAPI 依赖注入体系）
# ---------------------------------------------------------------------------

# Prometheus metrics endpoint at /metrics.
# Since app.mount() bypasses FastAPI's dependency system, we wrap the metrics
# ASGI app with ApiKeyAuthAsgiMiddleware to enforce API Key authentication at
# the ASGI layer when PRIVACY_AUTH_ENABLED=true. The middleware also enforces
# the "ops:diagnostics" scope, preventing low-privilege keys from accessing metrics.
# When auth is disabled, the middleware passes through transparently.
# Prometheus 指标端点 /metrics。
# 由于 app.mount() 绕过 FastAPI 依赖体系，因此用 ApiKeyAuthAsgiMiddleware
# 包裹指标 ASGI 应用，在 ASGI 层强制 API Key 认证（PRIVACY_AUTH_ENABLED=true 时）。
# 该中间件同时强制检查 "ops:diagnostics" scope（与 /v1/ops/* 同级），防止低权限 Key 访问指标。
# 认证未启用时，中间件透明透传。
app.mount("/metrics", ApiKeyAuthAsgiMiddleware(make_asgi_app()))

# ---------------------------------------------------------------------------
# Route registration — include domain-specific routers
# 路由注册 — 挂载按域拆分的子路由
# ---------------------------------------------------------------------------

# Dynamic classification router: 3-layer funnel (Rule -> NER -> LLM)
# for field/table/image data classification and security tagging.
# 动态分类路由：三层漏斗（规则 -> NER -> LLM），
# 用于字段/表/图片数据分类分级与安全标记。
app.include_router(dynclassification.router)

# Privacy primitives & operational routers:
#   health    — K8s liveness/readiness probes (/v1/health, /v1/readyz)
#   mask      — Field-name-aware PII masking (/v1/privacy/mask, /v1/privacy/mask_record)
#   dp        — Differential privacy: Laplace/Gaussian count/sum/mean (/v1/privacy/dp/*)
#   ldp       — Local differential privacy (/v1/privacy/ldp/*)
#   kano      — K-anonymity: per-record & dataset-level (/v1/privacy/k_anonymize/*)
#   qol       — Query obfuscation: dummy query injection (/v1/privacy/qol/*)
#   budget    — Privacy budget accounting & tracking (/v1/privacy/budget)
#   profile   — Privacy parameter profile management (/v1/privacy/profile)
#   file      — File upload, parsing & redaction (/v1/privacy/file/*)
#   ops       — Operational diagnostics & engine status (/v1/ops/diagnostics)
#   medical   — Medical data processing pipeline (/v1/medical/*)
#   pipeline  — Combined multi-step classification + privacy pipeline (/v1/pipeline/*)
# 隐私原语与运维路由：
#   health    — K8s 存活/就绪探针
#   mask      — 字段名感知的 PII 脱敏
#   dp        — 差分隐私：Laplace/Gaussian 计数/求和/均值
#   ldp       — 本地差分隐私
#   kano      — K-匿名：记录级与数据集级
#   qol       — 查询混淆：虚拟查询注入
#   budget    — 隐私预算核算与追踪
#   profile   — 隐私参数 profile 管理
#   file      — 文件上传、解析与打码
#   ops       — 运维诊断与引擎状态
#   medical   — 医疗数据处理流水线
#   pipeline  — 多步分类 + 隐私处理组合流水线
app.include_router(health.router)
app.include_router(mask.router)
app.include_router(dp.router)
app.include_router(ldp.router)
app.include_router(kano.router)
app.include_router(qol.router)
app.include_router(budget.router)
app.include_router(profile.router)
app.include_router(file.router)
app.include_router(ops.router)
app.include_router(ops.root_router)
app.include_router(agent_process.router)
app.include_router(medical.router)
app.include_router(pipeline_router)

# Auth kernel REST surface (Go ``pkg/auth.RegisterUserRoutes``): only mounted when
# a user-store backend is explicitly configured. Public paths are resolved by the
# shared identity dependency; handler-level ABAC lives in ``engine.auth.service``.
# 用户与权限管理面仅在显式配置用户库后挂载；公开路径由共享身份依赖放行。
_auth_user_store_enabled = AuthSettings.from_env().enabled
if _auth_user_store_enabled:
    app.include_router(auth_router)
    app.include_router(auth_well_known_router)


# ---------------------------------------------------------------------------
# OpenAPI 规范别名（Go 契约：/openapi.json 与 /docs/openapi.json）
# 认证启用时由依赖强制 ops:diagnostics 权限，保持与 Go 相同的暴露范围。
# ---------------------------------------------------------------------------
_OPENAPI_DEPS = [*SECURITY_DEPS, require_permission("ops:diagnostics")]


@app.get("/docs/openapi.json", include_in_schema=False, dependencies=_OPENAPI_DEPS)
def openapi_docs_alias():
    return app.openapi()


if _auth_enabled:

    @app.get("/openapi.json", include_in_schema=False, dependencies=_OPENAPI_DEPS)
    def openapi_alias():
        return app.openapi()


# =============================================================================
# Phase 5: Standalone entry point — ``python -m engine.main``
# Phase 5: 独立入口 — ``python -m engine.main``
# =============================================================================

# This block only executes when the file is run directly (not when imported).
# It parses CLI arguments and starts the Uvicorn ASGI server.
# Note: this entry point binds to 127.0.0.1 by default (local dev only).
# For production, use ``python -m engine.server`` which binds 0.0.0.0.
# 该代码块仅在直接运行文件时执行（导入时不执行）。
# 解析命令行参数并启动 Uvicorn ASGI 服务器。
# 注意：此入口默认绑定 127.0.0.1（仅本地开发）。
# 生产环境请使用 ``python -m engine.server``（默认绑定 0.0.0.0）。

if __name__ == "__main__":
    import argparse

    import uvicorn

    # CLI argument parser for standalone REST server mode
    # 独立 REST 服务器模式的命令行参数解析器
    parser = argparse.ArgumentParser(
        prog="engine.main",
        description="SecretFlow Local Privacy Agent REST server.",
    )
    # Bind to loopback only — safe default for local development
    # 仅绑定回环地址 — 本地开发的安全默认值
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="REST server host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8079,
        help="REST server port (default: 8079).",
    )
    # Parse CLI args and start the ASGI server with the configured app
    # 解析命令行参数，使用配置好的 app 启动 ASGI 服务器
    args = parser.parse_args()

    import os as _os

    # Production-hardened uvicorn configuration (aligned with server.py).
    # 生产加固的 uvicorn 配置（与 server.py 对齐）。
    # Even in standalone REST mode, apply timeout/concurrency limits
    # to prevent Slowloris DoS and unbounded resource consumption.
    # 即使独立 REST 模式也应用超时/并发限制，
    # 防止 Slowloris DoS 与无限制资源消耗。
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        limit_concurrency=int(_os.environ.get("PRIVACY_LIMIT_CONCURRENCY", "10000")),
        limit_max_requests=int(_os.environ.get("PRIVACY_LIMIT_MAX_REQUESTS", "100000")),
        timeout_keep_alive=int(_os.environ.get("PRIVACY_TIMEOUT_KEEP_ALIVE", "30")),
        timeout_graceful_shutdown=int(_os.environ.get("PRIVACY_TIMEOUT_GRACEFUL_SHUTDOWN", "10")),
    )
