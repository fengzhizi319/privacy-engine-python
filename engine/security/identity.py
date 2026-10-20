"""Identity model and endpoint-permission mappings.

定义调用者身份、服务类型、权限 scope，以及 REST 路径 / gRPC 方法到权限的映射。
Defines the caller Identity, service types, scopes, and permission mapping for REST
paths and gRPC methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 审计 scope 前缀：通配符 "*" 对审计权限显式失效（三权分立），与 Go
# pkg/auth.Identity.HasPermission 同口径。
AUDIT_SCOPE_PREFIX = "audit:"


@dataclass(frozen=True)
class Identity:
    """Authenticated caller identity.

    Attributes:
        service_type: "internal" for high-trust services, "external" for untrusted/public clients.
        name: Service or account name, used in logs and rate-limit keys.
        scopes: Granted permission strings. ["*"] means full access.
        subject: 责任主体（用户名）；仅登录会话与用户动态 Key 携带，静态服务密钥为空。
        role: 角色标识（admin/auditor/consumer/viewer/service_account）。
        principal_type: "user" 或 "workload"。
        tenant_id: 租户空间标识。
        mfa_authenticated: 该会话建立时是否已完成 TOTP 校验。
        session_auth: 身份是否来自交互式登录会话（而非 API Key / mTLS）。
        mfa_enabled: 账号当前是否已绑定 TOTP。
        allowed_audiences: 动态 Key 的受众白名单。
        audience: 凭据声明的目标受众。
        key_id: 动态 API Key 标识。
        cert_fingerprint: mTLS 证书指纹反查命中的账号指纹。
        binding_status: 公钥绑定状态（active / overlapping / drain_only / revoked）。
        binding_mode: 绑定档位（Go 目标态只有 required；空表示未声明）。
        bound_cn: 凭据绑定的客户端证书 CN（Go ``BoundCN``）。
        bound_cert_fingerprint: 凭据钉扎的叶证书 SHA-256 指纹（Go ``BoundCertFingerprint``）。
        bound_public_key_der: 绑定的原始 SPKI DER（base64，Go ``BoundPublicKeyDER``）。
        contract_version: 凭据登记的契约版本。
        contract_allowed_paths: 凭据登记的「METHOD /path」允许列表。
    """

    service_type: Literal["internal", "external", "public"]
    name: str
    scopes: list[str]
    subject: str = ""
    role: str = ""
    principal_type: str = ""
    tenant_id: str = ""
    mfa_authenticated: bool = False
    session_auth: bool = False
    mfa_enabled: bool = False
    allowed_audiences: list[str] = field(default_factory=list)
    audience: str = ""
    key_id: str = ""
    cert_fingerprint: str = ""
    binding_status: str = ""
    binding_mode: str = ""
    bound_cn: str = ""
    bound_cert_fingerprint: str = ""
    bound_public_key_der: str = ""
    contract_version: str = ""
    contract_allowed_paths: list[str] = field(default_factory=list)

    def has_credential_bindings(self) -> bool:
        """True when the credential declares any client-binding constraint."""
        return bool(self.bound_cn or self.bound_cert_fingerprint or self.bound_public_key_der)

    def has_permission(self, permission: str) -> bool:
        """Check whether this identity is allowed to perform ``permission``.

        通配符 "*" 授予全部**业务**权限，但显式不穿透 ``audit:`` 前缀
        （等保三级三权分立：系统管理员不得查看/篡改审计记录）。
        Scope matching is exact; no globbing or hierarchy is supported at P0.
        ``"a|b"`` 复合权限按逻辑或拆分，与 Go ``Identity.HasPermission`` 同口径。
        """
        if "|" in permission:
            return any(
                part.strip() and self.has_permission(part.strip())
                for part in permission.split("|")
            )
        if permission.startswith(AUDIT_SCOPE_PREFIX):
            return permission in self.scopes or "audit:*" in self.scopes
        return "*" in self.scopes or permission in self.scopes

    def is_subject(self, username: str) -> bool:
        """True 当且仅当本身份的责任主体就是 ``username``（空值绝不匹配）。"""
        if not self.subject:
            return False
        target = username.strip().lower()
        return bool(target) and self.subject.strip().lower() == target

    def matches_audience(self, target_audience: str) -> bool:
        """Go ``Identity.MatchesAudience``：受众白名单优先于单一受众声明。"""
        target = target_audience.strip()
        if not target:
            return False
        if self.allowed_audiences:
            return "*" in self.allowed_audiences or target in self.allowed_audiences
        if not self.audience:
            return False
        return self.audience in {"*", target}


# Default identity used when authentication is enabled but no credentials were
# presented and the path is not public: health probes only (Go AnonymousIdentity).
ANONYMOUS_IDENTITY = Identity("external", "anonymous", ["health:read"])

# Authentication disabled: every route is reachable. Go only skips route-level
# scope checks in that mode; FastAPI enforces scopes per route dependency, so the
# dev identity must carry the wildcard explicitly (Go DevModeIdentity equivalent).
DEV_MODE_IDENTITY = Identity("internal", "dev-anonymous", ["*", "admin"])


def permission_for_rest_path(path: str) -> str:
    """Map a REST path to the required permission string.

    .. note::
        **参考映射，当前未接线**：REST 路由均在 ``routers/*`` 中通过
        ``require_permission(...)`` 显式声明权限，本函数仅供文档/示例参考
        （见 ``docs/production_security/``），路由变更时需手动保持同步。
        gRPC 侧权限映射 ``permission_for_grpc_method`` 则是在接线状态
        （``AuthInterceptor`` 使用）。

    Paths are matched by prefix where it makes sense. Unknown paths require a
    generic wildcard permission for safety.
    """
    path = path.rstrip("/")
    if path in ("/health", "/livez", "/readyz", "/readyz/llm"):
        return "health:read"
    if path.startswith("/v1/privacy/mask"):
        # /v1/privacy/mask、/mask_record、/mask/batch、/mask/dataframe
        return "privacy:mask"
    if path == "/v1/privacy/hash":
        return "privacy:hash"
    if path.startswith("/v1/privacy/dp/") or path.startswith("/v1/privacy/ldp/"):
        return "privacy:dp"
    if path.startswith("/v1/privacy/k_anonymize"):
        return "privacy:kano"
    if path.startswith("/v1/privacy/qol/"):
        # /v1/privacy/qol/obfuscate、/qol/obfuscate/batch
        return "privacy:qol"
    if path == "/v1/privacy/budget":
        return "privacy:budget"
    if path == "/v1/privacy/profile/recommend":
        return "privacy:profile"
    if path == "/v1/privacy/process_file":
        return "privacy:mask"
    if path.startswith("/v1/privacy/classify/"):
        return "classification:read"
    if path.startswith("/v1/dynclassification"):
        # 写操作（热加载 / 生成配置）需要 write 权限，其余为只读。
        if path in (
            "/v1/dynclassification/profiles/reload",
            "/v1/dynclassification/generate_profile",
        ):
            return "dynclassification:write"
    if path.startswith("/v1/agent"):
        return "agent:process"
    if path.startswith("/v1/medical"):
        return "medical:process"
    if path.startswith("/v1/pipeline"):
        return "pipeline:process"
    if path.startswith("/v1/ops/"):
        # refresh 触发模型重载需更高权限 ops:admin（在端点内校验）。
        return "ops:diagnostics"
    # Conservative default for unknown routes.
    return "*"


def permission_for_grpc_method(method: str) -> str:
    """Map a gRPC full method name (e.g. '/privacy.local.PrivacyService/Mask') to a permission."""
    # Strip the package/service prefix if present.
    short = method.split("/")[-1] if "/" in method else method
    mapping = {
        "Mask": "privacy:mask",
        "MaskRecord": "privacy:mask",
        "MaskBatch": "privacy:mask",
        "MaskDataFrame": "privacy:mask",
        "Hash": "privacy:hash",
        "DPCount": "privacy:dp",
        "DPSum": "privacy:dp",
        "DPMean": "privacy:dp",
        "DPHistogram": "privacy:dp",
        "DPNoisyCount": "privacy:dp",
        "DPNoisySum": "privacy:dp",
        "DPNoisyMean": "privacy:dp",
        "DPNoisyHistogram": "privacy:dp",
        "DPChunkedCount": "privacy:dp",
        "DPChunkedSum": "privacy:dp",
        "DPChunkedMean": "privacy:dp",
        "DPChunkedHistogram": "privacy:dp",
        "DPAggregate": "privacy:dp",
        "DPVectorSum": "privacy:dp",
        "DPAdaptiveClip": "privacy:dp",
        "DPGroupBy": "privacy:dp",
        # 本地 DP（LDP）扰动与估计复用 privacy:dp 权限，与 REST 路由一致。
        "PerturbBinaryBatch": "privacy:dp",
        "PerturbCategoricalBatch": "privacy:dp",
        "EstimateBinaryFrequency": "privacy:dp",
        "EstimateCategoricalHistogram": "privacy:dp",
        "KAnonymizeRecord": "privacy:kano",
        "KAnonymizeTable": "privacy:kano",
        "KAnonymizeDataFrame": "privacy:kano",
        "ObfuscateQuery": "privacy:qol",
        "ObfuscateQueryBatch": "privacy:qol",
        "ClassifyField": "classification:read",
        "ClassifyRecord": "classification:read",
        "ClassifyTable": "classification:read",
        "DynClassify": "dynclassification:read",
        "Health": "health:read",
        "RecommendParams": "privacy:profile",
    }
    return mapping.get(short, "*")


def is_standard_health_path(path: str) -> bool:
    """Go ``IsStandardHealthPath`` 的严格白名单，供 REST 探针豁免使用。"""
    return path in (
        "/health",
        "/healthz",
        "/livez",
        "/readyz",
        "/readyz/llm",
        "/v1/privacy/health",
    )


_GRPC_HEALTH_METHODS = frozenset(
    {
        "grpc.health.v1.Health/Check",
        "grpc.health.v1.Health/Watch",
        "privacy.local.PrivacyService/Health",
        "servicehub.ServiceHubService/Health",
        "datasourcemgr.DataSourceManagerService/Health",
    }
)


def is_grpc_health_method(method: str) -> bool:
    return method.strip().lstrip("/") in _GRPC_HEALTH_METHODS


def is_health_path_or_method(path_or_method: str) -> bool:
    """Return True if the given REST path or gRPC method is a health probe."""
    normalized = path_or_method.rstrip("/")
    return normalized in ("/health", "/healthz", "/livez", "/readyz", "/readyz/llm", "/v1/privacy/health") or is_grpc_health_method(
        path_or_method
    )


# Go ``IsAuthPublicPath``: login / public self-registration / bootstrap /
# workload activation / public capability discovery. Matching strips a single
# trailing slash only (no prefix or suffix semantics).
AUTH_PUBLIC_PATHS = frozenset(
    {
        "/v1/auth/login",
        "/v1/auth/register",
        "/v1/auth/users/register",
        "/v1/auth/users/bootstrap",
        "/v1/auth/service-accounts/activate",
        "/v1/.well-known/service-capabilities/public",
    }
)


def is_auth_public_path(path: str) -> bool:
    normalized = path.rstrip("/") if len(path) > 1 else path
    return normalized in AUTH_PUBLIC_PATHS
