"""gRPC 服务入口模块。

基于 grpcio 与自动生成的 protobuf stub 实现 PrivacyService 的 gRPC 接口，
暴露与 REST 模块相对应的处理原语能力：脱敏、哈希、差分隐私、K-匿名、查询混淆与健康检查。

gRPC service entrypoint. Implements the protobuf-defined PrivacyService interface
using generated stubs and the shared PrivacyService business layer for processing
primitives.

执行流程 / Execution Flow:
```mermaid
graph TD
    A[模块加载] --> B[读取环境变量: PROFILE_PATH / NAMESPACE]
    B --> C[定义 _grpc_error_mapper 异常映射装饰器]
    C --> D[定义 PrivacyServicer 类: 封装所有 RPC 方法]
    D --> E[遍历公共方法, 统一包装异常映射]
    E --> F[serve 函数: 初始化日志/链路追踪]
    F --> G[构建拦截器链: 可观测性/认证/限流]
    G --> H[创建 grpc.server 并注册 Servicer]
    H --> I{TLS 启用?}
    I -->|是| J[add_secure_port 安全端口]
    I -->|否| K[add_insecure_port 非安全端口]
    J --> L[server.start 启动服务]
    K --> L
    L --> M[wait_for_termination 阻塞等待]
```
"""

# ─── 标准库导入 / Standard library imports ───
from .env_loader import load_env_file
load_env_file()

import os  # 用于读取环境变量配置（profile路径、命名空间、日志级别等）
import sys  # 用于 sys.platform 平台检测（Windows 兼容性）
from concurrent import futures  # 提供 ThreadPoolExecutor，作为 gRPC 服务器的工作线程池

# ─── 第三方库导入 / Third-party imports ───
import grpc  # gRPC Python 核心库，提供服务器创建、状态码、拦截器等基础设施

# ─── 项目内部模块导入 / Internal project imports ───
# 导入 protobuf 自动生成的消息类与服务桩代码
from . import privacy_pb2, privacy_pb2_grpc
# 导入日志配置工具：configure_logging 初始化全局日志，get_logger 获取模块级 logger
from .observability.logging_config import configure_logging, get_logger
# 导入 gRPC 可观测性拦截器：记录请求耗时、状态码等指标
from .observability.middleware import GrpcObservabilityInterceptor
# 导入 OpenTelemetry 链路追踪初始化函数
from .observability.tracing import init_tracing
# 导入隐私预算耗尽异常，用于在错误映射器中返回 RESOURCE_EXHAUSTED 状态码
from .privacy.budget import PrivacyBudgetExhausted
# 导入认证拦截器：校验请求中的 API Key
from .security.auth import AuthInterceptor
# 导入安全配置获取函数：读取 TLS/认证/限流等开关与参数
from .security.config import get_security_settings
# 导入速率限制拦截器：基于令牌桶或滑动窗口限流
from .security.ratelimit import RateLimitInterceptor
# 导入 TLS 凭证构建函数：根据配置加载证书并生成 gRPC ServerCredentials
from .security.tls import grpc_server_credentials
# 导入核心业务层 PrivacyService：所有隐私原语的统一入口
from .service import PrivacyService

import time as _time_mod  # noqa: E401  (用于 _extract_trace_id 回退生成)

# ─── 全局配置常量 / Global configuration constants ───
# 与 REST 模块共享环境变量配置，确保两种协议使用同一 profile 与命名空间
# Share env-var config with the REST module so both protocols use the same profile & namespace
PROFILE_PATH = os.environ.get("PRIVACY_PROFILE", "privacy-profile.yaml")  # 隐私参数 profile 文件路径
NAMESPACE = os.environ.get("PRIVACY_NAMESPACE", "default")  # 隐私预算命名空间，用于多租户隔离

# 获取当前模块的 logger 实例，日志名称为 "engine.grpc_server"
logger = get_logger(__name__)


def _extract_trace_id(context: grpc.ServicerContext) -> str:
    """Extract distributed trace ID from gRPC incoming metadata.

    从 gRPC 入站 metadata 中提取分布式追踪 ID。
    优先查找 x-request-id，其次 x-trace-id，均不存在时自动生成。

    与 bff-go WithTrace() 及 pkg/middleware/trace.go 的追踪上下文对齐，
    确保 HTTP → gRPC 跨协议全链路追踪不断链。

    Args:
        context: gRPC servicer context / gRPC 服务端 context

    Returns:
        Trace ID string / 追踪 ID 字符串
    """
    metadata = []
    if context is not None:
        metadata = context.invocation_metadata() or []
    for key, val in metadata:
        if key in ("x-request-id", "x-trace-id"):
            return val
    return f"req-{int(_time_mod.time())}"


def _level_rank(level: str) -> int:
    """提取等级 ID 中的数字部分用于比较（L1..L5 / C4 等）。"""
    digits = "".join(ch for ch in (level or "") if ch.isdigit())
    return int(digits) if digits else 0


def _format_grpc_value(value) -> str:
    """将记录值转为与 Go FormatRecordValue 一致的字符串。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _grpc_error_mapper(fn):
    """将 gRPC 方法异常映射到语义化 gRPC 状态码，避免全部返回 UNKNOWN。

    Map exceptions raised inside gRPC handler methods to semantically correct
    gRPC status codes, preventing all errors from surfacing as UNKNOWN.

    映射规则 / Mapping rules:
        PrivacyBudgetExhausted → RESOURCE_EXHAUSTED  (隐私预算耗尽)
        ValueError             → INVALID_ARGUMENT    (参数校验失败)
        其他 Exception         → INTERNAL            (未预期内部错误)

    集成追踪上下文：从 gRPC metadata 提取 trace_id 并注入结构化日志，
    确保 gRPC 入站请求与 REST 入站请求使用相同的追踪 ID 格式。

    Args:
        fn: 被装饰的 gRPC 方法（签名为 (self, request, context)）。

    Returns:
        包装后的函数，捕获异常并通过 context.set_code / set_details 设置错误信息。
    """
    def wrapper(self, request, context):
        trace_id = _extract_trace_id(context)
        try:
            # 正常执行被装饰的 RPC 方法并返回 protobuf 响应
            return fn(self, request, context)
        except PrivacyBudgetExhausted as e:
            # 隐私预算耗尽：返回 RESOURCE_EXHAUSTED，客户端可据此触发预算重置或降级
            logger.warning("Privacy budget exhausted", extra={"trace_id": trace_id})
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Privacy budget exhausted")  # Fixed message — avoid leaking internal state
        except ValueError as e:
            # 参数校验失败（如 epsilon<=0、空数据等）：返回 INVALID_ARGUMENT
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid request parameters")  # Fixed message — avoid leaking internals
            logger.debug("gRPC ValueError", extra={"error": str(e), "trace_id": trace_id})
        except Exception:
            # 未预期异常：记录完整堆栈到日志，向客户端仅暴露通用错误信息，避免泄露内部实现
            logger.exception("grpc_request_error", extra={"trace_id": trace_id})
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")  # 通用错误描述，不含敏感信息
    return wrapper


class PrivacyServicer(privacy_pb2_grpc.PrivacyServiceServicer):
    """PrivacyService gRPC 服务实现。

    将 protobuf 请求转换为 PrivacyService 业务方法调用，
    并将结果封装为 protobuf 响应返回给客户端。

    Translates incoming protobuf requests into PrivacyService business-layer
    invocations and wraps results back into protobuf response messages.

    Attributes:
        service: 共享的 PrivacyService 业务实例（默认命名空间）。
                 Shared PrivacyService instance for the default namespace.
        _service_cache: 按命名空间缓存的 PrivacyService 实例字典，
                        避免为同一命名空间重复创建实例。
                        Namespace-keyed cache of PrivacyService instances.
    """

    def __init__(self):
        """初始化 gRPC servicer，创建 PrivacyService 实例并复用它。

        Initialize the servicer: create a default-namespace PrivacyService
        and seed the namespace cache with it.
        """
        # 创建默认命名空间的 PrivacyService 实例，加载 profile 配置与预算管理器
        self.service = PrivacyService(profile_path=PROFILE_PATH, namespace=NAMESPACE)
        # 以命名空间为 key 缓存实例，后续 RecommendParams 等方法可按需创建其他命名空间的实例
        self._service_cache = {NAMESPACE: self.service}

    # ─── 脱敏类 RPC 方法 / Masking RPC methods ───

    def Mask(self, request, context):
        """单字段脱敏 gRPC 方法。

        Mask a single field value based on field-name-aware rules.

        Args:
            request: MaskRequest，包含 field_name（字段名）、value（原始值）、context（上下文）。
            context: gRPC 服务上下文，用于设置状态码与元数据。

        Returns:
            MaskResponse: 包含脱敏后的 result 字符串。
        """
        # 调用业务层 mask() 执行脱敏，传入字段名、原始值和可选上下文
        # 将脱敏结果封装为 MaskResponse protobuf 消息返回
        return privacy_pb2.MaskResponse(result=self.service.mask(request.field_name, request.value, request.context))

    def MaskRecord(self, request, context):
        """整记录脱敏 gRPC 方法。

        Mask all sensitive fields within a single record (dict).

        Args:
            request: MaskRecordRequest，包含 record（map<string,string> 记录）与 context。
            context: gRPC 服务上下文。

        Returns:
            MaskRecordResponse: 包含脱敏后的 result 字典。
        """
        # 将 protobuf map 转换为 Python dict，调用业务层整记录脱敏
        result = self.service.mask_record(dict(request.record), request.context)
        # 封装为 MaskRecordResponse 返回
        return privacy_pb2.MaskRecordResponse(result=result)

    def MaskBatch(self, request, context):
        """批量字段脱敏 gRPC 方法。

        Mask multiple field-value pairs in a single batch call.

        Args:
            request: MaskBatchRequest，包含 field_names（字段名列表）、values（值列表）、context。
            context: gRPC 服务上下文。

        Returns:
            MaskBatchResponse: 包含脱敏后的 results 列表。
        """
        # 将 protobuf repeated 字段转为 Python list，调用业务层批量脱敏
        results = self.service.mask_batch(
            list(request.field_names), list(request.values), request.context
        )
        # 封装为 MaskBatchResponse 返回
        return privacy_pb2.MaskBatchResponse(results=results)

    def MaskDataFrame(self, request, context):
        """DataFrame 脱敏 gRPC 方法。

        Mask every record row-by-row. Matches Go TypedServer.MaskDataFrame,
        which ignores ``columns``/``context`` and applies MaskRecord per row.

        Args:
            request: MaskDataFrameRequest，包含 data（RecordEntry 列表）、columns（可选目标列）、context。
            context: gRPC 服务上下文。

        Returns:
            MaskDataFrameResponse: 包含脱敏后的 data（RecordEntry 列表）。
        """
        # Go typed_server.go 逐行调用 MaskRecord，不区分 columns；
        # Python 保持相同语义，避免同一 RPC 两端结果不一致。
        rows = [
            privacy_pb2.RecordEntry(fields=self.service.mask_record(dict(row.fields)))
            for row in request.data
        ]
        return privacy_pb2.MaskDataFrameResponse(data=rows)

    # ─── 哈希 RPC 方法 / Hashing RPC method ───

    def Hash(self, request, context):
        """HMAC 哈希 gRPC 方法。

        Compute an HMAC-SHA256 hash of the given value with a caller-provided salt.

        Args:
            request: HashRequest，包含 value（待哈希值）与 salt（HMAC 盐值）。
            context: gRPC 服务上下文。

        Returns:
            HashResponse: 包含十六进制哈希字符串 result。
        """
        # Go gRPC Hash 走 svc.HashSM3（SM3 十六进制前 16 位），与 REST /hash 的
        # HMAC-SM3 base64 不同；此处必须保持一致，不能复用 service.hash()。
        return privacy_pb2.HashResponse(
            result=self.service.hash_sm3(request.value, request.salt)
        )

    # ─── 差分隐私辅助方法 / Differential Privacy helper methods ───

    def _dp_params_from_request(self, request) -> dict[str, object]:
        """从 DPRequest 构建参数字典。

        Extract DP parameters (epsilon, mechanism, delta, clip bounds) from a
        standard DPRequest protobuf message into a plain dict for the business layer.

        Args:
            request: DPRequest protobuf 消息。

        Returns:
            包含 epsilon、mechanism 及可选 delta/clip 参数的字典。
        """
        # 基础必填参数：隐私预算 epsilon 与噪声机制（laplace/gaussian）
        params: dict[str, object] = {
            "epsilon": request.epsilon,      # 隐私预算上限
            "mechanism": request.mechanism,  # 噪声机制名称
        }
        # proto3 默认值：delta=0.0, clip_lower/upper=0.0。仅当非零或显式设置时透传。
        # proto3 scalar fields default to 0; only forward delta when explicitly set (non-zero).
        if request.delta != 0.0:
            params["delta"] = request.delta  # Gaussian 机制所需的 δ 参数
        # 截断边界：仅当至少一个边界非零时才传递，避免误将默认 0 当作有效截断
        if request.clip_lower != 0.0 or request.clip_upper != 0.0:
            params["clip_lower"] = request.clip_lower  # 截断下界
            params["clip_upper"] = request.clip_upper  # 截断上界
        return params

    # ─── 差分隐私聚合 RPC 方法 / DP aggregate RPC methods ───

    def DPCount(self, request, context):
        """差分隐私计数 gRPC 方法。

        Compute a differentially private count over raw values.

        Args:
            request: DPRequest，包含 values（原始数据列表）与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPResponse: 包含加噪后的计数结果 result。
        """
        # 将 repeated values 转为 list，提取 DP 参数，调用业务层执行加噪计数
        result = self.service.dp_count(list(request.values), self._dp_params_from_request(request))
        return privacy_pb2.DPResponse(result=result)  # 封装为统一 DPResponse 返回

    def DPSum(self, request, context):
        """差分隐私求和 gRPC 方法。

        Compute a differentially private sum over raw values.

        Args:
            request: DPRequest，包含 values 与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPResponse: 包含加噪后的求和结果。
        """
        # 调用业务层 dp_sum，内部会执行截断 + 加噪 + 预算扣减
        result = self.service.dp_sum(list(request.values), self._dp_params_from_request(request))
        return privacy_pb2.DPResponse(result=result)

    def DPMean(self, request, context):
        """差分隐私均值 gRPC 方法。

        Compute a differentially private mean over raw values.

        Args:
            request: DPRequest，包含 values 与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPResponse: 包含加噪后的均值结果。
        """
        # 调用业务层 dp_mean，内部拆分为 noisy_sum / noisy_count 再求商
        result = self.service.dp_mean(list(request.values), self._dp_params_from_request(request))
        return privacy_pb2.DPResponse(result=result)

    def DPHistogram(self, request, context):
        """差分隐私直方图 gRPC 方法。

        Compute a differentially private histogram over categorical values.

        Args:
            request: DPHistogramRequest，包含 values、categories（类别列表）与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPHistogramResponse: 包含 map<string, double> 形式的加噪直方图。
        """
        # 直方图请求单独构建参数（无 clip 边界，仅需 epsilon/mechanism/delta）
        params = {
            "epsilon": request.epsilon,      # 隐私预算
            "mechanism": request.mechanism,  # 噪声机制
        }
        # 仅在显式设置 delta 时透传（Gaussian 机制需要）
        if request.delta != 0.0:
            params["delta"] = request.delta
        # 调用业务层直方图加噪：对每个类别的计数独立加噪
        res_dict = self.service.dp_histogram(
            list(request.values), list(request.categories), params
        )
        # 类型断言：确保业务层返回字典类型
        assert isinstance(res_dict, dict)
        # 将 key 转为 str、value 转为 float，以契合 protobuf map<string, double> 类型约束
        result = {str(k): float(v) for k, v in res_dict.items()}
        return privacy_pb2.DPHistogramResponse(result=result)

    # ─── Noisify（对已聚合结果加噪）辅助方法 / Noisify helper ───

    def _dp_params_from_noisy_request(self, request) -> dict[str, object]:
        """从 noisify 请求构建参数字典。

        Extract DP parameters from a NoisyXxxRequest protobuf message.
        Compared to _dp_params_from_request, this also handles sensitivity,
        clip bounds, and min_count fields used by the noisify endpoints.

        Args:
            request: NoisyCount/NoisySum/NoisyMean/NoisyHistogram 等请求消息。

        Returns:
            包含 epsilon、mechanism 及可选 delta/sensitivity/clip/min_count 的字典。
        """
        # 基础参数：隐私预算与噪声机制
        params: dict[str, object] = {
            "epsilon": request.epsilon,      # 本次查询的隐私预算
            "mechanism": request.mechanism,  # laplace 或 gaussian
        }
        # 可选 delta：Gaussian 机制必须，Laplace 机制可省略
        if request.delta != 0.0:
            params["delta"] = request.delta
        # proto3 默认值问题：sensitivity=0 视为未提供，依赖 clip 边界推导
        # proto3 defaults sensitivity to 0; treat 0 as "not provided" and let business layer infer from clip bounds
        if getattr(request, "sensitivity", 0.0) != 0.0:
            params["sensitivity"] = request.sensitivity  # 查询灵敏度（全局敏感度）
        # 截断边界：用于在加噪前对数据执行截断，控制灵敏度上界
        if getattr(request, "clip_lower", 0.0) != 0.0 or getattr(request, "clip_upper", 0.0) != 0.0:
            params["clip_lower"] = request.clip_lower  # 截断下界
            params["clip_upper"] = request.clip_upper  # 截断上界
        # 最小计数阈值：低于此值的类别将被抑制，防止小样本泄露
        if getattr(request, "min_count", 0.0) != 0.0:
            params["min_count"] = request.min_count
        return params

    # ─── Noisify RPC 方法（对已聚合值加噪） / Noisify RPC methods ───

    def DPNoisyCount(self, request, context):
        """对已聚合计数加噪的 gRPC 方法。

        Add DP noise to a pre-aggregated true count (Sidecar/Noisify pattern).

        Args:
            request: NoisyCountRequest，包含 true_count（真实计数）与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPResponse: 加噪后的计数结果。
        """
        # 提取 noisify 参数（epsilon、mechanism、sensitivity 等）
        params = self._dp_params_from_noisy_request(request)
        # 对真实计数加噪，内部执行预算扣减
        result = self.service.dp_noisy_count(request.true_count, params)
        return privacy_pb2.DPResponse(result=result)

    def DPNoisySum(self, request, context):
        """对已聚合求和加噪的 gRPC 方法。

        Add DP noise to a pre-aggregated true sum.

        Args:
            request: NoisySumRequest，包含 true_sum（真实求和）与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPResponse: 加噪后的求和结果。
        """
        # 提取 noisify 参数并对真实求和加噪
        params = self._dp_params_from_noisy_request(request)
        result = self.service.dp_noisy_sum(request.true_sum, params)
        return privacy_pb2.DPResponse(result=result)

    def DPNoisyMean(self, request, context):
        """对已聚合 sum/count 加噪得到均值的 gRPC 方法。

        Add DP noise to pre-aggregated sum and count, then compute noisy mean.

        Args:
            request: NoisyMeanRequest，包含 true_sum、true_count 与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPResponse: 加噪后的均值结果。
        """
        # 提取参数，对 sum 和 count 分别加噪后求商得到均值
        params = self._dp_params_from_noisy_request(request)
        result = self.service.dp_noisy_mean(
            request.true_sum, request.true_count, params
        )
        return privacy_pb2.DPResponse(result=result)

    def DPNoisyHistogram(self, request, context):
        """对已聚合直方图加噪的 gRPC 方法。

        Add DP noise to each bin of a pre-aggregated histogram.

        Args:
            request: NoisyHistogramRequest，包含 true_counts（map<string,int>）与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPHistogramResponse: 加噪后的直方图 map<string, double>。
        """
        # 提取 noisify 参数
        params = self._dp_params_from_noisy_request(request)
        # 将 protobuf map 转为 Python dict，对每个 bin 独立加噪
        res_dict = self.service.dp_noisy_histogram(dict(request.true_counts), params)
        # 类型断言：确保返回值为字典
        assert isinstance(res_dict, dict)
        # 转换类型以契合 protobuf map<string, double>
        result = {str(k): float(v) for k, v in res_dict.items()}
        return privacy_pb2.DPHistogramResponse(result=result)

    # ─── 分块流式 DP RPC 方法 / Chunked streaming DP RPC methods ───

    def _chunks_from_request(self, request) -> list[list[float]]:
        """从 chunked 请求中提取数据块列表。

        Extract a list of numeric chunks from a ChunkedXxxRequest message.
        Each chunk is a repeated-float message representing a data partition.

        Args:
            request: 包含 repeated DataChunk chunks 字段的请求消息。

        Returns:
            二维列表，每个子列表为一个数据分块的浮点数值。
        """
        # 遍历每个 DataChunk，将其 repeated values 转为 Python list
        return [list(chunk.values) for chunk in request.chunks]

    def DPChunkedCount(self, request, context):
        """分块流式 DP 计数 gRPC 方法。

        Compute DP count over chunked (partitioned) data for federated scenarios.

        Args:
            request: ChunkedCountRequest，包含 chunks（数据分块列表）与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPResponse: 加噪后的计数结果。
        """
        # 提取 noisify 参数（分块请求复用 noisify 参数结构）
        params = self._dp_params_from_noisy_request(request)
        # 对各分块先局部计数再汇总加噪
        result = self.service.dp_chunked_count(self._chunks_from_request(request), params)
        return privacy_pb2.DPResponse(result=result)

    def DPChunkedSum(self, request, context):
        """分块流式 DP 求和 gRPC 方法。

        Compute DP sum over chunked data partitions.

        Args:
            request: ChunkedSumRequest，包含 chunks 与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPResponse: 加噪后的求和结果。
        """
        # 提取参数，对各分块分别求和后汇总加噪
        params = self._dp_params_from_noisy_request(request)
        result = self.service.dp_chunked_sum(self._chunks_from_request(request), params)
        return privacy_pb2.DPResponse(result=result)

    def DPChunkedMean(self, request, context):
        """分块流式 DP 均值 gRPC 方法。

        Compute DP mean over chunked data partitions.

        Args:
            request: ChunkedMeanRequest，包含 chunks 与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPResponse: 加噪后的均值结果。
        """
        # 提取参数，对各分块汇总 sum/count 后加噪求均值
        params = self._dp_params_from_noisy_request(request)
        result = self.service.dp_chunked_mean(
            self._chunks_from_request(request), params
        )
        return privacy_pb2.DPResponse(result=result)

    def DPChunkedHistogram(self, request, context):
        """分块流式 DP 直方图 gRPC 方法。

        Compute DP histogram over chunked categorical data.

        Args:
            request: ChunkedHistogramRequest，包含 chunks、categories 与 DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPHistogramResponse: 加噪后的直方图。
        """
        # 提取 noisify 参数
        params = self._dp_params_from_noisy_request(request)
        # 将各分块的 repeated values 转为二维列表
        chunks = [list(chunk.values) for chunk in request.chunks]
        # 对各分块分别统计频次后汇总，再对每个 bin 加噪
        res_dict = self.service.dp_chunked_histogram(
            chunks, list(request.categories), params
        )
        # 类型断言：确保返回字典
        assert isinstance(res_dict, dict)
        # 类型转换以契合 protobuf map<string, double>
        result = {str(k): float(v) for k, v in res_dict.items()}
        return privacy_pb2.DPHistogramResponse(result=result)


    # ─── K-匿名 RPC 方法 / K-Anonymity RPC methods ───

    def KAnonymizeRecord(self, request, context):
        """单条记录 K-匿名 gRPC 方法。

        Matches Go grpcserver.Server.KAnonymizeRecord, which applies
        MaskRecord (not hierarchy generalization) for this RPC.

        Args:
            request: KAnonymizeRecordRequest，包含 record（map<string,string>）、qi_cols（准标识符列）、k。
            context: gRPC 服务上下文。

        Returns:
            KAnonymizeResponse: 包含脱敏后的 result 字典。
        """
        # Go 外层 Server 对该 RPC 直接调用 MaskRecord；qi_cols/k 不参与处理。
        return privacy_pb2.KAnonymizeResponse(
            result=self.service.mask_record(dict(request.record))
        )

    def KAnonymizeTable(self, request, context):
        """整张表 K-匿名泛化 gRPC 方法。

        Generalize an entire table with Go-compatible k clamping and the same
        mask fallback used by Go TypedServer.KAnonymizeTable.

        Args:
            request: KAnonymizeTableRequest，包含 rows（RecordEntry 列表）、qi_cols、k、max_depth。
            context: gRPC 服务上下文。

        Returns:
            KAnonymizeTableResponse: 包含泛化后的 rows（RecordEntry 列表）。
        """
        records = [dict(r.fields) for r in request.rows]
        # Go service 层 clampK：k < 2 收敛为 2
        resolved_k = request.k if request.k >= 2 else 2
        try:
            result = self.service.k_anonymize_table(
                records, list(request.qi_cols), resolved_k, request.max_depth
            )
            assert isinstance(result, list)
            return privacy_pb2.KAnonymizeTableResponse(
                rows=[privacy_pb2.RecordEntry(fields=r) for r in result]
            )
        except Exception:
            # 与 Go 对齐：K-匿名失败时回退为逐条脱敏，而非返回错误
            fallback = [
                privacy_pb2.RecordEntry(fields=self.service.mask_record(rec))
                for rec in records
            ]
            return privacy_pb2.KAnonymizeTableResponse(rows=fallback)

    def KAnonymizeDataFrame(self, request, context):
        """DataFrame K-匿名 gRPC 方法。

        Matches Go TypedServer.KAnonymizeDataFrame, which masks every record
        row-by-row (the RPC name is historical; no Mondrian is performed).

        Args:
            request: KAnonymizeDataFrameRequest，包含 data（RecordEntry 列表）、qi_cols、k、max_depth。
            context: gRPC 服务上下文。

        Returns:
            KAnonymizeDataFrameResponse: 包含脱敏后的 data。
        """
        rows = [
            privacy_pb2.RecordEntry(fields=self.service.mask_record(dict(row.fields)))
            for row in request.data
        ]
        return privacy_pb2.KAnonymizeDataFrameResponse(data=rows)

    # ─── 查询混淆 RPC 方法 / Query Obfuscation (QOL) RPC methods ───

    def ObfuscateQuery(self, request, context):
        """查询混淆 gRPC 方法。

        Inject dummy queries alongside the real query to provide K-anonymity
        for query patterns (Query Obfuscation Layer).

        Args:
            request: ObfuscateQueryRequest，包含 query、num_dummies、domain、
                     medical_pool、generic_pool、seed 等参数。
            context: gRPC 服务上下文。

        Returns:
            ObfuscateQueryResponse: 包含混淆后的查询列表 result。
        """
        # Go 外层 Server 对单条 gRPC 混淆默认 4 条诱饵（REST 默认 3），保持同口径；
        # 领域不设默认：仅 "medical" 走医疗池，其余（含空串）走 generic。
        num_dummies = request.num_dummies if request.num_dummies > 0 else 4
        domain = "medical" if request.domain == "medical" else "generic"
        result = self.service.obfuscate_query(
            request.query,           # 原始真实查询
            num_dummies,             # 需要注入的虚拟查询数量
            domain,                  # 查询领域
            medical_pool=list(request.medical_pool) if request.medical_pool else None,
            generic_pool=list(request.generic_pool) if request.generic_pool else None,
            seed=request.seed if request.seed != 0 else None,
        )
        return privacy_pb2.ObfuscateQueryResponse(result=result)

    def ObfuscateQueryBatch(self, request, context):
        """批量查询混淆 gRPC 方法。

        Obfuscate multiple queries in a single batch call.

        Args:
            request: ObfuscateQueryBatchRequest，包含 queries（查询列表）及混淆参数。
            context: gRPC 服务上下文。

        Returns:
            ObfuscateQueryBatchResponse: 包含每个查询的混淆结果列表。
        """
        # 对批量查询逐一执行混淆，共享相同的 dummy 数量与领域配置
        results = self.service.obfuscate_query_batch(
            list(request.queries),   # 待混淆的查询列表
            request.num_dummies,     # 每条查询注入的虚拟查询数
            request.domain,          # 查询领域
            medical_pool=list(request.medical_pool) if request.medical_pool else None,
            generic_pool=list(request.generic_pool) if request.generic_pool else None,
            seed=request.seed if request.seed != 0 else None,
        )
        # 将每个混淆结果封装为 ObfuscateQueryResponse，组装批量响应
        return privacy_pb2.ObfuscateQueryBatchResponse(
            results=[privacy_pb2.ObfuscateQueryResponse(result=r) for r in results]
        )

    # ─── 健康检查与参数推荐 / Health check & parameter recommendation ───

    def Health(self, request, context):
        """健康检查 gRPC 方法。

        Return service health status and current namespace.
        Used by load balancers and orchestrators for liveness/readiness probes.

        Args:
            request: HealthRequest（空消息）。
            context: gRPC 服务上下文。

        Returns:
            HealthResponse: 包含 status="ok" 与当前 namespace。
        """
        # 直接返回固定 "ok" 状态与当前命名空间，无业务逻辑
        return privacy_pb2.HealthResponse(status="ok", namespace=NAMESPACE)

    def RecommendParams(self, request, context):
        """隐私参数推荐 gRPC 方法。

        Analyze sample data and recommend optimal DP/K-anonymity parameters,
        then persist them to the profile for the specified namespace.

        Args:
            request: RecommendRequest，包含 namespace、rows/values（样本数据）、qi_cols。
            context: gRPC 服务上下文。

        Returns:
            RecommendResponse: 包含推荐参数的 JSON 字符串。
        """
        # 解析可选的样本数据：行记录或纯数值列表
        rows = None
        if request.rows:
            rows = [dict(r.fields) for r in request.rows]  # 将 RecordEntry 转为 dict 列表
        values = list(request.values) if request.values else None  # 纯数值样本
        qi_cols = list(request.qi_cols) if request.qi_cols else None  # 准标识符列名

        # 按命名空间查找或创建 PrivacyService 实例（懒加载缓存模式）
        rec_service = self._service_cache.get(request.namespace)
        if rec_service is None:
            # 首次访问该命名空间：创建新实例并缓存
            rec_service = PrivacyService(profile_path=PROFILE_PATH, namespace=request.namespace)
            self._service_cache[request.namespace] = rec_service
        # 执行参数推荐并持久化到 profile 文件
        recommended = rec_service.recommend_and_save_params(values, rows, qi_cols)

        # 将推荐结果序列化为 JSON 字符串返回
        import json
        return privacy_pb2.RecommendResponse(
            status="success",                          # 操作状态
            namespace=request.namespace,               # 目标命名空间
            recommended_params_json=json.dumps(recommended)  # 推荐参数 JSON
        )

    # ─── 本地差分隐私 (LDP) RPC 方法 / Local DP RPC methods ───

    def PerturbBinaryBatch(self, request, context):
        """二值本地 DP 扰动 gRPC 方法。

        Apply randomized response to a batch of binary (0/1) values for local DP.
        Each value is independently perturbed with probability derived from epsilon.

        Args:
            request: PerturbBinaryBatchRequest，包含 values（0/1 列表）与 epsilon。
            context: gRPC 服务上下文。

        Returns:
            PerturbBinaryBatchResponse: 包含扰动后的 results 列表。
        """
        # 对每个二值数据独立执行随机响应扰动（Warner 模型）
        results = self.service.perturb_binary_batch(list(request.values), request.epsilon)
        return privacy_pb2.PerturbBinaryBatchResponse(results=results)

    def PerturbCategoricalBatch(self, request, context):
        """类别型本地 DP 扰动 gRPC 方法。

        Apply k-ary randomized response to a batch of categorical values.

        Args:
            request: PerturbCategoricalBatchRequest，包含 values、categories（类别全集）、epsilon。
            context: gRPC 服务上下文。

        Returns:
            PerturbCategoricalBatchResponse: 包含扰动后的 results 列表。
        """
        # 对每个类别值执行 k-ary 随机响应：以概率 p 保留原值，否则随机替换为其他类别
        results = self.service.perturb_categorical_batch(
            list(request.values), list(request.categories), request.epsilon
        )
        return privacy_pb2.PerturbCategoricalBatchResponse(results=results)

    def EstimateBinaryFrequency(self, request, context):
        """二值频率估计 gRPC 方法。

        Estimate the true frequency of 1s from perturbed binary reports
        using the inverse of the randomized response matrix.

        Args:
            request: EstimateBinaryFrequencyRequest，包含 reported_values（扰动后的报告值）与 epsilon。
            context: gRPC 服务上下文。

        Returns:
            EstimateBinaryFrequencyResponse: 包含估计的真实频率 estimated_frequency。
        """
        # 通过逆矩阵校正从扰动报告中恢复真实频率估计
        estimated = self.service.estimate_binary_frequency(
            list(request.reported_values), request.epsilon
        )
        return privacy_pb2.EstimateBinaryFrequencyResponse(estimated_frequency=estimated)

    def EstimateCategoricalHistogram(self, request, context):
        """类别直方图估计 gRPC 方法。

        Estimate the true categorical distribution from perturbed reports
        using unbiased frequency estimation.

        Args:
            request: EstimateCategoricalHistogramRequest，包含 reported_values、categories、epsilon。
            context: gRPC 服务上下文。

        Returns:
            EstimateCategoricalHistogramResponse: 包含估计直方图 map<string, double>。
        """
        # 对每个类别执行无偏频率估计，校正随机响应引入的偏差
        est_dict = self.service.estimate_categorical_histogram(
            list(request.reported_values), list(request.categories), request.epsilon
        )
        # 将 key 转换为 str，value 转换为 float，以契合 protobuf map<string, double>
        estimated_histogram = {str(k): float(v) for k, v in est_dict.items()}
        return privacy_pb2.EstimateCategoricalHistogramResponse(
            estimated_histogram=estimated_histogram
        )

    # ─── 高级 DP 聚合 RPC 方法 / Advanced DP aggregate RPC methods ───

    def DPAggregate(self, request, context):
        """多规格 DP 聚合 gRPC 方法。

        Execute multiple DP aggregate queries (count/sum/mean) over a DataFrame
        in a single call, with per-spec column and aggregation configuration.

        Args:
            request: DPAggregateRequest，包含 rows、specs_json（聚合规格 JSON）、DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPAggregateResponse: 包含聚合结果 JSON 字符串。
        """
        # 延迟导入 json 与 pandas，避免模块级重量级依赖
        import json

        import pandas as pd
        # 将 protobuf repeated RecordEntry 转为 dict 列表，构建 DataFrame
        rows = [dict(r.fields) for r in request.rows]
        df = pd.DataFrame(rows)
        # 解析聚合规格 JSON：每个 spec 定义目标列、聚合函数、截断边界等
        specs = json.loads(request.specs_json)
        # 构建全局 DP 参数字典
        params = {
            "epsilon": request.epsilon,          # 总隐私预算（在所有 spec 间分配）
            "delta": request.delta,              # Gaussian 机制的 δ 参数
            "mechanism": request.mechanism,      # 噪声机制
            "return_details": request.return_details,  # 是否返回噪声尺度等详细信息
        }
        # 调用业务层执行多规格聚合，内部按 spec 数量均分预算
        res = self.service.dp_aggregate(df, specs, params)
        # 将结果序列化为 JSON（default=str 处理 numpy 类型等非原生对象）
        res_json = json.dumps(res, default=str)
        return privacy_pb2.DPAggregateResponse(results_json=res_json)

    def DPVectorSum(self, request, context):
        """DP 向量求和 gRPC 方法。

        Compute a differentially private sum over a collection of vectors,
        with per-vector norm clipping to bound sensitivity.

        Args:
            request: DPVectorSumRequest，包含 vectors（向量列表）、max_norm、DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPVectorSumResponse: 包含加噪向量及可选的详细信息。
        """
        # 延迟导入 numpy 用于向量运算
        import numpy as np
        # 将 protobuf repeated VectorChunk 转为二维 Python 列表
        vec_list = [list(chunk.values) for chunk in request.vectors]
        # 构建 numpy 二维数组 (n_vectors × dim)
        vectors = np.array(vec_list)
        # 构建参数字典：max_norm 用于 L2 截断以控制灵敏度
        params = {
            "max_norm": request.max_norm if request.max_norm > 0 else 1.0,
            "epsilon": request.epsilon if request.epsilon > 0 else 1.0,
            "delta": request.delta,
            "mechanism": request.mechanism if request.mechanism else "laplace",
            "return_details": request.return_details,
        }
        # 调用业务层执行向量求和 + 截断 + 加噪
        res = self.service.dp_vector_sum(vectors, params)

        # 根据业务层返回类型判断是否携带详细信息
        if hasattr(res, "value"):
            noisy_vec = list(res.value)
            noise_scale = (
                float(np.mean(res.noise_scale))
                if isinstance(res.noise_scale, (list, np.ndarray))
                else float(res.noise_scale)
            )
            dp_proto = privacy_pb2.DPResultProto(
                value_vector=noisy_vec,
                noise_mechanism=res.noise_mechanism,
                noise_scale=noise_scale,
                epsilon_spent=res.epsilon_spent,
                delta_spent=res.delta_spent,
            )
            return privacy_pb2.DPVectorSumResponse(noisy_vector=noisy_vec, result_details=dp_proto)
        else:
            noisy_vec = list(res)
            return privacy_pb2.DPVectorSumResponse(noisy_vector=noisy_vec)

    def DPVectorMean(self, request, context):
        import numpy as np
        vec_list = [list(chunk.values) for chunk in request.vectors]
        vectors = np.array(vec_list)
        params = {
            "max_norm": request.max_norm if request.max_norm > 0 else 1.0,
            "epsilon": request.epsilon if request.epsilon > 0 else 1.0,
            "delta": request.delta,
            "mechanism": request.mechanism if request.mechanism else "laplace",
            "min_count": request.min_count if request.min_count > 0 else 5.0,  # 对齐 REST 默认值（service.py:323）
            "return_details": request.return_details,
        }
        # 调用业务层执行向量均值：L2 截断 + 各向同性加噪 + noisy_count 归一化
        res = self.service.dp_vector_mean(vectors, params)

        # 根据业务层返回类型判断是否携带详细信息
        if hasattr(res, "value"):
            # 返回了 DPResult 对象：提取加噪均值向量与元数据
            mean_vec = list(res.value)  # 加噪后的均值向量
            # 噪声尺度可能是数组（每维不同）或标量，统一转为单个 float
            noise_scale = (
                float(np.mean(res.noise_scale))  # 多维取均值
                if isinstance(res.noise_scale, (list, np.ndarray))
                else float(res.noise_scale)      # 标量直接转换
            )
            # 构建详细信息 protobuf 消息
            dp_proto = privacy_pb2.DPResultProto(
                value_vector=mean_vec,            # 加噪均值向量
                noise_mechanism=res.noise_mechanism,  # 实际使用的噪声机制
                noise_scale=noise_scale,          # 噪声尺度
                epsilon_spent=res.epsilon_spent,  # 实际消耗的 ε
                delta_spent=res.delta_spent,      # 实际消耗的 δ
            )
            return privacy_pb2.DPVectorMeanResponse(mean_vector=mean_vec, result_details=dp_proto)
        else:
            # 仅返回纯向量（未请求详细信息）
            mean_vec = list(res)
            return privacy_pb2.DPVectorMeanResponse(mean_vector=mean_vec)

    def DPAdaptiveClip(self, request, context):
        """自适应截断边界估计 gRPC 方法。

        Iteratively estimate optimal clipping bounds for DP aggregation
        using a binary-search-like approach targeting a specified quantile.

        Args:
            request: DPAdaptiveClipRequest，包含 values、epsilon、target_quantile、
                     num_iterations、initial_clip。
            context: gRPC 服务上下文。

        Returns:
            DPAdaptiveClipResponse: 包含推荐的 clip_lower 与 clip_upper。
        """
        # 构建自适应截断参数
        # proto3 零值视为未设置，回退到与 REST 一致的默认值（service.py:333-335：
        # target_quantile=0.95 / num_iterations=15 / initial_clip=10.0），
        # 避免 proto3 默认 0 被业务层当作有效参数（如 0 次迭代导致估计失效）。
        params = {
            "epsilon": request.epsilon,              # 用于截断估计的隐私预算
            "target_quantile": request.target_quantile if request.target_quantile > 0 else 0.95,  # 目标分位数
            "num_iterations": request.num_iterations if request.num_iterations > 0 else 15,       # 二分搜索迭代次数
            "initial_clip": request.initial_clip if request.initial_clip > 0 else 10.0,           # 初始截断范围
        }
        # 调用业务层执行自适应截断估计，返回 (lower, upper) 元组
        lower, upper = self.service.dp_adaptive_clip(list(request.values), params)
        return privacy_pb2.DPAdaptiveClipResponse(clip_lower=lower, clip_upper=upper)

    def DPGroupBy(self, request, context):
        """DP 分组聚合 gRPC 方法。

        Perform a differentially private group-by aggregation on tabular data.

        Args:
            request: DPGroupByRequest，包含 rows、group_col、target_col、agg、DP 参数。
            context: gRPC 服务上下文。

        Returns:
            DPGroupByResponse: 包含分组聚合结果 JSON。
        """
        # 延迟导入 json 与 pandas
        import json

        import pandas as pd
        # 将 protobuf 行数据构建为 DataFrame
        rows = [dict(r.fields) for r in request.rows]
        df = pd.DataFrame(rows)
        # 构建 DP 参数：包含截断边界以控制组内灵敏度
        params = {
            "epsilon": request.epsilon,      # 隐私预算
            "delta": request.delta,          # δ 参数
            "mechanism": request.mechanism,  # 噪声机制
        }
        # proto3 标量默认为 0：clip_lower/clip_upper 双零时视为未设置，不透传，
        # 由业务层自行推导截断边界（参照 _dp_params_from_request 的 guard 写法）；
        # 直接透传 0.0/0.0 会被业务层误当作有效截断区间 [0, 0]。
        if request.clip_lower != 0.0 or request.clip_upper != 0.0:
            params["clip_lower"] = request.clip_lower  # 组内截断下界
            params["clip_upper"] = request.clip_upper  # 组内截断上界
        # 调用业务层执行分组聚合：按 group_col 分组，对 target_col 执行 agg 操作并加噪
        res = self.service.dp_groupby(df, request.group_col, request.target_col, request.agg, params)
        # 序列化为 JSON 返回
        res_json = json.dumps(res, default=str)
        return privacy_pb2.DPGroupByResponse(result_json=res_json)

    # ─── 动态分类分级 RPC 方法 / Dynamic classification RPC method ───

    def DynClassify(self, request, context):
        """动态分类求值 gRPC 方法。

        Classify a single field using the dynamic classification engine
        (rule engine → NER → LLM funnel). Returns security tags with
        sensitivity level, category, and audit information.

        Args:
            request: DynClassificationRequest，包含 field_name、field_value、domain、standard。
            context: gRPC 服务上下文。

        Returns:
            DynClassificationResponse: 包含 tags、max_level、audit_timestamp、engine_layer。
        """
        # 延迟导入动态分类服务，避免未使用分类功能时加载规则引擎依赖
        from .dynclassification import DynClassificationService

        # 懒初始化：首次调用时创建 DynClassificationService 实例并缓存
        if not hasattr(self, "_dyn_service") or self._dyn_service is None:
            self._dyn_service = DynClassificationService()

        # 检查规则文件是否变更，若有更新则热重载（支持运行中修改规则无需重启）。
        # 注意必须走 DynClassificationService.check_and_reload()（service.py:664），
        # 该方法在 loader 重载后连带清空 funnel/分类结果缓存与 NER/LLM 适配器；
        # 直接调用 loader.check_and_reload() 会导致旧引擎缓存继续生效，热重载形同虚设。
        self._dyn_service.check_and_reload()
        # 提取请求参数
        field_name = request.field_name    # 待分类的字段名
        value = request.field_value        # 字段值
        domain = request.domain or None    # 可选：限定领域（如 medical/finance）
        standard = request.standard or None  # 可选：限定标准（如 GB/T 35273）

        # 调用动态分类引擎执行字段分类（内部经过 L1规则 → L2 NER → L3 LLM 漏斗）
        result = self._dyn_service.classify_field(
            field_name=field_name,
            value=value,
            domain=domain,
            standard=standard,
        )

        # 初始化响应字段默认值
        tags_proto = []          # 安全标签列表（protobuf 消息）
        max_level = ""           # 最高敏感度等级
        audit_timestamp = ""     # 审计时间戳
        engine_layer = "L1_RULE"  # 命中的引擎层级，默认 L1 规则引擎

        # 若分类结果包含字段级结果，提取标签与元数据
        if result.field_result:
            max_level = result.field_result.final_level or ""  # 最终裁定的敏感度等级
            # 读取实际的 engine_layer，而非硬编码 L1_RULE
            engine_layer = result.field_result.engine_layer or "L1_RULE"
            # 遍历所有安全标签，逐一转换为 protobuf 消息
            for tag in result.field_result.tags:
                tags_proto.append(
                    privacy_pb2.DynSecurityTagProto(
                        level=tag.level,              # 敏感度等级（L1-L5）
                        category=tag.category,        # 数据类别（如 PERSONAL_INFO）
                        rule_id=tag.rule_id,          # 命中的规则 ID
                        source_engine=tag.source_engine,  # 来源引擎（L1_RULE/L2_NER/L3_LLM）
                        domain=tag.domain,            # 所属领域
                        standard_id=tag.standard_id,  # 关联标准编号
                        is_override=tag.is_override,  # 是否为强制覆盖标签
                        is_downgrade=tag.is_downgrade,  # 是否为降级标签
                        match_target=tag.match_target,  # 匹配目标（field_name/value）
                    )
                )

        # 若包含审计信息，提取时间戳
        if result.audit_info:
            audit_timestamp = result.audit_info.timestamp or ""

        # 组装并返回完整的动态分类响应
        return privacy_pb2.DynClassificationResponse(
            tags=tags_proto,            # 安全标签列表
            max_level=max_level,        # 最高敏感度等级
            audit_timestamp=audit_timestamp,  # 审计时间戳
            engine_layer=engine_layer,  # 命中的引擎层级
        )

    # ─── 动态分类扩展 RPC / Extended dynamic classification RPCs ───

    def _get_dyn_service(self):
        """懒初始化并热重载 DynClassificationService（与 REST 路由同口径）。"""
        if not hasattr(self, "_dyn_service") or self._dyn_service is None:
            from .dynclassification import DynClassificationService

            self._dyn_service = DynClassificationService()
        self._dyn_service.check_and_reload()
        return self._dyn_service

    @staticmethod
    def _dyn_eval_payload(field_name: str, value: str, field_result) -> dict:
        tags = field_result.tags or []
        tag = tags[0] if tags else None
        level = field_result.final_level
        return {
            "field": field_name,
            "value": value,
            "level": level,
            "level_id": level,
            "category": tag.category if tag else "unknown",
            "confidence": field_result.confidence,
            "matched_by": (
                f"rule:{tag.rule_id}"
                if tag.rule_id
                else (tag.source_engine or "rule").lower()
            )
            if tag
            else "default",
        }

    def DynEval(self, request, context):
        """动态单字段分类求值（与 Go TypedServer.DynEval 对齐）。"""
        svc = self._get_dyn_service()
        resp = svc.classify_field(
            field_name=request.field_name,
            value=request.value,
            domain=request.domain or None,
            standard=request.standard or None,
        )
        payload = self._dyn_eval_payload(request.field_name, request.value, resp.field_result)
        import json

        return privacy_pb2.DynEvalResponse(
            field=payload["field"],
            value=payload["value"],
            level=payload["level"],
            level_id=payload["level_id"],
            category=payload["category"],
            confidence=payload["confidence"],
            matched_by=payload["matched_by"],
            result_json=json.dumps(
                {**payload, "domain": request.domain, "standard": request.standard},
                ensure_ascii=False,
            ),
        )

    def DynEvalRecord(self, request, context):
        """动态记录/批量分类求值（与 Go TypedServer.DynEvalRecord 对齐）。"""
        import json

        svc = self._get_dyn_service()
        domain = request.domain or None
        standard = request.standard or None
        if request.records:
            classifications = []
            for entry in request.records:
                for field_name, value in dict(entry.fields).items():
                    resp = svc.classify_field(
                        field_name=field_name,
                        value=value,
                        domain=domain,
                        standard=standard,
                    )
                    classifications.append(
                        self._dyn_eval_payload(field_name, value, resp.field_result)
                    )
            overall = ""
            for item in classifications:
                if _level_rank(item["level"]) > _level_rank(overall):
                    overall = item["level"]
            result_json = json.dumps(
                {
                    "classifications": classifications,
                    "level": overall,
                    "overall_level": overall,
                },
                ensure_ascii=False,
            )
            return privacy_pb2.DynEvalRecordResponse(
                level=overall, overall_level=overall, result_json=result_json
            )

        record = dict(request.record)
        resp = svc.classify_record(record=record, domain=domain, standard=standard)
        record_result = resp.record_result
        results = {}
        for field_name, field_result in record_result.field_results.items():
            results[field_name] = self._dyn_eval_payload(
                field_name, _format_grpc_value(record.get(field_name, "")), field_result
            )
        overall = record_result.final_level
        result_json = json.dumps(
            {
                "result": results,
                "classifications": results,
                "level": overall,
                "overall_level": overall,
            },
            ensure_ascii=False,
        )
        return privacy_pb2.DynEvalRecordResponse(
            level=overall, overall_level=overall, result_json=result_json
        )

    def DynStandards(self, request, context):
        """列出动态分类标准及详情（与 Go TypedServer.DynStandards 对齐）。"""
        import json

        svc = self._get_dyn_service()
        standards = svc.list_standards()
        details = svc.list_standards_detail()
        return privacy_pb2.DynStandardsResponse(
            standards=standards,
            details_json=json.dumps(details, ensure_ascii=False),
        )

    def DynDomains(self, request, context):
        """列出可用领域包（与 Go TypedServer.DynDomains 对齐）。"""
        return privacy_pb2.DynDomainsResponse(domains=self._get_dyn_service().list_domains())

    def DynOperators(self, request, context):
        """列出可用匹配算子（与 Go TypedServer.DynOperators 对齐）。"""
        return privacy_pb2.DynOperatorsResponse(
            operators=self._get_dyn_service().list_operators()
        )

    def DynValidate(self, request, context):
        """校验规则 YAML 合法性（与 Go TypedServer.DynValidate 对齐）。"""
        from .dynclassification.validator import validate_rules_dir

        result = validate_rules_dir(request.rules_dir or "rules")
        return privacy_pb2.DynValidateResponse(
            is_valid=result.is_valid,
            valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
        )

    def DynGenerateProfile(self, request, context):
        """从标准文档生成规则配置（与 Go TypedServer.DynGenerateProfile 对齐）。"""
        svc = self._get_dyn_service()
        try:
            generated = svc.generate_profile_from_doc(request.doc_path)
        except Exception as exc:
            # 与 Go 一致：失败时以状态消息返回，不伪装成功。
            return privacy_pb2.DynGenerateProfileResponse(
                status="error",
                message=f"Failed to generate configuration: {exc}",
                generated_files={},
            )
        return privacy_pb2.DynGenerateProfileResponse(
            status="ok",
            message=f"Successfully generated profiles from {request.doc_path}",
            generated_files=generated,
        )


# ─── 统一异常映射包装 / Uniform exception mapping wrapper ───
# 遍历 PrivacyServicer 类的所有公共方法，用 _grpc_error_mapper 装饰器包装，
# 确保任何 RPC 方法抛出的异常都能被映射为语义化的 gRPC 状态码，
# 而不是默认返回 UNKNOWN。
# Wrap all public RPC methods with the error mapper decorator at class-definition time.
for _name in dir(PrivacyServicer):
    # 跳过私有/保护方法（以 _ 开头），仅包装公共 RPC 方法
    if _name.startswith("_"):
        continue
    _attr = getattr(PrivacyServicer, _name)  # 获取类属性
    if callable(_attr):
        # 用异常映射装饰器替换原方法，实现透明的错误码转换
        setattr(PrivacyServicer, _name, _grpc_error_mapper(_attr))


def serve(host: str = "0.0.0.0", port: int = 50051, max_workers: int | None = None, wait_for_termination: bool = True):
    """启动 gRPC 服务器。

    使用 ThreadPoolExecutor 作为工作线程池，注册 PrivacyServicer，
    监听指定端口，并阻塞或非阻塞等待连接。

    根据环境变量可启用 TLS/mTLS、认证鉴权、速率限制与可观测性拦截器。

    Start the gRPC server with a thread pool, register the PrivacyServicer,
    and optionally enable TLS/auth/rate-limit/observability interceptors.

    Args:
        host: gRPC 服务监听主机，默认 0.0.0.0。
        port: gRPC 服务监听端口，默认 50051。
        max_workers: 线程池最大工作线程数，默认从环境变量 PRIVACY_GRPC_MAX_WORKERS
            读取，未设置时回退到 64（高并发优化）。
        wait_for_termination: 是否阻塞等待服务器终止，默认 True。

    Returns:
        grpc.Server 实例（当 wait_for_termination=False 时可用于外部控制）。
    """
    # ── 步骤 1：初始化日志与链路追踪 / Step 1: Init logging & tracing ──
    # gRPC-only entrypoint: ensure logging/tracing are initialized.
    # 配置结构化日志：级别、格式（text/json）、服务名均从环境变量读取
    configure_logging(
        log_level=os.environ.get("PRIVACY_LOG_LEVEL", "INFO"),       # 日志级别
        json_format=os.environ.get("PRIVACY_LOG_FORMAT", "text").lower() == "json",  # 是否 JSON 格式
        service_name=os.environ.get("PRIVACY_SERVICE_NAME", "PrivShield"),  # 服务标识
    )
    # 初始化 OpenTelemetry 链路追踪：若配置了 OTLP endpoint 则启用导出
    init_tracing(
        endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),  # OTLP 收集器地址
        service_name=os.environ.get(
            "OTEL_SERVICE_NAME",  # 优先使用 OTEL 标准变量
            os.environ.get("PRIVACY_SERVICE_NAME", "PrivShield"),  # 回退到项目变量
        ),
    )

    # 高并发优化：max_workers 支持环境变量配置，默认 64
    if max_workers is None:
        max_workers = int(os.environ.get("PRIVACY_GRPC_MAX_WORKERS", "64"))

    # ── 步骤 2：构建拦截器链 / Step 2: Build interceptor chain ──
    # 读取安全配置（TLS/认证/限流开关与参数）
    settings = get_security_settings()
    # 拦截器列表：可观测性拦截器始终启用（记录请求指标与日志）
    interceptors: list[grpc.ServerInterceptor] = [
        GrpcObservabilityInterceptor(),  # 请求计数、耗时、状态码指标
    ]
    # 若启用 API Key 认证，追加认证拦截器（校验 metadata 中的 api-key）
    if settings.auth_enabled:
        interceptors.append(AuthInterceptor(settings))
    # 若启用速率限制，追加限流拦截器（基于令牌桶/滑动窗口）
    if settings.rate_limit_enabled:
        interceptors.append(RateLimitInterceptor(settings))

    # ── 步 3：创建 gRPC 服务器 / Step 3: Create gRPC server ──
    # 设置 gRPC 消息大小限制：默认仅 4 MiB，base64 编码的图片或大表分类
    # 场景极易超限导致服务端重置 HTTP/2 连接（表现为 connection reset by peer）。
    # 将收发上限均提升至 64 MiB，与 Go 客户端保持一致。
    _max_msg_size = 64 * 1024 * 1024  # 64 MiB
    _server_options = [
        # 接收消息上限 64 MiB
        ("grpc.max_receive_message_length", _max_msg_size),
        # 发送消息上限 64 MiB
        ("grpc.max_send_message_length", _max_msg_size),
        # 允许客户端在无活跃 RPC 时发送 keepalive PING，
        # 否则服务端会因 "too_many_pings" 发送 GOAWAY/ENHANCE_YOUR_CALM
        ("grpc.keepalive_permit_without_calls", 1),
        # 允许客户端最短每 5 秒发送一次 PING（高高频并发防护）
        ("grpc.http2.min_time_between_pings_ms", 5000),
        # 无数据交互时允许无限制次数发送 PING
        ("grpc.http2.max_pings_without_data", 0),
        # 禁用 PING 频率超限强制断开连接惩罚 (0 表示静默忽略过多 PING 而不断开 TCP)
        ("grpc.http2.max_ping_strikes", 0),
        # 连接最大存活时间 2h，到期后服务端发送 GOAWAY 通知客户端重连。
        # 防止长连接内存泄漏、促进负载均衡重分配、配合证书轮换。
        # 与 Go 服务端 MaxConnectionAge: 2h 对齐。
        ("grpc.max_connection_age_ms", 7_200_000),
        # 连接老化宽限期 5s：发送 GOAWAY 后等待在途 RPC 完成。
        # 与 Go 服务端 MaxConnectionAgeGrace: 5s 对齐。
        ("grpc.max_connection_age_grace_ms", 5_000),
        # 服务端主动 keepalive：每 30s 向客户端发送 PING，检测死连接。
        # 与 Go 服务端 KeepaliveParams.Time: 30s 对齐。
        ("grpc.keepalive_time_ms", 30_000),
        # keepalive PING 超时：10s 内未收到 ACK 视为连接断开。
        # 与 Go 服务端 KeepaliveParams.Timeout: 10s 对齐。
        ("grpc.keepalive_timeout_ms", 10_000),
        # 空闲连接最大存活时间 15min，超时后服务端发送 GOAWAY 关闭连接。
        # 防止空闲连接长期占用资源。与 Go 服务端 MaxConnectionIdle: 15min 对齐。
        ("grpc.max_connection_idle_ms", 900_000),
    ]
    # SO_REUSEPORT 内核级负载均衡仅 Linux 原生支持；
    # macOS 虽定义该常量但语义不同（无内核级连接分发），Windows 不支持。
    if sys.platform == "linux":
        _server_options.append(
            # 允许多进程绑定同一 gRPC 端口（SO_REUSEPORT）
            ("grpc.so_reuseport", 1),
        )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),  # 工作线程池
        interceptors=tuple(interceptors) if interceptors else None,  # 拦截器链
        options=_server_options,
    )
    # 将 PrivacyServicer 实例注册到 gRPC 服务器
    privacy_pb2_grpc.add_PrivacyServiceServicer_to_server(PrivacyServicer(), server)

    # ── 步骤 4：绑定端口 / Step 4: Bind port ──
    if settings.tls_enabled:
        # TLS/mTLS 模式：加载证书与私钥构建 ServerCredentials
        creds = grpc_server_credentials(settings)
        bound_port = server.add_secure_port(f"{host}:{port}", creds)  # 安全端口
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind gRPC secure port on {host}:{port}")
        logger.info("grpc_server_started", extra={"host": host, "port": bound_port, "tls": True})
    else:
        # 本地开发模式，使用非安全端口；生产环境建议启用 TLS/mTLS
        bound_port = server.add_insecure_port(f"{host}:{port}")  # 非安全端口
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind gRPC port on {host}:{port}")
        logger.info("grpc_server_started", extra={"host": host, "port": bound_port, "tls": False})

    # ── 步骤 5：启动服务 / Step 5: Start server ──
    server.start()  # 非阻塞启动，后台线程开始接受连接
    if wait_for_termination:
        # 阻塞主线程直到服务器被终止（Ctrl+C 或调用 server.stop()）
        server.wait_for_termination()
    return server  # 返回服务器实例，供外部控制（如优雅停机）


# ─── 命令行入口 / CLI entrypoint ───
# 支持通过 python -m engine.grpc_server 直接启动 gRPC 服务
if __name__ == "__main__":
    import argparse  # 命令行参数解析

    # 创建参数解析器，定义程序名与描述
    parser = argparse.ArgumentParser(
        prog="engine.grpc_server",
        description="SecretFlow Local Privacy Agent gRPC server.",
    )
    # --host 参数：gRPC 监听地址，优先读取环境变量 PRIVACY_GRPC_HOST
    parser.add_argument(
        "--host",
        default=os.environ.get("PRIVACY_GRPC_HOST", "0.0.0.0"),
        help="gRPC server host (default: 0.0.0.0 or PRIVACY_GRPC_HOST).",
    )
    # --port 参数：gRPC 监听端口，优先读取环境变量 PRIVACY_GRPC_PORT
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PRIVACY_GRPC_PORT", "50051")),
        help="gRPC server port (default: 50051 or PRIVACY_GRPC_PORT).",
    )
    # 解析命令行参数并启动服务
    args = parser.parse_args()

    # Non-blocking start so we can install signal handlers for graceful shutdown.
    # 非阻塞启动 gRPC 服务，以便注册信号处理器实现优雅停机。
    import signal as _signal
    import threading as _threading

    from .observability.logging_config import get_logger as _get_logger

    _logger = _get_logger(__name__)
    _grpc_server = serve(host=args.host, port=args.port, wait_for_termination=False)

    _shutdown_event = _threading.Event()

    def _handle_signal(signum, _frame):
        _logger.warning("gRPC standalone: received signal %s — initiating graceful shutdown", signum)
        _shutdown_event.set()

    _signal.signal(_signal.SIGTERM, _handle_signal)
    _signal.signal(_signal.SIGINT, _handle_signal)

    # Block main thread until a signal arrives.
    # 阻塞主线程直到收到关闭信号。
    _shutdown_event.wait()

    # Graceful stop: drain in-flight RPCs with a 5-second grace period.
    # 优雅停机：给在途 RPC 5 秒排空时间。
    _logger.info("gRPC standalone: stopping server (5s grace period)...")
    _stop_event = _grpc_server.stop(grace=5)
    if not _stop_event.wait(timeout=10):
        _logger.warning("gRPC standalone: server did not drain in-flight requests within timeout")
    _logger.info("gRPC standalone: server shut down gracefully")
