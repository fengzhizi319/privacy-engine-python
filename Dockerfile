# ==============================================================================
# PrivShield 多阶段构建（Multi-stage Build）
# ==============================================================================
# 为什么采用多阶段构建：
#   1. 分层缓存：base 层锁定基础镜像与核心依赖，业务代码变更不触发依赖重装，构建加速
#   2. 双目标镜像：core（默认轻量，仅隐私原语）/ ml（追加 torch 等重型 ML 依赖），按需选择
#   3. 安全最小化：仅 COPY 运行时必需文件，测试/文档/部署产物不进入镜像（见 .dockerignore）
#
# 构建流程总览：
#
#   python:3.13.13-slim-bookworm（锁定基础镜像，可追溯）
#        │
#        ▼
#   ┌─────────────────────────── [base] ───────────────────────────┐
#   │ ① 安装系统工具（curl + ca-certificates）                        │
#   │ ② 创建非 root 运行用户 privacy（安全最佳实践）                     │
#   │ ③ 安装核心依赖 requirements-core.txt（利用分层缓存）               │
#   └──────────────────────────────────────────────────────────────┘
#        │
#        ├───────────────► [core] 默认轻量运行镜像（推荐）
#        │                    │ 复制运行时文件（源码/规则/配置/proto/脚本）
#        │                    │ 复制并授权 entrypoint 脚本
#        │                    │ /app 归属 privacy 用户
#        │                    │ EXPOSE 8079(REST) + 50051(gRPC)
#        │                    │ HEALTHCHECK（/health，docker run 场景）
#        │                    │ USER privacy（非 root 运行）
#        │                    │ ENV 0.0.0.0 监听 + 日志无缓冲
#        │                    ▼
#        │                    CMD：python -m engine.server（REST+gRPC 一体进程）
#        │
#        └──────────────► [ml] 完整 ML 镜像
#                             │ USER root（安装需要 root 权限）
#                             │ 追加 requirements-ml.txt（torch/transformers/onnxruntime）
#                             │ 清理 __pycache__（体积优化）
#                             │ USER privacy（恢复非 root 安全基线）
#                             ▼
#                             CMD：与 core 相同的启动命令
#
# 示例：
#   docker build --target core -t privshield:2.0.0 .
#   docker build --target ml -t privshield:2.0.0-ml .

# ==============================================================================
# Stage 1: base —— 公共基础层（core / ml 两个目标共享）
# ==============================================================================
# 基础镜像选型：
#   - python:3.13-slim-bookworm : 与本地开发/ML 推理环境（conda pygpu、项目 .venv 均为 Python 3.13）一致；
#                              Qwen3.5 推理链路（transformers>=5.14 / fla-core>=0.5.2）实测验证于 3.13
#                              避免镜像与验证环境版本错位
#   - slim-bookworm         : Debian 精简版，减小镜像体积与攻击面
#   - 锁定 3.13-slim-bookworm : 保证构建可追溯、可复现
FROM python:3.13-slim-bookworm AS base

# 统一工作目录：后续所有 COPY/RUN 的默认路径，业务代码固定位于 /app
WORKDIR /app

# 安装基础系统工具并创建非 root 运行用户（安全最佳实践）：
#   - curl              : 供 K8s 探针（TLS 模式）与容器 HEALTHCHECK 调用
#   - ca-certificates   : HTTPS 访问外部服务（模型下载、OTLP 上报等）所需根证书
#   - --no-install-recommends : 只装必要包，避免连带安装多余软件（体积/攻击面）
#   - rm -rf /var/lib/apt/lists/* : 清理 apt 索引缓存，减薄镜像层
#   - groupadd/useradd  : 创建专用系统用户 privacy（-r 系统账户、无 shell、无 home），
#                         容器内进程以该用户运行，防止以 root 运行放大容器逃逸风险
RUN (sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true) \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r privacy \
    && useradd -r -g privacy -d /app -s /sbin/nologin privacy

# 先单独 COPY 依赖清单并安装（利用 Docker 分层缓存）：
#   - 依赖清单不变时，后续业务代码变更不会触发依赖重装，CI/本地构建显著加速
#   - requirements-core.txt 仅含核心运行依赖（FastAPI/gRPC/隐私原语/规则引擎），
#     重型 ML 依赖（torch/transformers/onnxruntime）不进入默认镜像
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ==============================================================================
# Stage 2: core —— 默认轻量运行镜像（推荐）
# ==============================================================================
FROM base AS core

# 分层 COPY：仅复制运行时必需文件，排除测试/文档/开发产物（排除规则见 .dockerignore）：
#   - engine/              : 隐私算力引擎源码（REST/gRPC/隐私原语/分类漏斗）
#   - rules/               : 分类规则与体系（YAML，运行时热加载）
#   - config/              : 运行配置（含 config/env/*.env 场景 profile；.env 已被
#                            .dockerignore 排除，敏感配置不进入镜像）
#   - proto/               : gRPC 协议 .proto 源（对应生成的 stub 已随主包复制）
COPY engine/ ./engine/
COPY rules/ ./rules/
COPY config/ ./config/
COPY proto/ ./proto/
COPY docs/standard/ ./docs/standard/
COPY pyproject.toml requirements-core.txt ./

# 复制 entrypoint 脚本并赋予执行权限：
#   - docker-entrypoint.sh 负责容器启动前的环境准备（如信号转发/前置检查），
#     必须可执行，否则 ENTRYPOINT 无法启动容器
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 创建数据与日志持久化挂载目录并授权 privacy 用户：
#   - /data/budget : SQLite 预算数据库
#   - /var/log/privacy : 审计日志
RUN mkdir -p /data/budget /var/log/privacy \
    && chown -R privacy:privacy /app /data /var/log/privacy

# 声明容器监听端口（文档性声明，实际端口映射由运行期 -p/--expose 决定）：
#   - 8079 : REST API（FastAPI）
#   - 50051: gRPC API
EXPOSE 8079 50051

# 容器级健康检查（Docker HEALTHCHECK）：
#   - 每 30s 同时探测 REST /health 与 gRPC :50051 TCP 连通性，超时 5s，启动宽限 10s
#   - 职责定位：服务于 docker run 单容器场景；K8s 部署时由 liveness/readiness
#     探针接管（见 deploy/helm/PrivShield/values.yaml 的 probes 节）
#   - REST: curl 探测 /health 端点返回 200
#   - gRPC: Python 轻量 TCP 连接检查（grpcio 已包含在核心依赖中）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8079/health > /dev/null && python -c "import socket; s=socket.create_connection(('localhost',50051),timeout=3); s.close()" || exit 1

# 切换到非 root 用户（安全最佳实践）：
#   - 自此之后所有 RUN/CMD/ENTRYPOINT 均以 privacy 用户身份执行
#   - 即使容器进程被攻破，权限也被限制在普通用户级，无法提权操作宿主机
USER privacy

# 运行时环境变量（镜像级默认值，可被运行期 -e/环境注入覆盖）：
#   - PYTHONUNBUFFERED=1     : 禁用 Python 输出缓冲，日志实时写入 stdout（容器日志采集依赖）
#   - PRIVACY_REST_HOST=0.0.0.0 : REST 监听所有网卡（容器内必须，否则宿主机无法访问）
#   - PRIVACY_GRPC_HOST=0.0.0.0 : gRPC 同上
#   - 注意：.env.example 中的 127.0.0.1 仅适用于宿主机直跑，容器内由此处覆盖
ENV PYTHONUNBUFFERED=1
ENV PRIVACY_REST_HOST=0.0.0.0
ENV PRIVACY_GRPC_HOST=0.0.0.0

# 入口定义（镜像启动协议）：
#   - ENTRYPOINT 固定为 entrypoint 脚本（启动前准备，一般不可被 docker run 参数覆盖）
#   - CMD 提供默认启动命令（REST + gRPC 一体进程），可用 docker run <image> <args> 覆盖
#     （如仅启动 gRPC：docker run <image> python -m engine.grpc_server）
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "engine.server"]

# ==============================================================================
# Stage 3: ml —— 完整 ML 镜像（torch / transformers / onnxruntime）
# ==============================================================================
# 在 core 基础上追加重型 ML 依赖，支撑三层分类漏斗：
#   - torch + transformers : Layer-3 本地 LLM 推理（Qwen3.5 分类模型）
#   - onnxruntime          : Layer-2 Small-NER 推理
# 适用场景：需要完整三层分类（规则 → NER → LLM），且有足够内存/GPU 资源
FROM core AS ml

# 追加系统级依赖需要 root 权限（core 阶段末尾已切换为 privacy 用户，此处切回 root）
USER root
COPY requirements-ml.txt .
# 安装 ML 依赖并清理字节码缓存：
#   - --no-cache-dir : 不缓存下载的 wheel 包，减小镜像体积
#   - find -exec rm  : 清理 site-packages 下的 __pycache__（体积优化，
#                      find 失败不影响构建——2>/dev/null + || true 容错）
RUN pip install --no-cache-dir -r requirements-ml.txt -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && find /usr/local/lib/python*/site-packages -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
# 重新切回非 root 用户（与 core 阶段的安全基线保持一致）
USER privacy

# ml 镜像默认启动命令（与 core 相同：REST + gRPC 一体进程）
CMD ["python", "-m", "engine.server"]

