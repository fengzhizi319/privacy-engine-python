# ==============================================================================
# PrivShield Python 引擎 TLCP / mTLS 终结器
# ==============================================================================
# 构建上下文是**本服务目录**（privacy-engine-python）：
#   docker build -f terminator/Dockerfile -t privshield-python-terminator:2.0.0 .
# 或在本目录执行 `make docker-terminator`。
#
# 终结器是唯一的 TLS 边界：对外提供 tlcp / both / tls 三档 mTLS，
# 对内以回环明文访问 Python 引擎，并注入 HMAC 签名的客户端证书身份头。
# 依赖的 Go pkg/ 子集已随仓库 vendor 在 terminator/third_party/pkg。
# ==============================================================================
FROM golang:1.26-alpine AS builder

WORKDIR /build
ENV CGO_ENABLED=0
ENV GOWORK=off
ENV GOPROXY=https://goproxy.cn,direct

# 先复制依赖清单命中 Docker 层缓存；replace 指向 third_party/pkg，
# 因此其 go.mod/go.sum 也必须先就位。
COPY terminator/go.mod terminator/go.sum ./terminator/
COPY terminator/third_party/pkg/go.mod terminator/third_party/pkg/go.sum \
     ./terminator/third_party/pkg/
RUN cd terminator && go mod download

COPY terminator/third_party/pkg/ ./terminator/third_party/pkg/
COPY terminator/ ./terminator/

ARG VERSION=dev
RUN cd terminator && \
    go build -trimpath -ldflags="-s -w -X 'main.Version=${VERSION}'" \
    -o /build/privshield-python-terminator .

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata \
    && addgroup -S terminator \
    && adduser -S terminator -G terminator

WORKDIR /app
COPY --from=builder /build/privshield-python-terminator /app/privshield-python-terminator

# 终结器不落盘、不写日志文件；只读根文件系统也可运行（/tmp 可按需挂 tmpfs）。
USER terminator

# 对外业务口 REST / gRPC 与运维口。Python 回环端口（18079/15051/19079）不得发布。
EXPOSE 8079 50051 9079

# 健康检查刻意不放在镜像里：运维口默认是 mTLS（both/tlcp），
# 镜像内置的 wget/curl 无法在未知证书材料下正确验证；由 compose/k8s
# 按实际协议栈配置探针（见 deploy/docker-compose.terminator.yml 与 README）。
ENTRYPOINT ["/app/privshield-python-terminator"]
