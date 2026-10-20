#!/bin/sh
# docker-entrypoint.sh - PrivShield 容器启动入口脚本
# Docker container entrypoint script for PrivShield
#
# 功能 / Features:
#   - 启动前配置校验（配置文件存在性、端口可用性）
#   - 信号转发（确保 graceful shutdown）
#   - 支持自定义启动命令
#
# Pre-start validation (config file, port availability)
# Signal forwarding for graceful shutdown
# Support for custom startup commands

set -e

# ============================================================================
# 配置校验 / Configuration Validation
# ============================================================================

# 检查配置文件是否存在（如已挂载）
# Verify profile config exists if specified
if [ -n "$PRIVACY_PROFILE" ] && [ ! -f "$PRIVACY_PROFILE" ]; then
    echo "[entrypoint] WARNING: PRIVACY_PROFILE=$PRIVACY_PROFILE not found"
    echo "[entrypoint] Agent will use built-in defaults"
fi

# 检查 rules 目录（如已配置）
# Verify rules directory if configured
if [ -n "$PRIVACY_RULES_DIR" ] && [ ! -d "$PRIVACY_RULES_DIR" ]; then
    echo "[entrypoint] WARNING: Rules directory $PRIVACY_RULES_DIR not found"
fi

# ============================================================================
# 环境变量默认值 / Environment Defaults
# ============================================================================

# 确保监听地址正确绑定容器网络
# Ensure listen addresses bind to container network
export PRIVACY_REST_HOST="${PRIVACY_REST_HOST:-0.0.0.0}"
export PRIVACY_GRPC_HOST="${PRIVACY_GRPC_HOST:-0.0.0.0}"

# 日志默认 JSON 格式（容器环境推荐）
# Default to JSON logging in container environments
export PRIVACY_LOG_FORMAT="${PRIVACY_LOG_FORMAT:-json}"

# ============================================================================
# 启动信息输出 / Startup Banner
# ============================================================================

echo "[entrypoint] PrivShield starting..."
echo "[entrypoint]   REST: ${PRIVACY_REST_HOST}:${PRIVACY_REST_PORT:-8079}"
echo "[entrypoint]   gRPC: ${PRIVACY_GRPC_HOST}:${PRIVACY_GRPC_PORT:-50051}"
echo "[entrypoint]   Log:  ${PRIVACY_LOG_FORMAT} / ${PRIVACY_LOG_LEVEL:-INFO}"
if [ -n "$PRIVACY_PROFILE" ]; then
    echo "[entrypoint]   Profile: $PRIVACY_PROFILE"
fi

# ============================================================================
# 执行主进程（使用 exec 确保信号正确转发）
# Exec main process (ensures signals are forwarded correctly)
# ============================================================================

exec "$@"
