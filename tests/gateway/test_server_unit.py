"""网关配置与 gRPC 监听器单元测试 (Gateway settings & gRPC listener).

覆盖以下测试点：
1. **GatewaySettings.from_env 配置加载**：
   - 默认值对齐 Go：127.0.0.1:8000 / :50000 / ops=业务口+1000 / p2c / tlcp；
   - ``PRIVACY_GATEWAY_*`` 环境变量覆盖；
   - ``TLS_STACK`` 枚举校验与缺证书 Fail-Fast；
   - 非环回绑定 + TLCP 档拒启（Python 无法终止国密）；
   - 生产姿态缺 inbound keyfile 拒启。
2. **start_grpc_gateway 服务启动器**：
   - 明文（终结器后环回）模式成功创建 grpc.aio.Server 并正常停止；
   - 启用 TLS/mTLS 时通过 grpc.ssl_server_credentials 正常启动。
"""

from __future__ import annotations

import socket
import pytest
from fastapi.testclient import TestClient

from engine.gateway.balancer import LoadBalancer
from engine.gateway.grpc_proxy import start_grpc_gateway
from engine.gateway.server import create_ops_app
from engine.gateway.settings import GatewaySettings


def find_free_port() -> int:
    """获取本地可用端口。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def _clean_env(monkeypatch):
    """清理所有网关相关的环境变量。"""
    for var in (
        "APP_ENV",
        "PRIVACY_GATEWAY_ENV",
        "PRIVACY_GATEWAY_HOST",
        "PRIVACY_GATEWAY_PORT",
        "PRIVACY_GATEWAY_GRPC_HOST",
        "PRIVACY_GATEWAY_GRPC_PORT",
        "PRIVACY_GATEWAY_OPS_PORT",
        "PRIVACY_GATEWAY_STRATEGY",
        "PRIVACY_GATEWAY_HEALTH_CHECK_INTERVAL",
        "PRIVACY_GATEWAY_TLS_STACK",
        "PRIVACY_GATEWAY_TLS_CERT_FILE",
        "PRIVACY_GATEWAY_TLS_KEY_FILE",
        "PRIVACY_GATEWAY_TLS_CA_FILE",
        "PRIVACY_GATEWAY_OPS_TLS_STACK",
        "PRIVACY_GATEWAY_OPS_TLS_CERT_FILE",
        "PRIVACY_GATEWAY_OPS_TLS_KEY_FILE",
        "PRIVACY_GATEWAY_OPS_TLS_CA_FILE",
        "PRIVACY_GATEWAY_BACKENDS",
        "PRIVACY_GATEWAY_GRPC_BACKENDS",
        "PRIVACY_GATEWAY_AUTH_ENABLED",
        "PRIVACY_GATEWAY_BACKEND_API_KEY",
        "PRIVACY_GATEWAY_INBOUND_KEYS_FILE",
        "PRIVACY_GATEWAY_METRICS_API_KEY",
        "PRIVACY_GATEWAY_ADMIN_API_KEY",
        "PRIVACY_GATEWAY_ALLOWED_CIDRS",
        "PRIVACY_GATEWAY_TRUSTED_PROXIES",
        "PRIVACY_GATEWAY_REQUIRE_TLS",
        "PRIVACY_GATEWAY_REQUIRE_FORWARDED_PROTO",
        "PRIVACY_GATEWAY_MAX_BODY_BYTES",
        "PRIVACY_GATEWAY_BACKEND_TLS_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)


def test_settings_defaults(_clean_env):
    """默认值与 Go internal/config.LoadGateway 一致。"""
    settings = GatewaySettings.from_env()
    assert settings.rest_host == "127.0.0.1"
    assert settings.rest_port == 8000
    assert settings.grpc_host == "127.0.0.1"
    assert settings.grpc_port == 50000
    assert settings.ops_port == 9000
    assert settings.strategy == "p2c"
    assert settings.health_check_interval == 5.0
    assert settings.tls_stack == "tlcp"
    assert settings.tls_enabled is False
    assert settings.ops_tls_stack == "both"
    assert settings.rest_backends == ("http://127.0.0.1:8079",)
    assert settings.grpc_backends == ("127.0.0.1:50051",)


def test_settings_env_overrides(_clean_env, monkeypatch):
    """PRIVACY_GATEWAY_* 环境变量覆盖默认值。"""
    monkeypatch.setenv("PRIVACY_GATEWAY_HOST", "127.0.0.1")
    monkeypatch.setenv("PRIVACY_GATEWAY_PORT", "9000")
    monkeypatch.setenv("PRIVACY_GATEWAY_GRPC_PORT", "60000")
    monkeypatch.setenv("PRIVACY_GATEWAY_STRATEGY", "weighted_rr")
    monkeypatch.setenv("PRIVACY_GATEWAY_HEALTH_CHECK_INTERVAL", "2.5")
    monkeypatch.setenv(
        "PRIVACY_GATEWAY_BACKENDS",
        "http://agent-1:8079,agent-2:8080",
    )
    monkeypatch.setenv(
        "PRIVACY_GATEWAY_GRPC_BACKENDS",
        "agent-1:50051,agent-2:50052",
    )

    settings = GatewaySettings.from_env()
    assert settings.rest_port == 9000
    assert settings.ops_port == 10000
    assert settings.grpc_port == 60000
    assert settings.strategy == "weighted_rr"
    assert settings.health_check_interval == 2.5
    assert settings.rest_backends == ("http://agent-1:8079", "http://agent-2:8080")
    assert settings.grpc_backends == ("agent-1:50051", "agent-2:50052")


@pytest.mark.parametrize("stack", ["ssl", "", "TLS1.2"])
def test_settings_rejects_unknown_tls_stack(_clean_env, monkeypatch, stack):
    monkeypatch.setenv("PRIVACY_GATEWAY_TLS_STACK", stack)
    with pytest.raises(ValueError, match="PRIVACY_GATEWAY_TLS_STACK"):
        GatewaySettings.from_env()


def test_settings_tls_stack_requires_cert_key_ca(_clean_env, monkeypatch):
    monkeypatch.setenv("PRIVACY_GATEWAY_TLS_STACK", "tls")
    with pytest.raises(ValueError, match="requires PRIVACY_GATEWAY_TLS_CERT_FILE"):
        GatewaySettings.from_env()

    monkeypatch.setenv("PRIVACY_GATEWAY_TLS_CERT_FILE", "/tmp/cert.pem")
    monkeypatch.setenv("PRIVACY_GATEWAY_TLS_KEY_FILE", "/tmp/key.pem")
    with pytest.raises(ValueError, match="PRIVACY_GATEWAY_TLS_CA_FILE is required"):
        GatewaySettings.from_env()


def test_settings_tlcp_requires_loopback_bind(_clean_env, monkeypatch):
    monkeypatch.setenv("PRIVACY_GATEWAY_HOST", "0.0.0.0")
    monkeypatch.setenv("PRIVACY_GATEWAY_AUTH_ENABLED", "true")
    monkeypatch.setenv("PRIVACY_GATEWAY_METRICS_API_KEY", "metrics-key")
    with pytest.raises(ValueError, match="cannot be terminated by the Python gateway"):
        GatewaySettings.from_env()


def test_settings_non_loopback_requires_credential(_clean_env, monkeypatch):
    monkeypatch.setenv("PRIVACY_GATEWAY_TLS_STACK", "tls")
    monkeypatch.setenv("PRIVACY_GATEWAY_HOST", "0.0.0.0")
    monkeypatch.setenv("PRIVACY_GATEWAY_TLS_CERT_FILE", "/tmp/cert.pem")
    monkeypatch.setenv("PRIVACY_GATEWAY_TLS_KEY_FILE", "/tmp/key.pem")
    monkeypatch.setenv("PRIVACY_GATEWAY_TLS_CA_FILE", "/tmp/ca.pem")
    # 运维口同样非环回，必须显式声明标准 TLS 终结
    monkeypatch.setenv("PRIVACY_GATEWAY_OPS_TLS_STACK", "tls")
    monkeypatch.setenv("PRIVACY_GATEWAY_OPS_TLS_CERT_FILE", "/tmp/ops-cert.pem")
    monkeypatch.setenv("PRIVACY_GATEWAY_OPS_TLS_KEY_FILE", "/tmp/ops-key.pem")
    monkeypatch.setenv("PRIVACY_GATEWAY_OPS_TLS_CA_FILE", "/tmp/ops-ca.pem")
    with pytest.raises(ValueError, match="requires PRIVACY_GATEWAY_AUTH_ENABLED"):
        GatewaySettings.from_env()


def test_settings_production_requires_inbound_keyfile(_clean_env, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(ValueError, match="PRIVACY_GATEWAY_INBOUND_KEYS_FILE"):
        GatewaySettings.from_env()


def test_ops_readyz_reports_healthy_backends(_clean_env):
    """运维口 /readyz 在存在健康后端时返回 200 与全量诊断字段。"""
    settings = GatewaySettings(metrics_api_key="metrics-key")
    balancer = LoadBalancer(strategy=settings.strategy)
    balancer.add_node(http_url="http://127.0.0.1:8079", grpc_address="")
    client = TestClient(create_ops_app(settings, balancer))

    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["component"] == "gateway"
    assert body["strategy"] == settings.strategy
    assert body["tls_stack"] == settings.tls_stack
    assert body["metrics_auth_mode"] == "bearer"
    assert body["rest_backends"] == 1
    assert body["rest_healthy"] == 1


def test_ops_readyz_503_when_all_backends_unhealthy(_clean_env):
    """所有后端被动摘除后 /readyz 返回 503，供探活熔断。"""
    settings = GatewaySettings()
    balancer = LoadBalancer(strategy=settings.strategy)
    balancer.add_node(http_url="http://127.0.0.1:8079", grpc_address="")
    balancer.nodes[0].mark_unhealthy(cooldown_seconds=60)
    client = TestClient(create_ops_app(settings, balancer))

    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unavailable"
    assert resp.json()["rest_healthy"] == 0


def test_ops_backends_and_metrics_require_bearer_keys(_clean_env):
    """配置运维 Key 后，/gateway/backends 与 /metrics 均拒绝无凭据访问。"""
    settings = GatewaySettings(
        metrics_api_key="metrics-key", admin_api_key="admin-key"
    )
    balancer = LoadBalancer(strategy=settings.strategy)
    balancer.add_node(http_url="http://127.0.0.1:8079", grpc_address="")
    client = TestClient(create_ops_app(settings, balancer))

    assert client.get("/gateway/backends").status_code == 401
    assert client.get("/metrics").status_code == 401
    assert (
        client.get(
            "/gateway/backends", headers={"Authorization": "Bearer admin-key"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/metrics", headers={"Authorization": "Bearer metrics-key"}
        ).status_code
        == 200
    )
    # 两把 Key 不互通
    assert (
        client.get(
            "/gateway/backends", headers={"Authorization": "Bearer metrics-key"}
        ).status_code
        == 401
    )


@pytest.mark.anyio
async def test_start_grpc_gateway_insecure_lifecycle():
    """测试终结器后的环回明文 gRPC 网关成功启动、绑定端口并正常停止。"""
    port = find_free_port()
    settings = GatewaySettings(grpc_port=port)
    balancer = LoadBalancer()
    server = await start_grpc_gateway(settings, balancer)
    assert server is not None
    await server.stop(grace=0.1)


@pytest.mark.anyio
async def test_start_grpc_gateway_tls_lifecycle(tmp_path):
    """测试自签名证书下的 TLS/mTLS gRPC 网关启动与停止。"""
    import subprocess
    import os

    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    ca = tmp_path / "ca.crt"
    ca_key = tmp_path / "ca.key"
    clean_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}

    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-days", "1", "-nodes",
            "-keyout", str(ca_key), "-out", str(ca),
            "-subj", "/CN=TestCA",
        ],
        check=True,
        capture_output=True,
        env=clean_env,
    )

    subprocess.run(
        [
            "openssl", "req", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(tmp_path / "server.csr"),
            "-subj", "/CN=localhost",
        ],
        check=True,
        capture_output=True,
        env=clean_env,
    )

    subprocess.run(
        [
            "openssl", "x509", "-req", "-days", "1",
            "-in", str(tmp_path / "server.csr"),
            "-CA", str(ca), "-CAkey", str(ca_key), "-CAcreateserial",
            "-out", str(cert),
        ],
        check=True,
        capture_output=True,
        env=clean_env,
    )

    port = find_free_port()
    settings = GatewaySettings(
        grpc_port=port,
        tls_stack="tls",
        tls_cert_file=str(cert),
        tls_key_file=str(key),
        tls_ca_file=str(ca),
    )
    balancer = LoadBalancer()
    server = await start_grpc_gateway(settings, balancer)
    assert server is not None
    await server.stop(grace=0.1)
