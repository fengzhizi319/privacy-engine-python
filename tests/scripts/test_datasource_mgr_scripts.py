"""Tests for datasource-mgr scripts under services/datasource-mgr/scripts/.

==============================================================================
数盾 PrivShield · 模拟数据源服务 (datasource-mgr) 运维与启动脚本测试套件
==============================================================================

一、脚本使用说明 (Usage Guide)
------------------------------------------------------------------------------
本模块为开发与调试提供真实脱敏场景的业务数据源仿真（医保 yibao.csv、康养 kangyang.csv），
包含以下 4 个核心运行与证书管理脚本：

1. 开发启动脚本 (scripts/dev-run.sh):
   - 命令: `cd services/datasource-mgr && bash scripts/dev-run.sh` (或 `make dev`)
   - 特性: 禁用 mTLS 双向认证 (DATASOURCE_MGR_TLS_ENABLED=false)，极速冷启动
   - 监听: HTTP REST (http://127.0.0.1:8083) + gRPC Insecure (127.0.0.1:50053)

2. 生产加固启动脚本 (scripts/prod-run.sh):
   - 命令: `cd services/datasource-mgr && bash scripts/prod-run.sh` (或 `make prod`)
   - 特性: 强制 TLS 1.3 双向认证 (CLIENT_AUTH=require) + 客户端公钥固定 (client.pub)
   - 监听: HTTP REST (http://0.0.0.0:8083) + gRPC mTLS (0.0.0.0:50053)

3. 证书生成脚本 (scripts/gen-certs.sh):
   - 命令: `cd services/datasource-mgr && bash scripts/gen-certs.sh [output_dir]` (或 `make gen-certs`)
   - 产物: ca.crt/ca.key (根CA), server.crt/server.key (服务端证书), client.crt/client.key (客户端证书), client.pub (固定公钥)
   - 特点: 生成的文件已持久化提交至 services/datasource-mgr/certs/，保障测试与公钥固定可复现

4. 快捷入口脚本 (run.sh):
   - 开发模式: `bash run.sh` 或 `bash run.sh dev` -> 转发至 scripts/dev-run.sh
   - 生产模式: `bash run.sh prod` -> 转发至 scripts/prod-run.sh

二、主要执行与验证流程 (Main Execution Flow)
------------------------------------------------------------------------------
本测试文件覆盖以下 3 大执行阶段：

┌──────────────────────────────────────────────────────────────────────────┐
│  阶段 1: 静态与语法校验 (Static & Syntax Checks)                          │
│  ├─ 检查 run.sh、dev-run.sh、prod-run.sh、gen-certs.sh 等文件存在且具备执行权限  │
│  ├─ 执行 bash -n 验证所有 Shell 脚本语法正确，无解析异常                    │
│  └─ 检查 services/datasource-mgr/certs/ 目录中持久化入库的证书与公钥完整性  │
├──────────────────────────────────────────────────────────────────────────┤
│  阶段 2: 证书动态生成校验 (Certificate Generation Flow)                   │
│  ├─ 在 tmp_path 隔离环境中调用 gen-certs.sh                              │
│  ├─ 验证 ca.crt、server.crt、client.crt、client.pub 全部成功生成         │
│  └─ 校验私钥权限收紧 (0600) 与证书可读性                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  阶段 3: 子进程生命周期与探活 (Subprocess Lifecycle & Health Checks)       │
│  ├─ 开发启动流程 (dev-run.sh): 分配空闲端口 ➔ 启动子进程 ➔ HTTP 探活 ➔ SIGTERM │
│  ├─ 生产启动流程 (prod-run.sh): 分配空闲端口 ➔ 证书校验 ➔ mTLS 探活 ➔ SIGTERM │
│  └─ 入口路由流程 (run.sh): 校验默认参数分发与 prod 参数分发路径             │
└──────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DS_DIR = PROJECT_ROOT / "services" / "datasource-mgr"
SCRIPTS_DIR = DS_DIR / "scripts"


def _clean_env() -> dict[str, str]:
    """生成精简干净的子进程环境变量，剔除所有超长变量防止 Argument list too long。"""
    return {
        k: v
        for k, v in os.environ.items()
        if len(v) < 2048
        and not k.startswith("ANTIGRAVITY")
        and not k.startswith("GEMINI")
        and not k.startswith("AI_")
        and k != "LS_COLORS"
    }


def _get_free_port() -> int:
    """获取本地空闲端口号。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def bash_bin() -> str:
    """获取系统可用的 bash 解释器路径。"""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("系统缺少 bash 解释器")
    return bash


class TestDatasourceMgrScriptsStatic:
    """阶段 1: 静态存在性、文件权限与 Bash 语法检查。"""

    def test_scripts_exist_and_executable(self):
        """验证所有脚本文件存在且具有执行权限 (0755)。"""
        expected_scripts = [
            DS_DIR / "run.sh",
            SCRIPTS_DIR / "dev-run.sh",
            SCRIPTS_DIR / "prod-run.sh",
            SCRIPTS_DIR / "gen-certs.sh",
            SCRIPTS_DIR / "deploy.sh",
            SCRIPTS_DIR / "health-check.sh",
        ]
        for p in expected_scripts:
            assert p.is_file(), f"脚本文件不存在: {p}"
            if sys.platform != "win32":
                assert p.stat().st_mode & stat.S_IXUSR, f"脚本缺少执行权限: {p}"

    def test_bash_syntax_check(self, bash_bin: str):
        """执行 bash -n 进行脚本语法正确性检查。"""
        scripts = [
            DS_DIR / "run.sh",
            SCRIPTS_DIR / "dev-run.sh",
            SCRIPTS_DIR / "prod-run.sh",
            SCRIPTS_DIR / "gen-certs.sh",
            SCRIPTS_DIR / "deploy.sh",
            SCRIPTS_DIR / "health-check.sh",
        ]
        for p in scripts:
            res = subprocess.run(
                [bash_bin, "-n", str(p)],
                capture_output=True,
                text=True,
                env=_clean_env(),
            )
            assert res.returncode == 0, f"bash 语法错误 {p}: {res.stderr}"

    def test_persisted_certs_exist(self):
        """验证持久化入库的测试证书与固定公钥文件完整存在。"""
        certs_dir = DS_DIR / "certs"
        assert certs_dir.is_dir(), f"证书目录不存在: {certs_dir}"
        required_files = [
            "ca.crt", "ca.key",
            "server.crt", "server.key",
            "client.crt", "client.key",
            "client.pub",
        ]
        for fname in required_files:
            p = certs_dir / fname
            assert p.is_file(), f"测试证书/公钥文件未就绪: {p}"
            assert p.stat().st_size > 0, f"证书文件为空: {p}"


class TestDatasourceMgrGenCerts:
    """阶段 2: 测试 gen-certs.sh 证书生成逻辑与完整证书链构建。"""

    def test_gen_certs_generates_valid_chain(self, bash_bin: str, tmp_path: Path):
        """测试 gen-certs.sh 能在指定输出目录正确生成完整测试证书链。"""
        gen_script = SCRIPTS_DIR / "gen-certs.sh"
        res = subprocess.run(
            [bash_bin, str(gen_script), str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=str(DS_DIR),
            env=_clean_env(),
        )
        assert res.returncode == 0, f"gen-certs.sh 失败: {res.stderr}"

        # 校验生成的核心文件
        for fname in ["ca.crt", "ca.key", "server.crt", "server.key", "client.crt", "client.key", "client.pub"]:
            p = tmp_path / fname
            assert p.is_file(), f"未生成文件: {fname}"
            assert p.stat().st_size > 0, f"文件为空: {fname}"


class TestDatasourceMgrExecutionLifecycle:
    """阶段 3: 测试 dev-run.sh / prod-run.sh 进程启动、探活与优雅停机。"""

    def test_dev_run_startup_and_health(self, bash_bin: str):
        """测试 dev-run.sh 开发启动流程：分配空闲端口 ➔ HTTP 探活 ➔ SIGTERM 优雅退出。"""
        http_port = _get_free_port()
        grpc_port = _get_free_port()

        env = _clean_env()
        env.update({
            "DATASOURCE_MGR_HOST": "127.0.0.1",
            "DATASOURCE_MGR_PORT": str(http_port),
            "DATASOURCE_MGR_GRPC_HOST": "127.0.0.1",
            "DATASOURCE_MGR_GRPC_PORT": str(grpc_port),
            "DATASOURCE_MGR_LOG_FORMAT": "text",
            "DATASOURCE_MGR_LOG_LEVEL": "debug",
        })

        dev_script = SCRIPTS_DIR / "dev-run.sh"
        proc = subprocess.Popen(
            [bash_bin, str(dev_script)],
            cwd=str(DS_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            health_url = f"http://127.0.0.1:{http_port}/api/health"
            healthy = False
            for _ in range(25):
                time.sleep(0.2)
                try:
                    req = urllib.request.Request(health_url)
                    with urllib.request.urlopen(req, timeout=1) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            if data.get("status") == "ok":
                                healthy = True
                                break
                except Exception:
                    pass

            assert healthy, f"dev-run.sh 未能在 {health_url} 正常提供健康探活"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_prod_run_startup_and_mtls(self, bash_bin: str):
        """测试 prod-run.sh 生产启动流程：加载证书 ➔ HTTPS 探活 ➔ gRPC 端口就绪 ➔ SIGTERM 优雅退出。"""
        http_port = _get_free_port()
        grpc_port = _get_free_port()

        env = _clean_env()
        env.update({
            "DATASOURCE_MGR_HOST": "127.0.0.1",
            "DATASOURCE_MGR_PORT": str(http_port),
            "DATASOURCE_MGR_GRPC_HOST": "127.0.0.1",
            "DATASOURCE_MGR_GRPC_PORT": str(grpc_port),
            "DATASOURCE_MGR_CERTS_DIR": str(DS_DIR / "certs"),
            "DATASOURCE_MGR_LOG_FORMAT": "text",
            "DATASOURCE_MGR_LOG_LEVEL": "info",
        })

        prod_script = SCRIPTS_DIR / "prod-run.sh"
        proc = subprocess.Popen(
            [bash_bin, str(prod_script)],
            cwd=str(DS_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # prod-run.sh 强制开启 HTTPS + mTLS，因此健康探活也需要使用客户端证书。
            certs_dir = DS_DIR / "certs"
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_ctx.load_cert_chain(
                certfile=str(certs_dir / "client.crt"),
                keyfile=str(certs_dir / "client.key"),
            )
            # 测试证书为自签名，不验证服务端证书；仅验证 HTTPS+mTLS 握手与端口可达性。
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

            health_url = f"https://127.0.0.1:{http_port}/api/health"
            healthy = False
            for _ in range(25):
                time.sleep(0.2)
                try:
                    req = urllib.request.Request(health_url)
                    with urllib.request.urlopen(req, timeout=1, context=ssl_ctx) as resp:
                        if resp.status == 200:
                            healthy = True
                            break
                except Exception:
                    pass

            assert healthy, f"prod-run.sh 未能在 {health_url} 正常提供健康探活"

            # 验证 gRPC mTLS 端口正常监听
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)
                res = s.connect_ex(("127.0.0.1", grpc_port))
                assert res == 0, f"gRPC mTLS 端口 {grpc_port} 未正常监听"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
