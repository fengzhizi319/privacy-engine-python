"""Tests for service-hub scripts under services/service-hub/scripts/.

==============================================================================
数盾 PrivShield · 数据服务调度中枢 (service-hub) 运维与启动脚本测试套件
==============================================================================

一、测试范围 (Scope):
1. 静态检查 (Static & Syntax Checks):
   - 验证 services/service-hub/scripts/ 下所有 Shell 脚本与根目录 run.sh 存在且具备可执行权限
   - 使用 bash -n 验证所有脚本的语法合规性
   - 检查持久化证书目录 (services/service-hub/certs/) 证书与公钥完整性

2. 命令行帮助与选项测试 (CLI Help & Parameter Tests):
   - deploy.sh --help
   - deploy-k8s.sh --help / -h
   - stop-k8s.sh --help / -h
   - stop-docker.sh --help / -h
   - gen-certs.sh --help / -h
   - health-check.sh --help / -h
   - simulate-pipeline.sh --help / -h
   - run.sh --help / -h

3. 证书生成流程测试 (Dynamic mTLS Certificate Generation):
   - 在 tmp_path 隔离目录下运行 gen-certs.sh
   - 验证 ca.crt, ca.key, server.crt, server.key, client.crt, client.key, client.pub 完整生成
   - 验证私钥权限收紧 (0600) 与证书有效性

4. Kubernetes 部署脚本演练测试 (K8s Deploy Dry-Run):
   - deploy-k8s.sh --dry-run
   - deploy-k8s.sh --dry-run --with-postgres

5. 服务健康探活与模拟器容错测试 (Health Check & Simulation Error Handling):
   - health-check.sh 在服务未运行时安全输出 FAILED (unreachable) 且不崩溃
   - simulate-pipeline.sh 在服务未运行时输出友好错误并返回退出码 1
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SERVICE_HUB_DIR = PROJECT_ROOT / "services" / "service-hub"
SCRIPTS_DIR = SERVICE_HUB_DIR / "scripts"


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


@pytest.fixture
def bash_bin() -> str:
    """获取系统可用的 bash 解释器路径。"""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("系统缺少 bash 解释器")
    return bash


class TestServiceHubScriptsStaticChecks:
    """静态文件存在性、权限与语法规范检查。"""

    def test_all_shell_scripts_exist_and_executable(self):
        """验证所有 service-hub Shell 脚本存在且具有执行权限。"""
        expected_scripts = [
            "deploy.sh",
            "deploy-k8s.sh",
            "stop-k8s.sh",
            "stop-docker.sh",
            "gen-certs.sh",
            "health-check.sh",
            "simulate-pipeline.sh",
        ]
        for name in expected_scripts:
            script_path = SCRIPTS_DIR / name
            assert script_path.is_file(), f"service-hub 脚本不存在: {script_path}"
            if sys.platform != "win32":
                assert script_path.stat().st_mode & stat.S_IXUSR, f"脚本缺少执行权限: {script_path}"

        root_run_sh = SERVICE_HUB_DIR / "run.sh"
        assert root_run_sh.is_file(), f"service-hub 快捷入口不存在: {root_run_sh}"
        if sys.platform != "win32":
            assert root_run_sh.stat().st_mode & stat.S_IXUSR, f"run.sh 缺少执行权限: {root_run_sh}"

    def test_bash_syntax_validation(self, bash_bin: str):
        """使用 bash -n 验证所有 Shell 脚本无语法错误。"""
        scripts_to_check = list(SCRIPTS_DIR.glob("*.sh")) + [SERVICE_HUB_DIR / "run.sh"]
        for script_path in scripts_to_check:
            if not script_path.is_file():
                continue
            res = subprocess.run(
                [bash_bin, "-n", str(script_path)],
                capture_output=True,
                text=True,
                env=_clean_env(),
            )
            assert res.returncode == 0, f"脚本语法解析错误: {script_path.name}\n{res.stderr}"

    def test_persisted_certs_directory_integrity(self):
        """验证 services/service-hub/certs/ 目录中持久化证书链与公钥完整性。"""
        certs_dir = SERVICE_HUB_DIR / "certs"
        assert certs_dir.is_dir(), f"证书目录不存在: {certs_dir}"
        expected_files = [
            "ca.crt",
            "ca.key",
            "server.crt",
            "server.key",
            "client.crt",
            "client.key",
            "client.pub",
        ]
        for cert_file in expected_files:
            file_path = certs_dir / cert_file
            assert file_path.is_file(), f"持久化证书文件缺失: {file_path}"
            assert file_path.stat().st_size > 0, f"证书文件为空: {file_path}"


class TestServiceHubScriptsHelpOptions:
    """测试各脚本的 --help 与 -h 选项响应。"""

    @pytest.mark.parametrize(
        ("script_relpath", "keyword"),
        [
            ("scripts/deploy.sh", "SERVICE_HUB_IMAGE"),
            ("scripts/deploy-k8s.sh", "Kubernetes"),
            ("scripts/stop-k8s.sh", "Kubernetes"),
            ("scripts/stop-docker.sh", "SERVICE_HUB_CONTAINER"),
            ("scripts/gen-certs.sh", "CERT_DAYS"),
            ("scripts/health-check.sh", "SERVICE_HUB_HOST"),
            ("scripts/simulate-pipeline.sh", "SERVICE_HUB_URL"),
            ("run.sh", "SERVICE_HUB_PORT"),
        ],
    )
    def test_script_help_options(self, bash_bin: str, script_relpath: str, keyword: str):
        """验证所有脚本在接收 --help 时退出码为 0 且输出包含关键提示词。"""
        script_path = SERVICE_HUB_DIR / script_relpath
        for flag in ["--help", "-h"]:
            res = subprocess.run(
                [bash_bin, str(script_path), flag],
                capture_output=True,
                text=True,
                env=_clean_env(),
            )
            assert res.returncode == 0, f"{script_path.name} {flag} 失败 (退出码 {res.returncode}):\n{res.stderr}"
            assert keyword in res.stdout, f"{script_path.name} 输出缺少关键字 '{keyword}':\n{res.stdout}"


class TestServiceHubDynamicCertsGeneration:
    """测试动态生成 mTLS 测试证书链流程。"""

    def test_gen_certs_in_temp_directory(self, bash_bin: str, tmp_path: Path):
        """验证 gen-certs.sh 在临时隔离目录下可正确生成完整证书链与公钥。"""
        if not shutil.which("openssl"):
            pytest.skip("系统未安装 openssl 命令")

        output_dir = tmp_path / "custom_certs"
        gen_certs_script = SCRIPTS_DIR / "gen-certs.sh"

        env = _clean_env()
        env["CERT_DAYS"] = "30"

        res = subprocess.run(
            [bash_bin, str(gen_certs_script), str(output_dir)],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode == 0, f"gen-certs.sh 执行失败:\n{res.stdout}\n{res.stderr}"

        # 验证生成的 7 个核心密钥与证书文件
        expected_files = [
            "ca.crt",
            "ca.key",
            "server.crt",
            "server.key",
            "client.crt",
            "client.key",
            "client.pub",
        ]
        for f in expected_files:
            file_path = output_dir / f
            assert file_path.is_file(), f"未生成预期文件: {file_path}"
            assert file_path.stat().st_size > 0, f"生成文件为空: {file_path}"

        # 验证 client.pub 包含公钥 PEM 格式标识
        client_pub_content = (output_dir / "client.pub").read_text(encoding="utf-8")
        assert "-----BEGIN PUBLIC KEY-----" in client_pub_content
        assert "-----END PUBLIC KEY-----" in client_pub_content

        # 验证私钥文件权限为 0600 (仅属主读写)
        if sys.platform != "win32":
            for key_file in ["ca.key", "server.key", "client.key"]:
                mode = stat.S_IMODE((output_dir / key_file).stat().st_mode)
                assert mode == 0o600, f"{key_file} 权限未收紧至 0600 (当前: {oct(mode)})"


class TestServiceHubK8sDeployDryRun:
    """测试 Kubernetes 部署脚本演练功能。"""

    def test_k8s_deploy_dry_run(self, bash_bin: str):
        """测试 deploy-k8s.sh 的 --dry-run 演练模式。"""
        kubectl = shutil.which("kubectl")
        if not kubectl:
            pytest.skip("系统未安装 kubectl 命令")

        deploy_k8s_script = SCRIPTS_DIR / "deploy-k8s.sh"

        # 1. 基础单服务演练
        res = subprocess.run(
            [bash_bin, str(deploy_k8s_script), "--dry-run", "-n", "test-ns"],
            capture_output=True,
            text=True,
            env=_clean_env(),
        )
        assert res.returncode == 0, f"deploy-k8s.sh --dry-run 失败:\n{res.stdout}\n{res.stderr}"
        assert "演练模式" in res.stdout or "dry-run" in res.stdout.lower()

        # 2. 包含 Phase B PostgreSQL 的联合演练
        res_pg = subprocess.run(
            [bash_bin, str(deploy_k8s_script), "--dry-run", "--with-postgres", "-n", "test-ns"],
            capture_output=True,
            text=True,
            env=_clean_env(),
        )
        assert res_pg.returncode == 0, f"deploy-k8s.sh --dry-run --with-postgres 失败:\n{res_pg.stdout}\n{res_pg.stderr}"
        assert "PostgreSQL" in res_pg.stdout


class TestServiceHubHealthCheckAndSimulator:
    """测试健康探活脚本与流水线模拟器的执行与异常处理。"""

    def test_health_check_unreachable_target(self, bash_bin: str):
        """验证 health-check.sh 在后端未运行时安全报告 FAILED (unreachable) 并不崩溃。"""
        health_check_script = SCRIPTS_DIR / "health-check.sh"
        env = _clean_env()
        # 指向一个绝对未监听的端口
        env["SERVICE_HUB_HOST"] = "127.0.0.1"
        env["SERVICE_HUB_PORT"] = "59999"

        res = subprocess.run(
            [bash_bin, str(health_check_script)],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode == 0, f"health-check.sh 在离线探测时应正常退出返回 0:\n{res.stdout}\n{res.stderr}"
        assert "FAILED (unreachable)" in res.stdout or "FAILED" in res.stdout

    def test_simulate_pipeline_unreachable_target(self, bash_bin: str):
        """验证 simulate-pipeline.sh 在后端未运行时输出错误提示并返回非 0 退出码。"""
        sim_script = SCRIPTS_DIR / "simulate-pipeline.sh"
        env = _clean_env()
        env["SERVICE_HUB_URL"] = "http://127.0.0.1:59999"

        res = subprocess.run(
            [bash_bin, str(sim_script), "5"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode != 0, "simulate-pipeline.sh 在目标离线时应返回失败退出码"
        assert "错误: Service Hub 未在" in res.stdout
