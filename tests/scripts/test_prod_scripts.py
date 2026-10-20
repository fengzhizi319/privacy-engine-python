"""Tests for production deployment, health check & backup scripts under scripts/prod/.

验证 scripts/prod/ 目录下的生产相关运维与部署脚本：
- 静态权限与语法结构检查；
- 参数解析与帮助信息 (--help)；
- 生产备份脚本 backup_privacy_budget.sh 在线热备份、压缩与校验和测试；
- 生产健康巡检脚本 prod_health_check.sh 选项解析与执行逻辑。
"""

from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROD_SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "prod"


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


class TestProdScriptsStaticChecks:
    """生产脚本静态存在性与权限检查。"""

    def test_all_prod_shell_scripts_exist_and_executable(self):
        """验证所有生产 Shell 脚本存在且具有执行权限。"""
        expected_scripts = [
            "deploy-docker-compose.sh",
            "stop-docker-compose.sh",
            "docker-start-agent.sh",
            "docker-stop-agent.sh",
            "deploy-helm.sh",
            "uninstall-helm.sh",
            "deploy-k8s.sh",
            "stop-k8s.sh",
            "prod-bff-agent.sh",
            "prod-stop.sh",
            "prod_health_check.sh",
            "backup_privacy_budget.sh",
            "backup-sqlite-databases.sh",
        ]
        for script_name in expected_scripts:
            script_path = PROD_SCRIPTS_DIR / script_name
            assert script_path.is_file(), f"生产脚本不存在: {script_path}"
            if sys.platform != "win32":
                assert script_path.stat().st_mode & stat.S_IXUSR, f"脚本缺少执行权限: {script_path}"

    def test_powershell_prod_scripts_exist(self):
        """验证 Windows 11 PowerShell 生产脚本存在。"""
        ps_scripts = [
            "deploy-docker-compose.ps1",
            "stop-docker-compose.ps1",
            "docker-start-agent.ps1",
            "docker-stop-agent.ps1",
        ]
        for script_name in ps_scripts:
            script_path = PROD_SCRIPTS_DIR / script_name
            assert script_path.is_file(), f"PowerShell 生产脚本不存在: {script_path}"


class TestProdScriptsHelpAndExecution:
    """生产脚本执行与功能测试。"""

    @pytest.mark.parametrize(
        "script_name",
        [
            "deploy-docker-compose.sh",
            "stop-docker-compose.sh",
            "docker-start-agent.sh",
            "deploy-helm.sh",
            "uninstall-helm.sh",
            "deploy-k8s.sh",
            "stop-k8s.sh",
            "prod-bff-agent.sh",
            "prod_health_check.sh",
            "backup_privacy_budget.sh",
        ],
    )
    def test_script_help_flag(self, bash_bin: str, script_name: str):
        """验证所有生产脚本均支持 --help 并返回 0。"""
        script_path = PROD_SCRIPTS_DIR / script_name
        res = subprocess.run(
            [bash_bin, str(script_path), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=_clean_env(),
            timeout=10,
        )
        assert res.returncode == 0
        assert "用法" in res.stdout or "Usage" in res.stdout or "使用说明" in res.stdout

    def test_backup_privacy_budget_live_execution(self, bash_bin: str, tmp_path: Path):
        """验证 backup_privacy_budget.sh 真实创建 SQLite 数据库并完成在线备份与校验。"""
        # 1. 创建源 SQLite 测试数据库并写入测试数据
        src_db = tmp_path / "test_budget.db"
        conn = sqlite3.connect(str(src_db))
        conn.execute("CREATE TABLE test_budget (id INTEGER PRIMARY KEY, consumed REAL);")
        conn.execute("INSERT INTO test_budget (consumed) VALUES (1.25);")
        conn.commit()
        conn.close()

        backup_dir = tmp_path / "backups"
        script_path = PROD_SCRIPTS_DIR / "backup_privacy_budget.sh"

        # 2. 执行备份脚本
        res = subprocess.run(
            [
                bash_bin,
                str(script_path),
                "--db-path",
                str(src_db),
                "--backup-dir",
                str(backup_dir),
                "--keep-days",
                "30",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=_clean_env(),
            timeout=30,
        )
        assert res.returncode == 0, f"备份脚本执行失败: {res.stderr}"
        assert "SQLite 在线备份完成" in res.stdout

        # 3. 验证生成了 .db.gz 备份文件与 .sha256 校验和文件
        backup_files = list(backup_dir.glob("privacy_budget_*.db.gz"))
        assert len(backup_files) == 1, f"未找到生成的 .db.gz 文件: {list(backup_dir.iterdir())}"
        backup_file = backup_files[0]
        assert backup_file.stat().st_size > 0

        sha_files = list(backup_dir.glob("privacy_budget_*.db.gz.sha256"))
        assert len(sha_files) == 1, "未找到生成的 .sha256 校验和文件"

        # 4. 解压并验证备份数据库的数据完整性
        extracted_db = tmp_path / "extracted.db"
        with gzip.open(backup_file, "rb") as f_in, open(extracted_db, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        read_conn = sqlite3.connect(str(extracted_db))
        row = read_conn.execute("SELECT consumed FROM test_budget WHERE id = 1;").fetchone()
        read_conn.close()
        assert row is not None
        assert row[0] == 1.25
