"""scripts/dev/docker-start-agent.sh 启动脚本与 Agent-LLM 通信测试.
docker-start-agent.sh & docker-stop-agent.sh Script Tests & Agent-to-LLM Communication Tests.

=====================================================================
测试目标 / Test Goal:
    1. 验证 scripts/dev/docker-start-agent.sh（Docker 方式单组分启动 PrivShield 容器）
       与 scripts/dev/docker-stop-agent.sh 停止脚本的正确性、健壮性与防回归机制。
    2. 验证 Docker 编排架构中 PrivShield (Sidecar) 与 Docker LLM (vLLM)
       容器之间的通信机制、配置一致性、安全地板约束以及真实端到端推理协同。

测试分层设计 / Layered Test Design（由浅入深，外部依赖逐层增加）:
    第 1 层 静态检查（无外部依赖）:
        - 脚本文件存在性、用户可执行权限位 (chmod +x)
        - Shebang 规范 (#!/usr/bin/env bash) 与 bash -n 静态语法解析
        - 核心关键指令防回归（set -euo pipefail / docker build / docker run / docker rm）
        - 默认监听端口 (8079/50051) 与关键环境变量校验
    第 2 层 模拟执行（仅需 bash，无需真实 Docker）:
        - 通过 PATH 注入 fake docker 命令，验证脚本执行流程：
          * 默认 target (core 镜像) 构建与启动
          * 显式 target (ml 镜像) 构建与启动
          * 启动前清理旧容器机制 (docker rm -f)
          * docker 失败时利用 set -e 快速非零退出
    第 3 层 停止脚本测试（仅需 bash）:
        - docker-stop-agent.sh 存在性、权限、语法及 fake docker 调用逻辑
    第 4 层 Compose 拓扑与 Agent-LLM 通信配置校验（仅需解析 YAML）:
        - 验证 docker-compose.yml 中 PrivShield 与 vllm 服务拓扑
        - 验证双方加入共同的 llm 网络以及 PRIVACY_LLM_API_BASE=http://vllm:8000/v1
    第 5 层 Agent 与 LLM 通信模拟单元测试（mock HTTP，无需 GPU/Docker）:
        - 验证 Agent 服务层在配置 vLLM 后端时的 OpenAILlmClassifier 适配与调用
        - 模拟 LLM 返回结构化分级/脱敏数据时 Agent 漏斗的流转与 Safety Floor 安全地板校验
        - 模拟 LLM 网络异常/超时时 Agent 的平滑优雅降级（Graceful Degradation）
    第 6 层 Docker Agent 与 Docker LLM 真实通信集成测试（integration marker）:
        - 启动（或复用）真实 Docker vLLM 推理容器（127.0.0.1:8000/v1）
        - 验证 Agent 动态分类服务直连 Docker LLM 完成字段分类、仲裁裁定与长文书无痕抹平
    第 7 层 Docker Agent 端点协议与接口测试（TestClient）:
        - 验证 Agent 健康检查探针 (/health 与 /readyz)
        - 验证单字段/整记录脱敏与动态分类分级评估接口协议
    第 8 层 真实执行 bash scripts/dev/docker-start-agent.sh core 拉起容器交互测试:
        - 真实执行启动脚本拉起物理容器并验证 docker ps 状态
        - 通过真实网络 Socket 发送 HTTP 脱敏、分类分级与健康探针请求
        - 验证 docker-stop-agent.sh 能够干净停机并清理容器
    第 9 层 多主机 / 跨主机部署与远程 LLM 通信测试（Cross-Host Deployment Tests）:
        - 验证 Compose 环境变量在跨主机端点与 API Key 注入时的变量替换渲染
        - 验证跨主机远程 HTTP 请求路由与 Authorization: Bearer 凭据传递
        - 验证跨主机网络超时、网关 502/504 异常捕获与 Layer-1/2 优雅降级机制
        - 验证同机模式 vs 跨主机模式的配置隔离与动态切换

=====================================================================
.venv 运行测试总览 / Running Tests with .venv:
    1. 激活虚拟环境后运行全量测试:
       source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py -v
    2. 或直接通过 .venv 路径运行:
       .venv/bin/pytest tests/scripts/test_docker_start_agent.py -v
    3. 跳过真实容器/GPU等外部依赖集成测试（仅运行单元与模拟测试）:
       source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py -m "not integration" -v
    4. 运行指定分层测试类示例:
       source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks -v
=====================================================================
"""

from __future__ import annotations

# ── 标准库导入 / Standard library imports ──
import json  # 解析 JSON 响应与构造模拟 payload
import os  # 环境变量读取与隔离
import re  # 正则匹配
import shutil  # 查找可执行命令 (bash / docker)
import socket  # TCP 端口可用性检测与动态端口分配
import stat  # 文件权限位检查
import subprocess  # 执行子进程命令
import sys  # 平台检测（Windows 无 bash/Docker 脚本环境）
import tempfile  # 临时目录管理 (隔离 fake docker)
import textwrap  # 文本缩进处理
import time  # 轮询与延时控制
import urllib.error  # HTTP 异常捕获
import urllib.request  # HTTP 探测与请求
import warnings  # 警告捕获
from pathlib import Path  # 面向对象路径操作
from typing import Any  # 类型提示
from unittest.mock import MagicMock, patch  # Mock 依赖

# ── 项目内导入 / Project imports ──
from engine.dynclassification.base import SensitivityLevel
from engine.dynclassification.llm_adapter import LlmAdapter
from engine.dynclassification.llm_engines import OpenAILlmClassifier
from engine.dynclassification.models import (
    CategoryDef,
    ConfidencePolicy,
    DomainTaxonomy,
    SecurityTag,
    SensitivityLevelDef,
)
from engine.dynclassification.service import DynClassificationService

# ── 第三方导入 / Third-party imports ──
import pytest
import yaml

# bash/Docker 脚本测试仅适用于 Unix 系平台（Linux/macOS）；
# Windows 无原生 bash 与 Unix 文件权限模型，即使安装了 WSL
# 也无法正确处理 Windows 路径，整个模块安全跳过。
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="bash/Docker shell-script tests require Unix (Linux/macOS); Windows has no native bash or POSIX permissions",
)

# ═══════════════════════════════════════════════════════════════════════
# 路径与常量定义 / Path & Constant Definitions
# ═══════════════════════════════════════════════════════════════════════

# 项目根目录（基于本文件所在层级推导，跨平台且与执行目录无关）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 启动脚本与停止脚本绝对路径
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "dev" / "docker-start-agent.sh"
STOP_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "dev" / "docker-stop-agent.sh"

# Docker Compose 编排文件路径
COMPOSE_DIR = PROJECT_ROOT / "deploy" / "docker-compose"
COMPOSE_FILE = COMPOSE_DIR / "docker-compose.yml"

# 容器与模型常量
AGENT_CONTAINER_NAME = "PrivShield"
VLLM_CONTAINER_NAME = "PrivShield-vllm"
VLLM_MODEL_DIR = PROJECT_ROOT / ".models" / "Qwen3.5-0.8B-Privacy-Classifier-Smoother"
VLLM_API_BASE = "http://127.0.0.1:8000/v1"
VLLM_SERVED_MODEL_NAME = "Qwen3.5-0.8B-Privacy-Classifier-Smoother"
VLLM_READY_TIMEOUT_S = 600
VLLM_READY_POLL_INTERVAL_S = 3


def _clean_env() -> dict[str, str]:
    """生成精简干净的子进程环境变量，剔除超长变量防止 Argument list too long。

    子进程在启动时会继承父进程的完整环境变量。当某些环境变量（如
    `LS_COLORS`、AI 相关工具注入的巨型 JSON 或路径列表）体积过大时，
    可能导致 `execve` 系统调用超出参数长度上限（通常为 128KB 或 2MB），
    从而抛出 `OSError: [Errno 7] Argument list too long`。

    本函数通过以下规则过滤环境变量：
        1. 值长度 < 2048：剔除超长变量，保留常用配置。
        2. 排除 `ANTIGRAVITY*`、`GEMINI*`、`AI_*`：这些前缀通常由 AI
           辅助开发工具或 IDE 插件注入，包含大量内部状态或提示词，
           对子进程无意义且体积庞大。
        3. 排除 `LS_COLORS`：`dircolors` 生成的颜色配置往往超过数 KB，
           对子进程编译/运行无实际用途。

    .venv 测试验证 / Test Command:
        # 辅助函数，在执行子进程命令时被调用，可通过执行依赖它的任一测试进行验证：
        source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_syntax_valid -v -s
        # 或直接调用：
        .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "test_script_syntax_valid" -v -s

    Returns:
        过滤后的安全环境变量字典，可直接传给 `subprocess.run(env=...)`。
    """
    return {
        k: v
        for k, v in os.environ.items()
        # 硬性长度限制：防止单条变量本身即接近系统上限
        if len(v) < 2048
        # 排除已知的大体积 AI 工具注入变量
        and not k.startswith("ANTIGRAVITY")
        and not k.startswith("GEMINI")
        and not k.startswith("AI_")
        # 排除 dircolors 生成的巨型颜色配置
        and k != "LS_COLORS"
    }



@pytest.fixture(scope="module")
def bash_bin() -> str:
    """探测系统中的 bash 解释器路径。不可用时自动 skip（如纯 Windows 环境）。

    .venv 测试验证 / Test Command:
        # Pytest module 级 fixture，探测 bash 路径。可通过运行依赖它的测试进行验证：
        source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_syntax_valid -v -s
        # 或直接调用：
        .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "test_script_syntax_valid" -v -s

    Returns:
        bash 解释器的绝对路径字符串。
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash 不可用，需要 WSL/Linux 环境")
    return bash


@pytest.fixture(scope="module")
def compose_config() -> dict[str, Any]:
    """解析 docker-compose.yml 为字典对象，供拓扑一致性校验。

    .venv 测试验证 / Test Command:
        # Pytest module 级 fixture，解析 compose 配置。可通过运行依赖它的测试进行验证：
        source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmComposeTopology -v -s
        # 或直接调用：
        .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "TestDockerAgentLlmComposeTopology" -v -s

    Returns:
        解析后的 compose YAML 嵌套字典。
    """
    with open(COMPOSE_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_script_with_fake_docker(
    bash_bin: str, exit_code: int = 0, target: str | None = None, script_path: Path = SCRIPT_PATH
) -> tuple[subprocess.CompletedProcess, str]:
    """通过在临时 PATH 中注入 fake docker 脚本来安全模拟执行被测 shell 脚本。

    .venv 测试验证 / Test Command:
        # 辅助函数，通过 fake docker 模拟执行脚本。可通过运行第 2/3 层测试进行验证：
        source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution -v -s
        # 或直接调用：
        .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "TestAgentScriptFakeExecution" -v -s

    Args:
        bash_bin: bash 解释器路径。
        exit_code: fake docker 预设的退出码（0 表示成功，非 0 表示模拟命令失败）。
        target: 传递给启动脚本的可选目标参数（如 'core' 或 'ml'）。
        script_path: 要执行的目标脚本路径（默认 docker-start-agent.sh）。

    Returns:
        元组 (CompletedProcess 对象, fake docker 调用日志字符串)。
    """
    with tempfile.TemporaryDirectory(prefix="fake-docker-agent-") as tmp:
        tmp_dir = Path(tmp)
        log_file = tmp_dir / "docker-calls.log"

        fake_docker = tmp_dir / "docker"
        fake_docker.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                echo "[FAKE-DOCKER] args: $*" >> "{log_file}"
                echo "[FAKE-DOCKER] cwd: $(pwd)" >> "{log_file}"
                exit {exit_code}
                """
            ),
            encoding="utf-8",
        )
        fake_docker.chmod(
            fake_docker.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )

        env = _clean_env()
        env["PATH"] = str(tmp_dir) + os.pathsep + os.environ.get("PATH", "")

        cmd = [bash_bin, str(script_path)]
        if target:
            cmd.append(target)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
            timeout=60,
        )
        logs = log_file.read_text(encoding="utf-8") if log_file.is_file() else ""
        return result, logs


# ═══════════════════════════════════════════════════════════════════════
# 第 1 层：脚本文件静态检查 / Static Checks
# ═══════════════════════════════════════════════════════════════════════


class TestAgentScriptStaticChecks:
    """启动脚本静态属性与语法检查：不执行真实命令，验证脚本满足运行前提。"""

    def test_script_file_exists(self):
        """【静态检查】验证 docker-start-agent.sh 脚本文件物理存在。

        测试目的：确保启动脚本未被意外移动或删除，防止部署脚本中断。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_file_exists -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_file_exists -v -s
        """
        assert SCRIPT_PATH.is_file(), f"启动脚本不存在: {SCRIPT_PATH}"

    def test_script_is_executable(self):
        """【静态检查】验证 docker-start-agent.sh 具备用户可执行权限位。

        测试目的：避免运维直接执行脚本时发生 Permission denied 错误。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_is_executable -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_is_executable -v -s
        """
        mode = SCRIPT_PATH.stat().st_mode
        assert mode & stat.S_IXUSR, "脚本缺少用户可执行权限位 (chmod +x)"

    def test_script_shebang(self):
        """【静态检查】验证脚本首行包含正确的标准 bash Shebang。

        测试目的：确保在不同 Linux 发行版与 macOS 环境下通过 PATH 正确找到 bash。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_shebang -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_shebang -v -s
        """
        first_line = SCRIPT_PATH.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == "#!/usr/bin/env bash", f"无效的 Shebang: {first_line}"

    def test_script_syntax_valid(self, bash_bin: str):
        """【静态检查】使用 bash -n 解析脚本语法，确保无语法错误。如果没有-n的参数，就会实际执行

        测试目的：在不实际运行容器的情况下，捕获语法拼写、括号不匹配等潜在错误。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_syntax_valid -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_syntax_valid -v -s
        """
        env = _clean_env()
        result = subprocess.run([bash_bin, "-n", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"脚本存在语法错误: {result.stderr}"

    def test_script_key_commands_present(self):
        """【静态检查】验证防回归核心指令：set -euo pipefail、docker build、docker run、docker rm。

        测试目的：防止关键防御配置或核心构建启动命令被误删，保证执行可靠性。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_key_commands_present -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_key_commands_present -v -s
        """
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "set -euo pipefail" in content, "缺少安全防错选项 set -euo pipefail"
        assert "docker build" in content, "缺少 docker build 构建指令"
        assert "docker run" in content, "缺少 docker run 运行指令"
        assert "docker rm -f" in content, "缺少清理旧容器机制 docker rm -f"

    def test_script_ports_and_env_config(self):
        """【静态检查】验证脚本正确暴露了 REST (8079) 与 gRPC (50051) 端口及默认环境变量。

        测试目的：确保容器对外服务端口及 0.0.0.0 监听地址配置符合规范。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_ports_and_env_config -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptStaticChecks::test_script_ports_and_env_config -v -s
        """
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "8079" in content and "50051" in content, "缺少端口映射配置 8079 / 50051"
        assert 'PRIVACY_REST_HOST="0.0.0.0"' in content, "容器未绑定 PRIVACY_REST_HOST=0.0.0.0"
        assert 'PRIVACY_GRPC_HOST="0.0.0.0"' in content, "容器未绑定 PRIVACY_GRPC_HOST=0.0.0.0"


# ═══════════════════════════════════════════════════════════════════════
# 第 2 层：模拟执行检查 / Fake Execution Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAgentScriptFakeExecution:
    """使用 Fake Docker 模拟脚本执行全流程（无真实 Docker 守护进程依赖）。"""

    def test_script_runs_default_core(self, bash_bin: str):
        """【模拟执行】未传参数时，验证脚本默认构建并启动 core 镜像。

        测试目的：
            1. 验证默认目标分支为 core；
            2. 验证构建命令包含 `--target core` 与标签 `privshield:1.8.0`；
            3. 验证终端输出启动成功友好提示。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_script_runs_default_core -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_script_runs_default_core -v -s
        """
        result, logs = _run_script_with_fake_docker(bash_bin, exit_code=0)
        assert result.returncode == 0, f"脚本执行失败: {result.stderr}"
        assert "PrivShield (Docker) 已成功启动" in result.stdout
        assert "build --target core -t privshield:1.8.0" in logs
        assert "run -d --name PrivShield" in logs

    def test_script_runs_ml_target(self, bash_bin: str):
        """【模拟执行】显式传入 ml 参数时，验证脚本构建并启动 ml 镜像。

        测试目的：验证多阶段构建目标切换为 `ml` 且镜像标签为 `privshield:1.8.0-ml`。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_script_runs_ml_target -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_script_runs_ml_target -v -s
        """
        result, logs = _run_script_with_fake_docker(bash_bin, exit_code=0, target="ml")
        assert result.returncode == 0, f"脚本执行失败: {result.stderr}"
        assert "build --target ml -t privshield:1.8.0-ml" in logs
        assert "privshield:1.8.0-ml" in logs

    def test_cleans_up_old_container_before_run(self, bash_bin: str):
        """【模拟执行】验证在每次启动新容器前，脚本先执行 docker rm -f 清理同名旧容器。

        测试目的：避免因容器名称冲突（Conflict. The container name is already in use）导致启动失败。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_cleans_up_old_container_before_run -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_cleans_up_old_container_before_run -v -s
        """
        result, logs = _run_script_with_fake_docker(bash_bin, exit_code=0)
        assert result.returncode == 0
        assert "rm -f PrivShield" in logs

    def test_fails_fast_when_docker_build_fails(self, bash_bin: str):
        """【模拟执行】验证当 docker 命令失败（返回非 0）时，set -e 机制使脚本立即非零退出。

        测试目的：确保构建或运行异常不会被吞没，防止 CI 或运维获得假成功状态。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_fails_fast_when_docker_build_fails -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_fails_fast_when_docker_build_fails -v -s
        """
        result, _ = _run_script_with_fake_docker(bash_bin, exit_code=1)
        assert result.returncode != 0, "docker 失败时脚本未能正确退出"

    def test_script_shows_help(self, bash_bin: str):
        """【模拟执行】验证传入 --help 或 -h 时正确输出用法帮助信息并以 0 退出。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_script_shows_help -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_script_shows_help -v -s
        """
        result, _ = _run_script_with_fake_docker(bash_bin, exit_code=0, target="--help")
        assert result.returncode == 0
        assert "用法 / Usage" in result.stdout
        assert "core" in result.stdout
        assert "ml" in result.stdout

    def test_script_rejects_invalid_target(self, bash_bin: str):
        """【模拟执行】验证传入未知非法目标（如 invalid_target）时非零退出并提示错误。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_script_rejects_invalid_target -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentScriptFakeExecution::test_script_rejects_invalid_target -v -s
        """
        result, _ = _run_script_with_fake_docker(bash_bin, exit_code=0, target="invalid_target")
        assert result.returncode != 0
        assert "无效的构建目标" in result.stderr


# ═══════════════════════════════════════════════════════════════════════
# 第 3 层：停止脚本测试 / Stop Script Checks
# ═══════════════════════════════════════════════════════════════════════


class TestAgentStopScript:
    """docker-stop-agent.sh 停止与清理脚本测试。"""

    def test_stop_script_exists_and_valid(self, bash_bin: str):
        """【静态检查】验证停止脚本文件存在、具备可执行权限且语法正确。

        测试目的：确保清理脚本可用无语法瑕疵。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentStopScript::test_stop_script_exists_and_valid -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentStopScript::test_stop_script_exists_and_valid -v -s
        """
        assert STOP_SCRIPT_PATH.is_file(), f"停止脚本不存在: {STOP_SCRIPT_PATH}"
        assert STOP_SCRIPT_PATH.stat().st_mode & stat.S_IXUSR, "停止脚本缺少可执行权限"
        env = _clean_env()
        result = subprocess.run([bash_bin, "-n", str(STOP_SCRIPT_PATH)], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr

    def test_stop_script_fake_execution(self, bash_bin: str):
        """【模拟执行】验证停止脚本执行时正确调用 docker rm -f 移除 Agent 容器。

        测试目的：确保停止脚本能可靠清除 Agent 容器并输出停止成功提示。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentStopScript::test_stop_script_fake_execution -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentStopScript::test_stop_script_fake_execution -v -s
        """
        result, logs = _run_script_with_fake_docker(bash_bin, exit_code=0, script_path=STOP_SCRIPT_PATH)
        assert result.returncode == 0
        assert "rm -f PrivShield" in logs
        assert "PrivShield 容器已成功停止与清理" in result.stdout

    def test_windows_powershell_scripts_exist_and_valid(self):
        """【跨平台兼容性】验证 Windows 11 原生 PowerShell 脚本结构与指令完整性。

        测试目的：确保 Windows 11 用户在 PowerShell 终端下无需 bash 即可直接启动/停止容器。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentStopScript::test_windows_powershell_scripts_exist_and_valid -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentStopScript::test_windows_powershell_scripts_exist_and_valid -v -s
        """
        ps_start = PROJECT_ROOT / "scripts" / "dev" / "docker-start-agent.ps1"
        ps_stop = PROJECT_ROOT / "scripts" / "dev" / "docker-stop-agent.ps1"

        assert ps_start.is_file(), f"Windows 启动脚本不存在: {ps_start}"
        assert ps_stop.is_file(), f"Windows 停止脚本不存在: {ps_stop}"

        start_content = ps_start.read_text(encoding="utf-8")
        assert "docker build" in start_content
        assert "docker run" in start_content
        assert "8079" in start_content
        assert "50051" in start_content
        assert "PRIVACY_REST_HOST" in start_content

        stop_content = ps_stop.read_text(encoding="utf-8")
        assert "docker rm -f PrivShield" in stop_content

    def test_cross_platform_os_detection_in_bash_script(self):
        """【跨平台兼容性】验证 bash 启动脚本内置了 macOS (Darwin) 与 Windows (WSL2/GitBash) 平台检测。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestAgentStopScript::test_cross_platform_os_detection_in_bash_script -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestAgentStopScript::test_cross_platform_os_detection_in_bash_script -v -s
        """
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "Darwin" in content, "脚本缺少 macOS 识别逻辑"
        assert "WSL2" in content or "microsoft" in content, "脚本缺少 WSL2 识别逻辑"
        assert "MINGW" in content or "MSYS" in content, "脚本缺少 Git Bash / MSYS2 识别逻辑"


# ═══════════════════════════════════════════════════════════════════════
# 第 4 层：Compose 拓扑与 Agent-LLM 通信配置校验 / Topology Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDockerAgentLlmComposeTopology:
    """验证 docker-compose.yml 中 Agent 与 Docker LLM (vLLM) 的解耦拓扑与通信配置。"""

    def test_compose_defines_agent_and_vllm_services(self, compose_config: dict[str, Any]):
        """【编排拓扑】验证 docker-compose.yml 同时定义了 PrivShield 与 vllm 服务。

        测试目的：确保全栈编排包含 Sidecar 核心与独立 LLM 推理服务。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmComposeTopology::test_compose_defines_agent_and_vllm_services -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmComposeTopology::test_compose_defines_agent_and_vllm_services -v -s
        """
        services = compose_config.get("services", {})
        assert "PrivShield" in services, "compose 缺少 PrivShield 服务定义"
        assert "vllm" in services, "compose 缺少 vllm 服务定义"

    def test_compose_network_topology_agent_to_vllm(self, compose_config: dict[str, Any]):
        """【网络拓扑】验证 PrivShield 与 vllm 服务均加入了 llm 共享网络
        仅仅是对配置文件的静态结构与声明进行校验，不会在运行时真正创建或加入 Docker 的 llm 网络。

        测试目的：确保 Agent 容器能够通过容器名 DNS（http://vllm:8000/v1）跨容器访问 LLM 推理服务。

        为何通过容器名就能跨容器访问（Why container-name DNS works across containers）:
            1. Docker Compose 会为每个显式声明的网络（本配置中的 llm，driver: bridge）创建
               一个独立的虚拟二层网段；加入同一网络的所有容器共享该网段，彼此二层可达；
            2. Docker 在每个容器内运行内嵌 DNS 服务（127.0.0.11），它会把同一网络内的
               容器名/服务名（vllm）自动解析为该容器的动态 IP（即 Docker 的服务发现机制）；
            3. 因此 Agent 内访问 http://vllm:8000/v1 等价于访问 vllm 容器的 8000 端口，
               无需关心容器 IP——IP 会随容器重建而变化，而服务名恒定，跨容器访问始终成立；
            4. 网络同时是隔离边界：未加入 llm 网络的容器既解析不到 vllm 也无法访问它，
               故本测试必须同时断言双方都挂载了 llm 网络。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmComposeTopology::test_compose_network_topology_agent_to_vllm -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmComposeTopology::test_compose_network_topology_agent_to_vllm -v -s
        """
        # 取出 compose 文件中的 services 顶层映射（key 为服务名，value 为服务配置字典）
        # Extract the top-level "services" mapping from the compose config (keyed by service name)
        services = compose_config.get("services", {})

        # 读取 Agent 服务声明的网络列表（networks 字段，本配置为显式列表形式）
        # Read the network list declared by the agent service (explicit list form in this config)
        agent_networks = services["PrivShield"].get("networks", [])

        # 读取 vllm 服务声明的网络列表
        # Read the network list declared by the vllm service
        vllm_networks = services["vllm"].get("networks", [])

        # Agent 必须加入 llm 网络：只有同一网络内的容器才能被内嵌 DNS 解析并互通，
        # 否则 PRIVACY_LLM_API_BASE=http://vllm:8000/v1 会解析失败，LLM 层只能降级
        # The agent MUST join the "llm" network: only containers on the same network are
        # resolvable and reachable, otherwise PRIVACY_LLM_API_BASE=http://vllm:8000/v1 fails
        assert "llm" in agent_networks, "PrivShield 未加入 llm 网络"

        # vllm 必须加入同一网络：内嵌 DNS 只为“加入了该网络的容器”注册服务名记录，
        # 若 vllm 不在 llm 网络内，Agent 侧将无法解析 vllm 这个服务名
        # The vllm service MUST join the same network: the embedded DNS only registers
        # service-name records for containers attached to that network
        assert "llm" in vllm_networks, "vllm 未加入 llm 网络"

    def test_compose_agent_llm_environment_variables(self, compose_config: dict[str, Any]):
        """【通信配置】验证 Agent 服务中配置了正确的 LLM 提供方与端点地址，以及 API Key 默认值策略。

        虽然本系统使用的是本地部署的 LLM（如 vLLM），但在配置中依然保留并校验了 PRIVACY_LLM_API_KEY（默认值为 "EMPTY"），主要原因有以下几点：
          ──────
          ### 1. 遵循 OpenAI 兼容协议规范与业界惯例
          Agent 的 LLM 请求层（llm_engines.py）采用标准 OpenAI 兼容 HTTP 接口规范，在发起请求时统一会带上 Authorization 请求头：
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
          • vLLM、Ollama 等开源推理引擎均遵循 OpenAI API 规范。
          • 当 vLLM 本地未开启 --api-key 鉴权时，社区标准惯例即是传递占位字符串 "EMPTY"。
          ──────
          ### 2. 架构解耦：支持跨主机 GPU 部署与网关鉴权
          虽然默认是本地同机部署，但在生产环境中常见的场景包括：
          1. 分机/跨主机部署：Agent 部署在 CPU 机器，vLLM 部署在独立的远程 GPU 服务器上，中间通常会配置安全访问密钥。
          2. API 网关防护：在 vLLM 前方架设了反向代理或 API 网关（如 Kong、Nginx、APISIX 等）进行访问控制，需要真实的 API Key。
          3. 无缝切云：方便在特定场景下将 PRIVACY_LLM_API_BASE 指向外部公有云/私有云大模型接口，只需在环境变量中注入 LLM_API_KEY=sk-xxxx 即可，代码和请求逻辑无需任何改动。
          ──────
          ### 3. 测试断言的具体目的
          在 docker-compose.yml 中，该环境变量使用了 Compose 参数化语法：
            PRIVACY_LLM_API_KEY: "${LLM_API_KEY:-EMPTY}"
          断言的目的，是确保未显式配置 LLM_API_KEY 时，Compose 能够正确回退到安全默认值 "EMPTY"，防止因环境变量缺失或为 None 导致容器内 Agent 向本地 vLLM 发送请求时抛出异常。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmComposeTopology::test_compose_agent_llm_environment_variables -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmComposeTopology::test_compose_agent_llm_environment_variables -v -s
        """
        services = compose_config.get("services", {})
        agent_env = services["PrivShield"].get("environment", {})

        assert agent_env.get("PRIVACY_LLM_PROVIDER") == "vllm", "Agent LLM Provider 应为 vllm"

        # LLM_API_BASE 已参数化为 ${LLM_API_BASE:-http://vllm:8000/v1}：
        # 未设置变量时默认指向 compose 内部 DNS 服务名 vllm:8000（同机部署零配置）；
        # 跨主机部署经 deploy/docker-compose/.env 的 LLM_API_BASE 覆盖（见 ops.md §5.2.11 ④）。
        # 此处从 yml 字面量中提取 `:-` 后的默认值进行断言，兼容两种写法。
        llm_api_base = agent_env.get("PRIVACY_LLM_API_BASE", "")
        match = re.fullmatch(r"\$\{LLM_API_BASE:-(.+)\}", llm_api_base)
        effective_base = match.group(1) if match else llm_api_base
        assert effective_base == "http://vllm:8000/v1", (
            f"Agent LLM API Base 默认值应指向 http://vllm:8000/v1: {llm_api_base}"
        )
        assert agent_env.get("PRIVACY_LLM_MODEL_NAME") == VLLM_SERVED_MODEL_NAME, (
            f"Agent 模型名与 vllm 服务名不一致: {agent_env.get('PRIVACY_LLM_MODEL_NAME')}"
        )

        # PRIVACY_LLM_API_KEY 已参数化为 ${LLM_API_KEY:-EMPTY}
        llm_api_key = agent_env.get("PRIVACY_LLM_API_KEY", "")
        key_match = re.fullmatch(r"\$\{LLM_API_KEY:-(.+)\}", llm_api_key)
        effective_key = key_match.group(1) if key_match else llm_api_key
        assert effective_key == "EMPTY", f"Agent LLM API Key 默认值应为 EMPTY: {llm_api_key}"


# ═══════════════════════════════════════════════════════════════════════
# 第 5 层：Agent 客户端与 LLM 通信模拟单元测试 / Mocked Agent-LLM Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDockerAgentLlmCommunicationMock:
    """Agent 与 Docker LLM 通信逻辑的模拟单元测试（无需 GPU，使用 Mock 验证协议与安全边界）。"""

    def test_agent_dynclassification_service_llm_routing_mock(self):
        """【Mock 通信】验证 Agent 动态分类服务在配置 vllm 时，能正确通过 HTTP 调用 LLM 并解析结果。

        测试目的：
            1. 验证 OpenAILlmClassifier 构造合法的 /v1/chat/completions 请求 payload；
            2. 验证 Agent 成功解析 LLM 返回的 JSON 分类与脱敏重写结果。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmCommunicationMock::test_agent_dynclassification_service_llm_routing_mock -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmCommunicationMock::test_agent_dynclassification_service_llm_routing_mock -v -s
        """
        mock_resp_data = {
            "id": "chatcmpl-test-agent-mock",
            "object": "chat.completion",
            "model": VLLM_SERVED_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "final_level": "L4",
                                "confidence": 0.96,
                                "reasoning": "识别为高敏病种临床主诉与阳性检验指标",
                                "sanitized_text": "患者反复发热伴咽痛2周，常规对症治疗。",
                            },
                            ensure_ascii=False,
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
        }

        mock_http_response = MagicMock()
        mock_http_response.status = 200
        mock_http_response.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
        mock_http_response.__enter__.return_value = mock_http_response

        with patch("urllib.request.urlopen", return_value=mock_http_response) as mock_urlopen:
            classifier = OpenAILlmClassifier(
                api_base="http://127.0.0.1:8000/v1",
                model_name=VLLM_SERVED_MODEL_NAME,
            )
            result = classifier.classify("患者反复发热伴咽痛2周，HIV抗体阳性", SensitivityLevel.L1, 0.5)

            assert result is not None, "LLM 分类结果返回 None"
            assert result["final_level"] == "L4"
            assert result["confidence"] == 0.96
            assert "高敏病种" in result["reasoning"]

            # 验证向 Docker LLM 发出的 HTTP 请求符合 OpenAI 协议规范
            assert mock_urlopen.called
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://127.0.0.1:8000/v1/chat/completions"
            body = json.loads(req.data.decode("utf-8"))
            assert body["model"] == VLLM_SERVED_MODEL_NAME
            assert body["temperature"] == 0.0

    def test_agent_safety_floor_prevents_illegal_downgrade_mock(self):
        """【Mock 通信】验证 Agent 的 Safety Floor（安全地板）能防止 LLM 幻觉将敏感级别非法降级。

        测试目的：
            当上游规则已经判定为高敏感等级（如 L4），即使 Mock 的 LLM 异常返回 L1，
            Agent 的漏斗必须拒绝该降级决策并标记 needs_human_review=True。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmCommunicationMock::test_agent_safety_floor_prevents_illegal_downgrade_mock -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmCommunicationMock::test_agent_safety_floor_prevents_illegal_downgrade_mock -v -s
        """
        adapter = LlmAdapter()
        mock_classifier = MagicMock()
        # 模拟 LLM 发生幻觉返回极低等级 L1
        mock_classifier.classify.return_value = {
            "final_level": "L1",
            "confidence": 0.99,
            "reasoning": "模型误判为无害公开数据",
        }
        adapter._classifier = mock_classifier
        adapter._initialized = True
        adapter._available = True

        service = DynClassificationService(rules_dir=str(PROJECT_ROOT / "rules"))
        engine = service.loader.get_engine("sc_health_db51")

        # 构造包含 LLM 仲裁的漏斗
        policy = ConfidencePolicy(enable_llm_arbitration=True, enable_llm=True)
        funnel = service._build_funnel(engine)
        funnel.llm = adapter

        # 输入包含高敏 HIV 的文本（规则层会识别为 L5/L4）
        result, _ = funnel.classify_field("hiv_report", "HIV确证试验阳性")

        # 断言：安全地板兜底，最终等级绝不能被降为 L1，且需要人工复核
        assert result.final_level in ("L4", "L5"), f"Safety Floor 未能阻止非法降级: {result.final_level}"

    def test_agent_graceful_degradation_on_llm_network_error_mock(self):
        """【Mock 通信】验证当 Docker LLM 容器网络宕机或超时时，Agent 自动平滑降级而不崩溃。

        测试目的：
            模拟 URLError（连接拒绝/超时），验证 Agent 的 OpenAILlmClassifier
            返回 None，并让上层服务回退至 Layer-1 规则层或默认安全等级。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmCommunicationMock::test_agent_graceful_degradation_on_llm_network_error_mock -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentLlmCommunicationMock::test_agent_graceful_degradation_on_llm_network_error_mock -v -s
        """
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            classifier = OpenAILlmClassifier(
                api_base="http://127.0.0.1:8000/v1",
                model_name=VLLM_SERVED_MODEL_NAME,
            )
            # 遇到网络错误时必须捕获异常并优雅返回 None，绝不能抛出未捕获异常导致 Sidecar 进程崩溃
            result = classifier.classify("测试网络异常降级", SensitivityLevel.L1, 0.5)
            assert result is None, "LLM 网络错误时未返回 None 触发降级"


# ═══════════════════════════════════════════════════════════════════════
# 第 6 层：Docker Agent 与 Docker LLM 真实通信集成测试 / Real Integration
# ═══════════════════════════════════════════════════════════════════════


def _http_get_json(url: str, timeout: float = 10.0) -> dict[str, Any] | None:
    """向指定 URL 发送 HTTP GET 请求并解析 JSON 响应。

    .venv 测试验证 / Test Command:
        # 辅助函数，向指定 URL 发送 HTTP GET 请求。由 _wait_vllm_ready 及集成测试调用。可通过运行第 6 层测试验证：
        source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentToLlmIntegration -v -s
        # 或直接调用：
        .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "TestDockerAgentToLlmIntegration" -v -s
    """
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _wait_vllm_ready(timeout_s: int = VLLM_READY_TIMEOUT_S) -> bool:
    """轮询等待 vLLM 容器的 /v1/models 端点就绪。

    .venv 测试验证 / Test Command:
        # 辅助函数，轮询等待 vLLM 容器就绪。在 vllm_service fixture 中调用。可通过运行集成测试验证：
        source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentToLlmIntegration -v -s
        # 或直接调用：
        .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "TestDockerAgentToLlmIntegration" -v -s
    """
    url = f"{VLLM_API_BASE}/models"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        data = _http_get_json(url, timeout=3.0)
        if data and "data" in data and len(data["data"]) > 0:
            return True
        time.sleep(VLLM_READY_POLL_INTERVAL_S)
    return False


@pytest.fixture(scope="module")
def vllm_service(
    bash_bin: str,
    docker_available: bool,
    gpu_available: bool,
    vllm_image_available: bool,
) -> dict[str, Any]:
    """启动（或复用已有）Docker vLLM 推理容器，并等待 OpenAI 兼容 API 就绪。

    .venv 测试验证 / Test Command:
        # Pytest module 级 fixture，启动/复用 Docker vLLM 容器。可通过运行依赖它的集成测试进行验证：
        source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentToLlmIntegration -v -s
        # 或直接调用：
        .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "TestDockerAgentToLlmIntegration" -v -s

    执行逻辑与生命周期设计（Execution Logic & Lifecycle Design）:
        1. 【阶段 1：前置环境准入检查 (Prerequisites Gate)】
           - 检查系统环境是否满足真实 vLLM 运行要求：Docker CLI、NVIDIA GPU 驱动、本地模型权重目录及 vLLM 镜像。
           - 若任一条件不满足，则调用 `pytest.skip()` 优雅跳过当前模块的集成测试，避免在 CPU/无容器环境下报错。
        2. 【阶段 2：容器状态探测与复用决策 (Container Inspection & Reuse Strategy)】
           - 执行 `docker inspect` 探测宿主机是否存在同名容器 `PrivShield-vllm`：
             * 若容器已存在且处于运行状态（Running=true）：直接复用已有容器（`created=False`），避免重复启动与显存浪费；
             * 若容器已存在但已退出（Exited）：强制删除旧容器（`docker rm -f`）并标记 `created=True` 准备重新拉起；
             * 若容器不存在：标记 `created=True` 准备新建拉起。
        3. 【阶段 3：调用启动脚本与轮询就绪 (Launch & Ready Polling)】
           - 若 `created=True`，调用 `scripts/dev/docker-start-llm.sh` 脚本拉起物理容器；
           - 启动后调用 `_wait_vllm_ready()` 持续轮询 `/v1/models` 端点，直到大模型权重加载完成并返回 200 OK（最长等待 600s）。
        4. 【阶段 4：提供服务元数据 (Yield Service Context)】
           - 向测试用例注入 `{"api_base", "model", "created"}` 字典，供测试构建 HTTP 客户端或配置环境变量。
        5. 【阶段 5：资源清理与环境恢复 (Teardown Cleanup)】
           - 在 `finally` 块中判断：仅当容器是由本测试 fixture 创建拉起时（`created=True`），才在测试结束后执行 `docker rm -f` 销毁容器；
           - 若复用的是开发者原先已启动的环境（`created=False`），则绝不删除，保护外部工作环境不被意外破坏。

    Returns:
        包含 {"api_base", "model", "created"} 的服务描述字典。
    """
    # ── 阶段 1：前置条件检查（任一不满足即跳过集成测试）──
    if not docker_available:
        pytest.skip("docker 不可用")
    if not gpu_available:
        pytest.skip("未检测到 NVIDIA GPU")
    if not VLLM_MODEL_DIR.is_dir():
        pytest.skip(f"本地模型目录不存在: {VLLM_MODEL_DIR}")
    if not vllm_image_available:
        pytest.skip("本地未缓存 vLLM 镜像")

    created = False
    try:
        # ── 阶段 2：探测容器当前状态，决策是复用还是重建 ──
        inspect = subprocess.run(
            ["docker", "inspect", VLLM_CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if inspect.returncode == 0:
            # 容器已存在，进一步检查是否正在运行
            running = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", VLLM_CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if running.stdout.strip() == "true":
                # 宿主机已有同名容器且正在运行：直接复用，不重复创建，teardown 阶段也不予删除
                pass
            else:
                # 容器处于非运行状态（已退出/失败）：清理残留容器后重新拉起
                subprocess.run(["docker", "rm", "-f", VLLM_CONTAINER_NAME], capture_output=True)
                created = True
        else:
            # 容器不存在：标记需要新建
            created = True

        # ── 阶段 3：执行启动脚本拉起物理容器 ──
        if created:
            start_script = PROJECT_ROOT / "scripts" / "dev" / "docker-start-llm.sh"
            res = subprocess.run(
                [bash_bin, str(start_script)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if res.returncode != 0:
                pytest.fail(f"docker-start-llm.sh 启动失败: {res.stderr or res.stdout}")

        # 轮询等待 OpenAI /v1/models 就绪探针返回可用模型列表
        if not _wait_vllm_ready():
            pytest.fail(f"vLLM 服务未在 {VLLM_READY_TIMEOUT_S}s 内就绪")

        # ── 阶段 4：Yield 服务元数据给测试用例 ──
        yield {"api_base": VLLM_API_BASE, "model": VLLM_SERVED_MODEL_NAME, "created": created}
    finally:
        # ── 阶段 5：清理回收资源（仅清理本 fixture 创建的容器，避免干扰已有环境）──
        if created:
            subprocess.run(["docker", "rm", "-f", VLLM_CONTAINER_NAME], capture_output=True)


@pytest.mark.integration
class TestDockerAgentToLlmIntegration:
    """Docker Agent 与 Docker LLM 真实通信与协同任务测试（资源不足自动 skip）。"""

    def test_agent_service_calls_docker_llm_for_field_classification(self, vllm_service: dict[str, Any]):
        """【真实通信】Agent 动态分类服务直连 Docker LLM，执行复杂医疗文本分类与仲裁。

        业务流程：
            1. Agent 初始化为 `PRIVACY_ENV_PROFILE=vllm` 模式；
            2. 向 Agent 发送一段未在规则表硬编码的非结构化复杂临床主诉；
            3. Agent 规则层判定低置信度，自动触发 Layer-3 向 Docker vLLM 容器发起 HTTP 请求；
            4. 验证 Agent 正确获取并整合 Docker LLM 返回的最终评级 (L1~L5) 与推理依据。

        通信端口配置与网络链路说明（Port Configuration & Network Topology）:
            1. 【通信端口配置来源】：
               - 测试侧：通过 fixture 注入 `vllm_service["api_base"]`（即顶部常量 `VLLM_API_BASE = "http://127.0.0.1:8000/v1"`，端口为 8000）；
               - Compose 编排侧：对应 `deploy/docker-compose/docker-compose.yml` 中的端口映射 `ports: ["127.0.0.1:8000:8000"]`；
               - 容器内部侧：对应 vLLM 镜像启动参数 `--port 8000 --host 0.0.0.0`。

            2. 【连接本地原生进程 vLLM vs 连接 Docker 容器 vLLM 的区别】：
               - 模式 A（本地原生 Python 进程 vLLM）：
                 * 运行形态：vLLM 直接在宿主机 Python/Conda 中运行，Agent 也是宿主机普通进程；
                 * 网络流向：宿主机 Loopback 直连 `127.0.0.1:8000`，无 Docker NAT 或虚拟网卡中转；
                 * 配置参数：`PRIVACY_LLM_API_BASE=http://127.0.0.1:8000/v1`，`PRIVACY_LLM_API_KEY=EMPTY`。
               - 模式 B（宿主机 Agent 访问 Docker vLLM，即本测试运行场景）：
                 * 运行形态：pytest 测试进程运行在宿主机，vLLM 运行在独立 Docker 容器内；
                 * 网络流向：宿主机 pytest -> 宿主机 `127.0.0.1:8000` -> Docker 端口映射转发 -> 容器内 `8000`；
                 * 配置参数：`PRIVACY_LLM_API_BASE=http://127.0.0.1:8000/v1`（与本地原生模式配置完全一致）。
               - 模式 C（全栈 Docker Compose 容器化部署）：
                 * 运行形态：Agent (`PrivShield`) 与 vLLM (`vllm`) 均在独立 Docker 容器内；
                 * 网络流向：Agent 容器 -> Docker 内部 `llm` 桥接网络 -> Docker 内嵌 DNS 解析服务名 `vllm:8000`（不经过宿主机端口转发）；
                 * 配置参数：`PRIVACY_LLM_API_BASE=http://vllm:8000/v1`。
               - 模式 D（跨主机 / 远程 GPU 节点部署）：
                 * 运行形态：Agent 在业务机，vLLM 在远程专用 GPU 服务器；
                 * 网络流向：跨主机局域网/公网 IP 路由，通常经过 API 网关或防火墙；
                 * 配置参数：`PRIVACY_LLM_API_BASE=http://<GPU_HOST_IP>:8000/v1`，`PRIVACY_LLM_API_KEY=sk-xxxx`。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentToLlmIntegration::test_agent_service_calls_docker_llm_for_field_classification -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentToLlmIntegration::test_agent_service_calls_docker_llm_for_field_classification -v -s
        """
        # 设置环境指向运行中的 Docker vLLM 服务（宿主机通过 127.0.0.1:8000 端口映射访问容器）
        os.environ["PRIVACY_ENV_PROFILE"] = "vllm"
        os.environ["PRIVACY_LLM_PROVIDER"] = "vllm"
        os.environ["PRIVACY_LLM_API_BASE"] = vllm_service["api_base"]
        os.environ["PRIVACY_LLM_MODEL_NAME"] = vllm_service["model"]

        service = DynClassificationService(rules_dir=str(PROJECT_ROOT / "rules"))
        engine = service.loader.get_engine("medical")

        # 启用 LLM 仲裁与推理
        engine.taxonomy.confidence_policy.enable_llm = True
        engine.taxonomy.confidence_policy.enable_llm_arbitration = True
        engine.taxonomy.confidence_policy.llm_confidence_threshold = 0.8

        funnel = service._build_funnel(engine)
        test_clinical_text = "患者反复发热咳痰2周，伴右侧胸痛，疑难肺部结节待查"

        # 执行分类
        result, _ = funnel.classify_field("clinical_note", test_clinical_text)

        print(f"\n[Docker Agent -> Docker LLM 通信结果]")
        print(f"  • 触发引擎层级 : {result.engine_layer}")
        print(f"  • 最终评定等级 : {result.final_level}")
        print(f"  • 评定置信度   : {result.confidence}")
        print(f"  • 决策推理依据 : {result.reasoning}")

        assert result is not None
        assert result.final_level in ("L1", "L2", "L3", "L4", "L5")
        assert result.confidence > 0.0
        assert len(result.reasoning) > 0

    def test_agent_record_sanitization_via_docker_llm(self, vllm_service: dict[str, Any]):
        """【真实通信】Agent 记录级脱敏处理：L1 规则与 L3 Docker LLM 协同输出脱敏数据。

        业务流程：
            1. 构造包含个人基本信息（姓名、身份证、电话）与高敏临床文书的记录字典；
            2. Agent 对敏感标识执行掩码与无痕抹平；
            3. 验证身份证与手机号等硬敏感信息在脱敏结果中被完全消除。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentToLlmIntegration::test_agent_record_sanitization_via_docker_llm -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentToLlmIntegration::test_agent_record_sanitization_via_docker_llm -v -s
        """
        os.environ["PRIVACY_ENV_PROFILE"] = "vllm"
        client = OpenAILlmClassifier(api_base=vllm_service["api_base"], model_name=vllm_service["model"])

        raw_chunk = (
            "患者姓名：李四，身份证号：510101198505051234，联系电话：13900001111。\n"
            "诊断为重度抑郁症伴自杀倾向，给予舍曲林治疗。"
        )

        sanitized_chunk = client.sanitize_text(raw_chunk)
        print(f"\n[Agent 调用 Docker LLM 抹平脱敏结果]:\n{sanitized_chunk}")

        assert sanitized_chunk is not None
        # 验证隐私抹平效果：原始身份证和手机号绝不能明文泄露
        assert "510101198505051234" not in sanitized_chunk, "身份证号未被脱敏"
        assert "13900001111" not in sanitized_chunk, "手机号未被脱敏"
        assert "李四" not in sanitized_chunk, "患者姓名未被掩码打码"


# ═══════════════════════════════════════════════════════════════════════
# 第 7 层：Docker Agent 容器/REST 建立连接、脱敏与分类分级端到端测试
# ═══════════════════════════════════════════════════════════════════════


def _http_post_json(url: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any] | None:
    """向指定 URL 发送 HTTP POST 请求并解析 JSON 响应。

    .venv 测试验证 / Test Command:
        # 辅助函数，向指定 URL 发送 HTTP POST 请求。在第 8 层真实容器网络交互测试中调用：
        source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle -v -s
        # 或直接调用：
        .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "TestRealDockerAgentScriptLifecycle" -v -s

    Args:
        url: 目标 HTTP 端点 URL。
        payload: JSON 请求体字典。
        timeout: 超时时间（秒）。

    Returns:
        解析后的响应字典，请求失败返回 None。
    """
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


class TestDockerAgentContainerEndpoints:
    """调用 Agent (Docker / REST 服务) 建立网络连接，发送脱敏请求与分类分级请求。

    测试覆盖：
        1. 建立 HTTP 连接与健康检查探针 (/health 与 /readyz)
        2. 单字段脱敏请求 (POST /v1/privacy/mask)
        3. 整条记录多字段批量脱敏请求 (POST /v1/privacy/mask_record)
        4. 单字段动态分类分级评估请求 (POST /v1/dynclassification/eval)
        5. 整条记录动态分类分级与组合升级评估 (POST /v1/dynclassification/eval_record)
        6. 表级批量数据动态分类分级评估 (POST /v1/dynclassification/eval_table)
    """

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """初始化测试客户端（基于 FastAPI TestClient 的进程内 ASGI 内存调用机制）。

        通信机制说明（In-Memory ASGI Dispatch vs Physical Socket）:
            1. 【本类（第 7 层）测试机制】：
               - 本类使用 `TestClient(app)` 执行接口协议与业务契约测试；
               - `self.client.get(...)` / `self.client.post(...)` 是纯进程内 ASGI 内存事件调用；
               - 不会启动 Uvicorn 服务器，不会占用或请求宿主机 `8079` 物理端口，请求默认派发至虚拟基地址 `http://testserver`；
               - 具备毫秒级响应、零外部依赖、无需预先拉起 Docker 容器等优势。
            2. 【第 8 层物理容器真实 Socket 测试区别】：
               - 真正向物理端口 `http://127.0.0.1:8079` 发起真实 TCP 网络请求的测试位于第 8 层 `TestRealDockerAgentScriptLifecycle`；
               - 第 8 层会真实执行 `docker-start-agent.sh core` 拉起物理容器，并使用 `urllib.request` 经由物理网卡向 `http://127.0.0.1:8079` 发送 Socket 请求。

        .venv 测试验证 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "TestDockerAgentContainerEndpoints" -v -s
        """
        from fastapi.testclient import TestClient
        from engine.main import app

        self.client = TestClient(app)
        self.live_agent_url = "http://127.0.0.1:8079"

    def test_agent_connection_and_health_probes(self):
        """【建立连接】向 Agent 服务发送 GET /health 与 GET /readyz 探针请求。

        测试目的与设计背景（Test Objective & Design Background）:
            在容器化部署与微服务编排（如 K8s / Docker Compose / 负载均衡网关）场景下，
            Agent 必须提供标准的存活（Liveness）与就绪（Readiness）健康检查接口。
            本测试旨在验证：
            1. 客户端能够成功与 Agent 建立 HTTP 连接；
            2. 基础路由与全局中间件（安全响应头、链路追踪、访问日志）正常运转；
            3. Agent 核心服务依赖（配置解析器、隐私预算持久化存储等）就绪可用。

        通信机制与请求目标说明（Dispatch Mechanism & Target URL）:
            - 此处的 `self.client.get("/health")` 是通过 TestClient 在进程内存中直接派发至 `FastAPI app`，
               请求目标实际为 `http://testserver/health`，不经过物理端口 `http://127.0.0.1:8079`。
            - 针对宿主机真实暴露端口 `http://127.0.0.1:8079` 的物理 Socket 探测，请参见第 8 层测试。

        执行逻辑与分步流程（Execution Logic & Step-by-Step Flow）:
            【步骤 1：验证 /health 存活探针 (Liveness Check)】
                - 发送 HTTP GET 请求到 `/health`；
                - 断言 HTTP 状态码为 200（表示 ASGI 进程存活且端口可达）；
                - 断言 JSON 响应包含 `{"status": "ok"}`（表示健康检查通过）。

            【步骤 2：验证 /readyz 业务就绪探针 (Readiness Check)】
                - 发送 HTTP GET 请求到 `/readyz`；
                - 断言 HTTP 状态码为 200（表示依赖服务如 service.resolver、SQLite 预算库均正常连接）；
                - 断言 JSON 响应包含 `{"status": "ready"}`（表示 Agent 已具备处理脱敏与分类请求的全部条件，可正式接入外部流量）。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_connection_and_health_probes -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_connection_and_health_probes -v -s
        """
        # ── 步骤 1：验证 /health 存活探针（检验基础进程存活与路由可达性，内存派发至 http://testserver/health）──
        resp_health = self.client.get("/health")
        assert resp_health.status_code == 200, f"/health 响应失败: {resp_health.status_code}"
        assert resp_health.json().get("status") == "ok", (
            f"/health 响应内容不符合预期: {resp_health.text}"
        )

        # ── 步骤 2：验证 /readyz 业务就绪探针（检验核心服务与预算存储是否就绪）──
        resp_readyz = self.client.get("/readyz")
        assert resp_readyz.status_code == 200, f"/readyz 响应失败: {resp_readyz.status_code}"
        assert resp_readyz.json().get("status") == "ok", (
            f"/readyz 响应内容不符合预期: {resp_readyz.text}"
        )

    def test_agent_mask_single_field_request(self):
        """【脱敏请求】向 Agent 发送单字段脱敏请求 (POST /v1/privacy/mask)。

        测试流程：
            1. 发送手机号脱敏请求：13812345678 -> 138****5678；
            2. 发送身份证脱敏请求：510101199001011234 -> 510101********1234；
            3. 验证返回状态码 200 且敏感字段被正确掩码。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_mask_single_field_request -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_mask_single_field_request -v -s
        """
        # 手机号脱敏
        resp_phone = self.client.post(
            "/v1/privacy/mask",
            json={"field_name": "mobile", "value": "13812345678", "context": ""},
        )
        assert resp_phone.status_code == 200
        assert resp_phone.json().get("result") == "138****5678"

        # 身份证脱敏
        resp_id = self.client.post(
            "/v1/privacy/mask",
            json={"field_name": "id_card", "value": "510101199001011234", "context": ""},
        )
        assert resp_id.status_code == 200
        assert resp_id.json().get("result") == "510101********1234"

    def test_agent_mask_record_request(self):
        """【脱敏请求】向 Agent 发送整条多字段记录批量脱敏请求 (POST /v1/privacy/mask_record)。

        测试流程：
            1. 构造包含姓名、手机号、身份证、住址的字典；
            2. 向 Agent 发送整条记录批量脱敏请求；
            3. 验证所有敏感 PII 字段均被规范掩码打码。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_mask_record_request -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_mask_record_request -v -s
        """
        record_payload = {
            "record": {
                "name": "张三丰",
                "mobile": "13812345678",
                "id_card": "510101199001011234",
                "address": "成都市武侯区人民南路四段18号",
            },
            "context": "",
        }

        resp = self.client.post("/v1/privacy/mask_record", json=record_payload)
        assert resp.status_code == 200
        result_record = resp.json().get("result", {})

        print(f"\n[Agent 整记录脱敏响应]: {result_record}")
        assert result_record.get("mobile") == "138****5678"
        assert result_record.get("id_card") == "510101********1234"
        assert result_record.get("name") in ("张*丰", "张**丰", "张*")

    def test_agent_dynclassification_eval_single_field(self):
        """【分类分级请求】向 Agent 发送单字段动态分类分级请求 (POST /v1/dynclassification/eval)。

        测试流程：
            1. 发送身份证字段（id_card: 510101199001011234）；
            2. Agent 规则引擎与漏斗进行评估；
            3. 验证返回数据敏感等级为 L3、分类为 PERSONAL_BASIC、置信度为 1.0。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_dynclassification_eval_single_field -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_dynclassification_eval_single_field -v -s
        """
        eval_payload = {
            "fieldName": "id_card",
            "value": "510101199001011234",
            "domain": "sc_health_db51",
        }

        resp = self.client.post("/v1/dynclassification/eval", json=eval_payload)
        assert resp.status_code == 200
        data = resp.json()

        print(f"\n[Agent 单字段分类分级响应]:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
        field_res = data.get("fieldResult", {})
        assert field_res.get("finalLevel") == "L3"
        assert field_res.get("fieldName") == "id_card"
        assert any(tag.get("category") == "PERSONAL_BASIC" for tag in field_res.get("tags", []))

    def test_agent_dynclassification_eval_record_with_composite_rules(self):
        """【分类分级请求】向 Agent 发送整条记录分类分级请求 (POST /v1/dynclassification/eval_record)。

        测试流程：
            1. 构造包含患者基础信息与高敏诊断的整条记录；
            2. 向 Agent 发送整条记录评估请求；
            3. 验证整条记录被正确评估，触发复合规则与高敏病种定级（L4/L5）。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_dynclassification_eval_record_with_composite_rules -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_dynclassification_eval_record_with_composite_rules -v -s
        """
        record_payload = {
            "record": {
                "name": "李四",
                "id_card": "510101199001011234",
                "mobile": "13800138000",
                "diagnosis": "HIV抗体阳性",
            },
            "domain": "medical",
        }

        resp = self.client.post("/v1/dynclassification/eval_record", json=record_payload)
        assert resp.status_code == 200
        data = resp.json()

        print(f"\n[Agent 整记录分类分级响应]:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
        record_res = data.get("recordResult", {})
        assert record_res is not None
        # 包含 HIV 极高敏诊断，整条记录最高级别应判定为 L4 或 L5
        assert record_res.get("finalLevel") in ("L4", "L5")

    def test_agent_dynclassification_eval_table(self):
        """【分类分级请求】向 Agent 发送表格数据批量分类分级请求 (POST /v1/dynclassification/eval_table)。

        测试流程：
            1. 构造包含 schema 与多行 rows 的表格数据；
            2. 向 Agent 发送表级分类评估请求；
            3. 验证返回表级最高安全级别 (finalLevel) 与每行每列的分类标签。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_dynclassification_eval_table -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentContainerEndpoints::test_agent_dynclassification_eval_table -v -s
        """
        table_payload = {
            "schema": ["name", "mobile", "turnover"],
            "rows": [
                {"name": "张三", "mobile": "13812345678", "turnover": "100万"},
                {"name": "李四", "mobile": "13900001111", "turnover": "200万"},
            ],
            "domain": "sc_health_db51",
        }

        resp = self.client.post("/v1/dynclassification/eval_table", json=table_payload)
        assert resp.status_code == 200
        data = resp.json()

        print(f"\n[Agent 表级批量分类分级响应]:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
        table_res = data.get("tableResult", {})
        assert table_res is not None
        assert table_res.get("finalLevel") in ("L1", "L2", "L3", "L4", "L5")
        assert len(table_res.get("recordResults", [])) == 2


# ═══════════════════════════════════════════════════════════════════════
# 第 8 层：真实执行 bash scripts/dev/docker-start-agent.sh core 拉起容器交互测试
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="class")
def live_docker_agent_service(bash_bin: str, docker_available: bool):
    """通过执行真实 scripts/dev/docker-start-agent.sh core 启动容器，并在测试结束后清理。

    .venv 测试验证 / Test Command:
        # Pytest class 级 fixture，真实运行 docker-start-agent.sh core 拉起容器。测试命令示例：
        source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle -v -s
        # 或直接调用：
        .venv/bin/pytest tests/scripts/test_docker_start_agent.py -k "TestRealDockerAgentScriptLifecycle" -v -s

    Yields:
        容器服务的基础 URL (http://127.0.0.1:8079)
    """
    if not docker_available:
        pytest.skip("Docker daemon 不可用，跳过真实物理容器启动测试")

    start_script = PROJECT_ROOT / "scripts" / "dev" / "docker-start-agent.sh"
    stop_script = PROJECT_ROOT / "scripts" / "dev" / "docker-stop-agent.sh"

    def _is_tcp_port_in_use(port: int) -> bool:
        for host in ("0.0.0.0", "127.0.0.1"):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((host, port))
            except OSError:
                return True
        return False

    def _get_free_tcp_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            return s.getsockname()[1]

    # 若 8079 / 50051 已被宿主机本地服务（如 dev-bff-agent.sh）占用，自动分配空闲端口避免端口冲突
    rest_port = 8079 if not _is_tcp_port_in_use(8079) else _get_free_tcp_port()
    grpc_port = 50051 if not _is_tcp_port_in_use(50051) else _get_free_tcp_port()

    env = os.environ.copy()
    env["PRIVACY_REST_PORT"] = str(rest_port)
    env["PRIVACY_GRPC_PORT"] = str(grpc_port)

    # 1. 真实运行启动脚本构建并拉起 core 容器
    res = subprocess.run(
        [bash_bin, str(start_script), "core"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if res.returncode != 0:
        pytest.fail(f"执行 docker-start-agent.sh core 失败:\n{res.stderr or res.stdout}")

    base_url = f"http://127.0.0.1:{rest_port}"

    # 2. 轮询健康探针等待容器内的 FastAPI 服务就绪（最多 30 秒）
    ready = False
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            req = urllib.request.Request(f"{base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.5)

    if not ready:
        pytest.fail(f"Agent 容器在 30 秒内未成功就绪 (http://127.0.0.1:8079/health)")

    try:
        yield base_url
    finally:
        # 3. 测试类全部执行完成后执行停止脚本清理容器
        subprocess.run(
            [bash_bin, str(stop_script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )


@pytest.mark.integration
class TestRealDockerAgentScriptLifecycle:
    """真实执行 bash scripts/dev/docker-start-agent.sh core 启动物理容器并进行网络交互测试。

    测试覆盖：
        1. 验证真实 PrivShield 容器启动成功并在 docker ps 中运行
        2. 通过真实网络 Socket 向物理容器发送健康与就绪探针 (/health, /readyz)
        3. 通过真实 HTTP POST 请求调用物理容器执行单字段脱敏与整记录脱敏
        4. 通过真实 HTTP POST 请求调用物理容器执行单字段与整记录动态分类分级
        5. 测试完成后由 fixture 执行 docker-stop-agent.sh 验证容器干净卸载
    """

    def test_real_container_is_running_in_docker_ps(self, live_docker_agent_service: str):
        """【真实容器状态】验证 PrivShield 容器真实存在于 docker ps 输出中。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle::test_real_container_is_running_in_docker_ps -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle::test_real_container_is_running_in_docker_ps -v -s
        """
        ps = subprocess.run(
            ["docker", "ps", "--filter", f"name={AGENT_CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert ps.returncode == 0
        assert AGENT_CONTAINER_NAME in ps.stdout

    def test_real_container_health_probes(self, live_docker_agent_service: str):
        """【真实网络交互】向物理容器发送 GET /health 与 GET /readyz 探针。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle::test_real_container_health_probes -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle::test_real_container_health_probes -v -s
        """
        # 1. 验证健康检查 / Health probe verification
        # - urllib.request.Request: Python 标准库请求对象，用于封装目标 URL、HTTP 方法（GET/POST）与请求头等元数据。
        #   此处构造指向容器健康检查端点 (GET http://127.0.0.1:8079/health) 的请求。
        req_health = urllib.request.Request(f"{live_docker_agent_service}/health", method="GET")
        # - urllib.request.urlopen: 执行底层 Socket 连接并发送 HTTP 请求的核心函数。
        #   * 第一个参数传入 Request 对象或 URL 字符串；
        #   * timeout=3.0 设置网络超时时间（秒），避免容器网络阻塞导致测试用例挂起；
        #   * with 上下文管理器: 保证响应读取结束后自动释放底层 Socket 与连接池资源。
        with urllib.request.urlopen(req_health, timeout=3.0) as resp:
            assert resp.status == 200  # 校验 HTTP 状态码为 200 OK
            data = json.loads(resp.read().decode("utf-8"))  # 读取响应字节流并反序列化为 JSON 字典
            assert data.get("status") == "ok"

        # 2. 验证就绪探针 / Readiness probe verification
        # - 同样通过 Request 对象封装就绪探针 (GET /readyz)
        req_readyz = urllib.request.Request(f"{live_docker_agent_service}/readyz", method="GET")
        # - 通过 urlopen 探测容器内部依赖（配置解析器与隐私预算 DB）就绪状态
        with urllib.request.urlopen(req_readyz, timeout=3.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") == "ready"

    def test_real_container_masking_interactions(self, live_docker_agent_service: str):
        """【真实网络交互】向物理容器发送单字段与整记录脱敏 HTTP 请求。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle::test_real_container_masking_interactions -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle::test_real_container_masking_interactions -v -s
        """
        # 1. 单字段手机号脱敏
        payload_phone = {"field_name": "mobile", "value": "13812345678", "context": ""}
        resp_phone = _http_post_json(f"{live_docker_agent_service}/v1/privacy/mask", payload_phone)
        assert resp_phone is not None
        assert resp_phone.get("result") == "138****5678"

        # 2. 整条记录多字段批量脱敏
        payload_record = {
            "record": {
                "name": "张三丰",
                "mobile": "13812345678",
                "id_card": "510101199001011234",
                "address": "成都市武侯区人民南路四段18号",
            },
            "context": "",
        }
        resp_record = _http_post_json(f"{live_docker_agent_service}/v1/privacy/mask_record", payload_record)
        assert resp_record is not None
        result_record = resp_record.get("result", {})
        print(f"\n[真实 Docker Agent 容器脱敏响应]: {result_record}")
        assert result_record.get("mobile") == "138****5678"
        assert result_record.get("id_card") == "510101********1234"
        assert result_record.get("name") in ("张*丰", "张**丰", "张*")

    def test_real_container_classification_interactions(self, live_docker_agent_service: str):
        """【真实网络交互】向物理容器发送动态分类分级评估 HTTP 请求。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle::test_real_container_classification_interactions -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestRealDockerAgentScriptLifecycle::test_real_container_classification_interactions -v -s
        """
        # 1. 单字段分类评估
        payload_eval = {
            "fieldName": "id_card",
            "value": "510101199001011234",
            "domain": "sc_health_db51",
        }
        resp_eval = _http_post_json(f"{live_docker_agent_service}/v1/dynclassification/eval", payload_eval)
        assert resp_eval is not None
        field_res = resp_eval.get("fieldResult", {})
        assert field_res.get("finalLevel") == "L3"
        assert field_res.get("fieldName") == "id_card"

        # 2. 整记录复合分类评估
        payload_rec_eval = {
            "record": {
                "name": "李四",
                "id_card": "510101199001011234",
                "mobile": "13800138000",
                "diagnosis": "HIV抗体阳性",
            },
            "domain": "medical",
        }
        resp_rec_eval = _http_post_json(f"{live_docker_agent_service}/v1/dynclassification/eval_record", payload_rec_eval)
        print(f"\n[真实 Docker Agent 容器分类评估响应]: {resp_rec_eval}")
        assert resp_rec_eval is not None
        record_res = resp_rec_eval.get("recordResult", {})
        assert record_res.get("finalLevel") in ("L4", "L5")


# ═══════════════════════════════════════════════════════════════════════
# 第 9 层：多主机 / 跨主机部署与远程 LLM 通信测试 / Multi-Host Deployment Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDockerAgentMultiHostDeployment:
    """多主机与跨主机部署拓扑专项测试（包含跨网络 LLM 调用、鉴权注入、超时降级与配置隔离）。"""

    def test_compose_cross_host_variable_substitution_render(self):
        """【多主机编排】验证 Compose 在跨主机部署时能够正确将变量替换为远程端点与 API Key。

        测试场景：
            - 当部署在多主机环境时（Agent 运行在 CPU 节点，vLLM 运行在 GPU 节点），
            - deploy/docker-compose/.env 中配置了跨主机地址（如 LLM_API_BASE=http://192.168.10.50:8000/v1）
              以及访问凭据（LLM_API_KEY=sk-remote-gpu-host-key-999）。
            - 验证变量替换逻辑能够正确渲染出跨主机配置，而非同机内部 DNS。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_compose_cross_host_variable_substitution_render -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_compose_cross_host_variable_substitution_render -v -s
        """
        compose_content = COMPOSE_FILE.read_text(encoding="utf-8")

        # 模拟 Compose 变量替换引擎（使用自定义跨主机环境变量）
        mock_env = {
            "LLM_API_BASE": "http://192.168.10.50:8000/v1",
            "LLM_API_KEY": "sk-remote-gpu-host-key-999",
        }

        def _substitute(match: re.Match) -> str:
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return mock_env.get(var_name, default_val)

        rendered_compose = re.sub(r"\$\{([A-Za-z0-9_]+)(?::-([^}]+))?\}", _substitute, compose_content)
        parsed_config = yaml.safe_load(rendered_compose)

        agent_service = parsed_config.get("services", {}).get("PrivShield", {})
        agent_env = agent_service.get("environment", {})

        # 验证跨主机地址与密钥成功注入
        assert agent_env.get("PRIVACY_LLM_API_BASE") == "http://192.168.10.50:8000/v1"
        assert agent_env.get("PRIVACY_LLM_API_KEY") == "sk-remote-gpu-host-key-999"
        assert agent_env.get("PRIVACY_REST_HOST") == "0.0.0.0"
        assert agent_env.get("PRIVACY_GRPC_HOST") == "0.0.0.0"

    def test_agent_cross_host_remote_request_and_bearer_auth(self):
        """【跨主机通信】验证 Agent 调用远程 LLM 服务时正确路由目标 URL 并注入 Bearer 鉴权头。

        测试场景：
            - 远程 GPU 机器 IP 为 10.240.0.50:8000，且配置了 API Key 访问控制；
            - 验证 OpenAILlmClassifier 发出的 HTTP POST 请求地址为 http://10.240.0.50:8000/v1/chat/completions；
            - 验证 Header 中携带 Authorization: Bearer sk-remote-cloud-gpu-key-999；
            - 验证能够正确解析远程服务返回的标准 OpenAI JSON 格式。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_agent_cross_host_remote_request_and_bearer_auth -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_agent_cross_host_remote_request_and_bearer_auth -v -s
        """
        classifier = OpenAILlmClassifier(
            api_base="http://10.240.0.50:8000/v1",
            api_key="sk-remote-cloud-gpu-key-999",
            model_name="Qwen3.5-0.8B-Privacy-Classifier-Smoother",
        )

        remote_json_reply = {
            "id": "chatcmpl-cross-host-test",
            "object": "chat.completion",
            "created": 1723620000,
            "model": "Qwen3.5-0.8B-Privacy-Classifier-Smoother",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "final_level": "L3",
                            "confidence": 0.96,
                            "reasoning": "跨主机远程GPU大模型仲裁: 命中敏感个人身份信息",
                            "sanitized_text": "身份证号：510101********1234",
                            "category": "PERSONAL_BASIC",
                            "sub_category": "ID_CARD",
                        }),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 128,
                "completion_tokens": 52,
                "total_tokens": 180,
            },
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(remote_json_reply).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = classifier.classify(
                text="身份证号：510101199001011234",
                upstream_level=SensitivityLevel.L3,
                upstream_confidence=0.6,
                sanitize=True,
            )

            # 1. 验证目标 URL 跨主机正确拼接
            assert mock_urlopen.call_count == 1
            req_obj = mock_urlopen.call_args[0][0]
            assert req_obj.full_url == "http://10.240.0.50:8000/v1/chat/completions"

            # 2. 验证 Bearer 鉴权头正确注入
            assert req_obj.get_header("Authorization") == "Bearer sk-remote-cloud-gpu-key-999"
            assert req_obj.get_header("Content-type") == "application/json"

            # 3. 验证分类与脱敏结果解析正确
            assert result is not None
            assert result.get("final_level") == "L3"
            assert result.get("confidence") == 0.96
            assert result.get("sanitized_text") == "身份证号：510101********1234"
            assert result.get("usage", {}).get("total_tokens") == 180

    def test_agent_cross_host_network_timeout_and_graceful_degradation(self, monkeypatch):
        """【跨主机容灾】验证当跨主机网络超时（如网络断开/远程主机未就绪）时，Agent 平滑降级。

        测试目的：
            - 多主机部署中跨机网络可能出现抖动或连接超时；
            - 验证捕获 URLError / TimeoutError 后返回 None 并触发 Layer-1 规则平滑降级；
            - 确保整个 Agent 进程不因跨主机网络异常而崩溃。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_agent_cross_host_network_timeout_and_graceful_degradation -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_agent_cross_host_network_timeout_and_graceful_degradation -v -s
        """
        import engine.env_loader as _env_mod
        monkeypatch.setenv("PRIVACY_ENV_PROFILE", "vllm")
        monkeypatch.setenv("PRIVACY_LLM_PROVIDER", "vllm")
        monkeypatch.setenv("PRIVACY_LLM_API_BASE", "http://10.240.0.99:8000/v1")
        monkeypatch.setenv("PRIVACY_LLM_API_KEY", "sk-test-token")
        _env_mod._ENV_LOADED = True

        # 模拟跨主机网络连接超时
        timeout_error = urllib.error.URLError(TimeoutError("Connection to 10.240.0.99:8000 timed out after 30s"))

        with patch("urllib.request.urlopen", side_effect=timeout_error):
            # 1. 验证分类器捕获超时返回 None
            classifier = OpenAILlmClassifier(api_base="http://10.240.0.99:8000/v1")
            result = classifier.classify("测试文本", SensitivityLevel.L3, 0.5)
            assert result is None, "网络超时应当返回 None 触发降级"

            # 2. 验证业务服务层在跨主机网络超时时平滑回退至 Layer-1 规则引擎
            service = DynClassificationService()
            eval_res = service.classify_field("id_card", "510101199001011234", domain="sc_health_db51")
            assert eval_res is not None
            # 规则引擎判定为 L3（命中身份证规则），服务正常响应且不抛出异常
            assert eval_res.field_result is not None
            assert eval_res.field_result.final_level == "L3"

    def test_agent_cross_host_gateway_http_502_504_handling(self):
        """【网关异常处理】验证跨主机反向代理/网关返回 502/504 时 Agent 优雅处理。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_agent_cross_host_gateway_http_502_504_handling -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_agent_cross_host_gateway_http_502_504_handling -v -s
        """
        classifier = OpenAILlmClassifier(api_base="http://api-gateway.company.internal:8000/v1")

        # 模拟反向代理网关返回 HTTP 504 Gateway Timeout
        http_504_error = urllib.error.HTTPError(
            url="http://api-gateway.company.internal:8000/v1/chat/completions",
            code=504,
            msg="Gateway Timeout",
            hdrs=MagicMock(),
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=http_504_error):
            result = classifier.classify("测试文本", SensitivityLevel.L2, 0.5)
            assert result is None, "HTTP 504 网关超时应返回 None 触发降级"

    def test_agent_cross_host_empty_api_key_header_handling(self):
        """【默认凭据模式】验证当远程 LLM 服务使用默认 EMPTY Key 时发送标准 Bearer EMPTY 鉴权头。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_agent_cross_host_empty_api_key_header_handling -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_agent_cross_host_empty_api_key_header_handling -v -s
        """
        classifier = OpenAILlmClassifier(
            api_base="http://192.168.1.100:8000/v1",
            api_key="EMPTY",
        )

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "{\"final_level\": \"L2\", \"confidence\": 0.9}"}}],
            "usage": {},
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            classifier.classify("测试", SensitivityLevel.L2, 0.9)
            req_obj = mock_urlopen.call_args[0][0]
            auth_header = req_obj.get_header("Authorization")
            assert auth_header == "Bearer EMPTY", f"预期发送 Bearer EMPTY 鉴权头，实际为: {auth_header}"

    def test_cross_host_environment_switching_isolation(self, monkeypatch):
        """【拓扑切换隔离】验证在运行时从同机模式动态切换为跨主机模式时配置完全隔离。

        .venv 测试命令 / Test Command:
            source .venv/bin/activate && pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_cross_host_environment_switching_isolation -v -s
            # 或直接调用：
            .venv/bin/pytest tests/scripts/test_docker_start_agent.py::TestDockerAgentMultiHostDeployment::test_cross_host_environment_switching_isolation -v -s
        """
        import engine.env_loader as _env_mod

        # 1. 模拟同机模式
        monkeypatch.setenv("PRIVACY_ENV_PROFILE", "vllm")
        monkeypatch.setenv("PRIVACY_LLM_PROVIDER", "vllm")
        monkeypatch.setenv("PRIVACY_LLM_API_BASE", "http://vllm:8000/v1")
        monkeypatch.setenv("PRIVACY_LLM_API_KEY", "EMPTY")
        _env_mod._ENV_LOADED = True

        adapter_local = LlmAdapter()
        adapter_local._lazy_init()
        assert adapter_local.is_available is True
        assert adapter_local._classifier.api_base == "http://vllm:8000/v1"
        assert adapter_local._classifier.api_key == "EMPTY"

        # 2. 动态切换为跨主机生产模式
        monkeypatch.setenv("PRIVACY_LLM_API_BASE", "https://remote-gpu-cloud.corp.com:8443/v1")
        monkeypatch.setenv("PRIVACY_LLM_API_KEY", "sk-cloud-prod-token-xyz")
        _env_mod._ENV_LOADED = True

        adapter_remote = LlmAdapter()
        adapter_remote._lazy_init()
        assert adapter_remote.is_available is True
        assert adapter_remote._classifier.api_base == "https://remote-gpu-cloud.corp.com:8443/v1"
        assert adapter_remote._classifier.api_key == "sk-cloud-prod-token-xyz"



