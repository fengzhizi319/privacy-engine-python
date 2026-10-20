"""tests/scripts pytest 配置 / pytest configuration for console script tests.

为 console/scripts 相关测试提供资源可用性探测 fixture：
- docker_available：docker CLI 与守护进程连通性
- docker_compose_available：docker compose 插件（v2）可用性
- gpu_available：Docker 能否向 vLLM 镜像分配 NVIDIA GPU
- vllm_image_available：本地是否已缓存 vLLM 镜像

这些 fixture 供集成测试使用；资源不可用时对应测试自动 skip，
保证 CI（无 GPU / 无 Docker）与本地环境均可安全运行。
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest


def _run_quiet(cmd: list[str]) -> bool:
    """执行命令并返回是否成功（静默，不向终端输出任何内容）。"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """检测 docker CLI 是否可用（docker version 能连通守护进程）。"""
    if shutil.which("docker") is None:
        return False
    return _run_quiet(["docker", "version"])


@pytest.fixture(scope="session")
def docker_compose_available() -> bool:
    """检测 docker compose 插件（v2）是否可用。"""
    if shutil.which("docker") is None:
        return False
    return _run_quiet(["docker", "compose", "version"])


@pytest.fixture(scope="session")
def gpu_available(docker_available: bool, vllm_image_available: bool) -> bool:
    """检测 Docker 是否能向 vLLM 镜像分配 NVIDIA GPU。"""
    if not docker_available or not vllm_image_available:
        return False
    tag = os.environ.get("VLLM_IMAGE_TAG", "latest")
    image = f"vllm/vllm-openai:{tag}"
    return _run_quiet(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "1",
            "--entrypoint",
            "python3",
            image,
            "-c",
            "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
        ]
    )


@pytest.fixture(scope="session")
def vllm_image_available(docker_available: bool) -> bool:
    """检测本地是否已缓存 vLLM 镜像（VLLM_IMAGE_TAG 环境变量或默认 latest）。"""
    if not docker_available:
        return False
    tag = os.environ.get("VLLM_IMAGE_TAG", "latest")
    return _run_quiet(["docker", "image", "inspect", f"vllm/vllm-openai:{tag}"])
