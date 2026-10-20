"""Environment variable loader for PrivShield with Profile Cascade Support.

Auto-detects and loads .env and profile-specific sub-config from config/env/ or project root
without requiring hardcoded logic in app code.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_LOADED = False


def _parse_and_load_file(filepath: Path, override: bool = False) -> bool:
    """内部辅助函数：解析并加载指定 Path 的 env 文件到 os.environ"""
    if not filepath.is_file():
        return False

    try:
        import dotenv

        dotenv.load_dotenv(dotenv_path=filepath, override=override)
        return True
    except ImportError:
        pass

    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key.startswith("export "):
                    key = key[7:].strip()
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]

                if override or key not in os.environ:
                    os.environ[key] = value
        return True
    except Exception:
        return False


def load_env_file(dotenv_path: str | Path | None = None, override: bool = False) -> bool:
    """自动级联加载项目根目录下的 .env 及特定场景文件 (config/env/<profile>.env) 至 os.environ。

    支持通过 PRIVACY_ENV_PROFILE / ENV_PROFILE / APP_ENV 指定场景模式 (如 vllm / qwen3 / mlx / openai)。

    Args:
        dotenv_path: .env 基础文件路径（默认自动从项目根目录寻找）。
        override: 是否覆盖已有的 os.environ 环境变量。

    Returns:
        True 表示成功找到并加载了 .env，False 表示未找到或加载失败。
    """
    global _ENV_LOADED

    if _ENV_LOADED and not override and not dotenv_path:
        return True

    pkg_root = Path(__file__).resolve().parent
    project_root = pkg_root.parent

    # 1. 寻找主 .env 文件
    target_path: Path | None = None
    if dotenv_path:
        target_path = Path(dotenv_path)
    else:
        possible_paths = [
            project_root / ".env",
            Path.cwd() / ".env",
        ]
        for p in possible_paths:
            if p.is_file():
                target_path = p
                break

    if not target_path or not target_path.is_file():
        return False

    # 记录显式传入的环境变量
    initial_profile = (
        os.environ.get("PRIVACY_ENV_PROFILE")
        or os.environ.get("ENV_PROFILE")
        or os.environ.get("APP_ENV")
    )

    # 加载基础 .env
    success = _parse_and_load_file(target_path, override=override)
    if not success:
        return False

    # 2. 检查环境变量中的场景 Profile
    profile = (
        initial_profile
        or os.environ.get("PRIVACY_ENV_PROFILE")
        or os.environ.get("ENV_PROFILE")
        or os.environ.get("APP_ENV")
    )

    if profile:
        profile = profile.strip().lower()
        # 优先在 config/env/ 目录下寻找 <profile>.env，实现集中式运维存放
        profile_candidates = [
            project_root / "config" / "env" / f"{profile}.env",
            target_path.parent / "config" / "env" / f"{profile}.env",
            project_root / "config" / "env" / f".env.{profile}",
            target_path.parent / f".env.{profile}",
            project_root / f".env.{profile}",
            Path.cwd() / f".env.{profile}",
        ]

        for p in profile_candidates:
            if p.is_file():
                # 场景配置文件优先级更高，使用 override=True 进行覆盖
                _parse_and_load_file(p, override=True)
                break

    _ENV_LOADED = True
    return True
