"""动态分类分级模块 CLI 入口 / Dynamic Classification CLI Entry Point.

提供命令行工具用于规则配置校验等操作。 / Provides command-line tools for operations like rule configuration validation.

Usage / 用法:
    python -m engine.dynclassification validate [rules_dir]
    python -m engine.dynclassification validate rules
    python -m engine.dynclassification validate /path/to/rules

Commands / 命令:
    validate    校验规则配置 YAML 文件合法性 / Validate rule configuration YAML files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """CLI 主入口。 / CLI main entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m engine.dynclassification",
        description="动态分类分级模块命令行工具 / Dynamic Classification CLI Tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令 / Available commands")

    # validate 子命令 / validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="校验规则配置 YAML 文件合法性 / Validate rule configuration YAML files",
    )
    validate_parser.add_argument(
        "rules_dir",
        nargs="?",
        default="rules",
        help="规则配置根目录路径（默认: rules） / Rules configuration root directory path (default: rules)",
    )

    args = parser.parse_args()

    if args.command == "validate":
        return _cmd_validate(args.rules_dir)
    else:
        parser.print_help()
        return 1


def _cmd_validate(rules_dir: str) -> int:
    """执行规则校验命令。 / Execute rule validation command."""
    from .validator import validate_rules_dir

    rules_path = Path(rules_dir)
    print(f"正在校验规则目录 / Validating rules directory: {rules_path.resolve()}")
    print("-" * 60)

    result = validate_rules_dir(rules_path)

    if result.errors:
        print(f"\n❌ 发现 {len(result.errors)} 个错误 / Found {len(result.errors)} errors:")
        for i, err in enumerate(result.errors, 1):
            print(f"  {i}. {err}")

    if result.warnings:
        print(f"\n⚠️  发现 {len(result.warnings)} 个警告 / Found {len(result.warnings)} warnings:")
        for i, warn in enumerate(result.warnings, 1):
            print(f"  {i}. {warn}")

    print("-" * 60)
    if result.is_valid:
        print("✅ 校验通过：所有规则配置文件合法 / Validation passed: All rule configuration files are valid")
        return 0
    else:
        print("❌ 校验失败：存在配置错误，请修复后重试 / Validation failed: Configuration errors exist, please fix and retry")
        return 1


if __name__ == "__main__":
    sys.exit(main())
