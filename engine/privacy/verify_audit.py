#!/usr/bin/env python3
"""Budget audit log HMAC signature verification script / 隐私预算审计日志 HMAC 签名校验工具。

校验 BudgetAuditLogger 写入的审计日志文件中每行的 HMAC-SHA256 签名是否完整、未被篡改。

审计日志格式（每行）：
    {timestamp}|{namespace}|{epsilon_total}|{delta_total}|{eps_spent}|{del_spent}|{hex_signature}

用法：
    # 使用环境变量 PRIVACY_AUDIT_KEY 校验默认审计日志
    PRIVACY_AUDIT_KEY=your-key python -m engine.privacy.verify_audit

    # 指定日志文件和密钥
    python -m engine.privacy.verify_audit --log-file /path/to/audit.log --key your-key

    # 从文件读取密钥
    python -m engine.privacy.verify_audit --log-file /path/to/audit.log --key-file /path/to/key

退出码：
    0 - 所有记录签名校验通过
    1 - 存在签名不匹配或格式错误的记录
    2 - 参数错误或文件不存在
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import sys


def verify_audit_log(log_file: str, secret_key: bytes) -> tuple[int, int, list[str]]:
    """校验审计日志文件中的 HMAC 签名。

    Args:
        log_file: 审计日志文件路径。
        secret_key: HMAC-SHA256 签名密钥。

    Returns:
        (total_lines, valid_count, errors) 元组：
        - total_lines: 总记录行数（跳过空行）；
        - valid_count: 签名校验通过的记录数；
        - errors: 校验失败的行号及原因列表。
    """
    total = 0
    valid = 0
    errors: list[str] = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1

            parts = line.split("|")
            if len(parts) != 7:
                errors.append(f"line {line_no}: expected 7 fields, got {len(parts)}")
                continue

            # 签名是最后一个字段，前面 6 个字段拼接为待校验消息
            signature = parts[6]
            message = "|".join(parts[:6])
            expected = hmac.new(secret_key, message.encode("utf-8"), hashlib.sha256).hexdigest()

            if hmac.compare_digest(signature, expected):
                valid += 1
            else:
                errors.append(
                    f"line {line_no}: signature mismatch "
                    f"(expected={expected[:16]}..., got={signature[:16]}...)"
                )

    return total, valid, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify HMAC-SHA256 signatures in budget audit log.",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("PRIVACY_BUDGET_AUDIT_LOG", "/tmp/budget_audit.log"),
        help="Path to the audit log file (default: $PRIVACY_BUDGET_AUDIT_LOG or /tmp/budget_audit.log).",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="HMAC secret key (hex string). Falls back to $PRIVACY_AUDIT_KEY env var.",
    )
    parser.add_argument(
        "--key-file",
        default=None,
        help="Path to a file containing the HMAC secret key (overrides --key and env var).",
    )
    args = parser.parse_args()

    # Resolve secret key: key-file > --key > env var
    secret_key: bytes | None = None
    if args.key_file:
        if not os.path.isfile(args.key_file):
            print(f"ERROR: key file not found: {args.key_file}", file=sys.stderr)
            sys.exit(2)
        with open(args.key_file, "r", encoding="utf-8") as f:
            secret_key = f.read().strip().encode("utf-8")
    elif args.key:
        secret_key = args.key.encode("utf-8")
    else:
        env_key = os.environ.get("PRIVACY_AUDIT_KEY")
        if env_key:
            secret_key = env_key.encode("utf-8")
        else:
            print(
                "ERROR: No audit key provided. Set PRIVACY_AUDIT_KEY env var, "
                "use --key, or --key-file.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Verify log file exists
    if not os.path.isfile(args.log_file):
        print(f"ERROR: audit log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(2)

    print(f"Verifying audit log: {args.log_file}")
    total, valid, errors = verify_audit_log(args.log_file, secret_key)

    print(f"Total records:  {total}")
    print(f"Valid records:   {valid}")
    print(f"Failed records:  {len(errors)}")

    if errors:
        print("\nFailed lines:")
        for err in errors[:50]:  # Show first 50 errors
            print(f"  {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more errors")
        print("\nRESULT: FAIL — some records have invalid signatures")
        sys.exit(1)
    else:
        if total == 0:
            print("\nRESULT: WARN — audit log is empty")
        else:
            print("\nRESULT: PASS — all records verified successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
