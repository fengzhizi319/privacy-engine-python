"""``python -m engine.auth.migrate`` — versioned auth schema migration CLI.

与 Go ``pkg/store/cmd/authmigrate`` 同口径：读取
``PRIVACY_AUTH_USER_STORE_DSN`` / ``PRIVACY_AUTH_USER_STORE_FILE`` 与
``PRIVACY_AUTH_USER_STORE_MIGRATIONS_DIR``，按文件名顺序幂等执行
``*.up.sql`` 并在 ``schema_migrations`` 去重。应用进程自身不做 DDL。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

from .migrations import apply_migrations, migration_files, pending_migrations
from .models import isoformat, now_utc
from .persistence import default_migrations_dir


def _resolve_target(dsn: str) -> Path:
    target = dsn.strip()
    if target.startswith(("postgres://", "postgresql://")):
        raise SystemExit(
            "postgres migrations are not available in this build: the psycopg driver "
            "is not installed; use the Go authmigrate Job for PostgreSQL"
        )
    if target.startswith("file:"):
        target = target[len("file:"):]
    target = target.split("?", 1)[0].strip()
    if not target:
        raise SystemExit(
            "no SQLite target configured: pass --dsn or set "
            "PRIVACY_AUTH_USER_STORE_FILE / PRIVACY_AUTH_USER_STORE_DSN"
        )
    return Path(target).expanduser()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="apply PrivShield auth migrations")
    parser.add_argument(
        "--dsn",
        default=(
            os.environ.get("PRIVACY_AUTH_USER_STORE_DSN", "").strip()
            or os.environ.get("PRIVACY_AUTH_USER_STORE_FILE", "").strip()
        ),
        help="sqlite path or file: URI (defaults to PRIVACY_AUTH_USER_STORE_FILE)",
    )
    parser.add_argument(
        "--dir",
        default=os.environ.get("PRIVACY_AUTH_USER_STORE_MIGRATIONS_DIR", "").strip(),
        help="directory holding *.up.sql (defaults to deploy/sql/auth)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print pending migrations only"
    )
    args = parser.parse_args(argv)

    target = _resolve_target(args.dsn)
    directory = Path(args.dir).expanduser() if args.dir else default_migrations_dir()
    if not directory.is_dir():
        raise SystemExit(f"auth migrations directory not found: {directory}")

    with sqlite3.connect(str(target)) as conn:
        if args.dry_run:
            pending = pending_migrations(conn, directory)
            if not pending:
                print("no pending auth migrations")
                return 0
            for file in pending:
                print(f"pending\t{file.name}")
            return 0

        before = {file.name for file in migration_files(directory)}
        applied = apply_migrations(conn, directory, applied_at=isoformat(now_utc()))
        if applied:
            for name in applied:
                print(f"applied\t{name}")
        else:
            print(f"no pending auth migrations ({len(before)} already applied)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
