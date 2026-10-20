"""Versioned auth schema migrations (SQLite side of ``deploy/sql/auth``).

Go 侧用户库启动不做 DDL，表结构只由 ``authmigrate`` Job 推进。Python 运行时
复用同一组 ``*.up.sql`` 迁移，并以本模块做「迁移是否完整」的启动校验；
真正的建表 / 升级由 ``python -m engine.auth.migrate`` 执行。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

VERSION_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    " version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
)


def migration_files(directory: str | Path) -> list[Path]:
    path = Path(directory)
    files = sorted(path.glob("*.up.sql"))
    if not files:
        raise RuntimeError(f"auth migrations not found in {path}")
    return files


def _version(file: Path) -> str:
    return file.name.split("_", 1)[0] or file.stem


def applied_versions(conn: sqlite3.Connection) -> set[str]:
    try:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {row[0] for row in rows}


def pending_migrations(conn: sqlite3.Connection, directory: str | Path) -> list[Path]:
    applied = applied_versions(conn)
    return [
        file
        for file in migration_files(directory)
        if _version(file) not in applied and file.name not in applied
    ]


def apply_migrations(
    conn: sqlite3.Connection, directory: str | Path, *, applied_at: str
) -> list[str]:
    """Apply pending migrations in filename order; returns the versions applied."""
    conn.execute(VERSION_TABLE_DDL)
    applied: list[str] = []
    for file in pending_migrations(conn, directory):
        script = file.read_text(encoding="utf-8")
        try:
            conn.executescript(script)
        except sqlite3.Error as exc:
            raise RuntimeError(f"failed to apply auth migration {file.name}: {exc}") from exc
        version = _version(file)
        conn.execute(
            "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, applied_at),
        )
        applied.append(file.name)
    conn.commit()
    return applied


def verify_schema(conn: sqlite3.Connection, directory: str | Path) -> None:
    """Fail closed when the database is behind the canonical migration set."""
    pending = pending_migrations(conn, directory)
    if pending:
        names = ", ".join(file.name for file in pending)
        raise RuntimeError(
            f"auth schema is not migrated ({names}); run `python -m engine.auth.migrate` "
            "before starting the service"
        )
