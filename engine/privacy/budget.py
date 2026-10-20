"""隐私预算记账模块。

提供单命名空间（namespace）下的隐私预算跟踪，通过 BudgetRegistry 注册表工厂
保证同一命名空间只存在一份预算状态，并通过锁保证并发安全。当累计消耗超过总预算
时抛出 PrivacyBudgetExhausted 异常。

支持基于时间窗口的预算重置：可通过构造函数参数或环境变量
PRIVACY_BUDGET_WINDOW_SECONDS 配置窗口长度，窗口到期后已消耗预算自动清零。

Privacy budget accounting module. Tracks per-namespace epsilon/delta consumption
using a thread-safe registry factory (BudgetRegistry) and raises PrivacyBudgetExhausted when
budget is exhausted. Supports optional time-window based reset to prevent
long-running sidecars from permanently exhausting their budget.
"""

import contextlib
import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time

from ..observability.logging_config import get_logger
from ..observability.metrics import BUDGET_REMAINING

# Module-level structured logger for budget events
logger = get_logger(__name__)


class BudgetAuditLogger:
    """不可篡改的 HMAC 签名隐私预算审计日志。

    审计密钥通过环境变量 PRIVACY_AUDIT_KEY 或显式构造参数配置；
    若均未提供，则使用 ``secrets.token_hex(32)`` 生成进程级随机密钥并输出警告。
    进程级随机密钥意味着进程重启后无法校验重启前写入的审计记录签名；
    生产环境务必通过环境变量或构造参数设置高强度随机密钥并妥善保管。
    """

    def __init__(self, secret_key: bytes | None = None, log_file: str | None = None) -> None:
        """初始化审计日志器。

        Args:
            secret_key: HMAC 签名密钥；为 None 时从环境变量 PRIVACY_AUDIT_KEY 读取；
                两者均未提供时生成进程级随机密钥（重启后旧记录不可校验）。
            log_file: 审计日志文件路径；为 None 时从环境变量 PRIVACY_BUDGET_AUDIT_LOG 读取。
        """
        env_key = os.environ.get("PRIVACY_AUDIT_KEY")
        if secret_key is not None:
            self.secret_key = secret_key
        elif env_key is not None:
            self.secret_key = env_key.encode("utf-8")
        else:
            # 不再使用随源码公开的硬编码默认密钥（HMAC 签名可被任何人伪造）；
            # 改为进程级随机密钥：签名在本进程生命周期内可校验，重启后旧记录不可校验。
            self.secret_key = secrets.token_hex(32).encode("utf-8")
            logger.warning(
                "audit_logger_process_random_key",
                extra={
                    "detail": "No audit key configured; generated a process-level random HMAC key. "
                    "Audit signatures cannot be verified after process restart.",
                    "recommendation": "Set PRIVACY_AUDIT_KEY env var (or pass secret_key) for production.",
                },
            )
        # Default audit path; configurable via PRIVACY_BUDGET_AUDIT_LOG env var.
        default_path = "/tmp/budget_audit.log"  # noqa: S108
        self.log_file = log_file or os.environ.get("PRIVACY_BUDGET_AUDIT_LOG", default_path)
        self._lock = threading.Lock()

    def log_spend(
        self, namespace: str, epsilon: float, delta: float, eps_spent: float, del_spent: float
    ) -> str:
        """记录一次预算消耗并返回 HMAC-SHA256 签名。

        Args:
            namespace: 预算命名空间。
            epsilon: 总 epsilon 预算。
            delta: 总 delta 预算。
            eps_spent: 已消耗 epsilon。
            del_spent: 已消耗 delta。

        Returns:
            HMAC 签名的十六进制字符串。
        """
        with self._lock:
            ts = time.time()
            msg = f"{ts:.4f}|{namespace}|{epsilon:.6f}|{delta:.8f}|{eps_spent:.6f}|{del_spent:.8f}"
            signature = hmac.new(self.secret_key, msg.encode("utf-8"), hashlib.sha256).hexdigest()
            log_line = f"{msg}|{signature}\n"
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_line)
            except Exception as e:
                import logging
                logging.getLogger("engine.privacy.budget").warning(
                    f"BudgetAuditLogger: failed to write audit log to '{self.log_file}': {e}"
                )
            return signature


class PrivacyBudgetExhaustedError(Exception):
    """隐私预算已耗尽异常。

    当某命名空间下的 epsilon 或 delta 累计消耗超过预设上限时抛出。
    Raised when the privacy budget (epsilon or delta) is exhausted for a namespace.
    """

    pass


# Deprecated alias for backward compatibility
PrivacyBudgetExhausted = PrivacyBudgetExhaustedError



class BudgetAccountant:
    """隐私预算会计师。

    记录指定 namespace 的总预算与已消耗预算。
    实例的创建与获取由 BudgetRegistry 统一管理（参见 default_registry）。

    **直接构造已关闭**：调用 ``BudgetAccountant("ns")`` 会抛出 ``TypeError``。
    请使用 ``default_registry.get_or_create("ns")`` 或便捷函数 ``get_budget("ns")``。

    注意：
    - ``__init__`` 是空操作保护：真实的初始化逻辑在 ``_init_instance`` 中，
      由 BudgetRegistry 在创建新实例时调用一次。Python 在 ``__new__`` 返回
      已有实例后仍会调用 ``__init__``，空实现可防止已有实例被重复初始化
      （例如已消耗预算被悄悄清零）。
    - 实例级锁 _mu 控制 spend/remaining 操作，确保多线程环境下的预算扣减是原子的。
    - 若配置了环境变量 PRIVACY_BUDGET_DB，则将预算存储至 SQLite 中以支持多进程/多节点共享。

    可选的时间窗口重置机制：
    - 通过 window_seconds 参数或 PRIVACY_BUDGET_WINDOW_SECONDS 环境变量配置窗口。
    - 窗口到期后，已消耗预算自动清零，新窗口从当前时间开始。
    - 在 SQLite 模式下，窗口信息也会持久化，以便多实例共享一致的时间边界。

    Attributes:
        namespace: 当前实例所属的命名空间。
        epsilon_total: epsilon 总预算。
        delta_total: delta 总预算。
        epsilon_spent: 已消耗 epsilon。
        delta_spent: 已消耗 delta。
        window_seconds: 预算重置时间窗口（秒），None 表示不重置。
        _window_start: 当前窗口开始时间（UNIX 时间戳）。
        _mu: 实例级锁，保护预算扣减与查询的原子性。
    """

    def __new__(
        cls,
        namespace: str,
        epsilon_total: float | None = None,
        delta_total: float | None = None,
        window_seconds: float | None = None,
    ) -> "BudgetAccountant":
        """禁止直接构造。

        BudgetAccountant 不支持直接实例化。请使用 Registry API 获取或创建实例：

        .. code-block:: python

            from engine.privacy.budget import default_registry
            acct = default_registry.get_or_create("my-namespace")

        Raises:
            TypeError: 始终抛出，提示使用 ``default_registry.get_or_create()``。
        """
        raise TypeError(
            "BudgetAccountant cannot be instantiated directly. "
            "Use default_registry.get_or_create() instead."
        )

    def __init__(
        self,
        namespace: str,
        epsilon_total: float | None = None,
        delta_total: float | None = None,
        window_seconds: float | None = None,
    ) -> None:
        """空操作保护。

        真实的初始化逻辑在 ``_init_instance`` 中，由 BudgetRegistry 创建新实例时
        调用一次。Python 在 ``__new__`` 返回已有实例后仍会调用 ``__init__``，
        这里保持空实现，避免已有实例的预算状态被重复初始化而悄悄清零。
        """

    def _init_instance(
        self,
        namespace: str,
        epsilon_total: float = 10.0,
        delta_total: float = 1e-4,
        window_seconds: float | None = None,
    ) -> None:
        """内部初始化方法（由 BudgetRegistry 创建实例时调用）。"""
        self.namespace = namespace
        self.epsilon_total = epsilon_total
        self.delta_total = delta_total
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self._mu = threading.Lock()
        # Reuse a single BudgetAuditLogger to avoid re-reading env vars and lock creation per spend
        self._audit_logger = BudgetAuditLogger()

        # 解析时间窗口：显式参数 > 环境变量 > 默认 None
        env_window = os.environ.get("PRIVACY_BUDGET_WINDOW_SECONDS")
        if window_seconds is None and env_window is not None:
            try:
                window_seconds = float(env_window)
            except ValueError:
                window_seconds = None
        self.window_seconds = window_seconds
        self._window_start = time.time()

        # 分布式 Redis 后端初始化（可选）
        self._redis_client = None
        self._redis_url = os.environ.get("PRIVACY_BUDGET_REDIS_URL")
        budget_backend = os.environ.get("PRIVACY_BUDGET_BACKEND", "").lower()
        if budget_backend == "redis" or self._redis_url:
            self._init_redis()

        self._init_db()

    def _init_redis(self) -> None:
        """初始化 Redis 分布式预算记账客户端（若安装了 redis 且配置了连接）。"""
        try:
            import redis
            url = self._redis_url or "redis://127.0.0.1:6379/0"
            self._redis_client = redis.Redis.from_url(url, socket_timeout=3.0, decode_responses=True)
            # 测试连接
            self._redis_client.ping()
            logger.info(
                "budget_redis_backend_initialized",
                extra={"namespace": self.namespace, "redis_url": url},
            )
        except ImportError:
            logger.warning(
                "budget_redis_not_installed",
                extra={
                    "detail": "redis python package is not installed; falling back to SQLite or in-memory mode.",
                    "recommendation": "Install redis (pip install redis) to enable distributed budget accounting.",
                },
            )
            self._redis_client = None
        except Exception as e:
            logger.warning(
                "budget_redis_connect_failed",
                extra={
                    "detail": f"Failed to connect to Redis at '{self._redis_url}': {e}. Falling back to SQLite/memory.",
                },
            )
            self._redis_client = None

    @contextlib.contextmanager
    def _db_conn(self, db_path: str):
        """按操作创建并关闭 SQLite 连接的上下文管理器。

        修复旧版 ``threading.local()`` 缓存导致的连接泄漏：
        线程池（如 FastAPI / gRPC 线程池、dynclassification ThreadPoolExecutor）
        中的线程退出时不会自动关闭 thread-local 连接，长期运行会造成文件描述符
        与 WAL 资源泄漏。改为每次操作独立打开/关闭连接，配合 WAL 模式与 busy
        timeout，仍能在中等并发下保持可用。

        连接创建时自动开启 WAL 模式与相关优化 pragma：
        - ``journal_mode=WAL``：读写不再互斥，写吞吐约 1K~3K TPS；
        - ``busy_timeout=10000``：遇锁时等待 10 秒而非立即报错；
        - ``synchronous=NORMAL``：WAL 模式下安全且 faster 的同步级别。

        Args:
            db_path: SQLite 数据库文件路径。

        Yields:
            sqlite3.Connection: 已配置好 WAL 与超时的数据库连接。
        """
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def _init_db(self) -> None:
        """初始化共享数据库，如果设置了 PRIVACY_BUDGET_DB 持久化路径。

        启动时执行 PRAGMA integrity_check 校验数据库完整性（#8）。配置的
        预算库损坏时必须拒绝初始化，不能回退到内存状态，否则重启后可能绕过
        已持久化的预算消耗。
        """
        db_path = os.environ.get("PRIVACY_BUDGET_DB")
        if db_path:
            # 启动时完整性校验（#8）：检测突然断电或文件系统故障导致的数据库损坏
            if os.path.exists(db_path):
                try:
                    check_conn = sqlite3.connect(db_path, timeout=10.0)
                    try:
                        result = check_conn.execute("PRAGMA integrity_check").fetchone()
                        if result and result[0] != "ok":
                            logger.error(
                                "budget_db_integrity_check_failed",
                                extra={"db_path": db_path, "result": result[0]},
                            )
                            raise RuntimeError(
                                f"privacy budget database integrity check failed: {result[0]}"
                            )
                    finally:
                        check_conn.close()
                except Exception as e:
                    logger.error(
                        "budget_db_integrity_check_error",
                        extra={"db_path": db_path, "error": str(e)},
                    )
                    raise RuntimeError(
                        f"privacy budget database integrity check failed: {e}"
                    ) from e

            parent_dir = os.path.dirname(db_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            conn = sqlite3.connect(db_path, timeout=10.0)
            try:
                # 初始化阶段也开启 WAL 模式，确保数据库文件本身已切换为 WAL 格式
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA synchronous=NORMAL")
                with conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS privacy_budgets ("
                        "namespace TEXT PRIMARY KEY, "
                        "epsilon_total REAL, "
                        "delta_total REAL, "
                        "epsilon_spent REAL, "
                        "delta_spent REAL, "
                        "window_seconds REAL, "
                        "window_start REAL"
                        ")"
                    )
                    # 兼容旧表：若无 window_seconds/window_start 列则添加
                    cursor = conn.execute(
                        "PRAGMA table_info(privacy_budgets)"
                    )
                    columns = {row[1] for row in cursor.fetchall()}
                    if "window_seconds" not in columns:
                        conn.execute(
                            "ALTER TABLE privacy_budgets ADD COLUMN window_seconds REAL"
                        )
                    if "window_start" not in columns:
                        conn.execute(
                            "ALTER TABLE privacy_budgets ADD COLUMN window_start REAL"
                        )
                    # 预插入/更新当前的 budget 信息（如果尚未存在）
                    conn.execute(
                        "INSERT OR IGNORE INTO privacy_budgets "
                        "(namespace, epsilon_total, delta_total, "
                        "epsilon_spent, delta_spent, window_seconds, window_start) "
                        "VALUES (?, ?, ?, 0.0, 0.0, ?, ?)",
                        (
                            self.namespace,
                            self.epsilon_total,
                            self.delta_total,
                            self.window_seconds,
                            self._window_start,
                        ),
                    )
                    # 若记录已存在但缺少窗口信息，补充更新
                    conn.execute(
                        "UPDATE privacy_budgets SET window_seconds = COALESCE(window_seconds, ?), "
                        "window_start = COALESCE(window_start, ?) WHERE namespace = ?",
                        (self.window_seconds, self._window_start, self.namespace),
                    )
            finally:
                conn.close()

    def _now(self) -> float:
        """返回当前 UNIX 时间戳（便于测试时 mock）。"""
        return time.time()

    def _window_expired(self, window_start: float | None) -> bool:
        """判断当前窗口是否已经过期。"""
        if self.window_seconds is None or self.window_seconds <= 0:
            return False
        if window_start is None:
            return True
        return self._now() >= window_start + self.window_seconds

    def _reset_window(self) -> None:
        """重置当前预算窗口（仅在持有锁时调用）。"""
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self._window_start = self._now()

    def spend(self, epsilon: float, delta: float = 0.0) -> None:
        """消耗隐私预算。

        在加锁环境下计算新的累计消耗量；若超过总预算则抛出异常，
        否则更新已消耗预算。若配置了时间窗口，窗口到期后先自动清零已消耗预算。
        若配置了 SQLite，则通过排他性写入事务保证多进程一致性。

        Args:
            epsilon: 本次操作需要消耗的 epsilon 预算（必须 > 0）。
            delta: 本次操作需要消耗的 delta 预算（必须 >= 0），默认 0。

        Raises:
            ValueError: 当 epsilon <= 0 或 delta < 0 时。
            PrivacyBudgetExhausted: 当累计 epsilon 或 delta 超过总预算时。
        """
        if epsilon <= 0.0 or delta < 0.0:
            raise ValueError(f"Budget amount to spend must be strictly positive (got epsilon={epsilon}, delta={delta})")

        # 1. 分布式 Redis 后端优先
        if self._redis_client:
            with self._mu:
                redis_key = f"privshield:budget:{self.namespace}"
                window_sec = self.window_seconds or 0.0
                now_ts = self._now()
                # Lua 脚本保证原子性判断与扣减
                lua_script = """
                local key = KEYS[1]
                local eps_spend = tonumber(ARGV[1])
                local del_spend = tonumber(ARGV[2])
                local eps_total = tonumber(ARGV[3])
                local del_total = tonumber(ARGV[4])
                local win_sec = tonumber(ARGV[5])
                local now_ts = tonumber(ARGV[6])

                local data = redis.call('HMGET', key, 'eps_spent', 'del_spent', 'win_start', 'eps_total', 'del_total')
                local eps_spent = tonumber(data[1]) or 0.0
                local del_spent = tonumber(data[2]) or 0.0
                local win_start = tonumber(data[3]) or now_ts
                local cur_eps_total = tonumber(data[4]) or eps_total
                local cur_del_total = tonumber(data[5]) or del_total

                if win_sec > 0 and (now_ts - win_start) >= win_sec then
                    eps_spent = 0.0
                    del_spent = 0.0
                    win_start = now_ts
                end

                local new_eps = eps_spent + eps_spend
                local new_del = del_spent + del_spend

                if new_eps > (cur_eps_total + 1e-12) or new_del > (cur_del_total + 1e-12) then
                    return {-1, eps_spent, del_spent, cur_eps_total, cur_del_total}
                end

                redis.call('HMSET', key, 'eps_spent', new_eps, 'del_spent', new_del, 'win_start', win_start, 'eps_total', cur_eps_total, 'del_total', cur_del_total)
                return {1, new_eps, new_del, cur_eps_total, cur_del_total}
                """
                try:
                    res = self._redis_client.eval(
                        lua_script, 1, redis_key,
                        epsilon, delta, self.epsilon_total, self.delta_total, window_sec, now_ts
                    )
                    status, new_eps, new_del, eps_tot, del_tot = res
                    if status == -1:
                        raise PrivacyBudgetExhausted(
                            f"Privacy budget exhausted in namespace {self.namespace} (Redis): "
                            f"epsilon={new_eps + epsilon}/{eps_tot}, delta={new_del + delta}/{del_tot}"
                        )
                    self.epsilon_spent = float(new_eps)
                    self.delta_spent = float(new_del)
                    self._update_metrics(float(eps_tot), float(del_tot), self.epsilon_spent, self.delta_spent)
                    self._audit_logger.log_spend(self.namespace, float(eps_tot), float(del_tot), self.epsilon_spent, self.delta_spent)
                    logger.info("budget_spend_completed", extra={"namespace": self.namespace, "epsilon": epsilon, "delta": delta, "eps_spent": self.epsilon_spent, "del_spent": self.delta_spent, "backend": "redis"})
                    return
                except PrivacyBudgetExhausted:
                    raise
                except Exception as e:
                    logger.warning("budget_redis_eval_failed", extra={"detail": f"Redis eval failed: {e}; falling back to SQLite/memory."})

        # 2. SQLite 持久化后端
        db_path = os.environ.get("PRIVACY_BUDGET_DB")
        if db_path:
            with self._mu, self._db_conn(db_path) as conn:
                try:
                    # Ensure a clean transaction state in case a previous error left it rolled back
                    conn.rollback()
                    # 使用 BEGIN IMMEDIATE 排他性事务锁定数据库
                    conn.execute("BEGIN IMMEDIATE")
                    cursor = conn.execute(
                        "SELECT epsilon_total, delta_total, epsilon_spent, delta_spent, window_seconds, window_start "
                        "FROM privacy_budgets WHERE namespace = ?",
                        (self.namespace,),
                    )
                    row = cursor.fetchone()
                    if row:
                        (
                            eps_total,
                            del_total,
                            eps_spent,
                            del_spent,
                            db_window_seconds,
                            db_window_start,
                        ) = row
                        # 同步窗口配置（首次从 DB 读取）
                        if self.window_seconds is None and db_window_seconds is not None:
                            self.window_seconds = float(db_window_seconds)
                    else:
                        eps_total, del_total, eps_spent, del_spent = (
                            self.epsilon_total,
                            self.delta_total,
                            0.0,
                            0.0,
                        )
                        db_window_start = self._window_start
                        conn.execute(
                            "INSERT INTO privacy_budgets "
                            "(namespace, epsilon_total, delta_total, "
                            "epsilon_spent, delta_spent, window_seconds, window_start) "
                            "VALUES (?, ?, ?, 0.0, 0.0, ?, ?)",
                            (self.namespace, eps_total, del_total, self.window_seconds, self._window_start),
                        )

                    # 窗口到期则重置
                    if self._window_expired(db_window_start):
                        eps_spent = 0.0
                        del_spent = 0.0
                        self._window_start = self._now()

                    new_eps = eps_spent + epsilon
                    new_delta = del_spent + delta

                    if new_eps > eps_total + 1e-12 or new_delta > del_total + 1e-12:
                        conn.rollback()
                        raise PrivacyBudgetExhausted(
                            f"Privacy budget exhausted in namespace {self.namespace}: "
                            f"epsilon={new_eps}/{eps_total}, delta={new_delta}/{del_total}"
                        )

                    conn.execute(
                        "UPDATE privacy_budgets SET epsilon_spent = ?, delta_spent = ?, window_start = ? "
                        "WHERE namespace = ?",
                        (new_eps, new_delta, self._window_start, self.namespace),
                    )
                    conn.commit()
                    self.epsilon_spent = new_eps
                    self.delta_spent = new_delta
                    self._update_metrics(eps_total, del_total, new_eps, new_delta)
                    self._audit_logger.log_spend(
                        self.namespace, eps_total, del_total, new_eps, new_delta
                    )
                    logger.info(
                        "budget_spend_completed",
                        extra={
                            "namespace": self.namespace,
                            "epsilon": epsilon,
                            "delta": delta,
                            "eps_spent": new_eps,
                            "del_spent": new_delta,
                            "backend": "sqlite",
                        },
                    )
                except PrivacyBudgetExhausted:
                    raise
                except Exception:
                    conn.rollback()
                    raise
        else:
            with self._mu:
                # 窗口到期则重置
                if self._window_expired(self._window_start):
                    self._reset_window()

                new_eps = self.epsilon_spent + epsilon
                new_delta = self.delta_spent + delta
                if new_eps > self.epsilon_total + 1e-12 or new_delta > self.delta_total + 1e-12:
                    raise PrivacyBudgetExhausted(
                        f"Privacy budget exhausted in namespace {self.namespace}: "
                        f"epsilon={new_eps}/{self.epsilon_total}, delta={new_delta}/{self.delta_total}"
                    )
                self.epsilon_spent = new_eps
                self.delta_spent = new_delta
                self._update_metrics(
                    self.epsilon_total, self.delta_total, new_eps, new_delta
                )
                self._audit_logger.log_spend(
                    self.namespace, self.epsilon_total, self.delta_total, new_eps, new_delta
                )
                logger.info(
                    "budget_spend_completed",
                    extra={
                        "namespace": self.namespace,
                        "epsilon": epsilon,
                        "delta": delta,
                        "eps_spent": new_eps,
                        "del_spent": new_delta,
                        "backend": "memory",
                    },
                )

    def _update_metrics(
        self,
        epsilon_total: float,
        delta_total: float,
        epsilon_spent: float,
        delta_spent: float,
    ) -> None:
        """Update Prometheus gauges for remaining budget."""
        BUDGET_REMAINING.labels(namespace=self.namespace, budget_type="epsilon").set(
            epsilon_total - epsilon_spent
        )
        BUDGET_REMAINING.labels(namespace=self.namespace, budget_type="delta").set(
            delta_total - delta_spent
        )

    def remaining(self) -> dict[str, float]:
        """查询剩余隐私预算。

        若配置了时间窗口且窗口已到期，会先自动重置已消耗预算，再返回剩余量。

        Returns:
            包含 epsilon 与 delta 剩余量的字典，例如
            {"epsilon": 8.0, "delta": 1e-4}。
        """
        # 1. Redis 分布式查询
        if self._redis_client:
            with self._mu:
                redis_key = f"privshield:budget:{self.namespace}"
                try:
                    data = self._redis_client.hmget(redis_key, ["eps_spent", "del_spent", "win_start", "eps_total", "del_total"])
                    eps_spent = float(data[0]) if data[0] is not None else 0.0
                    del_spent = float(data[1]) if data[1] is not None else 0.0
                    win_start = float(data[2]) if data[2] is not None else self._window_start
                    eps_total = float(data[3]) if data[3] is not None else self.epsilon_total
                    del_total = float(data[4]) if data[4] is not None else self.delta_total

                    if self._window_expired(win_start):
                        eps_spent = 0.0
                        del_spent = 0.0
                        self._window_start = self._now()
                        self._redis_client.hset(redis_key, mapping={"eps_spent": 0.0, "del_spent": 0.0, "win_start": self._window_start})

                    self._update_metrics(eps_total, del_total, eps_spent, del_spent)
                    return {
                        "epsilon": max(0.0, eps_total - eps_spent),
                        "delta": max(0.0, del_total - del_spent),
                    }
                except Exception as e:
                    logger.warning("budget_redis_remaining_failed", extra={"detail": f"Redis remaining query failed: {e}; falling back."})

        # 2. SQLite 持久化查询
        db_path = os.environ.get("PRIVACY_BUDGET_DB")
        if db_path:
            with self._mu, self._db_conn(db_path) as conn:
                try:
                    # Ensure a clean transaction state in case a previous error left it rolled back
                    conn.rollback()
                    # 使用 BEGIN IMMEDIATE 排他性事务：窗口检查 + 重置必须与其他进程的
                    # spend/remaining 互斥，否则并发下本进程的窗口重置 UPDATE 可能抹掉
                    # 其他进程已扣减的预算（竞态条件）。
                    conn.execute("BEGIN IMMEDIATE")
                    cursor = conn.execute(
                        "SELECT epsilon_total, delta_total, epsilon_spent, delta_spent, window_seconds, window_start "
                        "FROM privacy_budgets WHERE namespace = ?",
                        (self.namespace,),
                    )
                    row = cursor.fetchone()
                    if row:
                        (
                            eps_total,
                            del_total,
                            eps_spent,
                            del_spent,
                            db_window_seconds,
                            db_window_start,
                        ) = row
                        if self.window_seconds is None and db_window_seconds is not None:
                            self.window_seconds = float(db_window_seconds)
                        if self._window_expired(db_window_start):
                            eps_spent = 0.0
                            del_spent = 0.0
                            self._window_start = self._now()
                            conn.execute(
                                "UPDATE privacy_budgets SET epsilon_spent = 0.0, delta_spent = 0.0, window_start = ? "
                                "WHERE namespace = ?",
                                (self._window_start, self.namespace),
                            )
                        conn.commit()
                        self.epsilon_spent = eps_spent
                        self.delta_spent = del_spent
                        self._update_metrics(eps_total, del_total, eps_spent, del_spent)
                        return {
                            "epsilon": eps_total - eps_spent,
                            "delta": del_total - del_spent,
                        }
                    else:
                        conn.commit()
                        self._update_metrics(
                            self.epsilon_total, self.delta_total, 0.0, 0.0
                        )
                        return {
                            "epsilon": self.epsilon_total,
                            "delta": self.delta_total,
                        }
                except Exception:
                    conn.rollback()
                    raise
        else:
            with self._mu:
                if self._window_expired(self._window_start):
                    self._reset_window()
                self._update_metrics(
                    self.epsilon_total,
                    self.delta_total,
                    self.epsilon_spent,
                    self.delta_spent,
                )
                return {
                    "epsilon": self.epsilon_total - self.epsilon_spent,
                    "delta": self.delta_total - self.delta_spent,
                }

    def reset(self) -> None:
        """重置已消耗隐私预算为 0。"""
        with self._mu:
            reset_time = self._now()
            if self._redis_client:
                try:
                    redis_key = f"privshield:budget:{self.namespace}"
                    self._redis_client.hset(
                        redis_key,
                        mapping={
                            "eps_spent": 0.0,
                            "del_spent": 0.0,
                            "win_start": reset_time,
                        },
                    )
                except Exception as e:
                    raise RuntimeError("failed to persist privacy budget reset to Redis") from e
            db_path = os.environ.get("PRIVACY_BUDGET_DB")
            if db_path:
                try:
                    with self._db_conn(db_path) as conn:
                        conn.execute(
                            "UPDATE privacy_budgets SET epsilon_spent = 0.0, delta_spent = 0.0, window_start = ? WHERE namespace = ?",
                            (reset_time, self.namespace),
                        )
                        conn.commit()
                except Exception as e:
                    raise RuntimeError("failed to persist privacy budget reset to SQLite") from e
            self.epsilon_spent = 0.0
            self.delta_spent = 0.0
            self._window_start = reset_time
            self._update_metrics(self.epsilon_total, self.delta_total, 0.0, 0.0)

    def __repr__(self) -> str:
        """返回 BudgetAccountant 的可读字符串表示。"""
        eps_rem = self.epsilon_total - self.epsilon_spent
        del_rem = self.delta_total - self.delta_spent
        return (
            f"BudgetAccountant(namespace={self.namespace!r}, "
            f"epsilon={eps_rem:.4f}/{self.epsilon_total}, "
            f"delta={del_rem:.2e}/{self.delta_total})"
        )



class BudgetRegistry:
    """BudgetAccountant 的 namespace 级注册表工厂。

    集中管理预算会计师实例的创建、获取、检查、销毁与测试重置，
    替代原有的魔术方法单例机制。
    """

    def __init__(self) -> None:
        self._instances: dict[str, BudgetAccountant] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        namespace: str,
        epsilon_total: float | None = None,
        delta_total: float | None = None,
        window_seconds: float | None = None,
    ) -> BudgetAccountant:
        """获取已有实例，或在不存在时创建新实例。

        参数语义：
        - 实例不存在：未提供的参数使用默认值（epsilon_total=10.0, delta_total=1e-4）创建。
        - 实例已存在：仅对**显式提供**且与现有配置不一致的参数发出 UserWarning；
          未提供的参数视为"不关心"，不参与冲突检测。

        Returns:
            namespace 对应的 BudgetAccountant 实例（已有或新建）。
        """
        with self._lock:
            if namespace in self._instances:
                existing = self._instances[namespace]
                ignored = []
                if epsilon_total is not None and existing.epsilon_total != epsilon_total:
                    ignored.append(f"epsilon_total={epsilon_total}")
                if delta_total is not None and existing.delta_total != delta_total:
                    ignored.append(f"delta_total={delta_total}")
                if window_seconds is not None and existing.window_seconds != window_seconds:
                    ignored.append(f"window_seconds={window_seconds}")
                if ignored:
                    logger.warning(
                        "budget_registry_params_ignored",
                        extra={
                            "namespace": namespace,
                            "ignored_params": ", ".join(ignored),
                            "existing_epsilon_total": existing.epsilon_total,
                            "existing_delta_total": existing.delta_total,
                        },
                    )
                return existing

            accountant = object.__new__(BudgetAccountant)
            accountant._init_instance(
                namespace=namespace,
                epsilon_total=10.0 if epsilon_total is None else epsilon_total,
                delta_total=1e-4 if delta_total is None else delta_total,
                window_seconds=window_seconds,
            )
            self._instances[namespace] = accountant
            return accountant

    def get(self, namespace: str) -> BudgetAccountant | None:
        """获取已有实例，不存在则返回 None。"""
        with self._lock:
            return self._instances.get(namespace)

    def remove(self, namespace: str) -> BudgetAccountant | None:
        """销毁指定 namespace 的实例。"""
        with self._lock:
            return self._instances.pop(namespace, None)

    def reset(self) -> None:
        """清空所有注册的实例（测试隔离与全局重置用）。"""
        with self._lock:
            self._instances.clear()


# 全局默认注册表单例
default_registry = BudgetRegistry()


def get_budget(
    namespace: str,
    epsilon_total: float | None = None,
    delta_total: float | None = None,
    window_seconds: float | None = None,
) -> BudgetAccountant:
    """获取或创建指定命名空间的 BudgetAccountant 实例。

    模块级便捷函数，等价于 ``default_registry.get_or_create(namespace, ...)``。

    Args:
        namespace: 命名空间名称。
        epsilon_total: epsilon 总预算。
        delta_total: delta 总预算。
        window_seconds: 预算重置时间窗口（秒）。

    Returns:
        BudgetAccountant 实例。
    """
    return default_registry.get_or_create(
        namespace=namespace,
        epsilon_total=epsilon_total,
        delta_total=delta_total,
        window_seconds=window_seconds,
    )


class RDPAccountant:
    """Rényi 差分隐私（RDP）会计。

    基于 Mironov (2017) 提出的 Rényi Differential Privacy 组合定理：
    对于任意 alpha > 1，多个 RDP 查询的 epsilon_alpha 可以直接线性相加；
    在转换回经典 (epsilon, delta)-DP 时，选择最优的 alpha 值使 epsilon 最小：
        epsilon(delta) = min_{alpha > 1} ( epsilon_alpha + ln(1/delta) / (alpha - 1) )
    在多轮高斯加噪场景下相比传统 Basic Composition 可以节省大量隐私预算。
    """

    def __init__(self, target_delta: float = 1e-5) -> None:
        """初始化 RDPAccountant。

        Args:
            target_delta: 目标 delta 值，用于 RDP → (epsilon, delta)-DP 转换。
        """
        self.target_delta = target_delta
        # 记录各 order alpha 下累积的 RDP epsilon
        self.rdp_orders = [1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0]
        self.rdp_epsilons: dict[float, float] = dict.fromkeys(self.rdp_orders, 0.0)
        self._lock = threading.Lock()

    def record_gaussian(self, sigma: float, sensitivity: float = 1.0) -> None:
        """记录一次高斯机制下的 RDP 消耗。

        高斯机制在 order alpha 下的 RDP epsilon = alpha * sensitivity^2 / (2 * sigma^2)。
        """
        if sigma <= 0.0 or sensitivity <= 0.0:
            return
        with self._lock:
            for alpha in self.rdp_orders:
                eps_a = alpha * (sensitivity**2) / (2.0 * (sigma**2))
                self.rdp_epsilons[alpha] += eps_a

    def record_rdp(self, alpha: float, rdp_eps: float) -> None:
        """显式记录一次 (alpha, rdp_eps) 的 RDP 消耗。"""
        with self._lock:
            if alpha not in self.rdp_epsilons:
                self.rdp_orders.append(alpha)
                self.rdp_orders.sort()
                self.rdp_epsilons[alpha] = 0.0
            self.rdp_epsilons[alpha] += rdp_eps

    def get_epsilon(self, delta: float | None = None) -> float:
        """转换回经典 (epsilon, delta)-DP 下的最小 epsilon。"""
        d = delta if delta is not None else self.target_delta
        if d <= 0.0:
            return float("inf")
        with self._lock:
            best_eps = float("inf")
            import math

            for alpha in self.rdp_orders:
                eps_a = self.rdp_epsilons[alpha]
                # conversion formula: eps = eps_a + ln(1/delta)/(alpha - 1)
                eps = eps_a + math.log(1.0 / d) / (alpha - 1.0)
                if eps < best_eps:
                    best_eps = eps
            return best_eps

    def reset(self) -> None:
        """重置所有已累积的 RDP 消耗，保留原有 orders 配置。"""
        with self._lock:
            for alpha in self.rdp_orders:
                self.rdp_epsilons[alpha] = 0.0
