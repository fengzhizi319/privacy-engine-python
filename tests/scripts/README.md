# PrivShield 自动化运维与启动脚本测试套件 (Scripts Test Suite)

`tests/scripts/` 目录负责对 **数联天下 · 数盾 (`PrivShield`)** 平台下所有 Shell / PowerShell 运维、部署、容器化及服务启动脚本进行全方位的自动化回归测试与合规性验证。

---

## 1. 路径规范与执行基准说明 (Path & Working Directory Convention)

> 💡 **重要说明**：
> - **基准工作目录**：本文档中列出的所有 `pytest` 与 `make` 测试命令，均约定在 **项目根目录 (Project Root)** 下执行；
> - **路径自解析安全保证**：所有被测 Shell 脚本内部均采用 `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` 机制进行自定位，Python 测试用例采用 `Path(__file__).resolve()` 进行绝对路径推导。因此，**即使将本 `README.md` 复制或移动到其他目录，所有脚本与自动化测试的执行都不会受到任何路径影响**。

---

## 2. 目录结构与测试矩阵

```text
tests/scripts/
├── conftest.py                       # 共享 Pytest Fixture (系统解释器、环境净化、临时目录)
├── test_datasource_mgr_scripts.py    # 模拟数据源 (datasource-mgr) 启动与证书生成脚本测试
├── test_prod_scripts.py              # 生产运维、Helm/K8s 部署、备份与健康巡检脚本测试
├── test_docker_start_agent.py        # Agent (Core/ML) 容器启动脚本测试
├── test_docker_start_llm.py          # vLLM 大模型容器启动与 GPU 探测脚本测试
└── README.md                         # 本文档（脚本测试说明与执行手册）
```

### 测试用例覆盖矩阵

| 测试文件 | 目标被测脚本 | 核心测试阶段与验证内容 |
|---|---|---|
| **`test_datasource_mgr_scripts.py`** | • `services/datasource-mgr/scripts/dev-run.sh`<br/>• `services/datasource-mgr/scripts/prod-run.sh`<br/>• `services/datasource-mgr/scripts/gen-certs.sh`<br/>• `services/datasource-mgr/run.sh` | 1. 权限与静态文件存在性（0755 权限）<br/>2. `bash -n` 语法解析校验<br/>3. `gen-certs.sh` 动态生成证书链与客户端公钥固定文件 (`client.pub`)<br/>4. `dev-run.sh` 开发模式子进程拉起与 HTTP 探活<br/>5. `prod-run.sh` 生产 mTLS 模式拉起、端口绑定与优雅停机 |
| **`test_prod_scripts.py`** | • `scripts/prod/deploy-docker-compose.sh`<br/>• `scripts/prod/prod_health_check.sh`<br/>• `scripts/prod/backup_privacy_budget.sh`<br/>• `scripts/prod/deploy-helm.sh`<br/>• `scripts/prod/deploy-k8s.sh` 等 | 1. 生产 Shell/PowerShell 脚本完整性校验<br/>2. `--help` 帮助参数与选项解析逻辑<br/>3. 隐私预算 SQLite 在线热备份、Gzip 压缩与 SHA-256 校验和测试<br/>4. 生产健康巡检脚本各组件探活逻辑 |
| **`test_docker_start_agent.py`** | • `scripts/dev/docker-start-agent.sh`<br/>• `scripts/prod/docker-start-agent.sh` | 1. Core / ML 双架构镜像启动参数拼装<br/>2. 环境变量注入与数据卷映射校验<br/>3. 容器生命周期管理 |
| **`test_docker_start_llm.py`** | • `scripts/dev/docker-start-llm.sh`<br/>• `scripts/prod/docker-start-llm.sh` | 1. vLLM 模型参数解析与 GPU 显存配置<br/>2. 本地模型权重挂载路径检查<br/>3. 容器探活与异常回滚机制 |

---

## 3. 核心执行流程与阶段规范

本测试套件严格遵循分层测试规范，确保运维脚本在不同操作系统与环境下高可靠执行：

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ 阶段 1: 静态检查 (Static & Permission Checks)                               │
│ ├─ 确保所有目标脚本文件存在于代码库对应路径                                │
│ ├─ 校验 Linux/macOS 环境下文件具备所有者执行权限 (S_IXUSR / 0755)          │
│ └─ 校验持久化测试证书文件 (ca.crt, server.crt, client.pub 等) 完整就绪       │
├────────────────────────────────────────────────────────────────────────────┤
│ 阶段 2: 语法合规性检查 (Syntax Verification)                                │
│ ├─ 调用系统 bash 解释器执行 bash -n <script_path>                          │
│ └─ 杜绝未闭合引号、语法错误或平台特异性语法 break 问题                      │
├────────────────────────────────────────────────────────────────────────────┤
│ 阶段 3: 证书与数据产物动态验证 (Dynamic Artifact Checks)                    │
│ ├─ 在临时目录 (tmp_path) 隔离调用生成脚本                                  │
│ └─ 校验 X.509 证书 SAN、EKU 及提取公钥与证书的数学一致性                    │
├────────────────────────────────────────────────────────────────────────────┤
│ 阶段 4: 子进程生命周期与探活 (Subprocess Lifecycle & Graceful Shutdown)     │
│ ├─ 动态分配空闲随机端口（防端口冲突）                                      │
│ ├─ 净化子进程环境变量 (_clean_env 剔除超长变量防 Argument list too long)   │
│ ├─ 启动子进程 ➔ 轮询健康探活端点 (HTTP 200 / gRPC 监听)                     │
│ └─ 发送 SIGTERM 信号验证进程优雅注销并清理资源                              │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 运行测试指南

### 4.1 运行全部脚本测试

在项目根目录下执行：

```bash
# 激活 Python 虚拟环境并运行脚本测试
pytest tests/scripts/ -v
```

### 4.2 运行特定模块脚本测试

```bash
# 仅测试 datasource-mgr 启动与证书脚本
pytest tests/scripts/test_datasource_mgr_scripts.py -v
```

```bash
# 仅测试生产运维部署与备份脚本
pytest tests/scripts/test_prod_scripts.py -v
```

```bash
# 仅测试 Agent 容器启动脚本
pytest tests/scripts/test_docker_start_agent.py -v
```

```bash
# 仅测试 vLLM 本地大模型容器启动脚本
pytest tests/scripts/test_docker_start_llm.py -v
```

### 4.3 常用测试选项

```bash
# 显示标准输出与详细日志
pytest tests/scripts/ -v -s
```
```bash
# 遇到首个失败立即中断
pytest tests/scripts/ -x
```

```bash
# 仅执行快速测试（跳过较长的子进程拉起用例）
pytest tests/scripts/ -m "not slow"
```

---

## 5. 开发编写脚本测试注意事项

1. **环境变量防溢出**：在拉起子进程时，务必使用 `conftest.py` 或测试文件内的 `_clean_env()` 工具函数，过滤超长上下文环境变量，避免 Linux `E2BIG` (Argument list too long) 错误；
2. **端口隔离**：禁止在测试中使用固定端口（如 `:8083`），应统一通过 `_get_free_port()` 动态申请空闲端口；
3. **子进程安全清理**：所有 `subprocess.Popen` 调用必须放在 `try ... finally` 代码块中，在退出时先发送 `SIGTERM` 信号，若超时（如 3s）则强制 `kill()`，防止产生僵尸孤儿进程；
4. **跨平台兼容**：在 Windows 平台执行时自动跳过 POSIX 权限检查 (`S_IXUSR`) 与 Linux 专有信号。
