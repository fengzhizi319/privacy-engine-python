# PrivShield 全局 mTLS 测试证书库 (config/certs)

> 🔐 **用途说明**：本目录集中预置 PrivShield 各微服务与网关在本地联调、端到端集成测试（E2E）及 CI/CD 流水线中所需的自签名测试证书链（CA、服务端证书、客户端证书与 SPKI 公钥文件）。
> 
> **注意**：本目录下的证书仅供**开发调试与测试**使用，有效期设置为 **10 年（3650 天）**，避免频繁过期影响测试。**严禁用于生产环境**。

---

## 1. 证书文件清单

| 文件名 | 类型 / 格式 | 描述与用途 |
|---|---|---|
| `ca.crt` | X.509 证书 (PEM) | 受信任的自签名根证书授权机构（Root CA，4096-bit RSA） |
| `ca.key` | RSA 私钥 (PEM) | 根 CA 私钥（仅用于本地签发测试证书） |
| `server.crt` | X.509 证书 (PEM) | 服务端证书（SAN 包含 `localhost`、`127.0.0.1`，EKU: `serverAuth`） |
| `server.key` | RSA 私钥 (PEM) | 服务端私钥（RSA 2048-bit） |
| `client.crt` | X.509 证书 (PEM) | 客户端证书（包含 `clientAuth` 扩展密钥用法） |
| `client.key` | RSA 私钥 (PEM) | 客户端私钥（RSA 2048-bit） |
| `client.pub` | SPKI 公钥 (PEM) | 客户端公钥文件（用于应用层公钥固定 SPKI Pinning 比对校验） |

---

## 2. 各子服务测试证书对应目录

除了全局通用证书（`config/certs/`）外，各子服务亦维护其对应的专属测试证书：

* `console/bff-go/certs/`：Go BFF 代理网关测试证书
* `services/service-hub/certs/`：数据流通调度中枢测试证书
* `services/datasource-mgr/certs/`：数据源资产管理微服务测试证书
* `services/audit-log/certs/`：合规存证与审计日志微服务测试证书

---

## 3. 一键重新生成与轮换测试证书

如需重新生成全项目所有服务的测试证书，可直接运行根目录自动化脚本：

```bash
# 一键重新生成所有模块的 10 年有效期测试证书
bash ./scripts/dev/generate_all_test_certs.sh
```

或单独为某个子服务重新生成：

```bash
# 为 Go BFF 重新生成
bash ./console/bff-go/scripts/gen-certs.sh

# 为 Service Hub 重新生成
bash ./services/service-hub/scripts/gen-certs.sh

# 为 Datasource Mgr 重新生成
bash ./services/datasource-mgr/scripts/gen-certs.sh

# 为 Audit Log 重新生成
bash ./services/audit-log/scripts/gen-certs.sh
```
