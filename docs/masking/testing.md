# Python 数据脱敏测试指南 (Testing)

> **组件归属**：`services/privacy-engine-python`

---

## 1. 测试套件结构

- `tests/privacy/test_masking.py`：测试各类字段名推断、中国大陆手机号、18位身份证、银行卡 Luhn 规则、姓名、邮箱、地址掩码。
- `tests/privacy/test_masking_extended.py`：测试 HMAC-SM3 国密摘要一致性、FPE 保形加密可逆性、日期随机偏移与列洗牌。
- `tests/conformance/test_rest.py`：验证与 Go 引擎的跨语言逐字节一致性。

---

## 2. 自动化执行

```bash
.venv/bin/python -m pytest tests/privacy/test_masking*.py -v
```
