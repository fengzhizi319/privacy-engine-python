# 数据脱敏（Masking）模块

> 代码：`engine/privacy/masking.py` · REST：`/v1/privacy/mask*`、`/v1/privacy/hash` · 测试：`tests/privacy/test_masking.py`

## 定位

字段名感知的脱敏与哈希原语。根据字段名推断敏感类型（手机号、身份证、姓名、银行卡、邮箱、地址、医疗文本），输出格式保留的掩码结果。所有函数为无状态纯函数，批量/DataFrame/流式接口只是同一原语的编排层。

## 代码地图

| 文件/符号 | 职责 |
|---|---|
| `FieldType` | 字段类型枚举：`mobile` / `id_card` / `name` / `bank_card` / `email` / `address` / `medical` / `default` |
| `MaskingOperation` | 操作枚举：`mask_value`、`mask_value_batch`、`mask_dataframe`、`mask_record`、`hash_value`、`truncate`、`chunked_mask_records`、`fpe_encrypt`、`random_date_offset`、`column_shuffle` |
| `guess_field_type()` | 按字段名边界匹配推断类型；`tel`、`name`、`card` 等短词使用边界匹配避免误命中 |
| `mask_mobile/id_card/name/bank_card/email/address/medical/default` | 各类型的掩码实现；`mobile` 保留前 3 后 4，`id_card` 前 6 后 4，银行卡前 4 后 4，地址保留前 6 字符，姓名保留首尾 |
| `mask_value()` | 单值入口；返回字符串或 `MaskingResult` |
| `mask_value_batch()` | 等长字段名/值列表逐元素脱敏；整批只计一次指标 |
| `mask_record()` | 对 dict 中所有 `str` 值脱敏；支持 dict / Arrow IPC bytes / Arrow Table / ndarray / pandas Series 输入 |
| `mask_dataframe()` | pandas / PyArrow / Polars 等表格输入；存在 PyArrow 时走列级向量化路径 |
| `chunked_mask_records()` | 生成器式流式脱敏，控制大记录集内存占用 |
| `hash_value()` | HMAC-SM3（默认）或 HMAC-SHA-256（显式兼容开关），base64 后截取前 16 字符 |
| `truncate()` | 保留前缀并追加 `***` |
| `fpe_encrypt_numeric()` | 数字串保形加密（HMAC-SM3 计数器密钥流 + 逐位模映射），非 NIST FF1/FF3 |
| `random_date_offset()` / `shuffle_column()` | 日期统一随机偏移 / 列洗牌 |

## REST 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/privacy/mask` | 单值脱敏 |
| POST | `/v1/privacy/mask/record`、`/v1/privacy/mask_record` | 整记录脱敏（兼容别名） |
| POST | `/v1/privacy/mask/batch`、`/v1/privacy/mask_batch` | 批量字段脱敏 |
| POST | `/v1/privacy/mask/dataframe`、`/v1/privacy/mask_dataframe` | 表格脱敏 |
| POST | `/v1/privacy/hash` | HMAC 哈希 |
| POST | `/v1/privacy/process_file` | 文件级脱敏（CSV/Excel/JSON，见顶层 [api.md](../api.md)） |

接口所需的 `mask:read` / `mask:write` 等 scope 见 [auth.md](../auth.md)。

## 行为要点

- **HMAC 密钥而非公开盐**：`hash_value(value, salt)` 的 `salt` 是 HMAC 保密密钥。输出为 16 字符 base64（96 bit），超大规模去重需自行评估碰撞概率。
- **算法默认值与 Go 对齐**：默认 HMAC-SM3，输出与 Go `masking.HashHMAC` 逐字节一致；设置 `PRIVACY_HASH_LEGACY_SHA256=1` 才回退历史 SHA-256。
- **FPE 不是标准 FF1/FF3**：`fpe_encrypt_numeric` 使用 HMAC-SM3 计数器密钥流对每个数字位做模 10 映射，确定性且与 Go `FpeEncryptNumeric` 对齐；空密钥返回空串，调用方必须保证密钥管理。
- **medical 类型不做词库抹平**：`mask_value` 对 `medical` 仅做通用前 3 后 3 掩码。L4/L5 病史词剥离属于 `medical_pipeline`，不是本模块能力。
- **无 ASCII fast-path**：纯 Python 的逐字符掩码，在长文本/大表上比 Go 慢；PyArrow 列表路径可缓解表格场景，但 `mask_value_batch` 仍是逐元素调用。
- **输入校验 fail-fast**：空字段名、非字符串值、空盐、字段名/值长度不一致等直接抛 `ValueError`，不会静默截断。

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRIVACY_HASH_LEGACY_SHA256` | 未设置 | `1/true/yes/on` 时 HMAC 回退 SHA-256；用于历史兼容，不应用于新部署 |

## 测试

```bash
pytest tests/privacy/test_masking.py tests/privacy/test_privacy_adapters.py -q
make conformance   # Go/Python 共享夹具逐字节对齐（本仓库）
```

## 与 Go 的差异与边界

| 项 | Go 引擎 | Python 引擎 |
|---|---|---|
| ASCII fast-path | 有 | 无 |
| 批次内部并行 | 分块多核 | 逐元素循环（表格可走 PyArrow 列路径） |
| 国密算法 | SM3/SM4 | HMAC-SM3（哈希）；无 SM4 |
| 文件格式 | CSV/Excel/JSON/DICOM | CSV/Excel/JSON/JSONL/Parquet/SQLite + 图像脱敏在 `dynclassification` |

相关文档：[api.md](../api.md) · [architecture.md](../architecture.md) · [算法优化.md](../算法优化.md) · [testing.md](../testing.md)
