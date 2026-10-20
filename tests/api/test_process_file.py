"""数据文件隐私处理端点（/v1/privacy/process_file）REST 测试。

使用 FastAPI TestClient 以 multipart 形式上传 CSV/JSON 文件，
验证 DataFrame 脱敏、K-匿名、整表分类三种操作均能正确解析并处理，
同时对非法操作类型与不支持的文件格式返回 400。
"""

import json

from fastapi.testclient import TestClient

from engine.main import app

# 复用同一个 TestClient 实例，避免重复创建应用
client = TestClient(app)

# 含 PII 字段的 CSV：用于脱敏测试
CSV_PII = (
    b"email,phone,name\n"
    b"alice@example.com,13800138000,Alice\n"
    b"bob@example.com,13900139000,Bob\n"
)

# 含准标识符与敏感列的 CSV：用于 K-匿名测试（disease 为敏感列，应保持不变）
CSV_KANO = (
    b"age,zip,gender,disease\n"
    b"30,100000,F,A\n"
    b"31,100001,F,B\n"
    b"32,100002,M,C\n"
    b"33,100003,M,D\n"
)


def test_process_file_mask_csv():
    """上传 CSV 执行 DataFrame 脱敏，应返回与输入等量的脱敏记录。"""
    resp = client.post(
        "/v1/privacy/process_file",
        files={"file": ("data.csv", CSV_PII, "text/csv")},
        data={
            "operation": "mask_dataframe",
            "params": json.dumps({"columns": ["email", "phone"]}),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["operation"] == "mask_dataframe"
    assert body["rows_in"] == 2
    assert body["rows_out"] == 2
    # email 列应被脱敏，不再等于原始明文
    assert body["result"][0]["email"] != "alice@example.com"


def test_process_file_k_anonymize_csv():
    """上传 CSV 执行 K-匿名，敏感列保持不变，记录数与输入一致。"""
    resp = client.post(
        "/v1/privacy/process_file",
        files={"file": ("data.csv", CSV_KANO, "text/csv")},
        data={
            "operation": "k_anonymize",
            "params": json.dumps({"qi_cols": ["age", "zip", "gender"], "k": 2}),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["operation"] == "k_anonymize"
    assert body["rows_in"] == 4
    assert body["rows_out"] == 4
    # 敏感列 disease 不在 qi_cols 中，应保持原值
    assert {r["disease"] for r in body["result"]} == {"A", "B", "C", "D"}


def test_process_file_unsupported_operation():
    """不支持的操作类型应返回 400。"""
    resp = client.post(
        "/v1/privacy/process_file",
        files={"file": ("data.csv", CSV_PII, "text/csv")},
        data={"operation": "foobar"},
    )
    assert resp.status_code == 400


def test_process_file_unsupported_format():
    """不支持的文件格式应返回 400。"""
    resp = client.post(
        "/v1/privacy/process_file",
        files={"file": ("data.txt", b"hello", "text/plain")},
        data={"operation": "mask_dataframe"},
    )
    assert resp.status_code == 400


def _make_xlsx_bytes(rows: list[list[str]], sheet_title: str = "Sheet1") -> bytes:
    """使用 openpyxl 在内存中构造 .xlsx 文件字节。"""
    from io import BytesIO

    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    if ws is None:
        ws = wb.create_sheet(sheet_title)
    ws.title = sheet_title
    for row in rows:
        ws.append(row)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_process_file_mask_xlsx():
    """上传 Excel 执行 DataFrame 脱敏，应返回与输入等量的脱敏记录。"""
    xlsx = _make_xlsx_bytes(
        [
            ["email", "phone", "name"],
            ["alice@example.com", "13800138000", "Alice"],
            ["bob@example.com", "13900139000", "Bob"],
        ]
    )
    resp = client.post(
        "/v1/privacy/process_file",
        files={"file": ("data.xlsx", xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={
            "operation": "mask_dataframe",
            "params": json.dumps({"columns": ["email", "phone"]}),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["operation"] == "mask_dataframe"
    assert body["rows_in"] == 2
    assert body["rows_out"] == 2
    assert body["result"][0]["email"] != "alice@example.com"


def test_process_file_k_anonymize_xlsx():
    """上传 Excel 执行 K-匿名，敏感列应保持不变。"""
    xlsx = _make_xlsx_bytes(
        [
            ["age", "zip", "gender", "disease"],
            ["30", "100000", "F", "A"],
            ["31", "100001", "F", "B"],
            ["32", "100002", "M", "C"],
            ["33", "100003", "M", "D"],
        ]
    )
    resp = client.post(
        "/v1/privacy/process_file",
        files={"file": ("data.xlsx", xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={
            "operation": "k_anonymize",
            "params": json.dumps({"qi_cols": ["age", "zip", "gender"], "k": 2}),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["operation"] == "k_anonymize"
    assert body["rows_in"] == 4
    assert body["rows_out"] == 4
    assert {r["disease"] for r in body["result"]} == {"A", "B", "C", "D"}
