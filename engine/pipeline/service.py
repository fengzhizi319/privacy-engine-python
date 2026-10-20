"""流水线服务编排器 / Pipeline Service Orchestrator.

串联 DynClassificationService 分类分级与 privacy/masking 脱敏，输出 PipelineResult。
支持 CSV、JSON、JSONL、Excel(XLSX/XLS)、Parquet、SQLite 等多种数据源格式与数据库导入导出。
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from .classifier import classify_records
from .masker import mask_records
from .models import (
    ClassificationSummary,
    PipelineResult,
    RecordClassificationDetail,
)


class PipelineService:
    """医疗数据分类分级 + 脱敏流水线编排器。"""

    def __init__(
        self,
        rules_dir: str | Path | None = None,
        standard: str = "jrt0197",
        profile_path: str | Path | None = None,
    ) -> None:
        self.rules_dir = rules_dir
        self.standard = standard
        self.profile_path = profile_path

    def process_records(
        self,
        records: list[dict[str, Any]],
        *,
        standard: str | None = None,
        mask_l4: bool = True,
        mask_l5: bool = True,
    ) -> PipelineResult:
        """对字典记录列表执行分类分级与脱敏流水线。

        Args:
            records: 输入记录字典列表。
            standard: 使用的分类标准（默认使用初始化标准）。
            mask_l4: 是否掩码/剥离 L4 级数据。
            mask_l5: 是否掩码/剥离 L5 级数据。

        Returns:
            PipelineResult 包含分类汇总、明细与脱敏后记录。
        """
        start_t = time.perf_counter()
        std = standard or self.standard

        if not records:
            return PipelineResult(
                classification_summary=ClassificationSummary(
                    total_records=0,
                    level_distribution={"L1": 0, "L2": 0, "L3": 0, "L4": 0, "L5": 0},
                    high_risk_fields=[],
                    standard_id=std,
                    duration_ms=0.0,
                ),
                record_details=[],
                masked_records=[],
                masking_details=[],
            )

        # 1. 逐条执行 DynClassification 分类分级
        record_details: list[RecordClassificationDetail] = classify_records(
            records=records,
            standard=std,
            rules_dir=self.rules_dir,
        )

        # 2. 统计级别分布与 L4/L5 高风险字段
        level_dist: dict[str, int] = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "L5": 0}
        high_risk_set: set[str] = set()

        for rd in record_details:
            lvl = rd.final_level
            level_dist[lvl] = level_dist.get(lvl, 0) + 1
            for fd in rd.field_details:
                if fd.sensitivity_level in ("L4", "L5"):
                    high_risk_set.add(fd.field_name)

        # 3. 对记录与字段执行脱敏抹平
        masked_recs, mask_details = mask_records(
            records=records,
            record_details=record_details,
            mask_l4=mask_l4,
            mask_l5=mask_l5,
        )

        duration_ms = round((time.perf_counter() - start_t) * 1000, 2)

        summary = ClassificationSummary(
            total_records=len(records),
            level_distribution=level_dist,
            high_risk_fields=sorted(list(high_risk_set)),
            standard_id=std,
            duration_ms=duration_ms,
        )

        return PipelineResult(
            classification_summary=summary,
            record_details=record_details,
            masked_records=masked_recs,
            masking_details=mask_details,
        )

    def process_single_record(
        self,
        record: dict[str, Any],
        *,
        standard: str | None = None,
        mask_l4: bool = True,
        mask_l5: bool = True,
    ) -> PipelineResult:
        """处理单条病例记录字典。"""
        return self.process_records(
            records=[record],
            standard=standard,
            mask_l4=mask_l4,
            mask_l5=mask_l5,
        )

    def save_masked_file(
        self,
        masked_records: list[dict[str, Any]],
        output_path: str | Path,
        *,
        encoding: str = "utf-8-sig",
        table_name: str = "masked_data",
    ) -> Path:
        """将脱敏记录保存到本地文件或数据库（支持 .csv, .json, .jsonl, .xlsx, .parquet, .db/.sqlite）。"""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ext = out_p.suffix.lower()

        if not masked_records:
            if ext == ".csv":
                out_p.write_text("", encoding=encoding)
            return out_p

        if ext == ".csv":
            fieldnames = list(masked_records[0].keys())
            with open(out_p, "w", newline="", encoding=encoding) as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(masked_records)
        elif ext == ".json":
            out_p.write_text(json.dumps(masked_records, ensure_ascii=False, indent=2), encoding="utf-8")
        elif ext == ".jsonl":
            lines = [json.dumps(r, ensure_ascii=False) for r in masked_records]
            out_p.write_text("\n".join(lines), encoding="utf-8")
        elif ext in (".xlsx", ".xls", ".parquet"):
            import pandas as pd
            df = pd.DataFrame(masked_records)
            if ext == ".parquet":
                df.to_parquet(out_p, index=False)
            else:
                df.to_excel(out_p, index=False)
        elif ext in (".db", ".sqlite", ".sqlite3"):
            import pandas as pd
            df = pd.DataFrame(masked_records)
            conn = sqlite3.connect(out_p)
            try:
                df.to_sql(table_name, conn, if_exists="replace", index=False)
            finally:
                conn.close()
        else:
            # 默认作为 CSV 写入
            fieldnames = list(masked_records[0].keys())
            with open(out_p, "w", newline="", encoding=encoding) as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(masked_records)

        return out_p

    def process_csv(
        self,
        csv_path: str | Path,
        *,
        output_path: str | Path | None = None,
        standard: str | None = None,
        mask_l4: bool = True,
        mask_l5: bool = True,
        encoding: str = "utf-8-sig",
    ) -> PipelineResult:
        """从 CSV 文件读取数据并执行流水线，可保存到 output_path。"""
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        records: list[dict[str, Any]] = []
        with open(path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))

        res = self.process_records(
            records=records,
            standard=standard,
            mask_l4=mask_l4,
            mask_l5=mask_l5,
        )

        if output_path:
            self.save_masked_file(res.masked_records, output_path, encoding=encoding)

        return res

    def process_json(
        self,
        json_path: str | Path,
        *,
        output_path: str | Path | None = None,
        standard: str | None = None,
        mask_l4: bool = True,
        mask_l5: bool = True,
    ) -> PipelineResult:
        """处理 JSON 或 JSONL 文件。"""
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")

        content = path.read_text(encoding="utf-8").strip()
        records: list[dict[str, Any]] = []
        if path.suffix.lower() == ".jsonl":
            for line in content.splitlines():
                if line.strip():
                    records.append(json.loads(line.strip()))
        else:
            data = json.loads(content)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]

        res = self.process_records(records=records, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5)
        if output_path:
            self.save_masked_file(res.masked_records, output_path)
        return res

    def process_excel(
        self,
        excel_path: str | Path,
        *,
        sheet_name: str | int = 0,
        output_path: str | Path | None = None,
        standard: str | None = None,
        mask_l4: bool = True,
        mask_l5: bool = True,
    ) -> PipelineResult:
        """处理 Excel (.xlsx, .xls) 文件。"""
        import pandas as pd
        path = Path(excel_path)
        if not path.is_file():
            raise FileNotFoundError(f"Excel file not found: {path}")

        df = pd.read_excel(path, sheet_name=sheet_name)
        records = df.fillna("").to_dict(orient="records")
        res = self.process_records(records=records, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5)
        if output_path:
            self.save_masked_file(res.masked_records, output_path)
        return res

    def process_parquet(
        self,
        parquet_path: str | Path,
        *,
        output_path: str | Path | None = None,
        standard: str | None = None,
        mask_l4: bool = True,
        mask_l5: bool = True,
    ) -> PipelineResult:
        """处理 Parquet (.parquet) 文件。"""
        import pandas as pd
        path = Path(parquet_path)
        if not path.is_file():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        df = pd.read_parquet(path)
        records = df.fillna("").to_dict(orient="records")
        res = self.process_records(records=records, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5)
        if output_path:
            self.save_masked_file(res.masked_records, output_path)
        return res

    def process_sqlite(
        self,
        db_path: str | Path,
        query_or_table: str = "SELECT * FROM data",
        *,
        output_path: str | Path | None = None,
        output_table: str | None = None,
        standard: str | None = None,
        mask_l4: bool = True,
        mask_l5: bool = True,
    ) -> PipelineResult:
        """处理 SQLite (.db, .sqlite) 数据库查询或表数据。"""
        path = Path(db_path)
        if not path.is_file():
            raise FileNotFoundError(f"SQLite DB not found: {path}")

        if "SELECT" in query_or_table.upper():
            # 调用方显式传入完整 SQL（本地工具特性，原样执行） / Caller-provided full SQL passthrough
            sql = query_or_table  # noqa: S608
        else:
            # 表名拼接进 SQL：强制标识符校验，防 SQL 注入 / Validate identifier to prevent SQL injection
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", query_or_table):
                raise ValueError(f"非法表名（仅允许字母/数字/下划线）: {query_or_table!r}")
            sql = f"SELECT * FROM {query_or_table}"  # noqa: S608 —— 表名已经上方标识符校验
        conn = sqlite3.connect(path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            records = [dict(r) for r in rows]
        finally:
            conn.close()

        res = self.process_records(records=records, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5)
        if output_path:
            self.save_masked_file(res.masked_records, output_path, table_name=output_table or "masked_data")
        elif output_table:
            import pandas as pd
            df = pd.DataFrame(res.masked_records)
            c = sqlite3.connect(path)
            try:
                df.to_sql(output_table, c, if_exists="replace", index=False)
            finally:
                c.close()

        return res

    def process_file(
        self,
        file_path: str | Path,
        *,
        output_path: str | Path | None = None,
        standard: str | None = None,
        mask_l4: bool = True,
        mask_l5: bool = True,
        encoding: str = "utf-8-sig",
    ) -> PipelineResult:
        """多态通用文件入口：根据文件后缀自动调度对应加载器。

        支持后缀:
        - CSV: .csv
        - JSON / JSONL: .json, .jsonl
        - Excel: .xlsx, .xls
        - Parquet: .parquet
        - SQLite 数据库: .db, .sqlite, .sqlite3
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".csv":
            return self.process_csv(path, output_path=output_path, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5, encoding=encoding)
        elif ext in (".json", ".jsonl"):
            return self.process_json(path, output_path=output_path, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5)
        elif ext in (".xlsx", ".xls"):
            return self.process_excel(path, output_path=output_path, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5)
        elif ext == ".parquet":
            return self.process_parquet(path, output_path=output_path, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5)
        elif ext in (".db", ".sqlite", ".sqlite3"):
            return self.process_sqlite(path, output_path=output_path, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5)
        else:
            return self.process_csv(path, output_path=output_path, standard=standard, mask_l4=mask_l4, mask_l5=mask_l5, encoding=encoding)
