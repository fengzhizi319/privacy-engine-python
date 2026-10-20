#!/usr/bin/env python3
"""医保结算数据生成脚本 / Medical Insurance Data Generator (yibao.csv).

生成 50 条包含真实医保结算字段、ICD-10 诊断编码、诊断名称（含 L1-L5 分级诊疗）、
人员唯一标识、日期及院所编码的高仿真医保数据集。

Usage:
    python scripts/data/generate_yibao_data.py
    python scripts/data/generate_yibao_data.py --output engine/medical_pipeline/samples/yibao.csv --count 50
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path

# 18 个标准医保结算字段名 (英文 Key)
YIBAO_FIELDS = [
    "insurance_settlement_id",  # 1. 医保结算流水号
    "person_id",                # 2. 人员唯一标识（脱敏）
    "gender",                   # 3. 性别
    "birth_date",               # 4. 出生日期
    "admission_date",           # 5. 入院日期
    "discharge_date",           # 6. 出院日期
    "length_of_stay",           # 7. 住院天数
    "admission_dept",           # 8. 入院科室
    "discharge_dept",           # 9. 出院科室
    "hospital_code",            # 10. 定点医疗机构编码
    "medical_category",         # 11. 医疗类别
    "discharge_mode",           # 12. 离院方式
    "settlement_seq_no",        # 13. 明细结算流水号
    "diagnosis_seq",            # 14. 诊断序号
    "diagnosis_type",           # 15. 诊断类别
    "icd10_code",               # 16. 诊断编码(ICD-10)
    "diagnosis_name",           # 17. 诊断名称
    "admission_condition",      # 18. 入院病情
]

DEPARTMENTS = [
    "心血管内科", "呼吸与危重症医学科", "消化内科", "神经内科", "肿瘤科",
    "感染性疾病科", "精神心理科", "普通外科", "骨科", "妇产科",
    "皮肤性病科", "血液内科", "肾内科", "内分泌科", "急诊科"
]

MEDICAL_CATEGORIES = ["住院", "门诊慢特病", "普通门诊", "日间手术"]
DISCHARGE_MODES = ["医嘱离院", "医嘱转院", "非医嘱离院", "死亡"]
ADMISSION_CONDITIONS = ["一般", "急", "危", "重"]
HOSPITAL_CODES = [
    "H1101010001", "H1101080002", "H3101010005", "H4401030012",
    "H5101040008", "H3301020003", "H6101030009", "H4201020015"
]

# 诊疗诊断库 (含 ICD-10 编码、诊断名称与风险等级)
DIAGNOSIS_POOL = [
    # --- L1/L2 常规慢性病与普通诊疗 ---
    {"code": "I10.x00", "name": "原发性高血压病史10年", "level": "L1"},
    {"code": "E11.900", "name": "2型糖尿病", "level": "L1"},
    {"code": "K35.800", "name": "急性阑尾炎", "level": "L1"},
    {"code": "K81.100", "name": "慢性胆囊炎伴胆石症", "level": "L1"},
    {"code": "J18.900", "name": "社区获得性肺炎", "level": "L1"},
    {"code": "M17.900", "name": "双侧膝关节退行性骨关节炎", "level": "L1"},
    {"code": "K29.500", "name": "慢性浅表性胃炎", "level": "L1"},
    {"code": "I25.100", "name": "冠状动脉粥样硬化性心脏病", "level": "L2"},
    {"code": "E78.500", "name": "高脂血症", "level": "L1"},

    # --- L4 重大高敏诊疗 (恶性肿瘤、肝炎、严重心梗) ---
    {"code": "C34.900", "name": "原发性左肺上叶腺癌(T2N1M0)", "level": "L4"},
    {"code": "C16.900", "name": "胃体恶性肿瘤(进展期腺癌)", "level": "L4"},
    {"code": "B18.100", "name": "慢性乙型病毒性肝炎，建议启动恩替卡韦抗病毒治疗", "level": "L4"},
    {"code": "I21.900", "name": "急性前壁心肌梗死", "level": "L4"},
    {"code": "C50.900", "name": "右侧乳腺浸润性导管癌(III期)", "level": "L4"},
    {"code": "C18.900", "name": "升结肠恶性肿瘤", "level": "L4"},

    # --- L5 极高敏诊疗 (性病、HIV/AIDS、重度精神障碍、罕见遗传病) ---
    {"code": "B20.900", "name": "确诊HIV抗体阳性，CD4+细胞180/μL，行ART抗逆转录治疗", "level": "L5"},
    {"code": "A63.000", "name": "外阴多发菜花状赘生物，醋酸白试验阳性(尖锐湿疣)", "level": "L5"},
    {"code": "A51.000", "name": "硬下疳伴TPPA滴度1:64阳性(早期梅毒)", "level": "L5"},
    {"code": "F20.900", "name": "重度精神分裂症，存在偏执幻觉与命令性幻听", "level": "L5"},
    {"code": "G10.x00", "name": "基因检测提示亨廷顿舞蹈病(HTT基因CAG重复46次)", "level": "L5"},
]


def generate_yibao_record(idx: int) -> dict[str, str]:
    """生成单条高仿真医保结算记录。"""
    gender = random.choice(["男", "女"])
    age = random.randint(18, 85)
    birth_year = 2026 - age
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
    birth_date = f"{birth_year:04d}-{birth_month:02d}-{birth_day:02d}"

    # 入院与出院日期
    admission_dt = date(2025, random.randint(1, 12), random.randint(1, 20))
    stay_days = random.randint(2, 28)
    discharge_dt = admission_dt + timedelta(days=stay_days)

    admission_dept = random.choice(DEPARTMENTS)
    discharge_dept = admission_dept if random.random() > 0.15 else random.choice(DEPARTMENTS)

    diag = random.choice(DIAGNOSIS_POOL)
    
    # 医保流水号与人员 PID 格式
    seq_num = f"{idx + 1:04d}"
    insurance_settlement_id = f"YB2025{admission_dt.strftime('%m%d')}{seq_num}"
    person_id = f"PID{random.randint(10000000, 99999999)}"
    settlement_seq_no = f"MX{admission_dt.strftime('%Y%m%d')}{random.randint(1000, 9999)}"
    hospital_code = random.choice(HOSPITAL_CODES)

    return {
        "insurance_settlement_id": insurance_settlement_id,
        "person_id": person_id,
        "gender": gender,
        "birth_date": birth_date,
        "admission_date": admission_dt.strftime("%Y-%m-%d"),
        "discharge_date": discharge_dt.strftime("%Y-%m-%d"),
        "length_of_stay": str(stay_days),
        "admission_dept": admission_dept,
        "discharge_dept": discharge_dept,
        "hospital_code": hospital_code,
        "medical_category": random.choice(MEDICAL_CATEGORIES),
        "discharge_mode": random.choice(DISCHARGE_MODES),
        "settlement_seq_no": settlement_seq_no,
        "diagnosis_seq": "1",
        "diagnosis_type": "主要诊断",
        "icd10_code": diag["code"],
        "diagnosis_name": diag["name"],
        "admission_condition": random.choice(ADMISSION_CONDITIONS),
    }


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="生成医保结算数据 yibao.csv")
    parser.add_argument(
        "--output",
        default=str(project_root / "engine/medical_pipeline/samples/yibao.csv"),
        help="输出 CSV 文件路径",
    )
    parser.add_argument("--count", type=int, default=50, help="生成的记录条数 (默认 50)")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [generate_yibao_record(i) for i in range(args.count)]

    # 写入带 UTF-8 BOM 的 CSV
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=YIBAO_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ [成功] 顺利生成 {len(records)} 条医保结算仿真数据 -> {output_path}")

    console_go_path = project_root / "console/bff-go/internal/samples/yibao.csv"
    if console_go_path.parent.exists():
        console_go_path.write_bytes(output_path.read_bytes())
        print(f"✅ [副本] 成功将 yibao.csv 复制到 -> {console_go_path}")


if __name__ == "__main__":
    main()
