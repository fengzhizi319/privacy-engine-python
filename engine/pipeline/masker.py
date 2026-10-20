"""脱敏处理封装器 / Masking & Sanitization Wrapper.

封装 privacy/masking 原语与 L4/L5 降级逻辑，对字段进行脱敏处理。
L4/L5 敏感词库统一从 medical_pipeline.rules 导入，避免重复维护。
"""

from __future__ import annotations

from typing import Any

from ..privacy.masking import FieldType, mask_value
from .models import MaskingDetail, RecordClassificationDetail

# 从 medical_pipeline 统一导入 L4/L5 敏感词库（单一事实来源）
from ..medical_pipeline.rules import L4_PATTERNS, L5_PATTERNS


# 字段名到 FieldType 自动推断映射
FIELD_TYPE_MAP: dict[str, FieldType] = {
    "name": FieldType.NAME,
    "id_card_no": FieldType.ID_CARD,
    "registered_address": FieldType.ADDRESS,
    "disability_cert_no": FieldType.ID_CARD,
    "medical_insurance_no": FieldType.ID_CARD,
    "姓名": FieldType.NAME,
    "真实姓名": FieldType.NAME,
    "用户姓名": FieldType.NAME,
    "身份证": FieldType.ID_CARD,
    "身份证号": FieldType.ID_CARD,
    "居民身份证": FieldType.ID_CARD,
    "公民身份号码": FieldType.ID_CARD,
    "地址": FieldType.ADDRESS,
    "注册地址": FieldType.ADDRESS,
    "登记地址": FieldType.ADDRESS,
    "户籍地址": FieldType.ADDRESS,
    "居住地址": FieldType.ADDRESS,
    "居民住址": FieldType.ADDRESS,
    "家庭住址": FieldType.ADDRESS,
    "联系地址": FieldType.ADDRESS,
    "残疾证号": FieldType.ID_CARD,
    "残疾人证号": FieldType.ID_CARD,
    "医保卡号": FieldType.ID_CARD,
    "医保号": FieldType.ID_CARD,
    "医疗保险号": FieldType.ID_CARD,
}

# 需要强制进行 L4/L5 敏感词扫描的临床文本字段（不论分级结果如何）
CLINICAL_TEXT_FIELDS = frozenset({
    "diagnosis_name", "present_illness", "past_history",
    "progress_note", "family_history", "chief_complaint",
})


def mask_records(
    records: list[dict[str, Any]],
    record_details: list[RecordClassificationDetail],
    mask_l4: bool = True,
    mask_l5: bool = True,
) -> tuple[list[dict[str, Any]], list[MaskingDetail]]:
    """根据分级明细与策略对多条记录进行脱敏处理。

    Args:
        records: 原始记录列表。
        record_details: 分级明细列表。
        mask_l4: 是否对 L4 级数据进行脱敏/剥离。
        mask_l5: 是否对 L5 级数据进行脱敏/剥离。

    Returns:
        (masked_records, masking_details) 脱敏后记录列表与脱敏明细列表。
    """
    masked_records: list[dict[str, Any]] = []
    masking_details: list[MaskingDetail] = []

    # 建立 record_index -> RecordClassificationDetail 映射
    detail_map = {d.record_index: d for d in record_details}

    for idx, record in enumerate(records):
        rec_detail = detail_map.get(idx)
        masked_rec = dict(record)

        # 1. 对常规 PII 字段使用 privacy.masking 进行字段感知脱敏
        for fname, val in record.items():
            str_val = str(val) if val is not None else ""
            if not str_val:
                continue

            # 查寻找分类明细中对应的等级
            orig_level = "L1"
            if rec_detail:
                for fd in rec_detail.field_details:
                    if fd.field_name == fname:
                        orig_level = fd.sensitivity_level
                        break

            # PII 自动推断脱敏
            ftype = FIELD_TYPE_MAP.get(fname)
            if ftype:
                mval = mask_value(field_name=fname, value=str_val)
                if mval != str_val:
                    masked_rec[fname] = mval
                    masking_details.append(
                        MaskingDetail(
                            record_index=idx,
                            field_name=fname,
                            original_level=orig_level,
                            masking_type=ftype.value if hasattr(ftype, "value") else str(ftype),
                            original_value=str_val,
                            masked_value=mval,
                        )
                    )
                    str_val = mval  # 更新用于后续处理

            # 2. 对 L4/L5 级敏感数据进行高敏词汇强剥离
            #    使用统一词库（medical_pipeline.rules）中的 L5_PATTERNS + L4_PATTERNS
            if (orig_level in ("L4", "L5") and (mask_l4 or mask_l5)) or fname in CLINICAL_TEXT_FIELDS:
                cur_val = masked_rec[fname]
                new_val = cur_val
                # L5 优先替换（更高级别）
                if mask_l5:
                    for pattern, replacement in L5_PATTERNS:
                        if pattern.search(new_val):
                            new_val = pattern.sub(replacement, new_val)
                # L4 替换
                if mask_l4:
                    for pattern, replacement in L4_PATTERNS:
                        if pattern.search(new_val):
                            new_val = pattern.sub(replacement, new_val)

                if new_val != cur_val:
                    masked_rec[fname] = new_val
                    masking_details.append(
                        MaskingDetail(
                            record_index=idx,
                            field_name=fname,
                            original_level=orig_level,
                            masking_type="L4_L5_SANITY_STRIP",
                            original_value=cur_val,
                            masked_value=new_val,
                        )
                    )

        masked_records.append(masked_rec)

    return masked_records, masking_details
