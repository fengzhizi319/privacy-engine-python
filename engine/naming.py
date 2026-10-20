"""Single source of truth for canonical data source IDs and API codes in PrivShield engine.
数盾 Python 引擎数据源与 API 编码权威注册表。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from engine.observability.metrics import (
    API_ALIAS_REQUESTS_TOTAL,
    DATASOURCE_NORMALIZE_ERRORS_TOTAL,
)

_MODULE = "privshield-engine"

DS_YIBAO = "ds_yibao"
DS_KANGYANG = "ds_kangyang"
DS_MOCK3 = "ds_mock3"
DS_MOCK4 = "ds_mock4"

API1_YIBAO = "api1_yibao"
API2_KANGYANG = "api2_kangyang"

STATUS_ACTIVE = "active"
STATUS_RESERVED = "reserved"

_DATASOURCE_ID_RE = re.compile(r"^ds_[a-z][a-z0-9_]{1,30}$")
_API_CODE_RE = re.compile(r"^api[1-9]_[a-z][a-z0-9_]{1,30}$")


class UnknownDataSourceError(ValueError):
    """Raised when an inbound datasource value cannot be resolved."""


class ReservedDataSourceError(PermissionError):
    """Raised when an operation targets a registered but reserved (unimplemented) datasource."""


@dataclass(frozen=True)
class Entry:
    datasource_id: str
    api_code: str = ""
    seq: int = 0
    display_name: dict[str, str] = field(default_factory=dict)
    category: str = ""
    file_name: str = ""
    field_count: int = 0
    aliases: list[str] = field(default_factory=list)
    status: str = STATUS_ACTIVE


REGISTRY: list[Entry] = [
    Entry(
        api_code=API1_YIBAO,
        datasource_id=DS_YIBAO,
        seq=1,
        display_name={"zh-CN": "医保结算数据接口", "en-US": "Medical Insurance Settlement API"},
        category="medical",
        file_name="yibao.csv",
        field_count=18,
        aliases=["yibao", "yibao.csv", "medical.csv", "医保", "医保数据", "医保数据库", "医保结算", "medical", "medical_insurance"],
        status=STATUS_ACTIVE,
    ),
    Entry(
        api_code=API2_KANGYANG,
        datasource_id=DS_KANGYANG,
        seq=2,
        display_name={"zh-CN": "康养健康档案接口", "en-US": "Elderly-Care Health Record API"},
        category="healthcare",
        file_name="kangyang.csv",
        field_count=27,
        aliases=["kangyang", "kangyang.csv", "healthcare.csv", "康养", "康养数据", "康养数据库", "康养体检", "healthcare", "elderly_care"],
        status=STATUS_ACTIVE,
    ),
    Entry(
        datasource_id=DS_MOCK3,
        seq=3,
        display_name={"zh-CN": "预留政务数据源 3", "en-US": "Reserved Municipal Dataset 3"},
        category="reserved",
        file_name="mock3.csv",
        field_count=0,
        aliases=["mock3", "mock3.csv", "政务", "政务数据", "政务数据源"],
        status=STATUS_RESERVED,
    ),
    Entry(
        datasource_id=DS_MOCK4,
        seq=4,
        display_name={"zh-CN": "预留企业/金融数据源 4", "en-US": "Reserved Enterprise Dataset 4"},
        category="reserved",
        file_name="mock4.csv",
        field_count=0,
        aliases=["mock4", "mock4.csv", "企业", "金融", "企业数据", "金融数据"],
        status=STATUS_RESERVED,
    ),
]

_BY_DATASOURCE_ID: dict[str, Entry] = {e.datasource_id: e for e in REGISTRY}
_BY_API_CODE: dict[str, Entry] = {e.api_code: e for e in REGISTRY if e.api_code}
_ALIAS_INDEX: dict[str, Entry] = {}

for _entry in REGISTRY:
    for _a in _entry.aliases:
        _ALIAS_INDEX[_a.lower()] = _entry
        _ALIAS_INDEX[_a] = _entry


def active_datasource_ids() -> list[str]:
    return [e.datasource_id for e in REGISTRY if e.status == STATUS_ACTIVE]


def _record_alias(alias: str, canonical: str, target: str) -> None:
    API_ALIAS_REQUESTS_TOTAL.labels(
        alias=alias, canonical=canonical, target=target, module=_MODULE
    ).inc()


def _record_normalize_error(reason: str) -> None:
    DATASOURCE_NORMALIZE_ERRORS_TOTAL.labels(reason=reason, module=_MODULE).inc()


def normalize(raw: str) -> Entry:
    """Mirror Go ``inbound.Normalize`` (canonical → api_code → alias → fail-closed)."""
    v = (raw or "").strip()
    if not v:
        _record_normalize_error("empty")
        allowed = ", ".join(active_datasource_ids())
        raise UnknownDataSourceError(f"unknown datasource id: {raw!r} (allowed: {allowed})")
    if v in _BY_DATASOURCE_ID:
        return _BY_DATASOURCE_ID[v]
    if v in _BY_API_CODE:
        entry = _BY_API_CODE[v]
        _record_alias(v, entry.datasource_id, "api_code")
        return entry
    lowered = v.lower()
    if lowered in _ALIAS_INDEX:
        entry = _ALIAS_INDEX[lowered]
        _record_alias(v, entry.datasource_id, "datasource_id")
        return entry
    if v in _ALIAS_INDEX:
        entry = _ALIAS_INDEX[v]
        _record_alias(v, entry.datasource_id, "datasource_id")
        return entry

    _record_normalize_error("unknown")
    allowed = ", ".join(active_datasource_ids())
    raise UnknownDataSourceError(f"unknown datasource id: {raw!r} (allowed: {allowed})")


def normalize_datasource_id(raw: str) -> str:
    return normalize(raw).datasource_id


def resolve_inbound(raw: str) -> str:
    entry = normalize(raw)
    if entry.status != STATUS_ACTIVE:
        _record_normalize_error("reserved")
        raise ReservedDataSourceError(f"reserved datasource: {entry.datasource_id}")
    return entry.datasource_id


def api_code_for_datasource(datasource_id: str) -> str:
    entry = _BY_DATASOURCE_ID.get(datasource_id)
    return entry.api_code if entry else ""
