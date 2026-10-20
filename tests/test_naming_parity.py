"""Cross-language SSOT parity test for PrivShield naming registry.
Verifies Python engine/naming.py conforms to the exact canonical entries in Go and TypeScript.
"""

from engine.naming import (
    API1_YIBAO,
    API2_KANGYANG,
    DS_KANGYANG,
    DS_MOCK3,
    DS_MOCK4,
    DS_YIBAO,
    REGISTRY,
    ReservedDataSourceError,
    UnknownDataSourceError,
    active_datasource_ids,
    api_code_for_datasource,
    normalize_datasource_id,
    resolve_inbound,
)
import pytest


def test_registry_parity():
    assert len(REGISTRY) == 4

    # 1. ds_yibao
    e1 = REGISTRY[0]
    assert e1.datasource_id == DS_YIBAO
    assert e1.api_code == API1_YIBAO
    assert e1.seq == 1
    assert e1.status == "active"
    assert e1.field_count == 18
    assert "yibao" in e1.aliases
    assert "医保" in e1.aliases

    # 2. ds_kangyang
    e2 = REGISTRY[1]
    assert e2.datasource_id == DS_KANGYANG
    assert e2.api_code == API2_KANGYANG
    assert e2.seq == 2
    assert e2.status == "active"
    assert e2.field_count == 27
    assert "kangyang" in e2.aliases
    assert "康养" in e2.aliases

    # 3. ds_mock3
    e3 = REGISTRY[2]
    assert e3.datasource_id == DS_MOCK3
    assert e3.seq == 3
    assert e3.status == "reserved"

    # 4. ds_mock4
    e4 = REGISTRY[3]
    assert e4.datasource_id == DS_MOCK4
    assert e4.seq == 4
    assert e4.status == "reserved"


def test_active_datasources():
    active = active_datasource_ids()
    assert DS_YIBAO in active
    assert DS_KANGYANG in active
    assert DS_MOCK3 not in active
    assert DS_MOCK4 not in active


def test_normalization():
    assert normalize_datasource_id("api1_yibao") == DS_YIBAO
    assert normalize_datasource_id("ds_yibao") == DS_YIBAO
    assert normalize_datasource_id("yibao") == DS_YIBAO
    assert normalize_datasource_id("医保") == DS_YIBAO

    assert normalize_datasource_id("api2_kangyang") == DS_KANGYANG
    assert normalize_datasource_id("ds_kangyang") == DS_KANGYANG
    assert normalize_datasource_id("kangyang") == DS_KANGYANG
    assert normalize_datasource_id("康养") == DS_KANGYANG


def test_fail_closed():
    with pytest.raises(UnknownDataSourceError):
        normalize_datasource_id("shebao")

    with pytest.raises(ReservedDataSourceError):
        resolve_inbound("mock3")

    with pytest.raises(ReservedDataSourceError):
        resolve_inbound("ds_mock4")


def test_api_code_for_datasource():
    assert api_code_for_datasource(DS_YIBAO) == API1_YIBAO
    assert api_code_for_datasource(DS_KANGYANG) == API2_KANGYANG
    assert api_code_for_datasource(DS_MOCK3) == ""
