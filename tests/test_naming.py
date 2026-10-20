import pytest
from engine.naming import (
    DS_YIBAO,
    DS_KANGYANG,
    DS_MOCK3,
    DS_MOCK4,
    API1_YIBAO,
    API2_KANGYANG,
    REGISTRY,
    UnknownDataSourceError,
    ReservedDataSourceError,
    normalize,
    normalize_datasource_id,
    resolve_inbound,
    api_code_for_datasource,
)

def test_canonical_normalization():
    assert normalize_datasource_id("ds_yibao") == DS_YIBAO
    assert normalize_datasource_id("api1_yibao") == DS_YIBAO
    assert normalize_datasource_id("医保") == DS_YIBAO
    assert normalize_datasource_id("医保数据") == DS_YIBAO
    assert normalize_datasource_id("yibao.csv") == DS_YIBAO
    assert normalize_datasource_id("medical.csv") == DS_YIBAO
    assert normalize_datasource_id("ds_kangyang") == DS_KANGYANG
    assert normalize_datasource_id("api2_kangyang") == DS_KANGYANG
    assert normalize_datasource_id("康养") == DS_KANGYANG
    assert normalize_datasource_id("healthcare.csv") == DS_KANGYANG

def test_unknown_datasource_fail_closed():
    with pytest.raises(UnknownDataSourceError):
        normalize("unknown_source")
    with pytest.raises(UnknownDataSourceError):
        normalize("")

def test_resolve_inbound_reserved():
    with pytest.raises(ReservedDataSourceError):
        resolve_inbound("ds_mock3")
    with pytest.raises(ReservedDataSourceError):
        resolve_inbound("mock4")

def test_api_code_for_datasource():
    assert api_code_for_datasource(DS_YIBAO) == API1_YIBAO
    assert api_code_for_datasource(DS_KANGYANG) == API2_KANGYANG
    assert api_code_for_datasource(DS_MOCK3) == ""
