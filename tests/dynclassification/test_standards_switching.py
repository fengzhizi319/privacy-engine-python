"""三标准（四川/金融/广东）加载与切换测试。

覆盖：
- 三个标准 sc_health_db51 / jrt0197 / gd_health 的注册与等级体系
  （L1~L5 / C1~C4 / G1~G4）及默认等级；
- list_standards_detail 返回结构（供前端标准切换器渲染）：
  description / taxonomy / domains / default_level / levels（rank 升序）；
- 同一字段跨标准切换的分类一致性（字段级 / 记录级 / 组合规则）；
- audit_info.standard_id 与请求标准一致；
- 同一服务实例交替切换标准时结果稳定（引擎缓存安全）；
- REST 端点 GET /v1/dynclassification/standards 与
  POST /eval、/eval_record 携带 standard 的集成测试。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.dynclassification import DynClassificationService
from engine.main import app

# 项目根目录与规则目录
ROOT_DIR = Path(__file__).resolve().parents[2]
RULES_DIR = ROOT_DIR / "rules"

# 三标准期望等级体系：standard_id -> (按 rank 升序的等级 ID 列表, 默认等级)
EXPECTED_TAXONOMIES: dict[str, tuple[list[str], str]] = {
    "sc_health_db51": (["L1", "L2", "L3", "L4", "L5"], "L3"),
    "jrt0197": (["C1", "C2", "C3", "C4"], "C3"),
    "gd_health": (["G1", "G2", "G3", "G4"], "G2"),
}

# 跨标准字段级分类期望：(字段名, 字段值, {标准: 期望等级})
CROSS_STANDARD_FIELD_CASES = [
    (
        "id_card",
        "110101199001011237",
        {"sc_health_db51": "L3", "jrt0197": "C3", "gd_health": "G4"},
    ),
    (
        "mobile_phone",
        "13800138000",
        {"sc_health_db51": "L3", "jrt0197": "C3", "gd_health": "G4"},
    ),
    (
        "bank_card",
        "6222021234567890123",
        {"sc_health_db51": "L3", "jrt0197": "C4", "gd_health": "G3"},
    ),
]


@pytest.fixture()
def service() -> DynClassificationService:
    """构造使用项目 rules/ 目录的服务实例。"""
    return DynClassificationService(rules_dir=RULES_DIR)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """复用同一个 TestClient 实例，避免重复创建应用。"""
    return TestClient(app)


# ===========================================================================
# 标准目录与等级体系
# ===========================================================================


@pytest.mark.skipif(not RULES_DIR.exists(), reason="rules/ 目录不存在，跳过集成测试")
class TestStandardsCatalog:
    """标准注册与等级体系测试。"""

    def test_three_standards_registered(self, service):
        """三个标准均应出现在可用标准列表中。"""
        standards = service.list_standards()
        for sid in EXPECTED_TAXONOMIES:
            assert sid in standards, f"标准 {sid} 未注册"

    @pytest.mark.parametrize("sid", sorted(EXPECTED_TAXONOMIES))
    def test_taxonomy_levels(self, service, sid):
        """各标准的等级 ID 列表与默认等级应符合预期。"""
        expected_levels, expected_default = EXPECTED_TAXONOMIES[sid]
        std_def = service.loader.load_standard(sid)
        taxonomy = service.loader.load_taxonomy(std_def.taxonomy)
        level_ids = [
            lv.id for lv in sorted(taxonomy.levels.values(), key=lambda x: x.rank)
        ]
        assert level_ids == expected_levels
        assert taxonomy.default_level == expected_default

    def test_list_standards_detail_structure(self, service):
        """标准详情应包含切换器渲染所需的全部字段。"""
        details = service.list_standards_detail()
        by_id = {d["standard_id"]: d for d in details}
        for sid, (expected_levels, expected_default) in EXPECTED_TAXONOMIES.items():
            assert sid in by_id, f"标准详情缺少 {sid}"
            detail = by_id[sid]
            # 描述、taxonomy 引用、领域包列表不为空
            assert detail["description"]
            assert detail["taxonomy"]
            assert isinstance(detail["domains"], list) and detail["domains"]
            # 默认等级与等级体系符合预期
            assert detail["default_level"] == expected_default
            ids = [lv["id"] for lv in detail["levels"]]
            assert ids == expected_levels
            # levels 按 rank 升序排列，且每个等级有名称
            ranks = [lv["rank"] for lv in detail["levels"]]
            assert ranks == sorted(ranks)
            assert all(lv["name"] for lv in detail["levels"])
            # 规则总数为非负整数，且已注册标准至少含一条规则
            assert isinstance(detail["rule_count"], int)
            assert detail["rule_count"] > 0
            # 分类总数为非负整数，且已注册标准的 taxonomy 至少含一个分类
            assert isinstance(detail["category_count"], int)
            assert detail["category_count"] > 0

    def test_details_sorted_by_standard_id(self, service):
        """详情列表按 standard_id 排序，保证前端渲染顺序稳定。"""
        details = service.list_standards_detail()
        ids = [d["standard_id"] for d in details]
        assert ids == sorted(ids)

    def test_gd_domains_reference(self, service):
        """广东标准应引用 gd_health taxonomy 与领域包。"""
        std_def = service.loader.load_standard("gd_health")
        assert std_def.taxonomy == "gd_health"
        assert "gd_health" in std_def.domains


# ===========================================================================
# 跨标准切换分类一致性
# ===========================================================================


@pytest.mark.skipif(not RULES_DIR.exists(), reason="rules/ 目录不存在，跳过集成测试")
class TestCrossStandardClassification:
    """同一输入在不同标准下应得到各自体系的等级。"""

    @pytest.mark.parametrize(
        "field_name,value,expected",
        CROSS_STANDARD_FIELD_CASES,
        ids=[c[0] for c in CROSS_STANDARD_FIELD_CASES],
    )
    def test_field_level_per_standard(self, service, field_name, value, expected):
        """字段级分类：三标准下等级各自独立正确。"""
        for sid, level in expected.items():
            resp = service.classify_field(field_name, value, standard=sid)
            assert resp.field_result is not None
            assert resp.field_result.final_level == level, (
                f"{field_name} 在 {sid} 下应为 {level}，"
                f"实际 {resp.field_result.final_level}"
            )

    @pytest.mark.parametrize("sid", sorted(EXPECTED_TAXONOMIES))
    def test_audit_standard_id(self, service, sid):
        """审计信息中的 standard_id 应与请求标准一致。"""
        resp = service.classify_field(
            "id_card", "110101199001011237", standard=sid
        )
        assert resp.audit_info.standard_id == sid

    def test_gd_medication_g2(self, service):
        """广东标准：用药信息为 G2（较低敏感）。"""
        resp = service.classify_field("medication", "阿司匹林 100mg", standard="gd_health")
        assert resp.field_result is not None
        assert resp.field_result.final_level == "G2"

    def test_gd_hospital_basic_g1(self, service):
        """广东标准：医院基本数据为 G1（低敏感）。"""
        resp = service.classify_field("hospital_name", "华西医院", standard="gd_health")
        assert resp.field_result is not None
        assert resp.field_result.final_level == "G1"

    def test_gd_composite_rule(self, service):
        """广东标准组合规则：身份+通讯+健康关联叠加升至 G4。"""
        record = {
            "idcard": "110101199001011237",
            "mobile": "13800138000",
            "病历": "高血压病史",
        }
        resp = service.classify_record(record, standard="gd_health")
        assert resp.record_result is not None
        assert resp.record_result.final_level == "G4"

    @pytest.mark.parametrize(
        "sid,expected",
        [
            ("sc_health_db51", "L3"),
            ("jrt0197", "C3"),
            ("gd_health", "G4"),
        ],
    )
    def test_record_level_per_standard(self, service, sid, expected):
        """记录级分类：同一记录在三标准下取各自体系的最高等级。"""
        record = {"idcard": "110101199001011237", "mobile": "13800138000"}
        resp = service.classify_record(record, standard=sid)
        assert resp.record_result is not None
        assert resp.record_result.final_level == expected

    def test_alternating_switch_stability(self, service):
        """同一实例交替切换标准，多轮结果应保持一致（引擎缓存安全）。"""
        sequence = [
            ("sc_health_db51", "L3"),
            ("gd_health", "G4"),
            ("jrt0197", "C3"),
        ]
        for _ in range(3):
            for sid, level in sequence:
                resp = service.classify_field(
                    "id_card", "110101199001011237", standard=sid
                )
                assert resp.field_result is not None
                assert resp.field_result.final_level == level
                assert resp.audit_info.standard_id == sid


# ===========================================================================
# REST 端点集成测试
# ===========================================================================


@pytest.mark.skipif(not RULES_DIR.exists(), reason="rules/ 目录不存在，跳过集成测试")
class TestStandardsREST:
    """标准切换相关 REST 端点集成测试。"""

    def test_get_standards_returns_details(self, client):
        """GET /standards 应同时返回 standards 与 details。"""
        resp = client.get("/v1/dynclassification/standards")
        assert resp.status_code == 200
        data = resp.json()
        assert {"standards", "details"} <= set(data.keys())
        for sid in EXPECTED_TAXONOMIES:
            assert sid in data["standards"]
        detail_ids = {d["standard_id"] for d in data["details"]}
        assert set(EXPECTED_TAXONOMIES) <= detail_ids

    @pytest.mark.parametrize(
        "sid,expected",
        [
            ("sc_health_db51", "L3"),
            ("jrt0197", "C3"),
            ("gd_health", "G4"),
        ],
    )
    def test_eval_with_standard(self, client, sid, expected):
        """POST /eval 携带 standard 时应返回对应体系的等级与审计信息。"""
        resp = client.post(
            "/v1/dynclassification/eval",
            json={
                "fieldName": "id_card",
                "value": "110101199001011237",
                "standard": sid,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["fieldResult"]["finalLevel"] == expected
        assert body["auditInfo"]["standardId"] == sid

    def test_eval_record_switch_back_and_forth(self, client):
        """同一会话内反复切换标准，记录级结果均应正确。"""
        record = {
            "idcard": "110101199001011237",
            "mobile": "13800138000",
            "病历": "高血压病史",
        }
        for sid, expected in [
            ("gd_health", "G4"),
            ("sc_health_db51", "L3"),
            ("jrt0197", "C3"),
            ("gd_health", "G4"),
        ]:
            resp = client.post(
                "/v1/dynclassification/eval_record",
                json={"record": record, "standard": sid},
            )
            assert resp.status_code == 200
            assert resp.json()["recordResult"]["finalLevel"] == expected
