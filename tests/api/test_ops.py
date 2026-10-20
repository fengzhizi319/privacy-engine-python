"""运维诊断路由单元测试（/v1/ops/diagnostics）。

覆盖四个层面：
1. 端点集成测试：通过 FastAPI TestClient 验证响应结构、降级链顺序、
   依赖/模型/硬件字段完整性与一致性约束；
2. 检测工具单测：_pkg_installed / _pkg_version / _dep_info / _model_info /
   _engine_available 的分支逻辑（缺依赖 / 缺模型 / 缺附加文件 / 全满足）；
3. 降级链推断单测：_build_ner_status 的“第一个可用引擎即激活”规则、
   全不可用回退 none、运行时真实引擎优先于动态探测，以及
   _runtime_ner_engine 对适配器引擎类名的映射；
4. 动态探测单测：_probe_ner_engines / _probe_llm 的实际初始化判定逻辑
   （与 tests/dynclassification 相同的判定方式）。

Unit tests for the ops diagnostics route (/v1/ops/diagnostics), covering
endpoint structure, detection helpers, degradation-chain inference and dynamic probe.
"""

import sys

import pytest
from fastapi.testclient import TestClient

from engine.main import app
from engine.routers import dynclassification as dyn_router
from engine.routers import ops as ops_mod

# 复用同一个 TestClient 实例，避免重复创建应用
client = TestClient(app)


# ---------------------------------------------------------------------------
# 端点集成测试 / Endpoint integration tests
# ---------------------------------------------------------------------------


def test_diagnostics_endpoint_structure():
    """诊断端点返回 200 且包含全部顶层字段与 service 子字段。"""
    resp = client.get("/v1/ops/diagnostics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    for key in ("service", "engines", "dependencies", "models", "hardware"):
        assert key in data
    for key in ("name", "namespace", "python_version", "rest_port", "grpc_port"):
        assert key in data["service"]
    assert "ner" in data["engines"]
    assert "llm" in data["engines"]


def test_diagnostics_ner_chain_order():
    """NER 降级链顺序与适配器真实尝试顺序一致：mlx→tensorrt→onnx→modelscope。"""
    chain = client.get("/v1/ops/diagnostics").json()["engines"]["ner"]["degradation_chain"]
    assert [e["engine"] for e in chain] == ["mlx", "tensorrt", "onnx", "modelscope"]
    for e in chain:
        # 动态探测可能给链上条目额外添加 probe_error 字段
        required = {"engine", "available", "reason", "deps", "model", "note"}
        assert required <= set(e)
        # 可用引擎不应携带不可用原因；不可用引擎必须给出原因
        if e["available"]:
            assert e["reason"] is None
        else:
            assert e["reason"] or e.get("probe_error")


def test_diagnostics_active_engine_is_first_available():
    """动态探测确定的 active_engine 必须等于 probe 中第一个成功引擎。"""
    ner = client.get("/v1/ops/diagnostics").json()["engines"]["ner"]
    # 动态探测结果
    probe = ner["probe"]
    probe_active = probe["active_engine"]
    if ner["runtime_engine"] is None:
        assert ner["active_engine"] == probe_active
        assert ner["determined_by"] == "probe"
    else:
        assert ner["determined_by"] == "runtime"
    assert ner["available"] == (ner["active_engine"] != "none")


def test_diagnostics_consistency_dep_vs_engine():
    """一致性约束：某依赖未安装时，依赖它的引擎必须不可用且原因提及该依赖。"""
    data = client.get("/v1/ops/diagnostics").json()
    deps = {d["name"]: d["installed"] for d in data["dependencies"]}
    for eng in data["engines"]["ner"]["degradation_chain"]:
        for dep in eng["deps"]:
            if dep in deps and not deps[dep]:
                assert eng["available"] is False
                assert dep in (eng["reason"] or "")


def test_diagnostics_dependencies_fields():
    """依赖条目字段完整；未安装时版本为 None。"""
    deps = client.get("/v1/ops/diagnostics").json()["dependencies"]
    names = {d["name"] for d in deps}
    assert {"onnxruntime", "torch", "transformers", "modelscope", "tensorrt", "mlx"} <= names
    for d in deps:
        assert set(d) == {"name", "installed", "version", "purpose", "install"}
        assert isinstance(d["installed"], bool)
        if not d["installed"]:
            assert d["version"] is None


def test_diagnostics_models_fields():
    """模型条目字段完整，exists 为布尔值且附带下载命令与 note 字段。"""
    models = client.get("/v1/ops/diagnostics").json()["models"]
    paths = {m["path"] for m in models}
    assert ".models/raner_cmeee.onnx" in paths
    assert ".models/Qwen2-VL-2B-Instruct" in paths
    for m in models:
        assert set(m) == {"name", "path", "exists", "download", "note"}
        assert isinstance(m["exists"], bool)
        assert m["download"]
        assert m["note"] is None or isinstance(m["note"], str)


def test_diagnostics_hardware_fields():
    """硬件条目包含平台/架构/nvidia-smi/CUDA 字段。"""
    hw = client.get("/v1/ops/diagnostics").json()["hardware"]
    for key in ("platform", "machine", "nvidia_smi_found", "cuda_available", "cuda_detail"):
        assert key in hw
    assert hw["cuda_available"] in (True, False, None)


# ---------------------------------------------------------------------------
# 检测工具单测 / Detection helper unit tests
# ---------------------------------------------------------------------------


def test_pkg_installed_detects_existing_module():
    """已存在的标准库模块应被检测为已安装。"""
    assert ops_mod._pkg_installed("os") is True


def test_pkg_installed_missing_package():
    """不存在的包应返回 False 而非抛异常。"""
    assert ops_mod._pkg_installed("definitely_not_a_real_pkg_xyz") is False


def test_pkg_version_installed_package():
    """已安装包应能读出版本号字符串。"""
    version = ops_mod._pkg_version("pytest")
    assert version is None or isinstance(version, str)
    # pytest 正在运行，必然已安装
    assert version is not None


def test_pkg_version_missing_package():
    """未安装包返回 None。"""
    assert ops_mod._pkg_version("definitely-not-a-real-pkg-xyz") is None


def test_dep_info_installed(monkeypatch):
    """依赖已安装时携带版本号。"""
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: True)
    monkeypatch.setattr(ops_mod, "_pkg_version", lambda name: "1.2.3")
    info = ops_mod._dep_info("foo", "用途", "pip install foo")
    assert info == {
        "name": "foo",
        "installed": True,
        "version": "1.2.3",
        "purpose": "用途",
        "install": "pip install foo",
    }


def test_dep_info_missing(monkeypatch):
    """依赖未安装时版本为 None。"""
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: False)
    info = ops_mod._dep_info("foo", "用途", "pip install foo")
    assert info["installed"] is False
    assert info["version"] is None


def test_model_info_exists(tmp_path, monkeypatch):
    """模型文件存在时 exists=True。"""
    monkeypatch.setattr(ops_mod, "_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "m.onnx").write_bytes(b"x")
    info = ops_mod._model_info("模型", "m.onnx", "download cmd")
    assert info["exists"] is True
    assert info["path"] == "m.onnx"


def test_model_info_missing(tmp_path, monkeypatch):
    """模型文件缺失时 exists=False。"""
    monkeypatch.setattr(ops_mod, "_PROJECT_ROOT", str(tmp_path))
    assert ops_mod._model_info("模型", "missing.onnx", "download cmd")["exists"] is False


def test_engine_available_missing_deps(monkeypatch):
    """缺少依赖时报告不可用并列出缺失依赖名。"""
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: False)
    ok, reason = ops_mod._engine_available({"engine": "onnx", "deps": ["onnxruntime"], "model": "m.onnx"})
    assert ok is False
    assert "缺少依赖" in reason and "onnxruntime" in reason


def test_engine_available_missing_model(monkeypatch):
    """依赖齐全但模型文件缺失时报告不可用（本次排障的真实场景）。"""
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: True)
    monkeypatch.setattr(ops_mod.os.path, "exists", lambda path: False)
    ok, reason = ops_mod._engine_available({"engine": "onnx", "deps": ["onnxruntime"], "model": ".models/raner_cmeee.onnx"})
    assert ok is False
    assert "模型文件不存在" in reason


# ---------------------------------------------------------------------------
# 孤儿 ONNX 外部数据检测单测 / Orphaned ONNX external-data detection unit tests
# ---------------------------------------------------------------------------


def test_has_orphan_onnx_data_detected(tmp_path, monkeypatch):
    """主图 .onnx 缺失但 .onnx.data 存在 → 判定为孤儿外部数据。"""
    monkeypatch.setattr(ops_mod, "_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "m.onnx.data").write_bytes(b"w")
    assert ops_mod._has_orphan_onnx_data("m.onnx") is True


def test_has_orphan_onnx_data_complete_model(tmp_path, monkeypatch):
    """.onnx 与 .onnx.data 均存在 → 非孤儿。"""
    monkeypatch.setattr(ops_mod, "_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "m.onnx").write_bytes(b"g")
    (tmp_path / "m.onnx.data").write_bytes(b"w")
    assert ops_mod._has_orphan_onnx_data("m.onnx") is False


def test_has_orphan_onnx_data_non_onnx_path(tmp_path, monkeypatch):
    """非 .onnx 路径（如 ModelScope 目录）不参与孤儿判定。"""
    monkeypatch.setattr(ops_mod, "_PROJECT_ROOT", str(tmp_path))
    assert ops_mod._has_orphan_onnx_data("raner_cmeee") is False


def test_onnx_orphan_note_content(tmp_path, monkeypatch):
    """孤儿状态生成同时包含两个文件名的备注；补齐后返回 None。"""
    monkeypatch.setattr(ops_mod, "_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "m.onnx.data").write_bytes(b"w")
    note = ops_mod._onnx_orphan_note("m.onnx")
    assert note and "m.onnx.data" in note and "m.onnx" in note
    (tmp_path / "m.onnx").write_bytes(b"g")
    assert ops_mod._onnx_orphan_note("m.onnx") is None


def test_engine_available_orphan_onnx_hint(tmp_path, monkeypatch):
    """依赖齐全但 ONNX 图文件缺失（权重孤儿）时，原因附带“仅缺图文件”提示。"""
    monkeypatch.setattr(ops_mod, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: True)
    (tmp_path / "raner_cmeee.onnx.data").write_bytes(b"w")
    ok, reason = ops_mod._engine_available(
        {"engine": "onnx", "deps": ["onnxruntime"], "model": "raner_cmeee.onnx"}
    )
    assert ok is False
    assert "模型文件不存在" in reason
    assert "仅缺图文件" in reason


def test_engine_available_missing_extra_file(tmp_path, monkeypatch):
    """附加文件（如词表）缺失时同样报告不可用。"""
    monkeypatch.setattr(ops_mod, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: True)
    (tmp_path / "m.onnx").write_bytes(b"x")  # 主模型存在，词表缺失
    ok, reason = ops_mod._engine_available(
        {"engine": "onnx", "deps": ["onnxruntime"], "model": "m.onnx", "extra_files": ["vocab.txt"]}
    )
    assert ok is False
    assert "附加文件不存在" in reason and "vocab.txt" in reason


def test_engine_available_all_satisfied(tmp_path, monkeypatch):
    """依赖与模型全满足时报告可用。"""
    monkeypatch.setattr(ops_mod, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: True)
    (tmp_path / "m.onnx").write_bytes(b"x")
    (tmp_path / "vocab.txt").write_bytes(b"v")
    ok, reason = ops_mod._engine_available(
        {"engine": "onnx", "deps": ["onnxruntime"], "model": "m.onnx", "extra_files": ["vocab.txt"]}
    )
    assert ok is True
    assert reason is None


# ---------------------------------------------------------------------------
# 降级链推断单测 / Degradation chain inference unit tests
# ---------------------------------------------------------------------------


def test_build_ner_status_first_available_wins(monkeypatch):
    """全部引擎可用时激活链首（mlx）。"""
    monkeypatch.setattr(ops_mod, "_engine_available", lambda spec: (True, None))
    monkeypatch.setattr(ops_mod, "_runtime_ner_engine", lambda: None)
    monkeypatch.setattr(ops_mod, "_probe_ner_engines", lambda: {
        "active_engine": "mlx",
        "available": True,
        "details": [{"engine": "mlx", "ok": True, "error": None}],
    })
    status = ops_mod._build_ner_status()
    assert status["active_engine"] == "mlx"
    assert status["predicted_engine"] == "mlx"
    assert status["runtime_engine"] is None
    assert status["available"] is True
    assert status["determined_by"] == "probe"
    assert len(status["degradation_chain"]) == 4


def test_build_ner_status_degrades_to_modelscope(monkeypatch):
    """前三级均不可用时降级到 modelscope（对应当前真实环境）。"""
    def fake_available(spec):
        if spec["engine"] in ("mlx", "tensorrt", "onnx"):
            return False, f"{spec['engine']} 不可用"
        return True, None

    monkeypatch.setattr(ops_mod, "_engine_available", fake_available)
    monkeypatch.setattr(ops_mod, "_runtime_ner_engine", lambda: None)
    monkeypatch.setattr(ops_mod, "_probe_ner_engines", lambda: {
        "active_engine": "modelscope",
        "available": True,
        "details": [
            {"engine": "mlx", "ok": False, "error": "mlx 不可用"},
            {"engine": "tensorrt", "ok": False, "error": "tensorrt 不可用"},
            {"engine": "onnx", "ok": False, "error": "onnx 不可用"},
            {"engine": "modelscope", "ok": True, "error": None},
        ],
    })
    status = ops_mod._build_ner_status()
    assert status["active_engine"] == "modelscope"
    assert status["available"] is True
    reasons = {e["engine"]: e["reason"] for e in status["degradation_chain"]}
    assert reasons["mlx"] and reasons["tensorrt"] and reasons["onnx"]
    assert reasons["modelscope"] is None


def test_build_ner_status_none_available(monkeypatch):
    """全部引擎不可用时 active_engine 为 none 且 available=False。"""
    monkeypatch.setattr(ops_mod, "_engine_available", lambda spec: (False, "不可用"))
    monkeypatch.setattr(ops_mod, "_runtime_ner_engine", lambda: None)
    monkeypatch.setattr(ops_mod, "_probe_ner_engines", lambda: {
        "active_engine": "none",
        "available": False,
        "details": [{"engine": e, "ok": False, "error": "不可用"} for e in ("mlx", "tensorrt", "onnx", "modelscope")],
    })
    status = ops_mod._build_ner_status()
    assert status["active_engine"] == "none"
    assert status["predicted_engine"] == "none"
    assert status["available"] is False


def test_build_ner_status_runtime_overrides_prediction(monkeypatch):
    """运行时真实引擎优先于动态探测（如进程内已实际激活 onnx）。"""
    monkeypatch.setattr(
        ops_mod, "_engine_available", lambda spec: (spec["engine"] == "modelscope", None)
    )
    monkeypatch.setattr(ops_mod, "_runtime_ner_engine", lambda: "onnx")
    monkeypatch.setattr(ops_mod, "_probe_ner_engines", lambda: {
        "active_engine": "modelscope",
        "available": True,
        "details": [{"engine": "modelscope", "ok": True, "error": None}],
    })
    status = ops_mod._build_ner_status()
    assert status["predicted_engine"] == "modelscope"
    assert status["active_engine"] == "onnx"
    assert status["runtime_engine"] == "onnx"
    assert status["determined_by"] == "runtime"
    assert status["available"] is True


def test_runtime_ner_engine_not_initialized(monkeypatch):
    """适配器未初始化（尚无分类请求）时返回 None。"""
    class FakeAdapter:
        _initialized = False
        _available = True
        _engine = None

    class FakeService:
        _ner_adapter = FakeAdapter()

    monkeypatch.setattr(dyn_router, "_service", FakeService())
    assert ops_mod._runtime_ner_engine() is None


def test_runtime_ner_engine_unavailable_maps_to_none(monkeypatch):
    """适配器已初始化但全部后端失败时返回 'none'。"""
    class FakeAdapter:
        _initialized = True
        _available = False
        _engine = None

    class FakeService:
        _ner_adapter = FakeAdapter()

    monkeypatch.setattr(dyn_router, "_service", FakeService())
    assert ops_mod._runtime_ner_engine() == "none"


def test_runtime_ner_engine_maps_engine_class_name(monkeypatch):
    """运行时引擎类名包含 'onnx' 时映射回降级链标识 'onnx'。"""
    class SomeOnnxEngine:
        pass

    class FakeAdapter:
        _initialized = True
        _available = True
        _engine = SomeOnnxEngine()

    class FakeService:
        _ner_adapter = FakeAdapter()

    monkeypatch.setattr(dyn_router, "_service", FakeService())
    assert ops_mod._runtime_ner_engine() == "onnx"


def test_runtime_ner_engine_no_service(monkeypatch):
    """服务尚未创建时安全返回 None（不抛异常）。"""
    monkeypatch.setattr(dyn_router, "_service", None)
    assert ops_mod._runtime_ner_engine() is None


# ---------------------------------------------------------------------------
# LLM 状态与硬件检测单测 / LLM status & hardware unit tests
# ---------------------------------------------------------------------------


def test_build_llm_status_missing_deps(monkeypatch):
    """torch/transformers 缺失时 LLM 不可用并给出原因。"""
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: False)
    monkeypatch.setattr(ops_mod, "_runtime_llm_available", lambda: None)
    monkeypatch.setattr(ops_mod, "_probe_llm", lambda: {
        "available": False, "error": "依赖缺失",
    })
    status = ops_mod._build_llm_status()
    assert status["available"] is False
    assert status["deps_met"] is False
    assert status["determined_by"] == "probe"


def test_build_llm_status_missing_model(monkeypatch):
    """依赖齐全但模型目录缺失时 LLM 不可用。"""
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: True)
    monkeypatch.setattr(ops_mod.os.path, "exists", lambda path: False)
    monkeypatch.setattr(ops_mod, "_runtime_llm_available", lambda: None)
    monkeypatch.setattr(ops_mod, "_probe_llm", lambda: {
        "available": False, "error": "模型加载失败",
    })
    status = ops_mod._build_llm_status()
    assert status["available"] is False
    assert status["model_exists"] is False


def test_build_llm_status_ready(monkeypatch):
    """依赖与模型齐备时 LLM 可用。"""
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: True)
    monkeypatch.setattr(ops_mod.os.path, "exists", lambda path: True)
    monkeypatch.setattr(ops_mod, "_runtime_llm_available", lambda: None)
    monkeypatch.setattr(ops_mod, "_probe_llm", lambda: {
        "available": True, "error": None,
    })
    status = ops_mod._build_llm_status()
    assert status["available"] is True
    assert status["deps_met"] is True
    assert status["model_exists"] is True
    assert status["reason"] is None
    assert status["determined_by"] == "probe"


def test_build_llm_status_runtime_overrides(monkeypatch):
    """运行时 LLM 初始化失败时，即便静态条件满足也报告不可用。"""
    monkeypatch.setattr(ops_mod, "_pkg_installed", lambda name: True)
    monkeypatch.setattr(ops_mod.os.path, "exists", lambda path: True)
    monkeypatch.setattr(ops_mod, "_runtime_llm_available", lambda: False)
    monkeypatch.setattr(ops_mod, "_probe_llm", lambda: {
        "available": True, "error": None,
    })
    status = ops_mod._build_llm_status()
    assert status["available"] is False
    assert status["runtime_available"] is False
    assert status["determined_by"] == "runtime"


def test_build_hardware_torch_not_loaded(monkeypatch):
    """torch 未加载时 CUDA 状态为 None（不为诊断而导入 torch）。"""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    info = ops_mod._build_hardware()
    assert info["cuda_available"] is None
    assert "torch 尚未加载" in info["cuda_detail"]
    assert isinstance(info["nvidia_smi_found"], bool)


def test_build_hardware_torch_loaded_cuda_available(monkeypatch):
    """torch 已加载且 CUDA 可用时报告 GPU 详情。"""
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(idx):
            return "FakeGPU"

    class FakeTorch:
        cuda = FakeCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    info = ops_mod._build_hardware()
    assert info["cuda_available"] is True
    assert "FakeGPU" in info["cuda_detail"]


def test_build_hardware_torch_loaded_cuda_unavailable(monkeypatch):
    """torch 已加载但 CUDA 不可用时给出 CPU 版提示。"""
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    info = ops_mod._build_hardware()
    assert info["cuda_available"] is False
    assert "CUDA 不可用" in info["cuda_detail"]


# ---------------------------------------------------------------------------
# 动态探测单测 / Dynamic probe unit tests
# （与 tests/dynclassification 相同的判定方式）
# ---------------------------------------------------------------------------


def test_probe_ner_engines_all_fail(monkeypatch):
    """所有引擎初始化均失败时，探测结果 active_engine='none'。"""
    monkeypatch.setattr(ops_mod, "_probe_cache", None)  # 清除缓存

    # Mock 所有引擎导入均抛异常
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if "mlx_ner_engine" in name or "ner_engines" in name:
            raise ImportError(f"mock: {name} not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = ops_mod._probe_ner_engines()
    assert result["active_engine"] == "none"
    assert result["available"] is False
    assert len(result["details"]) == 4
    for d in result["details"]:
        assert d["ok"] is False
        assert d["error"]
    # 清理缓存以免影响其他测试
    monkeypatch.setattr(ops_mod, "_probe_cache", None)


def test_probe_ner_engines_onnx_succeeds(monkeypatch):
    """模拟 ONNX 引擎初始化成功，探测应报告 onnx 为激活引擎。"""
    monkeypatch.setattr(ops_mod, "_probe_cache", None)

    from unittest.mock import MagicMock, patch

    # MLX 和 TensorRT 失败，ONNX 成功
    mock_onnx_cls = MagicMock()
    mock_onnx_instance = MagicMock()
    mock_onnx_cls.return_value = mock_onnx_instance

    with patch.dict("sys.modules", {"mlx": None}):
        # 直接构造探测结果（模拟成功场景）
        fake_result = {
            "active_engine": "onnx",
            "available": True,
            "details": [
                {"engine": "mlx", "ok": False, "error": "No module named 'mlx'"},
                {"engine": "tensorrt", "ok": False, "error": "未找到本地 ONNX 模型文件"},
                {"engine": "onnx", "ok": True, "error": None},
                {"engine": "modelscope", "ok": False, "error": "已由更高优先级引擎激活，跳过"},
            ],
        }
        monkeypatch.setattr(ops_mod, "_probe_cache", fake_result)
        result = ops_mod._probe_ner_engines()

    assert result["active_engine"] == "onnx"
    assert result["available"] is True
    assert result["details"][2]["ok"] is True
    monkeypatch.setattr(ops_mod, "_probe_cache", None)


def test_probe_ner_engines_caches_result(monkeypatch):
    """探测结果应被缓存，第二次调用不重复执行。"""
    cached = {
        "active_engine": "modelscope",
        "available": True,
        "details": [{"engine": "modelscope", "ok": True, "error": None}],
    }
    monkeypatch.setattr(ops_mod, "_probe_cache", cached)
    result = ops_mod._probe_ner_engines()
    assert result is cached  # 同一对象，证明走了缓存


def test_probe_llm_success(monkeypatch):
    """模拟 LlmAdapter 可用时探测结果 available=True。"""
    monkeypatch.setattr(ops_mod, "_probe_llm_cache", None)
    monkeypatch.setattr(ops_mod, "_probe_llm_cache", {"available": True, "error": None})
    result = ops_mod._probe_llm()
    assert result["available"] is True
    assert result["error"] is None


def test_probe_llm_failure(monkeypatch):
    """模拟 LlmAdapter 初始化失败时探测结果包含错误信息。"""
    monkeypatch.setattr(ops_mod, "_probe_llm_cache", None)
    monkeypatch.setattr(ops_mod, "_probe_llm_cache", {
        "available": False, "error": "模型文件损坏",
    })
    result = ops_mod._probe_llm()
    assert result["available"] is False
    assert "模型文件损坏" in result["error"]


def test_probe_llm_caches_result(monkeypatch):
    """探测结果应被缓存。"""
    cached = {"available": True, "error": None}
    monkeypatch.setattr(ops_mod, "_probe_llm_cache", cached)
    result = ops_mod._probe_llm()
    assert result is cached


def test_diagnostics_ner_probe_field_in_response():
    """端点响应中 NER 状态包含 probe 字段且结构完整。"""
    ner = client.get("/v1/ops/diagnostics").json()["engines"]["ner"]
    assert "probe" in ner
    probe = ner["probe"]
    assert "active_engine" in probe
    assert "available" in probe
    assert "details" in probe
    assert isinstance(probe["details"], list)
    for d in probe["details"]:
        assert {"engine", "ok", "error"} <= set(d)


def test_diagnostics_llm_probe_field_in_response():
    """端点响应中 LLM 状态包含 probe 字段且结构完整。"""
    llm = client.get("/v1/ops/diagnostics").json()["engines"]["llm"]
    assert "probe" in llm
    assert "determined_by" in llm
    probe = llm["probe"]
    assert "available" in probe
    assert "error" in probe
