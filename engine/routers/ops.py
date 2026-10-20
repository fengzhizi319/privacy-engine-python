"""运维诊断路由（/v1/ops/diagnostics）。

为测试控制台提供一站式运维诊断信息，帮助快速定位问题出在哪一层：

    前端（浏览器） → 控制台后端（统一由 Go BFF 提供 REST/gRPC 代理） → Agent（本服务）

并进一步报告 Agent 内部：
    - 三层分类漏斗中 NER（Layer-2）/ LLM（Layer-3）引擎的降级链路；
    - 各 ML 依赖（onnxruntime / torch / transformers / modelscope / tensorrt / mlx）
      是否安装、版本号与安装方式；
    - 模型文件是否存在、如何下载；
    - CUDA / GPU 等硬件加速可用性。

设计要点：
    - **自动判断**：采用与 tests/dynclassification 相同的判定方式——实际尝试
      初始化各引擎（MLX → TensorRT → ONNX → ModelScope），第一个成功的即为真实
      激活引擎。这比纯静态文件检查更准确，能捕获依赖版本冲突、模型格式损坏、
      CUDA 不兼容等静态检查无法发现的问题。
    - **探测缓存**：动态探测结果缓存于模块级变量，整个进程生命周期内只执行一次，
      避免重复加载数百 MB 模型。
    - **降级链路推断**：按 NER 适配器的真实尝试顺序逐个判断，并保留各引擎的
      实际错误信息，供运维人员精准定位问题。
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..deps import NAMESPACE
from ..observability.logging_config import get_logger
from ..observability.middleware import record_auth_denial
from ..security.auth import get_current_identity, require_permission
from ..security.identity import Identity
from ..security.ratelimit import rate_limit_dependency

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/ops", tags=["Ops"])
# 根路径别名（Go 契约：/ops/diagnostics），与 /v1/ops/diagnostics 共用处理器。
root_router = APIRouter(tags=["Ops"])

# 诊断端点依赖：认证 + 限速 + 接口级权限。
# - 读取诊断信息需要 "ops:diagnostics" scope；
# - refresh=true 会真实加载数百 MB 模型（DoS 面），需要更高权限 "ops:admin"
#   （在端点函数体内校验，见 diagnostics()）。
# 注意：未启用认证时（本地开发默认）匿名身份持有 ["*"]，上述校验全部放行。
_OPS_DEPS = [
    Depends(get_current_identity),
    Depends(rate_limit_dependency),
    require_permission("ops:diagnostics"),
]

# 项目根目录：routers/ → PrivShield/ → 项目根。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 依赖与模型检测工具
# ---------------------------------------------------------------------------


def _pkg_installed(name: str) -> bool:
    """判断包是否可被导入（不真正 import，避免加载重量级库）。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _pkg_version(name: str) -> str | None:
    """读取已安装包的版本号（仅读元数据，不 import）；未安装返回 None。"""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _dep_info(name: str, purpose: str, install: str) -> dict[str, Any]:
    """构造单个依赖的检测条目。"""
    installed = _pkg_installed(name)
    return {
        "name": name,
        "installed": installed,
        "version": _pkg_version(name) if installed else None,
        "purpose": purpose,
        "install": install,
    }


def _model_info(name: str, rel_path: str, download: str, note: str | None = None) -> dict[str, Any]:
    """构造单个模型文件的检测条目。"""
    abs_path = os.path.join(_PROJECT_ROOT, rel_path)
    return {
        "name": name,
        "path": rel_path,
        "exists": os.path.exists(abs_path),
        "download": download,
        "note": note,
    }


def _has_orphan_onnx_data(rel_path: str) -> bool:
    """判断某 ONNX 模型是否处于“主图文件缺失、但外部权重文件（.onnx.data）存在”的孤儿状态。

    ONNX 外部数据模型由两个文件配对组成：.onnx（计算图定义） + .onnx.data（张量权重）。
    若仅存在 .onnx.data，通常是下载中断或仅拉取了权重文件；缺少图文件时
    onnxruntime 无法加载，引擎必须继续降级。识别该状态可避免运维人员
    误以为“模型已存在”。
    """
    if not rel_path.endswith(".onnx"):
        return False
    if os.path.exists(os.path.join(_PROJECT_ROOT, rel_path)):
        return False
    return os.path.exists(os.path.join(_PROJECT_ROOT, rel_path + ".data"))


def _onnx_orphan_note(rel_path: str) -> str | None:
    """为孤儿 ONNX 外部数据文件生成解释性备注；非孤儿状态返回 None。"""
    if not _has_orphan_onnx_data(rel_path):
        return None
    return (
        f"检测到外部权重文件 {rel_path}.data，但主图文件 {rel_path} 缺失。"
        "ONNX 模型需 .onnx（图）+.onnx.data（权重）配对，可能下载中断，"
        "请重新下载补齐，否则 ONNX/TensorRT 引擎无法加载。"
    )


# ---------------------------------------------------------------------------
# NER / LLM 降级链路推断
# ---------------------------------------------------------------------------

# NER 引擎降级链（与 NerAdapter._lazy_init 的尝试顺序严格一致）。
# 每个引擎给出：所需依赖、模型文件（相对项目根）、用途说明。
_NER_CHAIN: list[dict[str, Any]] = [
    {
        "engine": "mlx",
        "deps": ["mlx"],
        "model": ".models/raner_cmeee-mlx",
        "note": "Apple Silicon Metal GPU 加速（仅 macOS）",
    },
    {
        "engine": "tensorrt",
        "deps": ["onnxruntime", "tensorrt"],
        "model": ".models/raner_cmeee.onnx",
        "extra_files": [".models/vocab.txt"],
        "note": "NVIDIA GPU TensorRT 硬件加速（FP16）",
    },
    {
        "engine": "onnx",
        "deps": ["onnxruntime"],
        "model": ".models/raner_cmeee.onnx",
        "extra_files": [".models/vocab.txt"],
        "note": "ONNX Runtime 推理（CPU / CUDA，轻量推荐）",
    },
    {
        "engine": "modelscope",
        "deps": ["modelscope", "torch", "transformers"],
        "model": ".models/raner_cmeee",
        "note": "ModelScope 官方管道（需 PyTorch 全家桶）",
    },
]


def _engine_available(spec: dict[str, Any]) -> tuple[bool, str | None]:
    """判断某个 NER 引擎是否可用（静态检查），返回 (可用?, 不可用原因)。

    可用条件：全部依赖已安装 且 模型文件（含词表等附加文件）存在。
    注意：这是轻量级静态推断，不触发模型加载。
    """
    missing_deps = [d for d in spec["deps"] if not _pkg_installed(d)]
    if missing_deps:
        return False, "缺少依赖: " + ", ".join(missing_deps)

    model_abs = os.path.join(_PROJECT_ROOT, spec["model"])
    if not os.path.exists(model_abs):
        reason = f"模型文件不存在: {spec['model']}"
        if _has_orphan_onnx_data(spec["model"]):
            reason += "（权重文件 .onnx.data 已在，仅缺图文件）"
        return False, reason

    for extra in spec.get("extra_files", []):
        if not os.path.exists(os.path.join(_PROJECT_ROOT, extra)):
            return False, f"附加文件不存在: {extra}"

    return True, None


# ---------------------------------------------------------------------------
# 动态探测（与 tests/dynclassification 相同的判定方式）
# ---------------------------------------------------------------------------

# 模块级缓存：探测结果只执行一次（避免重复加载数百 MB 模型）。
# 使用 Lock 保护，防止并发请求重复初始化（竞态条件）。
_probe_cache: dict[str, Any] | None = None
_probe_llm_cache: dict[str, Any] | None = None
_probe_lock = threading.Lock()
_probe_llm_lock = threading.Lock()


def _invalidate_probe_cache() -> None:
    """失效动态探测缓存，使下一次诊断调用重新探测各引擎。

    适用场景：服务运行期间依赖或模型发生变化（如 agent 启动后才补装
    mlx / 下载模型）。探测结果缓存于进程生命周期内，若不失效，
    诊断页会持续展示陈旧的“未安装 / 不可用”结论，与实际环境不符。
    """
    global _probe_cache, _probe_llm_cache
    with _probe_lock:
        _probe_cache = None
    with _probe_llm_lock:
        _probe_llm_cache = None


def _probe_ner_engines() -> dict[str, Any]:
    """像 tests/dynclassification 一样，实际尝试初始化各 NER 引擎来自动判断。

    与 NerAdapter._lazy_init() 采用完全相同的尝试顺序（MLX → TensorRT → ONNX →
    ModelScope），逐个实例化并调用 _lazy_init()，第一个成功的即为真实激活引擎。
    这比静态文件检查更准确——它能捕获依赖版本冲突、模型格式损坏、CUDA 不兼容
    等静态检查无法发现的问题。

    结果缓存到模块级变量，整个进程生命周期内只探测一次。

    Returns:
        {
            "active_engine": str | "none",
            "available": bool,
            "details": [{"engine": ..., "ok": bool, "error": str|None}, ...],
        }
    """
    global _probe_cache
    if _probe_cache is not None:
        return _probe_cache

    with _probe_lock:
        # Double-check：获取锁后再次检查（另一线程可能已完成初始化）。
        if _probe_cache is not None:
            return _probe_cache

        details: list[dict[str, Any]] = []
        active: str | None = None

        # 尝试 0: MLX（Apple Silicon Metal GPU，仅 macOS）
        if sys.platform == "darwin":
            try:
                from ..dynclassification.mlx_ner_engine import MLXSmallNerEngine

                engine = MLXSmallNerEngine()
                engine._lazy_init()
                details.append({"engine": "mlx", "ok": True, "error": None})
                active = "mlx"
            except Exception as e:
                details.append({"engine": "mlx", "ok": False, "error": str(e)})
        else:
            details.append({"engine": "mlx", "ok": False, "error": "仅 macOS 支持 MLX"})

        # 尝试 1: TensorRT（NVIDIA GPU 硬件加速）
        if active is None:
            try:
                from ..dynclassification.ner_engines import TensorRTSmallNerEngine

                engine = TensorRTSmallNerEngine()
                engine._lazy_init()
                details.append({"engine": "tensorrt", "ok": True, "error": None})
                active = "tensorrt"
            except Exception as e:
                details.append({"engine": "tensorrt", "ok": False, "error": str(e)})
        else:
            details.append({"engine": "tensorrt", "ok": False, "error": "已由更高优先级引擎激活，跳过"})

        # 尝试 2: ONNX Runtime（轻量推荐）
        if active is None:
            try:
                from ..dynclassification.ner_engines import ONNXSmallNerEngine

                engine = ONNXSmallNerEngine()
                engine._lazy_init()
                details.append({"engine": "onnx", "ok": True, "error": None})
                active = "onnx"
            except Exception as e:
                details.append({"engine": "onnx", "ok": False, "error": str(e)})
        else:
            details.append({"engine": "onnx", "ok": False, "error": "已由更高优先级引擎激活，跳过"})

        # 尝试 3: ModelScope（需 PyTorch 全家桶）
        if active is None:
            try:
                from ..dynclassification.ner_engines import ModelScopeSmallNerEngine

                engine = ModelScopeSmallNerEngine()
                engine._lazy_init()
                details.append({"engine": "modelscope", "ok": True, "error": None})
                active = "modelscope"
            except Exception as e:
                details.append({"engine": "modelscope", "ok": False, "error": str(e)})
        else:
            details.append({"engine": "modelscope", "ok": False, "error": "已由更高优先级引擎激活，跳过"})

        _probe_cache = {
            "active_engine": active or "none",
            "available": active is not None,
            "details": details,
        }
        return _probe_cache


def _probe_llm() -> dict[str, Any]:
    """动态探测 LLM 可用性（与 tests/dynclassification/test_real_models.py 相同逻辑）。

    实际尝试实例化 LlmAdapter 并检查 is_available，而非仅检查文件存在。
    结果缓存到模块级变量。
    """
    global _probe_llm_cache
    if _probe_llm_cache is not None:
        return _probe_llm_cache

    with _probe_llm_lock:
        # Double-check：获取锁后再次检查。
        if _probe_llm_cache is not None:
            return _probe_llm_cache

        try:
            from ..dynclassification.llm_adapter import LlmAdapter

            adapter = LlmAdapter()
            available = adapter.is_available
            _probe_llm_cache = {
                "available": available,
                "error": None if available else "LlmAdapter 初始化失败（依赖或模型不完整）",
            }
        except Exception as e:
            _probe_llm_cache = {
                "available": False,
                "error": str(e),
            }
        return _probe_llm_cache


def _runtime_ner_engine() -> str | None:
    """读取运行时已激活的 NER 引擎名（若适配器尚未初始化则返回 None）。

    不会触发任何模型加载——仅当分类请求已经发生过、适配器已完成懒初始化时，
    才能读到真实的激活引擎；否则返回 None，由调用方回退到动态探测。
    """
    try:
        from . import dynclassification as dyn_router

        svc = getattr(dyn_router, "_service", None)
        if svc is None:
            return None
        adapter = getattr(svc, "_ner_adapter", None)
        if adapter is None or not getattr(adapter, "_initialized", False):
            return None
        if not getattr(adapter, "_available", False):
            return "none"
        engine = getattr(adapter, "_engine", None)
        if engine is None:
            return "none"
        # 将引擎类名映射回降级链中的标识。
        cls = type(engine).__name__.lower()
        for key in ("mlx", "tensorrt", "onnx", "modelscope"):
            if key in cls:
                return key
        return cls
    except Exception:
        return None


def _runtime_llm_available() -> bool | None:
    """读取运行时 LLM 适配器的可用状态（未初始化则返回 None）。"""
    try:
        from . import dynclassification as dyn_router

        svc = getattr(dyn_router, "_service", None)
        if svc is None:
            return None
        adapter = getattr(svc, "_llm_adapter", None)
        if adapter is None or not getattr(adapter, "_initialized", False):
            return None
        return bool(getattr(adapter, "_available", False))
    except Exception:
        return None


def _build_ner_status() -> dict[str, Any]:
    """构造 NER 引擎降级链路状态。

    判定优先级（与 tests/dynclassification 一致）：
    1. 运行时已初始化的适配器（最权威——分类请求已发生过）
    2. 动态探测（实际尝试初始化各引擎，与 test_real_models.py 逻辑相同）
    3. 静态推断（轻量文件/依赖检查，作为补充参考）
    """
    # 静态分析（轻量，始终执行，用于展示每个引擎的详细依赖/模型状态）
    chain: list[dict[str, Any]] = []
    static_active: str | None = None
    for spec in _NER_CHAIN:
        available, reason = _engine_available(spec)
        if available and static_active is None:
            static_active = spec["engine"]
        chain.append(
            {
                "engine": spec["engine"],
                "available": available,
                "reason": reason,
                "deps": spec["deps"],
                "model": spec["model"],
                "note": spec["note"],
            }
        )

    # 运行时状态（若分类请求已触发过适配器初始化）
    runtime = _runtime_ner_engine()

    # 动态探测（与 tests/dynclassification 相同的判定方式）
    probe = _probe_ner_engines()

    # 将探测的实际错误信息合并到 chain 中
    probe_errors = {d["engine"]: d["error"] for d in probe["details"] if not d["ok"] and d["error"]}
    for item in chain:
        eng = item["engine"]
        if eng in probe_errors:
            item["probe_error"] = probe_errors[eng]
        # 如果探测成功但静态检查认为不可用，以探测为准
        probe_ok = any(d["engine"] == eng and d["ok"] for d in probe["details"])
        if probe_ok and not item["available"]:
            item["available"] = True
            item["reason"] = None

    # 最终激活引擎判定：runtime > probe > static
    if runtime is not None:
        active_engine = runtime
    else:
        active_engine = probe["active_engine"]

    return {
        # 最终判定的激活引擎（自动判断结果）
        "active_engine": active_engine,
        "available": active_engine != "none",
        # 判定来源：runtime（运行时已初始化）/ probe（动态探测）/ static（静态推断）
        "determined_by": "runtime" if runtime is not None else "probe",
        # 动态探测详情（与 tests/dynclassification 相同的尝试结果）
        "probe": probe,
        # 静态推断值（仅供参考）
        "predicted_engine": static_active or "none",
        # 运行时激活引擎（未初始化时为 None）
        "runtime_engine": runtime,
        # 降级链详情（含静态分析 + 探测错误信息）
        "degradation_chain": chain,
    }


def _build_llm_status() -> dict[str, Any]:
    """构造 LLM 引擎状态。

    判定优先级（与 tests/dynclassification/test_real_models.py 一致）：
    1. 运行时已初始化的适配器
    2. 动态探测（实际尝试实例化 LlmAdapter）
    3. 静态推断（依赖 + 模型目录存在性）
    """
    deps = ["torch", "transformers"]
    missing_deps = [d for d in deps if not _pkg_installed(d)]
    model_rel = ".models/Qwen2-VL-2B-Instruct"
    model_exists = os.path.exists(os.path.join(_PROJECT_ROOT, model_rel))

    deps_met = not missing_deps

    if missing_deps:
        reason: str | None = "缺少依赖: " + ", ".join(missing_deps)
    elif not model_exists:
        reason = f"模型目录不存在: {model_rel}"
    else:
        reason = None

    # 运行时状态
    runtime = _runtime_llm_available()

    # 动态探测（与 test_real_models.py 相同：实例化 LlmAdapter 并检查 is_available）
    probe = _probe_llm()

    # 最终判定：runtime > probe > static
    if runtime is not None:
        available = runtime
        determined_by = "runtime"
    else:
        available = probe["available"]
        determined_by = "probe"
        # 如果探测失败，用探测错误信息补充 reason
        if not available and probe.get("error"):
            reason = probe["error"]

    return {
        "backend": "qwen2vl",
        "available": available,
        "determined_by": determined_by,
        "runtime_available": runtime,
        "probe": probe,
        "deps": deps,
        "deps_met": deps_met,
        "model": model_rel,
        "model_exists": model_exists,
        "reason": reason,
        "note": "Qwen2-VL-2B-Instruct 多模态大模型（Layer-3 深度分类 / 仲裁）",
    }


# ---------------------------------------------------------------------------
# 硬件加速检测
# ---------------------------------------------------------------------------


def _build_hardware() -> dict[str, Any]:
    """检测 CUDA / GPU 硬件加速可用性（安全、不触发 torch 导入）。"""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "nvidia_smi_found": shutil.which("nvidia-smi") is not None,
    }

    # 仅当 torch 已被加载（例如 LLM/ModelScope 引擎已激活）时才读取 CUDA 状态，
    # 避免为了诊断而首次导入数 GB 的 torch。
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            cuda_available = bool(torch_mod.cuda.is_available())
            info["cuda_available"] = cuda_available
            info["cuda_detail"] = (
                f"{torch_mod.cuda.device_count()} GPU(s): "
                + torch_mod.cuda.get_device_name(0)
                if cuda_available
                else "torch 已加载但 CUDA 不可用（可能为 CPU 版 torch 或缺少驱动）"
            )
        except Exception as e:  # noqa: BLE001
            info["cuda_available"] = False
            info["cuda_detail"] = f"CUDA 检测失败: {e}"
    else:
        info["cuda_available"] = None
        info["cuda_detail"] = (
            "torch 尚未加载，无法检测 CUDA。"
            + (" 已检测到 nvidia-smi，说明宿主机装有 NVIDIA 驱动。" if info["nvidia_smi_found"] else "")
        )
    return info


# ---------------------------------------------------------------------------
# 诊断端点
# ---------------------------------------------------------------------------


@root_router.get("/ops/diagnostics", dependencies=_OPS_DEPS)
@router.get("/diagnostics", dependencies=_OPS_DEPS)
def diagnostics(
    refresh: bool = False,
    identity: Identity = Depends(get_current_identity),
) -> dict[str, Any]:
    """一站式运维诊断接口。

    返回服务信息、NER/LLM 降级链路、依赖安装情况、模型文件与硬件加速状态，
    供控制台“运维诊断”页面渲染，用于快速判断问题出在前端 / 后端 / Agent 哪一层，
    以及 Agent 内部 NER/LLM 降级到了哪一级、缺少哪些驱动与如何安装。

    Args:
        refresh: 为 True 时先失效引擎探测缓存再重新探测。重新探测会真实加载
            数百 MB 模型、耗时较长，属于高成本管理操作，因此除
            ``ops:diagnostics`` 外还要求调用者持有 ``ops:admin`` scope，
            防止低权限调用者反复触发模型加载造成 DoS。
    """
    if refresh:
        if not identity.has_permission("ops:admin"):
            record_auth_denial("forbidden")
            logger.warning(
                "Diagnostics refresh rejected: missing ops:admin scope",
                extra={"identity_name": identity.name},
            )
            raise HTTPException(
                status_code=403,
                detail="Forbidden: refresh requires ops:admin scope",
            )
        _invalidate_probe_cache()
    dependencies = [
        _dep_info(
            "onnxruntime",
            "NER ONNX / TensorRT 推理引擎（Layer-2）",
            "pip install onnxruntime（CPU）或 pip install onnxruntime-gpu（GPU）",
        ),
        _dep_info(
            "torch",
            "NER ModelScope 引擎 / LLM Qwen2-VL 推理（Layer-2/3）",
            "pip install torch（GPU 版参见 https://pytorch.org）",
        ),
        _dep_info(
            "transformers",
            "LLM Qwen2-VL 模型加载（Layer-3）",
            "pip install transformers",
        ),
        _dep_info(
            "modelscope",
            "NER ModelScope 官方管道（Layer-2 备选）",
            "pip install modelscope",
        ),
        _dep_info(
            "tensorrt",
            "NER TensorRT 硬件加速（Layer-2 可选）",
            "pip install tensorrt（需 NVIDIA GPU + CUDA）",
        ),
        _dep_info(
            "mlx",
            "NER MLX Metal 加速（Layer-2，仅 Apple Silicon）",
            "pip install mlx（仅 macOS Apple Silicon）",
        ),
        _dep_info(
            "onnxruntime-gpu",
            "ONNX Runtime GPU 版（与 onnxruntime 二选一）",
            "pip install onnxruntime-gpu",
        ),
    ]

    models = [
        _model_info(
            "NER ONNX 模型（CMeEE）",
            ".models/raner_cmeee.onnx",
            "python -m engine.privacy.download_ner_model",
            note=_onnx_orphan_note(".models/raner_cmeee.onnx"),
        ),
        _model_info(
            "NER 词表 vocab.txt",
            ".models/vocab.txt",
            "python -m engine.privacy.download_ner_model",
        ),
        _model_info(
            "NER ModelScope 模型目录",
            ".models/raner_cmeee",
            "python -m engine.privacy.download_ner_model",
        ),
        _model_info(
            "LLM Qwen2-VL-2B-Instruct",
            ".models/Qwen2-VL-2B-Instruct",
            "python -m engine.privacy.download_model",
        ),
    ]

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": {
            "name": os.environ.get("PRIVACY_SERVICE_NAME", "PrivShield"),
            "namespace": NAMESPACE,
            "python_version": platform.python_version(),
            "rest_port": int(os.environ.get("PRIVACY_REST_PORT", "8079")),
            "grpc_port": int(os.environ.get("PRIVACY_GRPC_PORT", "50051")),
        },
        "engines": {
            "ner": _build_ner_status(),
            "llm": _build_llm_status(),
        },
        "dependencies": dependencies,
        "models": models,
        "hardware": _build_hardware(),
    }
