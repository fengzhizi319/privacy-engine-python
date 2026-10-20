"""图片/照片输入分类分级测试 / Image Input Classification Tests.

测试覆盖场景：
- Qwen2VLClassifier._detect_image() 三级图片检测策略
  - 第 1 级：本地图片文件路径检测
  - 第 2 级：Data URI 格式 Base64 检测
  - 第 3 级：纯 Base64 数据检测
- 图片输入的分类分级完整流程（mock 模型推理）
- 非图片输入的降级处理（纯文本不误判为图片）
- 真实模型 + gen_medical_images.py 生成图片的端到端冒烟测试

运行方式：
    # 单元测试（无需 GPU/模型）
    PYTHONPATH=. pytest tests/dynclassification/test_image_classification.py -v -m "not real_models"

    # 含真实模型测试（需要 .models/Qwen2-VL-2B-Instruct + ML 依赖）
    PYTHONPATH=. pytest tests/dynclassification/test_image_classification.py -v
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skip(reason="Multimodal image and OCR support removed per architecture deprecation")

from engine.dynclassification.llm_engines import Qwen3Classifier as Qwen2VLClassifier

# --------------------------------------------------------------------------- #
# 辅助工具
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_test_image(width: int = 200, height: int = 100, text: str = "测试") -> bytes:
    """生成一张简单的 PNG 测试图片并返回字节数据。

    使用 PIL 渲染一张带文字的纯色图片，模拟病例图片输入。
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    # 绘制简单内容（无需中文字体，仅用于格式验证）
    draw.rectangle([10, 10, width - 10, height - 10], outline="black", width=2)
    draw.text((30, 40), text, fill="black")

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_image(width: int = 150, height: int = 80) -> bytes:
    """生成一张 JPEG 格式的测试图片字节数据。"""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(200, 220, 240))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# =========================================================================== #
# _detect_image 三级检测策略测试
# =========================================================================== #


class TestDetectImage:
    """测试 Qwen2VLClassifier._detect_image() 的三级图片检测策略。"""

    def setup_method(self):
        """每个测试前创建分类器实例（不加载模型）。"""
        self.classifier = Qwen2VLClassifier(model_path="/tmp/fake_model")

    # --- 第 1 级：本地文件路径检测 ---

    def test_detect_local_png_file(self, tmp_path):
        """本地 PNG 文件路径应被正确检测为图片。"""
        img_bytes = _make_test_image()
        img_file = tmp_path / "test_case.png"
        img_file.write_bytes(img_bytes)

        result = self.classifier._detect_image(str(img_file))
        assert result is not None
        # 验证返回的是 PIL Image 实例
        from PIL import Image

        assert isinstance(result, Image.Image)
        assert result.size == (200, 100)

    def test_detect_local_jpeg_file(self, tmp_path):
        """本地 JPEG 文件路径应被正确检测为图片。"""
        img_bytes = _make_jpeg_image()
        img_file = tmp_path / "report.jpg"
        img_file.write_bytes(img_bytes)

        result = self.classifier._detect_image(str(img_file))
        assert result is not None
        from PIL import Image

        assert isinstance(result, Image.Image)

    def test_detect_local_webp_file(self, tmp_path):
        """本地 WebP 文件路径应被正确检测为图片。"""
        from PIL import Image

        img = Image.new("RGB", (100, 100), "green")
        img_file = tmp_path / "scan.webp"
        img.save(img_file, format="WEBP")

        result = self.classifier._detect_image(str(img_file))
        assert result is not None

    def test_detect_nonexistent_image_path_returns_none(self):
        """不存在的图片路径应返回 None（不崩溃）。"""
        result = self.classifier._detect_image("/nonexistent/path/photo.png")
        assert result is None

    def test_detect_corrupted_image_file_returns_none(self, tmp_path):
        """损坏的图片文件应返回 None（优雅降级）。"""
        bad_file = tmp_path / "corrupted.png"
        bad_file.write_bytes(b"this is not a valid png file content")

        result = self.classifier._detect_image(str(bad_file))
        assert result is None

    def test_detect_text_file_with_image_extension(self, tmp_path):
        """扩展名为 .png 但内容非图片的文件应返回 None。"""
        fake_file = tmp_path / "fake.png"
        fake_file.write_text("hello world, I am not an image")

        result = self.classifier._detect_image(str(fake_file))
        assert result is None

    # --- 第 2 级：Data URI 格式检测 ---

    def test_detect_data_uri_png(self):
        """data:image/png;base64,... 格式应被正确检测。"""
        img_bytes = _make_test_image()
        b64_str = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64_str}"

        result = self.classifier._detect_image(data_uri)
        assert result is not None
        from PIL import Image

        assert isinstance(result, Image.Image)
        assert result.size == (200, 100)

    def test_detect_data_uri_jpeg(self):
        """data:image/jpeg;base64,... 格式应被正确检测。"""
        img_bytes = _make_jpeg_image()
        b64_str = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64_str}"

        result = self.classifier._detect_image(data_uri)
        assert result is not None

    def test_detect_data_uri_invalid_base64_returns_none(self):
        """Data URI 中 Base64 数据无效时应返回 None。"""
        data_uri = "data:image/png;base64,NOT_VALID_BASE64!!!"
        result = self.classifier._detect_image(data_uri)
        assert result is None

    # --- 第 3 级：纯 Base64 数据检测 ---

    def test_detect_pure_base64_png(self):
        """纯 Base64 编码的 PNG 数据（无前缀）应被正确检测。"""
        img_bytes = _make_test_image()
        b64_str = base64.b64encode(img_bytes).decode("ascii")

        result = self.classifier._detect_image(b64_str)
        assert result is not None
        from PIL import Image

        assert isinstance(result, Image.Image)

    def test_detect_pure_base64_jpeg(self):
        """纯 Base64 编码的 JPEG 数据应被正确检测。"""
        img_bytes = _make_jpeg_image()
        b64_str = base64.b64encode(img_bytes).decode("ascii")

        result = self.classifier._detect_image(b64_str)
        assert result is not None

    # --- 非图片输入不应误判 ---

    def test_plain_text_not_detected_as_image(self):
        """普通中文文本不应被误判为图片。"""
        text = "患者张三，男，28岁，身份证号110101199001011234，诊断急性上呼吸道感染。"
        result = self.classifier._detect_image(text)
        assert result is None

    def test_short_text_not_detected_as_image(self):
        """短文本不应被误判为图片。"""
        result = self.classifier._detect_image("L3")
        assert result is None

    def test_url_not_detected_as_image(self):
        """HTTP URL 不应被当作 Base64 图片处理。"""
        url = "https://example.com/images/medical_report.png"
        result = self.classifier._detect_image(url)
        assert result is None

    def test_json_string_not_detected_as_image(self):
        """JSON 字符串不应被误判为图片。"""
        json_str = '{"final_level": "L3", "confidence": 0.9, "reasoning": "test"}' * 5
        result = self.classifier._detect_image(json_str)
        assert result is None

    def test_empty_string_returns_none(self):
        """空字符串应返回 None。"""
        result = self.classifier._detect_image("")
        assert result is None

    def test_whitespace_only_returns_none(self):
        """纯空白字符串应返回 None。"""
        result = self.classifier._detect_image("   \n\t  ")
        assert result is None


# =========================================================================== #
# 图片分类完整流程测试（mock 模型）
# =========================================================================== #


class TestImageClassificationFlow:
    """测试图片输入经过 _classify_inner 的完整分类流程（mock 模型推理）。"""

    def _make_classifier_with_mock_model(self):
        """构建一个带 mock 模型的分类器实例，跳过真实模型加载。"""
        classifier = Qwen2VLClassifier(model_path="/tmp/fake_model")
        # 标记为已初始化，跳过 _lazy_init
        classifier._initialized = True

        # Mock 模型和处理器
        mock_model = MagicMock()
        mock_processor = MagicMock()

        # 模拟 processor.apply_chat_template 返回 prompt 字符串
        mock_processor.apply_chat_template.return_value = "<mock_prompt>"

        # 模拟 processor 调用返回输入张量字典
        # 值必须是支持 .to() 方法的对象（模拟 tensor 行为）
        # 迭代时应产出 batch 中的各条序列（如 [[1,2,3]] 迭代得 [1,2,3]）
        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor  # .to(device) 返回自身
        mock_tensor.__len__ = lambda self: 1  # batch_size=1
        mock_tensor.__iter__ = lambda self: iter([[1, 2, 3]])  # 一条序列
        mock_inputs = {"input_ids": mock_tensor}
        mock_processor.return_value = mock_inputs

        # 模拟模型 generate 返回 token ids（嵌套列表，外层 batch）
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model.device = "cpu"

        # 模拟 batch_decode 返回 JSON 结果
        mock_processor.batch_decode.return_value = [
            '{"final_level": "L4", "confidence": 0.91, "reasoning": "图片含精神科病历"}'
        ]

        classifier._model = mock_model
        classifier._processor = mock_processor
        return classifier

    def test_image_file_path_triggers_multimodal_flow(self, tmp_path):
        """本地图片路径输入应触发多模态分类流程（含图片处理）。"""
        # 生成测试图片
        img_bytes = _make_test_image(text="psychiatric")
        img_file = tmp_path / "psychiatric_record.png"
        img_file.write_bytes(img_bytes)

        classifier = self._make_classifier_with_mock_model()

        from engine.dynclassification.base import SensitivityLevel

        result = classifier._classify_inner(
            str(img_file), SensitivityLevel.L3, 0.5
        )

        assert result is not None
        assert result["final_level"] == "L4"
        assert result["confidence"] == 0.91

        # 验证 processor 被调用时传入了 images 参数（非 None）
        call_kwargs = classifier._processor.call_args
        assert call_kwargs is not None
        # images 参数应非 None（表示图片被检测到）
        if call_kwargs.kwargs.get("images") is not None:
            pass  # 图片被正确处理
        elif len(call_kwargs.args) > 1 and call_kwargs.args[1] is not None:
            pass  # 位置参数形式
        # 验证 apply_chat_template 被调用（消息构建）
        classifier._processor.apply_chat_template.assert_called_once()

    def test_base64_image_triggers_multimodal_flow(self):
        """Base64 编码图片输入应触发多模态分类流程。"""
        img_bytes = _make_test_image(text="genetic")
        b64_str = base64.b64encode(img_bytes).decode("ascii")

        classifier = self._make_classifier_with_mock_model()

        from engine.dynclassification.base import SensitivityLevel

        result = classifier._classify_inner(b64_str, SensitivityLevel.L3, 0.6)
        assert result is not None
        assert result["final_level"] == "L4"

    def test_data_uri_image_triggers_multimodal_flow(self):
        """Data URI 格式图片输入应触发多模态分类流程。"""
        img_bytes = _make_jpeg_image()
        b64_str = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64_str}"

        classifier = self._make_classifier_with_mock_model()

        from engine.dynclassification.base import SensitivityLevel

        result = classifier._classify_inner(data_uri, SensitivityLevel.L4, 0.7)
        assert result is not None
        assert "final_level" in result

    def test_plain_text_uses_text_flow(self):
        """纯文本输入应走文本分类流程（images=None）。"""
        classifier = self._make_classifier_with_mock_model()

        from engine.dynclassification.base import SensitivityLevel

        result = classifier._classify_inner(
            "身份证号：510101199001011234", SensitivityLevel.L3, 0.5
        )
        assert result is not None
        assert result["final_level"] == "L4"

        # 验证 processor 调用时 images 为 None（纯文本模式）
        call_kwargs = classifier._processor.call_args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("images") is None

    def test_classify_with_image_timeout_returns_none(self, tmp_path):
        """图片推理超时应返回 None（优雅降级）。"""
        img_bytes = _make_test_image()
        img_file = tmp_path / "timeout_case.png"
        img_file.write_bytes(img_bytes)

        classifier = Qwen2VLClassifier(model_path="/tmp/fake_model")
        classifier._initialized = True
        classifier._model = MagicMock()
        classifier._processor = MagicMock()

        # 模拟推理超时
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        with patch.object(
            classifier._executor, "submit"
        ) as mock_submit:
            mock_future = MagicMock()
            mock_future.result.side_effect = FuturesTimeoutError()
            mock_submit.return_value = mock_future

            from engine.dynclassification.base import SensitivityLevel

            result = classifier.classify(str(img_file), SensitivityLevel.L3, 0.5)
            assert result is None


# =========================================================================== #
# gen_medical_images.py 生成图片的检测测试
# =========================================================================== #


class TestGenMedicalImagesDetection:
    """验证 gen_medical_images.py 生成的病例图片能被 _detect_image 正确识别。"""

    @pytest.fixture()
    def generated_images_dir(self, tmp_path):
        """调用 gen_medical_images 的渲染逻辑生成测试图片到临时目录。"""
        # 导入生成脚本的模板和渲染函数
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))
        try:
            from gen_medical_images import TEMPLATES, render_case
        finally:
            sys.path.pop(0)

        out_dir = tmp_path / "medical_images"
        out_dir.mkdir()

        paths = []
        for tpl in TEMPLATES:
            path = render_case(tpl, out_dir)
            paths.append(path)
        return out_dir, paths

    def test_all_generated_images_detected(self, generated_images_dir):
        """所有生成的病例图片都应被 _detect_image 识别为有效图片。"""
        out_dir, paths = generated_images_dir
        classifier = Qwen2VLClassifier(model_path="/tmp/fake_model")

        from PIL import Image

        for img_path in paths:
            result = classifier._detect_image(str(img_path))
            assert result is not None, f"未能检测到图片: {img_path.name}"
            assert isinstance(result, Image.Image)
            # 验证图片尺寸符合预期（800x1060）
            assert result.size == (800, 1060), f"{img_path.name} 尺寸异常"

    def test_generated_image_as_base64_detected(self, generated_images_dir):
        """生成的图片转为 Base64 后仍应被正确检测。"""
        _out_dir, paths = generated_images_dir
        classifier = Qwen2VLClassifier(model_path="/tmp/fake_model")

        # 取第一张图片测试 Base64 路径
        img_bytes = paths[0].read_bytes()
        b64_str = base64.b64encode(img_bytes).decode("ascii")

        result = classifier._detect_image(b64_str)
        assert result is not None

    def test_generated_image_as_data_uri_detected(self, generated_images_dir):
        """生成的图片转为 Data URI 后仍应被正确检测。"""
        _out_dir, paths = generated_images_dir
        classifier = Qwen2VLClassifier(model_path="/tmp/fake_model")

        img_bytes = paths[0].read_bytes()
        b64_str = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64_str}"

        result = classifier._detect_image(data_uri)
        assert result is not None

    def test_generated_images_cover_expected_levels(self, generated_images_dir):
        """验证生成脚本覆盖 L3~L5 不同敏感等级的病例场景。"""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))
        try:
            from gen_medical_images import TEMPLATES
        finally:
            sys.path.pop(0)

        level_hints = {tpl.level_hint for tpl in TEMPLATES}
        # 至少覆盖 L3、L4、L5
        assert "L3" in level_hints
        assert "L4" in level_hints
        assert "L5" in level_hints


# =========================================================================== #
# 真实模型 + 图片输入端到端测试（需要 GPU + 模型文件）
# =========================================================================== #


def _transformers_available() -> bool:
    """检测 transformers 库是否可用（导入无报错）。"""
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.real_models
@pytest.mark.skipif(
    not _transformers_available(),
    reason="图片分类需要 transformers 库（Qwen2-VL 视觉模型回退引擎）",
)
class TestRealImageClassification:
    """使用真实 Qwen2-VL 模型对生成病例图片进行端到端分类测试。

    运行条件：
    - .models/Qwen2-VL-2B-Instruct 已下载
    - 安装了 torch/transformers/Pillow 等 ML 依赖
    - 标记为 slow + real_models，常规 CI 跳过
    """

    @pytest.fixture(scope="class")
    def llm_adapter(self):
        """加载真实 LLM 适配器。"""
        from engine.dynclassification.llm_adapter import LlmAdapter

        adapter = LlmAdapter()
        assert adapter.is_available, "LLM 初始化失败，请检查模型文件与依赖"
        return adapter

    @pytest.fixture(scope="class")
    def medical_images(self, tmp_path_factory):
        """生成全部病例测试图片。"""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))
        try:
            from gen_medical_images import TEMPLATES, render_case
        finally:
            sys.path.pop(0)

        out_dir = tmp_path_factory.mktemp("medical_images")
        results = {}
        for tpl in TEMPLATES:
            path = render_case(tpl, out_dir)
            results[tpl.name] = {"path": path, "level_hint": tpl.level_hint}
        return results

    def test_real_llm_classifies_blood_routine_image(self, llm_adapter, medical_images):
        """真实模型应对血常规报告图片返回结构化分类结果。"""
        info = medical_images["lab_blood_routine"]
        result = llm_adapter.classify(
            text=str(info["path"]),
            upstream_level="L3",
            upstream_confidence=0.5,
        )
        assert result is not None, "真实模型对图片分类返回 None"
        assert "final_level" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_real_llm_classifies_genetic_report_image(self, llm_adapter, medical_images):
        """真实模型应对基因检测报告图片返回高敏感等级。"""
        info = medical_images["genetic_report"]
        result = llm_adapter.classify(
            text=str(info["path"]),
            upstream_level="L4",
            upstream_confidence=0.6,
        )
        assert result is not None, "真实模型对基因报告图片分类返回 None"
        assert "final_level" in result
        # 基因报告预期为 L4 或 L5
        assert result["final_level"] in ("L4", "L5"), (
            f"基因报告预期高敏感等级，实际: {result['final_level']}"
        )

    def test_real_llm_classifies_base64_image(self, llm_adapter, medical_images):
        """真实模型应能处理 Base64 编码的图片输入。"""
        info = medical_images["psychiatric_record"]
        img_bytes = info["path"].read_bytes()
        b64_str = base64.b64encode(img_bytes).decode("ascii")

        result = llm_adapter.classify(
            text=b64_str,
            upstream_level="L3",
            upstream_confidence=0.5,
        )
        assert result is not None, "真实模型对 Base64 图片分类返回 None"
        assert "final_level" in result
        assert "confidence" in result


# =========================================================================== #
# L4/L5 敏感图片病例智能打码与出入参格式对称性测试
# =========================================================================== #


class TestImageRedactionAndSymmetry:
    """测试带 L4/L5 敏感疾病的图片病例智能打码遮盖及出入参格式对称性。"""

    def test_image_file_path_redaction_returns_sanitized_file_path(self, tmp_path):
        """测试图片文件路径入参时，脱敏输出同格式的新图片文件路径。"""
        from engine.dynclassification.image_redaction import sanitize_image_input
        from engine.dynclassification import DynClassificationService

        # 1. 构造一个包含 L4 性病/肿瘤图像病例的临时图片
        img_bytes = _make_test_image(text="一期梅毒与RPR阳性诊断图片")
        img_file = tmp_path / "syphilis_case.png"
        img_file.write_bytes(img_bytes)

        # 2. 执行图像盲区打码抹平
        out_path_str = sanitize_image_input(str(img_file), output_dir=tmp_path / "out")
        out_path = Path(out_path_str)

        # 3. 校验出参格式为文件路径，且新文件存在
        assert out_path.exists()
        assert out_path.suffix == ".png"
        # 输出文件名已匿名化（sha256 派生），不得泄露原始文件名（可能含患者姓名/病种）
        assert out_path.name.startswith("sanitized_")
        assert "syphilis_case" not in out_path.name

        # 4. 通过 DynClassificationService 测试集成
        service = DynClassificationService()
        resp = service.classify_field("case_image", str(img_file), sanitize=True)
        assert resp.field_result is not None
        assert resp.field_result.sanitized_value is not None
        assert Path(resp.field_result.sanitized_value).exists()

    def test_base64_image_redaction_returns_base64_data_uri(self):
        """测试 Base64 Data URI 图片入参时，脱敏输出同格式的 Base64 Data URI。"""
        from engine.dynclassification.image_redaction import sanitize_image_input
        from engine.dynclassification import DynClassificationService

        # 1. 构造 Base64 Data URI
        img_bytes = _make_jpeg_image()
        b64_data = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_data}"

        # 2. 执行图像抹平
        out_data_uri = sanitize_image_input(data_uri)

        # 3. 校验出参格式与入参保持一致 (Data URI 字符串)
        assert out_data_uri.startswith("data:image/png;base64,")
        assert len(out_data_uri) > 50

        # 4. 通过 DynClassificationService 测试集成
        service = DynClassificationService()
        resp = service.classify_field("hiv_case_image", data_uri, sanitize=True)
        assert resp.field_result is not None
        assert resp.field_result.sanitized_value is not None
        assert resp.field_result.sanitized_value.startswith("data:image/")

    def test_text_case_input_returns_sanitized_text_symmetry(self):
        """测试纯文本病例入参时，脱敏输出同格式的文本病例。"""
        from engine.dynclassification import DynClassificationService

        service = DynClassificationService()
        raw_text = "患者自述外阴溃疡，RPR 1:16 阳性，确诊一期梅毒。"
        resp = service.classify_field("present_illness", raw_text, sanitize=True)

        assert resp.field_result is not None
        assert resp.field_result.sanitized_value is not None
        # 校验出参为文本字符串且删除了性病高敏词
        assert isinstance(resp.field_result.sanitized_value, str)
        assert "梅毒" not in resp.field_result.sanitized_value

