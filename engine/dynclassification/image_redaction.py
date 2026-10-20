"""图像病例与医学影像智能抹平/打码处理模块。
Image Case & Medical Image Smart Redaction / Sanitization Module.

===================================================================================
              图像处理执行流程 / Image Processing Execution Flow
===================================================================================

  sanitize_image_input(val_str, output_dir, boxes)
    │
    ├─① 输入类型判断
    │   ├─ 文件路径 (xxx.jpg/png/dcm) → 沙箱校验 _is_path_allowed()
    │   ├─ Base64 Data URI (data:image/...) → base64 解码
    │   └─ 均不匹配 → 返回原文或失败占位符
    │
    ├─② 安全防护
    │   ├─ 路径穿越防护: resolve() + 白名单目录前缀匹配
    │   ├─ DecompressionBomb: MAX_IMAGE_PIXELS = 25M
    │   └─ OOM防护: 超 2048x2048 自动下采样 (LANCZOS)
    │
    ├─③ 敏感区域遮挡
    │   ├─ 默认遮挡区: 头部 16% + 底部 18% (姓名/诊断/签名)
    │   └─ 自定义 boxes: [(ymin, xmin, ymax, xmax), ...] 比例或像素坐标
    │
    ├─④ 输出与清理
    │   ├─ 文件路径: sha256(文件名)[:12] 匿名命名 + 原子替换 (tmp→rename)
    │   ├─ Base64: 统一输出 PNG 格式
    │   └─ 磁盘防满: 自动清理超过 200 个旧文件
    │
    └─⑤ fail-closed: DICOM 等无法安全派生的格式返回 [IMAGE-REDACTION-FAILED]
===================================================================================
"""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import os
from pathlib import Path
from typing import Optional
import struct
import tempfile

from ..observability.logging_config import get_logger

logger = get_logger(__name__)

IMAGE_REDACTION_FAILURE = "[IMAGE-REDACTION-FAILED]"

# 常见图像文件扩展名（含 DICOM 医学影像）
IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dcm", ".dicom")

# DICOM 标签与 VR 定义 (借鉴 Go 引擎原生二进制脱敏设计)
_TAG_PATIENT_NAME = 0x00100010
_TAG_PATIENT_ID = 0x00100020
_TAG_PATIENT_BIRTH_DATE = 0x00100030
_TAG_PATIENT_SEX = 0x00100040
_TAG_PATIENT_AGE = 0x00101010
_TAG_PATIENT_ADDRESS = 0x00101040
_TAG_PATIENT_COMMENTS = 0x00104000
_TAG_REFERRING_PHYSICIAN_NAME = 0x00080090
_TAG_INSTITUTION_NAME = 0x00080080
_TAG_STUDY_DESCRIPTION = 0x00081030
_TAG_SERIES_DESCRIPTION = 0x0008103E
_TAG_STUDY_INSTANCE_UID = 0x0020000D
_TAG_SERIES_INSTANCE_UID = 0x0020000E
_TAG_SOP_INSTANCE_UID = 0x00080018
_TAG_PIXEL_DATA = 0x7FE00010

_EXPLICIT_VRS = {
    b"AE", b"AS", b"AT", b"CS", b"DA", b"DS", b"DT", b"FL", b"FD", b"IS",
    b"LO", b"LT", b"OB", b"OF", b"OW", b"PN", b"SH", b"SL", b"SQ", b"SS",
    b"ST", b"TM", b"UI", b"UL", b"UN", b"US", b"UT"
}
_LONG_VRS = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"}


def is_dicom_data(data: bytes) -> bool:
    """检查二进制数据是否具有标准 DICOM 文件格式魔数 (128-byte preamble + 'DICM')。"""
    return len(data) >= 132 and data[128:132] == b"DICM"


def anonymize_dicom(data: bytes) -> bytes:
    """对 DICOM 二进制文件流进行纯 Python 原生 Header 深度脱敏与匿名化重构。
    
    保留图像像素（PixelData），重写患者敏感 Tag 与 UID。
    """
    if not is_dicom_data(data):
        raise ValueError("Not a valid DICOM byte stream")

    out = bytearray()
    # 复制 128 字节前导码与 4 字节 DICM 标识
    out.extend(data[:132])
    offset = 132
    n = len(data)

    while offset + 4 <= n:
        group, elem = struct.unpack_from("<HH", data, offset)
        tag = (group << 16) | elem
        offset += 4

        is_explicit = False
        vr = b""
        if offset + 2 <= n and data[offset:offset+2] in _EXPLICIT_VRS:
            is_explicit = True
            vr = data[offset:offset+2]
            offset += 2

        val_len = 0
        if is_explicit:
            if vr in _LONG_VRS:
                offset += 2  # 2 字节保留空位
                if offset + 4 > n:
                    break
                val_len = struct.unpack_from("<I", data, offset)[0]
                offset += 4
            else:
                if offset + 2 > n:
                    break
                val_len = struct.unpack_from("<H", data, offset)[0]
                offset += 2
        else:
            if offset + 4 > n:
                break
            val_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4

        # 遇到 PixelData 或未定义长度 0xFFFFFFFF：直接拷贝剩余所有字节并退出
        if tag == _TAG_PIXEL_DATA or val_len == 0xFFFFFFFF:
            out.extend(struct.pack("<HH", group, elem))
            if is_explicit:
                out.extend(vr)
                if vr in _LONG_VRS:
                    out.extend(b"\x00\x00")
                    out.extend(struct.pack("<I", val_len))
                else:
                    out.extend(struct.pack("<H", val_len))
            else:
                out.extend(struct.pack("<I", val_len))
            out.extend(data[offset:])
            return bytes(out)

        if offset + val_len > n:
            val_len = n - offset

        raw_val = data[offset:offset+val_len]
        offset += val_len

        # 对敏感 Tag 进行脱敏重写
        new_val = raw_val
        val_str = raw_val.decode("utf-8", errors="ignore").rstrip("\x00 ").strip()

        if tag == _TAG_PATIENT_NAME:
            new_val = b"ANONYMOUS^PATIENT"
        elif tag == _TAG_PATIENT_ID:
            h = hashlib.sha256(val_str.encode("utf-8")).hexdigest()[:8]
            new_val = f"ANON_{h}".encode("ascii")
        elif tag == _TAG_PATIENT_BIRTH_DATE:
            if len(val_str) >= 6:
                new_val = (val_str[:6] + "01").encode("ascii")
            else:
                new_val = b"19000101"
        elif tag in (_TAG_PATIENT_ADDRESS, _TAG_REFERRING_PHYSICIAN_NAME, _TAG_INSTITUTION_NAME):
            new_val = b"***"
        elif tag == _TAG_PATIENT_AGE:
            new_val = b"000Y"
        elif tag == _TAG_PATIENT_COMMENTS:
            new_val = b""
        elif tag in (_TAG_STUDY_DESCRIPTION, _TAG_SERIES_DESCRIPTION):
            new_val = b"SANITIZED_STUDY"
        elif tag in (_TAG_STUDY_INSTANCE_UID, _TAG_SERIES_INSTANCE_UID, _TAG_SOP_INSTANCE_UID):
            uid_h = hashlib.sha256(val_str.encode("utf-8")).hexdigest()[:16]
            new_val = f"1.2.826.0.1.3680043.9.{uid_h}".encode("ascii")

        # 偶数对齐补齐
        if len(new_val) % 2 != 0:
            new_val = new_val + b" "

        # 写回 Tag
        out.extend(struct.pack("<HH", group, elem))
        if is_explicit:
            out.extend(vr)
            if vr in _LONG_VRS:
                out.extend(b"\x00\x00")
                out.extend(struct.pack("<I", len(new_val)))
            else:
                out.extend(struct.pack("<H", len(new_val)))
        else:
            out.extend(struct.pack("<I", len(new_val)))
        out.extend(new_val)

    return bytes(out)


# 文件路径长度上限：超过视为非文件路径（防超长字符串误判）
_MAX_PATH_LEN = 512

# 输出格式白名单：文件后缀 → PIL 保存格式。
_OUTPUT_FORMAT_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def _allowed_image_dirs() -> list[Path]:
    """返回允许读取图片文件的目录白名单（已 resolve 的绝对路径）。

    仅允许受信任的图片存储子目录（如 data/images, uploads, samples, tempdir）。
    """
    env = os.environ.get("PRIVACY_IMAGE_ALLOWED_DIRS")
    if env and env.strip():
        roots = [Path(p).expanduser() for p in env.split(os.pathsep) if p.strip()]
    else:
        cwd = Path.cwd()
        roots = [
            cwd / "data",
            cwd / "uploads",
            cwd / "samples",
            cwd / "medical_images",
            Path(tempfile.gettempdir()),
        ]
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(root.resolve())
        except OSError:
            continue
    return resolved


def _is_path_allowed(path: Path) -> bool:
    """校验图片路径是否位于允许的目录白名单内（沙箱校验）。

    先 resolve（跟随符号链接）再做前缀判断——既可拦截 ``../../`` 目录
    穿越，也可拦截指向沙箱外的 symlink 逃逸。resolve 失败一律拒绝。
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved.is_relative_to(root) for root in _allowed_image_dirs())


_IMAGE_EXTENSIONS_TUPLE = tuple(ext.lower() for ext in IMAGE_FILE_EXTENSIONS)


def is_image_input(val_str: str) -> bool:
    """判断输入是否为图像（文件路径或 Base64 Data URI）。"""
    if not val_str:
        return False
    stripped = val_str.strip()
    low = stripped.lower()
    return (
        len(stripped) < _MAX_PATH_LEN
        and low.endswith(_IMAGE_EXTENSIONS_TUPLE)
    ) or low.startswith(("data:image/", "image:"))


def _cleanup_old_sanitized_images(output_dir: Path, max_files: int = 200) -> None:
    """自动清理旧的打码图片，防止磁盘空间被满存（Disk Exhaustion 防护）。"""
    try:
        if not output_dir.exists():
            return
        files = sorted(output_dir.glob("sanitized_*"), key=lambda p: p.stat().st_mtime)
        if len(files) > max_files:
            for f in files[:-max_files]:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass


def sanitize_image_input(
    val_str: str,
    output_dir: str | Path = "data/sanitized_images",
    boxes: Optional[list[tuple[float, float, float, float]]] = None,
) -> str:
    """对图片病例（文件路径或 Base64 Data URI）进行图像盲区遮挡与敏感区域智能抹平，保持出参类型与入参一致。
    
    内部优化包含：
    1. with Image.open(...) 上下文管理，100% 避免 OS 文件句柄泄漏；
    2. 超高分辨率自动下采样 (2048x2048 thumbnail)，防止 10000x10000 大图耗尽 RAM/VRAM (OOM 防护)；
    3. 磁盘文件数自动淘汰轮转，防止硬盘空间爆满。
    """
    if not val_str:
        return val_str

    try:
        from PIL import Image, ImageDraw
        # 设置 DecompressionBomb 像素上限保护
        Image.MAX_IMAGE_PIXELS = 25_000_000
    except ImportError:
        logger.warning("PIL (Pillow) 未安装，拒绝输出未脱敏图像")
        return IMAGE_REDACTION_FAILURE

    val_stripped = val_str.strip()
    is_data_uri = val_stripped.lower().startswith("data:image/")
    is_image_marker = val_stripped.lower().startswith("image:")
    is_file_path = (
        len(val_stripped) < _MAX_PATH_LEN
        and any(val_stripped.lower().endswith(ext) for ext in IMAGE_FILE_EXTENSIONS)
    )

    img: Optional[Image.Image] = None
    file_path_obj: Optional[Path] = Path(val_stripped) if is_file_path else None

    # 1. 尝试从文件路径或 Base64 安全加载并立即分离文件句柄
    if file_path_obj is not None and os.path.exists(file_path_obj):
        # 沙箱校验（fail-closed）：路径 resolve 后必须位于允许的目录白名单内，
        # 拒绝 ../../ 目录穿越与指向沙箱外的 symlink 逃逸，
        # 防止任意文件读取（如 /etc 下伪装成 .png 的敏感文件）。
        if not _is_path_allowed(file_path_obj):
            logger.warning("拒绝访问沙箱外的图片路径（任意文件读取防护）")
            return IMAGE_REDACTION_FAILURE

        # 原生 DICOM 二进制文件脱敏分支 (无需依赖 PIL)
        if file_path_obj.suffix.lower() in (".dcm", ".dicom"):
            try:
                with open(file_path_obj, "rb") as df:
                    dcm_bytes = df.read()
                if not is_dicom_data(dcm_bytes):
                    logger.warning("无效的 DICOM 文件魔数，拒绝输出未脱敏图像")
                    return IMAGE_REDACTION_FAILURE
                out_dcm = anonymize_dicom(dcm_bytes)
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                _cleanup_old_sanitized_images(out_dir, max_files=200)
                name_digest = hashlib.sha256(file_path_obj.name.encode("utf-8")).hexdigest()[:12]
                out_file = out_dir / f"sanitized_{name_digest}.dcm"
                with open(out_file, "wb") as wf:
                    wf.write(out_dcm)
                return str(out_file)
            except Exception as e:
                logger.warning("DICOM 脱敏失败: %s", e)
                return IMAGE_REDACTION_FAILURE

        input_path = file_path_obj
        try:
            with Image.open(input_path) as raw_img:
                img = raw_img.convert("RGB")
        except Exception as e:
            logger.warning("无法打开图片文件，拒绝输出未脱敏图像: %s", e)
            return IMAGE_REDACTION_FAILURE
    elif is_data_uri:
        try:
            parts = val_stripped.split(",", 1)
            if len(parts) == 2:
                img_bytes = base64.b64decode(parts[1])
                with Image.open(BytesIO(img_bytes)) as raw_img:
                    img = raw_img.convert("RGB")
        except Exception as e:
            logger.warning("无法解码 Base64 图像，拒绝输出未脱敏图像: %s", e)
            return IMAGE_REDACTION_FAILURE

    if img is None:
        # 只要输入看起来是图片，就不能把未知格式当普通文本返回。
        return IMAGE_REDACTION_FAILURE if is_file_path or is_data_uri or is_image_marker else val_str

    # 2. 超高分辨率内存与显存保护 (OOM Prevention)
    # 若图像分辨率超 2048x2048，自动高质量下采样缩放
    max_dim = 2048
    if img.width > max_dim or img.height > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    width, height = img.size
    draw = ImageDraw.Draw(img)

    # 3. 图像盲区遮挡绘制
    default_boxes = [
        (0.0, 0.0, 1.0, 0.16),    # 头部个人身份姓名/身份证遮挡区
        (0.0, 0.82, 1.0, 1.0),    # 底部敏感诊断/签名遮挡区
    ]
    target_boxes = boxes if boxes else default_boxes

    for box in target_boxes:
        ymin, xmin, ymax, xmax = box
        if all(0.0 <= c <= 1.0 for c in (ymin, xmin, ymax, xmax)):
            x0 = int(xmin * width)
            y0 = int(ymin * height)
            x1 = int(xmax * width)
            y1 = int(ymax * height)
        else:
            x0, y0, x1, y1 = int(xmin), int(ymin), int(xmax), int(ymax)

        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

    # 4. 输出格式对齐与磁盘防满清理
    if is_file_path and file_path_obj is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_sanitized_images(out_dir, max_files=200)

        # 输出文件名脱敏：原始文件名可能包含患者姓名/病种等敏感信息，
        # 使用 sha256(原始文件名)[:12] 派生确定性的匿名文件名。
        name_digest = hashlib.sha256(file_path_obj.name.encode("utf-8")).hexdigest()[:12]
        out_suffix = file_path_obj.suffix.lower()
        out_file = out_dir / f"sanitized_{name_digest}{out_suffix}"
        if out_file.is_symlink():
            img.close()
            logger.warning("拒绝覆盖符号链接形式的图片输出: %s", out_file.name)
            return IMAGE_REDACTION_FAILURE
        # 格式白名单派生：仅允许可安全派生的格式（JPG→JPEG、TIF→TIFF 等）；
        # 无法安全派生的格式（含 DICOM .dcm/.dicom）返回失败占位符，不崩溃。
        fmt = _OUTPUT_FORMAT_MAP.get(out_suffix)
        if fmt is None:
            img.close()
            logger.warning(
                "无法安全派生输出格式（后缀 %s），拒绝输出未脱敏图像", out_suffix
            )
            return IMAGE_REDACTION_FAILURE
        # 在同一目录创建临时文件并原子替换，避免并发请求读到半成品。
        fd, tmp_name = tempfile.mkstemp(prefix=".sanitized_", suffix=out_file.suffix, dir=out_dir)
        os.close(fd)
        tmp_file = Path(tmp_name)
        try:
            img.save(tmp_file, format=fmt)
            tmp_file.replace(out_file)
            return str(out_file)
        finally:
            img.close()
            tmp_file.unlink(missing_ok=True)
    elif is_data_uri:
        buf = BytesIO()
        img.save(buf, format="PNG")
        img.close()
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    img.close()
    return val_str
