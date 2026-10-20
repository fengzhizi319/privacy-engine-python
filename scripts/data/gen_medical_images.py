#!/usr/bin/env python3
"""生成数据分类分级测试用的中文病例图片。

本脚本使用 Pillow 在本地渲染多张「模拟医院病例 / 检验报告」图片，
覆盖 L1~L5 不同敏感等级，用于前端测试控制台「分类分级」功能对
多模态（图片）分类能力的可视化验证。

生成的 PNG 输出到 ``console/web/src/assets/medical/``，由 Vite 在
构建时打包为静态资源；前端通过 ``import.meta.glob`` 引入并展示缩略图，
用户点选后转为 base64 data URI 注入 ``/v1/privacy/classify/field`` 请求体。

字体策略（按优先级回退）：
    1. 环境变量 ``PLA_CJK_FONT`` 指定的路径；
    2. 项目根 ``.fonts/NotoSansSC.ttf``（可由脚本自动下载缓存）；
    3. 常见系统 CJK 字体路径；
    4. 全部缺失时退化为 PIL 默认字体（中文会显示为方块，仅用于占位）。

用法::

    python scripts/data/gen_medical_images.py            # 生成全部
    python scripts/data/gen_medical_images.py --list     # 列出可用模板
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "console" / "web" / "src" / "assets" / "medical"
FONT_CACHE = PROJECT_ROOT / ".fonts" / "NotoSansSC.ttf"

# 字体下载地址（开源 Noto Sans CJK SC，仅在本地缓存缺失时拉取）。
FONT_URL = (
    "https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
)

# 常见系统 CJK 字体候选路径（按平台覆盖 Linux / macOS）。
_SYSTEM_CJK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
]

# 图片基础尺寸（宽 x 高），模拟 A4 报告的近似比例。
PAGE_W, PAGE_H = 800, 1060


# --------------------------------------------------------------------------- #
# 字体加载
# --------------------------------------------------------------------------- #


def _download_font(target: Path) -> bool:
    """尝试下载 Noto Sans SC 到 ``target``，成功返回 True。"""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[font] downloading CJK font -> {target}")
        urllib.request.urlretrieve(FONT_URL, target)  # nosec - 固定可信源
        return target.exists() and target.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001 - 网络异常一律降级
        print(f"[font] download failed: {exc}")
        return False


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """按优先级加载 CJK 字体，全部失败时退化为默认字体。"""
    candidates: List[Path] = []

    env_font = os.environ.get("PLA_CJK_FONT")
    if env_font:
        candidates.append(Path(env_font))
    candidates.append(FONT_CACHE)
    candidates.extend(Path(p) for p in _SYSTEM_CJK_FONTS)

    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:  # noqa: BLE001 - 字体损坏则继续尝试下一个
                continue

    # 缓存与系统字体都缺失时，尝试下载一次。
    if _download_font(FONT_CACHE):
        try:
            return ImageFont.truetype(str(FONT_CACHE), size=size)
        except Exception:  # noqa: BLE001
            pass

    print("[font] WARNING: no CJK font found, falling back to default bitmap font")
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# 病例模板定义
# --------------------------------------------------------------------------- #


@dataclass
class CaseTemplate:
    """单张病例图片的渲染模板。"""

    name: str                       # 输出文件名（不含扩展名）
    title: str                      # 报告标题
    level_hint: str                 # 预期敏感等级（仅用于说明）
    accent: Tuple[int, int, int]    # 主题色（标题栏）
    fields: List[Tuple[str, str]]   # 结构化字段（label, value）
    body: List[str] = field(default_factory=list)  # 正文段落
    footer: str = ""                # 页脚说明


# 覆盖 L3~L5 的典型医疗场景；颜色仅作视觉区分。
TEMPLATES: List[CaseTemplate] = [
    CaseTemplate(
        name="lab_blood_routine",
        title="门诊检验报告单 · 血常规",
        level_hint="L3",
        accent=(37, 99, 235),
        fields=[
            ("姓名", "张伟"),
            ("性别 / 年龄", "男 / 28 岁"),
            ("身份证号", "110101199001011234"),
            ("联系电话", "13800001111"),
            ("送检科室", "内科门诊"),
            ("样本类型", "静脉血"),
        ],
        body=[
            "检验项目        结果        参考范围",
            "白细胞 WBC      6.8         4.0-10.0  ×10^9/L",
            "红细胞 RBC      4.9         4.0-5.5   ×10^12/L",
            "血红蛋白 HGB    152         120-160   g/L",
            "血小板 PLT      210         100-300   ×10^9/L",
        ],
        footer="本报告仅对本次送检样本负责，结果供临床参考。",
    ),
    CaseTemplate(
        name="outpatient_record",
        title="门诊病历记录",
        level_hint="L3",
        accent=(13, 148, 136),
        fields=[
            ("姓名", "李娜"),
            ("性别 / 年龄", "女 / 34 岁"),
            ("手机号", "13900002222"),
            ("就诊卡号", "MZ20240618003"),
            ("科室", "呼吸内科"),
        ],
        body=[
            "主诉：咳嗽、咽痛 3 天，伴低热。",
            "现病史：患者 3 天前受凉后出现阵发性干咳，",
            "        体温最高 37.8℃，无胸闷气促。",
            "既往史：否认高血压、糖尿病史。",
            "诊断：急性上呼吸道感染。",
            "处理：对症退热，多饮水，必要时复诊。",
        ],
        footer="医师签名：王医生    日期：2024-06-18",
    ),
    CaseTemplate(
        name="discharge_summary",
        title="住院出院小结",
        level_hint="L4",
        accent=(220, 38, 38),
        fields=[
            ("姓名", "王强"),
            ("性别 / 年龄", "男 / 41 岁"),
            ("住院号", "ZY20240512077"),
            ("身份证号", "310101198301015678"),
            ("入院日期", "2024-05-12"),
            ("出院日期", "2024-05-26"),
            ("入院诊断", "2 型糖尿病伴酮症"),
        ],
        body=[
            "入院情况：患者因“多饮多尿 1 月，恶心呕吐 2 天”入院，",
            "          入院时血糖 21.3 mmol/L，尿酮体 (+++)。",
            "诊疗经过：予以胰岛素强化降糖、补液纠酮等治疗，",
            "          复查血糖平稳，尿酮转阴。",
            "出院诊断：2 型糖尿病伴酮症酸中毒（已纠正）。",
            "出院医嘱：规律用药，糖尿病饮食，1 个月后内分泌科复诊。",
        ],
        footer="主管医师：赵医生    科主任：钱主任",
    ),
    CaseTemplate(
        name="psychiatric_record",
        title="精神科住院病历",
        level_hint="L4",
        accent=(124, 58, 237),
        fields=[
            ("姓名", "刘洋"),
            ("性别 / 年龄", "女 / 29 岁"),
            ("住院号", "JS20240301021"),
            ("联系电话", "13600004444"),
            ("科室", "精神科"),
            ("入院诊断", "精神分裂症"),
        ],
        body=[
            "主诉：反复言行紊乱、幻听 2 年，加重 1 周。",
            "精神检查：意识清楚，接触被动，存在评论性幻听，",
            "          情感淡漠，自知力缺乏。",
            "诊疗计划：予抗精神病药物系统治疗，配合心理康复训练，",
            "          注意评估自伤及冲动风险。",
        ],
        footer="本病历涉及个人精神健康隐私，须严格保密。",
    ),
    CaseTemplate(
        name="genetic_report",
        title="基因检测报告",
        level_hint="L5",
        accent=(190, 24, 93),
        fields=[
            ("姓名", "陈杰"),
            ("性别 / 年龄", "男 / 38 岁"),
            ("样本编号", "GENE-2024-0087"),
            ("身份证号", "440101198601019012"),
            ("检测项目", "遗传性肿瘤相关基因突变检测"),
        ],
        body=[
            "检测结果：",
            "  BRCA1 基因    c.5266dupC    致病性突变（阳性）",
            "  TP53 基因     未检出致病突变",
            "  MLH1 基因     未检出致病突变",
            "结论：检出 BRCA1 致病性突变，提示遗传性乳腺癌-",
            "      卵巢癌综合征风险升高，建议遗传咨询与随访。",
        ],
        footer="基因信息属个人敏感生物信息，须最高级别保护。",
    ),
    CaseTemplate(
        name="hiv_lab_report",
        title="检验报告单 · 感染免疫",
        level_hint="L5",
        accent=(154, 52, 18),
        fields=[
            ("姓名", "杨敏"),
            ("性别 / 年龄", "女 / 45 岁"),
            ("样本类型", "血清"),
            ("联系电话", "13400006666"),
            ("送检科室", "感染科"),
        ],
        body=[
            "检验项目              结果        参考",
            "HIV 抗体初筛          阳性        阴性",
            "HIV 确证试验          阳性        阴性",
            "CD4+ T 淋巴细胞       320         500-1600  /μL",
            "梅毒螺旋体抗体        阴性        阴性",
        ],
        footer="本结果涉及重大传染病隐私，仅限授权人员查阅。",
    ),
]


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #


def render_case(tpl: CaseTemplate, out_dir: Path) -> Path:
    """渲染单张病例图片并保存为 PNG，返回输出路径。"""
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)

    font_title = load_font(34)
    font_label = load_font(22)
    font_body = load_font(21)
    font_small = load_font(16)

    # 顶部标题栏
    bar_h = 84
    draw.rectangle([0, 0, PAGE_W, bar_h], fill=tpl.accent)
    draw.text((32, 24), tpl.title, font=font_title, fill="white")

    # 右上角敏感等级提示（仅测试用途的水印说明）
    hint = f"测试样例 · 预期 {tpl.level_hint}"
    draw.text((PAGE_W - 220, 30), hint, font=font_small, fill=(255, 255, 255, 220))

    # 结构化字段区（两列布局）
    y = bar_h + 28
    col_w = (PAGE_W - 64) // 2
    for idx, (label, value) in enumerate(tpl.fields):
        col = idx % 2
        x = 32 + col * col_w
        draw.text((x, y), f"{label}：", font=font_label, fill=(100, 116, 139))
        # 值紧随标签后绘制（近似对齐）
        draw.text((x + 130, y), value, font=font_label, fill=(15, 23, 42))
        if col == 1:
            y += 40
    if len(tpl.fields) % 2 == 1:
        y += 40

    # 分隔线
    y += 6
    draw.line([(32, y), (PAGE_W - 32, y)], fill=(226, 232, 240), width=2)
    y += 22

    # 正文段落
    for line in tpl.body:
        draw.text((32, y), line, font=font_body, fill=(30, 41, 59))
        y += 34

    # 页脚
    if tpl.footer:
        draw.line([(32, PAGE_H - 70), (PAGE_W - 32, PAGE_H - 70)], fill=(226, 232, 240), width=2)
        draw.text((32, PAGE_H - 52), tpl.footer, font=font_small, fill=(148, 163, 184))

    out_path = out_dir / f"{tpl.name}.png"
    img.save(out_path, format="PNG", optimize=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成分类分级测试用病例图片")
    parser.add_argument("--list", action="store_true", help="仅列出可用模板")
    parser.add_argument(
        "--out", default=str(OUTPUT_DIR), help="输出目录（默认 console/web/src/assets/medical）"
    )
    args = parser.parse_args()

    if args.list:
        for tpl in TEMPLATES:
            print(f"{tpl.name:24s} {tpl.level_hint:4s} {tpl.title}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen] output dir: {out_dir}")
    for tpl in TEMPLATES:
        path = render_case(tpl, out_dir)
        size_kb = path.stat().st_size / 1024
        print(f"[gen] {tpl.level_hint} {tpl.name:24s} -> {path.name} ({size_kb:.1f} KB)")
    print(f"[gen] done, {len(TEMPLATES)} images generated.")


if __name__ == "__main__":
    main()
