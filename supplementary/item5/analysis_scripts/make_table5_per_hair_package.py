from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_table5_submission_package import (
    SRC, OUT, METHODS, SELECTED_SEEDS, SELECTED_HAIR,
    read_csv, make_supplement, make_docx, safe_num, get_font,
)


def parse_json(text, default):
    try:
        return json.loads(text)
    except Exception:
        return default


def safe_tag(text):
    return text.replace(" ", "_").replace("/", "-")


def image_for_row(row):
    p = Path(row["image_path"])
    with Image.open(p) as im0:
        im = im0.convert("RGB")
        scale = float(row.get("source_scale_to_hr") or 1.0)
        if abs(scale - 1.0) > 1e-6:
            im = im.resize((3840, 2160), Image.Resampling.BICUBIC)
        roi = parse_json(row["roi_coordinates"], [0, 0, 512, 512])
        x, y, w, h = [int(round(float(v))) for v in roi]
        return im.crop((x, y, x + w, y + h))


def make_images(raw_rows):
    plain = OUT / "02_论文图片候选" / "原始无毛发标注"
    lines = OUT / "02_论文图片候选" / "仅毛发标注"
    montages = OUT / "02_论文图片候选" / "对比拼图"
    for d in (plain, lines, montages):
        d.mkdir(parents=True, exist_ok=True)
    montage_groups = {(s, SELECTED_HAIR[s]): [] for s in SELECTED_SEEDS}
    method_pairs = 0
    for row in raw_rows:
        method = row["method"]
        seed = int(row["seed_id"])
        hair = row["hair_id"]
        crop = image_for_row(row)
        annotated = crop.copy()
        pts = parse_json(row["display_polyline_vertices"], [])
        if method == "HR":
            pts = parse_json(row.get("hr_base_reference_anchors", ""), pts)
        xy = [(float(x), float(y)) for x, y in pts]
        d = ImageDraw.Draw(annotated)
        # ImageJ-like segmented line: straight segments and small vertex handles.
        if len(xy) >= 2:
            d.line(xy, fill=(255, 255, 0), width=4)
            for x, y in xy:
                d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        length = safe_num(row.get("length_um", ""), 1).replace("—", "NA")
        base = f"seed{seed:02d}_{hair}_{safe_tag(method)}_L{length}um"
        crop.save(plain / f"{base}_original.png")
        annotated.save(lines / f"{base}_hairline.png")
        method_pairs += 1
        key = (seed, hair)
        if key in montage_groups:
            a = crop.resize((256, 256), Image.Resampling.LANCZOS)
            b = annotated.resize((256, 256), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (512, 292), "white")
            panel.paste(a, (0, 36)); panel.paste(b, (256, 36))
            dd = ImageDraw.Draw(panel)
            dd.text((5, 7), method, fill=(0, 0, 0), font=get_font(22))
            dd.text((170, 8), "原图 | 折线", fill=(55, 55, 55), font=get_font(16))
            montage_groups[key].append(panel)
    montage_paths = []
    for (seed, hair), panels in montage_groups.items():
        montage = Image.new("RGB", (512, 292 * len(panels)), "white")
        for i, panel in enumerate(panels):
            montage.paste(panel, (0, i * 292))
        p = montages / f"seed{seed:02d}_{hair}_comparison.png"
        montage.save(p)
        montage_paths.append(p)
    return method_pairs, montage_paths


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for name in [
        "01_中文版Word",
        "02_论文图片候选/原始无毛发标注",
        "02_论文图片候选/仅毛发标注",
        "02_论文图片候选/对比拼图",
        "03_审稿补充材料",
        "04_原始数据留存",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    raw_rows = read_csv(SRC / "raw" / "auto_hair_measurements.csv")
    summary_rows = read_csv(SRC / "results" / "table5_hair_measurement.csv")
    n_pairs, montage_paths = make_images(raw_rows)
    make_supplement(raw_rows, summary_rows)
    shutil.copytree(SRC, OUT / "04_原始数据留存" / SRC.name)
    (OUT / "04_原始数据留存" / "来源说明.txt").write_text(
        "完整留存 table5_hair_measurement_auto_60_guided_salvage_v2 原始输出，未修改。\n"
        "比例尺：1 pixel = 1.722488 μm；20 颗种子、60 根固定毛发。\n", encoding="utf-8")
    doc = make_docx(summary_rows, montage_paths[0])
    (OUT / "README_四类文件说明.txt").write_text(
        "Table 5 毛发长度实验整理包（60 根固定毛）\n\n"
        "01_中文版Word：中文版论文风格 Word，含 Table 5 和代表性图。\n"
        "02_论文图片候选：每根毛发的 512×512 局部 ROI。共 60 根×9 个方法×2 类图，"
        "文件名包含 seed、hair_id、方法和该图的长度（μm）。原始图不带标注；折线图只显示 3–5 个直线段和顶点。\n"
        "03_审稿补充材料：方法汇总、逐根毛数据、QC、HR 手工线一致性和严格/引导结果统计。\n"
        "04_原始数据留存：完整保留 v2 原始输出。\n\n"
        "HR 图使用 hr_base 手工锚点；其他方法使用 HR 线作为 shape prior 后独立跟踪。比例尺：1 pixel = 1.722488 μm。\n",
        encoding="utf-8")
    print(f"Created {OUT}")
    print(f"Per-hair method pairs: {n_pairs}")
    print(f"Word: {doc}")


if __name__ == "__main__":
    main()
