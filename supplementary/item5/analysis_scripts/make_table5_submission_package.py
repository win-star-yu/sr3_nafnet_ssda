from __future__ import annotations

import csv
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


ROOT = Path(r"./redacted/path")
SRC = ROOT / "revision_jobs" / "table5_hair_measurement_auto_60_guided_salvage_v2"
OUT = ROOT / "revision_jobs" / "table5_hair_measurement_submission_package_20260921"
SCALE = 1.722488
METHODS = [
    "HR",
    "Original LR",
    "Bicubic",
    "HAN",
    "SwinIR",
    "SRFormer",
    "Real-ESRGAN",
    "SR3",
    "Ours",
]
METHOD_DIR = {
    "HR": "01_HR",
    "Original LR": "09_LR_synthetic_480x270",
    "Bicubic": "02_Bicubic",
    "HAN": "03_HAN",
    "SwinIR": "04_SwinIR",
    "SRFormer": "05_SRFormer",
    "Real-ESRGAN": "06_RealESRGAN",
    "SR3": "07_SR3",
    "Ours": "08_Ours",
}
SELECTED_SEEDS = [1, 10, 20]
SELECTED_HAIR = {1: "00001_hair01", 10: "00010_hair01", 20: "00020_hair01"}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_json_field(text, default):
    try:
        return json.loads(text)
    except Exception:
        return default


def safe_num(x, digits=2):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


def get_font(size=18):
    for p in [
        r"./redacted/path",
        r"./redacted/path",
        r"./redacted/path",
    ]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def prepare_dirs():
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


def make_candidates(raw_rows):
    by_key = {(r["method"], r["hair_id"]): r for r in raw_rows}
    montage_files = []
    label_font = get_font(22)
    for seed in SELECTED_SEEDS:
        hair = SELECTED_HAIR[seed]
        panels = []
        for method in METHODS:
            r = by_key[(method, hair)]
            src_path = Path(r["image_path"])
            if not src_path.exists():
                continue
            roi = parse_json_field(r["roi_coordinates"], [0, 0, 512, 512])
            x, y, w, h = [int(round(float(v))) for v in roi]
            with Image.open(src_path) as im:
                im = im.convert("RGB")
                # Keep the exact measurement ROI; avoid adding annotation text to either candidate image.
                crop = im.crop((x, y, x + w, y + h))
            pts = parse_json_field(r["display_polyline_vertices"], [])
            annotated = crop.copy()
            d = ImageDraw.Draw(annotated)
            xy = [(float(px), float(py)) for px, py in pts]
            if len(xy) >= 2:
                # Straight segments, matching an ImageJ segmented-line measurement.
                d.line(xy, fill=(235, 35, 35), width=4)
                # ImageJ-like endpoint handles; no text, scale bar, or method name on the image.
                for px, py in [xy[0], xy[-1]]:
                    d.ellipse((px - 4, py - 4, px + 4, py + 4), outline=(235, 35, 35), width=2)
            base = f"seed{seed:02d}_{hair}_{method.replace(' ', '_').replace('/', '-') }"
            original_path = OUT / "02_论文图片候选" / "原始无毛发标注" / f"{base}_original.png"
            line_path = OUT / "02_论文图片候选" / "仅毛发标注" / f"{base}_hairline.png"
            crop.save(original_path)
            annotated.save(line_path)
            # Candidate montage uses a compact pair for each method.
            a = crop.resize((256, 256), Image.Resampling.LANCZOS)
            b = annotated.resize((256, 256), Image.Resampling.LANCZOS)
            pair = Image.new("RGB", (512, 292), "white")
            pair.paste(a, (0, 36))
            pair.paste(b, (256, 36))
            pd = ImageDraw.Draw(pair)
            pd.text((5, 7), method, fill=(0, 0, 0), font=label_font)
            pd.text((170, 8), "原图 | 折线", fill=(55, 55, 55), font=get_font(16))
            panels.append(pair)
        if panels:
            montage = Image.new("RGB", (512, 292 * len(panels)), "white")
            for i, panel in enumerate(panels):
                montage.paste(panel, (0, i * 292))
            mp = OUT / "02_论文图片候选" / "对比拼图" / f"seed{seed:02d}_{hair}_comparison.png"
            montage.save(mp)
            montage_files.append(mp)
    return montage_files


def seed_bbox(hr_path: Path):
    """Find a common seed crop from HR only; the same crop is used for every method."""
    im = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
    if im is None:
        return (0, 0, 3840, 2160)
    b, g, r = cv2.split(im)
    mask = ((r > 40) & (g > 30) & (r > 1.12 * b) & (g > 1.08 * b)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return (0, 0, im.shape[1], im.shape[0])
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = [int(v) for v in stats[i]]
    margin = 150
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(im.shape[1], x + w + margin), min(im.shape[0], y + h + margin)
    return (x0, y0, x1, y1)


def make_full_seed_candidates(raw_rows):
    """Write all 20 seeds for every method, with and without the measured polylines."""
    by_seed_method = {}
    for r in raw_rows:
        by_seed_method[(r["method"], int(r["seed_id"]))] = r
    out_plain = OUT / "02_论文图片候选" / "全种子_原始无毛发标注"
    out_line = OUT / "02_论文图片候选" / "全种子_仅毛发折线标注"
    out_plain.mkdir(parents=True, exist_ok=True)
    out_line.mkdir(parents=True, exist_ok=True)
    font = get_font(20)
    count = 0
    for seed in range(1, 21):
        hr_row = by_seed_method[("HR", seed)]
        bx0, by0, bx1, by1 = seed_bbox(Path(hr_row["image_path"]))
        for method in METHODS:
            rows = [r for r in raw_rows if r["method"] == method and int(r["seed_id"]) == seed]
            if not rows:
                continue
            first = rows[0]
            src_path = Path(first["image_path"])
            if not src_path.exists():
                continue
            with Image.open(src_path) as im0:
                im = im0.convert("RGB")
                scale = float(first.get("source_scale_to_hr") or 1.0)
                if abs(scale - 1.0) > 1e-6:
                    im = im.resize((3840, 2160), Image.Resampling.BICUBIC)
                # Keep a common HR-coordinate field of view for all methods.
                crop = im.crop((bx0, by0, bx1, by1))
            annotated = crop.copy()
            d = ImageDraw.Draw(annotated)
            for r in rows:
                pts = parse_json_field(r["display_polyline_vertices"], [])
                if method == "HR":
                    # The HR panel uses the recorded manual ImageJ-style anchor polyline.
                    pts = parse_json_field(r.get("hr_base_reference_anchors", ""), pts)
                roi = parse_json_field(r["roi_coordinates"], [0, 0, 512, 512])
                if len(pts) < 2:
                    continue
                ox, oy = float(roi[0]), float(roi[1])
                xy = [(float(px) + ox - bx0, float(py) + oy - by0) for px, py in pts]
                d.line(xy, fill=(235, 35, 35), width=4)
                for px, py in (xy[0], xy[-1]):
                    d.ellipse((px - 4, py - 4, px + 4, py + 4), outline=(235, 35, 35), width=2)
            # Reduce only for convenient paper selection; geometry is drawn before resizing.
            max_w = 1600
            if crop.width > max_w:
                nh = int(round(crop.height * max_w / crop.width))
                crop = crop.resize((max_w, nh), Image.Resampling.LANCZOS)
                annotated = annotated.resize((max_w, nh), Image.Resampling.LANCZOS)
            base = f"seed{seed:02d}_{method.replace(' ', '_').replace('/', '-') }"
            crop.save(out_plain / f"{base}_original.png")
            annotated.save(out_line / f"{base}_hairline.png")
            count += 1
    return count


def make_supplement(raw_rows, summary_rows):
    supp = OUT / "03_审稿补充材料"
    summary_fields = [
        "Method", "N", "Success_rate_pct", "Paired_N", "Hair_MAE_um", "Hair_MRE_pct",
        "RMSE_um", "Bias_um", "SD_difference_um", "LoA_lower_um", "LoA_upper_um", "Pearson_r",
    ]
    write_csv(supp / "table5_method_summary.csv", summary_rows, summary_fields)
    per_fields = [
        "method", "seed_id", "hair_id", "measurable", "failure_reason", "length_um",
        "confidence_score", "mean_distance_to_hr_prior_px", "max_distance_to_hr_prior_px",
        "endpoint_offset_root_px", "endpoint_offset_tip_px", "path_curvature_score",
        "branch_crossing_flag", "corridor_violation_flag", "branch_detector_raw_flag", "qc_warning",
    ]
    write_csv(supp / "table5_per_hair_measurements.csv", raw_rows, per_fields)
    qc_fields = [
        "method", "seed_id", "hair_id", "measurable", "failure_reason", "mean_distance_to_hr_prior_px",
        "max_distance_to_hr_prior_px", "endpoint_offset_root_px", "endpoint_offset_tip_px",
        "path_curvature_score", "branch_crossing_flag", "corridor_violation_flag", "branch_detector_raw_flag",
        "branch_qc_status", "qc_warning",
    ]
    write_csv(supp / "table5_qc.csv", raw_rows, qc_fields)
    shutil.copy2(SRC / "results" / "manual_vs_auto_hr.csv", supp / "manual_vs_auto_hr.csv")
    shutil.copy2(SRC / "results" / "strict_vs_guided_counts.csv", supp / "strict_vs_guided_counts.csv")
    (supp / "README_补充材料.txt").write_text(
        "Table 5 补充材料\n\n"
        "数据包含 9 个图像来源（HR、Original LR、Bicubic、HAN、SwinIR、SRFormer、Real-ESRGAN、SR3、Ours），"
        "20 颗种子、每颗 3 根固定毛发，共 60 根毛。各方法使用同一组 ROI 和 HR 手工中心线先验，"
        "但在各自图像中独立跟踪中心线。比例尺为 1 pixel = 1.722488 μm。\n\n"
        "table5_method_summary.csv：方法级均值、误差和一致性统计；\n"
        "table5_per_hair_measurements.csv：逐根毛最终测量值；\n"
        "table5_qc.csv：逐根毛的距离、端点、曲率、分叉和走廊 QC；\n"
        "manual_vs_auto_hr.csv：HR 手工折线与自动 HR 线的一致性；\n"
        "strict_vs_guided_counts.csv：严格分支检测与走廊约束后结果数量。\n"
        "\n表中长度误差均以 μm 计算，最终长度采用 raw centerline；展示折线仅用于图示。\n",
        encoding="utf-8",
    )


def wtag(tag, text=None, attrs=""):
    if text is None:
        return f"<{tag}{attrs}/>"
    return f"<{tag}{attrs}>{escape(str(text))}</{tag}>"


def make_docx(summary_rows, montage_path):
    doc_dir = OUT / "01_中文版Word"
    media_dir = doc_dir / "_word_media"
    media_dir.mkdir(exist_ok=True)
    image_name = montage_path.name
    shutil.copy2(montage_path, media_dir / image_name)
    body = []

    def para(text="", bold=False, size=24, align=None):
        rpr = f'<w:rPr><w:sz w:val="{size}"/>' + ("<w:b/>" if bold else "") + "</w:rPr>"
        ppr = f'<w:pPr><w:jc w:val="{align}"/></w:pPr>' if align else ""
        return f"<w:p>{ppr}<w:r>{rpr}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"

    body.append(para("表5  不同超分辨率方法的毛发长度定量测量", True, 32, "center"))
    body.append(para("在同一组 20 颗水稻种子上选取 60 根固定毛发进行测量。各方法使用相同 ROI 和 HR 手工中心线先验，并在各自图像中独立跟踪中心线。比例尺：1 pixel = 1.722488 μm。", False, 21))
    body.append(para("表5  毛发长度测量结果", True, 24))

    headers = ["方法", "N", "成功率(%)", "MAE(μm)", "MRE(%)", "RMSE(μm)", "Bias(μm)", "SD(μm)", "95% LoA(μm)", "Pearson r"]
    rows_xml = []
    rows_xml.append("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{escape(h)}</w:t></w:r></w:p></w:tc>" for h in headers) + "</w:tr>")
    zh = {"Original LR": "原始LR", "Bicubic": "Bicubic", "HAN": "HAN", "SwinIR": "SwinIR", "SRFormer": "SRFormer", "Real-ESRGAN": "Real-ESRGAN", "SR3": "SR3", "Ours": "Ours", "HR": "HR"}
    for r in summary_rows:
        vals = [
            zh.get(r["Method"], r["Method"]), r["N"], safe_num(r["Success_rate_pct"], 1),
            safe_num(r["Hair_MAE_um"]), safe_num(r["Hair_MRE_pct"]), safe_num(r["RMSE_um"]),
            safe_num(r["Bias_um"]), safe_num(r["SD_difference_um"]),
            f"[{safe_num(r['LoA_lower_um'])}, {safe_num(r['LoA_upper_um'])}]", safe_num(r["Pearson_r"], 4),
        ]
        rows_xml.append("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{escape(str(v))}</w:t></w:r></w:p></w:tc>" for v in vals) + "</w:tr>")
    body.append("<w:tbl><w:tblPr><w:tblBorders><w:top w:val=\"single\"/><w:left w:val=\"single\"/><w:bottom w:val=\"single\"/><w:right w:val=\"single\"/><w:insideH w:val=\"single\"/><w:insideV w:val=\"single\"/></w:tblBorders></w:tblPr>" + "".join(rows_xml) + "</w:tbl>")
    body.append(para("注：MAE、RMSE、Bias、SD 和 LoA 的单位为 μm；LoA 为 95% limits of agreement；成功率按 60 根固定毛发统计。", False, 18))

    with Image.open(montage_path) as im:
        iw, ih = im.size
    cx = int(9.0 * 914400)
    cy = int(cx * ih / iw)
    body.append(para("图5  代表性毛发 ROI：左侧为原图，右侧为基于 HR 先验的 3–5 点直线折线标注。", True, 22))
    drawing = (
        '<w:p><w:r><w:drawing><wp:inline xmlns:wp="htt./redacted/path" '
        'xmlns:a="htt./redacted/path" '
        'xmlns:pic="htt./redacted/path">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="1" name="Table5 hair candidates"/>'
        '<a:graphic><a:graphicData uri="htt./redacted/path">'
        '<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="table5_candidates.png"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="rId4" xmlns:r="htt./redacted/path"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    )
    body.append(drawing)
    body.append(para("图中折线仅用于展示毛发中心线，原始无标注图和仅折线图均另存于“论文图片候选”目录。", False, 18))

    sect = '<w:sectPr><w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr>'
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="htt./redacted/path">'
                '<w:body>' + "".join(body) + sect + '</w:body></w:document>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="htt./redacted/path">'
            '<Relationship Id="rId1" Type="htt./redacted/path" Target="word/document.xml"/>'
            '</Relationships>')
    doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="htt./redacted/path">'
                '<Relationship Id="rId4" Type="htt./redacted/path" Target="media/' + image_name + '"/>'
                '</Relationships>')
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<w:styles xmlns:w="htt./redacted/path">'
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style></w:styles>')
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="htt./redacted/path">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Default Extension="png" ContentType="image/png"/>'
                     '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                     '</Types>')
    package_path = doc_dir / "Table5_毛发长度定量测量_中文版.docx"
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.write(media_dir / image_name, "word/media/" + image_name)
    shutil.rmtree(media_dir)
    return package_path


def main():
    if not SRC.exists():
        raise SystemExit(f"Source folder not found: {SRC}")
    prepare_dirs()
    raw_rows = read_csv(SRC / "raw" / "auto_hair_measurements.csv")
    summary_rows = read_csv(SRC / "results" / "table5_hair_measurement.csv")
    montage_files = make_candidates(raw_rows)
    full_seed_count = make_full_seed_candidates(raw_rows)
    if not montage_files:
        raise SystemExit("No candidate montages were created")
    make_supplement(raw_rows, summary_rows)
    # Preserve the complete latest audit output without altering it.
    shutil.copytree(SRC, OUT / "04_原始数据留存" / SRC.name)
    (OUT / "04_原始数据留存" / "来源说明.txt").write_text(
        "本目录完整留存 table5_hair_measurement_auto_60_guided_salvage_v2 原始输出，未对源文件做修改。\n"
        "该版本包含 raw、results、config、figures、manuscript 及 GUIDED_SALVAGE_REPORT。\n"
        "比例尺：1 pixel = 1.722488 μm；固定 60 根毛。\n",
        encoding="utf-8",
    )
    doc_path = make_docx(summary_rows, montage_files[0])
    (OUT / "README_四类文件说明.txt").write_text(
        "Table 5 毛发长度实验整理包（60 根固定毛）\n\n"
        "来源：revision_jobs/table5_hair_measurement_auto_60_guided_salvage_v2。\n"
        "该版本将 HR 手工中心线作为 shape prior，在每个方法图像中独立跟踪，并输出端点、距离、曲率、分叉和 corridor QC。\n\n"
        "01_中文版Word：可直接编辑的中文版 Word，按论文风格放置 Table 5 和代表性图。\n"
        "02_论文图片候选：包含 20 颗种子×9 个方法的 180 张无标注图和 180 张仅折线标注图；每张标注图叠加该种子的 3 根毛发折线。折线由 3–5 个直线段组成，不含额外文字。\n"
        "03_审稿补充材料：精简后的方法级汇总、逐根毛数据、QC、HR 手工线一致性和严格/引导结果统计。\n"
        "04_原始数据留存：完整保留 v2 原始输出，用于复核和追溯。\n\n"
        "比例尺：1 pixel = 1.722488 μm；共 20 颗种子、60 根固定毛发。\n",
        encoding="utf-8",
    )
    print(f"Created: {OUT}")
    print(f"Word: {doc_path}")
    print(f"Montages: {len(montage_files)}")
    print(f"Full seed method pairs: {full_seed_count}")


if __name__ == "__main__":
    main()
