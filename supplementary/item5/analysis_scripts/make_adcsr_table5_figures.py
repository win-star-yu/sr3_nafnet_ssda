#!/usr/bin/env python3
"""Create clean ADCSR Table-5 presentation figures from measured raw paths."""
from __future__ import annotations

import ast
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import splprep, splev


def smooth(points: np.ndarray, samples: int = 320, smoothing: float = 0.5) -> np.ndarray:
    p = np.asarray(points, np.float64)
    if len(p) < 3:
        return p
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    u = np.r_[0.0, np.cumsum(d)]
    if u[-1] <= 1e-8:
        return p
    u /= u[-1]
    try:
        k = min(3, len(p) - 1)
        tck, _ = splprep(p.T, u=u, s=smoothing, k=k)
        q = np.asarray(splev(np.linspace(0, 1, samples), tck)).T
        q[0], q[-1] = p[0], p[-1]
        return q
    except Exception:
        return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"./redacted/path"))
    ap.add_argument("--config", type=Path, default=Path(r"./redacted/path"))
    ap.add_argument("--one-context-per-hair", action="store_true",
                    help="write one full-seed red-box image for each hair instead of one image with three boxes")
    args = ap.parse_args()
    root = args.root
    cfg = args.config
    refs_path = Path(r"./redacted/path")
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    with (root / "raw" / "auto_hair_measurements.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("method") == "ADCSR"]

    clean_dir = root / "figures" / "paper_candidates" / "ADCSR_smooth_hair_only"
    crop_dir = root / "figures" / "paper_candidates" / "ADCSR_unannotated_crops"
    context_dir = root / "figures" / "paper_candidates" / "ADCSR_seed_context_redbox"
    clean_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    # One clean local crop per fixed hair: only an anti-aliased smooth yellow line.
    for row in rows:
        roi = row["roi"]
        seed = int(row["seed_id"])
        ref = refs[roi]
        x0, y0 = int(ref["x0_hr"]), int(ref["y0_hr"])
        src = cfg / "08_ADCSR" / f"seed{seed:02d}.png"
        full = np.asarray(Image.open(src).convert("RGB"))
        crop = full[y0:y0 + 512, x0:x0 + 512]
        Image.fromarray(crop).save(crop_dir / f"{roi}_ADCSR_original.png")
        pts = np.asarray(ast.literal_eval(row["raw_centerline_vertices"]), np.float64)
        curve = np.rint(smooth(pts)).astype(np.int32).reshape(-1, 1, 2)
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.polylines(bgr, [curve], False, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
        length = float(row["length_um"])
        out_name = f"{roi}_ADCSR_L{length:.1f}um_smooth.png"
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(clean_dir / out_name)

    # Full seed context images.  By default this makes one image per seed with
    # three boxes; the paper package uses one image per hair with one box.
    by_seed = {}
    for roi, ref in refs.items():
        by_seed.setdefault(int(ref["image_id"]), []).append((roi, ref))
    if args.one_context_per_hair:
        for seed, seed_refs in sorted(by_seed.items()):
            src = cfg / "08_ADCSR" / f"seed{seed:02d}.png"
            for roi, ref in seed_refs:
                im = Image.open(src).convert("RGB")
                d = ImageDraw.Draw(im)
                x0, y0 = int(ref["x0_hr"]), int(ref["y0_hr"])
                d.rectangle((x0, y0, x0 + 512, y0 + 512), outline=(255, 0, 0), width=8)
                im.save(context_dir / f"{roi}_ADCSR_redbox.png")
    else:
        for seed, seed_refs in sorted(by_seed.items()):
            src = cfg / "08_ADCSR" / f"seed{seed:02d}.png"
            im = Image.open(src).convert("RGB")
            d = ImageDraw.Draw(im)
            for roi, ref in seed_refs:
                x0, y0 = int(ref["x0_hr"]), int(ref["y0_hr"])
                d.rectangle((x0, y0, x0 + 512, y0 + 512), outline=(255, 0, 0), width=8)
            im.save(context_dir / f"seed{seed:02d}_ADCSR_redbox.png")

    # A compact ADCSR-only audit table and README for the generated figures.
    fields = ["roi", "seed_id", "hair_id", "length_um", "raw_centerline_length_px", "display_polyline_length_um",
              "measurable", "failure_reason", "confidence_score", "mean_distance_to_hr_prior_px",
              "max_distance_to_hr_prior_px", "endpoint_offset_root_px", "endpoint_offset_tip_px",
              "path_curvature_score", "branch_crossing_flag", "corridor_violation_flag"]
    with (root / "results" / "adcsr_per_hair_measurements.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    summary = next(csv.DictReader((root / "results" / "table5_hair_measurement.csv").open(encoding="utf-8-sig", newline="")))
    with (root / "results" / "adcsr_table5_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
    context_note = ("The red-box context images contain one corresponding approximate hair location per full ADCSR seed image.\n"
                    if args.one_context_per_hair else
                    "The red-box context images show the corresponding approximate hair locations on each full ADCSR seed image.\n")
    (root / "README_ADCSR_Table5.txt").write_text(
        "ADCSR Table 5 measurement output\n"
        "================================\n"
        "The measurement uses the verified table5-auto-centerline-v3.0-guided-tracker protocol.\n"
        "Sixty fixed HR hair ROIs (20 seeds x 3 hairs) are used as shape priors; ADCSR is independently tracked in each crop.\n"
        "Configuration: 40 reference points, 14 px corridor half-width, 10 px endpoint search radius, guided branch salvage, 1 px = 1.722488 um.\n"
        "The clean presentation crops contain only the independently measured smooth yellow centerline; no white control points or text are drawn.\n"
        + context_note
        + "The displayed line is for visual presentation; the reported length uses the raw guided centerline.\n",
        encoding="utf-8",
    )
    print(f"clean={len(rows)} contexts={len(by_seed)} output={root}")


if __name__ == "__main__":
    main()
