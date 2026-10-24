"""Unified paper-table evaluator for full-image super-resolution results.

Metrics:
  RGB PSNR, RGB SSIM, LPIPS-Alex, LPIPS-VGG, and DISTS.

The same HR reference directory is used for every method. Results are
written as one per-image CSV (method + image_id) and one mean/sample-SD CSV.

Example (Linux/4090):
  python eval_table_metrics.py \
    --ref_dir ./data/test/hr \
    --method Ours=./data/test/ours \
    --method AdcSR=./data/test/adcsr \
    --out_dir ./data/test/metrics \
    --crop 8 --lpips_net both --batch 1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_index(root: Path, recursive: bool) -> dict[str, Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    result: dict[str, Path] = {}
    for path in iterator:
        if path.is_file() and path.suffix.lower() in EXTS:
            key = path.relative_to(root).with_suffix("").as_posix()
            if key in result:
                raise RuntimeError(f"duplicate image id in {root}: {key}")
            result[key] = path
    return result


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def crop_border(image: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return image
    if image.shape[0] <= 2 * border or image.shape[1] <= 2 * border:
        raise ValueError(f"crop={border} is too large for image shape {image.shape}")
    return image[border:-border, border:-border]


def psnr_rgb(pred: np.ndarray, ref: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - ref.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * math.log10(255.0**2 / mse))


def ssim_channel(pred: np.ndarray, ref: np.ndarray) -> float:
    kernel_1d = cv2.getGaussianKernel(11, 1.5)
    window = kernel_1d @ kernel_1d.T
    pred = pred.astype(np.float64)
    ref = ref.astype(np.float64)
    pred_mean = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
    ref_mean = cv2.filter2D(ref, -1, window)[5:-5, 5:-5]
    pred_var = cv2.filter2D(pred * pred, -1, window)[5:-5, 5:-5] - pred_mean**2
    ref_var = cv2.filter2D(ref * ref, -1, window)[5:-5, 5:-5] - ref_mean**2
    covariance = cv2.filter2D(pred * ref, -1, window)[5:-5, 5:-5] - pred_mean * ref_mean
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    value = ((2 * pred_mean * ref_mean + c1) * (2 * covariance + c2)) / (
        (pred_mean**2 + ref_mean**2 + c1) * (pred_var + ref_var + c2)
    )
    return float(value.mean())


def ssim_rgb(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean([ssim_channel(pred[..., c], ref[..., c]) for c in range(3)]))


def to_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack([image.transpose(2, 0, 1) for image in images], axis=0)
    return torch.from_numpy(np.ascontiguousarray(array)).float().div_(255.0).to(device)


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must use NAME=DIRECTORY")
    name, directory = value.split("=", 1)
    if not name.strip() or not directory.strip():
        raise argparse.ArgumentTypeError("--method must use NAME=DIRECTORY")
    return name.strip(), Path(directory).expanduser()


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def make_models(args: argparse.Namespace, device: torch.device):
    import lpips

    lpips_models = {}
    if args.lpips_net in {"alex", "both"}:
        lpips_models["lpips_alex"] = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
    if args.lpips_net in {"vgg", "both"}:
        lpips_models["lpips_vgg"] = lpips.LPIPS(net="vgg", verbose=False).eval().to(device)

    from DISTS_pytorch import DISTS

    old_prefix = sys.prefix
    if args.dists_prefix:
        sys.prefix = args.dists_prefix
    try:
        dists = DISTS().eval().to(device)
    finally:
        sys.prefix = old_prefix
    return lpips_models, dists


def evaluate_method(
    method: str,
    ref_images: dict[str, Path],
    pred_images: dict[str, Path],
    args: argparse.Namespace,
    device: torch.device,
    lpips_models: dict[str, torch.nn.Module],
    dists: torch.nn.Module,
) -> list[dict[str, object]]:
    missing = sorted(set(ref_images) - set(pred_images))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"{method} is missing {len(missing)} reference image(s), e.g. {preview}")

    names = sorted(ref_images)
    rows: list[dict[str, object]] = []
    for start in range(0, len(names), args.batch):
        batch_names = names[start : start + args.batch]
        pred_batch: list[np.ndarray] = []
        ref_batch: list[np.ndarray] = []
        for image_id in batch_names:
            ref = read_rgb(ref_images[image_id])
            pred = read_rgb(pred_images[image_id])
            if ref.shape != pred.shape:
                raise RuntimeError(
                    f"size mismatch for {method}/{image_id}: ref={ref.shape}, pred={pred.shape}"
                )
            ref_crop = crop_border(ref, args.crop)
            pred_crop = crop_border(pred, args.crop)
            rows.append(
                {
                    "method": method,
                    "image_id": image_id,
                    "psnr_rgb": psnr_rgb(pred_crop, ref_crop),
                    "ssim_rgb": ssim_rgb(pred_crop, ref_crop),
                }
            )
            pred_batch.append(pred_crop)
            ref_batch.append(ref_crop)

        pred_tensor = to_tensor(pred_batch, device)
        ref_tensor = to_tensor(ref_batch, device)
        with torch.inference_mode():
            for field, model in lpips_models.items():
                values = model(pred_tensor, ref_tensor, normalize=True).flatten().cpu().numpy()
                for row, value in zip(rows[-len(batch_names) :], values):
                    row[field] = float(value)
            values = dists(pred_tensor, ref_tensor).flatten().cpu().numpy()
            for row, value in zip(rows[-len(batch_names) :], values):
                row["dists"] = float(value)
        print(f"{method}: {min(start + len(batch_names), len(names))}/{len(names)}", flush=True)
    return rows


def write_outputs(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["method", "image_id", "psnr_rgb", "ssim_rgb", "lpips_alex", "lpips_vgg", "dists"]
    with (out_dir / "metrics_per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metric_fields = fields[2:]
    summary_fields = ["method", "count"]
    for field in metric_fields:
        summary_fields.extend([field, f"{field}_std"])
    methods = sorted({str(row["method"]) for row in rows})
    with (out_dir / "metrics_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            record: dict[str, object] = {"method": method, "count": len(method_rows)}
            for field in metric_fields:
                values = [float(row[field]) for row in method_rows]
                mean, sd = mean_sd(values)
                record[field] = mean
                record[f"{field}_std"] = sd
            writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref_dir", required=True, type=Path)
    parser.add_argument("--method", required=True, action="append", type=parse_method,
                        help="method and prediction directory, repeatable: NAME=DIRECTORY")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--crop", type=int, default=8, help="border crop; default: 8")
    parser.add_argument("--batch", type=int, default=1, help="metric batch size; default: 1")
    parser.add_argument("--recursive", action="store_true", help="match images by relative path")
    parser.add_argument("--lpips_net", choices=["alex", "vgg", "both"], default="both")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--dists_prefix", default="",
                        help="optional vendor prefix used to locate DISTS weights")
    args = parser.parse_args()
    if args.crop < 0 or args.batch < 1:
        parser.error("--crop must be >= 0 and --batch must be >= 1")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    ref_images = image_index(args.ref_dir, args.recursive)
    if not ref_images:
        raise RuntimeError(f"no images found in reference directory: {args.ref_dir}")
    if len({name for name, _ in args.method}) != len(args.method):
        raise RuntimeError("method names must be unique")

    lpips_models, dists = make_models(args, device)
    rows: list[dict[str, object]] = []
    for method, pred_dir in args.method:
        pred_images = image_index(pred_dir, args.recursive)
        rows.extend(evaluate_method(method, ref_images, pred_images, args, device, lpips_models, dists))
    write_outputs(rows, args.out_dir)
    print(f"wrote {len(rows)} rows to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
