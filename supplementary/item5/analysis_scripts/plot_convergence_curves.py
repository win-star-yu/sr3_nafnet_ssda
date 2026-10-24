"""Plot a convergence curve from a CSV exported by the training log merger."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--x", default="iteration")
    parser.add_argument("--y", default="psnr_sr")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row.get("stage", "all")].append((float(row[args.x]), float(row[args.y])))

    for stage, points in sorted(groups.items()):
        points.sort()
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=stage)
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
