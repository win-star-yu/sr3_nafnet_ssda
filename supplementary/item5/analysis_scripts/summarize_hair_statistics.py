"""Summarize an existing per-hair measurement CSV without changing its raw values."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--method-column", default="method")
    parser.add_argument("--value-column", default="length_um")
    args = parser.parse_args()

    groups = defaultdict(list)
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                groups[row[args.method_column]].append(float(row[args.value_column]))
            except (KeyError, TypeError, ValueError):
                continue
    rows = []
    for method, values in sorted(groups.items()):
        rows.append({
            "method": method,
            "n": len(values),
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        })
    if not rows:
        raise RuntimeError("No numeric hair measurements found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
