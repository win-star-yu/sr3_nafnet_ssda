"""Compute a reproducible bootstrap mean and 95% interval for an existing CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def bootstrap(values, repeats, seed):
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(repeats):
        estimates.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return sum(estimates) / len(estimates), percentile(estimates, 0.025), percentile(estimates, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--group-column", default="")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    groups = {}
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            group = row.get(args.group_column, "all") if args.group_column else "all"
            try:
                groups.setdefault(group, []).append(float(row[args.value_column]))
            except (KeyError, TypeError, ValueError):
                continue
    if not groups or any(not values for values in groups.values()):
        raise RuntimeError("No numeric values found")

    rows = []
    for group, values in groups.items():
        mean, low, high = bootstrap(values, args.repeats, args.seed)
        rows.append({"group": group, "n": len(values), "mean": mean, "ci95_low": low, "ci95_high": high})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
