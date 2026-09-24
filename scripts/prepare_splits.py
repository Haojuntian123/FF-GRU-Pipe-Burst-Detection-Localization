"""Partition source groups and assemble training files.

Manifest columns: group_id, kind, pressure, flow, label, partition_id. Each
path identifies one source block. Burst sources need either a label file with
one 1-based partition ID per window or a constant partition_id in the manifest.
All blocks belonging to one parent event or operating period share group_id.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path


def assign_groups(rows, seed):
    rng = random.Random(seed)
    assignment = {}
    normal_groups = {row["group_id"] for row in rows if row["kind"] == "normal"}
    burst_groups = {row["group_id"] for row in rows if row["kind"] == "burst"}
    if normal_groups & burst_groups:
        if normal_groups != burst_groups:
            raise ValueError("mixed shared and unpaired source groups need an explicit allocation")
        groups = sorted(normal_groups)
        if len(groups) < 3:
            raise ValueError("at least three shared source groups are needed")
        rng.shuffle(groups)
        val_count = max(1, round(len(groups) / 10))
        test_count = max(1, round(len(groups) / 10))
        if val_count + test_count >= len(groups):
            raise ValueError("too few shared groups for train/val/test")
        for i, group in enumerate(groups):
            split = "val" if i < val_count else "test" if i < val_count + test_count else "train"
            for kind in ("normal", "burst"):
                assignment[(kind, group)] = split
        return assignment
    for kind in ("normal", "burst"):
        groups = sorted({row["group_id"] for row in rows if row["kind"] == kind})
        if len(groups) < 3:
            raise ValueError(f"{kind}: at least three source groups are needed for train/val/test")
        rng.shuffle(groups)
        val_count = max(1, round(len(groups) / 10))
        test_count = max(1, round(len(groups) / 10))
        if val_count + test_count >= len(groups):
            raise ValueError(f"{kind}: too few groups for the requested allocation")
        for i, group in enumerate(groups):
            split = "val" if i < val_count else "test" if i < val_count + test_count else "train"
            assignment[(kind, group)] = split
    return assignment


def source_rows(path, columns=None):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        indices = None
        if columns is not None:
            header = next(reader)
            if len(set(header)) != len(header) or not set(columns).issubset(header):
                raise ValueError(f"missing or duplicate sensor columns in {path}")
            indices = [header.index(name) for name in columns]
        for row in reader:
            if indices is not None:
                row = [row[i] for i in indices]
            if not row or any(not cell.strip() for cell in row):
                raise ValueError(f"empty value in {path}")
            yield row


def count_rows(path, columns=None):
    return sum(1 for _ in source_rows(path, columns))


def prepare(manifest, out_dir, seed=0, window=60, burst_normal_ratio=16,
            pressure_sensors=None, flow_sensors=None):
    if window <= 0 or burst_normal_ratio <= 0:
        raise ValueError("window and burst_normal_ratio must be positive")
    manifest, out_dir = Path(manifest).resolve(), Path(out_dir).resolve()
    if (pressure_sensors is None) != (flow_sensors is None):
        raise ValueError("both sensor lists are needed for generated CSV inputs")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"group_id", "kind", "pressure", "flow"}.issubset(reader.fieldnames or []):
            raise ValueError("manifest requires group_id, kind, pressure, flow")
        rows = list(reader)
    if not rows:
        raise ValueError("empty manifest")
    seen_paths = set()
    counts = defaultdict(int)
    for row in rows:
        kind = row["kind"].strip().lower()
        if kind not in {"normal", "burst"} or not row["group_id"].strip():
            raise ValueError("kind must be normal or burst; group_id cannot be empty")
        row["kind"] = kind
        for field in ("pressure", "flow"):
            path = (manifest.parent / row[field]).resolve()
            if path in seen_paths:
                raise ValueError(f"source used more than once: {path}")
            seen_paths.add(path)
            row[field] = path
        if kind == "burst":
            label_path = (row.get("label") or "").strip()
            partition_id = (row.get("partition_id") or "").strip()
            if bool(label_path) == bool(partition_id):
                raise ValueError("burst sources need exactly one of label or partition_id")
            row["label"] = (manifest.parent / label_path).resolve() if label_path else None
            if partition_id and (not partition_id.isdigit() or int(partition_id) < 1):
                raise ValueError("partition_id must be a positive 1-based class ID")
        n = count_rows(row["pressure"], pressure_sensors)
        if n == 0 or n % window or count_rows(row["flow"], flow_sensors) != n:
            raise ValueError(f"pressure/flow rows must match and be divisible by {window}: {row['pressure']}")
        if kind == "burst" and row["label"] and count_rows(row["label"]) != n // window:
            raise ValueError(f"label count does not match windows: {row['label']}")
        row["windows"] = n // window

    assignment = assign_groups(rows, seed)
    for row in rows:
        counts[(assignment[(row["kind"], row["group_id"])], row["kind"])] += row["windows"]
    normal_count = counts[("train", "normal")]
    burst_count = counts[("train", "burst")]
    normal_take = min(normal_count, burst_count // burst_normal_ratio)
    burst_take = normal_take * burst_normal_ratio
    if normal_take == 0:
        raise ValueError("not enough training burst windows for the chosen ratio")
    rng = random.Random(seed)
    selected = {
        "normal": set(rng.sample(range(normal_count), normal_take)),
        "burst": set(rng.sample(range(burst_count), burst_take)),
    }
    offsets = defaultdict(int)
    out_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        writers = {}
        for split in ("train", "val", "test"):
            for kind in ("normal", "burst"):
                for field in ("pressure", "flow", "ids") + (("label",) if kind == "burst" else ()):
                    path = out_dir / f"{split}_{kind}_{field}.csv"
                    writers[(split, kind, field)] = csv.writer(stack.enter_context(path.open("w", newline="", encoding="utf-8")))
        for source_index, row in enumerate(rows):
            kind = row["kind"]
            split = assignment[(kind, row["group_id"])]
            labels = source_rows(row["label"]) if kind == "burst" and row["label"] else None
            pressure = source_rows(row["pressure"], pressure_sensors)
            flow = source_rows(row["flow"], flow_sensors)
            for index in range(row["windows"]):
                p_block = [next(pressure) for _ in range(window)]
                q_block = [next(flow) for _ in range(window)]
                label = (next(labels)[0] if labels else row["partition_id"]) if kind == "burst" else None
                offset = offsets[(split, kind)]
                offsets[(split, kind)] += 1
                if split == "train" and offset not in selected[kind]:
                    continue
                sample_id = f"{kind}:{row['group_id']}:{source_index}:{index}"
                writers[(split, kind, "ids")].writerow([sample_id])
                for p, q in zip(p_block, q_block):
                    writers[(split, kind, "pressure")].writerow(p)
                    writers[(split, kind, "flow")].writerow(q)
                if label is not None:
                    writers[(split, kind, "label")].writerow([sample_id, label])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--burst-normal-ratio", type=int, default=16)
    parser.add_argument("--pressure-sensors", nargs="+", help="ordered pressure column names in generator CSVs")
    parser.add_argument("--flow-sensors", nargs="+", help="ordered flow column names in generator CSVs")
    args = parser.parse_args()
    counts = prepare(args.manifest, args.out_dir, args.seed, args.window, args.burst_normal_ratio,
                     args.pressure_sensors, args.flow_sensors)
    print("Source-group allocation complete:", dict(counts))


if __name__ == "__main__":
    main()
