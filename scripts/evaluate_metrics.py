"""Evaluate detection and localization prediction files.

Each input CSV must contain ``sample_id``, ``y_true`` and ``y_pred`` columns.
The detection and localization files may be produced by separate training
runs; localization rows must match the true burst IDs in the detection file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.metrics import diagnostic_metrics, write_json


def _read_predictions(path: str | Path) -> dict[str, list]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no prediction rows")
    required = {"sample_id", "y_true", "y_pred"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return {
        "sample_id": [row["sample_id"] for row in rows],
        "y_true": [int(row["y_true"]) for row in rows],
        "y_pred": [int(row["y_pred"]) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection", required=True, help="local detection prediction CSV")
    parser.add_argument("--localization", required=True, help="local localization prediction CSV")
    parser.add_argument("--output", required=True, help="local JSON report path")
    args = parser.parse_args()

    detection = _read_predictions(args.detection)
    localization = _read_predictions(args.localization)
    report = diagnostic_metrics(
        detection["sample_id"], detection["y_true"], detection["y_pred"],
        localization["sample_id"], localization["y_true"], localization["y_pred"],
    )
    write_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
