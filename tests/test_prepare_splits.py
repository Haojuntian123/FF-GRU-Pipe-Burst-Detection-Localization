"""Test source-group separation and paired task IDs."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_splits import assign_groups, prepare


class PrepareSplitsTest(unittest.TestCase):
    def test_mixed_source_groups_require_explicit_allocation(self):
        rows = [{"group_id": group, "kind": kind}
                for kind, groups in (("normal", ("shared", "n2", "n3")),
                                     ("burst", ("shared", "b2", "b3")))
                for group in groups]
        with self.assertRaisesRegex(ValueError, "mixed shared"):
            assign_groups(rows, 1)

    def test_shared_source_groups_cannot_cross_splits(self):
        rows = [{"group_id": group, "kind": kind}
                for group in ("period1", "period2", "period3", "period4")
                for kind in ("normal", "burst")]
        assignment = assign_groups(rows, 42)
        for group in ("period1", "period2", "period3", "period4"):
            self.assertEqual(assignment[("normal", group)], assignment[("burst", group)])

    def test_generated_csvs_keep_shared_burst_ids_and_training_only_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["group_id", "kind", "pressure", "flow", "partition_id"])
                for group in range(4):
                    for kind in ("normal", "burst"):
                        stem = f"{group}_{kind}"
                        for suffix, sensor in (("p", "P1"), ("q", "Q1")):
                            with (root / f"{stem}_{suffix}.csv").open("w", newline="", encoding="utf-8") as data:
                                output = csv.writer(data)
                                output.writerow(["time", sensor])
                                for i in range(16):
                                    output.writerow([i, i + 1])
                        writer.writerow([f"period{group}", kind, f"{stem}_p.csv", f"{stem}_q.csv", 1 if kind == "burst" else ""])
            output = root / "splits"
            prepare(manifest, output, seed=3, window=2, burst_normal_ratio=2,
                    pressure_sensors=["P1"], flow_sensors=["Q1"])

            def read(name):
                with (output / name).open(newline="", encoding="utf-8") as handle:
                    return list(csv.reader(handle))

            for split in ("train", "val", "test"):
                ids = read(f"{split}_burst_ids.csv")
                labels = read(f"{split}_burst_label.csv")
                self.assertEqual([row[0] for row in ids], [row[0] for row in labels])
                self.assertEqual(len(read(f"{split}_burst_pressure.csv")), len(ids) * 2)
                if split != "train":
                    self.assertEqual(len(ids), len(read(f"{split}_normal_ids.csv")))
            self.assertEqual(len(read("train_burst_ids.csv")), 2 * len(read("train_normal_ids.csv")))
            split_groups = [set(row[0].split(":")[1] for row in read(f"{split}_burst_ids.csv"))
                            for split in ("train", "val", "test")]
            self.assertTrue(all(a.isdisjoint(b) for i, a in enumerate(split_groups) for b in split_groups[i + 1:]))


if __name__ == "__main__":
    unittest.main()
