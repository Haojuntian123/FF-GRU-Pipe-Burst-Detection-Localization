"""Test coordinated accuracy for all windows and burst windows."""

import unittest

from scripts.metrics import diagnostic_metrics


class DiagnosticMetricsTest(unittest.TestCase):
    def test_cac_pb_excludes_true_negatives(self):
        report = diagnostic_metrics(
            ["normal", "burst1", "burst2"], [0, 1, 1], [0, 1, 0],
            ["burst2", "burst1"], [2, 1], [2, 1],
        )
        self.assertEqual(report["CAC"], 2 / 3)
        self.assertEqual(report["CAC_PB"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
