"""Regression coverage for shared measurements and legacy report interfaces."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fiber_analysis.metrics import measure_fibers
from fiber_analysis.reporting import analyze_and_save
from src.utils import metrics as legacy3d
from src_2d.utils import metrics as legacy2d


class MetricsTests(unittest.TestCase):
    def test_shared_measurements_preserve_spacing_angles_and_label_order(self):
        labels = np.zeros((12, 12, 12), dtype=np.uint32)
        labels[1, 1, 1:9] = 2
        labels[2:10, 4, 3] = 7
        labels[10, 10, 10] = 12
        indices = np.arange(2, 9)
        labels[indices, indices, 10] = 18
        original = labels.copy()

        records = measure_fibers(labels, voxel_size=(2.0, 1.5, 0.5))
        self.assertEqual([r["label_id"] for r in records], [2, 7, 12, 18])
        np.testing.assert_allclose(
            [r["ep_ep_length"] for r in records], [3.5, 14.0, 0.0, 15.0]
        )
        np.testing.assert_allclose(
            [[r["angle_x"], r["angle_y"], r["angle_z"]] for r in records],
            [
                [0, 90, 90],
                [90, 90, 0],
                [0, 90, 90],
                [90, np.degrees(np.arccos(0.6)), np.degrees(np.arccos(0.8))],
            ],
        )
        self.assertEqual([r["voxel_count"] for r in records], [8, 8, 1, 7])
        np.testing.assert_array_equal(labels, original)
        for legacy in (legacy3d, legacy2d):
            self.assertEqual(legacy.measure_fibers(labels, (2.0, 1.5, 0.5)), records)

    def test_numeric_measurements_do_not_import_matplotlib(self):
        code = (
            "import sys; from fiber_analysis.metrics import measure_fibers; "
            "assert 'matplotlib' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reporting_preserves_files_columns_and_compatibility_exports(self):
        labels = np.zeros((8, 8, 8), dtype=np.uint32)
        labels[2, 3, 1:7] = 9
        expected = measure_fibers(labels)
        with tempfile.TemporaryDirectory() as tmp:
            for index, reporting in enumerate(
                (analyze_and_save, legacy3d.analyze_and_save, legacy2d.analyze_and_save)
            ):
                with self.subTest(interface=index):
                    folder = Path(tmp) / str(index)
                    output = reporting(labels, folder, prefix="sample")
                    self.assertEqual(output["records"], expected)
                    for key, suffix in (
                        ("csv_path", "_metrics.csv"),
                        ("length_hist_path", "_length_hist.png"),
                        ("orientation_hist_path", "_orientation_hist.png"),
                    ):
                        path = folder / f"sample{suffix}"
                        self.assertEqual(output[key], str(path))
                        self.assertGreater(path.stat().st_size, 0)
                    with Path(output["csv_path"]).open(newline="") as stream:
                        reader = csv.DictReader(stream)
                        rows = list(reader)
                        self.assertEqual(reader.fieldnames, list(expected[0]))
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["label_id"], "9")
                    self.assertEqual(float(rows[0]["ep_ep_length"]), 5.0)

    def test_empty_report_still_writes_only_csv_and_invalid_dimensions_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = analyze_and_save(np.zeros((3, 3, 3), dtype=np.uint32), tmp)
            self.assertEqual(output["records"], [])
            self.assertTrue(Path(output["csv_path"]).is_file())
            self.assertFalse(Path(output["length_hist_path"]).exists())
            self.assertFalse(Path(output["orientation_hist_path"]).exists())
        with self.assertRaisesRegex(ValueError, "must be a 3D array"):
            measure_fibers(np.zeros((3, 3), dtype=np.uint32))


if __name__ == "__main__":
    unittest.main()
