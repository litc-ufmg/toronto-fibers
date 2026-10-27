"""CSV export and the existing measurement-plus-histograms workflow."""

import csv
import os
from typing import TypedDict

import numpy as np

from .metrics import FiberMeasurement, measure_fibers
from .plotting import plot_length_histograms, plot_orientation_histograms


class MeasurementReport(TypedDict):
    """Measurements and the requested output paths (plots require records)."""

    records: list[FiberMeasurement]
    csv_path: str
    length_hist_path: str
    orientation_hist_path: str


def save_measurements_csv(records, output_csv_path):
    """Save measurements with the historical column order, including a header."""
    fields = [
        "label_id",
        "voxel_count",
        "ep_ep_length",
        "dir_z",
        "dir_y",
        "dir_x",
        "angle_x",
        "angle_y",
        "angle_z",
    ]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def analyze_and_save(
    labeled_volume: np.ndarray,
    output_dir: str | os.PathLike,
    prefix: str = "fiber",
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> MeasurementReport:
    """Measure a labeled volume and save a CSV plus length/orientation PNGs.

    Output names and return keys match the original ``utils.metrics`` API.
    Empty measurements still write a header-only CSV, but no PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    records = measure_fibers(labeled_volume, voxel_size=voxel_size)

    csv_path = os.path.join(output_dir, f"{prefix}_metrics.csv")
    length_hist_path = os.path.join(output_dir, f"{prefix}_length_hist.png")
    orientation_hist_path = os.path.join(output_dir, f"{prefix}_orientation_hist.png")

    save_measurements_csv(records, csv_path)
    plot_length_histograms(records, length_hist_path)
    plot_orientation_histograms(records, orientation_hist_path)

    return {
        "records": records,
        "csv_path": csv_path,
        "length_hist_path": length_hist_path,
        "orientation_hist_path": orientation_hist_path,
    }
