"""
Fiber-level measurements and plots for 3D labeled volumes.

The expected input is a 3D integer array where each non-zero label is one
fiber ID. For each fiber, this module computes:
    - skeleton voxel count (length proxy by number of skeleton voxels)
    - EP-EP Euclidean length from skeleton endpoints
    - orientation angles relative to X, Y and Z axes
"""

import csv
import os

import matplotlib
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_N26 = np.ones((3, 3, 3), dtype=np.uint8)
_N26[1, 1, 1] = 0

_PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fcfcfc",
    "axes.edgecolor": "#303030",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#d8d8d8",
    "grid.linestyle": "-",
    "grid.alpha": 0.45,
    "axes.titleweight": "semibold",
    "axes.labelweight": "regular",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def _format_count_axis(ax):
    """Formats y-axis ticks for histogram counts."""
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def _find_endpoints_3d(skeleton):
    """Returns endpoint coordinates (degree == 1) for a 3D skeleton."""
    deg = convolve(skeleton.astype(np.uint8), _N26, mode="constant", cval=0)
    return np.argwhere((skeleton > 0) & (deg == 1))


def _farthest_pair(coords):
    """Finds the farthest pair of coordinates and the distance between them."""
    if len(coords) < 2:
        return None, None, 0.0

    best_i, best_j = 0, 1
    best_dist = float(np.linalg.norm(coords[1] - coords[0]))
    for i in range(len(coords) - 1):
        for j in range(i + 1, len(coords)):
            dist = float(np.linalg.norm(coords[j] - coords[i]))
            if dist > best_dist:
                best_dist = dist
                best_i, best_j = i, j
    return best_i, best_j, best_dist


def _principal_axis(coords_phys):
    """Computes a robust unit direction vector with SVD (PCA axis)."""
    if len(coords_phys) < 2:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    centered = coords_phys - coords_phys.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return direction / norm


def measure_fibers(labeled_volume, voxel_size=(1.0, 1.0, 1.0)):
    """
    Measures each non-zero label in a 3D labeled volume.

    Args:
        labeled_volume (np.ndarray): 3D volume of integer labels.
        voxel_size (tuple): Physical spacing as (dz, dy, dx).

    Returns:
        list[dict]: Per-fiber measurements.
    """
    if labeled_volume.ndim != 3:
        raise ValueError("labeled_volume must be a 3D array.")

    scale = np.array(voxel_size, dtype=float)
    labels = np.unique(labeled_volume)
    labels = labels[labels != 0]
    records = []

    for label_id in labels:
        mask = labeled_volume == label_id
        if not np.any(mask):
            continue

        coords = np.argwhere(mask)

        # Compute all length features from a local skeleton for this component.
        lo = coords.min(axis=0)
        hi = coords.max(axis=0) + 1
        sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub_skel = skeletonize(sub)
        skel_local = np.argwhere(sub_skel)
        voxel_count = int(len(skel_local))
        skel_phys = (skel_local + lo) * scale if voxel_count else np.empty((0, 3), dtype=float)
        endpoints_local = _find_endpoints_3d(sub_skel)

        if len(endpoints_local) >= 2:
            endpoints_phys = (endpoints_local + lo) * scale
            i, j, ep_ep_length = _farthest_pair(endpoints_phys)
            if i is not None and j is not None and ep_ep_length > 0:
                vec = endpoints_phys[j] - endpoints_phys[i]
                direction = vec / ep_ep_length
            else:
                direction = _principal_axis(skel_phys)
                proj = skel_phys @ direction
                ep_ep_length = float(proj.max() - proj.min()) if len(proj) else 0.0
        else:
            # Skeleton-only fallback for closed loops or degenerate tiny components.
            direction = _principal_axis(skel_phys)
            proj = skel_phys @ direction
            ep_ep_length = float(proj.max() - proj.min()) if len(proj) else 0.0

        # Angles to axis directions in ZYX convention.
        angle_x = float(np.degrees(np.arccos(np.clip(abs(direction[2]), 0.0, 1.0))))
        angle_y = float(np.degrees(np.arccos(np.clip(abs(direction[1]), 0.0, 1.0))))
        angle_z = float(np.degrees(np.arccos(np.clip(abs(direction[0]), 0.0, 1.0))))

        records.append(
            {
                "label_id": int(label_id),
                "voxel_count": voxel_count,
                "ep_ep_length": ep_ep_length,
                "dir_z": float(direction[0]),
                "dir_y": float(direction[1]),
                "dir_x": float(direction[2]),
                "angle_x": angle_x,
                "angle_y": angle_y,
                "angle_z": angle_z,
            }
        )

    return records


def save_measurements_csv(records, output_csv_path):
    """Saves per-fiber measurements to CSV."""
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


def plot_length_histograms(records, output_path, bins=30):
    """Plots histogram for EP-EP Euclidean lengths only."""
    if not records:
        return

    ep_ep_lengths = np.asarray([r["ep_ep_length"] for r in records], dtype=float)

    with plt.style.context("seaborn-v0_8-whitegrid"), matplotlib.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.8), dpi=160)
        ax.hist(
            ep_ep_lengths,
            bins=bins,
            color="#2a9d8f",
            edgecolor="#202020",
            linewidth=0.6,
            alpha=0.88,
        )
        mean_v = float(np.mean(ep_ep_lengths))
        median_v = float(np.median(ep_ep_lengths))
        ax.axvline(mean_v, color="#8e2a2a", linestyle="--", linewidth=1.2, label=f"Mean: {mean_v:.2f}")
        ax.axvline(
            median_v,
            color="#5c3c92",
            linestyle=":",
            linewidth=1.6,
            label=f"Median: {median_v:.2f}",
        )
        ax.set_title("Fiber Length by Endpoint Distance")
        ax.set_xlabel("Length (physical units)")
        ax.set_ylabel("Count")
        _format_count_axis(ax)
        ax.legend(loc="upper right", fontsize=9)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_orientation_histograms(records, output_path, bins=18):
    """Plots orientation histograms relative to X, Y and Z axes."""
    if not records:
        return

    angles_x = np.asarray([r["angle_x"] for r in records], dtype=float)
    angles_y = np.asarray([r["angle_y"] for r in records], dtype=float)
    angles_z = np.asarray([r["angle_z"] for r in records], dtype=float)

    if isinstance(bins, int):
        bins = np.linspace(0.0, 90.0, bins + 1)

    with plt.style.context("seaborn-v0_8-whitegrid"), matplotlib.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=160, sharey=True)
        for ax, values, axis_name, color in (
            (axes[0], angles_x, "X", "#3b6fb6"),
            (axes[1], angles_y, "Y", "#c06c2f"),
            (axes[2], angles_z, "Z", "#4f8a3b"),
        ):
            weights = np.full(values.shape, 100.0 / max(len(values), 1), dtype=float)
            ax.hist(
                values,
                bins=bins,
                range=(0, 90),
                weights=weights,
                color=color,
                edgecolor="#1f1f1f",
                linewidth=0.6,
                alpha=0.86,
            )
            mean_v = float(np.mean(values))
            std_v = float(np.std(values))
            ax.axvline(mean_v, color="#7a1f4b", linestyle="--", linewidth=1.2)
            ax.text(
                0.98,
                0.96,
                f"mean={mean_v:.1f}\nstd={std_v:.1f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2.0},
            )
            ax.set_xlim(0, 90)
            ax.set_title(f"Orientation Relative to {axis_name} Axis")
            ax.set_xlabel("Angle (deg)")

        axes[0].set_ylabel("Fibers (%)")
        fig.suptitle("Fiber Orientation Distributions", y=1.03)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def analyze_and_save(labeled_volume, output_dir, prefix="fiber", voxel_size=(1.0, 1.0, 1.0)):
    """Runs measurement and persists CSV + histogram plots."""
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