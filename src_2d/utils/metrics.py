"""
Fiber-level measurements and plots for 3D labeled volumes.

The expected input is a 3D integer array where each non-zero label is one
fiber ID. For each fiber, this module computes:
    - voxel count (length proxy by number of pixels/voxels)
    - EP-EP Euclidean length from skeleton endpoints
    - orientation angles relative to X, Y and Z axes
"""

import csv
import os

import matplotlib
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_N26 = np.ones((3, 3, 3), dtype=np.uint8)
_N26[1, 1, 1] = 0


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
        voxel_count = int(mask.sum())
        if voxel_count == 0:
            continue

        coords = np.argwhere(mask)
        coords_phys = coords * scale

        # Compute EP-EP length from a local skeleton for this component.
        lo = coords.min(axis=0)
        hi = coords.max(axis=0) + 1
        sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub_skel = skeletonize(sub)
        endpoints_local = _find_endpoints_3d(sub_skel)

        if len(endpoints_local) >= 2:
            endpoints_phys = (endpoints_local + lo) * scale
            i, j, ep_ep_length = _farthest_pair(endpoints_phys)
            if i is not None and j is not None and ep_ep_length > 0:
                vec = endpoints_phys[j] - endpoints_phys[i]
                direction = vec / ep_ep_length
            else:
                direction = _principal_axis(coords_phys)
                proj = coords_phys @ direction
                ep_ep_length = float(proj.max() - proj.min())
        else:
            # Fallback for closed loops or degenerate tiny components.
            direction = _principal_axis(coords_phys)
            proj = coords_phys @ direction
            ep_ep_length = float(proj.max() - proj.min()) if len(proj) else 0.0

        # Angles to axis directions in ZYX convention.
        angle_z = float(np.degrees(np.arccos(np.clip(abs(direction[0]), 0.0, 1.0))))
        angle_y = float(np.degrees(np.arccos(np.clip(abs(direction[1]), 0.0, 1.0))))
        angle_x = float(np.degrees(np.arccos(np.clip(abs(direction[2]), 0.0, 1.0))))

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
    """Plots side-by-side histograms for voxel-count and EP-EP lengths."""
    if not records:
        return

    voxel_lengths = [r["voxel_count"] for r in records]
    ep_ep_lengths = [r["ep_ep_length"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(voxel_lengths, bins=bins, color="#1f77b4", edgecolor="black")
    axes[0].set_title("Fiber Length by Voxel Count")
    axes[0].set_xlabel("Length (voxels)")
    axes[0].set_ylabel("Count")

    axes[1].hist(ep_ep_lengths, bins=bins, color="#ff7f0e", edgecolor="black")
    axes[1].set_title("Fiber Length by EP-EP Euclidean Distance")
    axes[1].set_xlabel("Length (physical units)")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_orientation_histograms(records, output_path, bins=18):
    """Plots orientation histograms relative to X, Y and Z axes."""
    if not records:
        return

    angles_x = [r["angle_x"] for r in records]
    angles_y = [r["angle_y"] for r in records]
    angles_z = [r["angle_z"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, values, axis_name, color in (
        (axes[0], angles_x, "X", "#2ca02c"),
        (axes[1], angles_y, "Y", "#d62728"),
        (axes[2], angles_z, "Z", "#9467bd"),
    ):
        ax.hist(values, bins=bins, range=(0, 90), color=color, edgecolor="black")
        ax.set_xlim(0, 90)
        ax.set_title(f"Orientation Relative to {axis_name} Axis")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
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