"""Numerical fiber measurements, independent of plotting and file output.

Volumes and spacing use (Z, Y, X) order. Length is the greatest Euclidean
distance between local skeleton endpoints, with an SVD projection fallback.
Orientation angles are acute, treating opposite directions as equivalent.
"""

from typing import TypedDict

import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize


class FiberMeasurement(TypedDict):
    """One labeled instance; ``voxel_count`` counts skeleton voxels, not volume."""

    label_id: int
    voxel_count: int
    ep_ep_length: float
    dir_z: float
    dir_y: float
    dir_x: float
    angle_x: float
    angle_y: float
    angle_z: float


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


def measure_fibers(
    labeled_volume: np.ndarray,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[FiberMeasurement]:
    """Measure each nonzero label without modifying the input volume.

    Args:
        labeled_volume: 3D integer array; each nonzero value is one fiber ID.
        voxel_size: Physical spacing (dz, dy, dx). Unit spacing yields lengths
            in voxel units; micrometer spacing yields lengths in micrometers.

    Returns:
        Records ordered by label ID. Length uses farthest skeleton endpoints;
        if fewer than two endpoints exist, it uses the skeleton's extent along
        its principal axis. Angles are in degrees in [0, 90]. The historical
        zero-length/default-X direction for degenerate skeletons is preserved.
    """
    if labeled_volume.ndim != 3:
        raise ValueError("labeled_volume must be a 3D array.")

    scale = np.array(voxel_size, dtype=float)
    labels = np.unique(labeled_volume)
    labels = labels[labels != 0]
    records: list[FiberMeasurement] = []

    for label_id in labels:
        mask = labeled_volume == label_id
        if not np.any(mask):
            continue

        coords = np.argwhere(mask)

        # Compute all length features from a local skeleton for this component.
        lo = coords.min(axis=0)
        hi = coords.max(axis=0) + 1
        sub = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        sub_skel = skeletonize(sub)
        skel_local = np.argwhere(sub_skel)
        voxel_count = len(skel_local)
        skel_phys = (
            (skel_local + lo) * scale if voxel_count else np.empty((0, 3), dtype=float)
        )
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
