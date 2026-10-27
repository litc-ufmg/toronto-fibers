"""Named pipeline outputs; arrays retain their original dtypes and coordinates."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Direct3DResult:
    """Direct 3D intermediates and final labels.

    ``raw_volume`` is a copy of the input. All other arrays use the coordinate
    frame of ``cropped_volume``, after removal of zero-intensity margins.
    Arrays are mutable; no additional copies are made when packaging results.
    """

    raw_volume: np.ndarray
    cropped_volume: np.ndarray
    binary_mask: np.ndarray
    pruned_skeleton: np.ndarray
    labels: np.ndarray


@dataclass
class SliceWiseResult:
    """Slice-wise intermediates, diagnostic skeleton, and reconstructed labels.

    All arrays use the input coordinate frame. ``skeleton_3d`` is an unpruned
    diagnostic skeleton, not the collection of per-slice pruning markers.
    ``slicewise_labels`` have independent IDs per slice; only ``labels`` have
    globally reconciled IDs. ``skeleton_rgb`` has shape (Z, Y, X, 3).
    """

    raw_volume: np.ndarray
    binary_mask: np.ndarray
    skeleton_3d: np.ndarray
    junction_mask: np.ndarray
    skeleton_rgb: np.ndarray
    slicewise_labels: np.ndarray
    labels: np.ndarray
