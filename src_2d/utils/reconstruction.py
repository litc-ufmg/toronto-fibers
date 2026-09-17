"""
3D label reconciliation from a stack of independently labeled 2D slices.

Each 2D slice is segmented in isolation, so a single physical fiber receives
unrelated label IDs across consecutive slices. ``propagate_labels`` walks
through the stack along the chosen axis and links 2D components in adjacent
slices by spatial overlap, producing a single globally consistent labeled
volume.

Strategy
--------
- The first slice is relabeled with sequential global IDs.
- For every subsequent slice:
    * For each 2D component in the current slice, count the overlap (in
      pixels) with each labeled component of the previous slice.
    * If the largest overlap is at least ``min_overlap`` pixels AND covers
      at least ``min_overlap_ratio`` of the smaller component, the current
      component inherits that global ID.
    * Otherwise a new global ID is created.
- Conflicts (two current components claiming the same previous ID) are
  resolved by giving the inherited ID to the component with the highest
  overlap; the others receive new IDs.
"""

import numpy as np
from scipy.ndimage import label as cc_label


def _iter_slices(volume, axis):
    if axis == "x":
        for i in range(volume.shape[0]):
            yield i, volume[i, :, :]
    elif axis == "y":
        for i in range(volume.shape[1]):
            yield i, volume[:, i, :]
    elif axis == "z":
        for i in range(volume.shape[2]):
            yield i, volume[:, :, i]
    else:
        raise ValueError("axis must be one of 'x', 'y', 'z'.")


def _write_slice(volume, axis, index, data):
    if axis == "x":
        volume[index, :, :] = data
    elif axis == "y":
        volume[:, index, :] = data
    else:
        volume[:, :, index] = data


def _relabel_sequential(slice_2d, next_id):
    """Relabel a 2D slice with consecutive IDs starting at ``next_id``."""
    unique = np.unique(slice_2d)
    unique = unique[unique != 0]
    out = np.zeros_like(slice_2d, dtype=np.uint32)
    mapping = {}
    for lbl in unique:
        mapping[int(lbl)] = next_id
        out[slice_2d == lbl] = next_id
        next_id += 1
    return out, mapping, next_id


def _augment_with_unlabeled_components(slice_labels, foreground_slice):
    """
    Adds temporary local labels to foreground connected components that are
    still unlabeled (value 0) in a 2D slice.

    This allows unlabeled remnants from the 2D stage to participate in the
    same overlap-based propagation logic as regular labeled components.
    """
    if foreground_slice is None:
        return slice_labels.astype(np.uint32, copy=False)

    augmented = slice_labels.astype(np.uint32, copy=True)
    unlabeled_fg = np.asarray(foreground_slice, dtype=bool) & (augmented == 0)
    if not np.any(unlabeled_fg):
        return augmented

    # 8-connectivity in 2D to match the slicewise labeling behavior.
    structure = np.ones((3, 3), dtype=np.uint8)
    components, n_comp = cc_label(unlabeled_fg, structure=structure)
    next_local_id = int(augmented.max()) + 1
    for cid in range(1, n_comp + 1):
        augmented[components == cid] = next_local_id
        next_local_id += 1

    return augmented


def propagate_labels(
    labeled_volume,
    axis="z",
    foreground_mask=None,
    min_overlap=1,
    min_overlap_ratio=0.0,
    verbose=True,
):
    """
    Reconciles independently labeled 2D slices into a globally consistent
    3D labeling by linking components that overlap across consecutive slices.

    Args:
        labeled_volume (np.ndarray): 3D volume of per-slice labels (output of
            ``segment_volume_slicewise``).
        axis (str): Slicing axis used during the 2D segmentation
            ('x', 'y' or 'z').
        foreground_mask (np.ndarray | None): Optional 3D binary/boolean mask
            of the fiber foreground. When provided, connected components in
            foreground voxels that are unlabeled in ``labeled_volume`` are
            also propagated and receive global IDs.
        min_overlap (int): Minimum number of overlapping pixels required to
            inherit a previous label.
        min_overlap_ratio (float): Minimum overlap relative to the smaller of
            the two components (in [0, 1]) required to inherit a label.
        verbose (bool): If True, prints a tqdm progress bar.

    Returns:
        np.ndarray: 3D uint32 volume with globally unique fiber IDs.
    """
    if labeled_volume.ndim != 3:
        raise ValueError("labeled_volume must be a 3D array.")
    if foreground_mask is not None:
        if foreground_mask.shape != labeled_volume.shape:
            raise ValueError("foreground_mask must have the same shape as labeled_volume.")

    out = np.zeros(labeled_volume.shape, dtype=np.uint32)
    next_id = 1

    slice_iter = list(_iter_slices(labeled_volume, axis))
    if foreground_mask is not None:
        fg_iter = list(_iter_slices(foreground_mask, axis))
    else:
        fg_iter = None

    if verbose:
        from tqdm import tqdm

        bar = tqdm(total=len(slice_iter), desc=f"Propagating labels along {axis.upper()}")
    else:
        bar = None

    # Initialize with the first non-empty slice
    first_idx, first_slice = slice_iter[0]
    first_fg = fg_iter[0][1] if fg_iter is not None else None
    first_augmented = _augment_with_unlabeled_components(first_slice, first_fg)
    first_global, _, next_id = _relabel_sequential(first_augmented, next_id)
    _write_slice(out, axis, first_idx, first_global)
    prev_global = first_global
    if bar is not None:
        bar.update(1)

    for pos, (idx, current) in enumerate(slice_iter[1:], start=1):
        current_fg = fg_iter[pos][1] if fg_iter is not None else None
        current_augmented = _augment_with_unlabeled_components(current, current_fg)
        new_slice = np.zeros_like(current, dtype=np.uint32)

        cur_labels = np.unique(current_augmented)
        cur_labels = cur_labels[cur_labels != 0]

        # Compute overlap table: rows = current labels, cols = best prev IDs
        candidates = []  # (overlap, ratio, cur_lbl, prev_id)
        for cur_lbl in cur_labels:
            mask = current_augmented == cur_lbl
            cur_size = int(mask.sum())
            overlap_region = prev_global[mask]
            overlap_region = overlap_region[overlap_region != 0]
            if overlap_region.size == 0:
                candidates.append((0, 0.0, int(cur_lbl), 0))
                continue
            prev_ids, counts = np.unique(overlap_region, return_counts=True)
            best = np.argmax(counts)
            best_prev = int(prev_ids[best])
            best_overlap = int(counts[best])
            prev_size = int((prev_global == best_prev).sum())
            ratio = best_overlap / max(1, min(cur_size, prev_size))
            candidates.append((best_overlap, ratio, int(cur_lbl), best_prev))

        # Resolve conflicts: each previous ID can be inherited by at most one
        # current component (the one with the largest overlap).
        candidates.sort(key=lambda t: t[0], reverse=True)
        used_prev = set()
        assignments = {}
        for overlap, ratio, cur_lbl, prev_id in candidates:
            if (
                prev_id != 0
                and overlap >= min_overlap
                and ratio >= min_overlap_ratio
                and prev_id not in used_prev
            ):
                assignments[cur_lbl] = prev_id
                used_prev.add(prev_id)
            else:
                assignments[cur_lbl] = next_id
                next_id += 1

        for cur_lbl, gid in assignments.items():
            new_slice[current_augmented == cur_lbl] = gid

        _write_slice(out, axis, idx, new_slice)
        prev_global = new_slice

        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    return out
