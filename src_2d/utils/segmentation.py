"""
2D fiber segmentation: per-slice skeletonization, orientation-based junction
pruning and marker-based watershed labelling.

Implementation ported from the notebook ``notebooks/fiber_seg_2D.ipynb``.
The orientation-pruning logic preserves the most colinear pair of branches at
every skeleton junction and disconnects the remaining ones.
"""

from itertools import combinations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skan import Skeleton, summarize
from skimage.measure import label
from skimage.morphology import remove_small_objects, skeletonize
from skimage.segmentation import watershed


def _branch_angle(j_coords, other_coords):
    dy = other_coords[0] - j_coords[0]
    dx = other_coords[1] - j_coords[1]
    return np.degrees(np.arctan2(-dy, dx))


def prune_skeleton_by_orientation(
    skeleton_img,
    max_angle_deviation=45,
    local_analysis=True,
    local_range=20,
):
    """
    Prunes branches at every skeleton junction by keeping the most colinear
    pair (angular sum closest to 180 degrees) and disconnecting the rest.

    Args:
        skeleton_img (np.ndarray): 2D binary skeleton image.
        max_angle_deviation (float): Maximum allowed deviation from 180 degrees
            for a pair of branches to be accepted as continuous. If the best
            pair at a junction exceeds this value, every branch at that
            junction is disconnected.
        local_analysis (bool): If True, the branch direction is estimated using
            the voxel located ``local_range`` pixels away from the junction
            along the branch. Otherwise the opposite skeleton node is used.
        local_range (int): Number of pixels used for the local direction.

    Returns:
        np.ndarray: Pruned 2D skeleton image.
    """
    skeleton_obj = Skeleton(skeleton_img)
    summary = summarize(skeleton_obj, separator="-")

    if summary.shape[0] == 0:
        return np.copy(skeleton_img)

    pruned = np.copy(skeleton_img)
    junction_ids = np.where(skeleton_obj.degrees >= 3)[0]

    for junction_id in junction_ids:
        j_coords = skeleton_obj.coordinates[junction_id]
        connected = summary[
            (summary["node-id-src"] == junction_id)
            | (summary["node-id-dst"] == junction_id)
        ]

        if len(connected) < 3:
            continue

        branches_info = []
        for _, branch in connected.iterrows():
            branch_pixels = skeleton_obj.path_coordinates(int(branch.name))
            branch_len = len(branch_pixels)
            junc_is_src = int(branch["node-id-src"] == junction_id)

            if local_analysis:
                if junc_is_src:
                    other_coords = (
                        branch_pixels[local_range]
                        if branch_len > local_range
                        else skeleton_obj.coordinates[int(branch["node-id-dst"])]
                    )
                else:
                    other_coords = (
                        branch_pixels[-(local_range - 1)]
                        if branch_len > local_range
                        else skeleton_obj.coordinates[int(branch["node-id-src"])]
                    )
            else:
                other_node_id = (
                    branch["node-id-dst"] if junc_is_src else branch["node-id-src"]
                )
                other_coords = skeleton_obj.coordinates[int(other_node_id)]

            angle = _branch_angle(j_coords, other_coords)
            branches_info.append(
                {"id": branch.name, "angle": angle, "junc_is_src": junc_is_src}
            )

        # Find pair closest to 180 degrees
        best_pair = None
        best_diff = 180.0
        for r1, r2 in combinations(branches_info, 2):
            diff = abs(r1["angle"] - r2["angle"])
            diff_from_180 = abs(diff - 180)
            if diff_from_180 > 180:
                diff_from_180 = 360 - diff_from_180
            if diff_from_180 < best_diff:
                best_diff = diff_from_180
                best_pair = {r1["id"], r2["id"]}

        if best_pair and best_diff < max_angle_deviation:
            branches_to_prune = [b for b in branches_info if b["id"] not in best_pair]
        else:
            branches_to_prune = list(branches_info)

        for branch in branches_to_prune:
            path_coords = skeleton_obj.path_coordinates(int(branch["id"]))
            if len(path_coords) <= 1:
                continue
            if branch["junc_is_src"]:
                pruned[path_coords[1, 0], path_coords[1, 1]] = 0
            else:
                pruned[path_coords[-2, 0], path_coords[-2, 1]] = 0

    return pruned


def label_fibers_watershed_2d(binary_img, labeled_skeleton):
    """
    Propagates skeleton labels to the full fiber area using marker-based
    watershed on the EDT of the binary mask.

    Args:
        binary_img (np.ndarray): 2D binary mask of the fibers.
        labeled_skeleton (np.ndarray): 2D labeled skeleton (markers).

    Returns:
        np.ndarray: 2D labeled image where every fiber pixel carries the
        identifier of its skeleton component.
    """
    distance = distance_transform_edt(binary_img)
    return watershed(-distance, markers=labeled_skeleton, mask=binary_img)


def segment_fibers_2d(
    binary_img,
    max_angle_deviation=45,
    remove_small_skeletons=True,
    min_skeleton_size=20,
    local_analysis=True,
    local_range=20,
):
    """
    Full 2D segmentation pipeline for a single slice.

    Steps:
        1. Skeletonize the binary mask.
        2. Prune skeleton junctions using orientation continuity.
        3. Label the disconnected skeleton components.
        4. Optionally drop tiny skeleton fragments.
        5. Expand the labels back to the full fiber area with watershed.

    Args:
        binary_img (np.ndarray): 2D binary mask of the fibers.
        max_angle_deviation (float): Pruning threshold (see
            ``prune_skeleton_by_orientation``).
        remove_small_skeletons (bool): Drop skeleton components below
            ``min_skeleton_size`` after pruning.
        min_skeleton_size (int): Minimum skeleton size in pixels.
        local_analysis (bool): Use local direction at the junction.
        local_range (int): Pixel offset for the local direction estimate.

    Returns:
        tuple: (labeled_fibers, labeled_skeleton)
    """
    skeleton = skeletonize(binary_img)
    pruned = prune_skeleton_by_orientation(
        skeleton,
        max_angle_deviation=max_angle_deviation,
        local_analysis=local_analysis,
        local_range=local_range,
    )

    labeled_skeleton = label(pruned, connectivity=2)
    if remove_small_skeletons:
        labeled_skeleton = remove_small_objects(
            labeled_skeleton, min_size=min_skeleton_size
        )

    labeled_fibers = label_fibers_watershed_2d(binary_img, labeled_skeleton)
    return labeled_fibers, labeled_skeleton


def segment_volume_slicewise(
    binary_volume,
    axis="z",
    max_angle_deviation=45,
    remove_small_skeletons=True,
    min_skeleton_size=20,
    local_analysis=True,
    local_range=20,
    verbose=True,
):
    """
    Runs ``segment_fibers_2d`` over every slice of a 3D binary volume along
    the selected axis. Each slice is labeled independently (labels are NOT yet
    reconciled across slices; see ``reconstruction.propagate_labels``).

    Args:
        binary_volume (np.ndarray): 3D binary mask with shape (Z, Y, X).
        axis (str): Slicing axis ('x', 'y' or 'z').
        max_angle_deviation, remove_small_skeletons, min_skeleton_size,
        local_analysis, local_range: forwarded to ``segment_fibers_2d``.
        verbose (bool): If True, prints a tqdm progress bar.

    Returns:
        tuple: (labeled_fibers_3d, labeled_skeletons_3d) as uint32 arrays.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be one of 'x', 'y', 'z'.")

    fibers_3d = np.zeros(binary_volume.shape, dtype=np.uint32)
    skeletons_3d = np.zeros(binary_volume.shape, dtype=np.uint32)

    if axis == "x":
        n_slices = binary_volume.shape[0]
    elif axis == "y":
        n_slices = binary_volume.shape[1]
    else:
        n_slices = binary_volume.shape[2]

    iterator = range(n_slices)
    if verbose:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=f"2D segmentation along {axis.upper()}")

    for i in iterator:
        if axis == "x":
            sl = binary_volume[i, :, :]
        elif axis == "y":
            sl = binary_volume[:, i, :]
        else:
            sl = binary_volume[:, :, i]

        fibers, skeletons = segment_fibers_2d(
            sl,
            max_angle_deviation=max_angle_deviation,
            remove_small_skeletons=remove_small_skeletons,
            min_skeleton_size=min_skeleton_size,
            local_analysis=local_analysis,
            local_range=local_range,
        )

        if axis == "x":
            fibers_3d[i, :, :] = fibers
            skeletons_3d[i, :, :] = skeletons
        elif axis == "y":
            fibers_3d[:, i, :] = fibers
            skeletons_3d[:, i, :] = skeletons
        else:
            fibers_3d[:, :, i] = fibers
            skeletons_3d[:, :, i] = skeletons

    return fibers_3d, skeletons_3d
