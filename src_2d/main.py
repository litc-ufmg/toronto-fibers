"""
End-to-end test runner for the 2D fiber segmentation + 3D label propagation
pipeline. Mirrors ``src/testing.py`` but uses the slice-wise 2D method
described in ``notebooks/fiber_seg_2D.ipynb``.

Pipeline
--------
1. Load the micro-CT TIFF and convert to 8-bit.
2. Optionally extract a centered cubic sub-volume of configurable edge length.
3. Binarize with Otsu and apply a 3D morphological opening.
4. Run the 2D fiber segmentation slice-by-slice along a chosen axis.
5. Reconcile the per-slice labels into a globally consistent 3D labeling
   via overlap-based label propagation across consecutive slices.
6. Persist all intermediate volumes as TIFF files for inspection.
"""

import os
import sys
from pathlib import Path

import numpy as np
import tifffile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fiber_analysis.config import SliceWiseConfig
from fiber_analysis.pipelines import (
    SliceWiseFiberPipeline,
    skeleton_with_red_junctions,
)
from src_2d.utils import io as custom_io
from src_2d.utils import metrics
from src_2d.utils import preprocessing as prep

__all__ = ["generate_synthetic_crossing", "run_pipeline", "skeleton_with_red_junctions"]


def generate_synthetic_crossing():
    """
    Builds a tiny synthetic volume with two crossing fibers. Useful as a
    fallback when no real TIFF is provided.
    """
    print("Generating synthetic 3D crossing volume...")
    vol = np.zeros((50, 50, 50), dtype=np.uint8)
    vol[23:27, 23:27, 10:40] = 255  # Fiber along X
    vol[23:27, 10:40, 23:27] = 255  # Fiber along Y
    return vol


def run_pipeline(
    image_path=None,
    cube_edge=None,
    cube_position="center",
    slicing_axis="z",
    max_angle_deviation=45,
    remove_small_skeletons=True,
    min_skeleton_size=20,
    min_branch_length=10,
    local_analysis=True,
    local_range=12,
    propagation_min_overlap=1,
    propagation_min_overlap_ratio=0.0,
):
    """
    Load/select an image and delegate to ``SliceWiseFiberPipeline``.

    The original defaults, sample-specific crop, synthetic fallback, and
    seven-array tuple are preserved. The class API returns named results.

    Args:
        image_path (str | None): Path to the micro-CT TIFF.
        cube_edge (int | None): Edge length of the cubic sub-volume in voxels.
            If None, the full image is used.
        cube_position (str): Cube location: ``"edge"``, ``"intermediate"``
            or ``"center"``.
        slicing_axis (str): Axis used for the per-slice 2D segmentation.
        max_angle_deviation (float): Threshold for the orientation pruning.
        remove_small_skeletons (bool): Drop tiny skeleton fragments per slice.
        min_skeleton_size (int): Minimum skeleton component size in pixels.
        min_branch_length (int): Minimum EP--EP and JP--EP branch length.
        local_analysis (bool): Estimate branch direction locally at junctions.
        local_range (int): Pixel offset for the local direction estimate.
        propagation_min_overlap (int): Min overlap to inherit a label.
        propagation_min_overlap_ratio (float): Min overlap ratio in [0, 1].

    Returns:
        tuple: (
            raw_extracted_volume,
            binary_volume,
            skeleton_3d,
            junction_mask,
            skeleton_rgb,
            slicewise_labels,
            global_labels,
        )
    """
    print("========================================")
    print("    STARTING 2D + PROPAGATION PIPELINE")
    print("========================================")

    # 1. LOAD IMAGE
    if image_path and os.path.exists(image_path):
        print(f"[1/5] Loading image from: {image_path}")
        volume = custom_io.load_microct_image(image_path)
    else:
        print("[1/5] Image path not provided or not found. Using synthetic data.")
        volume = generate_synthetic_crossing()

    if image_path == "data/PPGF40.tif":
        volume = volume[:, 150:820, 129:491]

    # 2. CUBIC CROP (optional)
    if cube_edge is not None:
        print(f"[2/5] Extracting {cube_position} cube of edge {cube_edge} voxels...")
        volume = prep.select_cube_region(volume, cube_edge, cube_position)
    else:
        print("[2/5] No cube_edge given. Using the full volume.")

    config = SliceWiseConfig(
        slicing_axis=slicing_axis,
        max_angle_deviation=max_angle_deviation,
        remove_small_skeletons=remove_small_skeletons,
        min_skeleton_size=min_skeleton_size,
        min_branch_length=min_branch_length,
        local_analysis=local_analysis,
        local_range=local_range,
        propagation_min_overlap=propagation_min_overlap,
        propagation_min_overlap_ratio=propagation_min_overlap_ratio,
    )
    result = SliceWiseFiberPipeline(config).segment(volume)
    return (
        result.raw_volume,
        result.binary_mask,
        result.skeleton_3d,
        result.junction_mask,
        result.skeleton_rgb,
        result.slicewise_labels,
        result.labels,
    )


if __name__ == "__main__":
    paths = [
        "data/PPGF40.tif",
        "data/PPGF40HNT0-1.0.tif",
        "data/PPGF40HNT0-1.5.tif",
        "data/PPGF40HNT0-5.tif",
    ]
    idx = 0
    sample_path = paths[idx]

    # Select "edge", "intermediate" or "center".
    cube_edge = 200
    cube_position = "center"
    slicing_axis = "z"
    max_angle = 45
    min_ske = 20
    min_branch_length = 10
    local_range = 20

    (
        raw_extracted_volume,
        binary_mask,
        skeleton_3d,
        junction_mask,
        skeleton_rgb,
        slicewise_labels,
        global_labels,
    ) = run_pipeline(
        image_path=sample_path,
        cube_edge=cube_edge,
        cube_position=cube_position,
        slicing_axis=slicing_axis,
        max_angle_deviation=max_angle,
        remove_small_skeletons=True,
        min_skeleton_size=min_ske,
        min_branch_length=min_branch_length,
        local_analysis=True,
        local_range=local_range,
        propagation_min_overlap=1,
        propagation_min_overlap_ratio=0.0,
    )

    output_dir = (
        f"./results/fiber_img/2d-res_img-{idx}_cube-{cube_edge}"
        f"_position-{cube_position}_maxAng-{max_angle}_minSke-{min_ske}"
    )
    os.makedirs(output_dir, exist_ok=True)

    out_global_path = os.path.join(output_dir, "labeled_fibers_2d_global.tif")
    out_slice_path = os.path.join(output_dir, "labeled_fibers_2d_slicewise.tif")
    out_mask_path = os.path.join(output_dir, "binary_mask_2d.tif")
    out_skeleton_path = os.path.join(output_dir, "skeleton_3d.tif")
    out_junction_mask_path = os.path.join(output_dir, "junction_mask_3d.tif")
    out_skeleton_rgb_path = os.path.join(output_dir, "skeleton_junctions_red_3d.tif")
    out_raw_cube_path = os.path.join(output_dir, "raw_extracted_cube_2d.tif")

    print(f"Saving raw extracted cube to {out_raw_cube_path}...")
    tifffile.imwrite(out_raw_cube_path, raw_extracted_volume)

    print(f"Saving global 3D labels to {out_global_path}...")
    tifffile.imwrite(out_global_path, global_labels.astype(np.uint32))

    print(f"Saving per-slice labels to {out_slice_path}...")
    tifffile.imwrite(out_slice_path, slicewise_labels.astype(np.uint32))

    print(f"Saving binary mask to {out_mask_path}...")
    tifffile.imwrite(out_mask_path, (binary_mask.astype(np.uint8) * 255))

    print(f"Saving 3D skeleton to {out_skeleton_path}...")
    tifffile.imwrite(out_skeleton_path, (skeleton_3d.astype(np.uint8) * 255))

    print(f"Saving junction mask to {out_junction_mask_path}...")
    tifffile.imwrite(out_junction_mask_path, (junction_mask.astype(np.uint8) * 255))

    print(f"Saving skeleton RGB (junctions in red) to {out_skeleton_rgb_path}...")
    tifffile.imwrite(out_skeleton_rgb_path, skeleton_rgb)

    print("Computing per-fiber metrics and histograms...")
    metric_outputs = metrics.analyze_and_save(
        global_labels,
        output_dir,
        prefix="labeled_fibers_2d_global",
        voxel_size=(1.0, 1.0, 1.0),
    )
    print(f"Saved metrics CSV to {metric_outputs['csv_path']}")
    print(f"Saved length histogram to {metric_outputs['length_hist_path']}")
    print(f"Saved orientation histogram to {metric_outputs['orientation_hist_path']}")

    print("\n=======================================================================")
    print("Open the global labels in Napari with:")
    print(f"  napari {out_global_path} --layer-type labels")
    print("=======================================================================\n")
