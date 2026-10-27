"""Array-based orchestration of the existing segmentation algorithms.

The classes perform no file I/O, intensity normalization, or sample-specific
ROI/cube selection. Supply the grayscale array to analyze (the legacy runners
normalize TIFFs to uint8). Measuring and exporting results are separate steps.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

import numpy as np
from scipy.ndimage import binary_opening, convolve
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from src.utils import analysis
from src.utils import preprocessing as prep3d
from src.utils import segmentation as seg3d
from src_2d.utils import preprocessing as prep2d
from src_2d.utils import reconstruction as recon
from src_2d.utils import segmentation as seg2d

from .config import Direct3DConfig, SliceWiseConfig
from .results import Direct3DResult, SliceWiseResult


@contextmanager
def _stage(message: str, *, verbose: bool) -> Iterator[None]:
    if not verbose:
        yield
        return
    print(message, flush=True)
    started = perf_counter()
    yield
    print(f"      Completed in {perf_counter() - started:.2f} s.", flush=True)


def skeleton_with_red_junctions(
    binary_volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the existing 3D diagnostic skeleton and red-junction RGB overlay.

    Returns the boolean skeleton, boolean mask of voxels with >= 3 neighbors
    (26-connectivity), and uint8 RGB volume with a white skeleton/red junctions.
    """
    skeleton_3d = skeletonize(binary_volume).astype(bool)

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = convolve(
        skeleton_3d.astype(np.uint8), kernel, mode="constant", cval=0
    )
    junction_mask = skeleton_3d & (neighbor_count >= 3)

    skeleton_rgb = np.zeros(skeleton_3d.shape + (3,), dtype=np.uint8)
    skeleton_rgb[skeleton_3d] = (255, 255, 255)
    skeleton_rgb[junction_mask] = (255, 0, 0)

    return skeleton_3d, junction_mask, skeleton_rgb


class Direct3DFiberPipeline:
    """Preprocess, prune, and label a grayscale volume directly in 3D.

    ``verbose=True`` reports configuration, numbered stages, counts and timings.
    ``verbose=False`` disables progress, including output from nested helpers.
    Warnings and exceptions remain visible.
    """

    def __init__(
        self, config: Direct3DConfig | None = None, *, verbose: bool = True
    ) -> None:
        self.config = config if config is not None else Direct3DConfig()
        self.verbose = verbose

    def segment(self, volume: np.ndarray) -> Direct3DResult:
        """Segment an already selected volume without modifying it.

        Preserve the original order: auto-crop, Otsu, 3D opening,
        skeletonization, short-branch filtering, pruning, and watershed.
        Pruning uses the existing single topology snapshot and greedy pairing.
        """
        started = perf_counter()
        config = self.config
        verbose = self.verbose
        if verbose:
            print(
                f"Direct3D | input (Z, Y, X): {volume.shape} | dtype: {volume.dtype}\n"
                f"Configuration: {config}",
                flush=True,
            )
        raw_extracted_volume = volume.copy()

        with _stage(
            "[Direct3D 1/6] Auto-cropping zero-intensity margins", verbose=verbose
        ):
            cropped_volume = prep3d.auto_crop_volume(volume, verbose=verbose)

        with _stage("[Direct3D 2/6] Global Otsu and 3D opening", verbose=verbose):
            thresh = threshold_otsu(cropped_volume)
            binary_image = cropped_volume > thresh
            binary_image = binary_opening(binary_image)
            if verbose:
                print(
                    f"      Otsu threshold: {thresh} | "
                    f"Foreground after opening: {int(binary_image.sum())} voxels",
                    flush=True,
                )

        with _stage("[Direct3D 3/6] Extracting the 3D skeleton", verbose=verbose):
            skeleton_3d = skeletonize(binary_image)
            if verbose:
                print(f"      Skeleton: {int(skeleton_3d.sum())} voxels", flush=True)

        with _stage("[Direct3D 4/6] Filtering short branches", verbose=verbose):
            skeleton_3d = analysis.filter_skeleton(
                skeleton_3d,
                tam_min=config.min_branch_length,
                remove_isolated=config.remove_isolated_branches,
                isolated_min=config.min_isolated_length,
                verbose=verbose,
            )
            if verbose:
                print(
                    f"      Filtered skeleton: {int(skeleton_3d.sum())} voxels",
                    flush=True,
                )

        with _stage("[Direct3D 5/6] Pruning junctions by continuity", verbose=verbose):
            pruned_skeleton = analysis.prune_skeleton(
                skeleton_3d,
                local_search=config.local_search,
                local_range=config.local_range,
                min_angle=config.max_angle,
                retraction_dist=config.retraction_dist,
                verbose=verbose,
            )
            if verbose:
                print(
                    f"      Pruned skeleton: {int(pruned_skeleton.sum())} voxels",
                    flush=True,
                )

        with _stage("[Direct3D 6/6] Marker-based 3D watershed", verbose=verbose):
            labeled_fibers = seg3d.label_fibers_watershed(
                binary_image, pruned_skeleton, verbose=verbose
            )

        if verbose:
            num_fibers = int(np.count_nonzero(np.unique(labeled_fibers)))
            print(
                f"Direct3D complete: {num_fibers} instances | "
                f"labels (Z, Y, X): {labeled_fibers.shape} | "
                f"total: {perf_counter() - started:.2f} s",
                flush=True,
            )

        return Direct3DResult(
            raw_volume=raw_extracted_volume,
            cropped_volume=cropped_volume,
            binary_mask=binary_image,
            pruned_skeleton=pruned_skeleton,
            labels=labeled_fibers,
        )


class SliceWiseFiberPipeline:
    """Segment independent 2D slices and reconcile adjacent labels by overlap.

    ``verbose=True`` reports configuration, stages, timings and slice progress.
    ``verbose=False`` disables progress messages and both nested progress bars.
    Warnings and exceptions remain visible.
    """

    def __init__(
        self, config: SliceWiseConfig | None = None, *, verbose: bool = True
    ) -> None:
        self.config = config if config is not None else SliceWiseConfig()
        self.verbose = verbose

    def segment(self, volume: np.ndarray) -> SliceWiseResult:
        """Segment an already selected volume without modifying it.

        Otsu and opening operate in 3D before slice-wise segmentation. The
        diagnostic 3D skeleton does not seed watershed. Reconstruction keeps
        the legacy axis mapping and does not augment unlabeled foreground.
        """
        started = perf_counter()
        config = self.config
        verbose = self.verbose
        if verbose:
            print(
                f"SliceWise | input (Z, Y, X): {volume.shape} | dtype: {volume.dtype}\n"
                f"Configuration: {config}",
                flush=True,
            )
        raw_extracted_volume = volume.copy()

        with _stage("[SliceWise 1/4] Global Otsu and 3D opening", verbose=verbose):
            binary_volume, threshold = prep2d.binarize_volume_otsu(volume)
            binary_volume = binary_opening(binary_volume)
            if verbose:
                print(
                    f"      Otsu threshold: {threshold} | "
                    f"Foreground after opening: {int(binary_volume.sum())} voxels",
                    flush=True,
                )

        with _stage(
            "[SliceWise 2/4] Diagnostic 3D skeleton and junctions", verbose=verbose
        ):
            skeleton_3d, junction_mask, skeleton_rgb = skeleton_with_red_junctions(
                binary_volume
            )
            if verbose:
                print(
                    f"      Skeleton: {int(skeleton_3d.sum())} voxels | "
                    f"Junctions: {int(junction_mask.sum())} voxels "
                    "(diagnostics only, not watershed markers)",
                    flush=True,
                )

        with _stage(
            "[SliceWise 3/4] Segmenting independent 2D slices", verbose=verbose
        ):
            if verbose and config.slicing_axis in ("x", "y", "z"):
                dimension = ("x", "y", "z").index(config.slicing_axis)
                physical_axis, plane = (("Z", "XY"), ("Y", "XZ"), ("X", "YZ"))[
                    dimension
                ]
                print(
                    f"      {volume.shape[dimension]} {plane} planes along {physical_axis} "
                    f"| array dimension: {dimension} | legacy axis: '{config.slicing_axis}'",
                    flush=True,
                )
            slicewise_fibers, _ = seg2d.segment_volume_slicewise(
                binary_volume,
                axis=config.slicing_axis,
                max_angle_deviation=config.max_angle_deviation,
                remove_small_skeletons=config.remove_small_skeletons,
                min_skeleton_size=config.min_skeleton_size,
                min_branch_length=config.min_branch_length,
                local_analysis=config.local_analysis,
                local_range=config.local_range,
                verbose=verbose,
            )

        with _stage(
            "[SliceWise 4/4] Reconstructing 3D labels by overlap", verbose=verbose
        ):
            global_labels = recon.propagate_labels(
                slicewise_fibers,
                axis=config.slicing_axis,
                min_overlap=config.propagation_min_overlap,
                min_overlap_ratio=config.propagation_min_overlap_ratio,
                verbose=verbose,
            )

        if verbose:
            n_fibers = int(np.count_nonzero(np.unique(global_labels)))
            print(
                f"SliceWise complete: {n_fibers} instances | "
                f"labels (Z, Y, X): {global_labels.shape} | "
                f"total: {perf_counter() - started:.2f} s",
                flush=True,
            )

        return SliceWiseResult(
            raw_volume=raw_extracted_volume,
            binary_mask=binary_volume,
            skeleton_3d=skeleton_3d,
            junction_mask=junction_mask,
            skeleton_rgb=skeleton_rgb,
            slicewise_labels=slicewise_fibers,
            labels=global_labels,
        )
