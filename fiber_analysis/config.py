"""Pipeline parameters; defaults match the existing ``run_pipeline`` functions."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Direct3DConfig:
    """Direct 3D segmentation settings, without loading or crop options.

    Branch lengths and retraction distance are voxel counts. ``local_range``
    is the number of voxels used in the SVD direction fit when ``local_search``
    is enabled. Despite its legacy name, ``max_angle`` is a *minimum* accepted
    angle between outgoing branch vectors, in degrees. ``min_branch_length``
    filters JP--EP spurs; ``min_isolated_length`` filters EP--EP branches only
    when ``remove_isolated_branches`` is enabled.
    """

    min_branch_length: int = 5
    remove_isolated_branches: bool = True
    min_isolated_length: int = 10
    local_search: bool = True
    local_range: int = 10
    max_angle: float = 45.0
    retraction_dist: int = 4


@dataclass(frozen=True)
class SliceWiseConfig:
    """2D segmentation and adjacent-slice reconstruction settings.

    The historical slicing names are preserved: x, y, z select array
    dimensions 0, 1, 2 respectively, even for (Z, Y, X) volumes.
    ``max_angle_deviation`` is the strict maximum deviation from a straight
    pair (180 degrees). ``min_branch_length`` filters JP--EP/EP--EP paths
    before pruning; ``min_skeleton_size`` filters components after pruning.
    ``local_range`` is a pixel offset, not the 3D SVD window size.
    Reconstruction accepts overlap >= ``propagation_min_overlap`` pixels and
    overlap / min(region sizes) >= ``propagation_min_overlap_ratio``.
    """

    slicing_axis: Literal["x", "y", "z"] = "z"
    max_angle_deviation: float = 45
    remove_small_skeletons: bool = True
    min_skeleton_size: int = 20
    min_branch_length: int = 10
    local_analysis: bool = True
    local_range: int = 12
    propagation_min_overlap: int = 1
    propagation_min_overlap_ratio: float = 0.0
