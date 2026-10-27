# Toronto Fiber

Skeleton-based instance segmentation and geometric characterization of glass
fibers in micro-CT images of reinforced polypropylene. The repository compares
slice-wise 2D segmentation followed by 3D label reconstruction with segmentation
performed directly on a 3D skeleton.

These are classical image-processing pipelines, not trained machine-learning
models. Both methods can oversegment touching fibers, so a label is a detected
instance or fragment, not a guarantee of one complete physical fiber.

## Code organization

| Location | Responsibility |
| --- | --- |
| `fiber_analysis/config.py` | Immutable configurations for the two pipelines |
| `fiber_analysis/results.py` | Named intermediate volumes and final labels |
| `fiber_analysis/pipelines.py` | Array-based orchestration of existing stages |
| `fiber_analysis/metrics.py` | Numerical length and orientation measurements |
| `fiber_analysis/plotting.py` | Headless histogram rendering |
| `fiber_analysis/reporting.py` | CSV export and measurement-plus-plot reports |
| `src/utils/analysis.py` | 3D skeleton filtering, SVD directions, branch pairing and pruning |
| `src/utils/segmentation.py` | 3D skeleton markers and watershed |
| `src/utils/visualization.py` | Legacy interactive volume and histogram helpers |
| `src_2d/utils/segmentation.py` | 2D branch filtering, angular pruning and watershed |
| `src_2d/utils/reconstruction.py` | Overlap-based propagation of labels between adjacent slices |
| Both `utils/preprocessing.py` and `utils/io.py` modules | Existing image normalization, binarization and region selection helpers |
| `src/testing.py`, `src_2d/testing.py` | Compatible experiment runners: load, select ROI/cube, segment and save |
| Both `utils/metrics.py` modules | Compatibility imports from the shared measurement/report modules |
| `notebooks/` | Exploratory segmentation and planar/blockwise Hermans analyses |
| `exec_full.py` | Legacy block-processing experiment with different pruning and metrics |
| `tests/` | Regression checks for orchestration, measurements and compatibility |

`data/` and `results/` are local/ignored; they are not all included in a fresh
clone. `notebooks/fiber_seg_2D.ipynb` and
`notebooks/fiber_seg_3D.ipynb` are step-by-step methodology examples using cubic
sections of `data/PPGF40.tif` and equivalent class API calls. Their TIFF input is
memory-mapped, preserving full-volume maximum normalization while converting
only the selected cubes to uint8.
The Hermans notebook and `exec_full.py` remain specialized research analyses.

## Environment and execution

Run commands from the repository root in an environment containing NumPy,
SciPy, scikit-image, skan, pandas, tifffile, Matplotlib and tqdm. The notebooks
also use Jupyter, Plotly and other visualization libraries.

`my_vedo_env.yml` and `requirements.txt` are historical environment snapshots,
not authoritative minimal dependency manifests. In particular, the existing
2D segmentation uses `remove_small_objects(max_size=...)`, which requires
scikit-image 0.26 or later; the YAML snapshot predates that API.

The original experiment commands remain supported:

```bash
conda run --no-capture-output -n my_vedo_env python src/testing.py
conda run --no-capture-output -n my_vedo_env python src_2d/testing.py
```

Module execution also works:

```bash
conda run --no-capture-output -n my_vedo_env python -m src.testing
conda run --no-capture-output -n my_vedo_env python -m src_2d.testing
```

The settings in each script's `__main__` block still select the experiment.
These scripts write TIFF intermediates, a measurement CSV, and histogram PNGs
under their existing `results/fiber_img/` directories. They can load the entire
TIFF before selecting a cube, so memory requirements are unchanged.

The legacy `run_pipeline(...)` functions retain their parameters and positional
tuple outputs. Their existing missing-file behavior also remains: a message is
printed and a synthetic crossing is used. The array-based classes do not load
files or choose synthetic input.

## Reusable array API

Loading, region selection, segmentation, measurement and export are separate:

```python
from fiber_analysis.config import SliceWiseConfig
from fiber_analysis.metrics import measure_fibers
from fiber_analysis.pipelines import SliceWiseFiberPipeline
from fiber_analysis.reporting import analyze_and_save
from src_2d.utils.io import load_microct_image
from src_2d.utils.preprocessing import select_cube_region

volume = load_microct_image("data/PPGF40.tif")
volume = volume[:, 150:820, 129:491]  # Match this sample's legacy runner ROI.
cube = select_cube_region(volume, edge_length=100, position="center")

config = SliceWiseConfig(local_range=20)
result = SliceWiseFiberPipeline(config, verbose=True).segment(cube)

labels = result.labels
records = measure_fibers(labels, voxel_size=(1.0, 1.0, 1.0))
# Optional persistence; this measures again and creates the output directory.
report = analyze_and_save(labels, "results/example", prefix="fibers")
```

For direct 3D segmentation of an already selected grayscale array:

```python
from fiber_analysis.config import Direct3DConfig
from fiber_analysis.pipelines import Direct3DFiberPipeline

config = Direct3DConfig(
    min_branch_length=10,
    min_isolated_length=15,
    local_range=12,
    max_angle=75,
    retraction_dist=4,
)
result = Direct3DFiberPipeline(config, verbose=True).segment(cube)
records = measure_fibers(result.labels)
```

Neither class performs intensity normalization, file I/O, sample-specific ROI
selection, or cube selection. The example loader preserves the original
normalization: divide by the volume maximum, multiply by 255, cast to uint8.
The stage order is unchanged. Both constructors accept `verbose=True` (default)
for configuration, input/output dimensions, consistently numbered stages,
stage/total timings, and useful voxel/instance counts. Slice-wise progress also
identifies the physical slicing direction and includes per-slice progress bars.
Use `SliceWiseFiberPipeline(config, verbose=False)` or
`Direct3DFiberPipeline(config, verbose=False)` to disable routine messages and
nested progress bars, for example inside a block-processing loop. Warnings and
exceptions are not suppressed; `verbose` is separate from the algorithm config.

Results expose named arrays instead of positional tuples:

| Result | Arrays |
| --- | --- |
| `Direct3DResult` | `raw_volume`, `cropped_volume`, `binary_mask`, `pruned_skeleton`, `labels` |
| `SliceWiseResult` | `raw_volume`, `binary_mask`, `skeleton_3d`, `junction_mask`, `skeleton_rgb`, `slicewise_labels`, `labels` |

`raw_volume` is a copy of the input. Direct 3D processing then removes
zero-intensity margins, so its remaining arrays use the cropped coordinate
frame. Slice-wise arrays keep the input frame. The slice-wise `skeleton_3d` and
RGB junction overlay are diagnostics, not the 2D watershed markers.
Result arrays are mutable; packaging them does not add further copies.

## Methodology and preserved conventions

**Slice-wise:** global Otsu and 3D opening; then, for each slice, skeletonization,
short-branch filtering, orientation-based pruning, connected-component labeling,
small-marker removal, and watershed on the negative Euclidean distance transform.
Adjacent slices are reconciled by maximum overlap. A predecessor ID can be
inherited by at most one current region; competing regions are ranked by
absolute overlap. Unmatched regions receive new IDs.

**Direct 3D:** auto-crop, Otsu, 3D opening and skeletonization, short-branch
filtering, local SVD direction estimation (or endpoint vectors), greedy branch
pairing and retraction, then connected-component markers and 3D watershed.
Pruning uses a single topology snapshot, without recomputing after each cut.
Accepted branch pairs are not assigned separate graph identities: branches
remaining connected at a shared junction can still receive one component label.

This structural refactor deliberately does not resolve discrepancies between
the manuscript, historical experiments and current code:

| Convention | Actual preserved behavior |
| --- | --- |
| Array coordinates and metrics | `(Z, Y, X)`; spacing is `(dz, dy, dx)` |
| Legacy slice names | `"x"` selects `volume[i, :, :]`, `"y"` selects `volume[:, i, :]`, `"z"` selects `volume[:, :, i]` |
| Physical meaning of `"z"` | Under `(Z, Y, X)`, YZ planes along X, not the XY planes described in the methodology |
| 2D angular threshold | Accept the best pair only when its deviation from 180 degrees is strictly less than `max_angle_deviation` |
| 3D angular threshold | `max_angle` is passed to the existing `min_angle`: the pair angle must be at least that value |
| Reconstruction | The runners/classes do not pass the optional `foreground_mask`; unlabeled foreground is not automatically recovered |
| Sample-specific ROI | The 2D runner applies `[:, 150:820, 129:491]` only for the exact string `"data/PPGF40.tif"` |
| Class defaults | Match callable `run_pipeline` defaults, not necessarily script experiment settings or low-level helper defaults |

The 3D callable/class defaults use branch minimum 5, isolated minimum 10, local
range 10 and angle 45. Its script experiment uses 10, 15, 12 and 75 respectively.
The 2D callable/class defaults use local range 12; its script experiment uses
20, a 100-voxel center cube and a post-pruning skeleton minimum of 20.
All of these differences are retained.

## Measurements and outputs

Each nonzero label is re-skeletonized in its bounding box. Length is the greatest
physical Euclidean distance between skeleton endpoints (26-connectivity).
With fewer than two endpoints, the existing fallback measures the skeleton's
projected extent along its SVD principal axis. Degenerate skeletons retain the
historical zero length and default X direction. Curved-fiber arc length is not
computed.

Orientation uses the same endpoint direction, or the fallback principal axis.
Angles to X, Y and Z are `acos(abs(direction component))` in degrees, in
`[0, 90]`. Spacing affects both lengths and directions. Unit spacing reports
lengths in voxel units; to obtain micrometers, supply calibrated micrometer
spacing explicitly. No hardcoded 4-micrometer conversion is applied.

The CSV columns remain `label_id`, `voxel_count`, `ep_ep_length`, `dir_z`,
`dir_y`, `dir_x`, `angle_x`, `angle_y`, `angle_z`. `voxel_count` counts skeleton
voxels, not all foreground voxels.

`analyze_and_save` returns records and paths to `<prefix>_metrics.csv`,
`<prefix>_length_hist.png`, and `<prefix>_orientation_hist.png`. Orientation
histograms show percentages of instances, not length- or volume-weighted
orientation. Empty inputs produce a header-only CSV and no PNGs, as before.
Importing `fiber_analysis.metrics` does not load Matplotlib; importing plotting,
reporting, or the legacy metrics facades selects the existing Agg backend.

## Regression checks

The checks use standard-library unittest and the scientific runtime dependencies,
without requiring a separate test framework or the large micro-CT dataset:

```bash
conda run --no-capture-output -n my_vedo_env python -m unittest discover -s tests -v
```

They compare pipeline intermediates with the original low-level stage sequence,
cover all legacy slice names and nondefault settings, check runner loading/crops
and output ordering, and exercise shared measurements and report exports.
