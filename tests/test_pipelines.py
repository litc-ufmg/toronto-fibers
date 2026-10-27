"""Characterize orchestration against the original low-level stage sequence."""

import contextlib
import io
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
from scipy.ndimage import binary_opening, convolve
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from fiber_analysis.config import Direct3DConfig, SliceWiseConfig
from fiber_analysis.pipelines import Direct3DFiberPipeline, SliceWiseFiberPipeline
from src import testing as runner3d
from src.utils import analysis
from src.utils import preprocessing as prep3d
from src.utils import segmentation as seg3d
from src_2d import testing as runner2d
from src_2d.utils import io as image_io
from src_2d.utils import preprocessing as prep2d
from src_2d.utils import reconstruction
from src_2d.utils import segmentation as seg2d


def crossing_volume():
    volume = np.full((32, 36, 7), 5, dtype=np.uint16)
    volume[14:18, 2:34, :] = 250
    volume[2:30, 16:20, :] = 250
    return volume


def reference_3d(volume, config):
    """Stage order and keyword mapping from the original 3D runner."""
    cropped = prep3d.auto_crop_volume(volume)
    binary = binary_opening(cropped > threshold_otsu(cropped))
    skeleton = analysis.filter_skeleton(
        skeletonize(binary),
        tam_min=config.min_branch_length,
        remove_isolated=config.remove_isolated_branches,
        isolated_min=config.min_isolated_length,
    )
    pruned = analysis.prune_skeleton(
        skeleton,
        local_search=config.local_search,
        local_range=config.local_range,
        min_angle=config.max_angle,
        retraction_dist=config.retraction_dist,
    )
    return volume, cropped, binary, pruned, seg3d.label_fibers_watershed(binary, pruned)


def reference_slicewise(volume, config):
    """Stage order and keyword mapping from the original 2D runner."""
    binary, _ = prep2d.binarize_volume_otsu(volume)
    binary = binary_opening(binary)
    skeleton = skeletonize(binary).astype(bool)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    junctions = skeleton & (
        convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0) >= 3
    )
    rgb = np.zeros(skeleton.shape + (3,), dtype=np.uint8)
    rgb[skeleton] = (255, 255, 255)
    rgb[junctions] = (255, 0, 0)
    slices, _ = seg2d.segment_volume_slicewise(
        binary,
        axis=config.slicing_axis,
        max_angle_deviation=config.max_angle_deviation,
        remove_small_skeletons=config.remove_small_skeletons,
        min_skeleton_size=config.min_skeleton_size,
        min_branch_length=config.min_branch_length,
        local_analysis=config.local_analysis,
        local_range=config.local_range,
        verbose=False,
    )
    labels = reconstruction.propagate_labels(
        slices,
        axis=config.slicing_axis,
        min_overlap=config.propagation_min_overlap,
        min_overlap_ratio=config.propagation_min_overlap_ratio,
        verbose=False,
    )
    return volume, binary, skeleton, junctions, rgb, slices, labels


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def assert_arrays_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for array, reference in zip(actual, expected):
            self.assertEqual(array.dtype, reference.dtype)
            np.testing.assert_array_equal(array, reference)

    def test_direct_pipeline_preserves_stages_and_configuration(self):
        volume = crossing_volume()
        original = volume.copy()
        for config in (
            Direct3DConfig(),
            Direct3DConfig(
                min_branch_length=3,
                remove_isolated_branches=False,
                min_isolated_length=15,
                local_search=False,
                local_range=6,
                max_angle=100,
                retraction_dist=2,
            ),
        ):
            with self.subTest(config=config):
                result = Direct3DFiberPipeline(config).segment(volume)
                self.assert_arrays_equal(
                    (
                        result.raw_volume,
                        result.cropped_volume,
                        result.binary_mask,
                        result.pruned_skeleton,
                        result.labels,
                    ),
                    reference_3d(volume, config),
                )
                self.assertFalse(np.shares_memory(result.raw_volume, volume))
                np.testing.assert_array_equal(volume, original)

    def test_slicewise_pipeline_preserves_all_legacy_axes_and_settings(self):
        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            volume = np.moveaxis(crossing_volume(), 2, index)
            original = volume.copy()
            config = SliceWiseConfig(
                slicing_axis=axis,
                max_angle_deviation=60,
                min_skeleton_size=3,
                min_branch_length=3,
                local_analysis=axis != "y",
                local_range=6,
                propagation_min_overlap=2,
                propagation_min_overlap_ratio=0.25,
            )
            with self.subTest(axis=axis):
                result = SliceWiseFiberPipeline(config).segment(volume)
                self.assert_arrays_equal(
                    (
                        result.raw_volume,
                        result.binary_mask,
                        result.skeleton_3d,
                        result.junction_mask,
                        result.skeleton_rgb,
                        result.slicewise_labels,
                        result.labels,
                    ),
                    reference_slicewise(volume, config),
                )
                self.assertFalse(np.shares_memory(result.raw_volume, volume))
                np.testing.assert_array_equal(volume, original)

    def test_pipeline_verbosity_controls_nested_output_without_changing_arrays(self):
        cases = [
            (Direct3DFiberPipeline, Direct3DConfig(), crossing_volume(), "Direct3D", 6)
        ]
        for axis, dimension in (("x", 0), ("y", 1), ("z", 2)):
            cases.append(
                (
                    SliceWiseFiberPipeline,
                    SliceWiseConfig(slicing_axis=axis),
                    np.moveaxis(crossing_volume(), 2, dimension),
                    "SliceWise",
                    4,
                )
            )
        for pipeline_class, config, volume, name, stages in cases:
            with self.subTest(pipeline=name, config=config):
                loud_out, loud_err = io.StringIO(), io.StringIO()
                with (
                    contextlib.redirect_stdout(loud_out),
                    contextlib.redirect_stderr(loud_err),
                ):
                    loud = pipeline_class(config).segment(volume)
                quiet_out, quiet_err = io.StringIO(), io.StringIO()
                with (
                    contextlib.redirect_stdout(quiet_out),
                    contextlib.redirect_stderr(quiet_err),
                ):
                    quiet = pipeline_class(config, verbose=False).segment(volume)

                self.assertEqual(quiet_out.getvalue(), "")
                self.assertEqual(quiet_err.getvalue(), "")
                self.assert_arrays_equal(
                    tuple(getattr(quiet, field.name) for field in fields(quiet)),
                    tuple(getattr(loud, field.name) for field in fields(loud)),
                )
                output = loud_out.getvalue()
                self.assertEqual(
                    re.findall(rf"\[{name} (\d+)/{stages}\]", output),
                    [str(step) for step in range(1, stages + 1)],
                )
                self.assertIn(f"input (Z, Y, X): {volume.shape}", output)
                self.assertIn("Otsu threshold:", output)
                self.assertIn("Configuration:", output)
                count = np.count_nonzero(np.unique(loud.labels))
                self.assertIn(f"{name} complete: {count} instances", output)
                self.assertRegex(output, r"Completed in \d+\.\d{2} s\.")
                self.assertRegex(output, r"total: \d+\.\d{2} s")
                if name == "SliceWise":
                    dimension = ("x", "y", "z").index(config.slicing_axis)
                    physical_axis, plane = (("Z", "XY"), ("Y", "XZ"), ("X", "YZ"))[
                        dimension
                    ]
                    self.assertIn(
                        f"{volume.shape[dimension]} {plane} planes along {physical_axis}",
                        output,
                    )
                    self.assertIn("2D segmentation", loud_err.getvalue())
                    self.assertIn("Propagating labels", loud_err.getvalue())

    def test_verbose_summary_counts_present_ids_without_assuming_background(self):
        volume = crossing_volume()
        labels = np.full(volume.shape, 2, dtype=np.uint32)
        labels[volume.shape[0] // 2 :] = 7
        for pipeline_class, module, operation, name in (
            (Direct3DFiberPipeline, seg3d, "label_fibers_watershed", "Direct3D"),
            (SliceWiseFiberPipeline, reconstruction, "propagate_labels", "SliceWise"),
        ):
            with self.subTest(pipeline=name):
                output = io.StringIO()
                with (
                    contextlib.redirect_stdout(output),
                    patch.object(module, operation, return_value=labels),
                ):
                    result = pipeline_class().segment(volume)
                self.assertIs(result.labels, labels)
                self.assertIn(f"{name} complete: 2 instances", output.getvalue())

    def test_verbosity_does_not_hide_errors_or_report_false_completion(self):
        for pipeline_class, operation in (
            (Direct3DFiberPipeline, "fiber_analysis.pipelines.threshold_otsu"),
            (
                SliceWiseFiberPipeline,
                "fiber_analysis.pipelines.prep2d.binarize_volume_otsu",
            ),
        ):
            for verbose in (False, True):
                with self.subTest(pipeline=pipeline_class.__name__, verbose=verbose):
                    output, errors = io.StringIO(), io.StringIO()
                    with (
                        contextlib.redirect_stdout(output),
                        contextlib.redirect_stderr(errors),
                        patch(operation, side_effect=ValueError("Otsu failed")),
                        self.assertRaisesRegex(ValueError, "Otsu failed"),
                    ):
                        pipeline_class(verbose=verbose).segment(crossing_volume())
                    self.assertNotIn(" complete:", output.getvalue())
                    self.assertEqual(errors.getvalue(), "")
                    if not verbose:
                        self.assertEqual(output.getvalue(), "")

    def test_legacy_runners_preserve_loading_crops_and_tuple_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crossing.tif"
            tifffile.imwrite(path, crossing_volume())
            loaded = image_io.load_microct_image(path)
            config3d = Direct3DConfig(min_branch_length=3, local_range=6)
            self.assert_arrays_equal(
                runner3d.run_pipeline(str(path), subcube_size=28, **asdict(config3d)),
                reference_3d(loaded[2:30, 4:32, :], config3d),
            )
            config2d = SliceWiseConfig(min_skeleton_size=3, min_branch_length=3)
            cropped = prep2d.select_cube_region(loaded, 28, "intermediate")
            self.assert_arrays_equal(
                runner2d.run_pipeline(
                    str(path),
                    cube_edge=28,
                    cube_position="intermediate",
                    **asdict(config2d),
                ),
                reference_slicewise(cropped, config2d),
            )

    def test_legacy_sample_roi_still_depends_on_exact_path_string(self):
        volume = np.arange(10 * 830 * 500, dtype=np.uint32).reshape(10, 830, 500)
        with (
            patch.object(runner2d.os.path, "exists", return_value=True),
            patch.object(runner2d.custom_io, "load_microct_image", return_value=volume),
            patch.object(SliceWiseFiberPipeline, "segment") as segment,
        ):
            runner2d.run_pipeline("data/PPGF40.tif")
            np.testing.assert_array_equal(
                segment.call_args.args[0], volume[:, 150:820, 129:491]
            )
            runner2d.run_pipeline(Path("data/PPGF40.tif"))
            self.assertIs(segment.call_args.args[0], volume)

    def test_legacy_synthetic_fallback_does_not_apply_3d_file_crop(self):
        volume = runner3d.generate_synthetic_crossing()
        self.assert_arrays_equal(
            runner3d.run_pipeline(image_path="missing.tif", subcube_size=10),
            reference_3d(volume, Direct3DConfig()),
        )

    def test_script_imports_work_outside_repository_and_without_utils_alias(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            for directory in ("src", "src_2d"):
                with self.subTest(directory=directory):
                    path = str(root / directory / "testing.py")
                    code = (
                        f"import runpy; namespace = runpy.run_path({path!r}); "
                        "assert callable(namespace['run_pipeline'])"
                    )
                    process = subprocess.run(
                        [sys.executable, "-c", code],
                        cwd=tmp,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
