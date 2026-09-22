"""Command-line arguments shared by ``fit_case`` and ``repair_case``.

Both entry points expose the same repair stage, so its flags are defined once here.
The three flags whose defaults legitimately differ between the two are parameters of
:func:`add_repair_arguments` rather than hard-coded values.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_case_arguments(parser: argparse.ArgumentParser) -> None:
    """Which case to process and where to write the results."""
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def add_artifact_filter_arguments(parser: argparse.ArgumentParser) -> None:
    """Component-distance filtering of false-positive islands, applied before repair."""
    group = parser.add_argument_group("artifact filter")
    group.add_argument("--mask-artifact-filter", choices=["none", "component_distance"], default="none")
    group.add_argument("--artifact-keep-near-main-mm", type=float, default=12.0)
    group.add_argument("--artifact-remove-distance-mm", type=float, default=25.0)
    group.add_argument(
        "--artifact-max-remove-voxels",
        type=int,
        default=50,
        help="Maximum size of far components to remove. Use 0 to remove all far components by distance only.",
    )
    group.add_argument("--artifact-connectivity", type=int, choices=[6, 18, 26], default=26)


def add_bridge_repair_arguments(
    parser: argparse.ArgumentParser,
    *,
    geometric_repair_method: str,
) -> None:
    """Mesh-guided bridge repair: the mesh-graph and endpoint strategies and the cleanup."""
    group = parser.add_argument_group("bridge repair")
    group.add_argument(
        "--geometric-repair-method",
        choices=["voxel_or", "sdf_cc", "component_sdf", "both", "mesh_path_connect", "mask_endpoint_connect"],
        default=geometric_repair_method,
    )
    group.add_argument("--path-repair-anchor-mm", type=float, default=2.0)
    group.add_argument("--path-repair-radius-mm", type=float, default=1.0)
    group.add_argument(
        "--path-repair-max-radius-mm",
        type=float,
        default=0.0,
        help="Maximum bridge tube radius. Use 0 for auto: anchor_mm plus one voxel.",
    )
    group.add_argument(
        "--path-repair-min-accept-radius-mm",
        type=float,
        default=0.0,
        help="Keep growing an accepted bridge until at least this radius. Use 0 to accept at path-repair-radius-mm.",
    )
    group.add_argument(
        "--path-repair-radius-mode",
        choices=["fixed", "adaptive_local"],
        default="fixed",
        help="Use a fixed bridge radius or adapt it from local component thickness.",
    )
    group.add_argument("--path-repair-local-radius-window-mm", type=float, default=6.0)
    group.add_argument("--path-repair-radius-percentile", type=float, default=80.0)
    group.add_argument("--path-repair-radius-scale", type=float, default=1.0)
    group.add_argument("--path-repair-min-component-voxels", type=int, default=20)
    group.add_argument("--path-repair-max-added-fraction", type=float, default=0.03)
    group.add_argument("--path-repair-connectivity", type=int, choices=[6, 18, 26], default=26)
    group.add_argument(
        "--path-repair-selection",
        choices=["shortest", "edge_filtered_shortest"],
        default="shortest",
        help="Bridge selector. shortest preserves the original behavior.",
    )
    group.add_argument("--path-repair-min-path-edges", type=int, default=2)
    group.add_argument(
        "--path-repair-max-mesh-edge-mm",
        type=float,
        default=0.0,
        help="For edge_filtered_shortest, ignore mesh graph edges longer than this. Use 0 to disable edge cap.",
    )
    group.add_argument(
        "--path-repair-edge-filter-fallback",
        choices=["old_shortest", "skip"],
        default="old_shortest",
        help="What to do if edge-filtered selection finds no valid bridge.",
    )
    group.add_argument(
        "--endpoint-repair-max-gap-mm",
        type=float,
        default=12.0,
        help="For mask_endpoint_connect, only bridge component endpoints this close in physical space.",
    )
    group.add_argument(
        "--endpoint-repair-mesh-support-mm",
        type=float,
        default=3.0,
        help="For mask_endpoint_connect, line samples must be this close to fitted mesh support.",
    )
    group.add_argument(
        "--endpoint-repair-min-mesh-support-fraction",
        type=float,
        default=0.50,
        help="For mask_endpoint_connect, minimum fraction of line samples close to mesh support.",
    )
    group.add_argument(
        "--post-repair-cleanup",
        choices=["none", "remove_mesh_uncovered_components"],
        default="none",
    )
    group.add_argument("--post-cleanup-mesh-distance-mm", type=float, default=2.0)
    group.add_argument("--post-cleanup-min-close-fraction", type=float, default=0.01)
    group.add_argument("--post-cleanup-max-remove-voxels", type=int, default=1000)
    group.add_argument("--post-cleanup-connectivity", type=int, choices=[6, 18, 26], default=26)


def add_voxel_repair_arguments(
    parser: argparse.ArgumentParser,
    *,
    sdf_repair_threshold_mm: float,
    voxelize_backend: str,
) -> None:
    """Whole-mesh voxelisation and signed-distance repair baselines.

    These predate the bridge-based repair used in the paper and are kept so the
    earlier comparisons stay runnable.
    """
    group = parser.add_argument_group("voxel and SDF repair baselines")
    group.add_argument("--repair-dilation-mm", type=float, default=8.0)
    group.add_argument(
        "--repair-region-method", choices=["edt_crop", "edt", "binary_dilation"], default="edt_crop"
    )
    group.add_argument("--sdf-repair-threshold-mm", type=float, default=sdf_repair_threshold_mm)
    group.add_argument("--sdf-repair-connectivity", type=int, choices=[6, 18, 26], default=26)
    group.add_argument(
        "--component-sdf-touch-original-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument("--sdf-clip-mm", type=float, default=16.0)
    group.add_argument("--save-mesh-sdf", action="store_true")
    group.add_argument("--voxelize-backend", choices=["pyvista", "multigeomed"], default=voxelize_backend)
    group.add_argument("--voxelize-margin-voxels", type=int, default=3)
    group.add_argument("--voxelize-slab-depth", type=int, default=16)


def add_repair_arguments(
    parser: argparse.ArgumentParser,
    *,
    geometric_repair_method: str,
    sdf_repair_threshold_mm: float,
    voxelize_backend: str,
) -> None:
    """Every flag of the repair stage, in one call.

    The three keyword arguments are the defaults that differ between ``fit_case``
    and ``repair_case``; everything else is identical in both.
    """
    add_artifact_filter_arguments(parser)
    add_bridge_repair_arguments(parser, geometric_repair_method=geometric_repair_method)
    add_voxel_repair_arguments(
        parser,
        sdf_repair_threshold_mm=sdf_repair_threshold_mm,
        voxelize_backend=voxelize_backend,
    )
    parser.add_argument("--disable-qa", action="store_true")
