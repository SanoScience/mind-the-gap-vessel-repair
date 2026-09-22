"""The repair stage, shared by ``fit_case`` and ``repair_case``.

``fit_case`` fits a mesh and then repairs; ``repair_case`` loads a mesh someone
already fitted and repairs. The repair itself is the same work in both, so it lives
here: filter the mask, propose and validate bridges, clean up what the mesh does not
support, write the masks out.

The filter is a separate call rather than part of :func:`run_bridge_repair` because
``fit_case`` applies it before fitting, so the fit target is the cleaned mask, while
``repair_case`` applies it immediately before repairing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .artifact_filter import filter_mask_artifacts_by_component_distance
from .endpoint_repair import repair_mask_with_endpoint_paths
from .io_utils import ImageGeometry, save_label_like, save_mask_like
from .mesh_path_repair import MeshPathRepairResult, repair_mask_with_mesh_paths
from .post_repair_cleanup import remove_mesh_uncovered_components

MESH_PATH_CONNECT = "mesh_path_connect"
MASK_ENDPOINT_CONNECT = "mask_endpoint_connect"

#: Repair methods that bridge components with the mesh instead of voxelising it.
BRIDGE_METHODS = frozenset({MESH_PATH_CONNECT, MASK_ENDPOINT_CONNECT})


@dataclass(frozen=True)
class BridgeNames:
    """Output names and the metrics key, which differ per bridge strategy."""

    metrics_key: str
    repaired: str
    bridge: str
    labels: str
    precleanup: str
    timing: str


_NAMES = {
    MESH_PATH_CONNECT: BridgeNames(
        metrics_key="mesh_path_repair",
        repaired="repaired_mask_mesh_path_connect.nii.gz",
        bridge="mesh_path_bridge_mask.nii.gz",
        labels="mesh_path_component_labels.nii.gz",
        precleanup="repaired_mask_mesh_path_connect_precleanup.nii.gz",
        timing="mesh_path_repair_seconds",
    ),
    MASK_ENDPOINT_CONNECT: BridgeNames(
        metrics_key="endpoint_repair",
        repaired="repaired_mask_endpoint_connect.nii.gz",
        bridge="endpoint_bridge_mask.nii.gz",
        labels="endpoint_component_labels.nii.gz",
        precleanup="repaired_mask_endpoint_connect_precleanup.nii.gz",
        timing="endpoint_repair_seconds",
    ),
}


def names_for(method: str) -> BridgeNames:
    """Output naming for one bridge strategy."""
    try:
        return _NAMES[method]
    except KeyError:
        raise ValueError(f"not a bridge repair method: {method!r}") from None


@dataclass
class FilterOutcome:
    """Result of the pre-repair artifact filter."""

    mask: np.ndarray
    metrics: dict | None


@dataclass
class BridgeRepairOutcome:
    """Everything the entry points need to report after repair."""

    repaired: np.ndarray
    bridge_mask: np.ndarray
    component_labels: np.ndarray
    names: BridgeNames
    repair_metrics: dict
    cleanup_metrics: dict | None


def filter_artifacts(
    mask_zyx: np.ndarray,
    geometry: ImageGeometry,
    reference_image: sitk.Image,
    output_dir: Path,
    args,
    *,
    stage: str = "repair",
) -> FilterOutcome:
    """Drop far false-positive components, and save what was removed.

    Returns the mask unchanged when the filter is off, so callers can apply this
    unconditionally. ``stage`` only names the step in the progress message:
    ``fit_case`` filters before fitting, ``repair_case`` before repairing.
    """
    if args.mask_artifact_filter != "component_distance":
        return FilterOutcome(mask=mask_zyx, metrics=None)

    print(f"Filtering distal mask artifacts before {stage}...")
    result = filter_mask_artifacts_by_component_distance(
        mask_zyx,
        geometry,
        keep_near_main_mm=args.artifact_keep_near_main_mm,
        remove_distance_mm=args.artifact_remove_distance_mm,
        max_remove_voxels=args.artifact_max_remove_voxels,
        connectivity=args.artifact_connectivity,
    )
    save_mask_like(result.cleaned_mask, reference_image, output_dir / "artifact_cleaned_mask.nii.gz")
    save_mask_like(result.removed_mask, reference_image, output_dir / "artifact_removed_mask.nii.gz")
    save_label_like(result.component_labels, reference_image, output_dir / "artifact_component_labels.nii.gz")
    print(
        "Artifact filter removed "
        f"{result.metrics['removed_voxels']} voxels "
        f"from {len(result.metrics['removed_components'])} components"
    )
    return FilterOutcome(mask=result.cleaned_mask, metrics=result.metrics)


def connect_components(
    mask_zyx: np.ndarray,
    vertices_physical_xyz: np.ndarray,
    faces: np.ndarray,
    edge_index: np.ndarray | None,
    geometry: ImageGeometry,
    args,
) -> MeshPathRepairResult:
    """Propose and validate bridges with the strategy named by ``--geometric-repair-method``."""
    if args.geometric_repair_method == MESH_PATH_CONNECT:
        print("Repairing with mesh-guided path connector...")
        return repair_mask_with_mesh_paths(
            mask_zyx,
            vertices_physical_xyz,
            faces,
            edge_index,
            geometry,
            anchor_mm=args.path_repair_anchor_mm,
            radius_mm=args.path_repair_radius_mm,
            max_radius_mm=args.path_repair_max_radius_mm,
            min_accept_radius_mm=args.path_repair_min_accept_radius_mm,
            radius_mode=args.path_repair_radius_mode,
            local_radius_window_mm=args.path_repair_local_radius_window_mm,
            radius_percentile=args.path_repair_radius_percentile,
            radius_scale=args.path_repair_radius_scale,
            min_component_voxels=args.path_repair_min_component_voxels,
            max_added_fraction=args.path_repair_max_added_fraction,
            connectivity=args.path_repair_connectivity,
            path_selection=args.path_repair_selection,
            min_path_edges=args.path_repair_min_path_edges,
            max_mesh_edge_mm=args.path_repair_max_mesh_edge_mm,
            edge_filter_fallback=args.path_repair_edge_filter_fallback,
        )

    print("Repairing with local endpoint connector...")
    return repair_mask_with_endpoint_paths(
        mask_zyx,
        vertices_physical_xyz,
        faces,
        geometry,
        max_gap_mm=args.endpoint_repair_max_gap_mm,
        mesh_support_mm=args.endpoint_repair_mesh_support_mm,
        min_mesh_support_fraction=args.endpoint_repair_min_mesh_support_fraction,
        radius_mm=args.path_repair_radius_mm,
        max_radius_mm=args.path_repair_max_radius_mm,
        min_accept_radius_mm=args.path_repair_min_accept_radius_mm,
        radius_mode=args.path_repair_radius_mode,
        local_radius_window_mm=args.path_repair_local_radius_window_mm,
        radius_percentile=args.path_repair_radius_percentile,
        radius_scale=args.path_repair_radius_scale,
        min_component_voxels=args.path_repair_min_component_voxels,
        max_added_fraction=args.path_repair_max_added_fraction,
        connectivity=args.path_repair_connectivity,
    )


def cleanup_after_repair(
    repaired: np.ndarray,
    original_mask_zyx: np.ndarray,
    vertices_physical_xyz: np.ndarray,
    faces: np.ndarray,
    geometry: ImageGeometry,
    reference_image: sitk.Image,
    output_dir: Path,
    names: BridgeNames,
    args,
) -> tuple[np.ndarray, dict | None]:
    """Remove leftover components the fitted mesh does not support.

    The repaired mask before cleanup is saved first, so both versions stay on disk.
    """
    if args.post_repair_cleanup != "remove_mesh_uncovered_components":
        return repaired, None

    print("Removing mesh-uncovered leftover components...")
    save_mask_like(repaired, reference_image, output_dir / names.precleanup)
    result = remove_mesh_uncovered_components(
        repaired,
        vertices_physical_xyz,
        faces,
        geometry,
        mesh_distance_mm=args.post_cleanup_mesh_distance_mm,
        min_close_fraction=args.post_cleanup_min_close_fraction,
        max_remove_voxels=args.post_cleanup_max_remove_voxels,
        connectivity=args.post_cleanup_connectivity,
    )
    metrics = result.metrics
    metrics["removed_original_voxels"] = int((result.removed_mask & original_mask_zyx).sum())
    save_mask_like(result.removed_mask, reference_image, output_dir / "post_cleanup_removed_mask.nii.gz")
    save_label_like(result.component_labels, reference_image, output_dir / "post_cleanup_component_labels.nii.gz")
    print(
        "Post-cleanup removed "
        f"{metrics['removed_voxels']} voxels "
        f"from {len(metrics['removed_components'])} components"
    )
    return result.cleaned_mask, metrics


def save_bridge_outputs(
    repaired: np.ndarray,
    result: MeshPathRepairResult,
    reference_image: sitk.Image,
    output_dir: Path,
    names: BridgeNames,
) -> None:
    """Write the final mask, the bridges on their own, and the component labels."""
    save_mask_like(repaired, reference_image, output_dir / "repaired_mask.nii.gz")
    save_mask_like(repaired, reference_image, output_dir / names.repaired)
    save_mask_like(result.bridge_mask, reference_image, output_dir / names.bridge)
    save_label_like(result.component_labels, reference_image, output_dir / names.labels)


def run_bridge_repair(
    mask_zyx: np.ndarray,
    vertices_physical_xyz: np.ndarray,
    faces: np.ndarray,
    edge_index: np.ndarray | None,
    geometry: ImageGeometry,
    reference_image: sitk.Image,
    output_dir: Path,
    args,
    timings: dict[str, float],
    *,
    save_timing_key: str = "save_outputs_seconds",
) -> BridgeRepairOutcome:
    """Bridge repair end to end: connect, clean up, save. Records its own timings."""
    names = names_for(args.geometric_repair_method)

    started = time.perf_counter()
    result = connect_components(mask_zyx, vertices_physical_xyz, faces, edge_index, geometry, args)
    timings[names.timing] = time.perf_counter() - started

    started = time.perf_counter()
    repaired, cleanup_metrics = cleanup_after_repair(
        result.repaired, mask_zyx, vertices_physical_xyz, faces, geometry,
        reference_image, output_dir, names, args,
    )
    if cleanup_metrics is not None:
        timings["post_repair_cleanup_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    save_bridge_outputs(repaired, result, reference_image, output_dir, names)
    timings[save_timing_key] = time.perf_counter() - started

    return BridgeRepairOutcome(
        repaired=repaired,
        bridge_mask=result.bridge_mask,
        component_labels=result.component_labels,
        names=names,
        repair_metrics=result.metrics,
        cleanup_metrics=cleanup_metrics,
    )


def growth_metrics(original_mask_zyx: np.ndarray, repaired: np.ndarray, geometry: ImageGeometry) -> dict:
    """How much foreground the repair added, in voxels and in millimetres cubed."""
    voxel_volume_mm3 = float(np.prod(geometry.spacing_xyz))
    added = int((repaired & ~original_mask_zyx).sum())
    return {
        "original_voxels": int(original_mask_zyx.sum()),
        "repaired_voxels": int(repaired.sum()),
        "added_voxels": added,
        "added_volume_mm3": float(added * voxel_volume_mm3),
        "voxel_volume_mm3": voxel_volume_mm3,
    }
