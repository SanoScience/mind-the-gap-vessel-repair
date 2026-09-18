from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .io_utils import ImageGeometry
from .mesh_path_repair import (
    MeshPathRepairResult,
    _component_sizes,
    _connectivity_structure,
    _local_component_radius_mm,
    _rasterize_path_tube,
)


@dataclass(frozen=True)
class _EndpointCandidate:
    component_label: int
    endpoint_distance_mm: float
    main_point_xyz: np.ndarray
    candidate_point_xyz: np.ndarray
    main_voxel_zyx: np.ndarray
    candidate_voxel_zyx: np.ndarray
    mesh_support_fraction: float
    mesh_support_min_mm: float
    mesh_support_median_mm: float
    mesh_support_max_mm: float


def _surface_voxels(mask_zyx: np.ndarray, structure: np.ndarray) -> np.ndarray:
    mask = mask_zyx.astype(bool, copy=False)
    if not mask.any():
        return np.empty((0, 3), dtype=np.int64)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    boundary = mask & ~eroded
    voxels = np.argwhere(boundary)
    if voxels.size == 0:
        voxels = np.argwhere(mask)
    return voxels.astype(np.int64, copy=False)


def _component_surface_cache(
    component_labels: np.ndarray,
    component_order: list[int],
    structure: np.ndarray,
    geometry: ImageGeometry,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    mask = component_labels > 0
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    boundary = mask & ~eroded
    boundary_voxels = np.argwhere(boundary).astype(np.int64, copy=False)
    boundary_labels = (
        component_labels[boundary_voxels[:, 0], boundary_voxels[:, 1], boundary_voxels[:, 2]]
        if boundary_voxels.size
        else np.empty((0,), dtype=component_labels.dtype)
    )

    surface_voxels_by_label: dict[int, np.ndarray] = {}
    surface_points_by_label: dict[int, np.ndarray] = {}
    for label in component_order:
        label = int(label)
        voxels = boundary_voxels[boundary_labels == label] if boundary_voxels.size else np.empty((0, 3), dtype=np.int64)
        if voxels.size == 0:
            voxels = np.argwhere(component_labels == label).astype(np.int64, copy=False)
        surface_voxels_by_label[label] = voxels
        surface_points_by_label[label] = (
            _voxels_to_physical_xyz(voxels, geometry) if voxels.size else np.empty((0, 3), dtype=np.float64)
        )
    return surface_voxels_by_label, surface_points_by_label


def _voxels_to_physical_xyz(voxels_zyx: np.ndarray, geometry: ImageGeometry) -> np.ndarray:
    index_xyz = voxels_zyx[:, [2, 1, 0]].astype(np.float64, copy=False)
    return geometry.continuous_index_xyz_to_physical(index_xyz)


def _line_samples_xyz(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    step_mm: float,
) -> np.ndarray:
    start = np.asarray(start_xyz, dtype=np.float64)
    end = np.asarray(end_xyz, dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    steps = max(int(np.ceil(length / max(float(step_mm), 1e-6))), 1)
    t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
    return start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]


def _mesh_support_points(vertices_physical_xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices_physical_xyz, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return vertices
    valid = np.all((faces >= 0) & (faces < vertices.shape[0]), axis=1)
    if not np.any(valid):
        return vertices
    centroids = vertices[faces[valid]].mean(axis=1)
    return np.concatenate([vertices, centroids], axis=0)


def _nearest_endpoint_pair(
    main_voxels: np.ndarray,
    main_points: np.ndarray,
    main_tree: cKDTree,
    candidate_voxels: np.ndarray,
    candidate_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if (
        main_voxels.size == 0
        or main_points.size == 0
        or candidate_voxels.size == 0
        or candidate_points.size == 0
    ):
        return None
    try:
        distances, nearest = main_tree.query(candidate_points, k=1, workers=-1)
    except TypeError:  # pragma: no cover - older scipy compatibility
        distances, nearest = main_tree.query(candidate_points, k=1)
    best_idx = int(np.argmin(distances))
    main_idx = int(nearest[best_idx])
    return (
        float(distances[best_idx]),
        main_points[main_idx].astype(np.float32, copy=False),
        candidate_points[best_idx].astype(np.float32, copy=False),
        main_voxels[main_idx].astype(np.int64, copy=False),
        candidate_voxels[best_idx].astype(np.int64, copy=False),
    )


def _evaluate_mesh_support(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    support_tree: cKDTree,
    geometry: ImageGeometry,
    support_mm: float,
) -> tuple[float, float, float, float]:
    min_spacing = float(np.min(np.asarray(geometry.spacing_xyz, dtype=float)))
    samples = _line_samples_xyz(start_xyz, end_xyz, step_mm=max(0.5 * min_spacing, 0.25))
    try:
        distances, _nearest = support_tree.query(samples, k=1, workers=-1)
    except TypeError:  # pragma: no cover
        distances, _nearest = support_tree.query(samples, k=1)
    distances = np.asarray(distances, dtype=np.float64)
    if distances.size == 0:
        return 0.0, float("inf"), float("inf"), float("inf")
    support_fraction = float(np.mean(distances <= float(support_mm)))
    return (
        support_fraction,
        float(np.min(distances)),
        float(np.median(distances)),
        float(np.max(distances)),
    )


def _find_best_endpoint_candidate(
    component_labels: np.ndarray,
    component_sizes: np.ndarray,
    main_labels: set[int],
    candidate_labels: list[int],
    geometry: ImageGeometry,
    structure: np.ndarray,
    support_tree: cKDTree,
    surface_voxels_by_label: dict[int, np.ndarray],
    surface_points_by_label: dict[int, np.ndarray],
    *,
    max_gap_mm: float,
    mesh_support_mm: float,
    min_mesh_support_fraction: float,
) -> tuple[_EndpointCandidate | None, list[dict[str, object]]]:
    main_voxel_chunks = [surface_voxels_by_label[label] for label in main_labels if label in surface_voxels_by_label]
    main_point_chunks = [surface_points_by_label[label] for label in main_labels if label in surface_points_by_label]
    main_voxels = np.concatenate(main_voxel_chunks, axis=0) if main_voxel_chunks else np.empty((0, 3), dtype=np.int64)
    main_points = (
        np.concatenate(main_point_chunks, axis=0) if main_point_chunks else np.empty((0, 3), dtype=np.float64)
    )
    if main_voxels.size == 0:
        return None, [
            {
                "component_label": int(candidate_label),
                "component_voxels": int(component_sizes[int(candidate_label)]),
                "status": "rejected_no_main_endpoint_surface",
            }
            for candidate_label in candidate_labels
        ]
    main_tree = cKDTree(main_points)
    rejections: list[dict[str, object]] = []
    endpoint_options: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for candidate_label in candidate_labels:
        candidate_voxels = surface_voxels_by_label.get(int(candidate_label))
        candidate_points = surface_points_by_label.get(int(candidate_label))
        if candidate_voxels is None or candidate_points is None:
            rejections.append(
                {
                    "component_label": int(candidate_label),
                    "component_voxels": int(component_sizes[int(candidate_label)]),
                    "status": "rejected_no_candidate_endpoint_surface",
                }
            )
            continue
        pair = _nearest_endpoint_pair(
            main_voxels,
            main_points,
            main_tree,
            candidate_voxels,
            candidate_points,
        )
        if pair is None:
            rejections.append(
                {
                    "component_label": int(candidate_label),
                    "component_voxels": int(component_sizes[int(candidate_label)]),
                    "status": "rejected_no_endpoint_pair",
                }
            )
            continue
        distance, main_point, candidate_point, main_voxel, candidate_voxel = pair
        if distance > float(max_gap_mm):
            rejections.append(
                {
                    "component_label": int(candidate_label),
                    "component_voxels": int(component_sizes[int(candidate_label)]),
                    "status": "rejected_endpoint_gap_too_large",
                    "endpoint_distance_mm": float(distance),
                    "max_gap_mm": float(max_gap_mm),
                }
            )
            continue
        endpoint_options.append(
            (
                float(distance),
                int(candidate_label),
                main_point,
                candidate_point,
                main_voxel,
                candidate_voxel,
            )
        )

    endpoint_options.sort(key=lambda item: item[0])
    for distance, candidate_label, main_point, candidate_point, main_voxel, candidate_voxel in endpoint_options:
        support_fraction, support_min, support_median, support_max = _evaluate_mesh_support(
            main_point,
            candidate_point,
            support_tree,
            geometry,
            support_mm=float(mesh_support_mm),
        )
        if support_fraction < float(min_mesh_support_fraction):
            rejections.append(
                {
                    "component_label": int(candidate_label),
                    "component_voxels": int(component_sizes[int(candidate_label)]),
                    "status": "rejected_insufficient_mesh_support",
                    "endpoint_distance_mm": float(distance),
                    "mesh_support_fraction": float(support_fraction),
                    "min_mesh_support_fraction": float(min_mesh_support_fraction),
                    "mesh_support_mm": float(mesh_support_mm),
                    "mesh_support_min_mm": float(support_min),
                    "mesh_support_median_mm": float(support_median),
                    "mesh_support_max_mm": float(support_max),
                }
            )
            continue
        best = _EndpointCandidate(
            component_label=int(candidate_label),
            endpoint_distance_mm=float(distance),
            main_point_xyz=main_point,
            candidate_point_xyz=candidate_point,
            main_voxel_zyx=main_voxel,
            candidate_voxel_zyx=candidate_voxel,
            mesh_support_fraction=float(support_fraction),
            mesh_support_min_mm=float(support_min),
            mesh_support_median_mm=float(support_median),
            mesh_support_max_mm=float(support_max),
        )
        return best, rejections
    return None, rejections


def repair_mask_with_endpoint_paths(
    original_mask_zyx: np.ndarray,
    vertices_physical_xyz: np.ndarray,
    faces: np.ndarray,
    geometry: ImageGeometry,
    *,
    max_gap_mm: float = 12.0,
    mesh_support_mm: float = 3.0,
    min_mesh_support_fraction: float = 0.50,
    radius_mm: float = 0.4,
    max_radius_mm: float | None = None,
    min_accept_radius_mm: float | None = None,
    radius_mode: str = "fixed",
    local_radius_window_mm: float = 6.0,
    radius_percentile: float = 70.0,
    radius_scale: float = 0.8,
    min_component_voxels: int = 1,
    max_added_fraction: float = 0.03,
    connectivity: int = 26,
) -> MeshPathRepairResult:
    """Conservatively bridge nearest mask endpoints with mesh support validation.

    Unlike ``mesh_path_connect``, this does not follow mesh graph edges. It
    chooses the closest endpoint-to-endpoint gap between connected components in
    the mask, then checks that the straight local bridge is supported by nearby
    fitted-mesh vertices/face centroids.
    """
    original = original_mask_zyx.astype(bool, copy=False)
    radius_mode = str(radius_mode).strip().lower()
    if radius_mode not in {"fixed", "adaptive_local"}:
        raise ValueError("radius_mode must be one of: fixed, adaptive_local")

    structure = _connectivity_structure(connectivity)
    component_labels, original_components = ndimage.label(original, structure=structure)
    component_labels = component_labels.astype(np.int32, copy=False)
    component_sizes = _component_sizes(component_labels, original_components)
    original_voxels = int(original.sum())
    max_added_voxels = int(max(1, np.floor(float(max_added_fraction) * max(original_voxels, 1))))
    spacing_zyx = np.asarray([geometry.spacing_xyz[2], geometry.spacing_xyz[1], geometry.spacing_xyz[0]], dtype=float)
    radius_step_mm = float(max(np.min(spacing_zyx), 1e-6))
    if max_radius_mm is None or float(max_radius_mm) <= 0:
        max_radius_mm = max(float(radius_mm), float(max_gap_mm))
    max_radius_mm = float(max(float(radius_mm), float(max_radius_mm)))
    if min_accept_radius_mm is None or float(min_accept_radius_mm) <= 0:
        min_accept_radius_mm = float(radius_mm)
    min_accept_radius_mm = float(min(max(float(radius_mm), float(min_accept_radius_mm)), max_radius_mm))

    current = original.copy()
    bridge_mask = np.zeros_like(original, dtype=bool)
    metrics: dict[str, object] = {
        "repair_status": "",
        "original_components": int(original_components),
        "repaired_components": int(original_components),
        "accepted_paths": 0,
        "rejected_paths": 0,
        "added_voxels": 0,
        "max_added_voxels": int(max_added_voxels),
        "max_gap_mm": float(max_gap_mm),
        "mesh_support_mm": float(mesh_support_mm),
        "min_mesh_support_fraction": float(min_mesh_support_fraction),
        "radius_mm": float(radius_mm),
        "max_radius_mm": float(max_radius_mm),
        "min_accept_radius_mm": float(min_accept_radius_mm),
        "radius_mode": radius_mode,
        "local_radius_window_mm": float(local_radius_window_mm),
        "radius_percentile": float(radius_percentile),
        "radius_scale": float(radius_scale),
        "min_component_voxels": int(min_component_voxels),
        "max_added_fraction": float(max_added_fraction),
        "connectivity": int(connectivity),
        "paths": [],
    }
    if original_components <= 1:
        metrics["repair_status"] = "noop_already_connected"
        return MeshPathRepairResult(current, bridge_mask, component_labels, metrics)
    if original_voxels <= 0:
        metrics["repair_status"] = "noop_empty_mask"
        return MeshPathRepairResult(current, bridge_mask, component_labels, metrics)

    support_points = _mesh_support_points(vertices_physical_xyz, faces)
    if support_points.size == 0:
        metrics["repair_status"] = "noop_empty_mesh_support"
        return MeshPathRepairResult(current, bridge_mask, component_labels, metrics)
    support_tree = cKDTree(support_points)
    metrics["support_points"] = int(support_points.shape[0])

    component_order = [
        int(label)
        for label in np.argsort(component_sizes[1:])[::-1] + 1
        if component_sizes[int(label)] >= int(min_component_voxels)
    ]
    if not component_order:
        metrics["repair_status"] = "noop_no_large_components"
        return MeshPathRepairResult(current, bridge_mask, component_labels, metrics)

    main_labels = {int(component_order[0])}
    candidate_labels = [label for label in component_order[1:]]
    metrics["main_component_label"] = int(component_order[0])
    metrics["candidate_component_labels"] = [int(x) for x in candidate_labels]

    surface_voxels_by_label, surface_points_by_label = _component_surface_cache(
        component_labels,
        component_order,
        structure,
        geometry,
    )
    metrics["cached_component_surfaces"] = int(len(surface_voxels_by_label))

    accepted = 0
    rejected = 0
    while candidate_labels:
        _, cur_components = ndimage.label(current, structure=structure)
        if cur_components <= 1:
            break
        best, rejections = _find_best_endpoint_candidate(
            component_labels,
            component_sizes,
            main_labels,
            candidate_labels,
            geometry,
            structure,
            support_tree,
            surface_voxels_by_label,
            surface_points_by_label,
            max_gap_mm=float(max_gap_mm),
            mesh_support_mm=float(mesh_support_mm),
            min_mesh_support_fraction=float(min_mesh_support_fraction),
        )
        if best is None:
            rejected += len(rejections)
            metrics["paths"].extend(rejections)
            break

        candidate_label = int(best.component_label)
        path_points = np.stack([best.main_point_xyz, best.candidate_point_xyz], axis=0).astype(np.float32, copy=False)
        main_seed = np.isin(component_labels, list(main_labels)) | bridge_mask
        candidate_seed = component_labels == candidate_label

        main_endpoint_label = int(component_labels[tuple(best.main_voxel_zyx)])
        candidate_endpoint_label = int(component_labels[tuple(best.candidate_voxel_zyx)])
        main_local_radius = float("nan")
        candidate_local_radius = float("nan")
        adaptive_accept_radius = float(min_accept_radius_mm)
        if radius_mode == "adaptive_local":
            if main_endpoint_label > 0:
                main_local_radius = _local_component_radius_mm(
                    component_labels,
                    main_endpoint_label,
                    best.main_point_xyz,
                    geometry,
                    window_mm=float(local_radius_window_mm),
                    percentile=float(radius_percentile),
                )
            if candidate_endpoint_label > 0:
                candidate_local_radius = _local_component_radius_mm(
                    component_labels,
                    candidate_endpoint_label,
                    best.candidate_point_xyz,
                    geometry,
                    window_mm=float(local_radius_window_mm),
                    percentile=float(radius_percentile),
                )
            valid_radii = [value for value in (main_local_radius, candidate_local_radius) if np.isfinite(value) and value > 0]
            if valid_radii:
                adaptive_accept_radius = float(np.mean(valid_radii) * float(radius_scale))
            adaptive_accept_radius = float(
                min(max(float(radius_mm), adaptive_accept_radius, float(min_accept_radius_mm)), max_radius_mm)
            )

        radius_values = np.arange(float(radius_mm), max_radius_mm + 0.5 * radius_step_mm, radius_step_mm)
        extra_radius_values = [float(max_radius_mm)]
        if radius_mode == "adaptive_local":
            extra_radius_values.append(float(adaptive_accept_radius))
        radius_values = np.concatenate([radius_values, np.asarray(extra_radius_values, dtype=float)])
        radius_values = sorted(float(x) for x in np.unique(np.round(radius_values, decimals=6)) if x <= max_radius_mm)

        tube = np.zeros_like(current, dtype=bool)
        added = np.zeros_like(current, dtype=bool)
        added_voxels = 0
        selected_radius = float(radius_mm)
        merges = False
        too_many_added = False
        for trial_radius in radius_values:
            trial_tube = _rasterize_path_tube(path_points, geometry, float(trial_radius))
            trial_added = trial_tube & ~current
            trial_added_voxels = int(trial_added.sum())
            trial_merges = bool(np.any(trial_tube & main_seed) and np.any(trial_tube & candidate_seed))
            if trial_merges:
                if float(trial_radius) < float(adaptive_accept_radius):
                    continue
                if trial_added_voxels > max_added_voxels:
                    too_many_added = True
                    tube = trial_tube
                    added = trial_added
                    added_voxels = trial_added_voxels
                    selected_radius = float(trial_radius)
                    break
                tube = trial_tube
                added = trial_added
                added_voxels = trial_added_voxels
                selected_radius = float(trial_radius)
                merges = True
                break

        path_metrics = {
            "component_label": int(candidate_label),
            "component_voxels": int(component_sizes[candidate_label]),
            "endpoint_distance_mm": float(best.endpoint_distance_mm),
            "mesh_support_fraction": float(best.mesh_support_fraction),
            "mesh_support_min_mm": float(best.mesh_support_min_mm),
            "mesh_support_median_mm": float(best.mesh_support_median_mm),
            "mesh_support_max_mm": float(best.mesh_support_max_mm),
            "added_voxels": int(added_voxels),
            "selected_radius_mm": float(selected_radius),
            "radius_mode": radius_mode,
            "adaptive_accept_radius_mm": float(adaptive_accept_radius),
        }
        if radius_mode == "adaptive_local":
            path_metrics["main_local_radius_mm"] = float(main_local_radius)
            path_metrics["candidate_local_radius_mm"] = float(candidate_local_radius)
        if not merges:
            rejected += 1
            path_metrics["status"] = "rejected_too_many_added_voxels" if too_many_added else "rejected_did_not_merge"
            if too_many_added:
                path_metrics["max_added_voxels"] = int(max_added_voxels)
            metrics["paths"].append(path_metrics)
            candidate_labels = [x for x in candidate_labels if x != candidate_label]
            continue

        current |= tube
        bridge_mask |= added
        main_labels.add(candidate_label)
        candidate_labels = [x for x in candidate_labels if x != candidate_label]
        accepted += 1
        path_metrics["status"] = "accepted"
        metrics["paths"].append(path_metrics)

    _, repaired_components = ndimage.label(current, structure=structure)
    metrics["accepted_paths"] = int(accepted)
    metrics["rejected_paths"] = int(rejected)
    metrics["repaired_components"] = int(repaired_components)
    metrics["added_voxels"] = int((current & ~original).sum())
    if accepted == 0:
        metrics["repair_status"] = "noop_no_valid_bridge"
    elif repaired_components <= 1:
        metrics["repair_status"] = "connected"
    else:
        metrics["repair_status"] = "partial_connected"
    return MeshPathRepairResult(current, bridge_mask, component_labels, metrics)
