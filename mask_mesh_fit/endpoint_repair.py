from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .bridge_tube import (
    RadiusPolicy,
    TubeTrial,
    grow_tube_until_merge,
    local_component_radius_mm,
)
from .io_utils import ImageGeometry
from .mesh_path_repair import (
    MeshPathRepairResult,
    _component_sizes,
    _connectivity_structure,
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


@dataclass(frozen=True)
class _EndpointOptions:
    """Validated settings for endpoint bridge repair.

    How wide a bridge may be is delegated to :class:`RadiusPolicy`, shared with the
    mesh-graph strategy. The fields here govern which endpoint pairs are even
    considered and how much mesh evidence a straight bridge needs.
    """

    max_gap_mm: float
    mesh_support_mm: float
    min_mesh_support_fraction: float
    radius: RadiusPolicy
    min_component_voxels: int
    max_added_fraction: float
    connectivity: int


def _resolve_endpoint_options(
    geometry: ImageGeometry,
    *,
    max_gap_mm: float,
    mesh_support_mm: float,
    min_mesh_support_fraction: float,
    radius_mm: float,
    max_radius_mm: float | None,
    min_accept_radius_mm: float | None,
    radius_mode: str,
    local_radius_window_mm: float,
    radius_percentile: float,
    radius_scale: float,
    min_component_voxels: int,
    max_added_fraction: float,
    connectivity: int,
) -> _EndpointOptions:
    """Check the option values and settle the radius bounds."""
    # With no explicit cap, a bridge may grow as wide as the largest gap this run is
    # willing to cross.
    auto_max_radius_mm = max(float(radius_mm), float(max_gap_mm))
    policy = RadiusPolicy.resolve(
        geometry,
        radius_mm=radius_mm,
        max_radius_mm=max_radius_mm,
        auto_max_radius_mm=auto_max_radius_mm,
        min_accept_radius_mm=min_accept_radius_mm,
        mode=radius_mode,
        local_window_mm=local_radius_window_mm,
        percentile=radius_percentile,
        scale=radius_scale,
    )
    return _EndpointOptions(
        max_gap_mm=float(max_gap_mm),
        mesh_support_mm=float(mesh_support_mm),
        min_mesh_support_fraction=float(min_mesh_support_fraction),
        radius=policy,
        min_component_voxels=int(min_component_voxels),
        max_added_fraction=float(max_added_fraction),
        connectivity=int(connectivity),
    )


def _initial_endpoint_metrics(
    options: _EndpointOptions,
    original_components: int,
    max_added_voxels: int,
) -> dict[str, object]:
    """The metrics record, pre-filled with the settings the run used."""
    return {
        "repair_status": "",
        "original_components": int(original_components),
        "repaired_components": int(original_components),
        "accepted_paths": 0,
        "rejected_paths": 0,
        "added_voxels": 0,
        "max_added_voxels": int(max_added_voxels),
        "max_gap_mm": options.max_gap_mm,
        "mesh_support_mm": options.mesh_support_mm,
        "min_mesh_support_fraction": options.min_mesh_support_fraction,
        "radius_mm": options.radius.radius_mm,
        "max_radius_mm": options.radius.max_radius_mm,
        "min_accept_radius_mm": options.radius.min_accept_radius_mm,
        "radius_mode": options.radius.mode,
        "local_radius_window_mm": options.radius.local_window_mm,
        "radius_percentile": options.radius.percentile,
        "radius_scale": options.radius.scale,
        "min_component_voxels": options.min_component_voxels,
        "max_added_fraction": options.max_added_fraction,
        "connectivity": options.connectivity,
        "paths": [],
    }


def _endpoint_local_radii_mm(
    best: _EndpointCandidate,
    component_labels: np.ndarray,
    geometry: ImageGeometry,
    options: _EndpointOptions,
) -> tuple[float, float]:
    """Local vessel calibre of the mask at each end of the proposed bridge."""
    main_label = int(component_labels[tuple(best.main_voxel_zyx)])
    candidate_label = int(component_labels[tuple(best.candidate_voxel_zyx)])
    main_radius = float("nan")
    candidate_radius = float("nan")
    if main_label > 0:
        main_radius = local_component_radius_mm(
            component_labels,
            main_label,
            best.main_point_xyz,
            geometry,
            window_mm=options.radius.local_window_mm,
            percentile=options.radius.percentile,
        )
    if candidate_label > 0:
        candidate_radius = local_component_radius_mm(
            component_labels,
            candidate_label,
            best.candidate_point_xyz,
            geometry,
            window_mm=options.radius.local_window_mm,
            percentile=options.radius.percentile,
        )
    return main_radius, candidate_radius


def _endpoint_bridge_metrics(
    best: _EndpointCandidate,
    trial: TubeTrial,
    component_sizes: np.ndarray,
    main_local_radius: float,
    candidate_local_radius: float,
    adaptive_accept_radius: float,
    options: _EndpointOptions,
) -> dict[str, object]:
    """One bridge's record: the gap it crossed, its mesh support, and what it cost."""
    label = int(best.component_label)
    metrics: dict[str, object] = {
        "component_label": label,
        "component_voxels": int(component_sizes[label]),
        "endpoint_distance_mm": float(best.endpoint_distance_mm),
        "mesh_support_fraction": float(best.mesh_support_fraction),
        "mesh_support_min_mm": float(best.mesh_support_min_mm),
        "mesh_support_median_mm": float(best.mesh_support_median_mm),
        "mesh_support_max_mm": float(best.mesh_support_max_mm),
        "added_voxels": int(trial.added_voxels),
        "selected_radius_mm": float(trial.selected_radius_mm),
        "radius_mode": options.radius.mode,
        "adaptive_accept_radius_mm": float(adaptive_accept_radius),
    }
    if options.radius.adaptive:
        metrics["main_local_radius_mm"] = float(main_local_radius)
        metrics["candidate_local_radius_mm"] = float(candidate_local_radius)
    return metrics


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
    """Reconnect a fragmented mask with short straight bridges the mesh vouches for.

    Unlike the mesh-graph strategy this does not follow mesh edges. It takes the
    closest endpoint-to-endpoint gap between components and checks that the straight
    line across it runs close to the fitted mesh, so the mesh validates a bridge
    rather than proposing one. That suits dense branching trees such as the pulmonary
    arteries, where graph paths across a 20k-vertex mesh are long and unreliable.

    A proposal is rasterised as a thin tube and kept only if it truly merges the two
    components within the foreground-growth budget.
    """
    original = original_mask_zyx.astype(bool, copy=False)
    options = _resolve_endpoint_options(
        geometry,
        max_gap_mm=max_gap_mm,
        mesh_support_mm=mesh_support_mm,
        min_mesh_support_fraction=min_mesh_support_fraction,
        radius_mm=radius_mm,
        max_radius_mm=max_radius_mm,
        min_accept_radius_mm=min_accept_radius_mm,
        radius_mode=radius_mode,
        local_radius_window_mm=local_radius_window_mm,
        radius_percentile=radius_percentile,
        radius_scale=radius_scale,
        min_component_voxels=min_component_voxels,
        max_added_fraction=max_added_fraction,
        connectivity=connectivity,
    )

    structure = _connectivity_structure(options.connectivity)
    component_labels, original_components = ndimage.label(original, structure=structure)
    component_labels = component_labels.astype(np.int32, copy=False)
    component_sizes = _component_sizes(component_labels, original_components)
    original_voxels = int(original.sum())
    max_added_voxels = int(max(1, np.floor(options.max_added_fraction * max(original_voxels, 1))))

    current = original.copy()
    bridge_mask = np.zeros_like(original, dtype=bool)
    metrics = _initial_endpoint_metrics(options, original_components, max_added_voxels)

    def finished(status: str) -> MeshPathRepairResult:
        metrics["repair_status"] = status
        return MeshPathRepairResult(current, bridge_mask, component_labels, metrics)

    if original_components <= 1:
        return finished("noop_already_connected")
    if original_voxels <= 0:
        return finished("noop_empty_mask")

    support_points = _mesh_support_points(vertices_physical_xyz, faces)
    if support_points.size == 0:
        return finished("noop_empty_mesh_support")
    support_tree = cKDTree(support_points)
    metrics["support_points"] = int(support_points.shape[0])

    component_order = [
        int(label)
        for label in np.argsort(component_sizes[1:])[::-1] + 1
        if component_sizes[int(label)] >= options.min_component_voxels
    ]
    if not component_order:
        return finished("noop_no_large_components")

    main_labels = {int(component_order[0])}
    candidate_labels = list(component_order[1:])
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
            max_gap_mm=options.max_gap_mm,
            mesh_support_mm=options.mesh_support_mm,
            min_mesh_support_fraction=options.min_mesh_support_fraction,
        )
        if best is None:
            rejected += len(rejections)
            metrics["paths"].extend(rejections)
            break

        label = int(best.component_label)
        path_points = np.stack(
            [best.main_point_xyz, best.candidate_point_xyz], axis=0
        ).astype(np.float32, copy=False)
        # Bridges accepted earlier count as part of the main component, so a later
        # bridge may legitimately land on one of them.
        main_seed = np.isin(component_labels, list(main_labels)) | bridge_mask
        candidate_seed = component_labels == label

        main_local_radius, candidate_local_radius = (float("nan"), float("nan"))
        if options.radius.adaptive:
            main_local_radius, candidate_local_radius = _endpoint_local_radii_mm(
                best, component_labels, geometry, options
            )
        adaptive_accept_radius = options.radius.accept_radius_mm(
            main_local_radius, candidate_local_radius
        )

        trial = grow_tube_until_merge(
            path_points,
            current,
            main_seed,
            candidate_seed,
            geometry,
            options.radius,
            adaptive_accept_radius,
            max_added_voxels,
        )

        path_metrics = _endpoint_bridge_metrics(
            best,
            trial,
            component_sizes,
            main_local_radius,
            candidate_local_radius,
            adaptive_accept_radius,
            options,
        )
        candidate_labels = [x for x in candidate_labels if x != label]

        if not trial.merges:
            rejected += 1
            path_metrics["status"] = (
                "rejected_too_many_added_voxels" if trial.too_many_added else "rejected_did_not_merge"
            )
            if trial.too_many_added:
                path_metrics["max_added_voxels"] = int(max_added_voxels)
            metrics["paths"].append(path_metrics)
            continue

        current |= trial.tube
        bridge_mask |= trial.added
        main_labels.add(label)
        accepted += 1
        path_metrics["status"] = "accepted"
        metrics["paths"].append(path_metrics)

    _, repaired_components = ndimage.label(current, structure=structure)
    metrics["accepted_paths"] = int(accepted)
    metrics["rejected_paths"] = int(rejected)
    metrics["repaired_components"] = int(repaired_components)
    metrics["added_voxels"] = int((current & ~original).sum())
    if accepted == 0:
        return finished("noop_no_valid_bridge")
    if repaired_components <= 1:
        return finished("connected")
    return finished("partial_connected")
