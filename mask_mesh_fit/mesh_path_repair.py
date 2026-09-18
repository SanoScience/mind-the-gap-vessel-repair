from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .io_utils import ImageGeometry
from .voxelize import physical_ball_structure


@dataclass(frozen=True)
class MeshPathRepairResult:
    repaired: np.ndarray
    bridge_mask: np.ndarray
    component_labels: np.ndarray
    metrics: dict[str, object]


def edge_index_from_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
            faces[:, [1, 0]],
            faces[:, [2, 1]],
            faces[:, [0, 2]],
        ],
        axis=0,
    )
    return np.unique(edges, axis=0).T.astype(np.int64, copy=False)


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if int(connectivity) == 6:
        return ndimage.generate_binary_structure(3, 1)
    if int(connectivity) == 18:
        return ndimage.generate_binary_structure(3, 2)
    if int(connectivity) == 26:
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError("connectivity must be one of 6, 18, or 26")


def _component_sizes(labels: np.ndarray, num_labels: int) -> np.ndarray:
    if num_labels <= 0:
        return np.zeros((1,), dtype=np.int64)
    return np.bincount(labels.ravel(), minlength=num_labels + 1).astype(np.int64, copy=False)


def _mesh_adjacency(vertices: np.ndarray, edge_index: np.ndarray) -> list[list[tuple[int, float]]]:
    num_vertices = int(vertices.shape[0])
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_vertices)]
    if edge_index.size == 0:
        return adjacency
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.shape[0] != 2:
        edges = edges.T
    undirected = np.unique(np.sort(edges.T, axis=1), axis=0)
    for a, b in undirected:
        if a < 0 or b < 0 or a >= num_vertices or b >= num_vertices or a == b:
            continue
        weight = float(np.linalg.norm(vertices[int(a)] - vertices[int(b)]))
        if not np.isfinite(weight) or weight <= 0:
            continue
        adjacency[int(a)].append((int(b), weight))
        adjacency[int(b)].append((int(a), weight))
    return adjacency


def _shortest_mesh_path(
    adjacency: list[list[tuple[int, float]]],
    sources: np.ndarray,
    targets: np.ndarray,
) -> tuple[list[int], float] | None:
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if sources.size == 0 or targets.size == 0:
        return None
    target_set = {int(x) for x in targets}
    num_vertices = len(adjacency)
    dist = np.full((num_vertices,), np.inf, dtype=np.float64)
    prev = np.full((num_vertices,), -1, dtype=np.int64)
    heap: list[tuple[float, int]] = []
    for source in np.unique(sources):
        source_i = int(source)
        if source_i < 0 or source_i >= num_vertices:
            continue
        dist[source_i] = 0.0
        heapq.heappush(heap, (0.0, source_i))

    found = -1
    while heap:
        cur_dist, cur = heapq.heappop(heap)
        if cur_dist != dist[cur]:
            continue
        if cur in target_set:
            found = cur
            break
        for nxt, weight in adjacency[cur]:
            cand = cur_dist + weight
            if cand < dist[nxt]:
                dist[nxt] = cand
                prev[nxt] = cur
                heapq.heappush(heap, (cand, nxt))

    if found < 0 or not np.isfinite(dist[found]):
        return None

    path = [found]
    while prev[path[-1]] >= 0:
        path.append(int(prev[path[-1]]))
    path.reverse()
    return path, float(dist[found])


def _shortest_mesh_path_min_edges(
    adjacency: list[list[tuple[int, float]]],
    sources: np.ndarray,
    targets: np.ndarray,
    min_edges: int,
) -> tuple[list[int], float] | None:
    min_edges = max(int(min_edges), 0)
    first = _shortest_mesh_path(adjacency, sources, targets)
    if first is None or len(first[0]) - 1 >= min_edges:
        return first

    # If the ordinary shortest path is a direct source-target shortcut, remove
    # those direct cross-component graph edges and try once more. This avoids
    # accepting a one-edge mesh spike while still keeping the selector cheap.
    source_set = {int(x) for x in np.asarray(sources, dtype=np.int64)}
    target_set = {int(x) for x in np.asarray(targets, dtype=np.int64)}
    pruned: list[list[tuple[int, float]]] = []
    removed_any = False
    for idx, neighbors in enumerate(adjacency):
        filtered_neighbors = []
        for nxt, weight in neighbors:
            direct_cross_edge = (idx in source_set and nxt in target_set) or (idx in target_set and nxt in source_set)
            if direct_cross_edge:
                removed_any = True
                continue
            filtered_neighbors.append((nxt, weight))
        pruned.append(filtered_neighbors)
    if not removed_any:
        return None

    second = _shortest_mesh_path(pruned, sources, targets)
    if second is None or len(second[0]) - 1 < min_edges:
        return None
    return second


def _filter_adjacency_by_edge_length(
    adjacency: list[list[tuple[int, float]]],
    max_edge_mm: float,
) -> list[list[tuple[int, float]]]:
    if float(max_edge_mm) <= 0:
        return adjacency
    max_edge_mm = float(max_edge_mm)
    return [[(nxt, weight) for nxt, weight in neighbors if weight <= max_edge_mm] for neighbors in adjacency]


def _path_edge_stats(vertices: np.ndarray, path: list[int]) -> dict[str, float | int]:
    path_edges = max(len(path) - 1, 0)
    if path_edges <= 0:
        return {
            "path_edges": 0,
            "path_max_edge_mm": 0.0,
            "path_mean_edge_mm": 0.0,
        }
    points = vertices[np.asarray(path, dtype=np.int64)]
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return {
        "path_edges": int(path_edges),
        "path_max_edge_mm": float(np.max(lengths)),
        "path_mean_edge_mm": float(np.mean(lengths)),
    }


def _mesh_vertex_component_anchors(
    vertices_physical_xyz: np.ndarray,
    labels: np.ndarray,
    geometry: ImageGeometry,
    anchor_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_zyx = np.argwhere(labels > 0)
    if mask_zyx.size == 0:
        return (
            np.zeros((vertices_physical_xyz.shape[0],), dtype=np.int32),
            np.full((vertices_physical_xyz.shape[0],), np.inf, dtype=np.float32),
            np.full((vertices_physical_xyz.shape[0], 3), np.nan, dtype=np.float32),
        )

    vertex_labels = np.zeros((vertices_physical_xyz.shape[0],), dtype=np.int32)
    vertex_dist = np.full((vertices_physical_xyz.shape[0],), np.inf, dtype=np.float32)
    vertex_nearest_physical_xyz = np.full((vertices_physical_xyz.shape[0], 3), np.nan, dtype=np.float32)

    mask_index_xyz = mask_zyx[:, [2, 1, 0]].astype(np.float64, copy=False)
    mask_physical_xyz = geometry.continuous_index_xyz_to_physical(mask_index_xyz)
    tree = cKDTree(mask_physical_xyz)
    query_points = np.asarray(vertices_physical_xyz, dtype=np.float64)
    try:
        distances, nearest_idx = tree.query(
            query_points,
            k=1,
            distance_upper_bound=float(anchor_mm),
            workers=-1,
        )
    except TypeError:  # pragma: no cover - compatibility with older scipy
        distances, nearest_idx = tree.query(
            query_points,
            k=1,
            distance_upper_bound=float(anchor_mm),
        )
    valid = np.isfinite(distances) & (nearest_idx < mask_zyx.shape[0])
    if np.any(valid):
        nearest_mask_zyx = mask_zyx[nearest_idx[valid]]
        vertex_labels[valid] = labels[
            nearest_mask_zyx[:, 0],
            nearest_mask_zyx[:, 1],
            nearest_mask_zyx[:, 2],
        ].astype(np.int32, copy=False)
        vertex_dist[valid] = distances[valid].astype(np.float32, copy=False)
        vertex_nearest_physical_xyz[valid] = mask_physical_xyz[nearest_idx[valid]].astype(np.float32, copy=False)
    return vertex_labels, vertex_dist, vertex_nearest_physical_xyz


def _rasterize_path_tube(
    path_vertices_physical_xyz: np.ndarray,
    geometry: ImageGeometry,
    radius_mm: float,
) -> np.ndarray:
    output = np.zeros(geometry.shape_zyx, dtype=bool)
    if path_vertices_physical_xyz.shape[0] == 0:
        return output
    index_xyz = geometry.physical_to_continuous_index_xyz(path_vertices_physical_xyz)
    index_zyx = index_xyz[:, [2, 1, 0]]
    shape = np.asarray(geometry.shape_zyx, dtype=np.int64)
    spacing_zyx = np.asarray([geometry.spacing_xyz[2], geometry.spacing_xyz[1], geometry.spacing_xyz[0]], dtype=float)
    min_spacing = float(max(np.min(spacing_zyx), 1e-6))
    voxel_chunks: list[np.ndarray] = []

    for start, end in zip(index_zyx[:-1], index_zyx[1:], strict=False):
        physical_len = float(np.linalg.norm((end - start) * spacing_zyx))
        steps = max(int(np.ceil(physical_len / (0.5 * min_spacing))), 1)
        ts = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)
        samples = start[None, :] * (1.0 - ts[:, None]) + end[None, :] * ts[:, None]
        voxels = np.rint(samples).astype(np.int64)
        in_bounds = np.all((voxels >= 0) & (voxels < shape[None, :]), axis=1)
        voxels = voxels[in_bounds]
        if voxels.size > 0:
            voxel_chunks.append(voxels)

    if path_vertices_physical_xyz.shape[0] == 1:
        voxel = np.rint(index_zyx[0]).astype(np.int64)
        if np.all((voxel >= 0) & (voxel < shape)):
            voxel_chunks.append(voxel[None, :])

    if not voxel_chunks:
        return output

    voxels = np.unique(np.concatenate(voxel_chunks, axis=0), axis=0)
    if float(radius_mm) <= 0:
        output[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
        return output

    radii_xyz = np.maximum(np.ceil(float(radius_mm) / np.asarray(geometry.spacing_xyz, dtype=float)).astype(int), 1)
    margin_zyx = np.asarray([radii_xyz[2], radii_xyz[1], radii_xyz[0]], dtype=np.int64) + 1
    lo = np.maximum(voxels.min(axis=0) - margin_zyx, 0)
    hi = np.minimum(voxels.max(axis=0) + margin_zyx + 1, shape)
    crop_slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    crop_shape = tuple(int(b - a) for a, b in zip(lo, hi, strict=True))
    centerline_crop = np.zeros(crop_shape, dtype=bool)
    crop_voxels = voxels - lo[None, :]
    centerline_crop[crop_voxels[:, 0], crop_voxels[:, 1], crop_voxels[:, 2]] = True
    structure = physical_ball_structure(geometry.spacing_xyz, float(radius_mm))
    output[crop_slices] = ndimage.binary_dilation(centerline_crop, structure=structure)
    return output


def _local_component_radius_mm(
    labels: np.ndarray,
    component_label: int,
    anchor_physical_xyz: np.ndarray,
    geometry: ImageGeometry,
    window_mm: float,
    percentile: float,
) -> float:
    if int(component_label) <= 0 or not np.all(np.isfinite(anchor_physical_xyz)):
        return float("nan")
    anchor_index_xyz = geometry.physical_to_continuous_index_xyz(anchor_physical_xyz[None, :])[0]
    anchor_zyx = np.rint(anchor_index_xyz[[2, 1, 0]]).astype(np.int64)
    shape = np.asarray(labels.shape, dtype=np.int64)
    if np.any(anchor_zyx < 0) or np.any(anchor_zyx >= shape):
        return float("nan")

    spacing_zyx = np.asarray([geometry.spacing_xyz[2], geometry.spacing_xyz[1], geometry.spacing_xyz[0]], dtype=float)
    margin_zyx = np.maximum(np.ceil(float(window_mm) / spacing_zyx).astype(np.int64), 1)
    lo = np.maximum(anchor_zyx - margin_zyx, 0)
    hi = np.minimum(anchor_zyx + margin_zyx + 1, shape)
    crop_slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    component_crop = labels[crop_slices] == int(component_label)
    if not np.any(component_crop):
        return float("nan")

    dist = ndimage.distance_transform_edt(component_crop, sampling=tuple(float(x) for x in spacing_zyx))
    values = dist[component_crop]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, float(percentile)))


def repair_mask_with_mesh_paths(
    original_mask_zyx: np.ndarray,
    vertices_physical_xyz: np.ndarray,
    faces: np.ndarray,
    edge_index: np.ndarray | None,
    geometry: ImageGeometry,
    *,
    anchor_mm: float = 2.0,
    radius_mm: float = 1.0,
    max_radius_mm: float | None = None,
    min_accept_radius_mm: float | None = None,
    radius_mode: str = "fixed",
    local_radius_window_mm: float = 6.0,
    radius_percentile: float = 80.0,
    radius_scale: float = 1.0,
    min_component_voxels: int = 20,
    max_added_fraction: float = 0.03,
    connectivity: int = 26,
    path_selection: str = "shortest",
    min_path_edges: int = 2,
    max_mesh_edge_mm: float = 0.0,
    edge_filter_fallback: str = "old_shortest",
) -> MeshPathRepairResult:
    original = original_mask_zyx.astype(bool, copy=False)
    radius_mode = str(radius_mode).strip().lower()
    if radius_mode not in {"fixed", "adaptive_local"}:
        raise ValueError("radius_mode must be one of: fixed, adaptive_local")
    path_selection = str(path_selection).strip().lower()
    if path_selection not in {"shortest", "edge_filtered_shortest"}:
        raise ValueError("path_selection must be one of: shortest, edge_filtered_shortest")
    edge_filter_fallback = str(edge_filter_fallback).strip().lower()
    if edge_filter_fallback not in {"old_shortest", "skip"}:
        raise ValueError("edge_filter_fallback must be one of: old_shortest, skip")
    min_path_edges = max(int(min_path_edges), 0)
    structure = _connectivity_structure(connectivity)
    component_labels, original_components = ndimage.label(original, structure=structure)
    component_labels = component_labels.astype(np.int32, copy=False)
    component_sizes = _component_sizes(component_labels, original_components)
    original_voxels = int(original.sum())
    max_added_voxels = int(max(1, np.floor(float(max_added_fraction) * max(original_voxels, 1))))
    spacing_zyx = np.asarray([geometry.spacing_xyz[2], geometry.spacing_xyz[1], geometry.spacing_xyz[0]], dtype=float)
    radius_step_mm = float(max(np.min(spacing_zyx), 1e-6))
    if max_radius_mm is None or float(max_radius_mm) <= 0:
        max_radius_mm = max(float(radius_mm), float(anchor_mm) + radius_step_mm)
    max_radius_mm = float(max(float(radius_mm), float(max_radius_mm)))
    if min_accept_radius_mm is None or float(min_accept_radius_mm) <= 0:
        min_accept_radius_mm = float(radius_mm)
    min_accept_radius_mm = float(min(max(float(radius_mm), float(min_accept_radius_mm)), max_radius_mm))

    bridge_mask = np.zeros_like(original, dtype=bool)
    current = original.copy()
    metrics: dict[str, object] = {
        "repair_status": "",
        "original_components": int(original_components),
        "repaired_components": int(original_components),
        "accepted_paths": 0,
        "rejected_paths": 0,
        "added_voxels": 0,
        "max_added_voxels": int(max_added_voxels),
        "anchor_mm": float(anchor_mm),
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
    if path_selection != "shortest":
        metrics.update(
            {
                "path_selection": path_selection,
                "min_path_edges": int(min_path_edges),
                "max_mesh_edge_mm": float(max_mesh_edge_mm),
                "edge_filter_fallback": edge_filter_fallback,
            }
        )

    if original_components <= 1:
        metrics["repair_status"] = "noop_already_connected"
        return MeshPathRepairResult(
            repaired=current,
            bridge_mask=bridge_mask,
            component_labels=component_labels,
            metrics=metrics,
        )
    if original_voxels <= 0:
        metrics["repair_status"] = "noop_empty_mask"
        return MeshPathRepairResult(
            repaired=current,
            bridge_mask=bridge_mask,
            component_labels=component_labels,
            metrics=metrics,
        )

    if edge_index is None or np.asarray(edge_index).size == 0:
        edge_index = edge_index_from_faces(faces)
    adjacency = _mesh_adjacency(vertices_physical_xyz, np.asarray(edge_index, dtype=np.int64))
    if not any(adjacency):
        metrics["repair_status"] = "noop_empty_mesh_graph"
        return MeshPathRepairResult(
            repaired=current,
            bridge_mask=bridge_mask,
            component_labels=component_labels,
            metrics=metrics,
        )
    filtered_adjacency = _filter_adjacency_by_edge_length(adjacency, max_mesh_edge_mm)

    vertex_component, vertex_distance, vertex_nearest_physical_xyz = _mesh_vertex_component_anchors(
        vertices_physical_xyz,
        component_labels,
        geometry,
        anchor_mm=float(anchor_mm),
    )
    anchor_counts = {
        int(label): int(np.count_nonzero(vertex_component == label))
        for label in range(1, original_components + 1)
    }
    metrics["anchor_counts"] = anchor_counts

    component_order = [
        int(label)
        for label in np.argsort(component_sizes[1:])[::-1] + 1
        if component_sizes[int(label)] >= int(min_component_voxels)
    ]
    if not component_order:
        metrics["repair_status"] = "noop_no_large_components"
        return MeshPathRepairResult(
            repaired=current,
            bridge_mask=bridge_mask,
            component_labels=component_labels,
            metrics=metrics,
        )

    main_labels = {int(component_order[0])}
    candidate_labels = [label for label in component_order[1:] if anchor_counts.get(int(label), 0) > 0]
    metrics["main_component_label"] = int(component_order[0])
    metrics["candidate_component_labels"] = [int(x) for x in candidate_labels]

    accepted = 0
    rejected = 0
    while candidate_labels:
        _, cur_components = ndimage.label(current, structure=structure)
        if cur_components <= 1:
            break

        best: dict[str, object] | None = None
        fallback_best: dict[str, object] | None = None
        source_vertices = np.flatnonzero(np.isin(vertex_component, list(main_labels)))
        if source_vertices.size == 0:
            break
        for candidate_label in candidate_labels:
            target_vertices = np.flatnonzero(vertex_component == int(candidate_label))
            path_result = None
            if path_selection == "edge_filtered_shortest":
                path_result = _shortest_mesh_path_min_edges(
                    filtered_adjacency,
                    source_vertices,
                    target_vertices,
                    min_edges=min_path_edges,
                )
            else:
                path_result = _shortest_mesh_path(adjacency, source_vertices, target_vertices)

            if path_result is None:
                if path_selection == "edge_filtered_shortest" and edge_filter_fallback == "old_shortest":
                    fallback_result = _shortest_mesh_path(adjacency, source_vertices, target_vertices)
                    if fallback_result is not None:
                        fallback_path, fallback_length = fallback_result
                        if fallback_best is None or fallback_length < float(fallback_best["path_length_mm"]):
                            fallback_best = {
                                "component_label": int(candidate_label),
                                "path": fallback_path,
                                "path_length_mm": float(fallback_length),
                                "filtered_path_used": False,
                                "fallback_used": True,
                            }
                        continue
                rejected += 1
                path_metrics = {
                    "component_label": int(candidate_label),
                    "status": "rejected_no_mesh_path",
                    "component_voxels": int(component_sizes[int(candidate_label)]),
                    "anchor_count": int(anchor_counts.get(int(candidate_label), 0)),
                }
                if path_selection == "edge_filtered_shortest":
                    path_metrics.update(
                        {
                            "path_selection": path_selection,
                            "filtered_path_used": False,
                            "fallback_used": False,
                        }
                    )
                metrics["paths"].append(path_metrics)
                continue
            path, path_length = path_result
            if best is None or path_length < float(best["path_length_mm"]):
                best = {
                    "component_label": int(candidate_label),
                    "path": path,
                    "path_length_mm": float(path_length),
                    "filtered_path_used": path_selection == "edge_filtered_shortest",
                    "fallback_used": False,
                }

        if best is None and fallback_best is not None:
            best = fallback_best
        if best is None:
            break

        candidate_label = int(best["component_label"])
        path = list(best["path"])
        path_points = vertices_physical_xyz[np.asarray(path, dtype=np.int64)]
        if path:
            start_anchor = vertex_nearest_physical_xyz[int(path[0])]
            end_anchor = vertex_nearest_physical_xyz[int(path[-1])]
            extra_points = []
            if np.all(np.isfinite(start_anchor)):
                extra_points.append(start_anchor[None, :])
            extra_points.append(path_points)
            if np.all(np.isfinite(end_anchor)):
                extra_points.append(end_anchor[None, :])
            path_points = np.concatenate(extra_points, axis=0)

        main_seed = np.isin(component_labels, list(main_labels))
        candidate_seed = component_labels == candidate_label
        main_endpoint_label = int(vertex_component[int(path[0])]) if path else 0
        candidate_endpoint_label = int(vertex_component[int(path[-1])]) if path else 0
        main_local_radius = float("nan")
        candidate_local_radius = float("nan")
        adaptive_accept_radius = float(min_accept_radius_mm)
        if radius_mode == "adaptive_local":
            if path and main_endpoint_label > 0:
                main_local_radius = _local_component_radius_mm(
                    component_labels,
                    main_endpoint_label,
                    vertex_nearest_physical_xyz[int(path[0])],
                    geometry,
                    window_mm=float(local_radius_window_mm),
                    percentile=float(radius_percentile),
                )
            if path and candidate_endpoint_label > 0:
                candidate_local_radius = _local_component_radius_mm(
                    component_labels,
                    candidate_endpoint_label,
                    vertex_nearest_physical_xyz[int(path[-1])],
                    geometry,
                    window_mm=float(local_radius_window_mm),
                    percentile=float(radius_percentile),
                )
            valid_local_radii = [
                value for value in (main_local_radius, candidate_local_radius) if np.isfinite(value) and value > 0
            ]
            if valid_local_radii:
                adaptive_accept_radius = float(np.mean(valid_local_radii) * float(radius_scale))
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
        merges = False
        selected_radius = float(radius_mm)
        too_many_added = False
        for trial_radius in radius_values:
            trial_tube = _rasterize_path_tube(
                path_points,
                geometry,
                float(trial_radius),
            )
            trial_added = trial_tube & ~current
            trial_added_voxels = int(trial_added.sum())
            # The bridge tube is rasterized from a continuous mesh path and is
            # connected by construction. It merges components as soon as it
            # touches both seeds, so avoid a full-volume connected-component
            # relabel for every candidate radius.
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
                merges = True
                selected_radius = float(trial_radius)
                break

        path_metrics = {
            "component_label": int(candidate_label),
            "component_voxels": int(component_sizes[candidate_label]),
            "anchor_count": int(anchor_counts.get(candidate_label, 0)),
            "path_vertices": int(len(path)),
            "path_length_mm": float(best["path_length_mm"]),
            "added_voxels": int(added_voxels),
            "selected_radius_mm": float(selected_radius),
            "radius_mode": radius_mode,
            "adaptive_accept_radius_mm": float(adaptive_accept_radius),
        }
        if path_selection == "edge_filtered_shortest":
            path_metrics.update(
                {
                    "path_selection": path_selection,
                    "filtered_path_used": bool(best.get("filtered_path_used", False)),
                    "fallback_used": bool(best.get("fallback_used", False)),
                    **_path_edge_stats(vertices_physical_xyz, path),
                }
            )
        if radius_mode == "adaptive_local":
            path_metrics["main_local_radius_mm"] = float(main_local_radius)
            path_metrics["candidate_local_radius_mm"] = float(candidate_local_radius)
            path_metrics["local_radius_window_mm"] = float(local_radius_window_mm)
            path_metrics["radius_percentile"] = float(radius_percentile)
            path_metrics["radius_scale"] = float(radius_scale)
        if not merges:
            rejected += 1
            path_metrics["status"] = (
                "rejected_too_many_added_voxels" if too_many_added else "rejected_did_not_merge"
            )
            if too_many_added:
                path_metrics["max_added_voxels"] = int(max_added_voxels)
            metrics["paths"].append(path_metrics)
            candidate_labels = [x for x in candidate_labels if x != candidate_label]
            continue
        if added_voxels > max_added_voxels:
            rejected += 1
            path_metrics["status"] = "rejected_too_many_added_voxels"
            path_metrics["max_added_voxels"] = int(max_added_voxels)
            metrics["paths"].append(path_metrics)
            candidate_labels = [x for x in candidate_labels if x != candidate_label]
            continue

        test = current | tube
        current = test
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
    return MeshPathRepairResult(
        repaired=current,
        bridge_mask=bridge_mask,
        component_labels=component_labels,
        metrics=metrics,
    )
