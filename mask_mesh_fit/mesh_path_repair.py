from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .bridge_tube import (
    RadiusPolicy,
    TubeTrial,
    grow_tube_until_merge,
    local_component_radius_mm,
    voxel_step_mm,
)
from .io_utils import ImageGeometry


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


@dataclass(frozen=True)
class _BridgeOptions:
    """Validated settings for mesh-graph bridge repair.

    Everything about how wide a bridge may be lives in :class:`RadiusPolicy`, which
    the endpoint strategy shares; the fields here are the ones specific to walking
    the mesh graph.
    """

    anchor_mm: float
    radius: RadiusPolicy
    min_component_voxels: int
    max_added_fraction: float
    connectivity: int
    path_selection: str
    min_path_edges: int
    max_mesh_edge_mm: float
    edge_filter_fallback: str

    @property
    def edge_filtered(self) -> bool:
        return self.path_selection == "edge_filtered_shortest"


def _resolve_bridge_options(
    geometry: ImageGeometry,
    *,
    anchor_mm: float,
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
    path_selection: str,
    min_path_edges: int,
    max_mesh_edge_mm: float,
    edge_filter_fallback: str,
) -> _BridgeOptions:
    """Check the option values and settle the radius bounds."""
    path_selection = str(path_selection).strip().lower()
    if path_selection not in {"shortest", "edge_filtered_shortest"}:
        raise ValueError("path_selection must be one of: shortest, edge_filtered_shortest")
    edge_filter_fallback = str(edge_filter_fallback).strip().lower()
    if edge_filter_fallback not in {"old_shortest", "skip"}:
        raise ValueError("edge_filter_fallback must be one of: old_shortest, skip")

    # With no explicit cap, a bridge may grow to the anchor distance plus one voxel:
    # wide enough to reach the mask it was anchored to, and no wider.
    auto_max_radius_mm = max(float(radius_mm), float(anchor_mm) + voxel_step_mm(geometry))
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

    return _BridgeOptions(
        anchor_mm=float(anchor_mm),
        radius=policy,
        min_component_voxels=int(min_component_voxels),
        max_added_fraction=float(max_added_fraction),
        connectivity=int(connectivity),
        path_selection=path_selection,
        min_path_edges=max(int(min_path_edges), 0),
        max_mesh_edge_mm=float(max_mesh_edge_mm),
        edge_filter_fallback=edge_filter_fallback,
    )


def _initial_bridge_metrics(
    options: _BridgeOptions,
    original_components: int,
    max_added_voxels: int,
) -> dict[str, object]:
    """The metrics record, pre-filled with the settings the run used."""
    metrics: dict[str, object] = {
        "repair_status": "",
        "original_components": int(original_components),
        "repaired_components": int(original_components),
        "accepted_paths": 0,
        "rejected_paths": 0,
        "added_voxels": 0,
        "max_added_voxels": int(max_added_voxels),
        "anchor_mm": options.anchor_mm,
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
    if options.edge_filtered:
        metrics.update(
            {
                "path_selection": options.path_selection,
                "min_path_edges": options.min_path_edges,
                "max_mesh_edge_mm": options.max_mesh_edge_mm,
                "edge_filter_fallback": options.edge_filter_fallback,
            }
        )
    return metrics


@dataclass
class _BridgeCandidate:
    """The shortest mesh path found to one disconnected component."""

    component_label: int
    path: list[int]
    path_length_mm: float
    filtered_path_used: bool
    fallback_used: bool


def _select_best_path(
    adjacency: list[list[tuple[int, float]]],
    filtered_adjacency: list[list[tuple[int, float]]],
    vertex_component: np.ndarray,
    main_labels: set[int],
    candidate_labels: list[int],
    component_sizes: np.ndarray,
    anchor_counts: dict[int, int],
    options: _BridgeOptions,
) -> tuple[_BridgeCandidate | None, list[dict[str, object]]]:
    """Find the shortest mesh path from the main component to any candidate.

    Returns the winner and a record for every candidate that had no usable path.
    Under ``edge_filtered_shortest`` a candidate that only has an unfiltered path is
    held back as a fallback and used only if nothing else wins.
    """
    rejections: list[dict[str, object]] = []
    source_vertices = np.flatnonzero(np.isin(vertex_component, list(main_labels)))
    if source_vertices.size == 0:
        return None, rejections

    best: _BridgeCandidate | None = None
    fallback_best: _BridgeCandidate | None = None
    for candidate_label in candidate_labels:
        target_vertices = np.flatnonzero(vertex_component == int(candidate_label))
        if options.edge_filtered:
            path_result = _shortest_mesh_path_min_edges(
                filtered_adjacency,
                source_vertices,
                target_vertices,
                min_edges=options.min_path_edges,
            )
        else:
            path_result = _shortest_mesh_path(adjacency, source_vertices, target_vertices)

        if path_result is None:
            if options.edge_filtered and options.edge_filter_fallback == "old_shortest":
                fallback_result = _shortest_mesh_path(adjacency, source_vertices, target_vertices)
                if fallback_result is not None:
                    fallback_path, fallback_length = fallback_result
                    if fallback_best is None or fallback_length < fallback_best.path_length_mm:
                        fallback_best = _BridgeCandidate(
                            component_label=int(candidate_label),
                            path=fallback_path,
                            path_length_mm=float(fallback_length),
                            filtered_path_used=False,
                            fallback_used=True,
                        )
                    continue
            rejection: dict[str, object] = {
                "component_label": int(candidate_label),
                "status": "rejected_no_mesh_path",
                "component_voxels": int(component_sizes[int(candidate_label)]),
                "anchor_count": int(anchor_counts.get(int(candidate_label), 0)),
            }
            if options.edge_filtered:
                rejection.update(
                    {
                        "path_selection": options.path_selection,
                        "filtered_path_used": False,
                        "fallback_used": False,
                    }
                )
            rejections.append(rejection)
            continue

        path, path_length = path_result
        if best is None or path_length < best.path_length_mm:
            best = _BridgeCandidate(
                component_label=int(candidate_label),
                path=path,
                path_length_mm=float(path_length),
                filtered_path_used=options.edge_filtered,
                fallback_used=False,
            )

    if best is None:
        best = fallback_best
    return best, rejections


def _bridge_path_points(
    path: list[int],
    vertices_physical_xyz: np.ndarray,
    vertex_nearest_physical_xyz: np.ndarray,
) -> np.ndarray:
    """Physical points of the bridge, extended to the mask voxels at both ends.

    The mesh path runs between vertices near the two components, not to the mask
    itself, so the nearest mask point at each end is prepended and appended. Without
    that the rasterised tube can stop short and fail to merge.
    """
    path_points = vertices_physical_xyz[np.asarray(path, dtype=np.int64)]
    if not path:
        return path_points

    start_anchor = vertex_nearest_physical_xyz[int(path[0])]
    end_anchor = vertex_nearest_physical_xyz[int(path[-1])]
    pieces = []
    if np.all(np.isfinite(start_anchor)):
        pieces.append(start_anchor[None, :])
    pieces.append(path_points)
    if np.all(np.isfinite(end_anchor)):
        pieces.append(end_anchor[None, :])
    return np.concatenate(pieces, axis=0)


def _endpoint_local_radii_mm(
    path: list[int],
    component_labels: np.ndarray,
    vertex_component: np.ndarray,
    vertex_nearest_physical_xyz: np.ndarray,
    geometry: ImageGeometry,
    options: _BridgeOptions,
) -> tuple[float, float]:
    """Local vessel radius of the mask at each end of the bridge."""
    main_radius = float("nan")
    candidate_radius = float("nan")
    if not path:
        return main_radius, candidate_radius

    main_label = int(vertex_component[int(path[0])])
    candidate_label = int(vertex_component[int(path[-1])])
    if main_label > 0:
        main_radius = local_component_radius_mm(
            component_labels,
            main_label,
            vertex_nearest_physical_xyz[int(path[0])],
            geometry,
            window_mm=options.radius.local_window_mm,
            percentile=options.radius.percentile,
        )
    if candidate_label > 0:
        candidate_radius = local_component_radius_mm(
            component_labels,
            candidate_label,
            vertex_nearest_physical_xyz[int(path[-1])],
            geometry,
            window_mm=options.radius.local_window_mm,
            percentile=options.radius.percentile,
        )
    return main_radius, candidate_radius


def _bridge_metrics(
    candidate: _BridgeCandidate,
    trial: TubeTrial,
    component_sizes: np.ndarray,
    anchor_counts: dict[int, int],
    vertices_physical_xyz: np.ndarray,
    main_local_radius: float,
    candidate_local_radius: float,
    adaptive_accept_radius: float,
    options: _BridgeOptions,
) -> dict[str, object]:
    """One bridge's record: which component, how long, how wide, how much it added."""
    label = int(candidate.component_label)
    metrics: dict[str, object] = {
        "component_label": label,
        "component_voxels": int(component_sizes[label]),
        "anchor_count": int(anchor_counts.get(label, 0)),
        "path_vertices": int(len(candidate.path)),
        "path_length_mm": float(candidate.path_length_mm),
        "added_voxels": int(trial.added_voxels),
        "selected_radius_mm": float(trial.selected_radius_mm),
        "radius_mode": options.radius.mode,
        "adaptive_accept_radius_mm": float(adaptive_accept_radius),
    }
    if options.edge_filtered:
        metrics.update(
            {
                "path_selection": options.path_selection,
                "filtered_path_used": bool(candidate.filtered_path_used),
                "fallback_used": bool(candidate.fallback_used),
                **_path_edge_stats(vertices_physical_xyz, candidate.path),
            }
        )
    if options.radius.adaptive:
        metrics["main_local_radius_mm"] = float(main_local_radius)
        metrics["candidate_local_radius_mm"] = float(candidate_local_radius)
        metrics["local_radius_window_mm"] = options.radius.local_window_mm
        metrics["radius_percentile"] = options.radius.percentile
        metrics["radius_scale"] = options.radius.scale
    return metrics


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
    """Reconnect a fragmented mask along paths on the fitted mesh graph.

    Disconnected components are anchored to nearby mesh vertices, and the shortest
    path on the mesh graph between the main component and each candidate becomes a
    bridge proposal. A proposal is rasterised as a thin tube and kept only if it
    actually merges the two components while adding fewer voxels than the
    foreground-growth budget allows. Accepted bridges are absorbed into the main
    component and the search repeats until nothing is left to join.

    The mesh is never voxelised: only accepted tubes are added to the mask.
    """
    original = original_mask_zyx.astype(bool, copy=False)
    options = _resolve_bridge_options(
        geometry,
        anchor_mm=anchor_mm,
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
        path_selection=path_selection,
        min_path_edges=min_path_edges,
        max_mesh_edge_mm=max_mesh_edge_mm,
        edge_filter_fallback=edge_filter_fallback,
    )

    structure = _connectivity_structure(options.connectivity)
    component_labels, original_components = ndimage.label(original, structure=structure)
    component_labels = component_labels.astype(np.int32, copy=False)
    component_sizes = _component_sizes(component_labels, original_components)
    original_voxels = int(original.sum())
    max_added_voxels = int(max(1, np.floor(options.max_added_fraction * max(original_voxels, 1))))

    bridge_mask = np.zeros_like(original, dtype=bool)
    current = original.copy()
    metrics = _initial_bridge_metrics(options, original_components, max_added_voxels)

    def finished(status: str) -> MeshPathRepairResult:
        metrics["repair_status"] = status
        return MeshPathRepairResult(
            repaired=current,
            bridge_mask=bridge_mask,
            component_labels=component_labels,
            metrics=metrics,
        )

    if original_components <= 1:
        return finished("noop_already_connected")
    if original_voxels <= 0:
        return finished("noop_empty_mask")

    if edge_index is None or np.asarray(edge_index).size == 0:
        edge_index = edge_index_from_faces(faces)
    adjacency = _mesh_adjacency(vertices_physical_xyz, np.asarray(edge_index, dtype=np.int64))
    if not any(adjacency):
        return finished("noop_empty_mesh_graph")
    filtered_adjacency = _filter_adjacency_by_edge_length(adjacency, options.max_mesh_edge_mm)

    vertex_component, _vertex_distance, vertex_nearest_physical_xyz = _mesh_vertex_component_anchors(
        vertices_physical_xyz,
        component_labels,
        geometry,
        anchor_mm=options.anchor_mm,
    )
    anchor_counts = {
        int(label): int(np.count_nonzero(vertex_component == label))
        for label in range(1, original_components + 1)
    }
    metrics["anchor_counts"] = anchor_counts

    component_order = [
        int(label)
        for label in np.argsort(component_sizes[1:])[::-1] + 1
        if component_sizes[int(label)] >= options.min_component_voxels
    ]
    if not component_order:
        return finished("noop_no_large_components")

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

        candidate, rejections = _select_best_path(
            adjacency,
            filtered_adjacency,
            vertex_component,
            main_labels,
            candidate_labels,
            component_sizes,
            anchor_counts,
            options,
        )
        rejected += len(rejections)
        metrics["paths"].extend(rejections)
        if candidate is None:
            break

        label = int(candidate.component_label)
        path_points = _bridge_path_points(candidate.path, vertices_physical_xyz, vertex_nearest_physical_xyz)
        main_seed = np.isin(component_labels, list(main_labels))
        candidate_seed = component_labels == label

        main_local_radius, candidate_local_radius = (float("nan"), float("nan"))
        if options.radius.adaptive:
            main_local_radius, candidate_local_radius = _endpoint_local_radii_mm(
                candidate.path,
                component_labels,
                vertex_component,
                vertex_nearest_physical_xyz,
                geometry,
                options,
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

        path_metrics = _bridge_metrics(
            candidate,
            trial,
            component_sizes,
            anchor_counts,
            vertices_physical_xyz,
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

        current = current | trial.tube
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
