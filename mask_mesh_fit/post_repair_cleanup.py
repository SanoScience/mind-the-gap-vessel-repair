from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .io_utils import ImageGeometry


@dataclass(frozen=True)
class PostRepairCleanupResult:
    cleaned_mask: np.ndarray
    removed_mask: np.ndarray
    component_labels: np.ndarray
    metrics: dict[str, object]


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if int(connectivity) == 6:
        return ndimage.generate_binary_structure(3, 1)
    if int(connectivity) == 18:
        return ndimage.generate_binary_structure(3, 2)
    if int(connectivity) == 26:
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError("connectivity must be one of 6, 18, or 26")


def _points_zyx_to_physical_xyz(points_zyx: np.ndarray, geometry: ImageGeometry) -> np.ndarray:
    points_zyx = np.asarray(points_zyx, dtype=np.float64)
    points_xyz = points_zyx[:, [2, 1, 0]]
    return geometry.continuous_index_xyz_to_physical(points_xyz)


def _mesh_support_points(vertices_physical_xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices_physical_xyz, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_physical_xyz must have shape (N, 3)")
    if faces.ndim == 2 and faces.shape[1] == 3 and faces.size > 0:
        valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
        centroids = vertices[faces[valid]].mean(axis=1) if np.any(valid) else np.empty((0, 3), dtype=np.float64)
        return np.concatenate([vertices, centroids], axis=0)
    return vertices


def remove_mesh_uncovered_components(
    repaired_mask_zyx: np.ndarray,
    vertices_physical_xyz: np.ndarray,
    faces: np.ndarray,
    geometry: ImageGeometry,
    *,
    mesh_distance_mm: float = 2.0,
    min_close_fraction: float = 0.01,
    max_remove_voxels: int = 1000,
    connectivity: int = 26,
) -> PostRepairCleanupResult:
    repaired = repaired_mask_zyx.astype(bool, copy=False)
    structure = _connectivity_structure(connectivity)
    labels, num_components = ndimage.label(repaired, structure=structure)
    labels = labels.astype(np.int32, copy=False)
    sizes = np.bincount(labels.ravel(), minlength=num_components + 1).astype(np.int64, copy=False)
    removed = np.zeros_like(repaired, dtype=bool)
    cleaned = repaired.copy()
    support_points = _mesh_support_points(vertices_physical_xyz, faces)
    voxel_volume_mm3 = float(np.prod(geometry.spacing_xyz))

    metrics: dict[str, object] = {
        "method": "remove_mesh_uncovered_components",
        "connectivity": int(connectivity),
        "mesh_distance_mm": float(mesh_distance_mm),
        "min_close_fraction": float(min_close_fraction),
        "max_remove_voxels": int(max_remove_voxels),
        "pre_cleanup_components": int(num_components),
        "post_cleanup_components": int(num_components),
        "main_component_label": 0,
        "main_component_voxels": 0,
        "removed_components": [],
        "kept_mesh_supported_components": [],
        "kept_large_uncovered_components": [],
        "removed_voxels": 0,
        "removed_volume_mm3": 0.0,
        "pre_cleanup_voxels": int(repaired.sum()),
        "post_cleanup_voxels": int(repaired.sum()),
    }

    if num_components <= 1 or repaired.sum() == 0:
        return PostRepairCleanupResult(
            cleaned_mask=cleaned,
            removed_mask=removed,
            component_labels=labels,
            metrics=metrics,
        )

    main_label = int(np.argmax(sizes[1:]) + 1)
    metrics["main_component_label"] = main_label
    metrics["main_component_voxels"] = int(sizes[main_label])

    if support_points.size == 0:
        metrics["support_points"] = 0
        return PostRepairCleanupResult(
            cleaned_mask=cleaned,
            removed_mask=removed,
            component_labels=labels,
            metrics=metrics,
        )

    support_tree = cKDTree(support_points)
    metrics["support_points"] = int(len(support_points))

    for label in range(1, num_components + 1):
        if label == main_label:
            continue
        component_voxels = int(sizes[label])
        component_mask = labels == label
        points_zyx = np.argwhere(component_mask)
        if points_zyx.size == 0:
            continue
        points_physical = _points_zyx_to_physical_xyz(points_zyx, geometry)
        try:
            distances, _nearest = support_tree.query(points_physical, k=1, workers=-1)
        except TypeError:  # pragma: no cover - older scipy compatibility
            distances, _nearest = support_tree.query(points_physical, k=1)
        close_fraction = float(np.mean(distances <= float(mesh_distance_mm))) if distances.size else 0.0
        min_distance_mm = float(np.min(distances)) if distances.size else float("inf")
        median_distance_mm = float(np.median(distances)) if distances.size else float("inf")
        entry = {
            "label": int(label),
            "voxels": component_voxels,
            "close_fraction": close_fraction,
            "min_distance_to_mesh_mm": min_distance_mm,
            "median_distance_to_mesh_mm": median_distance_mm,
        }

        if component_voxels > int(max_remove_voxels):
            metrics["kept_large_uncovered_components"].append(entry)
            continue
        if close_fraction < float(min_close_fraction):
            removed |= component_mask
            cleaned[component_mask] = False
            metrics["removed_components"].append(entry)
        else:
            metrics["kept_mesh_supported_components"].append(entry)

    _, post_components = ndimage.label(cleaned, structure=structure)
    removed_voxels = int(removed.sum())
    metrics["post_cleanup_components"] = int(post_components)
    metrics["removed_voxels"] = removed_voxels
    metrics["removed_volume_mm3"] = float(removed_voxels * voxel_volume_mm3)
    metrics["post_cleanup_voxels"] = int(cleaned.sum())
    return PostRepairCleanupResult(
        cleaned_mask=cleaned,
        removed_mask=removed,
        component_labels=labels,
        metrics=metrics,
    )
