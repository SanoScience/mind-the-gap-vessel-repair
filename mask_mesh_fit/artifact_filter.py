from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .io_utils import ImageGeometry


@dataclass(frozen=True)
class MaskArtifactFilterResult:
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


def filter_mask_artifacts_by_component_distance(
    mask_zyx: np.ndarray,
    geometry: ImageGeometry,
    *,
    keep_near_main_mm: float = 12.0,
    remove_distance_mm: float = 25.0,
    max_remove_voxels: int = 50,
    connectivity: int = 26,
) -> MaskArtifactFilterResult:
    mask = mask_zyx.astype(bool, copy=False)
    structure = _connectivity_structure(connectivity)
    component_labels, original_components = ndimage.label(mask, structure=structure)
    component_labels = component_labels.astype(np.int32, copy=False)
    component_sizes = np.bincount(component_labels.ravel(), minlength=original_components + 1).astype(
        np.int64,
        copy=False,
    )
    voxel_volume_mm3 = float(np.prod(geometry.spacing_xyz))
    cleaned = mask.copy()
    removed = np.zeros_like(mask, dtype=bool)

    metrics: dict[str, object] = {
        "method": "component_distance",
        "connectivity": int(connectivity),
        "keep_near_main_mm": float(keep_near_main_mm),
        "remove_distance_mm": float(remove_distance_mm),
        "max_remove_voxels": int(max_remove_voxels),
        "max_remove_voxels_disabled": bool(int(max_remove_voxels) <= 0),
        "original_components": int(original_components),
        "cleaned_components": int(original_components),
        "main_component_label": 0,
        "main_component_voxels": 0,
        "removed_components": [],
        "kept_near_components": [],
        "large_far_kept_components": [],
        "kept_mid_distance_components": [],
        "removed_voxels": 0,
        "removed_volume_mm3": 0.0,
        "original_voxels": int(mask.sum()),
        "cleaned_voxels": int(mask.sum()),
    }

    if original_components <= 1:
        return MaskArtifactFilterResult(
            cleaned_mask=cleaned,
            removed_mask=removed,
            component_labels=component_labels,
            metrics=metrics,
        )

    main_label = int(np.argmax(component_sizes[1:]) + 1)
    metrics["main_component_label"] = main_label
    metrics["main_component_voxels"] = int(component_sizes[main_label])

    main_points_zyx = np.argwhere(component_labels == main_label)
    if main_points_zyx.size == 0:
        return MaskArtifactFilterResult(
            cleaned_mask=cleaned,
            removed_mask=removed,
            component_labels=component_labels,
            metrics=metrics,
        )

    main_points_physical = _points_zyx_to_physical_xyz(main_points_zyx, geometry)
    main_tree = cKDTree(main_points_physical)

    for label in range(1, original_components + 1):
        if label == main_label:
            continue
        component_voxels = int(component_sizes[label])
        component_points_zyx = np.argwhere(component_labels == label)
        if component_points_zyx.size == 0:
            continue
        component_points_physical = _points_zyx_to_physical_xyz(component_points_zyx, geometry)
        try:
            distances, _nearest = main_tree.query(component_points_physical, k=1, workers=-1)
        except TypeError:  # pragma: no cover - older scipy compatibility
            distances, _nearest = main_tree.query(component_points_physical, k=1)
        min_distance_mm = float(np.min(distances)) if distances.size else float("inf")

        entry = {
            "label": int(label),
            "voxels": component_voxels,
            "min_distance_to_main_mm": min_distance_mm,
        }
        if min_distance_mm <= float(keep_near_main_mm):
            metrics["kept_near_components"].append(entry)
            continue
        size_cap_disabled = int(max_remove_voxels) <= 0
        if min_distance_mm >= float(remove_distance_mm) and (
            size_cap_disabled or component_voxels <= int(max_remove_voxels)
        ):
            component_mask = component_labels == label
            removed |= component_mask
            cleaned[component_mask] = False
            metrics["removed_components"].append(entry)
            continue
        if (
            not size_cap_disabled
            and min_distance_mm >= float(remove_distance_mm)
            and component_voxels > int(max_remove_voxels)
        ):
            metrics["large_far_kept_components"].append(entry)
        else:
            metrics["kept_mid_distance_components"].append(entry)

    _, cleaned_components = ndimage.label(cleaned, structure=structure)
    removed_voxels = int(removed.sum())
    metrics["cleaned_components"] = int(cleaned_components)
    metrics["removed_voxels"] = removed_voxels
    metrics["removed_volume_mm3"] = float(removed_voxels * voxel_volume_mm3)
    metrics["cleaned_voxels"] = int(cleaned.sum())
    return MaskArtifactFilterResult(
        cleaned_mask=cleaned,
        removed_mask=removed,
        component_labels=component_labels,
        metrics=metrics,
    )
