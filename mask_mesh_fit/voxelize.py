from __future__ import annotations

import warnings

import numpy as np
import pyvista as pv
from scipy import ndimage
import torch

from .io_utils import ImageGeometry


def _faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
    faces = faces.astype(np.int64, copy=False)
    return np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()


def voxelize_mesh_to_mask(
    vertices_physical_xyz: np.ndarray,
    faces: np.ndarray,
    geometry: ImageGeometry,
    margin_voxels: int = 3,
    slab_depth: int = 16,
    backend: str = "pyvista",
) -> np.ndarray:
    backend = str(backend).strip().lower()
    if backend == "pyvista":
        return _voxelize_mesh_to_mask_pyvista(
            vertices_physical_xyz,
            faces,
            geometry,
            margin_voxels=margin_voxels,
            slab_depth=slab_depth,
        )
    if backend == "multigeomed":
        return _voxelize_mesh_to_mask_multigeomed(
            vertices_physical_xyz,
            faces,
            geometry,
            margin_voxels=margin_voxels,
        )
    raise ValueError(f"Unsupported voxelize backend: {backend}")


def _mesh_bbox_in_image(
    vertices_physical_xyz: np.ndarray,
    geometry: ImageGeometry,
    margin_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_xyz = geometry.physical_to_continuous_index_xyz(vertices_physical_xyz).astype(np.float32)
    shape_z, shape_y, shape_x = geometry.shape_zyx
    lo_xyz = np.floor(index_xyz.min(axis=0)).astype(int) - int(margin_voxels)
    hi_xyz = np.ceil(index_xyz.max(axis=0)).astype(int) + int(margin_voxels)
    lo_xyz = np.maximum(lo_xyz, np.array([0, 0, 0], dtype=int))
    hi_xyz = np.minimum(hi_xyz, np.array([shape_x - 1, shape_y - 1, shape_z - 1], dtype=int))
    if np.any(hi_xyz < lo_xyz):
        raise RuntimeError(f"Mesh bbox is outside image grid: lo={lo_xyz}, hi={hi_xyz}, shape={geometry.shape_zyx}")
    return index_xyz, lo_xyz, hi_xyz


def _voxelize_mesh_to_mask_pyvista(
    vertices_physical_xyz: np.ndarray,
    faces: np.ndarray,
    geometry: ImageGeometry,
    margin_voxels: int = 3,
    slab_depth: int = 16,
) -> np.ndarray:
    index_xyz, lo_xyz, hi_xyz = _mesh_bbox_in_image(vertices_physical_xyz, geometry, margin_voxels)

    surface = pv.PolyData(index_xyz[:, [0, 1, 2]], _faces_to_pyvista(faces))
    occupancy = np.zeros(geometry.shape_zyx, dtype=bool)

    x_values = np.arange(lo_xyz[0], hi_xyz[0] + 1, dtype=np.float32)
    y_values = np.arange(lo_xyz[1], hi_xyz[1] + 1, dtype=np.float32)
    z_start, z_end = int(lo_xyz[2]), int(hi_xyz[2]) + 1
    slab_depth = max(int(slab_depth), 1)

    for z0 in range(z_start, z_end, slab_depth):
        z_values = np.arange(z0, min(z0 + slab_depth, z_end), dtype=np.float32)
        zz, yy, xx = np.meshgrid(z_values, y_values, x_values, indexing="ij")
        points_xyz = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        cloud = pv.PolyData(points_xyz)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*select_interior_points.*")
            selected = cloud.select_enclosed_points(surface, tolerance=0.0, check_surface=False)
        inside = np.asarray(selected["SelectedPoints"], dtype=bool).reshape(
            len(z_values), len(y_values), len(x_values)
        )
        occupancy[
            z0 : z0 + len(z_values),
            int(lo_xyz[1]) : int(hi_xyz[1]) + 1,
            int(lo_xyz[0]) : int(hi_xyz[0]) + 1,
        ] = inside
    return occupancy


def _voxelize_mesh_to_mask_multigeomed(
    vertices_physical_xyz: np.ndarray,
    faces: np.ndarray,
    geometry: ImageGeometry,
    margin_voxels: int = 3,
) -> np.ndarray:
    """Voxelize through multigeomed's trimesh converter on the mesh crop.

    This is much faster than PyVista's per-slab enclosed-points query, but the
    multigeomed converter rasterizes/fills inside a crop grid and can be more
    generous than PyVista. For conservative comparisons, keep using the pyvista
    backend.
    """
    try:
        from multigeomed.converters import from_surface_mesh
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("multigeomed voxelization backend is not available") from exc

    index_xyz, lo_xyz, hi_xyz = _mesh_bbox_in_image(vertices_physical_xyz, geometry, margin_voxels)
    crop_shape_xyz = (hi_xyz - lo_xyz + 1).astype(int)
    vertices_crop_xyz = index_xyz - lo_xyz[None, :]
    metadata = {
        "spacing": (1.0, 1.0, 1.0),
        "origin": (0.0, 0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        "voxel_grid_shape": tuple(int(x) for x in crop_shape_xyz),
    }
    grid_xyz = from_surface_mesh.to_voxel_grid(
        torch.as_tensor(vertices_crop_xyz, dtype=torch.float32),
        torch.as_tensor(faces.astype(np.int64, copy=False), dtype=torch.long),
        metadata,
        tuple(int(x) for x in crop_shape_xyz),
        backend="trimesh",
        pitch_iso=1.0,
    )
    crop_occ_xyz = np.asarray(grid_xyz.detach().cpu()) > 0.5
    occupancy = np.zeros(geometry.shape_zyx, dtype=bool)
    occupancy[
        int(lo_xyz[2]) : int(hi_xyz[2]) + 1,
        int(lo_xyz[1]) : int(hi_xyz[1]) + 1,
        int(lo_xyz[0]) : int(hi_xyz[0]) + 1,
    ] = np.transpose(crop_occ_xyz, (2, 1, 0))
    return occupancy


def physical_ball_structure(spacing_xyz: np.ndarray, radius_mm: float) -> np.ndarray:
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float64)
    radius_mm = float(radius_mm)
    radii_xyz = np.maximum(np.ceil(radius_mm / spacing_xyz).astype(int), 1)
    dz, dy, dx = radii_xyz[2], radii_xyz[1], radii_xyz[0]
    z, y, x = np.ogrid[-dz : dz + 1, -dy : dy + 1, -dx : dx + 1]
    dist = (
        (x * spacing_xyz[0]) ** 2
        + (y * spacing_xyz[1]) ** 2
        + (z * spacing_xyz[2]) ** 2
    )
    return dist <= radius_mm**2


def _mask_bbox_slices(mask_zyx: np.ndarray, margin_zyx: np.ndarray) -> tuple[slice, slice, slice] | None:
    mask = mask_zyx.astype(bool, copy=False)
    z_idx = np.flatnonzero(mask.any(axis=(1, 2)))
    y_idx = np.flatnonzero(mask.any(axis=(0, 2)))
    x_idx = np.flatnonzero(mask.any(axis=(0, 1)))
    if z_idx.size == 0 or y_idx.size == 0 or x_idx.size == 0:
        return None
    shape = np.asarray(mask.shape, dtype=int)
    lo = np.asarray([z_idx[0], y_idx[0], x_idx[0]], dtype=int) - margin_zyx
    hi = np.asarray([z_idx[-1], y_idx[-1], x_idx[-1]], dtype=int) + margin_zyx + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))


def repair_mask_with_mesh(
    original_mask_zyx: np.ndarray,
    mesh_mask_zyx: np.ndarray,
    geometry: ImageGeometry,
    dilation_mm: float = 8.0,
    method: str = "edt_crop",
) -> tuple[np.ndarray, np.ndarray]:
    original = original_mask_zyx.astype(bool, copy=False)
    mesh = mesh_mask_zyx.astype(bool, copy=False)
    method = str(method).strip().lower()
    if method == "binary_dilation":
        structure = physical_ball_structure(geometry.spacing_xyz, dilation_mm)
        local_region = ndimage.binary_dilation(original, structure=structure)
        repaired = original | (mesh & local_region)
        return repaired, local_region
    if method not in {"edt", "edt_crop"}:
        raise ValueError(f"Unsupported repair mask method: {method}")

    local_region = np.zeros_like(original, dtype=bool)
    bbox_mask = mesh if method == "edt_crop" else np.ones_like(mesh, dtype=bool)
    spacing_zyx = np.asarray([geometry.spacing_xyz[2], geometry.spacing_xyz[1], geometry.spacing_xyz[0]], dtype=float)
    margin_zyx = np.maximum(np.ceil(float(dilation_mm) / spacing_zyx).astype(int), 1)
    crop = _mask_bbox_slices(bbox_mask, margin_zyx)
    if crop is not None:
        original_crop = original[crop]
        if original_crop.any():
            dist = ndimage.distance_transform_edt(~original_crop, sampling=tuple(float(x) for x in spacing_zyx))
            local_region[crop] = dist <= float(dilation_mm)
    repaired = original | (mesh & local_region)
    return repaired, local_region


def keep_components_connected_to_seed(
    mask_zyx: np.ndarray,
    seed_zyx: np.ndarray,
    connectivity: int = 26,
) -> np.ndarray:
    mask = mask_zyx.astype(bool, copy=False)
    seed = seed_zyx.astype(bool, copy=False)
    if not mask.any() or not seed.any():
        return mask.copy()
    if int(connectivity) == 6:
        structure = ndimage.generate_binary_structure(3, 1)
    elif int(connectivity) == 18:
        structure = ndimage.generate_binary_structure(3, 2)
    elif int(connectivity) == 26:
        structure = ndimage.generate_binary_structure(3, 3)
    else:
        raise ValueError("connectivity must be one of 6, 18, or 26")
    labels, num_labels = ndimage.label(mask, structure=structure)
    if num_labels == 0:
        return mask.copy()
    seed_labels = np.unique(labels[seed & (labels > 0)])
    if seed_labels.size == 0:
        return seed & mask
    return np.isin(labels, seed_labels)


def filter_components_touching_seed(
    candidate_zyx: np.ndarray,
    seed_zyx: np.ndarray,
    connectivity: int = 26,
) -> np.ndarray:
    candidate = candidate_zyx.astype(bool, copy=False)
    seed = seed_zyx.astype(bool, copy=False)
    if not candidate.any() or not seed.any():
        return np.zeros_like(candidate, dtype=bool)
    if int(connectivity) == 6:
        structure = ndimage.generate_binary_structure(3, 1)
    elif int(connectivity) == 18:
        structure = ndimage.generate_binary_structure(3, 2)
    elif int(connectivity) == 26:
        structure = ndimage.generate_binary_structure(3, 3)
    else:
        raise ValueError("connectivity must be one of 6, 18, or 26")
    labels, num_labels = ndimage.label(candidate, structure=structure)
    if num_labels == 0:
        return np.zeros_like(candidate, dtype=bool)
    touching_labels = np.unique(labels[seed & (labels > 0)])
    if touching_labels.size == 0:
        return np.zeros_like(candidate, dtype=bool)
    return np.isin(labels, touching_labels)


def repair_mask_with_mesh_sdf_blend(
    original_mask_zyx: np.ndarray,
    mesh_mask_zyx: np.ndarray,
    mesh_sdf_zyx: np.ndarray,
    geometry: ImageGeometry,
    dilation_mm: float = 8.0,
    sdf_threshold_mm: float = 1.5,
    local_region_method: str = "edt_crop",
    local_region_zyx: np.ndarray | None = None,
    connectivity: int = 26,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original = original_mask_zyx.astype(bool, copy=False)
    mesh = mesh_mask_zyx.astype(bool, copy=False)
    mesh_sdf = np.asarray(mesh_sdf_zyx, dtype=np.float32)
    if local_region_zyx is None:
        _, local_region = repair_mask_with_mesh(
            original,
            mesh,
            geometry,
            dilation_mm=dilation_mm,
            method=local_region_method,
        )
    else:
        local_region = local_region_zyx.astype(bool, copy=False)
    candidate = (mesh_sdf <= float(sdf_threshold_mm)) & local_region
    repaired_raw = original | candidate
    repaired = keep_components_connected_to_seed(repaired_raw, original, connectivity=connectivity)
    kept_candidate = repaired & ~original
    return repaired, candidate, local_region, kept_candidate


def repair_mask_with_component_gated_sdf(
    original_mask_zyx: np.ndarray,
    mesh_mask_zyx: np.ndarray,
    mesh_sdf_zyx: np.ndarray,
    geometry: ImageGeometry,
    dilation_mm: float = 8.0,
    sdf_threshold_mm: float = 1.5,
    local_region_method: str = "edt_crop",
    local_region_zyx: np.ndarray | None = None,
    connectivity: int = 26,
    touch_original_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original = original_mask_zyx.astype(bool, copy=False)
    mesh_sdf = np.asarray(mesh_sdf_zyx, dtype=np.float32)
    if local_region_zyx is None:
        _, local_region = repair_mask_with_mesh(
            original,
            mesh_mask_zyx,
            geometry,
            dilation_mm=dilation_mm,
            method=local_region_method,
        )
    else:
        local_region = local_region_zyx.astype(bool, copy=False)
    full_candidate = mesh_sdf <= float(sdf_threshold_mm)
    touch_seed = original | local_region if bool(touch_original_mask) else local_region
    kept_candidate = filter_components_touching_seed(
        full_candidate,
        touch_seed,
        connectivity=connectivity,
    )
    repaired = original | kept_candidate
    return repaired, full_candidate, local_region, kept_candidate


def mesh_signed_distance(
    mesh_mask_zyx: np.ndarray,
    spacing_xyz: np.ndarray,
    clip_mm: float = 16.0,
    crop_to_mesh: bool = True,
) -> np.ndarray:
    mask = mesh_mask_zyx.astype(bool, copy=False)
    sampling_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    clip_mm = float(max(clip_mm, 1e-6))
    if not crop_to_mesh:
        outside = ndimage.distance_transform_edt(~mask, sampling=sampling_zyx)
        inside = ndimage.distance_transform_edt(mask, sampling=sampling_zyx)
        sdf = outside - inside
        return np.clip(sdf, -clip_mm, clip_mm).astype(np.float32, copy=False)
    sdf_full = np.full(mask.shape, clip_mm, dtype=np.float32)
    if not mask.any():
        return sdf_full
    margin_zyx = np.maximum(np.ceil(clip_mm / np.asarray(sampling_zyx)).astype(int), 1) + 1
    crop = _mask_bbox_slices(mask, margin_zyx)
    if crop is None:
        return sdf_full
    crop_mask = mask[crop]
    outside = ndimage.distance_transform_edt(~crop_mask, sampling=sampling_zyx)
    inside = ndimage.distance_transform_edt(crop_mask, sampling=sampling_zyx)
    sdf_crop = np.clip(outside - inside, -clip_mm, clip_mm).astype(np.float32, copy=False)
    sdf_full[crop] = sdf_crop
    return sdf_full
