from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from skimage import measure


@dataclass(frozen=True)
class ImageGeometry:
    shape_zyx: tuple[int, int, int]
    spacing_xyz: np.ndarray
    origin_xyz: np.ndarray
    direction_xyz: np.ndarray

    @classmethod
    def from_image(cls, image: sitk.Image, shape_zyx: tuple[int, int, int]) -> "ImageGeometry":
        return cls(
            shape_zyx=tuple(int(v) for v in shape_zyx),
            spacing_xyz=np.asarray(image.GetSpacing(), dtype=np.float64),
            origin_xyz=np.asarray(image.GetOrigin(), dtype=np.float64),
            direction_xyz=np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3),
        )

    def continuous_index_xyz_to_physical(self, index_xyz: np.ndarray) -> np.ndarray:
        index_xyz = np.asarray(index_xyz, dtype=np.float64)
        scaled = index_xyz * self.spacing_xyz[None, :]
        return self.origin_xyz[None, :] + scaled @ self.direction_xyz.T

    def physical_to_continuous_index_xyz(self, points_xyz: np.ndarray) -> np.ndarray:
        points_xyz = np.asarray(points_xyz, dtype=np.float64)
        scaled = (points_xyz - self.origin_xyz[None, :]) @ self.direction_xyz
        return scaled / self.spacing_xyz[None, :]


@dataclass(frozen=True)
class Normalization:
    center_xyz: np.ndarray
    scale: float

    def to_norm(self, points_xyz: np.ndarray) -> np.ndarray:
        return ((points_xyz - self.center_xyz[None, :]) / self.scale).astype(np.float32)

    def to_physical(self, points_norm_xyz: np.ndarray) -> np.ndarray:
        return (points_norm_xyz * self.scale + self.center_xyz[None, :]).astype(np.float32)


def parse_case_id(mask_path: Path) -> str:
    name = mask_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return mask_path.stem


def load_mask(mask_path: Path) -> tuple[np.ndarray, sitk.Image, ImageGeometry]:
    image = sitk.ReadImage(str(mask_path))
    array = sitk.GetArrayFromImage(image)
    mask = array > 0
    if mask.ndim != 3:
        raise RuntimeError(f"{mask_path}: expected 3D mask, got shape {mask.shape}")
    if not np.any(mask):
        raise RuntimeError(f"{mask_path}: mask is empty")
    return mask.astype(bool, copy=False), image, ImageGeometry.from_image(image, mask.shape)


def save_mask_like(mask_zyx: np.ndarray, reference_image: sitk.Image, output_path: Path) -> None:
    out = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    out.CopyInformation(reference_image)
    sitk.WriteImage(out, str(output_path))


def save_label_like(labels_zyx: np.ndarray, reference_image: sitk.Image, output_path: Path) -> None:
    out = sitk.GetImageFromArray(labels_zyx.astype(np.int32, copy=False))
    out.CopyInformation(reference_image)
    sitk.WriteImage(out, str(output_path))


def save_float_like(array_zyx: np.ndarray, reference_image: sitk.Image, output_path: Path) -> None:
    out = sitk.GetImageFromArray(array_zyx.astype(np.float32, copy=False))
    out.CopyInformation(reference_image)
    sitk.WriteImage(out, str(output_path))


def extract_surface_from_mask(
    mask_zyx: np.ndarray,
    geometry: ImageGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(mask_zyx.shape) < 2:
        raise RuntimeError(f"Mask shape is too small for marching cubes: {mask_zyx.shape}")
    padded = np.pad(mask_zyx.astype(np.float32, copy=False), 1, mode="constant", constant_values=0)
    verts_zyx, faces, _normals, _values = measure.marching_cubes(
        padded,
        level=0.5,
        allow_degenerate=False,
    )
    verts_zyx = verts_zyx - 1.0
    index_xyz = verts_zyx[:, [2, 1, 0]]
    physical_xyz = geometry.continuous_index_xyz_to_physical(index_xyz)
    return (
        physical_xyz.astype(np.float32, copy=False),
        faces.astype(np.int64, copy=False),
        verts_zyx.astype(np.float32, copy=False),
    )


def make_bbox_normalization(points_xyz: np.ndarray, eps: float = 1e-6) -> Normalization:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    lo = points_xyz.min(axis=0)
    hi = points_xyz.max(axis=0)
    center = 0.5 * (lo + hi)
    scale = 0.5 * float(np.max(hi - lo))
    if not np.isfinite(scale) or scale < eps:
        raise RuntimeError(f"Invalid normalization scale from bbox: {scale}")
    return Normalization(center_xyz=center.astype(np.float32), scale=scale)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
