from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch

from mask_mesh_fit.geometry import load_template
from mask_mesh_fit.io_utils import (
    extract_surface_from_mask,
    load_mask,
    make_bbox_normalization,
    save_float_like,
    save_mask_like,
    save_npz,
)
from mask_mesh_fit.optimize import DecoderConfig, StageConfig, fit_mesh_to_target
from mask_mesh_fit.voxelize import mesh_signed_distance, voxelize_mesh_to_mask


FIXED_A_VAL_CASES = [
    "A049",
    "A055",
    "A061",
    "A062",
    "A064",
    "A069",
    "A073",
    "A080",
    "A084",
    "A093",
    "A097",
    "A099",
    "A110",
    "A114",
    "A115",
    "A129",
    "A130",
    "A135",
    "A137",
    "A141",
]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def case_id_from_path(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def list_cases(pred_mask_dir: Path, gt_mask_dir: Path | None, case_prefix: str = "") -> list[str]:
    cases = []
    for path in sorted(pred_mask_dir.glob("*.nii.gz")):
        case_id = case_id_from_path(path)
        if case_prefix and not case_id.startswith(case_prefix):
            continue
        if gt_mask_dir is not None and not (gt_mask_dir / f"{case_id}.nii.gz").exists():
            continue
        cases.append(case_id)
    return cases


def fixed_a_16_4_split(cases: list[str]) -> tuple[list[str], list[str]]:
    available = [case_id for case_id in FIXED_A_VAL_CASES if case_id in set(cases)]
    if len(available) < 20:
        available = sorted(cases)
    if len(available) < 5:
        raise RuntimeError(f"Need at least 5 cases for fixed_a_16_4 split, got {len(available)}")
    return available[:16], available[16:20]


def split_cases(
    pred_mask_dir: Path,
    gt_mask_dir: Path,
    case_prefix: str,
    split_mode: str,
    train_cases: list[str] | None = None,
    val_cases: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    if train_cases and val_cases:
        return list(train_cases), list(val_cases)
    cases = list_cases(pred_mask_dir, gt_mask_dir, case_prefix=case_prefix)
    if split_mode == "fixed_a_16_4":
        return fixed_a_16_4_split(cases)
    if split_mode == "all_train":
        return cases, cases[: min(4, len(cases))]
    raise ValueError(f"Unknown split_mode: {split_mode}")


def read_bool_mask(path: Path) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3:
        raise RuntimeError(f"{path}: expected 3D image, got shape {array.shape}")
    return (array > 0), image


def read_float_image(path: Path) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    if array.ndim != 3:
        raise RuntimeError(f"{path}: expected 3D image, got shape {array.shape}")
    return array, image


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    inter = float(np.logical_and(pred, target).sum())
    return (2.0 * inter + eps) / (float(pred.sum()) + float(target.sum()) + eps)


def mask_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_b = pred.astype(bool, copy=False)
    target_b = target.astype(bool, copy=False)
    tp = float(np.logical_and(pred_b, target_b).sum())
    fp = float(np.logical_and(pred_b, ~target_b).sum())
    fn = float(np.logical_and(~pred_b, target_b).sum())
    return {
        "dice": dice_score(pred_b, target_b),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "pred_voxels": float(pred_b.sum()),
        "target_voxels": float(target_b.sum()),
    }


def crop_slices_from_mask(mask_zyx: np.ndarray, spacing_xyz: np.ndarray, margin_mm: float) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask_zyx.astype(bool, copy=False))
    if coords.size == 0:
        return (slice(0, mask_zyx.shape[0]), slice(0, mask_zyx.shape[1]), slice(0, mask_zyx.shape[2]))
    spacing_zyx = np.asarray([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float64)
    margin_vox = np.maximum(np.ceil(float(margin_mm) / spacing_zyx).astype(int), 1)
    lo = np.maximum(coords.min(axis=0) - margin_vox, 0)
    hi = np.minimum(coords.max(axis=0) + margin_vox + 1, np.asarray(mask_zyx.shape))
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def make_refiner_input(
    pred_mask_zyx: np.ndarray,
    mesh_mask_zyx: np.ndarray,
    mesh_sdf_zyx: np.ndarray,
    spacing_xyz: np.ndarray,
    crop_margin_mm: float,
    sdf_clip_mm: float,
) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    union = pred_mask_zyx.astype(bool, copy=False) | mesh_mask_zyx.astype(bool, copy=False)
    crop = crop_slices_from_mask(union, spacing_xyz=spacing_xyz, margin_mm=crop_margin_mm)
    pred = pred_mask_zyx[crop].astype(np.float32, copy=False)
    mesh = mesh_mask_zyx[crop].astype(np.float32, copy=False)
    sdf = np.clip(mesh_sdf_zyx[crop], -sdf_clip_mm, sdf_clip_mm).astype(np.float32, copy=False)
    sdf = sdf / max(float(sdf_clip_mm), 1e-6)
    return np.stack([pred, mesh, sdf], axis=0).astype(np.float32, copy=False), crop


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def default_mesh_fit_configs(
    fit_steps: int,
    target_points: int,
    stage_max_offsets: tuple[float, ...],
    lr: float,
) -> tuple[StageConfig, StageConfig, StageConfig, DecoderConfig]:
    align_cfg = StageConfig(
        name="align",
        steps=250,
        lr=0.02,
        target_points=min(2048, target_points),
        lambda_chamfer=0.10,
        lambda_bbox=1.0,
    )
    coarse_cfg = StageConfig(
        name="coarse",
        steps=0,
        lr=lr,
        target_points=target_points,
        lambda_chamfer=1.0,
    )
    detail_cfg = StageConfig(
        name="detail",
        steps=fit_steps,
        lr=lr,
        target_points=target_points,
        lambda_chamfer=1.0,
        lambda_edge=0.03125,
        lambda_laplacian=1.25,
        lambda_normal=0.00025,
        lambda_face_area_var=2.5e-05,
    )
    decoder_cfg = DecoderConfig(
        deformation_mode="decoder",
        stages=len(stage_max_offsets),
        latent_dim=128,
        local_feature_dim=128,
        hidden_dim=128,
        num_blocks=3,
        graph_layer="gcn",
        edge_features="none",
        stage_max_offsets=stage_max_offsets,
        stage_loss_weight=0.0,
        fit_steps=fit_steps,
        train_alignment=False,
    )
    return align_cfg, coarse_cfg, detail_cfg, decoder_cfg


def ensure_case_mesh_cache(
    case_id: str,
    pred_mask_dir: Path,
    template_path: Path,
    cache_dir: Path,
    device: torch.device,
    *,
    cache_mode: str = "on_demand",
    force_rebuild: bool = False,
    template_radius: float = 0.8,
    target_sampling: str = "random",
    fps_candidate_points: int = 50000,
    fit_steps: int = 500,
    target_points: int = 4096,
    stage_max_offsets: tuple[float, ...] = (0.3, 0.15, 0.08, 0.04),
    lr: float = 0.012,
    sdf_clip_mm: float = 16.0,
    voxelize_margin_voxels: int = 3,
    voxelize_slab_depth: int = 16,
    voxelize_backend: str = "pyvista",
) -> dict[str, Path]:
    case_cache = cache_dir / case_id
    mesh_path = case_cache / "mesh_voxelized.nii.gz"
    sdf_path = case_cache / "mesh_sdf.nii.gz"
    metrics_path = case_cache / "mesh_fit_metrics.json"
    if mesh_path.exists() and sdf_path.exists() and not force_rebuild:
        return {"mesh_mask": mesh_path, "mesh_sdf": sdf_path, "metrics": metrics_path}
    if cache_mode == "require_existing":
        raise RuntimeError(f"Missing mesh cache for {case_id}: {case_cache}")

    case_cache.mkdir(parents=True, exist_ok=True)
    pred_path = pred_mask_dir / f"{case_id}.nii.gz"
    mask_zyx, reference_image, geometry = load_mask(pred_path)
    target_physical, target_faces, target_zyx = extract_surface_from_mask(mask_zyx, geometry)
    norm = make_bbox_normalization(target_physical)
    target_norm = norm.to_norm(target_physical)
    template = load_template(template_path, radius=template_radius)
    align_cfg, coarse_cfg, detail_cfg, decoder_cfg = default_mesh_fit_configs(
        fit_steps=fit_steps,
        target_points=target_points,
        stage_max_offsets=stage_max_offsets,
        lr=lr,
    )
    stage_results, final_metrics = fit_mesh_to_target(
        template_vertices_norm=template.vertices,
        faces=template.faces,
        edge_index=template.edge_index,
        target_points_norm=target_norm,
        align_cfg=align_cfg,
        coarse_cfg=coarse_cfg,
        detail_cfg=detail_cfg,
        sampling_mode=target_sampling,
        fps_candidate_points=fps_candidate_points,
        seed=42,
        device=device,
        decoder_cfg=decoder_cfg,
        optimization_loop="lightning",
        enable_progress_bar=False,
        status_fn=lambda _msg: None,
        tb_log_fn=None,
    )
    final_vertices_physical = norm.to_physical(stage_results[-1].vertices_norm)
    mesh_mask = voxelize_mesh_to_mask(
        final_vertices_physical,
        template.faces,
        geometry,
        margin_voxels=voxelize_margin_voxels,
        slab_depth=voxelize_slab_depth,
        backend=voxelize_backend,
    )
    mesh_sdf = mesh_signed_distance(mesh_mask, geometry.spacing_xyz, clip_mm=sdf_clip_mm)
    save_mask_like(mesh_mask, reference_image, mesh_path)
    save_float_like(mesh_sdf, reference_image, sdf_path)
    save_npz(
        case_cache / "fitted_mesh_detail.npz",
        vertices_physical_xyz=final_vertices_physical,
        vertices_norm_xyz=stage_results[-1].vertices_norm,
        faces=template.faces,
        edge_index=template.edge_index,
        target_points_physical_xyz=target_physical,
        target_faces=target_faces,
        target_vertices_pre_zyx=target_zyx,
    )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            _jsonable(
                {
                    "case_id": case_id,
                    "template": template_path,
                    "final": final_metrics,
                    "decoder_config": asdict(decoder_cfg),
                    "detail_config": asdict(detail_cfg),
                    "voxelize_backend": voxelize_backend,
                }
            ),
            f,
            indent=2,
        )
    return {"mesh_mask": mesh_path, "mesh_sdf": sdf_path, "metrics": metrics_path}
