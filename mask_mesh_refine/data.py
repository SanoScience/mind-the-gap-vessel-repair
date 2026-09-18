from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import make_refiner_input, read_bool_mask, read_float_image


class MeshRefineDataset(Dataset):
    def __init__(
        self,
        cases: list[str],
        pred_mask_dir: Path,
        gt_mask_dir: Path,
        mesh_cache_dir: Path,
        crop_margin_mm: float,
        sdf_clip_mm: float,
    ) -> None:
        self.cases = list(cases)
        self.pred_mask_dir = Path(pred_mask_dir)
        self.gt_mask_dir = Path(gt_mask_dir)
        self.mesh_cache_dir = Path(mesh_cache_dir)
        self.crop_margin_mm = float(crop_margin_mm)
        self.sdf_clip_mm = float(sdf_clip_mm)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_id = self.cases[index]
        pred_mask, pred_image = read_bool_mask(self.pred_mask_dir / f"{case_id}.nii.gz")
        gt_mask, _gt_image = read_bool_mask(self.gt_mask_dir / f"{case_id}.nii.gz")
        mesh_mask, _mesh_image = read_bool_mask(self.mesh_cache_dir / case_id / "mesh_voxelized.nii.gz")
        mesh_sdf, _sdf_image = read_float_image(self.mesh_cache_dir / case_id / "mesh_sdf.nii.gz")

        spacing_xyz = np.asarray(pred_image.GetSpacing(), dtype=np.float64)
        inputs, crop = make_refiner_input(
            pred_mask,
            mesh_mask,
            mesh_sdf,
            spacing_xyz=spacing_xyz,
            crop_margin_mm=self.crop_margin_mm,
            sdf_clip_mm=self.sdf_clip_mm,
        )
        target = gt_mask[crop].astype(np.float32, copy=False)[None, ...]
        return {
            "case_id": case_id,
            "x": torch.from_numpy(inputs),
            "y": torch.from_numpy(target),
        }
