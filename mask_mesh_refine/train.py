from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
try:
    import omegaconf

    if not hasattr(omegaconf, "Container"):
        omegaconf.Container = (omegaconf.DictConfig, omegaconf.ListConfig)
except Exception:
    pass

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .data import MeshRefineDataset
from .model import RefinerLightningModule
from .utils import ensure_case_mesh_cache, resolve_device, split_cases


class HistoryCallback(Callback):
    def __init__(self) -> None:
        self.records: list[dict[str, float | int]] = []

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        record: dict[str, float | int] = {"epoch": int(trainer.current_epoch)}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                record[str(key)] = float(value.detach().cpu().item())
        self.records.append(record)


class SafeTensorBoardLogger(TensorBoardLogger):
    def log_hyperparams(self, params: Any, metrics: Any | None = None) -> None:
        del params, metrics
        return

    def save(self) -> None:
        self.experiment.flush()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned mask refiner from nnUNet masks and mesh priors.")
    parser.add_argument("--pred-mask-dir", type=Path, required=True)
    parser.add_argument("--gt-mask-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-prefix", default="A")
    parser.add_argument("--split-mode", choices=["fixed_a_16_4", "all_train"], default="fixed_a_16_4")
    parser.add_argument("--train-cases", nargs="*", default=None)
    parser.add_argument("--val-cases", nargs="*", default=None)
    parser.add_argument("--mesh-cache-mode", choices=["on_demand", "require_existing"], default="on_demand")
    parser.add_argument("--mesh-cache-dir", type=Path, default=None)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lambda-bce", type=float, default=0.5)
    parser.add_argument("--lambda-dice", type=float, default=1.0)
    parser.add_argument("--crop-margin-mm", type=float, default=16.0)
    parser.add_argument("--sdf-clip-mm", type=float, default=16.0)
    parser.add_argument("--mesh-fit-steps", type=int, default=500)
    parser.add_argument("--mesh-fit-target-points", type=int, default=4096)
    parser.add_argument("--mesh-fit-lr", type=float, default=0.012)
    parser.add_argument("--mesh-stage-max-offsets", type=float, nargs="+", default=[0.3, 0.15, 0.08, 0.04])
    parser.add_argument("--template-radius", type=float, default=0.8)
    parser.add_argument("--target-sampling", choices=["random", "fps"], default="random")
    parser.add_argument("--fps-candidate-points", type=int, default=50000)
    parser.add_argument("--voxelize-backend", choices=["pyvista", "multigeomed"], default="pyvista")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise RuntimeError("Use --batch-size 1: cropped full-resolution cases have variable shapes.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.mesh_cache_dir if args.mesh_cache_dir is not None else args.output_dir / "cache"
    device = resolve_device(args.device)
    L.seed_everything(args.seed, workers=True)

    train_cases, val_cases = split_cases(
        args.pred_mask_dir,
        args.gt_mask_dir,
        args.case_prefix,
        args.split_mode,
        train_cases=args.train_cases,
        val_cases=args.val_cases,
    )
    split = {"train": train_cases, "val": val_cases}
    with (args.output_dir / "case_split.json").open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(vars(args)), f, indent=2)

    print(f"Train cases ({len(train_cases)}): {train_cases}")
    print(f"Val cases ({len(val_cases)}): {val_cases}")
    print(f"Mesh cache: {cache_dir}")
    for case_id in train_cases + val_cases:
        print(f"Ensuring mesh cache: {case_id}")
        ensure_case_mesh_cache(
            case_id,
            pred_mask_dir=args.pred_mask_dir,
            template_path=args.template,
            cache_dir=cache_dir,
            device=device,
            cache_mode=args.mesh_cache_mode,
            force_rebuild=args.force_rebuild_cache,
            template_radius=args.template_radius,
            target_sampling=args.target_sampling,
            fps_candidate_points=args.fps_candidate_points,
            fit_steps=args.mesh_fit_steps,
            target_points=args.mesh_fit_target_points,
            stage_max_offsets=tuple(float(x) for x in args.mesh_stage_max_offsets),
            lr=args.mesh_fit_lr,
            sdf_clip_mm=args.sdf_clip_mm,
            voxelize_backend=args.voxelize_backend,
        )

    train_ds = MeshRefineDataset(
        train_cases,
        pred_mask_dir=args.pred_mask_dir,
        gt_mask_dir=args.gt_mask_dir,
        mesh_cache_dir=cache_dir,
        crop_margin_mm=args.crop_margin_mm,
        sdf_clip_mm=args.sdf_clip_mm,
    )
    val_ds = MeshRefineDataset(
        val_cases,
        pred_mask_dir=args.pred_mask_dir,
        gt_mask_dir=args.gt_mask_dir,
        mesh_cache_dir=cache_dir,
        crop_margin_mm=args.crop_margin_mm,
        sdf_clip_mm=args.sdf_clip_mm,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    module = RefinerLightningModule(
        in_channels=3,
        base_channels=args.base_channels,
        lr=args.lr,
        lambda_bce=args.lambda_bce,
        lambda_dice=args.lambda_dice,
        crop_margin_mm=args.crop_margin_mm,
        sdf_clip_mm=args.sdf_clip_mm,
    )
    history_cb = HistoryCallback()
    ckpt_cb = ModelCheckpoint(
        dirpath=args.output_dir / "lightning_ckpts",
        filename="epoch{epoch:04d}-val_dice{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    logger = SafeTensorBoardLogger(save_dir=str(args.output_dir), name="tensorboard", version="")
    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[history_cb, ckpt_cb],
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history_cb.records, f, indent=2)
    print(f"Best checkpoint: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
