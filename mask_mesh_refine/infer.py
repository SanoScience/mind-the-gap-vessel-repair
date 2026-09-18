from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    import omegaconf

    if not hasattr(omegaconf, "Container"):
        omegaconf.Container = (omegaconf.DictConfig, omegaconf.ListConfig)
except Exception:
    pass

from mask_mesh_fit.io_utils import ImageGeometry, save_float_like, save_mask_like
from mask_mesh_fit.voxelize import (
    repair_mask_with_component_gated_sdf,
    repair_mask_with_mesh,
    repair_mask_with_mesh_sdf_blend,
)

from .model import RefinerLightningModule
from .qa import save_refine_qa
from .utils import (
    ensure_case_mesh_cache,
    list_cases,
    make_refiner_input,
    mask_metrics,
    read_bool_mask,
    read_float_image,
    resolve_device,
)


def _hparam(module: RefinerLightningModule, key: str, default: float) -> float:
    return float(module.hparams.get(key, default))


@torch.no_grad()
def refine_case_arrays(
    checkpoint_path: Path,
    pred_mask_zyx: np.ndarray,
    mesh_mask_zyx: np.ndarray,
    mesh_sdf_zyx: np.ndarray,
    reference_image: Any,
    output_dir: Path,
    output_mask_name: str = "refined_mask.nii.gz",
    output_prob_name: str = "refined_prob.nii.gz",
    device: str = "auto",
    crop_margin_mm: float | None = None,
    sdf_clip_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    torch_device = resolve_device(device)
    module = RefinerLightningModule.load_from_checkpoint(str(checkpoint_path), map_location=torch_device)
    module.to(torch_device)
    module.eval()
    spacing_xyz = np.asarray(reference_image.GetSpacing(), dtype=np.float64)
    crop_margin_mm = _hparam(module, "crop_margin_mm", 16.0) if crop_margin_mm is None else float(crop_margin_mm)
    sdf_clip_mm = _hparam(module, "sdf_clip_mm", 16.0) if sdf_clip_mm is None else float(sdf_clip_mm)
    inputs, crop = make_refiner_input(
        pred_mask_zyx,
        mesh_mask_zyx,
        mesh_sdf_zyx,
        spacing_xyz=spacing_xyz,
        crop_margin_mm=crop_margin_mm,
        sdf_clip_mm=sdf_clip_mm,
    )
    x = torch.from_numpy(inputs).unsqueeze(0).to(device=torch_device, dtype=torch.float32)
    logits = module(x)
    prob_crop = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    prob_full = pred_mask_zyx.astype(np.float32, copy=True)
    prob_full[crop] = prob_crop
    refined = prob_full >= 0.5
    output_dir.mkdir(parents=True, exist_ok=True)
    save_float_like(prob_full, reference_image, output_dir / output_prob_name)
    save_mask_like(refined, reference_image, output_dir / output_mask_name)
    return prob_full, refined


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer learned mask refinement from nnUNet masks and mesh priors.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pred-mask-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-mask-dir", type=Path, default=None)
    parser.add_argument("--case-prefix", default="")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--mesh-cache-mode", choices=["on_demand", "require_existing"], default="on_demand")
    parser.add_argument("--mesh-cache-dir", type=Path, default=None)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-margin-mm", type=float, default=None)
    parser.add_argument("--sdf-clip-mm", type=float, default=None)
    parser.add_argument("--mesh-fit-steps", type=int, default=500)
    parser.add_argument("--mesh-fit-target-points", type=int, default=4096)
    parser.add_argument("--mesh-fit-lr", type=float, default=0.012)
    parser.add_argument("--mesh-stage-max-offsets", type=float, nargs="+", default=[0.3, 0.15, 0.08, 0.04])
    parser.add_argument("--template-radius", type=float, default=0.8)
    parser.add_argument("--target-sampling", choices=["random", "fps"], default="random")
    parser.add_argument("--fps-candidate-points", type=int, default=50000)
    parser.add_argument("--voxelize-backend", choices=["pyvista", "multigeomed"], default="pyvista")
    parser.add_argument("--repair-dilation-mm", type=float, default=8.0)
    parser.add_argument("--repair-region-method", choices=["edt_crop", "edt", "binary_dilation"], default="edt_crop")
    parser.add_argument("--sdf-repair-threshold-mm", type=float, default=1.5)
    parser.add_argument("--sdf-repair-connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument(
        "--component-sdf-touch-original-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.mesh_cache_dir if args.mesh_cache_dir is not None else args.output_dir / "cache"
    device = resolve_device(args.device)
    cases = list(args.cases) if args.cases else list_cases(args.pred_mask_dir, args.gt_mask_dir, args.case_prefix)
    if not cases:
        raise RuntimeError("No cases found for inference")

    all_metrics: dict[str, Any] = {"cases": {}, "mean": {}}
    for case_id in cases:
        print(f"Infer learned refiner: {case_id}")
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
            sdf_clip_mm=args.sdf_clip_mm if args.sdf_clip_mm is not None else 16.0,
            voxelize_backend=args.voxelize_backend,
        )
        case_dir = args.output_dir / case_id
        pred_mask, reference_image = read_bool_mask(args.pred_mask_dir / f"{case_id}.nii.gz")
        mesh_mask, _ = read_bool_mask(cache_dir / case_id / "mesh_voxelized.nii.gz")
        mesh_sdf, _ = read_float_image(cache_dir / case_id / "mesh_sdf.nii.gz")
        prob, refined = refine_case_arrays(
            checkpoint_path=args.checkpoint,
            pred_mask_zyx=pred_mask,
            mesh_mask_zyx=mesh_mask,
            mesh_sdf_zyx=mesh_sdf,
            reference_image=reference_image,
            output_dir=case_dir,
            device=str(device),
            crop_margin_mm=args.crop_margin_mm,
            sdf_clip_mm=args.sdf_clip_mm,
        )
        del prob
        save_mask_like(pred_mask, reference_image, case_dir / "nnunet_mask.nii.gz")
        save_mask_like(mesh_mask, reference_image, case_dir / "mesh_voxelized.nii.gz")
        save_float_like(mesh_sdf, reference_image, case_dir / "mesh_sdf.nii.gz")

        metrics: dict[str, Any] = {"case_id": case_id}
        gt_mask = None
        if args.gt_mask_dir is not None and (args.gt_mask_dir / f"{case_id}.nii.gz").exists():
            gt_mask, _ = read_bool_mask(args.gt_mask_dir / f"{case_id}.nii.gz")
            voxel_repaired, local_region = repair_mask_with_mesh(
                pred_mask,
                mesh_mask,
                ImageGeometry.from_image(reference_image, pred_mask.shape),
                dilation_mm=args.repair_dilation_mm,
                method=args.repair_region_method,
            )
            sdf_repaired, sdf_candidate, _sdf_local, sdf_kept = repair_mask_with_mesh_sdf_blend(
                pred_mask,
                mesh_mask,
                mesh_sdf,
                ImageGeometry.from_image(reference_image, pred_mask.shape),
                dilation_mm=args.repair_dilation_mm,
                sdf_threshold_mm=args.sdf_repair_threshold_mm,
                local_region_method=args.repair_region_method,
                local_region_zyx=local_region,
                connectivity=args.sdf_repair_connectivity,
            )
            component_repaired, component_candidate, _component_local, component_kept = (
                repair_mask_with_component_gated_sdf(
                    pred_mask,
                    mesh_mask,
                    mesh_sdf,
                    ImageGeometry.from_image(reference_image, pred_mask.shape),
                    dilation_mm=args.repair_dilation_mm,
                    sdf_threshold_mm=args.sdf_repair_threshold_mm,
                    local_region_method=args.repair_region_method,
                    local_region_zyx=local_region,
                    connectivity=args.sdf_repair_connectivity,
                    touch_original_mask=args.component_sdf_touch_original_mask,
                )
            )
            save_mask_like(voxel_repaired, reference_image, case_dir / "repaired_mask_voxelized.nii.gz")
            save_mask_like(sdf_repaired, reference_image, case_dir / "repaired_mask_sdf_cc.nii.gz")
            save_mask_like(sdf_candidate, reference_image, case_dir / "sdf_repair_candidate.nii.gz")
            save_mask_like(sdf_kept, reference_image, case_dir / "sdf_repair_kept_candidate.nii.gz")
            save_mask_like(component_repaired, reference_image, case_dir / "repaired_mask_component_sdf.nii.gz")
            save_mask_like(component_candidate, reference_image, case_dir / "component_sdf_candidate.nii.gz")
            save_mask_like(component_kept, reference_image, case_dir / "component_sdf_kept_candidate.nii.gz")
            metrics["nnunet"] = mask_metrics(pred_mask, gt_mask)
            metrics["voxelized_repair"] = mask_metrics(voxel_repaired, gt_mask)
            metrics["sdf_cc_repair"] = mask_metrics(sdf_repaired, gt_mask)
            metrics["component_sdf_repair"] = mask_metrics(component_repaired, gt_mask)
            metrics["learned_refiner"] = mask_metrics(refined, gt_mask)
            metrics["learned_minus_nnunet_dice"] = (
                metrics["learned_refiner"]["dice"] - metrics["nnunet"]["dice"]
            )
            metrics["voxelized_minus_nnunet_dice"] = (
                metrics["voxelized_repair"]["dice"] - metrics["nnunet"]["dice"]
            )
            metrics["sdf_cc_minus_nnunet_dice"] = (
                metrics["sdf_cc_repair"]["dice"] - metrics["nnunet"]["dice"]
            )
            metrics["component_sdf_minus_nnunet_dice"] = (
                metrics["component_sdf_repair"]["dice"] - metrics["nnunet"]["dice"]
            )
        save_refine_qa(
            pred_mask=pred_mask,
            mesh_mask=mesh_mask,
            refined_mask=refined,
            gt_mask=gt_mask,
            output_png=case_dir / "qa_overlay.png",
            output_pdf=case_dir / "qa_overlay.pdf",
            title=case_id,
        )
        with (case_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(metrics), f, indent=2)
        all_metrics["cases"][case_id] = metrics

    if all_metrics["cases"] and args.gt_mask_dir is not None:
        for group in ("nnunet", "voxelized_repair", "sdf_cc_repair", "component_sdf_repair", "learned_refiner"):
            values = [
                case_metrics[group]["dice"]
                for case_metrics in all_metrics["cases"].values()
                if group in case_metrics
            ]
            if values:
                all_metrics["mean"][f"{group}_dice"] = float(np.mean(values))
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(all_metrics), f, indent=2)
    print(f"Saved inference metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
