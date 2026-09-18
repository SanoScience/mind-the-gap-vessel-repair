from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from chamferdist import ChamferDistance

from .geometry import adjacent_face_pairs, load_template, sample_points
from .io_utils import extract_surface_from_mask, load_mask, make_bbox_normalization
from .losses import MeshTensors
from .optimize import (
    DecoderConfig,
    StageConfig,
    _FourStageDecoderFitModule,
    _weighted_deform_loss,
)
from .reptile_meta_init import (
    ALLOWED_PREFIXES,
    TOPCOW_MASK_DIR,
    TOPCOW_SOLVED_STATE_GLOB,
    TOPCOW_TEMPLATE,
    average_states,
    checkpoint_decoder_config,
    discover_cases,
    discover_solved_states,
    load_meta_state,
    save_meta_checkpoint,
    write_lines,
)


TOPCOW_FOMAML_OUTPUT_DIR = Path(
    "/home/gniewosz/segmentation/frameworks/v24/runs/TopCow/FOMAML_meta_init_torus10k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a TOPCOW torus mesh_fit_state.pt initialization with first-order MAML. "
            "Each task adapts on support surface points and updates the shared init with "
            "the adapted query loss gradient."
        )
    )
    parser.add_argument("--mask-dir", type=Path, default=TOPCOW_MASK_DIR)
    parser.add_argument("--template", type=Path, default=TOPCOW_TEMPLATE)
    parser.add_argument("--solved-state-glob", default=TOPCOW_SOLVED_STATE_GLOB)
    parser.add_argument("--output-dir", type=Path, default=TOPCOW_FOMAML_OUTPUT_DIR)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=25)
    parser.add_argument(
        "--train-cases-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one training case id per line. "
            "When set, overrides --train-count slicing."
        ),
    )
    parser.add_argument(
        "--val-cases-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one validation/test case id per line. "
            "When set, overrides --val-count slicing."
        ),
    )
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-val-cases", type=int, default=0)
    parser.add_argument("--meta-epochs", type=int, default=3)
    parser.add_argument("--inner-steps", type=int, default=100)
    parser.add_argument("--support-points", type=int, default=4096)
    parser.add_argument("--query-points", type=int, default=4096)
    parser.add_argument("--inner-lr", type=float, default=0.006)
    parser.add_argument("--meta-lr-start", type=float, default=1e-3)
    parser.add_argument("--meta-lr-end", type=float, default=3e-4)
    parser.add_argument("--meta-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-meta-val", action="store_true")
    return parser.parse_args()


def normalize_case_id(value: str) -> str:
    case = value.strip()
    if case.endswith(".nii.gz"):
        case = case[: -len(".nii.gz")]
    return case


def read_case_file(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"Case-list file does not exist: {path}")
    cases = []
    for line in path.read_text().splitlines():
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        cases.append(normalize_case_id(value))
    if not cases:
        raise RuntimeError(f"Case-list file is empty: {path}")
    duplicates = sorted({case for case in cases if cases.count(case) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate cases in {path}: {duplicates[:10]}")
    return cases


def select_train_val_cases(args: argparse.Namespace, all_cases: list[str]) -> tuple[list[str], list[str]]:
    available = set(all_cases)
    if args.train_cases_file is not None:
        train_cases = read_case_file(args.train_cases_file)
    else:
        train_cases = all_cases[: args.train_count]

    if args.val_cases_file is not None:
        val_cases = read_case_file(args.val_cases_file)
    else:
        if args.train_cases_file is not None:
            remaining = [case for case in all_cases if case not in set(train_cases)]
            val_cases = remaining[: args.val_count]
        else:
            val_cases = all_cases[args.train_count : args.train_count + args.val_count]

    missing_train = [case for case in train_cases if case not in available]
    missing_val = [case for case in val_cases if case not in available]
    if missing_train:
        raise RuntimeError(f"Train cases missing masks: {missing_train[:10]}")
    if missing_val:
        raise RuntimeError(f"Validation cases missing masks: {missing_val[:10]}")

    overlap = sorted(set(train_cases) & set(val_cases))
    if overlap:
        raise RuntimeError(f"Train/validation case overlap is not allowed: {overlap[:10]}")

    if args.train_cases_file is None and args.val_cases_file is None:
        if args.train_count + args.val_count > len(all_cases):
            raise RuntimeError(
                f"Requested train_count+val_count={args.train_count + args.val_count} "
                f"but only found {len(all_cases)} masks"
            )

    if args.max_train_cases > 0:
        train_cases = train_cases[: args.max_train_cases]
    if args.max_val_cases > 0:
        val_cases = val_cases[: args.max_val_cases]
    return train_cases, val_cases


def linear_meta_lr(start: float, end: float, index: int, total: int) -> float:
    if total <= 1:
        return float(start)
    alpha = float(index) / float(total - 1)
    return float(start) * (1.0 - alpha) + float(end) * alpha


def load_target_norm(mask_dir: Path, case: str) -> np.ndarray:
    mask_path = mask_dir / f"{case}.nii.gz"
    if not mask_path.exists():
        raise RuntimeError(f"Missing mask: {mask_path}")
    mask_zyx, _reference_image, geometry = load_mask(mask_path)
    target_physical, _target_faces, _target_zyx = extract_surface_from_mask(mask_zyx, geometry)
    norm = make_bbox_normalization(target_physical)
    return norm.to_norm(target_physical).astype(np.float32, copy=False)


def sample_support_query(
    target_norm: np.ndarray,
    *,
    support_points: int,
    query_points: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_np = sample_points(
        target_norm,
        support_points,
        "random",
        seed,
        device,
        fps_candidate_points=50000,
    )
    query_np = sample_points(
        target_norm,
        query_points,
        "random",
        seed + 100_000,
        device,
        fps_candidate_points=50000,
    )
    support = torch.as_tensor(support_np, dtype=torch.float32, device=device)
    query = torch.as_tensor(query_np, dtype=torch.float32, device=device)
    return support, query


def build_module(
    *,
    template_path: Path,
    target_norm: np.ndarray,
    support: torch.Tensor,
    device: torch.device,
    inner_lr: float,
) -> _FourStageDecoderFitModule:
    template = load_template(template_path, radius=0.8)
    base = torch.as_tensor(template.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(template.faces.astype(np.int64, copy=False), dtype=torch.long, device=device)
    normal_pairs = torch.as_tensor(adjacent_face_pairs(template.faces), dtype=torch.long, device=device)
    edge_index = torch.as_tensor(template.edge_index.astype(np.int64, copy=False), dtype=torch.long, device=device)
    align_cfg = StageConfig(
        name="align",
        steps=0,
        lr=0.02,
        target_points=int(support.shape[0]),
        lambda_chamfer=0.10,
        lambda_bbox=1.0,
        log_every=0,
    )
    mesh_cfg = StageConfig(
        name="detail",
        steps=0,
        lr=float(inner_lr),
        target_points=int(support.shape[0]),
        lambda_chamfer=1.0,
        lambda_edge=0.000625,
        lambda_laplacian=0.025,
        lambda_normal=0.00025,
        lambda_face_area_var=2.5e-05,
        log_every=0,
    )
    decoder_cfg = DecoderConfig(
        deformation_mode="decoder",
        stages=4,
        latent_dim=128,
        local_feature_dim=128,
        hidden_dim=128,
        num_blocks=3,
        graph_layer="gcn",
        edge_features="none",
        stage_max_offsets=(0.35, 0.20, 0.10, 0.05),
        stage_loss_weight=0.10,
        fit_steps=0,
        train_alignment=False,
    )
    return _FourStageDecoderFitModule(
        base=base,
        target_align=support,
        target_fit=support,
        target_all=target_norm,
        template_radius=0.8,
        faces=faces,
        normal_pairs=normal_pairs,
        edge_index=edge_index,
        align_cfg=align_cfg,
        mesh_cfg=mesh_cfg,
        decoder_cfg=decoder_cfg,
        lr=float(inner_lr),
        tb_log_fn=None,
    ).to(device)


def load_meta_into_module(module: torch.nn.Module, meta_state: dict[str, torch.Tensor]) -> int:
    current = module.state_dict()
    next_state = dict(current)
    loaded = 0
    for key, value in meta_state.items():
        if not key.startswith(ALLOWED_PREFIXES):
            continue
        if key not in current:
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            continue
        next_state[key] = value.to(device=current[key].device, dtype=current[key].dtype)
        loaded += 1
    module.load_state_dict(next_state, strict=True)
    if loaded == 0:
        raise RuntimeError("Loaded zero meta tensors into FOMAML module")
    return loaded


def fit_loss(
    module: _FourStageDecoderFitModule,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    aligned, stage_vertices = module.forward_stages()
    final_vertices = stage_vertices[-1]
    mesh = MeshTensors(faces=module.faces, normal_pairs=module.normal_pairs)
    chamfer = ChamferDistance()
    final_loss, final_parts = _weighted_deform_loss(
        final_vertices,
        target,
        mesh,
        module.mesh_cfg,
        chamfer=chamfer,
        num_pred_samples=int(target.shape[0]),
        reference=aligned.detach(),
    )
    stage_loss = final_loss.new_zeros(())
    if module.decoder_cfg.stage_loss_weight > 0 and len(stage_vertices) > 1:
        stage_losses = []
        for vertices in stage_vertices[:-1]:
            s_loss, _s_parts = _weighted_deform_loss(
                vertices,
                target,
                mesh,
                module.mesh_cfg,
                chamfer=chamfer,
                num_pred_samples=int(target.shape[0]),
                reference=aligned.detach(),
            )
            stage_losses.append(s_loss)
        stage_loss = torch.stack(stage_losses).mean()
    total = final_loss + module.decoder_cfg.stage_loss_weight * stage_loss
    parts = {f"final_{key}": value for key, value in final_parts.items()}
    parts["loss"] = total
    parts["stage_loss"] = stage_loss
    return total, parts


def named_meta_parameters(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    params = {}
    for key, value in module.named_parameters():
        if key.startswith(ALLOWED_PREFIXES):
            params[key] = value
    return params


def global_grad_norm(params: dict[str, torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params.values():
        if param.grad is None:
            continue
        total += float(torch.sum(param.grad.detach() ** 2).cpu())
    return math.sqrt(total)


def apply_fomaml_update(
    meta_state: dict[str, torch.Tensor],
    params: dict[str, torch.nn.Parameter],
    *,
    meta_lr: float,
    grad_clip_norm: float,
) -> tuple[int, float, float]:
    grad_norm = global_grad_norm(params)
    scale = 1.0
    if grad_clip_norm > 0 and grad_norm > grad_clip_norm:
        scale = float(grad_clip_norm) / max(grad_norm, 1e-12)
    updated = 0
    for key, param in params.items():
        if param.grad is None or key not in meta_state:
            continue
        grad = param.grad.detach().cpu().to(dtype=meta_state[key].dtype) * scale
        meta_state[key].add_(grad, alpha=-float(meta_lr))
        updated += 1
    return updated, grad_norm, scale


def save_current(
    path: Path,
    state: dict[str, torch.Tensor],
    *,
    decoder_config: dict | None,
    metadata: dict,
) -> None:
    save_meta_checkpoint(path, state, decoder_config=decoder_config, metadata=metadata)


def csv_writer(path: Path) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_obj = path.open("w", newline="")
    writer = csv.DictWriter(
        file_obj,
        fieldnames=[
            "phase",
            "epoch",
            "case",
            "meta_lr",
            "support_loss",
            "support_chamfer",
            "query_loss",
            "query_chamfer",
            "updated_tensors",
            "grad_norm",
            "grad_scale",
            "loaded_tensors",
            "seconds",
        ],
    )
    writer.writeheader()
    return writer, file_obj


def adapt_and_query(
    *,
    case: str,
    meta_state: dict[str, torch.Tensor],
    mask_dir: Path,
    template: Path,
    device: torch.device,
    support_points: int,
    query_points: int,
    seed: int,
    inner_steps: int,
    inner_lr: float,
) -> tuple[_FourStageDecoderFitModule, dict[str, float]]:
    target_norm = load_target_norm(mask_dir, case)
    support, query = sample_support_query(
        target_norm,
        support_points=support_points,
        query_points=query_points,
        seed=seed,
        device=device,
    )
    module = build_module(
        template_path=template,
        target_norm=target_norm,
        support=support,
        device=device,
        inner_lr=inner_lr,
    )
    loaded = load_meta_into_module(module, meta_state)
    optimizer = torch.optim.Adam(module.parameters(), lr=float(inner_lr))

    support_loss_value = math.nan
    support_chamfer_value = math.nan
    module.train()
    for _step in range(max(1, int(inner_steps))):
        optimizer.zero_grad(set_to_none=True)
        support_loss, support_parts = fit_loss(module, support)
        support_loss.backward()
        optimizer.step()
        support_loss_value = float(support_loss.detach().cpu())
        support_chamfer = support_parts.get("final_chamfer")
        if support_chamfer is not None:
            support_chamfer_value = float(support_chamfer.detach().cpu())

    optimizer.zero_grad(set_to_none=True)
    query_loss, query_parts = fit_loss(module, query)
    query_loss.backward()
    query_chamfer = query_parts.get("final_chamfer")
    metrics = {
        "support_loss": support_loss_value,
        "support_chamfer": support_chamfer_value,
        "query_loss": float(query_loss.detach().cpu()),
        "query_chamfer": float(query_chamfer.detach().cpu()) if query_chamfer is not None else math.nan,
        "loaded_tensors": float(loaded),
    }
    return module, metrics


def main() -> None:
    args = parse_args()
    if not args.mask_dir.is_dir():
        raise RuntimeError(f"Mask directory does not exist: {args.mask_dir}")
    if not args.template.is_file():
        raise RuntimeError(f"Template does not exist: {args.template}")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = discover_cases(args.mask_dir)
    train_cases, val_cases = select_train_val_cases(args, all_cases)
    write_lines(args.output_dir / "meta_train_cases.txt", train_cases)
    write_lines(args.output_dir / "meta_val_cases.txt", val_cases)

    solved_states = discover_solved_states(args.solved_state_glob)
    missing_solved = [case for case in train_cases if case not in solved_states]
    if missing_solved:
        raise RuntimeError(f"Missing solved states for train cases: {missing_solved[:10]}")

    avg_path = args.output_dir / "meta_init_average.pt"
    if not avg_path.exists() or args.overwrite:
        avg_state, decoder_config, averaged_count = average_states(solved_states[case] for case in train_cases)
        save_current(
            avg_path,
            avg_state,
            decoder_config=decoder_config,
            metadata={
                "algorithm": "average_solved_train_states",
                "averaged_count": averaged_count,
                "allowed_prefixes": ALLOWED_PREFIXES,
                "train_cases": train_cases,
                "val_cases": val_cases,
            },
        )
    else:
        avg_state = load_meta_state(avg_path)
        decoder_config = checkpoint_decoder_config(avg_path)

    meta_state = {key: value.detach().cpu().clone() for key, value in avg_state.items()}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    history_path = args.output_dir / "meta_history.csv"
    writer, history_file = csv_writer(history_path)
    try:
        total_updates = max(1, int(args.meta_epochs) * len(train_cases))
        update_index = 0
        for epoch in range(1, int(args.meta_epochs) + 1):
            for case_index, case in enumerate(train_cases):
                start = time.time()
                meta_lr = linear_meta_lr(args.meta_lr_start, args.meta_lr_end, update_index, total_updates)
                seed = int(args.seed) + epoch * 10_000 + case_index
                module, metrics = adapt_and_query(
                    case=case,
                    meta_state=meta_state,
                    mask_dir=args.mask_dir,
                    template=args.template,
                    device=device,
                    support_points=args.support_points,
                    query_points=args.query_points,
                    seed=seed,
                    inner_steps=args.inner_steps,
                    inner_lr=args.inner_lr,
                )
                params = named_meta_parameters(module)
                updated, grad_norm, grad_scale = apply_fomaml_update(
                    meta_state,
                    params,
                    meta_lr=meta_lr,
                    grad_clip_norm=args.meta_grad_clip_norm,
                )
                update_index += 1
                seconds = time.time() - start
                writer.writerow(
                    {
                        "phase": "train",
                        "epoch": epoch,
                        "case": case,
                        "meta_lr": meta_lr,
                        "support_loss": metrics["support_loss"],
                        "support_chamfer": metrics["support_chamfer"],
                        "query_loss": metrics["query_loss"],
                        "query_chamfer": metrics["query_chamfer"],
                        "updated_tensors": updated,
                        "grad_norm": grad_norm,
                        "grad_scale": grad_scale,
                        "loaded_tensors": metrics["loaded_tensors"],
                        "seconds": seconds,
                    }
                )
                history_file.flush()
                print(
                    f"[epoch {epoch}/{args.meta_epochs}] {case}: "
                    f"query_chamfer={metrics['query_chamfer']:.6g} "
                    f"grad_norm={grad_norm:.4g} scale={grad_scale:.4g} "
                    f"seconds={seconds:.1f}"
                )
                del module
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            epoch_path = args.output_dir / f"meta_init_epoch{epoch:03d}.pt"
            save_current(
                epoch_path,
                meta_state,
                decoder_config=decoder_config,
                metadata={
                    "algorithm": "fomaml",
                    "epoch": epoch,
                    "inner_steps": args.inner_steps,
                    "support_points": args.support_points,
                    "query_points": args.query_points,
                    "meta_lr_start": args.meta_lr_start,
                    "meta_lr_end": args.meta_lr_end,
                    "meta_grad_clip_norm": args.meta_grad_clip_norm,
                    "allowed_prefixes": ALLOWED_PREFIXES,
                },
            )

        final_path = args.output_dir / "meta_init_final.pt"
        save_current(
            final_path,
            meta_state,
            decoder_config=decoder_config,
            metadata={
                "algorithm": "fomaml",
                "inner_steps": args.inner_steps,
                "support_points": args.support_points,
                "query_points": args.query_points,
                "meta_lr_start": args.meta_lr_start,
                "meta_lr_end": args.meta_lr_end,
                "meta_grad_clip_norm": args.meta_grad_clip_norm,
                "allowed_prefixes": ALLOWED_PREFIXES,
                "train_cases": train_cases,
                "val_cases": val_cases,
            },
        )

        if args.run_meta_val:
            for case_index, case in enumerate(val_cases):
                start = time.time()
                seed = int(args.seed) + 1_000_000 + case_index
                module, metrics = adapt_and_query(
                    case=case,
                    meta_state=meta_state,
                    mask_dir=args.mask_dir,
                    template=args.template,
                    device=device,
                    support_points=args.support_points,
                    query_points=args.query_points,
                    seed=seed,
                    inner_steps=args.inner_steps,
                    inner_lr=args.inner_lr,
                )
                writer.writerow(
                    {
                        "phase": "val",
                        "epoch": int(args.meta_epochs),
                        "case": case,
                        "meta_lr": "",
                        "support_loss": metrics["support_loss"],
                        "support_chamfer": metrics["support_chamfer"],
                        "query_loss": metrics["query_loss"],
                        "query_chamfer": metrics["query_chamfer"],
                        "updated_tensors": "",
                        "grad_norm": "",
                        "grad_scale": "",
                        "loaded_tensors": metrics["loaded_tensors"],
                        "seconds": time.time() - start,
                    }
                )
                history_file.flush()
                del module
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        print(f"Final FOMAML meta-init: {final_path}")
        print(f"History: {history_path}")
    finally:
        history_file.close()


if __name__ == "__main__":
    main()
