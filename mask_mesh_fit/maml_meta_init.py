from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from pathlib import Path

import torch
from chamferdist import ChamferDistance

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - old torch fallback
    from torch.nn.utils.stateless import functional_call

from .losses import MeshTensors
from .optimize import _FourStageDecoderFitModule, _weighted_deform_loss
from .fomaml_meta_init import (
    TOPCOW_FOMAML_OUTPUT_DIR,
    build_module,
    load_meta_into_module,
    load_target_norm,
    sample_support_query,
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


TOPCOW_MAML_OUTPUT_DIR = TOPCOW_FOMAML_OUTPUT_DIR.parent / "MAML_meta_init_torus10k_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a TOPCOW torus mesh_fit_state.pt initialization with true "
            "second-order MAML. This is intended as a small smoke experiment; "
            "large inner-step counts are very memory intensive."
        )
    )
    parser.add_argument("--mask-dir", type=Path, default=TOPCOW_MASK_DIR)
    parser.add_argument("--template", type=Path, default=TOPCOW_TEMPLATE)
    parser.add_argument("--solved-state-glob", default=TOPCOW_SOLVED_STATE_GLOB)
    parser.add_argument("--output-dir", type=Path, default=TOPCOW_MAML_OUTPUT_DIR)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=25)
    parser.add_argument("--max-train-cases", type=int, default=5)
    parser.add_argument("--max-val-cases", type=int, default=5)
    parser.add_argument("--meta-epochs", type=int, default=1)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--support-points", type=int, default=1024)
    parser.add_argument("--query-points", type=int, default=1024)
    parser.add_argument("--inner-lr", type=float, default=0.006)
    parser.add_argument("--meta-lr-start", type=float, default=1e-4)
    parser.add_argument("--meta-lr-end", type=float, default=1e-4)
    parser.add_argument("--meta-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5252)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-meta-val", action="store_true")
    return parser.parse_args()


def linear_meta_lr(start: float, end: float, index: int, total: int) -> float:
    if total <= 1:
        return float(start)
    alpha = float(index) / float(total - 1)
    return float(start) * (1.0 - alpha) + float(end) * alpha


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
            "inner_grad_norm",
            "meta_grad_norm",
            "grad_scale",
            "loaded_tensors",
            "seconds",
        ],
    )
    writer.writeheader()
    return writer, file_obj


def named_trainable_meta_parameters(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    params = {}
    for key, value in module.named_parameters():
        if key.startswith(ALLOWED_PREFIXES):
            params[key] = value
    if not params:
        raise RuntimeError("No trainable MAML meta-parameters found")
    return params


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def functional_forward_stages(
    module: _FourStageDecoderFitModule,
    fast_state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    aligned = module.aligned_vertices()
    cur = aligned.unsqueeze(0)
    latent = fast_state["latent0"]
    stage_vertices: list[torch.Tensor] = []
    stage_max_offsets = fast_state.get("decoder.stage_max_offsets", module.decoder.stage_max_offsets)

    cached_edge_index = module.decoder._cached_undirected_edge_index(module.edge_index, cur.device)
    for stage_idx, local in enumerate(module.local_features):
        local_key = f"local_features.{stage_idx}"
        local_tensor = fast_state[local_key]
        stage_prefix = f"decoder.stages_refine.{stage_idx}."
        stage_state = _strip_prefix(fast_state, stage_prefix)
        stage_latent, raw_offsets = functional_call(
            module.decoder.stages_refine[stage_idx],
            stage_state,
            (latent, local_tensor, cur, cached_edge_index),
        )
        offsets = torch.tanh(raw_offsets) * stage_max_offsets[stage_idx]
        cur = cur + offsets
        latent = stage_latent
        stage_vertices.append(cur.squeeze(0))

    return aligned, stage_vertices


def functional_fit_loss(
    module: _FourStageDecoderFitModule,
    fast_state: dict[str, torch.Tensor],
    target: torch.Tensor,
    chamfer: ChamferDistance,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    aligned, stage_vertices = functional_forward_stages(module, fast_state)
    final_vertices = stage_vertices[-1]
    mesh = MeshTensors(faces=module.faces, normal_pairs=module.normal_pairs)
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


def tensor_global_norm(values: list[torch.Tensor | None]) -> float:
    total = 0.0
    for value in values:
        if value is None:
            continue
        total += float(torch.sum(value.detach() ** 2).cpu())
    return math.sqrt(total)


def apply_maml_update(
    meta_state: dict[str, torch.Tensor],
    keys: list[str],
    grads: list[torch.Tensor | None],
    *,
    meta_lr: float,
    grad_clip_norm: float,
) -> tuple[int, float, float]:
    grad_norm = tensor_global_norm(grads)
    scale = 1.0
    if grad_clip_norm > 0 and grad_norm > grad_clip_norm:
        scale = float(grad_clip_norm) / max(grad_norm, 1e-12)

    updated = 0
    for key, grad in zip(keys, grads):
        if grad is None or key not in meta_state:
            continue
        update = grad.detach().cpu().to(dtype=meta_state[key].dtype) * scale
        meta_state[key].add_(update, alpha=-float(meta_lr))
        updated += 1
    return updated, grad_norm, scale


def adapt_and_query_maml(
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
) -> tuple[list[str], list[torch.Tensor | None], dict[str, float]]:
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
    trainable = named_trainable_meta_parameters(module)
    trainable_keys = list(trainable)
    fast_state: dict[str, torch.Tensor] = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in meta_state.items()
        if key.startswith(ALLOWED_PREFIXES)
    }
    for key, value in trainable.items():
        fast_state[key] = value

    chamfer = ChamferDistance()
    support_loss_value = math.nan
    support_chamfer_value = math.nan
    inner_grad_norm = math.nan

    for _step in range(max(1, int(inner_steps))):
        support_loss, support_parts = functional_fit_loss(module, fast_state, support, chamfer)
        grads = torch.autograd.grad(
            support_loss,
            [fast_state[key] for key in trainable_keys],
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        inner_grad_norm = tensor_global_norm(list(grads))
        for key, grad in zip(trainable_keys, grads):
            if grad is not None:
                fast_state[key] = fast_state[key] - float(inner_lr) * grad
        support_loss_value = float(support_loss.detach().cpu())
        support_chamfer = support_parts.get("final_chamfer")
        if support_chamfer is not None:
            support_chamfer_value = float(support_chamfer.detach().cpu())

    query_loss, query_parts = functional_fit_loss(module, fast_state, query, chamfer)
    meta_grads = torch.autograd.grad(
        query_loss,
        [trainable[key] for key in trainable_keys],
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    query_chamfer = query_parts.get("final_chamfer")
    metrics = {
        "support_loss": support_loss_value,
        "support_chamfer": support_chamfer_value,
        "query_loss": float(query_loss.detach().cpu()),
        "query_chamfer": float(query_chamfer.detach().cpu()) if query_chamfer is not None else math.nan,
        "inner_grad_norm": inner_grad_norm,
        "loaded_tensors": float(loaded),
    }
    del module, fast_state, support, query, query_loss
    return trainable_keys, list(meta_grads), metrics


def save_current(
    path: Path,
    state: dict[str, torch.Tensor],
    *,
    decoder_config: dict | None,
    metadata: dict,
) -> None:
    save_meta_checkpoint(path, state, decoder_config=decoder_config, metadata=metadata)


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
    if args.train_count + args.val_count > len(all_cases):
        raise RuntimeError(
            f"Requested train_count+val_count={args.train_count + args.val_count} "
            f"but only found {len(all_cases)} masks"
        )
    train_cases = all_cases[: args.train_count]
    val_cases = all_cases[args.train_count : args.train_count + args.val_count]
    if args.max_train_cases > 0:
        train_cases = train_cases[: args.max_train_cases]
    if args.max_val_cases > 0:
        val_cases = val_cases[: args.max_val_cases]
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
                keys, grads, metrics = adapt_and_query_maml(
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
                updated, meta_grad_norm, grad_scale = apply_maml_update(
                    meta_state,
                    keys,
                    grads,
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
                        "inner_grad_norm": metrics["inner_grad_norm"],
                        "meta_grad_norm": meta_grad_norm,
                        "grad_scale": grad_scale,
                        "loaded_tensors": metrics["loaded_tensors"],
                        "seconds": seconds,
                    }
                )
                history_file.flush()
                print(
                    f"[epoch {epoch}/{args.meta_epochs}] {case}: "
                    f"query_chamfer={metrics['query_chamfer']:.6g} "
                    f"meta_grad_norm={meta_grad_norm:.4g} scale={grad_scale:.4g} "
                    f"seconds={seconds:.1f}"
                )
                del grads
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            epoch_path = args.output_dir / f"meta_init_epoch{epoch:03d}.pt"
            save_current(
                epoch_path,
                meta_state,
                decoder_config=decoder_config,
                metadata={
                    "algorithm": "maml",
                    "epoch": epoch,
                    "inner_update": "sgd",
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
                "algorithm": "maml",
                "inner_update": "sgd",
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
                keys, grads, metrics = adapt_and_query_maml(
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
                del keys, grads
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
                        "inner_grad_norm": metrics["inner_grad_norm"],
                        "meta_grad_norm": "",
                        "grad_scale": "",
                        "loaded_tensors": metrics["loaded_tensors"],
                        "seconds": time.time() - start,
                    }
                )
                history_file.flush()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        print(f"Final MAML meta-init: {final_path}")
        print(f"History: {history_path}")
    finally:
        history_file.close()


if __name__ == "__main__":
    main()
