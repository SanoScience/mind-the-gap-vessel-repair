from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import torch


ALLOWED_PREFIXES = ("decoder.", "latent0", "local_features.")
TOPCOW_MASK_DIR = Path(
    "/home/gniewosz/segmentation/baseline/helios/nnunet/results/"
    "Dataset003_TOPCOW_CT_BINARY/nnUNetTrainer__nnUNetPlans__3d_fullres/oof_predictions"
)
TOPCOW_TEMPLATE = Path(
    "/home/gniewosz/segmentation/Voxel_Mesh/mesh_gt/topcow_torus_template_10k_thin_stagefixed.npz"
)
TOPCOW_SOLVED_STATE_GLOB = (
    "/home/gniewosz/segmentation/frameworks/v24/runs/TopCow/Oof_torus10k/"
    "*_voxelrepair_torus10k/mesh_fit_state.pt"
)
TOPCOW_OUTPUT_DIR = Path(
    "/home/gniewosz/segmentation/frameworks/v24/runs/TopCow/Reptile_meta_init_torus10k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a TOPCOW mesh_fit_state.pt initialization with Reptile-style "
            "meta-updates over short fit_case adaptations."
        )
    )
    parser.add_argument("--mask-dir", type=Path, default=TOPCOW_MASK_DIR)
    parser.add_argument("--template", type=Path, default=TOPCOW_TEMPLATE)
    parser.add_argument("--solved-state-glob", default=TOPCOW_SOLVED_STATE_GLOB)
    parser.add_argument("--output-dir", type=Path, default=TOPCOW_OUTPUT_DIR)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=25)
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-val-cases", type=int, default=0)
    parser.add_argument("--meta-epochs", type=int, default=3)
    parser.add_argument("--inner-steps", type=int, default=500)
    parser.add_argument("--meta-lr-start", type=float, default=0.1)
    parser.add_argument("--meta-lr-end", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-meta-val", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(disable_progress_bar=True)
    parser.add_argument("--enable-progress-bar", dest="disable_progress_bar", action="store_false")
    parser.add_argument("--keep-adapted-outputs", action="store_true")
    return parser.parse_args()


def strip_nii_gz(path: Path) -> str:
    name = path.name
    if not name.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz mask name, got {path}")
    return name[: -len(".nii.gz")]


def discover_cases(mask_dir: Path) -> list[str]:
    cases = [strip_nii_gz(path) for path in sorted(mask_dir.glob("*.nii.gz"))]
    if not cases:
        raise RuntimeError(f"No .nii.gz masks found in {mask_dir}")
    return cases


def case_from_state_path(path: Path) -> str:
    name = path.parent.name
    suffixes = ("_voxelrepair_torus10k", "_voxelrepair")
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def discover_solved_states(pattern: str) -> dict[str, Path]:
    paths = sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:]))
    states: dict[str, Path] = {}
    for path in paths:
        case = case_from_state_path(path)
        if case in states:
            raise RuntimeError(f"Duplicate solved state for {case}: {states[case]} and {path}")
        states[case] = path
    if not states:
        raise RuntimeError(f"No solved mesh_fit_state.pt files matched: {pattern}")
    return states


def checkpoint_state(checkpoint: object, path: Path) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"{path}: expected checkpoint dict or state_dict")
    return state


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{path}: expected checkpoint dict")
    return checkpoint


def allowed_float_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = load_checkpoint(path)
    state = checkpoint_state(checkpoint, path)
    result = {}
    for key, value in state.items():
        if not key.startswith(ALLOWED_PREFIXES):
            continue
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            continue
        result[key] = value.detach().cpu().clone()
    if not result:
        raise RuntimeError(f"{path}: no allowed floating tensors found")
    return result


def checkpoint_decoder_config(path: Path) -> dict | None:
    checkpoint = load_checkpoint(path)
    decoder_config = checkpoint.get("decoder_config")
    return dict(decoder_config) if isinstance(decoder_config, dict) else None


def average_states(paths: Iterable[Path]) -> tuple[dict[str, torch.Tensor], dict | None, int]:
    iterator = iter(paths)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Cannot average zero states") from exc

    avg = allowed_float_state(first)
    decoder_config = checkpoint_decoder_config(first)
    count = 1
    reference_shapes = {key: tuple(value.shape) for key, value in avg.items()}

    for path in iterator:
        state = allowed_float_state(path)
        missing = sorted(set(reference_shapes) - set(state))
        extra = sorted(set(state) - set(reference_shapes))
        bad_shape = [
            key
            for key, shape in reference_shapes.items()
            if key in state and tuple(state[key].shape) != shape
        ]
        if missing or extra or bad_shape:
            raise RuntimeError(
                f"{path}: incompatible state for averaging "
                f"missing={missing[:5]} extra={extra[:5]} bad_shape={bad_shape[:5]}"
            )
        for key, value in state.items():
            avg[key].add_(value)
        count += 1

    for key in avg:
        avg[key].div_(float(count))
    return avg, decoder_config, count


def save_meta_checkpoint(
    path: Path,
    state: dict[str, torch.Tensor],
    *,
    decoder_config: dict | None,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "v24_mask_mesh_fit_state_v1",
            "state_dict": {key: value.detach().cpu().clone() for key, value in state.items()},
            "decoder_config": decoder_config or {},
            "best_step": -1,
            "best_final_chamfer": float("nan"),
            "loaded_from": None,
            "meta": metadata,
        },
        str(path),
    )


def load_meta_state(path: Path) -> dict[str, torch.Tensor]:
    return allowed_float_state(path)


def reptile_update(
    meta_state: dict[str, torch.Tensor],
    adapted_state_path: Path,
    *,
    meta_lr: float,
) -> int:
    adapted = allowed_float_state(adapted_state_path)
    updated = 0
    for key, meta_value in meta_state.items():
        value = adapted.get(key)
        if value is None or tuple(value.shape) != tuple(meta_value.shape):
            continue
        meta_value.add_(value - meta_value, alpha=float(meta_lr))
        updated += 1
    return updated


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values))


def linear_meta_lr(start: float, end: float, index: int, total: int) -> float:
    if total <= 1:
        return float(start)
    alpha = float(index) / float(total - 1)
    return float(start) * (1.0 - alpha) + float(end) * alpha


def run_fit_case(
    *,
    python_bin: str,
    case: str,
    mask_dir: Path,
    template: Path,
    output_dir: Path,
    init_state: Path,
    save_fit_state: Path,
    inner_steps: int,
    device: str,
    disable_progress_bar: bool,
) -> None:
    cmd = [
        python_bin,
        "-m",
        "mask_mesh_fit.fit_case",
        "--case-id",
        case,
        "--mask-dir",
        str(mask_dir),
        "--template",
        str(template),
        "--output-dir",
        str(output_dir),
        "--save-fit-state",
        str(save_fit_state),
        "--init-fit-state",
        str(init_state),
        "--device",
        device,
        "--repair-mode",
        "voxelize",
        "--deformation-mode",
        "decoder",
        "--target-sampling",
        "random",
        "--coarse-target-points",
        "8192",
        "--detail-target-points",
        "8192",
        "--fit-steps",
        str(inner_steps),
        "--detail-lr",
        "0.006",
        "--detail-lambda-chamfer",
        "1.0",
        "--detail-lambda-edge",
        "0.000625",
        "--detail-lambda-laplacian",
        "0.025",
        "--detail-lambda-normal",
        "0.00025",
        "--detail-lambda-face-area-var",
        "2.5e-05",
        "--decoder-stage-max-offsets",
        "0.35",
        "0.20",
        "0.10",
        "0.05",
        "--geometric-repair-method",
        "component_sdf",
        "--sdf-repair-threshold-mm",
        "0.01",
        "--repair-dilation-mm",
        "0.5",
        "--sdf-repair-connectivity",
        "26",
        "--voxelize-backend",
        "multigeomed",
        "--skip-voxelize",
        "--disable-tensorboard",
        "--disable-qa",
    ]
    if disable_progress_bar:
        cmd.append("--disable-progress-bar")
    subprocess.run(cmd, check=True)


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
            "updated_tensors",
            "seconds",
            "meta_checkpoint",
            "adapted_checkpoint",
        ],
    )
    writer.writeheader()
    return writer, file_obj


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
        save_meta_checkpoint(
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

    current_meta_path = args.output_dir / "meta_init_epoch000.pt"
    if not current_meta_path.exists() or args.overwrite:
        save_meta_checkpoint(
            current_meta_path,
            avg_state,
            decoder_config=decoder_config,
            metadata={
                "algorithm": "reptile",
                "source": str(avg_path),
                "epoch": 0,
                "allowed_prefixes": ALLOWED_PREFIXES,
                "inner_steps": args.inner_steps,
            },
        )

    history_path = args.output_dir / "meta_history.csv"
    writer, history_file = csv_writer(history_path)
    try:
        total_updates = max(1, int(args.meta_epochs) * len(train_cases))
        update_index = 0
        for epoch in range(1, int(args.meta_epochs) + 1):
            meta_state = load_meta_state(current_meta_path)
            epoch_dir = args.output_dir / "adaptations" / f"epoch_{epoch:03d}"
            for case in train_cases:
                meta_lr = linear_meta_lr(args.meta_lr_start, args.meta_lr_end, update_index, total_updates)
                adapt_dir = epoch_dir / f"{case}_adapt"
                adapted_state = adapt_dir / "mesh_fit_state.pt"
                start = time.time()
                run_fit_case(
                    python_bin=args.python_bin,
                    case=case,
                    mask_dir=args.mask_dir,
                    template=args.template,
                    output_dir=adapt_dir,
                    init_state=current_meta_path,
                    save_fit_state=adapted_state,
                    inner_steps=args.inner_steps,
                    device=args.device,
                    disable_progress_bar=args.disable_progress_bar,
                )
                updated_tensors = reptile_update(meta_state, adapted_state, meta_lr=meta_lr)
                seconds = time.time() - start
                update_index += 1
                save_meta_checkpoint(
                    current_meta_path,
                    meta_state,
                    decoder_config=decoder_config,
                    metadata={
                        "algorithm": "reptile",
                        "epoch": epoch,
                        "last_case": case,
                        "last_meta_lr": meta_lr,
                        "update_index": update_index,
                        "total_updates": total_updates,
                        "allowed_prefixes": ALLOWED_PREFIXES,
                        "inner_steps": args.inner_steps,
                    },
                )
                writer.writerow(
                    {
                        "phase": "train",
                        "epoch": epoch,
                        "case": case,
                        "meta_lr": meta_lr,
                        "updated_tensors": updated_tensors,
                        "seconds": seconds,
                        "meta_checkpoint": str(current_meta_path),
                        "adapted_checkpoint": str(adapted_state),
                    }
                )
                history_file.flush()
                if not args.keep_adapted_outputs:
                    shutil.rmtree(adapt_dir, ignore_errors=True)

            epoch_path = args.output_dir / f"meta_init_epoch{epoch:03d}.pt"
            shutil.copy2(current_meta_path, epoch_path)

        final_path = args.output_dir / "meta_init_final.pt"
        shutil.copy2(current_meta_path, final_path)

        if args.run_meta_val:
            val_root = args.output_dir / "meta_val_adaptations"
            for case in val_cases:
                adapt_dir = val_root / f"{case}_adapt"
                adapted_state = adapt_dir / "mesh_fit_state.pt"
                start = time.time()
                run_fit_case(
                    python_bin=args.python_bin,
                    case=case,
                    mask_dir=args.mask_dir,
                    template=args.template,
                    output_dir=adapt_dir,
                    init_state=final_path,
                    save_fit_state=adapted_state,
                    inner_steps=args.inner_steps,
                    device=args.device,
                    disable_progress_bar=args.disable_progress_bar,
                )
                writer.writerow(
                    {
                        "phase": "val",
                        "epoch": int(args.meta_epochs),
                        "case": case,
                        "meta_lr": "",
                        "updated_tensors": "",
                        "seconds": time.time() - start,
                        "meta_checkpoint": str(final_path),
                        "adapted_checkpoint": str(adapted_state),
                    }
                )
                history_file.flush()

        print(f"Final Reptile meta-init: {args.output_dir / 'meta_init_final.pt'}")
        print(f"History: {history_path}")
    finally:
        history_file.close()


if __name__ == "__main__":
    main()
