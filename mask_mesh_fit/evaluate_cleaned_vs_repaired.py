from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import label
from skimage.morphology import skeletonize


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if int(connectivity) == 6:
        return np.zeros((3, 3, 3), dtype=bool) | np.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=bool,
        )
    if int(connectivity) == 18:
        from scipy import ndimage

        return ndimage.generate_binary_structure(3, 2)
    if int(connectivity) == 26:
        from scipy import ndimage

        return ndimage.generate_binary_structure(3, 3)
    raise ValueError("connectivity must be one of 6, 18, or 26")


def _strip_suffix(case_id: str, suffix: str) -> str:
    if suffix and case_id.endswith(suffix):
        return case_id[: -len(suffix)]
    return case_id


def _load_mask(path: Path, label_id: int | None) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if label_id is None:
        return arr > 0
    return arr == int(label_id)


def _skeleton(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask > 0, method="lee")


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum() + eps))


def largest_component(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    labels, n = label(mask, structure=structure)
    if n <= 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def cldice_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    skel_pred = _skeleton(pred)
    skel_gt = _skeleton(gt)
    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return {"cldice": 0.0, "tprec": 0.0, "trec": 0.0, "false_branch": 0.0}
    tprec = np.logical_and(skel_pred, gt).sum() / (skel_pred.sum() + eps)
    trec = np.logical_and(skel_gt, pred).sum() / (skel_gt.sum() + eps)
    cldice = 2.0 * tprec * trec / (tprec + trec + eps)
    return {
        "cldice": float(cldice),
        "tprec": float(tprec),
        "trec": float(trec),
        "false_branch": float(1.0 - tprec),
    }


def length_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    skel_pred = _skeleton(pred)
    skel_gt = _skeleton(gt)
    len_pred = int(skel_pred.sum())
    len_gt = int(skel_gt.sum())
    return {
        "length_pred": len_pred,
        "length_gt": len_gt,
        "length_ratio": float(len_pred / (len_gt + eps) if len_gt > 0 else 0.0),
        "length_error": int(abs(len_pred - len_gt)),
    }


def cc_metrics(pred: np.ndarray, gt: np.ndarray, structure: np.ndarray) -> dict[str, float]:
    cc_pred, n_pred = label(pred, structure=structure)
    cc_gt, n_gt = label(gt, structure=structure)
    pred_main = largest_component(pred, structure)
    gt_main = largest_component(gt, structure)
    union = np.logical_or(pred_main, gt_main).sum()
    largest_iou = float(np.logical_and(pred_main, gt_main).sum() / union) if union > 0 else 0.0
    return {
        "n_pred": int(n_pred),
        "n_gt": int(n_gt),
        "cc_diff": int(abs(n_pred - n_gt)),
        "cc_ratio": float(abs(n_pred - n_gt) / max(n_gt, 1)),
        "largest_component_iou": largest_iou,
    }


def split_errors(pred: np.ndarray, gt: np.ndarray, structure: np.ndarray) -> dict[str, int]:
    cc_pred, _n_pred = label(pred, structure=structure)
    cc_gt, n_gt = label(gt, structure=structure)
    fragmented_objects = 0
    fragmentation_degree = 0
    for gt_id in range(1, n_gt + 1):
        overlapping_pred = np.unique(cc_pred[cc_gt == gt_id])
        overlapping_pred = overlapping_pred[overlapping_pred > 0]
        if len(overlapping_pred) > 1:
            fragmented_objects += 1
            fragmentation_degree += len(overlapping_pred) - 1
    return {
        "fragmented_objects": int(fragmented_objects),
        "fragmentation_degree": int(fragmentation_degree),
    }


def beta1_fast(mask: np.ndarray) -> dict[str, int]:
    skel = _skeleton(mask > 0)
    coords = np.array(np.nonzero(skel)).T
    coord_set = set(map(tuple, coords))
    graph = nx.Graph()
    neighbors = [
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if not (i == 0 and j == 0 and k == 0)
    ]
    for coord in coord_set:
        graph.add_node(coord)
    for z, y, x in coord_set:
        for dz, dy, dx in neighbors:
            nb = (z + dz, y + dy, x + dx)
            if nb in coord_set:
                graph.add_edge((z, y, x), nb)
    if graph.number_of_nodes() == 0:
        return {"beta0": 0, "beta1": 0}
    vertex_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    component_count = nx.number_connected_components(graph)
    return {"beta0": int(component_count), "beta1": int(edge_count - vertex_count + component_count)}


def full_metrics(pred: np.ndarray, gt: np.ndarray, structure: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics["dice"] = dice(pred, gt)
    metrics["lcc_dice"] = dice(largest_component(pred, structure), largest_component(gt, structure))
    metrics.update(cldice_metrics(pred, gt))
    metrics.update(length_metrics(pred, gt))
    metrics.update(cc_metrics(pred, gt, structure))
    split = split_errors(pred, gt, structure)
    metrics.update(split)
    metrics["normalized_fragmentation"] = float(split["fragmentation_degree"] / (_skeleton(gt).sum() + 1e-8))
    beta_gt = beta1_fast(gt)
    beta_pred = beta1_fast(pred)
    metrics["beta0_gt"] = beta_gt["beta0"]
    metrics["beta1_gt"] = beta_gt["beta1"]
    metrics["beta0_pred"] = beta_pred["beta0"]
    metrics["beta1_pred"] = beta_pred["beta1"]
    metrics["beta0_error"] = abs(beta_pred["beta0"] - beta_gt["beta0"])
    metrics["beta1_error"] = abs(beta_pred["beta1"] - beta_gt["beta1"])
    metrics["beta1_score"] = float(beta_pred["beta1"] / beta_gt["beta1"]) if beta_gt["beta1"] > 0 else 0.0
    return metrics


def discover_cases(run_root: Path, suffix: str, cleaned_filename: str, repaired_filename: str) -> list[tuple[str, Path, Path]]:
    cases: list[tuple[str, Path, Path]] = []
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.endswith(suffix):
            continue
        case_id = _strip_suffix(run_dir.name, suffix)
        cleaned = run_dir / cleaned_filename
        repaired = run_dir / repaired_filename
        if cleaned.exists() and repaired.exists():
            cases.append((case_id, cleaned, repaired))
    return cases


def save_plots(df: pd.DataFrame, output_dir: Path, title: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{title}.pdf"
    cases = df["case"].tolist()
    x = np.arange(len(cases))
    colors = {"cleaned": "#d95f02", "repaired": "#1b9eae", "gt": "#c9a600"}

    def save_current(fig: plt.Figure, filename: str, pdf: PdfPages) -> None:
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        ax.axis("off")
        lines = [
            title,
            f"Cases: {len(df)}",
            "",
            f"Dice mean: cleaned {df['cleaned_dice'].mean():.4f} -> repaired {df['repaired_dice'].mean():.4f}",
            f"LCC Dice mean: cleaned {df['cleaned_lcc_dice'].mean():.4f} -> repaired {df['repaired_lcc_dice'].mean():.4f}",
            f"beta0 error mean: cleaned {df['cleaned_beta0_error'].mean():.3f} -> repaired {df['repaired_beta0_error'].mean():.3f}",
            f"fragmentation mean: cleaned {df['cleaned_fragmentation_degree'].mean():.3f} -> repaired {df['repaired_fragmentation_degree'].mean():.3f}",
        ]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=12)
        save_current(fig, f"{title}_summary.png", pdf)

        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        metrics = ["dice", "lcc_dice"]
        loc = np.arange(len(metrics))
        width = 0.36
        ax.bar(loc - width / 2, [df[f"cleaned_{m}"].mean() for m in metrics], width, label="artifact-cleaned nnUNet", color=colors["cleaned"], alpha=0.85)
        ax.bar(loc + width / 2, [df[f"repaired_{m}"].mean() for m in metrics], width, label="final repaired", color=colors["repaired"], alpha=0.85)
        ax.set_xticks(loc)
        ax.set_xticklabels(metrics)
        ax.set_title("Whole-mask Dice and largest-component Dice")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        save_current(fig, f"{title}_dice_mean.png", pdf)

        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        groups = ["beta0", "components"]
        gt_vals = [df["cleaned_beta0_gt"].mean(), df["cleaned_n_gt"].mean()]
        cleaned_vals = [df["cleaned_beta0_pred"].mean(), df["cleaned_n_pred"].mean()]
        repaired_vals = [df["repaired_beta0_pred"].mean(), df["repaired_n_pred"].mean()]
        loc = np.arange(len(groups))
        width = 0.30
        ax.bar(loc - width, gt_vals, width, label="GT", color=colors["gt"], alpha=0.85)
        ax.bar(loc, cleaned_vals, width, label="artifact-cleaned nnUNet", color=colors["cleaned"], alpha=0.85)
        ax.bar(loc + width, repaired_vals, width, label="final repaired", color=colors["repaired"], alpha=0.85)
        ax.set_xticks(loc)
        ax.set_xticklabels(groups)
        ax.set_title("Topology counts")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        save_current(fig, f"{title}_topology_counts_mean.png", pdf)

        fig, ax = plt.subplots(figsize=(10, 5.0))
        metrics = ["beta0_error", "fragmentation_degree", "cc_diff", "beta1_error"]
        loc = np.arange(len(metrics))
        width = 0.36
        ax.bar(loc - width / 2, [df[f"cleaned_{m}"].mean() for m in metrics], width, label="artifact-cleaned nnUNet", color=colors["cleaned"], alpha=0.85)
        ax.bar(loc + width / 2, [df[f"repaired_{m}"].mean() for m in metrics], width, label="final repaired", color=colors["repaired"], alpha=0.85)
        ax.set_xticks(loc)
        ax.set_xticklabels(metrics, rotation=20, ha="right")
        ax.set_title("Topology error metrics")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        save_current(fig, f"{title}_topology_errors_mean.png", pdf)

        for metric, ylabel in [
            ("dice", "Dice"),
            ("lcc_dice", "Largest-component Dice"),
            ("beta0_error", "Beta0 error"),
            ("fragmentation_degree", "Fragmentation degree"),
        ]:
            fig, ax = plt.subplots(figsize=(max(12, len(cases) * 0.55), 5.2))
            width = 0.42
            ax.bar(x - width / 2, df[f"cleaned_{metric}"], width, label="artifact-cleaned nnUNet", color=colors["cleaned"], alpha=0.85)
            ax.bar(x + width / 2, df[f"repaired_{metric}"], width, label="final repaired", color=colors["repaired"], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(cases, rotation=65, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Per-case {ylabel}")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            save_current(fig, f"{title}_per_case_{metric}.png", pdf)

        fig, axes = plt.subplots(3, 1, figsize=(max(12, len(cases) * 0.55), 10.0), sharex=True)
        deltas = [
            ("delta Dice", df["repaired_dice"] - df["cleaned_dice"]),
            ("delta LCC Dice", df["repaired_lcc_dice"] - df["cleaned_lcc_dice"]),
            ("delta beta0 error", df["repaired_beta0_error"] - df["cleaned_beta0_error"]),
        ]
        for ax, (name, values) in zip(axes, deltas):
            ax.bar(x, values, color=colors["repaired"], alpha=0.85)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"Repaired - cleaned: {name}")
            ax.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(cases, rotation=65, ha="right", fontsize=8)
        save_current(fig, f"{title}_per_case_deltas.png", pdf)

    return pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare artifact-cleaned nnUNet masks against final v24 repaired masks.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--run-suffix-filter", required=True)
    parser.add_argument("--cleaned-filename", default="artifact_cleaned_mask.nii.gz")
    parser.add_argument("--repaired-filename", default="repaired_mask_mesh_path_connect.nii.gz")
    parser.add_argument("--label-id", type=int, default=1)
    parser.add_argument("--title", default="aorta_cleaned_vs_repaired_mesh_path")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Evaluate only this case ID. Can be passed multiple times.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap for quick smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structure = _connectivity_structure(args.connectivity)
    cases = discover_cases(
        args.run_root,
        args.run_suffix_filter,
        args.cleaned_filename,
        args.repaired_filename,
    )
    if not cases:
        raise RuntimeError(f"No cases found in {args.run_root} with suffix {args.run_suffix_filter}")
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case[0] in wanted]
        missing = sorted(wanted - {case[0] for case in cases})
        if missing:
            raise RuntimeError(f"Requested cases not found: {missing}")
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]
    rows: list[dict[str, object]] = []
    for case_id, cleaned_path, repaired_path in cases:
        gt_path = args.gt_dir / f"{case_id}.nii.gz"
        if not gt_path.exists():
            print(f"Skipping {case_id}: missing GT {gt_path}")
            continue
        cleaned = _load_mask(cleaned_path, args.label_id)
        repaired = _load_mask(repaired_path, args.label_id)
        gt = _load_mask(gt_path, args.label_id)
        if cleaned.shape != gt.shape or repaired.shape != gt.shape:
            raise RuntimeError(
                f"{case_id}: shape mismatch cleaned={cleaned.shape}, repaired={repaired.shape}, gt={gt.shape}"
            )
        row: dict[str, object] = {
            "case": case_id,
            "cleaned_path": str(cleaned_path),
            "repaired_path": str(repaired_path),
            "gt_path": str(gt_path),
        }
        for prefix, pred in [("cleaned", cleaned), ("repaired", repaired)]:
            for key, value in full_metrics(pred, gt, structure).items():
                row[f"{prefix}_{key}"] = value
        row["delta_dice"] = row["repaired_dice"] - row["cleaned_dice"]
        row["delta_lcc_dice"] = row["repaired_lcc_dice"] - row["cleaned_lcc_dice"]
        row["delta_beta0_error"] = row["repaired_beta0_error"] - row["cleaned_beta0_error"]
        row["delta_fragmentation_degree"] = (
            row["repaired_fragmentation_degree"] - row["cleaned_fragmentation_degree"]
        )
        rows.append(row)
        print(
            f"{case_id}: cleaned dice={row['cleaned_dice']:.4f}, "
            f"repaired dice={row['repaired_dice']:.4f}, "
            f"cleaned beta0err={row['cleaned_beta0_error']}, "
            f"repaired beta0err={row['repaired_beta0_error']}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows were evaluated.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    csv_path = args.output_dir / f"evaluation_results_{timestamp}_{args.title}.csv"
    df.to_csv(csv_path, index=False)
    plots_dir = args.plots_dir if args.plots_dir is not None else args.output_dir / "connectivity_comparison_plots" / args.title
    pdf_path = save_plots(df, plots_dir, args.title)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plots: {plots_dir}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
