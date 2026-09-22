from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TOPCOW mesh Chamfer convergence curves.")
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--reptile-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="onecase_init_6000_first500")
    parser.add_argument("--reptile-label", default="reptile_500")
    parser.add_argument("--scalar-tag", default="mesh/final_chamfer")
    parser.add_argument("--best-scalar-tag", default="mesh/best_final_chamfer")
    parser.add_argument("--max-steps", type=int, default=500)
    return parser.parse_args()


def read_cases(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"Case list does not exist: {path}")
    cases = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not cases:
        raise RuntimeError(f"Case list is empty: {path}")
    return cases


def case_fit_dir(root: Path, case: str) -> Path:
    return root / f"{case}_voxelrepair_torus10k"


def event_source(fit_dir: Path) -> Path:
    tensorboard_dir = fit_dir / "tensorboard"
    files = sorted(tensorboard_dir.glob("events.out.tfevents*"))
    if not files:
        raise RuntimeError(f"No TensorBoard event files found in {tensorboard_dir}")
    return tensorboard_dir


def scalar_curve(path: Path, tag: str) -> list[tuple[int, float]]:
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        raise RuntimeError(f"{path}: missing scalar tag {tag!r}; available={tags}")
    return [(int(event.step), float(event.value)) for event in acc.Scalars(tag)]


def best_prefix(values: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(values)


def curve_at_steps(curve: list[tuple[int, float]], steps: np.ndarray) -> np.ndarray:
    if not curve:
        return np.full_like(steps, np.nan, dtype=np.float64)
    by_step = {int(step): float(value) for step, value in curve}
    xs = np.array(sorted(by_step), dtype=np.int64)
    ys = np.array([by_step[int(step)] for step in xs], dtype=np.float64)
    out = np.full(len(steps), np.nan, dtype=np.float64)
    for i, step in enumerate(steps):
        idx = np.searchsorted(xs, step, side="right") - 1
        if idx >= 0:
            out[i] = ys[idx]
    return out


def load_run_curves(root: Path, cases: list[str], tag: str, max_steps: int) -> dict[str, np.ndarray]:
    steps = np.arange(1, int(max_steps) + 1, dtype=np.int64)
    curves = {}
    missing = []
    for case in cases:
        fit_dir = case_fit_dir(root, case)
        try:
            raw = scalar_curve(event_source(fit_dir), tag)
        except RuntimeError as exc:
            missing.append(f"{case}: {exc}")
            continue
        values = curve_at_steps(raw, steps)
        curves[case] = values
    if missing:
        preview = "\n".join(missing[:10])
        raise RuntimeError(f"Missing convergence data for {len(missing)} case(s):\n{preview}")
    return curves


def summarize(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": math.nan, "median": math.nan, "std": math.nan}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def write_curve_csv(
    path: Path,
    steps: np.ndarray,
    baseline: dict[str, np.ndarray],
    reptile: dict[str, np.ndarray],
) -> None:
    cases = sorted(baseline)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "method", "step", "final_chamfer", "best_so_far_chamfer"])
        for case in cases:
            for method, curves in (("baseline", baseline), ("reptile", reptile)):
                vals = curves[case]
                best = best_prefix(vals)
                for step, value, best_value in zip(steps, vals, best):
                    writer.writerow([case, method, int(step), value, best_value])


def write_summary_csv(
    path: Path,
    steps: np.ndarray,
    baseline: dict[str, np.ndarray],
    reptile: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    selected_steps = sorted(set([1, 5, 10, 25, 50, 100, 200, 300, 400, int(steps[-1])]))
    rows = []
    with path.open("w", newline="") as f:
        fieldnames = [
            "step",
            "baseline_mean",
            "baseline_median",
            "reptile_mean",
            "reptile_median",
            "mean_delta_baseline_minus_reptile",
            "median_delta_baseline_minus_reptile",
            "cases_reptile_better",
            "num_cases",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in selected_steps:
            idx = int(step) - 1
            b = np.array([best_prefix(values)[idx] for values in baseline.values()], dtype=np.float64)
            r = np.array([best_prefix(values)[idx] for values in reptile.values()], dtype=np.float64)
            b_sum = summarize(b)
            r_sum = summarize(r)
            row = {
                "step": int(step),
                "baseline_mean": b_sum["mean"],
                "baseline_median": b_sum["median"],
                "reptile_mean": r_sum["mean"],
                "reptile_median": r_sum["median"],
                "mean_delta_baseline_minus_reptile": b_sum["mean"] - r_sum["mean"],
                "median_delta_baseline_minus_reptile": b_sum["median"] - r_sum["median"],
                "cases_reptile_better": int(np.sum(r < b)),
                "num_cases": int(len(b)),
            }
            rows.append(row)
            writer.writerow(row)
    return rows


def plot_curves(
    output_png: Path,
    steps: np.ndarray,
    baseline: dict[str, np.ndarray],
    reptile: dict[str, np.ndarray],
    baseline_label: str,
    reptile_label: str,
) -> None:
    b = np.stack([best_prefix(values) for values in baseline.values()], axis=0)
    r = np.stack([best_prefix(values) for values in reptile.values()], axis=0)
    b_mean = np.nanmean(b, axis=0)
    r_mean = np.nanmean(r, axis=0)
    b_sem = np.nanstd(b, axis=0, ddof=1) / math.sqrt(b.shape[0])
    r_sem = np.nanstd(r, axis=0, ddof=1) / math.sqrt(r.shape[0])

    plt.figure(figsize=(9, 5), dpi=160)
    plt.plot(steps, b_mean, label=baseline_label, color="#52616b")
    plt.fill_between(steps, b_mean - b_sem, b_mean + b_sem, color="#52616b", alpha=0.18, linewidth=0)
    plt.plot(steps, r_mean, label=reptile_label, color="#c94c4c")
    plt.fill_between(steps, r_mean - r_sem, r_mean + r_sem, color="#c94c4c", alpha=0.18, linewidth=0)
    plt.xlabel("adaptation step")
    plt.ylabel("best-so-far Chamfer")
    plt.title("TOPCOW held-out Chamfer convergence")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = read_cases(args.case_list)
    steps = np.arange(1, int(args.max_steps) + 1, dtype=np.int64)
    baseline = load_run_curves(args.baseline_root, cases, args.scalar_tag, args.max_steps)
    reptile = load_run_curves(args.reptile_root, cases, args.scalar_tag, args.max_steps)

    curve_csv = args.output_dir / "chamfer_convergence_curves.csv"
    summary_csv = args.output_dir / "chamfer_convergence_summary.csv"
    plot_png = args.output_dir / "chamfer_convergence_mean_sem.png"
    write_curve_csv(curve_csv, steps, baseline, reptile)
    rows = write_summary_csv(summary_csv, steps, baseline, reptile)
    plot_curves(plot_png, steps, baseline, reptile, args.baseline_label, args.reptile_label)

    metadata = {
        "case_list": str(args.case_list),
        "num_cases": len(cases),
        "baseline_root": str(args.baseline_root),
        "reptile_root": str(args.reptile_root),
        "scalar_tag": args.scalar_tag,
        "max_steps": args.max_steps,
        "curve_csv": str(curve_csv),
        "summary_csv": str(summary_csv),
        "plot_png": str(plot_png),
    }
    (args.output_dir / "chamfer_convergence_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote: {curve_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {plot_png}")
    print("Summary rows:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
