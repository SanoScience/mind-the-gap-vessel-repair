from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _mip(mask: np.ndarray, axis: int) -> np.ndarray:
    return np.max(mask.astype(np.float32, copy=False), axis=axis)


def save_refine_qa(
    pred_mask: np.ndarray,
    mesh_mask: np.ndarray,
    refined_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    output_png: Path,
    output_pdf: Path | None = None,
    title: str = "",
) -> None:
    panels = [
        ("nnUNet", pred_mask),
        ("Mesh", mesh_mask),
        ("Refined", refined_mask),
    ]
    if gt_mask is not None:
        panels.append(("GT", gt_mask))

    fig, axes = plt.subplots(len(panels), 3, figsize=(12, 3 * len(panels)))
    if len(panels) == 1:
        axes = np.asarray([axes])
    for row, (name, mask) in enumerate(panels):
        views = [_mip(mask, 0), _mip(mask, 1), _mip(mask, 2)]
        for col, view in enumerate(views):
            axes[row, col].imshow(view, cmap="gray")
            axes[row, col].set_axis_off()
            axes[row, col].set_title(f"{title} {name}" if col == 0 else name)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    if output_pdf is not None:
        fig.savefig(output_pdf)
    plt.close(fig)
