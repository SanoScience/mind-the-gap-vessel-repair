from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _set_equal_3d(ax, points: np.ndarray) -> None:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.max(hi - lo))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _sample_points(points: np.ndarray, max_points: int, seed: int = 13) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(points.shape[0], size=max_points, replace=False)]


def _plot_mesh(ax, vertices: np.ndarray, faces: np.ndarray, color: str, alpha: float, max_faces: int) -> None:
    if faces.shape[0] > max_faces:
        stride = int(np.ceil(faces.shape[0] / max_faces))
        faces = faces[::stride]
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color=color,
        alpha=alpha,
        linewidth=0.05,
        edgecolor="black",
    )


def save_qa_overlay(
    target_points_physical: np.ndarray,
    mesh_vertices_physical: np.ndarray,
    mesh_faces: np.ndarray,
    output_png: Path,
    output_pdf: Path | None = None,
    title: str = "",
    max_target_points: int = 12000,
    max_faces: int = 12000,
) -> None:
    target_sample = _sample_points(target_points_physical, max_target_points)
    all_points = np.vstack([target_sample, mesh_vertices_physical])
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(target_sample[:, 0], target_sample[:, 1], target_sample[:, 2], s=0.2, c="gold", alpha=0.35)
    ax1.set_title(f"{title} nnUNet mask surface")
    _set_equal_3d(ax1, all_points)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _plot_mesh(ax2, mesh_vertices_physical, mesh_faces, color="cyan", alpha=0.75, max_faces=max_faces)
    ax2.set_title(f"{title} fitted mesh")
    _set_equal_3d(ax2, all_points)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(target_sample[:, 0], target_sample[:, 1], target_sample[:, 2], s=0.2, c="gold", alpha=0.25)
    _plot_mesh(ax3, mesh_vertices_physical, mesh_faces, color="cyan", alpha=0.55, max_faces=max_faces)
    ax3.set_title(f"{title} overlay")
    _set_equal_3d(ax3, all_points)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    if output_pdf is not None:
        fig.savefig(output_pdf)
    plt.close(fig)
