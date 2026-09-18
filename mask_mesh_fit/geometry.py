from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from pointnet2_ops import pointnet2_utils
except Exception:
    pointnet2_utils = None


@dataclass(frozen=True)
class TemplateMesh:
    vertices: np.ndarray
    faces: np.ndarray
    edge_index: np.ndarray


def faces_to_edge_index(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
            faces[:, [1, 0]],
            faces[:, [2, 1]],
            faces[:, [0, 2]],
        ],
        axis=0,
    )
    return np.unique(edges, axis=0).T.astype(np.int64, copy=False)


def load_template(path: Path, radius: float = 0.8) -> TemplateMesh:
    data = np.load(path)
    if "vertices" not in data or "faces" not in data:
        raise RuntimeError(f"{path} must contain 'vertices' and 'faces'")
    vertices = data["vertices"].astype(np.float32, copy=False)
    faces = data["faces"].astype(np.int64, copy=False)
    edge_index = (
        data["edge_index"].astype(np.int64, copy=False)
        if "edge_index" in data
        else faces_to_edge_index(faces)
    )
    vertices = vertices - vertices.mean(axis=0, keepdims=True)
    max_radius = float(np.linalg.norm(vertices, axis=1).max())
    if not np.isfinite(max_radius) or max_radius <= 0:
        raise RuntimeError(f"{path}: invalid template radius {max_radius}")
    vertices = vertices / max_radius * float(radius)
    return TemplateMesh(vertices=vertices.astype(np.float32), faces=faces, edge_index=edge_index)


def undirected_edges_from_edge_index(edge_index: np.ndarray) -> np.ndarray:
    edges = edge_index.T.astype(np.int64, copy=False)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def adjacent_face_pairs(faces: np.ndarray) -> np.ndarray:
    edge_to_face: dict[tuple[int, int], list[int]] = {}
    for face_idx, face in enumerate(faces.astype(np.int64, copy=False)):
        tri_edges = (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        )
        for a, b in tri_edges:
            key = (a, b) if a < b else (b, a)
            edge_to_face.setdefault(key, []).append(face_idx)
    pairs = [faces_for_edge[:2] for faces_for_edge in edge_to_face.values() if len(faces_for_edge) == 2]
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec).clamp_min(1e-8)
    axis = rotvec / theta
    x, y, z = axis.unbind()
    zero = torch.zeros((), dtype=rotvec.dtype, device=rotvec.device)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + torch.sin(theta) * k + (1.0 - torch.cos(theta)) * (k @ k)


def rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(matrix).as_rotvec().astype(np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def pca_initial_transform(target_norm_xyz: np.ndarray, template_radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = target_norm_xyz.astype(np.float64, copy=False)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1.0
    local = centered @ basis
    lo = np.percentile(local, 1.0, axis=0)
    hi = np.percentile(local, 99.0, axis=0)
    half_extent = np.maximum(0.5 * (hi - lo), 0.05)
    scale = np.clip(half_extent / max(float(template_radius), 1e-6), 0.15, 2.5).astype(np.float32)
    translation = points.mean(axis=0).astype(np.float32)
    rotvec = rotation_matrix_to_rotvec(basis.astype(np.float64))
    return translation, np.log(scale).astype(np.float32), rotvec.astype(np.float32)


@torch.no_grad()
def sample_points(
    points: np.ndarray,
    count: int,
    mode: str,
    seed: int,
    device: torch.device,
    fps_candidate_points: int = 50000,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if count <= 0 or points.shape[0] <= count:
        return points.copy()
    rng = np.random.default_rng(seed)
    if mode == "random":
        idx = rng.choice(points.shape[0], size=count, replace=False)
        return points[idx].copy()
    if mode != "fps":
        raise ValueError(f"Unknown sampling mode: {mode}")

    candidate_count = min(int(fps_candidate_points), points.shape[0])
    candidate_idx = rng.choice(points.shape[0], size=candidate_count, replace=False)
    candidates = torch.as_tensor(points[candidate_idx], dtype=torch.float32, device=device)
    if pointnet2_utils is not None and candidates.is_cuda:
        fps_idx = pointnet2_utils.furthest_point_sample(
            candidates.unsqueeze(0).contiguous(),
            int(count),
        )[0].to(dtype=torch.long)
        return candidates.index_select(0, fps_idx).cpu().numpy()

    print(
        "FPS requested but pointnet2 CUDA FPS is unavailable in this runtime; "
        "falling back to random target subsampling."
    )
    idx = rng.choice(points.shape[0], size=count, replace=False)
    return points[idx].copy()
