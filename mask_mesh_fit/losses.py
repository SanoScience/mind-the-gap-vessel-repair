from __future__ import annotations

from dataclasses import dataclass

import torch
from chamferdist import ChamferDistance
from multigeomed.objectives.objectives_for_surface_mesh import (
    edge_length_regularization,
    face_area_regularization,
    laplacian_smoothing_loss,
    normal_consistency_loss,
)


@dataclass(frozen=True)
class MeshTensors:
    faces: torch.Tensor
    normal_pairs: torch.Tensor | None = None


def sample_points_from_faces(
    vertices: torch.Tensor,
    faces: torch.Tensor | None,
    num_samples: int,
) -> torch.Tensor:
    """Same surface sampling pattern used by the v23 Lightning mesh loss."""
    if faces is None or faces.numel() == 0:
        if vertices.shape[0] <= num_samples:
            return vertices
        idx = torch.randperm(vertices.shape[0], device=vertices.device)[:num_samples]
        return vertices[idx]

    tri = vertices[faces.long()]
    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    areas = torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1) * 0.5
    valid = torch.isfinite(areas) & (areas > 1e-8)
    if not torch.any(valid):
        if vertices.shape[0] <= num_samples:
            return vertices
        idx = torch.randperm(vertices.shape[0], device=vertices.device)[:num_samples]
        return vertices[idx]

    tri = tri[valid]
    areas = areas[valid]
    probs = areas / areas.sum().clamp_min(1e-8)
    face_idx = torch.multinomial(probs, num_samples=max(1, int(num_samples)), replacement=True)
    tri = tri[face_idx]

    r1 = torch.rand((tri.shape[0], 1), device=vertices.device, dtype=vertices.dtype)
    r2 = torch.rand((tri.shape[0], 1), device=vertices.device, dtype=vertices.dtype)
    sqrt_r1 = torch.sqrt(r1.clamp_min(1e-8))
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    return w0 * tri[:, 0, :] + w1 * tri[:, 1, :] + w2 * tri[:, 2, :]


def chamferdist_surface_loss(
    vertices: torch.Tensor,
    target_points: torch.Tensor,
    faces: torch.Tensor | None,
    chamfer: ChamferDistance,
    num_pred_samples: int,
) -> torch.Tensor:
    pred_samples = sample_points_from_faces(vertices, faces, num_samples=num_pred_samples)
    return chamfer(
        pred_samples.unsqueeze(0),
        target_points.unsqueeze(0),
        bidirectional=True,
        batch_reduction="mean",
        point_reduction="mean",
    )


def normalized_face_area_variance_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """v23-compatible normalized face-area variance regularizer."""
    if faces.numel() == 0:
        return vertices.new_zeros(())
    tri = vertices[faces.long()]
    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = torch.linalg.norm(cross, dim=-1) * 0.5
    if areas.numel() == 0:
        return vertices.new_zeros(())
    mean_area = areas.mean().clamp_min(eps)
    rel = (areas - mean_area) / mean_area
    return (rel * rel).mean()


def fast_normal_consistency_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    normal_pairs: torch.Tensor | None,
) -> torch.Tensor:
    """v23-style adjacent-face normal consistency with precomputed topology."""
    if faces.numel() == 0 or normal_pairs is None or normal_pairs.numel() == 0:
        return vertices.new_zeros(())
    tri = vertices[faces.long()]
    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp_min(1e-8)
    pairs = normal_pairs.to(device=vertices.device, dtype=torch.long)
    dot = (normals[pairs[:, 0]] * normals[pairs[:, 1]]).sum(dim=-1)
    return (1.0 - dot).mean()


def multigeomed_regularization_parts(vertices: torch.Tensor, faces: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "edge": edge_length_regularization(vertices, faces),
        "laplacian": laplacian_smoothing_loss(vertices, faces),
        "normal": normal_consistency_loss(vertices, faces),
        "face_area": face_area_regularization(vertices, faces),
        "face_area_var": normalized_face_area_variance_loss(vertices, faces),
    }


def multigeomed_regularization_parts_for_lambdas(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    lambda_edge: float,
    lambda_laplacian: float,
    lambda_normal: float,
    lambda_face_area: float,
    lambda_face_area_var: float,
    normal_pairs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    zero = vertices.new_zeros(())
    return {
        "edge": edge_length_regularization(vertices, faces) if lambda_edge > 0 else zero,
        "laplacian": laplacian_smoothing_loss(vertices, faces) if lambda_laplacian > 0 else zero,
        "normal": (
            fast_normal_consistency_loss(vertices, faces, normal_pairs)
            if lambda_normal > 0
            else zero
        ),
        "face_area": face_area_regularization(vertices, faces) if lambda_face_area > 0 else zero,
        "face_area_var": (
            normalized_face_area_variance_loss(vertices, faces) if lambda_face_area_var > 0 else zero
        ),
    }
