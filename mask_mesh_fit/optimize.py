from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from chamferdist import ChamferDistance
from torch.utils.data import DataLoader, Dataset

from .geometry import (
    adjacent_face_pairs,
    pca_initial_transform,
    rodrigues,
    sample_points,
)
from .losses import (
    MeshTensors,
    chamferdist_surface_loss,
    multigeomed_regularization_parts_for_lambdas,
)
from .mesh_deformation_decoder import MeshDecoder

TensorboardLogFn = Callable[[str, int, dict[str, torch.Tensor | float]], None]


@dataclass(frozen=True)
class StageConfig:
    name: str
    steps: int
    lr: float
    target_points: int
    lambda_chamfer: float
    lambda_bbox: float = 0.0
    lambda_vertex_chamfer: float = 0.0
    lambda_deform: float = 0.0
    lambda_edge: float = 0.0
    lambda_laplacian: float = 0.0
    lambda_normal: float = 0.0
    lambda_face_area: float = 0.0
    lambda_face_area_var: float = 0.0
    log_every: int = 50


@dataclass(frozen=True)
class DecoderConfig:
    deformation_mode: str = "decoder"
    stages: int = 4
    latent_dim: int = 128
    local_feature_dim: int = 128
    hidden_dim: int = 128
    num_blocks: int = 3
    graph_layer: str = "gcn"
    edge_features: str = "none"
    stage_max_offsets: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1)
    coarse_max_offset: float = 0.50
    detail_max_offset: float = 0.20
    stage_loss_weight: float = 0.10
    fit_steps: int = 0
    train_alignment: bool = False


@dataclass
class StageResult:
    name: str
    vertices_norm: np.ndarray
    metrics: dict[str, float]


class _StepDataset(Dataset):
    def __init__(self, steps: int) -> None:
        self.steps = int(max(1, steps))

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(index, dtype=torch.long)


def _make_lightning_trainer(
    steps: int,
    device: torch.device,
    enable_progress_bar: bool,
) -> L.Trainer:
    accelerator = "gpu" if device.type == "cuda" else "cpu"
    return L.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=1,
        limit_train_batches=int(max(1, steps)),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=bool(enable_progress_bar),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )


def _fit_lightning_stage(
    module: L.LightningModule,
    steps: int,
    device: torch.device,
    enable_progress_bar: bool,
) -> None:
    dataloader = DataLoader(_StepDataset(steps), batch_size=1, shuffle=False, num_workers=0)
    trainer = _make_lightning_trainer(steps=steps, device=device, enable_progress_bar=enable_progress_bar)
    trainer.fit(module, train_dataloaders=dataloader)


def _tensor_metrics(parts: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in parts.items()}


def _load_matching_fit_state(
    module: torch.nn.Module,
    checkpoint_path: Path,
    *,
    load_alignment: bool = False,
) -> dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location=module.device)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"{checkpoint_path}: expected a state_dict-like checkpoint")

    current = module.state_dict()
    allowed_prefixes = ("decoder.", "latent0", "local_features.")
    if load_alignment:
        allowed_prefixes = allowed_prefixes + ("translation", "log_scale", "rotvec")
    loaded: list[str] = []
    skipped: list[str] = []
    next_state = dict(current)
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            skipped.append(key)
            continue
        if not key.startswith(allowed_prefixes):
            skipped.append(key)
            continue
        if key not in current:
            skipped.append(key)
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        next_state[key] = value.to(device=current[key].device, dtype=current[key].dtype)
        loaded.append(key)
    module.load_state_dict(next_state, strict=True)
    return {
        "path": str(checkpoint_path),
        "loaded_keys": loaded,
        "skipped_keys": skipped,
        "num_loaded_keys": len(loaded),
        "num_skipped_keys": len(skipped),
        "load_alignment": bool(load_alignment),
    }


def _log_tensorboard(
    tb_log_fn: TensorboardLogFn | None,
    stage: str,
    step: int,
    parts: dict[str, torch.Tensor | float],
) -> None:
    if tb_log_fn is None:
        return
    tb_log_fn(stage, step, parts)


def _bbox_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_lo, pred_hi = pred.min(dim=0).values, pred.max(dim=0).values
    target_lo, target_hi = target.min(dim=0).values, target.max(dim=0).values
    return F.mse_loss(pred_lo, target_lo) + F.mse_loss(pred_hi, target_hi)


def _make_mesh_tensors(faces_np: np.ndarray, device: torch.device) -> MeshTensors:
    faces = torch.as_tensor(faces_np.astype(np.int64, copy=False), dtype=torch.long, device=device)
    normal_pairs_np = adjacent_face_pairs(faces_np)
    normal_pairs = torch.as_tensor(normal_pairs_np, dtype=torch.long, device=device)
    return MeshTensors(faces=faces, normal_pairs=normal_pairs)


class _AlignLightningModule(L.LightningModule):
    def __init__(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        template_radius: float,
        target_all: np.ndarray,
        cfg: StageConfig,
        tb_log_fn: TensorboardLogFn | None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tb_log_fn = tb_log_fn
        self.chamfer = ChamferDistance()
        self.register_buffer("base", base.detach().clone())
        self.register_buffer("target", target.detach().clone())
        init_t, init_log_s, init_r = pca_initial_transform(target_all, template_radius=template_radius)
        self.translation = torch.nn.Parameter(torch.as_tensor(init_t, dtype=torch.float32, device=base.device))
        self.log_scale = torch.nn.Parameter(torch.as_tensor(init_log_s, dtype=torch.float32, device=base.device))
        self.rotvec = torch.nn.Parameter(torch.as_tensor(init_r, dtype=torch.float32, device=base.device))
        self.latest_vertices = self.base.detach()
        self.latest_parts: dict[str, torch.Tensor] = {}

    def vertices(self) -> torch.Tensor:
        rot = rodrigues(self.rotvec)
        return (self.base * torch.exp(self.log_scale)[None, :]) @ rot.T + self.translation[None, :]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch
        aligned = self.vertices()
        chamfer_loss = self.chamfer(
            aligned.unsqueeze(0),
            self.target.unsqueeze(0),
            bidirectional=True,
            batch_reduction="mean",
            point_reduction="mean",
        )
        bbox = _bbox_loss(aligned, self.target)
        loss = self.cfg.lambda_chamfer * chamfer_loss + self.cfg.lambda_bbox * bbox
        parts = {
            "loss": loss,
            "chamfer": chamfer_loss,
            "bbox": bbox,
            "scale_x": torch.exp(self.log_scale)[0],
            "scale_y": torch.exp(self.log_scale)[1],
            "scale_z": torch.exp(self.log_scale)[2],
        }
        self.latest_vertices = aligned.detach()
        self.latest_parts = {key: value.detach() for key, value in parts.items()}
        for name, value in parts.items():
            self.log(f"{self.cfg.name}/{name}", value, on_step=True, prog_bar=name in {"loss", "chamfer"})
        _log_tensorboard(self.tb_log_fn, self.cfg.name, batch_idx + 1, parts)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([self.translation, self.log_scale, self.rotvec], lr=self.cfg.lr)


class _DirectDeformLightningModule(L.LightningModule):
    def __init__(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
        faces: torch.Tensor,
        normal_pairs: torch.Tensor,
        cfg: StageConfig,
        tb_log_fn: TensorboardLogFn | None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tb_log_fn = tb_log_fn
        self.chamfer = ChamferDistance()
        self.register_buffer("reference", reference.detach().clone())
        self.register_buffer("target", target.detach().clone())
        self.register_buffer("faces", faces.detach().clone())
        self.register_buffer("normal_pairs", normal_pairs.detach().clone())
        self.offsets = torch.nn.Parameter(torch.zeros_like(reference))
        self.latest_vertices = self.reference.detach()
        self.latest_parts: dict[str, torch.Tensor] = {}

    def vertices(self) -> torch.Tensor:
        return self.reference + self.offsets

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch
        vertices = self.vertices()
        loss, parts = _weighted_deform_loss(
            vertices,
            self.target,
            MeshTensors(faces=self.faces, normal_pairs=self.normal_pairs),
            self.cfg,
            chamfer=self.chamfer,
            num_pred_samples=self.cfg.target_points,
            reference=self.reference,
        )
        self.latest_vertices = vertices.detach()
        self.latest_parts = {key: value.detach() for key, value in parts.items()}
        for name, value in parts.items():
            self.log(f"{self.cfg.name}/{name}", value, on_step=True, prog_bar=name in {"loss", "chamfer"})
        _log_tensorboard(self.tb_log_fn, self.cfg.name, batch_idx + 1, parts)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([self.offsets], lr=self.cfg.lr)


class _DecoderDeformLightningModule(L.LightningModule):
    def __init__(
        self,
        aligned: torch.Tensor,
        faces: torch.Tensor,
        normal_pairs: torch.Tensor,
        edge_index: torch.Tensor,
        decoder_cfg: DecoderConfig,
        tb_log_fn: TensorboardLogFn | None,
    ) -> None:
        super().__init__()
        self.decoder_cfg = decoder_cfg
        self.tb_log_fn = tb_log_fn
        self.chamfer = ChamferDistance()
        self.register_buffer("reference", aligned.detach().clone().unsqueeze(0))
        self.register_buffer("faces", faces.detach().clone())
        self.register_buffer("normal_pairs", normal_pairs.detach().clone())
        self.register_buffer("edge_index", edge_index.detach().clone())
        num_vertices = int(aligned.shape[0])
        self.decoder = MeshDecoder(
            latent_dim=decoder_cfg.latent_dim,
            max_offset=max(float(decoder_cfg.coarse_max_offset), float(decoder_cfg.detail_max_offset)),
            stages=2,
            local_feature_dims=[decoder_cfg.local_feature_dim, decoder_cfg.local_feature_dim],
            stage_hidden_dims=[decoder_cfg.hidden_dim, decoder_cfg.hidden_dim],
            stage_max_offsets=[decoder_cfg.coarse_max_offset, decoder_cfg.detail_max_offset],
            num_gcn_blocks=decoder_cfg.num_blocks,
            graph_layer=decoder_cfg.graph_layer,
            edge_features=decoder_cfg.edge_features,
        )
        self.latent0 = torch.nn.Parameter(torch.zeros((1, num_vertices, decoder_cfg.latent_dim), dtype=torch.float32))
        self.local0 = torch.nn.Parameter(
            torch.zeros((1, num_vertices, decoder_cfg.local_feature_dim), dtype=torch.float32)
        )
        self.local1 = torch.nn.Parameter(
            torch.zeros((1, num_vertices, decoder_cfg.local_feature_dim), dtype=torch.float32)
        )
        self.active_cfg: StageConfig | None = None
        self.active_target: torch.Tensor | None = None
        self.active_lr = 1e-3
        self.active_stage = "coarse"
        self.latest_coarse_vertices = aligned.detach()
        self.latest_detail_vertices = aligned.detach()
        self.latest_parts: dict[str, torch.Tensor] = {}

    def set_stage(self, stage: str, cfg: StageConfig, target: torch.Tensor) -> None:
        self.active_stage = str(stage)
        self.active_cfg = cfg
        self.active_lr = float(cfg.lr)
        self.active_target = target.detach().clone()

    def forward_two_stages(self) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_vertices, coarse_latent, _gate0 = self.decoder.forward_stage(
            0,
            self.latent0,
            self.local0,
            self.reference,
            edge_index=self.edge_index,
        )
        detail_vertices, _detail_latent, _gate1 = self.decoder.forward_stage(
            1,
            coarse_latent,
            self.local1,
            coarse_vertices,
            edge_index=self.edge_index,
        )
        return coarse_vertices.squeeze(0), detail_vertices.squeeze(0)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch
        if self.active_cfg is None or self.active_target is None:
            raise RuntimeError("Decoder stage was not configured before Trainer.fit")
        cfg = self.active_cfg
        coarse_vertices, detail_vertices = self.forward_two_stages()
        mesh = MeshTensors(faces=self.faces, normal_pairs=self.normal_pairs)
        if self.active_stage == "coarse":
            loss, parts = _weighted_deform_loss(
                coarse_vertices,
                self.active_target,
                mesh,
                cfg,
                chamfer=self.chamfer,
                num_pred_samples=cfg.target_points,
                reference=self.reference.squeeze(0),
            )
        else:
            detail_loss, parts = _weighted_deform_loss(
                detail_vertices,
                self.active_target,
                mesh,
                cfg,
                chamfer=self.chamfer,
                num_pred_samples=cfg.target_points,
                reference=self.reference.squeeze(0),
            )
            if self.decoder_cfg.stage_loss_weight > 0:
                coarse_loss, _coarse_parts = _weighted_deform_loss(
                    coarse_vertices,
                    self.active_target,
                    mesh,
                    cfg,
                    chamfer=self.chamfer,
                    num_pred_samples=cfg.target_points,
                    reference=self.reference.squeeze(0),
                )
                loss = detail_loss + self.decoder_cfg.stage_loss_weight * coarse_loss
            else:
                coarse_loss = detail_loss.new_zeros(())
                loss = detail_loss
            parts = {**parts, "loss": loss, "detail_loss": detail_loss, "coarse_aux_loss": coarse_loss}

        self.latest_coarse_vertices = coarse_vertices.detach()
        self.latest_detail_vertices = detail_vertices.detach()
        self.latest_parts = {key: value.detach() for key, value in parts.items()}
        for name, value in parts.items():
            self.log(f"{cfg.name}/{name}", value, on_step=True, prog_bar=name in {"loss", "chamfer"})
        _log_tensorboard(self.tb_log_fn, cfg.name, batch_idx + 1, parts)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.active_lr)


class _FourStageDecoderFitModule(L.LightningModule):
    def __init__(
        self,
        base: torch.Tensor,
        target_align: torch.Tensor,
        target_fit: torch.Tensor,
        target_all: np.ndarray,
        template_radius: float,
        faces: torch.Tensor,
        normal_pairs: torch.Tensor,
        edge_index: torch.Tensor,
        align_cfg: StageConfig,
        mesh_cfg: StageConfig,
        decoder_cfg: DecoderConfig,
        lr: float,
        tb_log_fn: TensorboardLogFn | None,
    ) -> None:
        super().__init__()
        self.align_cfg = align_cfg
        self.mesh_cfg = mesh_cfg
        self.decoder_cfg = decoder_cfg
        self.tb_log_fn = tb_log_fn
        self.chamfer = ChamferDistance()
        self.register_buffer("base", base.detach().clone())
        self.register_buffer("target_align", target_align.detach().clone())
        self.register_buffer("target_fit", target_fit.detach().clone())
        self.register_buffer("faces", faces.detach().clone())
        self.register_buffer("normal_pairs", normal_pairs.detach().clone())
        self.register_buffer("edge_index", edge_index.detach().clone())
        init_t, init_log_s, init_r = pca_initial_transform(target_all, template_radius=template_radius)
        translation = torch.as_tensor(init_t, dtype=torch.float32, device=base.device)
        log_scale = torch.as_tensor(init_log_s, dtype=torch.float32, device=base.device)
        rotvec = torch.as_tensor(init_r, dtype=torch.float32, device=base.device)
        if decoder_cfg.train_alignment:
            self.translation = torch.nn.Parameter(translation)
            self.log_scale = torch.nn.Parameter(log_scale)
            self.rotvec = torch.nn.Parameter(rotvec)
        else:
            self.register_buffer("translation", translation)
            self.register_buffer("log_scale", log_scale)
            self.register_buffer("rotvec", rotvec)

        stages = int(max(1, decoder_cfg.stages))
        offsets = list(float(x) for x in decoder_cfg.stage_max_offsets)
        if len(offsets) != stages:
            raise ValueError("decoder stage_max_offsets length must equal decoder stages")
        self.decoder = MeshDecoder(
            latent_dim=decoder_cfg.latent_dim,
            max_offset=max(offsets),
            stages=stages,
            local_feature_dims=[decoder_cfg.local_feature_dim] * stages,
            stage_hidden_dims=[decoder_cfg.hidden_dim] * stages,
            stage_max_offsets=offsets,
            num_gcn_blocks=decoder_cfg.num_blocks,
            graph_layer=decoder_cfg.graph_layer,
            edge_features=decoder_cfg.edge_features,
        )
        num_vertices = int(base.shape[0])
        self.latent0 = torch.nn.Parameter(torch.zeros((1, num_vertices, decoder_cfg.latent_dim), dtype=torch.float32))
        self.local_features = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros((1, num_vertices, decoder_cfg.local_feature_dim), dtype=torch.float32)
                )
                for _ in range(stages)
            ]
        )
        self.lr = float(lr)
        self.latest_aligned = self.base.detach()
        self.latest_stage_vertices: list[torch.Tensor] = [self.base.detach()]
        self.latest_parts: dict[str, torch.Tensor] = {}
        self.best_aligned = self.base.detach().clone()
        self.best_stage_vertices: list[torch.Tensor] = [self.base.detach().clone()]
        self.best_parts: dict[str, torch.Tensor] = {}
        self.best_step = 0
        self.best_final_chamfer = float("inf")

    def aligned_vertices(self) -> torch.Tensor:
        rot = rodrigues(self.rotvec)
        return (self.base * torch.exp(self.log_scale)[None, :]) @ rot.T + self.translation[None, :]

    def forward_stages(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        aligned = self.aligned_vertices()
        cur = aligned.unsqueeze(0)
        latent = self.latent0
        stage_vertices = []
        for stage_idx, local in enumerate(self.local_features):
            cur, latent, _gate = self.decoder.forward_stage(
                stage_idx,
                latent,
                local,
                cur,
                edge_index=self.edge_index,
            )
            stage_vertices.append(cur.squeeze(0))
        return aligned, stage_vertices

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch
        aligned, stage_vertices = self.forward_stages()
        final_vertices = stage_vertices[-1]
        mesh = MeshTensors(faces=self.faces, normal_pairs=self.normal_pairs)

        align_chamfer = self.chamfer(
            aligned.unsqueeze(0),
            self.target_align.unsqueeze(0),
            bidirectional=True,
            batch_reduction="mean",
            point_reduction="mean",
        )
        align_bbox = _bbox_loss(aligned, self.target_align)
        align_loss = self.align_cfg.lambda_chamfer * align_chamfer + self.align_cfg.lambda_bbox * align_bbox

        final_loss, final_parts = _weighted_deform_loss(
            final_vertices,
            self.target_fit,
            mesh,
            self.mesh_cfg,
            chamfer=self.chamfer,
            num_pred_samples=self.mesh_cfg.target_points,
            reference=aligned.detach(),
        )
        stage_loss = final_loss.new_zeros(())
        if self.decoder_cfg.stage_loss_weight > 0 and len(stage_vertices) > 1:
            stage_losses = []
            for vertices in stage_vertices[:-1]:
                s_loss, _s_parts = _weighted_deform_loss(
                    vertices,
                    self.target_fit,
                    mesh,
                    self.mesh_cfg,
                    chamfer=self.chamfer,
                    num_pred_samples=self.mesh_cfg.target_points,
                    reference=aligned.detach(),
                )
                stage_losses.append(s_loss)
            stage_loss = torch.stack(stage_losses).mean()

        total = final_loss + self.decoder_cfg.stage_loss_weight * stage_loss
        if self.decoder_cfg.train_alignment:
            total = total + align_loss
        parts: dict[str, torch.Tensor] = {
            "loss": total,
            "align_loss": align_loss,
            "align_chamfer": align_chamfer,
            "align_bbox": align_bbox,
            "final_loss": final_loss,
            "stage_loss": stage_loss,
            "scale_x": torch.exp(self.log_scale)[0],
            "scale_y": torch.exp(self.log_scale)[1],
            "scale_z": torch.exp(self.log_scale)[2],
        }
        for key, value in final_parts.items():
            parts[f"final_{key}"] = value

        self.latest_aligned = aligned.detach()
        self.latest_stage_vertices = [v.detach() for v in stage_vertices]
        self.latest_parts = {key: value.detach() for key, value in parts.items()}
        chamfer_for_selection = final_parts.get("chamfer")
        if chamfer_for_selection is not None:
            chamfer_value = float(chamfer_for_selection.detach().cpu().item())
            if np.isfinite(chamfer_value) and chamfer_value < self.best_final_chamfer:
                self.best_final_chamfer = chamfer_value
                self.best_step = int(batch_idx) + 1
                self.best_aligned = aligned.detach().clone()
                self.best_stage_vertices = [v.detach().clone() for v in stage_vertices]
                self.best_parts = {key: value.detach().clone() for key, value in parts.items()}
        parts["best_final_chamfer"] = final_loss.new_tensor(self.best_final_chamfer)
        for name, value in parts.items():
            self.log(f"mesh/{name}", value, on_step=True, prog_bar=name in {"loss", "final_chamfer"})
        _log_tensorboard(self.tb_log_fn, "mesh", batch_idx + 1, parts)
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def _weighted_deform_loss(
    vertices: torch.Tensor,
    target: torch.Tensor,
    mesh: MeshTensors,
    cfg: StageConfig,
    chamfer: ChamferDistance,
    num_pred_samples: int,
    reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    chamfer_loss = chamferdist_surface_loss(
        vertices=vertices,
        target_points=target,
        faces=mesh.faces,
        chamfer=chamfer,
        num_pred_samples=num_pred_samples,
    )
    bbox = _bbox_loss(vertices, target) if cfg.lambda_bbox > 0 else vertices.new_zeros(())
    vertex_chamfer = vertices.new_zeros(())
    if cfg.lambda_vertex_chamfer > 0:
        vertex_chamfer = chamfer(
            vertices.unsqueeze(0),
            target.unsqueeze(0),
            bidirectional=True,
            batch_reduction="mean",
            point_reduction="mean",
        )
    deform = vertices.new_zeros(())
    if cfg.lambda_deform > 0 and reference is not None:
        deform = torch.mean((vertices - reference) ** 2)
    reg = multigeomed_regularization_parts_for_lambdas(
        vertices,
        mesh.faces,
        lambda_edge=cfg.lambda_edge,
        lambda_laplacian=cfg.lambda_laplacian,
        lambda_normal=cfg.lambda_normal,
        lambda_face_area=cfg.lambda_face_area,
        lambda_face_area_var=cfg.lambda_face_area_var,
        normal_pairs=mesh.normal_pairs,
    )
    total = (
        cfg.lambda_chamfer * chamfer_loss
        + cfg.lambda_bbox * bbox
        + cfg.lambda_vertex_chamfer * vertex_chamfer
        + cfg.lambda_deform * deform
        + cfg.lambda_edge * reg["edge"]
        + cfg.lambda_laplacian * reg["laplacian"]
        + cfg.lambda_normal * reg["normal"]
        + cfg.lambda_face_area * reg["face_area"]
        + cfg.lambda_face_area_var * reg["face_area_var"]
    )
    parts = {
        "loss": total,
        "chamfer": chamfer_loss,
        "bbox": bbox,
        "vertex_chamfer": vertex_chamfer,
        "deform": deform,
        **reg,
    }
    return total, parts


def _run_alignment(
    base: torch.Tensor,
    target_all: np.ndarray,
    template_radius: float,
    align_cfg: StageConfig,
    sampling_mode: str,
    fps_candidate_points: int,
    seed: int,
    device: torch.device,
    chamfer: ChamferDistance,
    status_fn: Callable[[str], None],
    tb_log_fn: TensorboardLogFn | None,
) -> tuple[torch.Tensor, StageResult]:
    target_align_np = sample_points(
        target_all,
        align_cfg.target_points,
        sampling_mode,
        seed,
        device,
        fps_candidate_points=fps_candidate_points,
    )
    target_align = torch.as_tensor(target_align_np, dtype=torch.float32, device=device)

    init_t, init_log_s, init_r = pca_initial_transform(target_all, template_radius=template_radius)
    translation = torch.nn.Parameter(torch.as_tensor(init_t, dtype=torch.float32, device=device))
    log_scale = torch.nn.Parameter(torch.as_tensor(init_log_s, dtype=torch.float32, device=device))
    rotvec = torch.nn.Parameter(torch.as_tensor(init_r, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([translation, log_scale, rotvec], lr=align_cfg.lr)

    align_parts: dict[str, torch.Tensor] = {}
    aligned = base
    for step in range(max(align_cfg.steps, 1)):
        optimizer.zero_grad(set_to_none=True)
        rot = rodrigues(rotvec)
        aligned = (base * torch.exp(log_scale)[None, :]) @ rot.T + translation[None, :]
        chamfer_loss = chamfer(
            aligned.unsqueeze(0),
            target_align.unsqueeze(0),
            bidirectional=True,
            batch_reduction="mean",
            point_reduction="mean",
        )
        bbox = _bbox_loss(aligned, target_align)
        loss = align_cfg.lambda_chamfer * chamfer_loss + align_cfg.lambda_bbox * bbox
        loss.backward()
        optimizer.step()
        align_parts = {
            "loss": loss,
            "chamfer": chamfer_loss,
            "bbox": bbox,
            "scale_x": torch.exp(log_scale)[0],
            "scale_y": torch.exp(log_scale)[1],
            "scale_z": torch.exp(log_scale)[2],
        }
        if align_cfg.log_every > 0 and (step == 0 or (step + 1) % align_cfg.log_every == 0):
            status_fn(f"[{align_cfg.name}] step {step + 1}/{align_cfg.steps}: loss={float(loss.detach().cpu()):.6f}")
            _log_tensorboard(tb_log_fn, align_cfg.name, step + 1, align_parts)

    return aligned.detach(), StageResult(
        name=align_cfg.name,
        vertices_norm=aligned.detach().cpu().numpy().astype(np.float32),
        metrics=_tensor_metrics(align_parts),
    )


def _run_alignment_lightning(
    base: torch.Tensor,
    target_all: np.ndarray,
    template_radius: float,
    align_cfg: StageConfig,
    sampling_mode: str,
    fps_candidate_points: int,
    seed: int,
    device: torch.device,
    tb_log_fn: TensorboardLogFn | None,
    enable_progress_bar: bool,
) -> tuple[torch.Tensor, StageResult]:
    target_align_np = sample_points(
        target_all,
        align_cfg.target_points,
        sampling_mode,
        seed,
        device,
        fps_candidate_points=fps_candidate_points,
    )
    target_align = torch.as_tensor(target_align_np, dtype=torch.float32, device=device)
    module = _AlignLightningModule(
        base=base,
        target=target_align,
        template_radius=template_radius,
        target_all=target_all,
        cfg=align_cfg,
        tb_log_fn=tb_log_fn,
    )
    _fit_lightning_stage(
        module,
        steps=align_cfg.steps,
        device=device,
        enable_progress_bar=enable_progress_bar,
    )
    vertices = module.vertices().detach()
    return vertices, StageResult(
        name=align_cfg.name,
        vertices_norm=vertices.cpu().numpy().astype(np.float32),
        metrics=_tensor_metrics(module.latest_parts),
    )


def _run_direct_deformation(
    aligned: torch.Tensor,
    faces: np.ndarray,
    target_all: np.ndarray,
    coarse_cfg: StageConfig,
    detail_cfg: StageConfig,
    sampling_mode: str,
    fps_candidate_points: int,
    seed: int,
    device: torch.device,
    chamfer: ChamferDistance,
    status_fn: Callable[[str], None],
    tb_log_fn: TensorboardLogFn | None,
) -> list[StageResult]:
    current = aligned.detach()
    mesh = _make_mesh_tensors(faces, device)
    results: list[StageResult] = []
    for stage_seed_offset, cfg in enumerate([coarse_cfg, detail_cfg], start=1):
        target_np = sample_points(
            target_all,
            cfg.target_points,
            sampling_mode,
            seed + stage_seed_offset,
            device,
            fps_candidate_points=fps_candidate_points,
        )
        target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
        reference = current.detach()
        offsets = torch.nn.Parameter(torch.zeros_like(reference))
        optimizer = torch.optim.Adam([offsets], lr=cfg.lr)
        parts: dict[str, torch.Tensor] = {}
        vertices = reference
        for step in range(max(cfg.steps, 1)):
            optimizer.zero_grad(set_to_none=True)
            vertices = reference + offsets
            loss, parts = _weighted_deform_loss(
                vertices,
                target,
                mesh,
                cfg,
                chamfer=chamfer,
                num_pred_samples=cfg.target_points,
                reference=reference,
            )
            loss.backward()
            optimizer.step()
            if cfg.log_every > 0 and (step == 0 or (step + 1) % cfg.log_every == 0):
                status_fn(f"[{cfg.name}] step {step + 1}/{cfg.steps}: loss={float(loss.detach().cpu()):.6f}")
                _log_tensorboard(tb_log_fn, cfg.name, step + 1, parts)
        current = vertices.detach()
        results.append(
            StageResult(
                name=cfg.name,
                vertices_norm=current.cpu().numpy().astype(np.float32),
                metrics={**asdict(cfg), **_tensor_metrics(parts), "deformation_mode": 0.0},
            )
        )
    return results


def _run_direct_deformation_lightning(
    aligned: torch.Tensor,
    faces: np.ndarray,
    target_all: np.ndarray,
    coarse_cfg: StageConfig,
    detail_cfg: StageConfig,
    sampling_mode: str,
    fps_candidate_points: int,
    seed: int,
    device: torch.device,
    tb_log_fn: TensorboardLogFn | None,
    enable_progress_bar: bool,
) -> list[StageResult]:
    current = aligned.detach()
    mesh = _make_mesh_tensors(faces, device)
    faces_t = mesh.faces
    normal_pairs_t = mesh.normal_pairs
    results: list[StageResult] = []
    for stage_seed_offset, cfg in enumerate([coarse_cfg, detail_cfg], start=1):
        target_np = sample_points(
            target_all,
            cfg.target_points,
            sampling_mode,
            seed + stage_seed_offset,
            device,
            fps_candidate_points=fps_candidate_points,
        )
        target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
        module = _DirectDeformLightningModule(
            reference=current,
            target=target,
            faces=faces_t,
            normal_pairs=normal_pairs_t,
            cfg=cfg,
            tb_log_fn=tb_log_fn,
        )
        _fit_lightning_stage(
            module,
            steps=cfg.steps,
            device=device,
            enable_progress_bar=enable_progress_bar,
        )
        current = module.vertices().detach()
        results.append(
            StageResult(
                name=cfg.name,
                vertices_norm=current.cpu().numpy().astype(np.float32),
                metrics={**asdict(cfg), **_tensor_metrics(module.latest_parts), "deformation_mode": 0.0},
            )
        )
    return results


def _run_decoder_deformation(
    aligned: torch.Tensor,
    faces: np.ndarray,
    edge_index: np.ndarray,
    target_all: np.ndarray,
    coarse_cfg: StageConfig,
    detail_cfg: StageConfig,
    decoder_cfg: DecoderConfig,
    sampling_mode: str,
    fps_candidate_points: int,
    seed: int,
    device: torch.device,
    chamfer: ChamferDistance,
    status_fn: Callable[[str], None],
    tb_log_fn: TensorboardLogFn | None,
) -> list[StageResult]:
    mesh = _make_mesh_tensors(faces, device)
    edge_index_t = torch.as_tensor(edge_index.astype(np.int64, copy=False), dtype=torch.long, device=device)
    num_vertices = int(aligned.shape[0])

    decoder = MeshDecoder(
        latent_dim=decoder_cfg.latent_dim,
        max_offset=max(float(decoder_cfg.coarse_max_offset), float(decoder_cfg.detail_max_offset)),
        stages=2,
        local_feature_dims=[decoder_cfg.local_feature_dim, decoder_cfg.local_feature_dim],
        stage_hidden_dims=[decoder_cfg.hidden_dim, decoder_cfg.hidden_dim],
        stage_max_offsets=[decoder_cfg.coarse_max_offset, decoder_cfg.detail_max_offset],
        num_gcn_blocks=decoder_cfg.num_blocks,
        graph_layer=decoder_cfg.graph_layer,
        edge_features=decoder_cfg.edge_features,
    ).to(device)
    latent0 = torch.nn.Parameter(torch.zeros((1, num_vertices, decoder_cfg.latent_dim), dtype=torch.float32, device=device))
    local0 = torch.nn.Parameter(
        torch.zeros((1, num_vertices, decoder_cfg.local_feature_dim), dtype=torch.float32, device=device)
    )
    local1 = torch.nn.Parameter(
        torch.zeros((1, num_vertices, decoder_cfg.local_feature_dim), dtype=torch.float32, device=device)
    )
    params = list(decoder.parameters()) + [latent0, local0, local1]
    optimizer = torch.optim.Adam(params, lr=coarse_cfg.lr)
    reference = aligned.detach().unsqueeze(0)

    def forward_two_stages() -> tuple[torch.Tensor, torch.Tensor]:
        coarse_vertices, coarse_latent, _gate0 = decoder.forward_stage(
            0,
            latent0,
            local0,
            reference,
            edge_index=edge_index_t,
        )
        detail_vertices, _detail_latent, _gate1 = decoder.forward_stage(
            1,
            coarse_latent,
            local1,
            coarse_vertices,
            edge_index=edge_index_t,
        )
        return coarse_vertices.squeeze(0), detail_vertices.squeeze(0)

    target_coarse = torch.as_tensor(
        sample_points(target_all, coarse_cfg.target_points, sampling_mode, seed + 1, device, fps_candidate_points),
        dtype=torch.float32,
        device=device,
    )
    coarse_parts: dict[str, torch.Tensor] = {}
    coarse_vertices = aligned.detach()
    best_coarse_vertices = coarse_vertices
    best_coarse_parts: dict[str, torch.Tensor] = {}
    best_coarse_chamfer = float("inf")
    best_coarse_step = 0
    for step in range(max(coarse_cfg.steps, 1)):
        optimizer.zero_grad(set_to_none=True)
        coarse_vertices, _detail_vertices = forward_two_stages()
        loss, coarse_parts = _weighted_deform_loss(
            coarse_vertices,
            target_coarse,
            mesh,
            coarse_cfg,
            chamfer=chamfer,
            num_pred_samples=coarse_cfg.target_points,
            reference=aligned.detach(),
        )
        loss.backward()
        optimizer.step()
        coarse_chamfer = float(coarse_parts["chamfer"].detach().cpu().item())
        if np.isfinite(coarse_chamfer) and coarse_chamfer < best_coarse_chamfer:
            best_coarse_chamfer = coarse_chamfer
            best_coarse_step = step + 1
            best_coarse_vertices = coarse_vertices.detach().clone()
            best_coarse_parts = {key: value.detach().clone() for key, value in coarse_parts.items()}
        if coarse_cfg.log_every > 0 and (step == 0 or (step + 1) % coarse_cfg.log_every == 0):
            status_fn(f"[{coarse_cfg.name}] step {step + 1}/{coarse_cfg.steps}: loss={float(loss.detach().cpu()):.6f}")
            _log_tensorboard(tb_log_fn, coarse_cfg.name, step + 1, coarse_parts)

    coarse_result = StageResult(
        name=coarse_cfg.name,
        vertices_norm=best_coarse_vertices.detach().cpu().numpy().astype(np.float32),
        metrics={
            **asdict(coarse_cfg),
            **asdict(decoder_cfg),
            **_tensor_metrics(best_coarse_parts or coarse_parts),
            "mesh_selection": "best_chamfer",
            "best_step": float(best_coarse_step),
            "best_chamfer": float(best_coarse_chamfer),
        },
    )

    optimizer = torch.optim.Adam(params, lr=detail_cfg.lr)
    target_detail = torch.as_tensor(
        sample_points(target_all, detail_cfg.target_points, sampling_mode, seed + 2, device, fps_candidate_points),
        dtype=torch.float32,
        device=device,
    )
    detail_parts: dict[str, torch.Tensor] = {}
    detail_vertices = coarse_vertices.detach()
    best_detail_vertices = detail_vertices
    best_detail_parts: dict[str, torch.Tensor] = {}
    best_detail_chamfer = float("inf")
    best_detail_step = 0
    for step in range(max(detail_cfg.steps, 1)):
        optimizer.zero_grad(set_to_none=True)
        coarse_vertices, detail_vertices = forward_two_stages()
        detail_loss, detail_parts = _weighted_deform_loss(
            detail_vertices,
            target_detail,
            mesh,
            detail_cfg,
            chamfer=chamfer,
            num_pred_samples=detail_cfg.target_points,
            reference=aligned.detach(),
        )
        if decoder_cfg.stage_loss_weight > 0:
            coarse_loss, _coarse_parts = _weighted_deform_loss(
                coarse_vertices,
                target_detail,
                mesh,
                detail_cfg,
                chamfer=chamfer,
                num_pred_samples=detail_cfg.target_points,
                reference=aligned.detach(),
            )
            loss = detail_loss + decoder_cfg.stage_loss_weight * coarse_loss
        else:
            loss = detail_loss
        loss.backward()
        optimizer.step()
        detail_parts = {**detail_parts, "loss": loss, "detail_loss": detail_loss}
        detail_chamfer = float(detail_parts["chamfer"].detach().cpu().item())
        if np.isfinite(detail_chamfer) and detail_chamfer < best_detail_chamfer:
            best_detail_chamfer = detail_chamfer
            best_detail_step = step + 1
            best_detail_vertices = detail_vertices.detach().clone()
            best_detail_parts = {key: value.detach().clone() for key, value in detail_parts.items()}
        if detail_cfg.log_every > 0 and (step == 0 or (step + 1) % detail_cfg.log_every == 0):
            status_fn(f"[{detail_cfg.name}] step {step + 1}/{detail_cfg.steps}: loss={float(loss.detach().cpu()):.6f}")
            _log_tensorboard(tb_log_fn, detail_cfg.name, step + 1, detail_parts)

    detail_result = StageResult(
        name=detail_cfg.name,
        vertices_norm=best_detail_vertices.detach().cpu().numpy().astype(np.float32),
        metrics={
            **asdict(detail_cfg),
            **asdict(decoder_cfg),
            **_tensor_metrics(best_detail_parts or detail_parts),
            "mesh_selection": "best_chamfer",
            "best_step": float(best_detail_step),
            "best_chamfer": float(best_detail_chamfer),
        },
    )
    return [coarse_result, detail_result]


def _run_decoder_deformation_lightning(
    aligned: torch.Tensor,
    faces: np.ndarray,
    edge_index: np.ndarray,
    target_all: np.ndarray,
    coarse_cfg: StageConfig,
    detail_cfg: StageConfig,
    decoder_cfg: DecoderConfig,
    sampling_mode: str,
    fps_candidate_points: int,
    seed: int,
    device: torch.device,
    tb_log_fn: TensorboardLogFn | None,
    enable_progress_bar: bool,
) -> list[StageResult]:
    mesh = _make_mesh_tensors(faces, device)
    faces_t = mesh.faces
    normal_pairs_t = mesh.normal_pairs
    edge_index_t = torch.as_tensor(edge_index.astype(np.int64, copy=False), dtype=torch.long, device=device)
    module = _DecoderDeformLightningModule(
        aligned=aligned,
        faces=faces_t,
        normal_pairs=normal_pairs_t,
        edge_index=edge_index_t,
        decoder_cfg=decoder_cfg,
        tb_log_fn=tb_log_fn,
    )

    target_coarse = torch.as_tensor(
        sample_points(target_all, coarse_cfg.target_points, sampling_mode, seed + 1, device, fps_candidate_points),
        dtype=torch.float32,
        device=device,
    )
    module.set_stage("coarse", coarse_cfg, target_coarse)
    _fit_lightning_stage(
        module,
        steps=coarse_cfg.steps,
        device=device,
        enable_progress_bar=enable_progress_bar,
    )
    coarse_vertices = module.latest_coarse_vertices.detach()
    coarse_result = StageResult(
        name=coarse_cfg.name,
        vertices_norm=coarse_vertices.cpu().numpy().astype(np.float32),
        metrics={**asdict(coarse_cfg), **asdict(decoder_cfg), **_tensor_metrics(module.latest_parts)},
    )

    target_detail = torch.as_tensor(
        sample_points(target_all, detail_cfg.target_points, sampling_mode, seed + 2, device, fps_candidate_points),
        dtype=torch.float32,
        device=device,
    )
    module.set_stage("detail", detail_cfg, target_detail)
    _fit_lightning_stage(
        module,
        steps=detail_cfg.steps,
        device=device,
        enable_progress_bar=enable_progress_bar,
    )
    detail_vertices = module.latest_detail_vertices.detach()
    detail_result = StageResult(
        name=detail_cfg.name,
        vertices_norm=detail_vertices.cpu().numpy().astype(np.float32),
        metrics={**asdict(detail_cfg), **asdict(decoder_cfg), **_tensor_metrics(module.latest_parts)},
    )
    return [coarse_result, detail_result]


def _run_four_stage_decoder_lightning(
    base: torch.Tensor,
    faces: np.ndarray,
    edge_index: np.ndarray,
    target_all: np.ndarray,
    template_radius: float,
    align_cfg: StageConfig,
    mesh_cfg: StageConfig,
    decoder_cfg: DecoderConfig,
    sampling_mode: str,
    fps_candidate_points: int,
    seed: int,
    device: torch.device,
    tb_log_fn: TensorboardLogFn | None,
    enable_progress_bar: bool,
    init_fit_state_path: Path | None,
    init_fit_state_load_alignment: bool,
    save_fit_state_path: Path | None,
) -> list[StageResult]:
    target_align = torch.as_tensor(
        sample_points(target_all, align_cfg.target_points, sampling_mode, seed, device, fps_candidate_points),
        dtype=torch.float32,
        device=device,
    )
    target_fit = torch.as_tensor(
        sample_points(target_all, mesh_cfg.target_points, sampling_mode, seed + 1, device, fps_candidate_points),
        dtype=torch.float32,
        device=device,
    )
    mesh = _make_mesh_tensors(faces, device)
    faces_t = mesh.faces
    normal_pairs_t = mesh.normal_pairs
    edge_index_t = torch.as_tensor(edge_index.astype(np.int64, copy=False), dtype=torch.long, device=device)
    module = _FourStageDecoderFitModule(
        base=base,
        target_align=target_align,
        target_fit=target_fit,
        target_all=target_all,
        template_radius=template_radius,
        faces=faces_t,
        normal_pairs=normal_pairs_t,
        edge_index=edge_index_t,
        align_cfg=align_cfg,
        mesh_cfg=mesh_cfg,
        decoder_cfg=decoder_cfg,
        lr=mesh_cfg.lr,
        tb_log_fn=tb_log_fn,
    )
    init_state_info: dict[str, object] | None = None
    if init_fit_state_path is not None:
        init_state_info = _load_matching_fit_state(
            module,
            init_fit_state_path,
            load_alignment=init_fit_state_load_alignment,
        )
        print(
            "Loaded warm-start fit state: "
            f"{init_fit_state_path} ({init_state_info['num_loaded_keys']} tensors)"
        )
    total_steps = int(decoder_cfg.fit_steps) if int(decoder_cfg.fit_steps) > 0 else (
        int(max(0, align_cfg.steps)) + int(max(0, mesh_cfg.steps))
    )
    _fit_lightning_stage(
        module,
        steps=max(total_steps, 1),
        device=device,
        enable_progress_bar=enable_progress_bar,
    )
    selected_aligned = module.best_aligned if module.best_step > 0 else module.latest_aligned
    selected_stage_vertices = module.best_stage_vertices if module.best_step > 0 else module.latest_stage_vertices
    selected_parts = module.best_parts if module.best_step > 0 else module.latest_parts
    if save_fit_state_path is not None:
        save_fit_state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "v24_mask_mesh_fit_state_v1",
                "state_dict": module.state_dict(),
                "decoder_config": asdict(decoder_cfg),
                "best_step": int(module.best_step),
                "best_final_chamfer": float(module.best_final_chamfer),
                "loaded_from": init_state_info,
            },
            str(save_fit_state_path),
        )
    selected_metrics = _tensor_metrics(selected_parts)
    selected_metrics.update(
        {
            "mesh_selection": "best_final_chamfer",
            "best_step": float(module.best_step),
            "best_final_chamfer": float(module.best_final_chamfer),
            "total_fit_steps": float(max(total_steps, 1)),
            "fit_state_saved": str(save_fit_state_path) if save_fit_state_path is not None else "",
            "fit_state_loaded": str(init_fit_state_path) if init_fit_state_path is not None else "",
            "fit_state_loaded_tensors": float(init_state_info["num_loaded_keys"]) if init_state_info else 0.0,
        }
    )
    results = [
        StageResult(
            name="align",
            vertices_norm=selected_aligned.cpu().numpy().astype(np.float32),
            metrics=selected_metrics,
        )
    ]
    for stage_idx, vertices in enumerate(selected_stage_vertices):
        name = "detail" if stage_idx == len(selected_stage_vertices) - 1 else f"stage{stage_idx + 1}"
        results.append(
            StageResult(
                name=name,
                vertices_norm=vertices.cpu().numpy().astype(np.float32),
                metrics={
                    **asdict(mesh_cfg),
                    **asdict(decoder_cfg),
                    **selected_metrics,
                    "stage_index": float(stage_idx),
                },
            )
        )
    return results


def fit_mesh_to_target(
    template_vertices_norm: np.ndarray,
    faces: np.ndarray,
    edge_index: np.ndarray,
    target_points_norm: np.ndarray,
    align_cfg: StageConfig,
    coarse_cfg: StageConfig,
    detail_cfg: StageConfig,
    sampling_mode: str,
    fps_candidate_points: int,
    seed: int,
    device: torch.device,
    decoder_cfg: DecoderConfig,
    optimization_loop: str = "lightning",
    enable_progress_bar: bool = True,
    status_fn: Callable[[str], None] = print,
    tb_log_fn: TensorboardLogFn | None = None,
    init_fit_state_path: Path | None = None,
    init_fit_state_load_alignment: bool = False,
    save_fit_state_path: Path | None = None,
) -> tuple[list[StageResult], dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    chamfer = ChamferDistance()
    base = torch.as_tensor(template_vertices_norm, dtype=torch.float32, device=device)
    template_radius = float(np.linalg.norm(template_vertices_norm, axis=1).max())
    target_all = target_points_norm.astype(np.float32, copy=False)

    loop = str(optimization_loop).strip().lower()
    if loop not in {"lightning", "manual"}:
        raise ValueError("--optimization-loop must be 'lightning' or 'manual'")

    mode = str(decoder_cfg.deformation_mode).strip().lower()
    if mode != "decoder":
        raise ValueError("v24 mask mesh fitting now supports only --deformation-mode decoder")
    if loop == "lightning" and mode == "decoder":
        results = _run_four_stage_decoder_lightning(
            base=base,
            faces=faces,
            edge_index=edge_index,
            target_all=target_all,
            template_radius=template_radius,
            align_cfg=align_cfg,
            mesh_cfg=replace(detail_cfg, steps=int(max(0, coarse_cfg.steps)) + int(max(0, detail_cfg.steps))),
            decoder_cfg=decoder_cfg,
            sampling_mode=sampling_mode,
            fps_candidate_points=fps_candidate_points,
            seed=seed,
            device=device,
            tb_log_fn=tb_log_fn,
            enable_progress_bar=enable_progress_bar,
            init_fit_state_path=init_fit_state_path,
            init_fit_state_load_alignment=init_fit_state_load_alignment,
            save_fit_state_path=save_fit_state_path,
        )
    elif loop == "lightning":
        aligned, align_result = _run_alignment_lightning(
            base,
            target_all,
            template_radius,
            align_cfg,
            sampling_mode,
            fps_candidate_points,
            seed,
            device,
            tb_log_fn,
            enable_progress_bar,
        )
        results = [align_result]
    else:
        aligned, align_result = _run_alignment(
            base,
            target_all,
            template_radius,
            align_cfg,
            sampling_mode,
            fps_candidate_points,
            seed,
            device,
            chamfer,
            status_fn,
            tb_log_fn,
        )
        results = [align_result]

    if mode == "direct":
        if loop == "lightning":
            results.extend(
                _run_direct_deformation_lightning(
                    aligned,
                    faces,
                    target_all,
                    coarse_cfg,
                    detail_cfg,
                    sampling_mode,
                    fps_candidate_points,
                    seed,
                    device,
                    tb_log_fn,
                    enable_progress_bar,
                )
            )
        else:
            results.extend(
                _run_direct_deformation(
                    aligned,
                    faces,
                    target_all,
                    coarse_cfg,
                    detail_cfg,
                    sampling_mode,
                    fps_candidate_points,
                    seed,
                    device,
                    chamfer,
                    status_fn,
                    tb_log_fn,
                )
            )
    elif mode == "decoder":
        if loop == "lightning":
            pass
        else:
            results.extend(
                _run_decoder_deformation(
                    aligned,
                    faces,
                    edge_index,
                    target_all,
                    coarse_cfg,
                    detail_cfg,
                    decoder_cfg,
                    sampling_mode,
                    fps_candidate_points,
                    seed,
                    device,
                    chamfer,
                    status_fn,
                    tb_log_fn,
                )
            )
    else:
        raise ValueError("v24 mask mesh fitting now supports only --deformation-mode decoder")

    final_vertices = torch.as_tensor(results[-1].vertices_norm, dtype=torch.float32, device=device)
    final_target = torch.as_tensor(
        sample_points(target_all, detail_cfg.target_points, sampling_mode, seed + 99, device, fps_candidate_points),
        dtype=torch.float32,
        device=device,
    )
    mesh = _make_mesh_tensors(faces, device)
    final_chamfer = chamferdist_surface_loss(
        vertices=final_vertices,
        target_points=final_target,
        faces=mesh.faces,
        chamfer=chamfer,
        num_pred_samples=detail_cfg.target_points,
    )
    final_metrics = {
        "final_chamfer": float(final_chamfer.detach().cpu().item()),
        "num_vertices": float(template_vertices_norm.shape[0]),
        "num_faces": float(faces.shape[0]),
        "deformation_mode": mode,
        "chamfer_backend": "chamferdist.ChamferDistance",
        "regularizer_backend": "multigeomed.objectives.objectives_for_surface_mesh",
        "optimization_loop": loop,
    }
    return results, final_metrics
