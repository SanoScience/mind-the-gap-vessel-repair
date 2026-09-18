from __future__ import annotations

"""v24-local copy of the v23 graph mesh deformation decoder.

This keeps the per-case v24 mesh fitter self-contained while preserving the
decoder architecture that worked well in v23.
"""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GENConv, GINEConv, GraphConv, SAGEConv, TransformerConv


GRAPH_LAYER_CHOICES = ("gcn", "sage", "graph", "gen", "gine", "transformer", "gatv2")
EDGE_FEATURE_CHOICES = ("none", "geometry")
EDGE_AWARE_GRAPH_LAYERS = {"gine", "gen", "transformer", "gatv2"}
EDGE_GEOMETRY_DIM = 7


def _to_undirected_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.dim() == 3:
        edge_index = edge_index[0]
    edge_index = edge_index.long()
    if edge_index.numel() == 0:
        return edge_index
    rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    edge_index = torch.unique(edge_index.t(), dim=0).t().contiguous()
    return edge_index


def _expand_edge_index(edge_index: torch.Tensor, batch_size: int, num_vertices: int) -> torch.Tensor:
    if batch_size == 1:
        return edge_index
    offsets = torch.arange(batch_size, device=edge_index.device).view(-1, 1, 1) * num_vertices
    batched = edge_index.unsqueeze(0) + offsets
    return batched.permute(1, 0, 2).reshape(2, -1)


def _canonical_graph_layer(name: str) -> str:
    layer = str(name).strip().lower()
    if layer not in GRAPH_LAYER_CHOICES:
        raise ValueError(f"mesh graph layer must be one of {GRAPH_LAYER_CHOICES}, got {name!r}")
    return layer


def _canonical_edge_features(name: str) -> str:
    edge_features = str(name).strip().lower()
    if edge_features not in EDGE_FEATURE_CHOICES:
        raise ValueError(f"mesh edge features must be one of {EDGE_FEATURE_CHOICES}, got {name!r}")
    return edge_features


def _make_graph_layer(
    graph_layer: str,
    in_dim: int,
    out_dim: int,
    *,
    edge_feature_dim: int | None = None,
) -> nn.Module:
    graph_layer = _canonical_graph_layer(graph_layer)
    in_dim = int(in_dim)
    out_dim = int(out_dim)
    if graph_layer == "gcn":
        return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
    if graph_layer == "sage":
        return SAGEConv(in_dim, out_dim)
    if graph_layer == "graph":
        return GraphConv(in_dim, out_dim)
    if graph_layer == "gen":
        return GENConv(in_dim, out_dim, edge_dim=edge_feature_dim)
    if graph_layer == "gine":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        return GINEConv(mlp, edge_dim=edge_feature_dim)
    if graph_layer == "transformer":
        return TransformerConv(in_dim, out_dim, heads=1, concat=True, edge_dim=edge_feature_dim)
    if graph_layer == "gatv2":
        return GATv2Conv(in_dim, out_dim, heads=1, concat=True, edge_dim=edge_feature_dim)
    raise AssertionError(f"Unhandled graph layer: {graph_layer}")


def _apply_graph_layer(
    layer: nn.Module,
    graph_layer: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
) -> torch.Tensor:
    graph_layer = _canonical_graph_layer(graph_layer)
    if graph_layer in EDGE_AWARE_GRAPH_LAYERS:
        return layer(x, edge_index, edge_attr=edge_attr)
    return layer(x, edge_index)


def _edge_geometry_attr(vertices: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor | None:
    if edge_index is None or edge_index.numel() == 0:
        return None
    if edge_index.dim() == 3:
        edge_index = edge_index[0]
    src = edge_index[0].long()
    dst = edge_index[1].long()
    delta = vertices[:, dst, :] - vertices[:, src, :]
    length = torch.linalg.norm(delta, dim=-1, keepdim=True)
    unit = delta / length.clamp_min(1e-8)
    edge_attr = torch.cat([delta, length, unit], dim=-1)
    return edge_attr.reshape(-1, EDGE_GEOMETRY_DIM).contiguous()


class _GraphResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        graph_layer: str = "gcn",
        edge_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.graph_layer = _canonical_graph_layer(graph_layer)
        self.graph1 = _make_graph_layer(
            self.graph_layer,
            hidden_dim,
            hidden_dim,
            edge_feature_dim=edge_feature_dim,
        )
        self.graph2 = _make_graph_layer(
            self.graph_layer,
            hidden_dim,
            hidden_dim,
            edge_feature_dim=edge_feature_dim,
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        if edge_index is None:
            out = F.gelu(self.norm1(self.linear1(x)))
            out = F.gelu(self.norm2(self.linear2(out)))
        else:
            out = F.gelu(_apply_graph_layer(self.graph1, self.graph_layer, x, edge_index, edge_attr))
            out = self.norm1(out)
            out = F.gelu(_apply_graph_layer(self.graph2, self.graph_layer, out, edge_index, edge_attr))
            out = self.norm2(out)
        return 0.5 * (residual + out)


class _MeshRefineStage(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        local_feature_dim: int,
        hidden_dim: int,
        num_blocks: int = 3,
        graph_layer: str = "gcn",
        edge_features: str = "none",
    ) -> None:
        super().__init__()
        in_dim = int(latent_dim) + int(local_feature_dim) + 3
        hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.local_feature_dim = int(local_feature_dim)
        self.hidden_dim = hidden_dim
        self.graph_layer = _canonical_graph_layer(graph_layer)
        self.edge_features = _canonical_edge_features(edge_features)
        edge_feature_dim = EDGE_GEOMETRY_DIM if self.edge_features == "geometry" else None
        self.input_graph = _make_graph_layer(
            self.graph_layer,
            in_dim,
            hidden_dim,
            edge_feature_dim=edge_feature_dim,
        )
        self.input_linear = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.res_blocks = nn.ModuleList(
            [
                _GraphResidualBlock(
                    hidden_dim,
                    graph_layer=self.graph_layer,
                    edge_feature_dim=edge_feature_dim,
                )
                for _ in range(max(0, int(num_blocks)))
            ]
        )
        self.offset_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        latent: torch.Tensor,
        local_ct: torch.Tensor,
        vertices: torch.Tensor,
        edge_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.shape[-1] != self.latent_dim:
            raise RuntimeError(
                f"Expected latent dim {self.latent_dim} for mesh stage, got {latent.shape[-1]}"
            )
        if local_ct.shape[-1] != self.local_feature_dim:
            raise RuntimeError(
                f"Expected local feature dim {self.local_feature_dim} for mesh stage, got {local_ct.shape[-1]}"
            )
        x = torch.cat([latent, local_ct, vertices], dim=-1)
        if edge_index is None:
            x = F.gelu(self.input_norm(self.input_linear(x)))
            for block in self.res_blocks:
                x = block(x, None)
        else:
            batch_size, num_vertices, _ = x.shape
            x_flat = x.reshape(batch_size * num_vertices, -1)
            edge_attr = _edge_geometry_attr(vertices, edge_index) if self.edge_features == "geometry" else None
            edge_index = _expand_edge_index(edge_index.to(device=x.device), batch_size, num_vertices)
            x_flat = F.gelu(_apply_graph_layer(self.input_graph, self.graph_layer, x_flat, edge_index, edge_attr))
            x_flat = self.input_norm(x_flat)
            for block in self.res_blocks:
                x_flat = block(x_flat, edge_index, edge_attr=edge_attr)
            x = x_flat.view(batch_size, num_vertices, -1)
        offsets = self.offset_head(x)
        return x, offsets


class MeshDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        max_offset: float = 0.2,
        stages: int = 3,
        adaptive_refine: bool = True,
        local_feature_dims: list[int] | tuple[int, ...] | None = None,
        stage_hidden_dims: list[int] | tuple[int, ...] | None = None,
        stage_max_offsets: list[float] | tuple[float, ...] | None = None,
        num_gcn_blocks: int = 3,
        graph_layer: str = "gcn",
        edge_features: str = "none",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.initial_latent_dim = int(latent_dim)
        self.max_offset = float(max_offset)
        self.stages = int(max(1, stages))
        # Kept in the constructor for older config compatibility. The decoder
        # now applies direct bounded offsets without learned/adaptive gates.
        del adaptive_refine
        self.graph_layer = _canonical_graph_layer(graph_layer)
        self.edge_features = _canonical_edge_features(edge_features)
        if self.edge_features != "none" and self.graph_layer not in EDGE_AWARE_GRAPH_LAYERS:
            raise ValueError(
                "--mesh-edge-features geometry is only supported with "
                "gine, gen, transformer, or gatv2 graph layers"
            )
        if self.graph_layer == "gine" and self.edge_features == "none":
            raise ValueError("GINEConv requires --mesh-edge-features geometry")

        if local_feature_dims is None:
            local_feature_dims = [self.latent_dim] * self.stages
        self.stage_local_feature_dims = [int(dim) for dim in local_feature_dims]
        if len(self.stage_local_feature_dims) != self.stages:
            raise ValueError("local_feature_dims must have one value per mesh decoder stage")

        if stage_hidden_dims is None:
            stage_hidden_dims = self.stage_local_feature_dims
        self.stage_hidden_dims = [int(dim) for dim in stage_hidden_dims]
        if len(self.stage_hidden_dims) != self.stages:
            raise ValueError("stage_hidden_dims must have one value per mesh decoder stage")

        prev_latent_dims = [self.initial_latent_dim] + self.stage_hidden_dims[:-1]
        self.stages_refine = nn.ModuleList(
            [
                _MeshRefineStage(
                    latent_dim=prev_latent_dims[stage_idx],
                    local_feature_dim=self.stage_local_feature_dims[stage_idx],
                    hidden_dim=self.stage_hidden_dims[stage_idx],
                    num_blocks=num_gcn_blocks,
                    graph_layer=self.graph_layer,
                    edge_features=self.edge_features,
                )
                for stage_idx in range(self.stages)
            ]
        )
        if stage_max_offsets is None:
            stage_max_offsets_tensor = torch.linspace(1.0, 0.35, self.stages) * self.max_offset
        else:
            if len(stage_max_offsets) != self.stages:
                raise ValueError("stage_max_offsets must have one value per mesh decoder stage")
            stage_max_offsets_tensor = torch.tensor([float(x) for x in stage_max_offsets], dtype=torch.float32)
        self.register_buffer("stage_max_offsets", stage_max_offsets_tensor)
        self._edge_index_cache: dict[tuple[str, str], torch.Tensor] = {}

    def _cached_undirected_edge_index(
        self,
        edge_index: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if edge_index is None:
            return None
        if edge_index.dim() == 3:
            edge_index = edge_index[0]
        edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.int64).contiguous()
        digest = hashlib.blake2b(edge_cpu.numpy().tobytes(), digest_size=16).hexdigest()
        cache_key = (digest, str(device))
        cached = self._edge_index_cache.get(cache_key)
        if cached is not None:
            return cached

        undirected = _to_undirected_edge_index(edge_cpu).to(device=device, dtype=torch.long)
        self._edge_index_cache[cache_key] = undirected
        return undirected

    def forward_stage(
        self,
        stage_idx: int,
        latent: torch.Tensor,
        local_ct: torch.Tensor,
        cur_vertices: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        vertex_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached_edge_index = self._cached_undirected_edge_index(edge_index, cur_vertices.device)
        stage_latent, raw_offsets = self.stages_refine[stage_idx](
            latent,
            local_ct,
            cur_vertices,
            edge_index=cached_edge_index,
        )
        del vertex_mask

        offsets = torch.tanh(raw_offsets) * self.stage_max_offsets[stage_idx]
        next_vertices = cur_vertices + offsets
        next_latent = stage_latent
        gate = torch.ones(cur_vertices.shape[:2], device=cur_vertices.device, dtype=cur_vertices.dtype)
        return next_vertices, next_latent, gate.squeeze(-1)

    def subdivide_features(
        self,
        latent: torch.Tensor,
        midpoint_pairs: torch.Tensor | None,
    ) -> torch.Tensor:
        if midpoint_pairs is None:
            return latent
        midpoint_pairs = midpoint_pairs.to(device=latent.device, dtype=torch.long)
        midpoint_latent = 0.5 * (
            latent[:, midpoint_pairs[:, 0], :] + latent[:, midpoint_pairs[:, 1], :]
        )
        return torch.cat([latent, midpoint_latent], dim=1)
