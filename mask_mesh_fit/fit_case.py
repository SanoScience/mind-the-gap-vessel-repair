from __future__ import annotations

import argparse
import json
import time
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .artifact_filter import filter_mask_artifacts_by_component_distance
from .endpoint_repair import repair_mask_with_endpoint_paths
from .geometry import load_template
from .io_utils import (
    extract_surface_from_mask,
    load_mask,
    make_bbox_normalization,
    save_label_like,
    save_float_like,
    save_mask_like,
    save_npz,
)
from .mesh_path_repair import repair_mask_with_mesh_paths
from .optimize import DecoderConfig, StageConfig, fit_mesh_to_target
from .post_repair_cleanup import remove_mesh_uncovered_components
from .qa import save_qa_overlay
from .voxelize import (
    mesh_signed_distance,
    repair_mask_with_component_gated_sdf,
    repair_mask_with_mesh,
    repair_mask_with_mesh_sdf_blend,
    voxelize_mesh_to_mask,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def _find_mask(mask_dir: Path, case_id: str) -> Path:
    path = mask_dir / f"{case_id}.nii.gz"
    if not path.exists():
        raise RuntimeError(f"Mask not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a sphere-topology mesh to one nnUNet predicted mask.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--template-radius", type=float, default=0.8)
    parser.add_argument("--target-sampling", choices=["fps", "random"], default="fps")
    parser.add_argument("--fps-candidate-points", type=int, default=50000)
    parser.add_argument("--optimization-loop", choices=["lightning", "manual"], default="lightning")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--deformation-mode", choices=["decoder"], default="decoder")
    parser.add_argument("--fit-steps", type=int, default=0)
    parser.add_argument("--decoder-stages", type=int, default=4)
    parser.add_argument("--decoder-stage-max-offsets", type=float, nargs="+", default=[1.0, 0.5, 0.2, 0.1])
    parser.add_argument("--decoder-latent-dim", type=int, default=128)
    parser.add_argument("--decoder-local-feature-dim", type=int, default=128)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-num-blocks", type=int, default=3)
    parser.add_argument(
        "--decoder-graph-layer",
        choices=["gcn", "sage", "graph", "gen", "gine", "transformer", "gatv2"],
        default="gcn",
    )
    parser.add_argument("--decoder-edge-features", choices=["none", "geometry"], default="none")
    parser.add_argument("--decoder-coarse-max-offset", type=float, default=0.50)
    parser.add_argument("--decoder-detail-max-offset", type=float, default=0.20)
    parser.add_argument("--decoder-stage-loss-weight", type=float, default=0.10)
    parser.add_argument(
        "--train-alignment-in-decoder",
        "--optimize-align-in-decoder",
        dest="train_alignment_in_decoder",
        action="store_true",
        help="Also optimize the PCA template alignment inside the decoder loop. Default keeps alignment fixed.",
    )

    parser.add_argument("--align-steps", type=int, default=250)
    parser.add_argument("--coarse-steps", type=int, default=700)
    parser.add_argument("--detail-steps", type=int, default=700)
    parser.add_argument("--align-lr", type=float, default=0.02)
    parser.add_argument("--coarse-lr", type=float, default=0.003)
    parser.add_argument("--detail-lr", type=float, default=0.0015)
    parser.add_argument("--align-target-points", type=int, default=2048)
    parser.add_argument("--coarse-target-points", type=int, default=4096)
    parser.add_argument("--detail-target-points", type=int, default=8192)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--align-lambda-chamfer", type=float, default=0.10)
    parser.add_argument("--align-lambda-bbox", type=float, default=1.0)
    parser.add_argument("--coarse-lambda-chamfer", type=float, default=1.0)
    parser.add_argument("--coarse-lambda-bbox", type=float, default=0.0)
    parser.add_argument("--coarse-lambda-vertex-chamfer", type=float, default=0.0)
    parser.add_argument("--coarse-lambda-deform", type=float, default=0.0)
    parser.add_argument("--coarse-lambda-edge", type=float, default=0.03)
    parser.add_argument("--coarse-lambda-laplacian", type=float, default=0.10)
    parser.add_argument("--coarse-lambda-normal", type=float, default=0.01)
    parser.add_argument("--coarse-lambda-face-area", type=float, default=0.0)
    parser.add_argument("--coarse-lambda-face-area-var", type=float, default=0.001)
    parser.add_argument("--detail-lambda-chamfer", type=float, default=1.0)
    parser.add_argument("--detail-lambda-bbox", type=float, default=0.0)
    parser.add_argument("--detail-lambda-vertex-chamfer", type=float, default=0.0)
    parser.add_argument("--detail-lambda-deform", type=float, default=0.0)
    parser.add_argument("--detail-lambda-edge", type=float, default=0.01)
    parser.add_argument("--detail-lambda-laplacian", type=float, default=0.03)
    parser.add_argument("--detail-lambda-normal", type=float, default=0.005)
    parser.add_argument("--detail-lambda-face-area", type=float, default=0.0)
    parser.add_argument("--detail-lambda-face-area-var", type=float, default=0.0005)

    parser.add_argument("--mask-artifact-filter", choices=["none", "component_distance"], default="none")
    parser.add_argument("--artifact-keep-near-main-mm", type=float, default=12.0)
    parser.add_argument("--artifact-remove-distance-mm", type=float, default=25.0)
    parser.add_argument(
        "--artifact-max-remove-voxels",
        type=int,
        default=50,
        help="Maximum size of far components to remove. Use 0 to remove all far components by distance only.",
    )
    parser.add_argument("--artifact-connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument("--repair-dilation-mm", type=float, default=8.0)
    parser.add_argument("--repair-region-method", choices=["edt_crop", "edt", "binary_dilation"], default="edt_crop")
    parser.add_argument(
        "--geometric-repair-method",
        choices=["voxel_or", "sdf_cc", "component_sdf", "both", "mesh_path_connect", "mask_endpoint_connect"],
        default="voxel_or",
    )
    parser.add_argument("--sdf-repair-threshold-mm", type=float, default=1.5)
    parser.add_argument("--sdf-repair-connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument(
        "--component-sdf-touch-original-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--repair-mode", choices=["voxelize", "learned", "both"], default="voxelize")
    parser.add_argument("--learned-refiner-checkpoint", type=Path, default=None)
    parser.add_argument("--sdf-clip-mm", type=float, default=16.0)
    parser.add_argument("--save-mesh-sdf", action="store_true")
    parser.add_argument("--voxelize-backend", choices=["pyvista", "multigeomed"], default="pyvista")
    parser.add_argument("--voxelize-margin-voxels", type=int, default=3)
    parser.add_argument("--voxelize-slab-depth", type=int, default=16)
    parser.add_argument("--skip-voxelize", action="store_true")
    parser.add_argument("--path-repair-anchor-mm", type=float, default=2.0)
    parser.add_argument("--path-repair-radius-mm", type=float, default=1.0)
    parser.add_argument(
        "--path-repair-max-radius-mm",
        type=float,
        default=0.0,
        help="Maximum bridge tube radius. Use 0 for auto: anchor_mm plus one voxel.",
    )
    parser.add_argument(
        "--path-repair-min-accept-radius-mm",
        type=float,
        default=0.0,
        help="Keep growing an accepted bridge until at least this radius. Use 0 to accept at path-repair-radius-mm.",
    )
    parser.add_argument(
        "--path-repair-radius-mode",
        choices=["fixed", "adaptive_local"],
        default="fixed",
        help="Use a fixed bridge radius or adapt it from local component thickness.",
    )
    parser.add_argument("--path-repair-local-radius-window-mm", type=float, default=6.0)
    parser.add_argument("--path-repair-radius-percentile", type=float, default=80.0)
    parser.add_argument("--path-repair-radius-scale", type=float, default=1.0)
    parser.add_argument("--path-repair-min-component-voxels", type=int, default=20)
    parser.add_argument("--path-repair-max-added-fraction", type=float, default=0.03)
    parser.add_argument("--path-repair-connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument(
        "--path-repair-selection",
        choices=["shortest", "edge_filtered_shortest"],
        default="shortest",
        help="Bridge selector. shortest preserves the original behavior.",
    )
    parser.add_argument("--path-repair-min-path-edges", type=int, default=2)
    parser.add_argument(
        "--path-repair-max-mesh-edge-mm",
        type=float,
        default=0.0,
        help="For edge_filtered_shortest, ignore mesh graph edges longer than this. Use 0 to disable edge cap.",
    )
    parser.add_argument(
        "--path-repair-edge-filter-fallback",
        choices=["old_shortest", "skip"],
        default="old_shortest",
        help="What to do if edge-filtered selection finds no valid bridge.",
    )
    parser.add_argument(
        "--endpoint-repair-max-gap-mm",
        type=float,
        default=12.0,
        help="For mask_endpoint_connect, only bridge component endpoints this close in physical space.",
    )
    parser.add_argument(
        "--endpoint-repair-mesh-support-mm",
        type=float,
        default=3.0,
        help="For mask_endpoint_connect, line samples must be this close to fitted mesh support.",
    )
    parser.add_argument(
        "--endpoint-repair-min-mesh-support-fraction",
        type=float,
        default=0.50,
        help="For mask_endpoint_connect, minimum fraction of line samples close to mesh support.",
    )
    parser.add_argument(
        "--post-repair-cleanup",
        choices=["none", "remove_mesh_uncovered_components"],
        default="none",
    )
    parser.add_argument("--post-cleanup-mesh-distance-mm", type=float, default=2.0)
    parser.add_argument("--post-cleanup-min-close-fraction", type=float, default=0.01)
    parser.add_argument("--post-cleanup-max-remove-voxels", type=int, default=1000)
    parser.add_argument("--post-cleanup-connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument(
        "--init-fit-state",
        type=Path,
        default=None,
        help="Warm-start decoder weights/latent features from a previous v24 mesh_fit_state.pt.",
    )
    parser.add_argument(
        "--init-fit-state-load-alignment",
        action="store_true",
        help="Also load previous translation/scale/rotation. Usually leave off for different cases.",
    )
    parser.add_argument(
        "--save-fit-state",
        type=Path,
        default=None,
        help="Where to save reusable decoder fit state. Default: <output-dir>/mesh_fit_state.pt.",
    )
    parser.add_argument("--disable-save-fit-state", action="store_true")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--disable-qa", action="store_true")
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repair_mode in {"learned", "both"} and args.learned_refiner_checkpoint is None:
        raise RuntimeError("--repair-mode learned/both requires --learned-refiner-checkpoint")
    if args.skip_voxelize and args.repair_mode in {"learned", "both"}:
        raise RuntimeError("--repair-mode learned/both requires voxelization; remove --skip-voxelize")
    if args.geometric_repair_method in {"mesh_path_connect", "mask_endpoint_connect"} and args.repair_mode in {
        "learned",
        "both",
    }:
        raise RuntimeError(
            "--geometric-repair-method mesh_path_connect/mask_endpoint_connect supports --repair-mode voxelize only"
        )
    if args.init_fit_state is not None and not args.init_fit_state.exists():
        raise RuntimeError(f"--init-fit-state does not exist: {args.init_fit_state}")
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    writer = None
    if not args.disable_tensorboard:
        tb_dir = args.tensorboard_dir if args.tensorboard_dir is not None else args.output_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_dir))

    mask_path = _find_mask(args.mask_dir, args.case_id)
    print(f"Case: {args.case_id}")
    print(f"Mask: {mask_path}")
    print(f"Template: {args.template}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    if writer is not None:
        print(f"TensorBoard: {writer.log_dir}")
        writer.add_text("config/command_args", json.dumps(_jsonable(vars(args)), indent=2), global_step=0)

    def tb_log(stage: str, step: int, parts: dict[str, torch.Tensor | float]) -> None:
        if writer is None:
            return
        for name, value in parts.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                scalar = float(value.detach().cpu().item())
            else:
                scalar = float(value)
            writer.add_scalar(f"{stage}/{name}", scalar, global_step=step)

    mask_zyx, reference_image, geometry = load_mask(mask_path)
    artifact_filter_metrics = None
    if args.mask_artifact_filter == "component_distance":
        print("Filtering distal mask artifacts before mesh fitting...")
        filter_result = filter_mask_artifacts_by_component_distance(
            mask_zyx,
            geometry,
            keep_near_main_mm=args.artifact_keep_near_main_mm,
            remove_distance_mm=args.artifact_remove_distance_mm,
            max_remove_voxels=args.artifact_max_remove_voxels,
            connectivity=args.artifact_connectivity,
        )
        mask_zyx = filter_result.cleaned_mask
        artifact_filter_metrics = filter_result.metrics
        save_mask_like(filter_result.cleaned_mask, reference_image, args.output_dir / "artifact_cleaned_mask.nii.gz")
        save_mask_like(filter_result.removed_mask, reference_image, args.output_dir / "artifact_removed_mask.nii.gz")
        save_label_like(
            filter_result.component_labels,
            reference_image,
            args.output_dir / "artifact_component_labels.nii.gz",
        )
        print(
            "Artifact filter removed "
            f"{artifact_filter_metrics['removed_voxels']} voxels "
            f"from {len(artifact_filter_metrics['removed_components'])} components"
        )
    target_physical, target_faces, target_zyx = extract_surface_from_mask(mask_zyx, geometry)
    norm = make_bbox_normalization(target_physical)
    target_norm = norm.to_norm(target_physical)
    template = load_template(args.template, radius=args.template_radius)

    save_npz(
        args.output_dir / "target_surface.npz",
        target_points_physical_xyz=target_physical,
        target_points_norm_xyz=target_norm,
        target_vertices_pre_zyx=target_zyx,
        target_faces=target_faces,
        norm_center_xyz=norm.center_xyz,
        norm_scale=np.asarray(norm.scale, dtype=np.float32),
        image_shape_zyx=np.asarray(geometry.shape_zyx, dtype=np.int64),
        image_spacing_xyz=geometry.spacing_xyz.astype(np.float32),
        image_origin_xyz=geometry.origin_xyz.astype(np.float32),
        image_direction_xyz=geometry.direction_xyz.astype(np.float32),
    )

    align_cfg = StageConfig(
        name="align",
        steps=args.align_steps,
        lr=args.align_lr,
        target_points=args.align_target_points,
        lambda_chamfer=args.align_lambda_chamfer,
        lambda_bbox=args.align_lambda_bbox,
        log_every=args.log_every,
    )
    coarse_cfg = StageConfig(
        name="coarse",
        steps=args.coarse_steps,
        lr=args.coarse_lr,
        target_points=args.coarse_target_points,
        lambda_chamfer=args.coarse_lambda_chamfer,
        lambda_bbox=args.coarse_lambda_bbox,
        lambda_vertex_chamfer=args.coarse_lambda_vertex_chamfer,
        lambda_deform=args.coarse_lambda_deform,
        lambda_edge=args.coarse_lambda_edge,
        lambda_laplacian=args.coarse_lambda_laplacian,
        lambda_normal=args.coarse_lambda_normal,
        lambda_face_area=args.coarse_lambda_face_area,
        lambda_face_area_var=args.coarse_lambda_face_area_var,
        log_every=args.log_every,
    )
    detail_cfg = StageConfig(
        name="detail",
        steps=args.detail_steps,
        lr=args.detail_lr,
        target_points=args.detail_target_points,
        lambda_chamfer=args.detail_lambda_chamfer,
        lambda_bbox=args.detail_lambda_bbox,
        lambda_vertex_chamfer=args.detail_lambda_vertex_chamfer,
        lambda_deform=args.detail_lambda_deform,
        lambda_edge=args.detail_lambda_edge,
        lambda_laplacian=args.detail_lambda_laplacian,
        lambda_normal=args.detail_lambda_normal,
        lambda_face_area=args.detail_lambda_face_area,
        lambda_face_area_var=args.detail_lambda_face_area_var,
        log_every=args.log_every,
    )
    decoder_cfg = DecoderConfig(
        deformation_mode=args.deformation_mode,
        stages=args.decoder_stages,
        latent_dim=args.decoder_latent_dim,
        local_feature_dim=args.decoder_local_feature_dim,
        hidden_dim=args.decoder_hidden_dim,
        num_blocks=args.decoder_num_blocks,
        graph_layer=args.decoder_graph_layer,
        edge_features=args.decoder_edge_features,
        stage_max_offsets=tuple(float(x) for x in args.decoder_stage_max_offsets),
        coarse_max_offset=args.decoder_coarse_max_offset,
        detail_max_offset=args.decoder_detail_max_offset,
        stage_loss_weight=args.decoder_stage_loss_weight,
        fit_steps=args.fit_steps,
        train_alignment=args.train_alignment_in_decoder,
    )

    print("Fitting mesh...")
    t_fit = time.perf_counter()
    save_fit_state_path = None
    if not args.disable_save_fit_state:
        save_fit_state_path = args.save_fit_state if args.save_fit_state is not None else args.output_dir / "mesh_fit_state.pt"

    stage_results, final_metrics = fit_mesh_to_target(
        template_vertices_norm=template.vertices,
        faces=template.faces,
        edge_index=template.edge_index,
        target_points_norm=target_norm,
        align_cfg=align_cfg,
        coarse_cfg=coarse_cfg,
        detail_cfg=detail_cfg,
        sampling_mode=args.target_sampling,
        fps_candidate_points=args.fps_candidate_points,
        seed=args.seed,
        device=device,
        decoder_cfg=decoder_cfg,
        optimization_loop=args.optimization_loop,
        enable_progress_bar=not args.disable_progress_bar,
        tb_log_fn=tb_log if writer is not None else None,
        init_fit_state_path=args.init_fit_state,
        init_fit_state_load_alignment=args.init_fit_state_load_alignment,
        save_fit_state_path=save_fit_state_path,
    )
    mesh_fit_seconds = time.perf_counter() - t_fit
    print(f"Mesh fitting/training: {mesh_fit_seconds:.2f}s")

    metrics: dict[str, Any] = {
        "case_id": args.case_id,
        "mask_path": mask_path,
        "template": args.template,
        "device": str(device),
        "num_target_surface_points": int(target_physical.shape[0]),
        "num_target_faces": int(target_faces.shape[0]),
        "num_template_vertices": int(template.vertices.shape[0]),
        "num_template_faces": int(template.faces.shape[0]),
        "deformation_mode": args.deformation_mode,
        "optimization_loop": args.optimization_loop,
        "decoder_config": decoder_cfg,
        "fit_state": {
            "init": str(args.init_fit_state) if args.init_fit_state is not None else None,
            "init_load_alignment": bool(args.init_fit_state_load_alignment),
            "saved": str(save_fit_state_path) if save_fit_state_path is not None else None,
        },
        "normalization": {"center_xyz": norm.center_xyz, "scale": norm.scale},
        "stages": {},
        "final": final_metrics,
        "timings": {},
    }
    if artifact_filter_metrics is not None:
        metrics["mask_artifact_filter"] = artifact_filter_metrics
    timings = metrics["timings"]
    timings["mesh_fit_seconds"] = float(mesh_fit_seconds)
    timings["training_seconds"] = float(mesh_fit_seconds)

    stage_name_to_file = {
        "align": "fitted_mesh_align.npz",
        "coarse": "fitted_mesh_coarse.npz",
        "detail": "fitted_mesh_detail.npz",
    }
    final_vertices_physical = None
    final_result_metrics: dict[str, Any] = {}
    for result in stage_results:
        vertices_physical = norm.to_physical(result.vertices_norm)
        final_vertices_physical = vertices_physical
        final_result_metrics = result.metrics
        stage_file = stage_name_to_file.get(result.name, f"fitted_mesh_{result.name}.npz")
        save_npz(
            args.output_dir / stage_file,
            vertices_norm_xyz=result.vertices_norm,
            vertices_physical_xyz=vertices_physical,
            faces=template.faces,
            edge_index=template.edge_index,
            norm_center_xyz=norm.center_xyz,
            norm_scale=np.asarray(norm.scale, dtype=np.float32),
            metrics_json=np.asarray(json.dumps(_jsonable(result.metrics))),
        )
        metrics["stages"][result.name] = result.metrics

    if final_vertices_physical is None:
        raise RuntimeError("No fitted mesh was produced")
    save_npz(
        args.output_dir / "fitted_mesh_best_chamfer.npz",
        vertices_norm_xyz=stage_results[-1].vertices_norm,
        vertices_physical_xyz=final_vertices_physical,
        faces=template.faces,
        edge_index=template.edge_index,
        norm_center_xyz=norm.center_xyz,
        norm_scale=np.asarray(norm.scale, dtype=np.float32),
        metrics_json=np.asarray(json.dumps(_jsonable(final_result_metrics))),
    )
    metrics["selected_mesh"] = {
        "source": "best_chamfer",
        "file": "fitted_mesh_best_chamfer.npz",
        "also_saved_as": stage_name_to_file.get(stage_results[-1].name, f"fitted_mesh_{stage_results[-1].name}.npz"),
        "metrics": final_result_metrics,
    }

    mesh_mask = None
    repaired = None
    if args.geometric_repair_method in {"mesh_path_connect", "mask_endpoint_connect"} and not args.skip_voxelize:
        if args.geometric_repair_method == "mesh_path_connect":
            print("Repairing with mesh-guided path connector...")
        else:
            print("Repairing with local endpoint connector...")
        t0 = time.perf_counter()
        if args.geometric_repair_method == "mesh_path_connect":
            path_result = repair_mask_with_mesh_paths(
                mask_zyx,
                final_vertices_physical,
                template.faces,
                template.edge_index,
                geometry,
                anchor_mm=args.path_repair_anchor_mm,
                radius_mm=args.path_repair_radius_mm,
                max_radius_mm=args.path_repair_max_radius_mm,
                min_accept_radius_mm=args.path_repair_min_accept_radius_mm,
                radius_mode=args.path_repair_radius_mode,
                local_radius_window_mm=args.path_repair_local_radius_window_mm,
                radius_percentile=args.path_repair_radius_percentile,
                radius_scale=args.path_repair_radius_scale,
                min_component_voxels=args.path_repair_min_component_voxels,
                max_added_fraction=args.path_repair_max_added_fraction,
                connectivity=args.path_repair_connectivity,
                path_selection=args.path_repair_selection,
                min_path_edges=args.path_repair_min_path_edges,
                max_mesh_edge_mm=args.path_repair_max_mesh_edge_mm,
                edge_filter_fallback=args.path_repair_edge_filter_fallback,
            )
            repair_metrics_key = "mesh_path_repair"
            repaired_specific_name = "repaired_mask_mesh_path_connect.nii.gz"
            bridge_name = "mesh_path_bridge_mask.nii.gz"
            labels_name = "mesh_path_component_labels.nii.gz"
            precleanup_name = "repaired_mask_mesh_path_connect_precleanup.nii.gz"
            timings["mesh_path_repair_seconds"] = time.perf_counter() - t0
        else:
            path_result = repair_mask_with_endpoint_paths(
                mask_zyx,
                final_vertices_physical,
                template.faces,
                geometry,
                max_gap_mm=args.endpoint_repair_max_gap_mm,
                mesh_support_mm=args.endpoint_repair_mesh_support_mm,
                min_mesh_support_fraction=args.endpoint_repair_min_mesh_support_fraction,
                radius_mm=args.path_repair_radius_mm,
                max_radius_mm=args.path_repair_max_radius_mm,
                min_accept_radius_mm=args.path_repair_min_accept_radius_mm,
                radius_mode=args.path_repair_radius_mode,
                local_radius_window_mm=args.path_repair_local_radius_window_mm,
                radius_percentile=args.path_repair_radius_percentile,
                radius_scale=args.path_repair_radius_scale,
                min_component_voxels=args.path_repair_min_component_voxels,
                max_added_fraction=args.path_repair_max_added_fraction,
                connectivity=args.path_repair_connectivity,
            )
            repair_metrics_key = "endpoint_repair"
            repaired_specific_name = "repaired_mask_endpoint_connect.nii.gz"
            bridge_name = "endpoint_bridge_mask.nii.gz"
            labels_name = "endpoint_component_labels.nii.gz"
            precleanup_name = "repaired_mask_endpoint_connect_precleanup.nii.gz"
            timings["endpoint_repair_seconds"] = time.perf_counter() - t0
        repaired = path_result.repaired
        timing_key = "mesh_path_repair_seconds" if args.geometric_repair_method == "mesh_path_connect" else "endpoint_repair_seconds"
        print(f"Direct repair: {timings[timing_key]:.2f}s")
        post_cleanup_metrics = None
        if args.post_repair_cleanup == "remove_mesh_uncovered_components":
            print("Removing mesh-uncovered leftover components...")
            t0 = time.perf_counter()
            save_mask_like(
                repaired,
                reference_image,
                args.output_dir / precleanup_name,
            )
            cleanup_result = remove_mesh_uncovered_components(
                repaired,
                final_vertices_physical,
                template.faces,
                geometry,
                mesh_distance_mm=args.post_cleanup_mesh_distance_mm,
                min_close_fraction=args.post_cleanup_min_close_fraction,
                max_remove_voxels=args.post_cleanup_max_remove_voxels,
                connectivity=args.post_cleanup_connectivity,
            )
            repaired = cleanup_result.cleaned_mask
            post_cleanup_metrics = cleanup_result.metrics
            post_cleanup_metrics["removed_original_voxels"] = int((cleanup_result.removed_mask & mask_zyx).sum())
            save_mask_like(
                cleanup_result.removed_mask,
                reference_image,
                args.output_dir / "post_cleanup_removed_mask.nii.gz",
            )
            save_label_like(
                cleanup_result.component_labels,
                reference_image,
                args.output_dir / "post_cleanup_component_labels.nii.gz",
            )
            timings["post_repair_cleanup_seconds"] = time.perf_counter() - t0
            print(
                "Post-cleanup removed "
                f"{post_cleanup_metrics['removed_voxels']} voxels "
                f"from {len(post_cleanup_metrics['removed_components'])} components"
            )

        t0 = time.perf_counter()
        save_mask_like(repaired, reference_image, args.output_dir / repaired_specific_name)
        save_mask_like(path_result.bridge_mask, reference_image, args.output_dir / bridge_name)
        save_label_like(
            path_result.component_labels,
            reference_image,
            args.output_dir / labels_name,
        )
        save_mask_like(repaired, reference_image, args.output_dir / "repaired_mask.nii.gz")
        timings["save_repair_outputs_seconds"] = time.perf_counter() - t0
        print(f"Save repair outputs: {timings['save_repair_outputs_seconds']:.2f}s")

        voxel_volume_mm3 = float(np.prod(geometry.spacing_xyz))
        metrics[repair_metrics_key] = {
            **path_result.metrics,
            "original_voxels": int(mask_zyx.sum()),
            "repaired_voxels": int(repaired.sum()),
            "added_voxels": int((repaired & ~mask_zyx).sum()),
            "voxel_volume_mm3": voxel_volume_mm3,
            "added_volume_mm3": float((repaired & ~mask_zyx).sum() * voxel_volume_mm3),
            "geometric_repair_method": args.geometric_repair_method,
        }
        if post_cleanup_metrics is not None:
            metrics["post_repair_cleanup"] = post_cleanup_metrics
        args.skip_voxelize = True
    if not args.skip_voxelize:
        print("Voxelizing fitted mesh...")
        t0 = time.perf_counter()
        mesh_mask = voxelize_mesh_to_mask(
            final_vertices_physical,
            template.faces,
            geometry,
            margin_voxels=args.voxelize_margin_voxels,
            slab_depth=args.voxelize_slab_depth,
            backend=args.voxelize_backend,
        )
        timings["voxelize_seconds"] = time.perf_counter() - t0
        print(f"Voxelization: {timings['voxelize_seconds']:.2f}s")
        t0 = time.perf_counter()
        voxel_or_repaired, local_region = repair_mask_with_mesh(
            mask_zyx,
            mesh_mask,
            geometry,
            dilation_mm=args.repair_dilation_mm,
            method=args.repair_region_method,
        )
        timings["voxel_or_repair_seconds"] = time.perf_counter() - t0
        print(f"Voxel OR repair: {timings['voxel_or_repair_seconds']:.2f}s")
        mesh_sdf = None
        needs_sdf = (
            args.save_mesh_sdf
            or args.repair_mode in {"learned", "both"}
            or args.geometric_repair_method in {"sdf_cc", "component_sdf", "both"}
        )
        if needs_sdf:
            t0 = time.perf_counter()
            mesh_sdf = mesh_signed_distance(mesh_mask, geometry.spacing_xyz, clip_mm=args.sdf_clip_mm)
            timings["mesh_sdf_seconds"] = time.perf_counter() - t0
            print(f"Mesh SDF: {timings['mesh_sdf_seconds']:.2f}s")
        sdf_cc_repaired = None
        sdf_candidate = None
        sdf_kept_candidate = None
        if args.geometric_repair_method in {"sdf_cc", "both"}:
            if mesh_sdf is None:
                raise RuntimeError("Internal error: mesh SDF was not computed for sdf_cc repair")
            t0 = time.perf_counter()
            sdf_cc_repaired, sdf_candidate, sdf_local_region, sdf_kept_candidate = repair_mask_with_mesh_sdf_blend(
                mask_zyx,
                mesh_mask,
                mesh_sdf,
                geometry,
                dilation_mm=args.repair_dilation_mm,
                sdf_threshold_mm=args.sdf_repair_threshold_mm,
                local_region_method=args.repair_region_method,
                local_region_zyx=local_region,
                connectivity=args.sdf_repair_connectivity,
            )
            del sdf_local_region
            timings["sdf_cc_repair_seconds"] = time.perf_counter() - t0
            print(f"SDF+CC repair: {timings['sdf_cc_repair_seconds']:.2f}s")
        component_sdf_repaired = None
        component_sdf_candidate = None
        component_sdf_kept_candidate = None
        if args.geometric_repair_method in {"component_sdf", "both"}:
            if mesh_sdf is None:
                raise RuntimeError("Internal error: mesh SDF was not computed for component_sdf repair")
            t0 = time.perf_counter()
            (
                component_sdf_repaired,
                component_sdf_candidate,
                component_sdf_local_region,
                component_sdf_kept_candidate,
            ) = repair_mask_with_component_gated_sdf(
                mask_zyx,
                mesh_mask,
                mesh_sdf,
                geometry,
                dilation_mm=args.repair_dilation_mm,
                sdf_threshold_mm=args.sdf_repair_threshold_mm,
                local_region_method=args.repair_region_method,
                local_region_zyx=local_region,
                connectivity=args.sdf_repair_connectivity,
                touch_original_mask=args.component_sdf_touch_original_mask,
            )
            del component_sdf_local_region
            timings["component_sdf_repair_seconds"] = time.perf_counter() - t0
            print(f"Component-gated SDF repair: {timings['component_sdf_repair_seconds']:.2f}s")
        if args.geometric_repair_method == "sdf_cc":
            repaired = sdf_cc_repaired
        elif args.geometric_repair_method == "both":
            repaired = component_sdf_repaired
        elif args.geometric_repair_method == "component_sdf":
            repaired = component_sdf_repaired
        else:
            repaired = voxel_or_repaired
        post_cleanup_metrics = None
        if args.post_repair_cleanup == "remove_mesh_uncovered_components":
            print("Removing mesh-uncovered leftover components...")
            t0 = time.perf_counter()
            save_mask_like(repaired, reference_image, args.output_dir / "repaired_mask_mesh_path_connect_precleanup.nii.gz")
            cleanup_result = remove_mesh_uncovered_components(
                repaired,
                final_vertices_physical,
                template.faces,
                geometry,
                mesh_distance_mm=args.post_cleanup_mesh_distance_mm,
                min_close_fraction=args.post_cleanup_min_close_fraction,
                max_remove_voxels=args.post_cleanup_max_remove_voxels,
                connectivity=args.post_cleanup_connectivity,
            )
            repaired = cleanup_result.cleaned_mask
            post_cleanup_metrics = cleanup_result.metrics
            post_cleanup_metrics["removed_original_voxels"] = int((cleanup_result.removed_mask & mask_zyx).sum())
            save_mask_like(
                cleanup_result.removed_mask,
                reference_image,
                args.output_dir / "post_cleanup_removed_mask.nii.gz",
            )
            save_label_like(
                cleanup_result.component_labels,
                reference_image,
                args.output_dir / "post_cleanup_component_labels.nii.gz",
            )
            timings["post_repair_cleanup_seconds"] = time.perf_counter() - t0
            print(
                "Post-cleanup removed "
                f"{post_cleanup_metrics['removed_voxels']} voxels "
                f"from {len(post_cleanup_metrics['removed_components'])} components"
            )
        t0 = time.perf_counter()
        save_mask_like(mesh_mask, reference_image, args.output_dir / "mesh_voxelized.nii.gz")
        if mesh_sdf is not None:
            save_float_like(mesh_sdf, reference_image, args.output_dir / "mesh_sdf.nii.gz")
        save_mask_like(local_region, reference_image, args.output_dir / "local_repair_region.nii.gz")
        if args.geometric_repair_method in {"voxel_or", "both"}:
            save_mask_like(voxel_or_repaired, reference_image, args.output_dir / "repaired_mask_voxel_or.nii.gz")
        if args.geometric_repair_method in {"sdf_cc", "both"}:
            save_mask_like(sdf_cc_repaired, reference_image, args.output_dir / "repaired_mask_sdf_cc.nii.gz")
            save_mask_like(sdf_candidate, reference_image, args.output_dir / "sdf_repair_candidate.nii.gz")
            save_mask_like(sdf_kept_candidate, reference_image, args.output_dir / "sdf_repair_kept_candidate.nii.gz")
        if args.geometric_repair_method in {"component_sdf", "both"}:
            save_mask_like(
                component_sdf_repaired,
                reference_image,
                args.output_dir / "repaired_mask_component_sdf.nii.gz",
            )
            save_mask_like(
                component_sdf_candidate,
                reference_image,
                args.output_dir / "component_sdf_candidate.nii.gz",
            )
            save_mask_like(
                component_sdf_kept_candidate,
                reference_image,
                args.output_dir / "component_sdf_kept_candidate.nii.gz",
            )
        if args.repair_mode == "both":
            save_mask_like(repaired, reference_image, args.output_dir / "repaired_mask_voxelized.nii.gz")
        elif args.repair_mode == "voxelize":
            save_mask_like(repaired, reference_image, args.output_dir / "repaired_mask.nii.gz")
        timings["save_repair_outputs_seconds"] = time.perf_counter() - t0
        print(f"Save repair outputs: {timings['save_repair_outputs_seconds']:.2f}s")
        if args.repair_mode in {"learned", "both"}:
            from mask_mesh_refine.infer import refine_case_arrays

            t0 = time.perf_counter()
            learned_name = "refined_mask_learned.nii.gz" if args.repair_mode == "both" else "refined_mask.nii.gz"
            refined_prob, refined_mask = refine_case_arrays(
                checkpoint_path=args.learned_refiner_checkpoint,
                pred_mask_zyx=mask_zyx,
                mesh_mask_zyx=mesh_mask,
                mesh_sdf_zyx=mesh_sdf,
                reference_image=reference_image,
                output_dir=args.output_dir,
                output_mask_name=learned_name,
                output_prob_name="refined_prob.nii.gz",
                device=str(device),
            )
            timings["learned_refiner_seconds"] = time.perf_counter() - t0
            print(f"Learned refiner: {timings['learned_refiner_seconds']:.2f}s")
            metrics["learned_refiner"] = {
                "checkpoint": str(args.learned_refiner_checkpoint),
                "refined_voxels": int(refined_mask.sum()),
                "refined_prob_min": float(refined_prob.min()),
                "refined_prob_max": float(refined_prob.max()),
            }
        voxel_volume_mm3 = float(np.prod(geometry.spacing_xyz))
        metrics["voxel_repair"] = {
            "original_voxels": int(mask_zyx.sum()),
            "mesh_voxels": int(mesh_mask.sum()),
            "repaired_voxels": int(repaired.sum()),
            "added_voxels": int((repaired & ~mask_zyx).sum()),
            "voxel_volume_mm3": voxel_volume_mm3,
            "added_volume_mm3": float((repaired & ~mask_zyx).sum() * voxel_volume_mm3),
            "repair_dilation_mm": args.repair_dilation_mm,
            "repair_region_method": args.repair_region_method,
            "geometric_repair_method": args.geometric_repair_method,
            "voxelize_backend": args.voxelize_backend,
        }
        if post_cleanup_metrics is not None:
            metrics["post_repair_cleanup"] = post_cleanup_metrics
        metrics["voxel_or_repair"] = {
            "repaired_voxels": int(voxel_or_repaired.sum()),
            "added_voxels": int((voxel_or_repaired & ~mask_zyx).sum()),
            "added_volume_mm3": float((voxel_or_repaired & ~mask_zyx).sum() * voxel_volume_mm3),
        }
        if sdf_cc_repaired is not None:
            metrics["sdf_cc_repair"] = {
                "sdf_threshold_mm": args.sdf_repair_threshold_mm,
                "connectivity": args.sdf_repair_connectivity,
                "candidate_voxels": int(sdf_candidate.sum()),
                "kept_candidate_voxels": int(sdf_kept_candidate.sum()),
                "repaired_voxels": int(sdf_cc_repaired.sum()),
                "added_voxels": int((sdf_cc_repaired & ~mask_zyx).sum()),
                "added_volume_mm3": float((sdf_cc_repaired & ~mask_zyx).sum() * voxel_volume_mm3),
            }
        if component_sdf_repaired is not None:
            metrics["component_sdf_repair"] = {
                "sdf_threshold_mm": args.sdf_repair_threshold_mm,
                "connectivity": args.sdf_repair_connectivity,
                "touch_original_mask": args.component_sdf_touch_original_mask,
                "candidate_voxels": int(component_sdf_candidate.sum()),
                "kept_candidate_voxels": int(component_sdf_kept_candidate.sum()),
                "repaired_voxels": int(component_sdf_repaired.sum()),
                "added_voxels": int((component_sdf_repaired & ~mask_zyx).sum()),
                "added_volume_mm3": float((component_sdf_repaired & ~mask_zyx).sum() * voxel_volume_mm3),
            }

    if not args.disable_qa:
        print("Saving QA overlay...")
        t0 = time.perf_counter()
        save_qa_overlay(
            target_points_physical=target_physical,
            mesh_vertices_physical=final_vertices_physical,
            mesh_faces=template.faces,
            output_png=args.output_dir / "qa_overlay.png",
            output_pdf=args.output_dir / "qa_overlay.pdf",
            title=args.case_id,
        )
        timings["qa_overlay_seconds"] = time.perf_counter() - t0
        print(f"QA overlay: {timings['qa_overlay_seconds']:.2f}s")

    metrics["runtime_seconds"] = time.time() - start_time
    if writer is not None:
        for name, value in final_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"final/{name}", float(value), global_step=0)
        if "voxel_repair" in metrics:
            for name, value in metrics["voxel_repair"].items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"voxel_repair/{name}", float(value), global_step=0)
        for name, value in timings.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"timings/{name}", float(value), global_step=0)
        writer.add_scalar("runtime/mesh_fit_seconds", float(mesh_fit_seconds), global_step=0)
        writer.add_scalar("runtime/seconds", float(metrics["runtime_seconds"]), global_step=0)
        writer.flush()
        writer.close()
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2)
    print(f"Done. Metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
