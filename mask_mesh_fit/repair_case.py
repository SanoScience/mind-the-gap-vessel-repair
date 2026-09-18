from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .artifact_filter import filter_mask_artifacts_by_component_distance
from .endpoint_repair import repair_mask_with_endpoint_paths
from .io_utils import extract_surface_from_mask, load_mask, save_float_like, save_label_like, save_mask_like
from .mesh_path_repair import repair_mask_with_mesh_paths
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
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _find_mask(mask_dir: Path, case_id: str) -> Path:
    path = mask_dir / f"{case_id}.nii.gz"
    if not path.exists():
        raise RuntimeError(f"Mask not found: {path}")
    return path


def _load_bool_image(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(bool, copy=False)


def _load_fitted_mesh(mesh_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(mesh_npz, allow_pickle=True)
    if "vertices_physical_xyz" in data:
        vertices = data["vertices_physical_xyz"].astype(np.float32, copy=False)
    elif "vertices_norm_xyz" in data and "norm_center_xyz" in data and "norm_scale" in data:
        vertices = (
            data["vertices_norm_xyz"].astype(np.float32, copy=False) * float(np.asarray(data["norm_scale"]))
            + data["norm_center_xyz"].astype(np.float32, copy=False)[None, :]
        )
    else:
        raise RuntimeError(
            f"{mesh_npz} must contain vertices_physical_xyz, or vertices_norm_xyz + norm_center_xyz + norm_scale. "
            f"Available keys: {list(data.files)}"
        )
    if "faces" not in data:
        raise RuntimeError(f"{mesh_npz} does not contain faces. Available keys: {list(data.files)}")
    faces = data["faces"].astype(np.int64, copy=False)
    edge_index = data["edge_index"].astype(np.int64, copy=False) if "edge_index" in data else None
    return vertices, faces, edge_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair one mask from an already fitted v24 mesh.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-voxelized", type=Path, default=None)
    parser.add_argument("--mesh-sdf", type=Path, default=None)
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
        default="component_sdf",
    )
    parser.add_argument("--sdf-repair-threshold-mm", type=float, default=1.0)
    parser.add_argument("--sdf-repair-connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument(
        "--component-sdf-touch-original-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sdf-clip-mm", type=float, default=16.0)
    parser.add_argument("--save-mesh-sdf", action="store_true")
    parser.add_argument("--voxelize-backend", choices=["pyvista", "multigeomed"], default="multigeomed")
    parser.add_argument("--voxelize-margin-voxels", type=int, default=3)
    parser.add_argument("--voxelize-slab-depth", type=int, default=16)
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
    parser.add_argument("--disable-qa", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = _find_mask(args.mask_dir, args.case_id)
    mask_zyx, reference_image, geometry = load_mask(mask_path)
    artifact_filter_metrics = None
    if args.mask_artifact_filter == "component_distance":
        print("Filtering distal mask artifacts before repair...")
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
    vertices_physical, faces, edge_index = _load_fitted_mesh(args.mesh_npz)

    timings: dict[str, float] = {}
    print(f"Case: {args.case_id}")
    print(f"Mask: {mask_path}")
    print(f"Mesh: {args.mesh_npz}")
    print(f"Output: {args.output_dir}")

    if args.geometric_repair_method in {"mesh_path_connect", "mask_endpoint_connect"}:
        if args.geometric_repair_method == "mesh_path_connect":
            print("Repairing with mesh-guided path connector...")
        else:
            print("Repairing with local endpoint connector...")
        t0 = time.perf_counter()
        if args.geometric_repair_method == "mesh_path_connect":
            path_result = repair_mask_with_mesh_paths(
                mask_zyx,
                vertices_physical,
                faces,
                edge_index,
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
                vertices_physical,
                faces,
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
                vertices_physical,
                faces,
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
        save_mask_like(repaired, reference_image, args.output_dir / "repaired_mask.nii.gz")
        save_mask_like(repaired, reference_image, args.output_dir / repaired_specific_name)
        save_mask_like(path_result.bridge_mask, reference_image, args.output_dir / bridge_name)
        save_label_like(
            path_result.component_labels,
            reference_image,
            args.output_dir / labels_name,
        )
        timings["save_outputs_seconds"] = time.perf_counter() - t0

        if not args.disable_qa:
            t0 = time.perf_counter()
            target_physical, _target_faces, _target_zyx = extract_surface_from_mask(mask_zyx, geometry)
            save_qa_overlay(
                target_points_physical=target_physical,
                mesh_vertices_physical=vertices_physical,
                mesh_faces=faces,
                output_png=args.output_dir / "qa_overlay.png",
                output_pdf=args.output_dir / "qa_overlay.pdf",
                title=args.case_id,
            )
            timings["qa_overlay_seconds"] = time.perf_counter() - t0

        voxel_volume_mm3 = float(np.prod(geometry.spacing_xyz))
        metrics = {
            "case_id": args.case_id,
            "mask_path": mask_path,
            "mesh_npz": args.mesh_npz,
            "geometric_repair_method": args.geometric_repair_method,
            "original_voxels": int(mask_zyx.sum()),
            "repaired_voxels": int(repaired.sum()),
            "added_voxels": int((repaired & ~mask_zyx).sum()),
            "added_volume_mm3": float((repaired & ~mask_zyx).sum() * voxel_volume_mm3),
            "voxel_volume_mm3": voxel_volume_mm3,
            "timings": timings,
            "runtime_seconds": time.time() - start,
        }
        metrics[repair_metrics_key] = path_result.metrics
        if artifact_filter_metrics is not None:
            metrics["mask_artifact_filter"] = artifact_filter_metrics
        if post_cleanup_metrics is not None:
            metrics["post_repair_cleanup"] = post_cleanup_metrics
        with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(metrics), f, indent=2)
        print(f"Saved repair outputs: {args.output_dir}")
        print(f"Metrics: {args.output_dir / 'metrics.json'}")
        return

    if args.mesh_voxelized is not None:
        t0 = time.perf_counter()
        mesh_mask = _load_bool_image(args.mesh_voxelized)
        if tuple(mesh_mask.shape) != tuple(mask_zyx.shape):
            raise RuntimeError(f"mesh_voxelized shape {mesh_mask.shape} != mask shape {mask_zyx.shape}")
        timings["load_mesh_voxelized_seconds"] = time.perf_counter() - t0
    else:
        print("Voxelizing fitted mesh...")
        t0 = time.perf_counter()
        mesh_mask = voxelize_mesh_to_mask(
            vertices_physical,
            faces,
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

    needs_sdf = args.save_mesh_sdf or args.mesh_sdf is not None or args.geometric_repair_method in {
        "sdf_cc",
        "component_sdf",
        "both",
    }
    mesh_sdf = None
    if needs_sdf:
        if args.mesh_sdf is not None:
            t0 = time.perf_counter()
            mesh_sdf = sitk.GetArrayFromImage(sitk.ReadImage(str(args.mesh_sdf))).astype(np.float32, copy=False)
            if tuple(mesh_sdf.shape) != tuple(mask_zyx.shape):
                raise RuntimeError(f"mesh_sdf shape {mesh_sdf.shape} != mask shape {mask_zyx.shape}")
            timings["load_mesh_sdf_seconds"] = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            mesh_sdf = mesh_signed_distance(mesh_mask, geometry.spacing_xyz, clip_mm=args.sdf_clip_mm)
            timings["mesh_sdf_seconds"] = time.perf_counter() - t0
            print(f"Mesh SDF: {timings['mesh_sdf_seconds']:.2f}s")

    sdf_cc_repaired = None
    sdf_candidate = None
    sdf_kept_candidate = None
    if args.geometric_repair_method in {"sdf_cc", "both"}:
        if mesh_sdf is None:
            raise RuntimeError("mesh_sdf is required for sdf_cc repair")
        t0 = time.perf_counter()
        sdf_cc_repaired, sdf_candidate, _sdf_local, sdf_kept_candidate = repair_mask_with_mesh_sdf_blend(
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
        del _sdf_local
        timings["sdf_cc_repair_seconds"] = time.perf_counter() - t0

    component_repaired = None
    component_candidate = None
    component_kept_candidate = None
    if args.geometric_repair_method in {"component_sdf", "both"}:
        if mesh_sdf is None:
            raise RuntimeError("mesh_sdf is required for component_sdf repair")
        t0 = time.perf_counter()
        component_repaired, component_candidate, _component_local, component_kept_candidate = (
            repair_mask_with_component_gated_sdf(
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
        )
        del _component_local
        timings["component_sdf_repair_seconds"] = time.perf_counter() - t0

    if args.geometric_repair_method == "voxel_or":
        repaired = voxel_or_repaired
    elif args.geometric_repair_method == "sdf_cc":
        repaired = sdf_cc_repaired
    elif args.geometric_repair_method == "component_sdf":
        repaired = component_repaired
    else:
        repaired = component_repaired
    post_cleanup_metrics = None
    if args.post_repair_cleanup == "remove_mesh_uncovered_components":
        print("Removing mesh-uncovered leftover components...")
        t0 = time.perf_counter()
        save_mask_like(repaired, reference_image, args.output_dir / "repaired_mask_mesh_path_connect_precleanup.nii.gz")
        cleanup_result = remove_mesh_uncovered_components(
            repaired,
            vertices_physical,
            faces,
            geometry,
            mesh_distance_mm=args.post_cleanup_mesh_distance_mm,
            min_close_fraction=args.post_cleanup_min_close_fraction,
            max_remove_voxels=args.post_cleanup_max_remove_voxels,
            connectivity=args.post_cleanup_connectivity,
        )
        repaired = cleanup_result.cleaned_mask
        post_cleanup_metrics = cleanup_result.metrics
        post_cleanup_metrics["removed_original_voxels"] = int((cleanup_result.removed_mask & mask_zyx).sum())
        save_mask_like(cleanup_result.removed_mask, reference_image, args.output_dir / "post_cleanup_removed_mask.nii.gz")
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
    save_mask_like(local_region, reference_image, args.output_dir / "local_repair_region.nii.gz")
    save_mask_like(voxel_or_repaired, reference_image, args.output_dir / "repaired_mask_voxel_or.nii.gz")
    if mesh_sdf is not None and args.save_mesh_sdf:
        save_float_like(mesh_sdf, reference_image, args.output_dir / "mesh_sdf.nii.gz")
    if sdf_cc_repaired is not None:
        save_mask_like(sdf_cc_repaired, reference_image, args.output_dir / "repaired_mask_sdf_cc.nii.gz")
        save_mask_like(sdf_candidate, reference_image, args.output_dir / "sdf_repair_candidate.nii.gz")
        save_mask_like(sdf_kept_candidate, reference_image, args.output_dir / "sdf_repair_kept_candidate.nii.gz")
    if component_repaired is not None:
        save_mask_like(component_repaired, reference_image, args.output_dir / "repaired_mask_component_sdf.nii.gz")
        save_mask_like(component_candidate, reference_image, args.output_dir / "component_sdf_candidate.nii.gz")
        save_mask_like(component_kept_candidate, reference_image, args.output_dir / "component_sdf_kept_candidate.nii.gz")
    save_mask_like(repaired, reference_image, args.output_dir / "repaired_mask.nii.gz")
    timings["save_outputs_seconds"] = time.perf_counter() - t0

    if not args.disable_qa:
        t0 = time.perf_counter()
        target_physical, _target_faces, _target_zyx = extract_surface_from_mask(mask_zyx, geometry)
        save_qa_overlay(
            target_points_physical=target_physical,
            mesh_vertices_physical=vertices_physical,
            mesh_faces=faces,
            output_png=args.output_dir / "qa_overlay.png",
            output_pdf=args.output_dir / "qa_overlay.pdf",
            title=args.case_id,
        )
        timings["qa_overlay_seconds"] = time.perf_counter() - t0

    voxel_volume_mm3 = float(np.prod(geometry.spacing_xyz))
    metrics = {
        "case_id": args.case_id,
        "mask_path": mask_path,
        "mesh_npz": args.mesh_npz,
        "geometric_repair_method": args.geometric_repair_method,
        "repair_dilation_mm": args.repair_dilation_mm,
        "repair_region_method": args.repair_region_method,
        "sdf_repair_threshold_mm": args.sdf_repair_threshold_mm,
        "sdf_repair_connectivity": args.sdf_repair_connectivity,
        "component_sdf_touch_original_mask": args.component_sdf_touch_original_mask,
        "voxelize_backend": args.voxelize_backend,
        "original_voxels": int(mask_zyx.sum()),
        "mesh_voxels": int(mesh_mask.sum()),
        "repaired_voxels": int(repaired.sum()),
        "added_voxels": int((repaired & ~mask_zyx).sum()),
        "added_volume_mm3": float((repaired & ~mask_zyx).sum() * voxel_volume_mm3),
        "voxel_volume_mm3": voxel_volume_mm3,
        "timings": timings,
        "runtime_seconds": time.time() - start,
    }
    if artifact_filter_metrics is not None:
        metrics["mask_artifact_filter"] = artifact_filter_metrics
    if post_cleanup_metrics is not None:
        metrics["post_repair_cleanup"] = post_cleanup_metrics
    if component_repaired is not None:
        metrics["component_sdf_repair"] = {
            "candidate_voxels": int(component_candidate.sum()),
            "kept_candidate_voxels": int(component_kept_candidate.sum()),
            "repaired_voxels": int(component_repaired.sum()),
            "added_voxels": int((component_repaired & ~mask_zyx).sum()),
        }
    if sdf_cc_repaired is not None:
        metrics["sdf_cc_repair"] = {
            "candidate_voxels": int(sdf_candidate.sum()),
            "kept_candidate_voxels": int(sdf_kept_candidate.sum()),
            "repaired_voxels": int(sdf_cc_repaired.sum()),
            "added_voxels": int((sdf_cc_repaired & ~mask_zyx).sum()),
        }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2)
    print(f"Saved repair outputs: {args.output_dir}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
