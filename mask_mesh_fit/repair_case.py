from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .io_utils import extract_surface_from_mask, load_mask, save_float_like, save_label_like, save_mask_like
from .post_repair_cleanup import remove_mesh_uncovered_components
from .qa import save_qa_overlay
from .repair_args import add_case_arguments, add_repair_arguments
from .repair_pipeline import BRIDGE_METHODS, filter_artifacts, growth_metrics, run_bridge_repair
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
    add_case_arguments(parser)
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--mesh-voxelized", type=Path, default=None)
    parser.add_argument("--mesh-sdf", type=Path, default=None)
    add_repair_arguments(
        parser,
        geometric_repair_method="component_sdf",
        sdf_repair_threshold_mm=1.0,
        voxelize_backend="multigeomed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = _find_mask(args.mask_dir, args.case_id)
    mask_zyx, reference_image, geometry = load_mask(mask_path)
    filtered = filter_artifacts(mask_zyx, geometry, reference_image, args.output_dir, args)
    mask_zyx = filtered.mask
    artifact_filter_metrics = filtered.metrics
    vertices_physical, faces, edge_index = _load_fitted_mesh(args.mesh_npz)

    timings: dict[str, float] = {}
    print(f"Case: {args.case_id}")
    print(f"Mask: {mask_path}")
    print(f"Mesh: {args.mesh_npz}")
    print(f"Output: {args.output_dir}")

    if args.geometric_repair_method in BRIDGE_METHODS:
        outcome = run_bridge_repair(
            mask_zyx,
            vertices_physical,
            faces,
            edge_index,
            geometry,
            reference_image,
            args.output_dir,
            args,
            timings,
            save_timing_key="save_outputs_seconds",
        )
        repaired = outcome.repaired

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

        metrics = {
            "case_id": args.case_id,
            "mask_path": mask_path,
            "mesh_npz": args.mesh_npz,
            "geometric_repair_method": args.geometric_repair_method,
            **growth_metrics(mask_zyx, repaired, geometry),
            "timings": timings,
            "runtime_seconds": time.time() - start,
        }
        metrics[outcome.names.metrics_key] = outcome.repair_metrics
        if artifact_filter_metrics is not None:
            metrics["mask_artifact_filter"] = artifact_filter_metrics
        if outcome.cleanup_metrics is not None:
            metrics["post_repair_cleanup"] = outcome.cleanup_metrics
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
