"""Reconstruct a ``repair_case`` invocation from the ``metrics.json`` of a stored run.

Every run writes the repair parameters it used into ``metrics.json``. That makes a
finished run self-describing: this module turns one back into the command line that
produced it, so a regression test can re-run it and compare the output masks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# metrics.json section -> {key in that section: CLI flag}
_FILTER_FLAGS = {
    "connectivity": "--artifact-connectivity",
    "keep_near_main_mm": "--artifact-keep-near-main-mm",
    "remove_distance_mm": "--artifact-remove-distance-mm",
    "max_remove_voxels": "--artifact-max-remove-voxels",
}
_RADIUS_FLAGS = {
    "radius_mm": "--path-repair-radius-mm",
    "max_radius_mm": "--path-repair-max-radius-mm",
    "min_accept_radius_mm": "--path-repair-min-accept-radius-mm",
    "radius_mode": "--path-repair-radius-mode",
    "local_radius_window_mm": "--path-repair-local-radius-window-mm",
    "radius_percentile": "--path-repair-radius-percentile",
    "radius_scale": "--path-repair-radius-scale",
    "min_component_voxels": "--path-repair-min-component-voxels",
    "max_added_fraction": "--path-repair-max-added-fraction",
    "connectivity": "--path-repair-connectivity",
}
_MESH_PATH_FLAGS = {**_RADIUS_FLAGS, "anchor_mm": "--path-repair-anchor-mm"}
_ENDPOINT_FLAGS = {
    **_RADIUS_FLAGS,
    "max_gap_mm": "--endpoint-repair-max-gap-mm",
    "mesh_support_mm": "--endpoint-repair-mesh-support-mm",
    "min_mesh_support_fraction": "--endpoint-repair-min-mesh-support-fraction",
}
_CLEANUP_FLAGS = {
    "connectivity": "--post-cleanup-connectivity",
    "mesh_distance_mm": "--post-cleanup-mesh-distance-mm",
    "min_close_fraction": "--post-cleanup-min-close-fraction",
    "max_remove_voxels": "--post-cleanup-max-remove-voxels",
}

REPAIRED_FILENAME = {
    "mesh_path_connect": "repaired_mask_mesh_path_connect.nii.gz",
    "mask_endpoint_connect": "repaired_mask_endpoint_connect.nii.gz",
}


@dataclass(frozen=True)
class GoldenCase:
    """One stored run, replayable into a fresh output directory."""

    run_dir: Path
    case_id: str
    method: str
    mask_path: Path
    mesh_npz: Path
    metrics: dict

    @property
    def repaired_name(self) -> str:
        return REPAIRED_FILENAME[self.method]

    @property
    def expected_mask(self) -> Path:
        return self.run_dir / self.repaired_name

    def missing_inputs(self) -> list[str]:
        missing = [str(p) for p in (self.mask_path, self.mesh_npz, self.expected_mask) if not p.exists()]
        return missing

    def command(self, output_dir: Path) -> list[str]:
        """The ``repair_case`` argument list that reproduces this run."""
        args = [
            "--case-id", self.case_id,
            "--mask-dir", str(self.mask_path.parent),
            "--mesh-npz", str(self.mesh_npz),
            "--output-dir", str(output_dir),
            "--geometric-repair-method", self.method,
            "--disable-qa",
        ]

        section = self.metrics.get("mask_artifact_filter") or {}
        if section.get("method"):
            args += ["--mask-artifact-filter", str(section["method"])]
            args += _flags(section, _FILTER_FLAGS)

        repair_key = "mesh_path_repair" if self.method == "mesh_path_connect" else "endpoint_repair"
        flags = _MESH_PATH_FLAGS if self.method == "mesh_path_connect" else _ENDPOINT_FLAGS
        args += _flags(self.metrics.get(repair_key) or {}, flags)

        section = self.metrics.get("post_repair_cleanup") or {}
        if section.get("method"):
            args += ["--post-repair-cleanup", str(section["method"])]
            args += _flags(section, _CLEANUP_FLAGS)

        return args


def _flags(section: dict, mapping: dict[str, str]) -> list[str]:
    out: list[str] = []
    for key, flag in mapping.items():
        if key in section and section[key] is not None:
            out += [flag, _fmt(section[key])]
    return out


def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def load_case(run_dir: Path) -> GoldenCase:
    """Build a :class:`GoldenCase` from a run directory containing ``metrics.json``."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    method = metrics.get("geometric_repair_method")
    if method not in REPAIRED_FILENAME:
        raise ValueError(f"{run_dir}: unsupported repair method {method!r}")
    return GoldenCase(
        run_dir=run_dir,
        case_id=str(metrics["case_id"]),
        method=str(method),
        mask_path=Path(metrics["mask_path"]),
        mesh_npz=Path(metrics["mesh_npz"]),
        metrics=metrics,
    )


def load_manifest(path: Path) -> list[GoldenCase]:
    """Load the run directories listed in a manifest JSON file.

    The manifest holds paths to local run directories, so it is not committed.
    See ``tests/README.md``.
    """
    path = Path(path)
    entries = json.loads(path.read_text())
    runs = entries["runs"] if isinstance(entries, dict) else entries
    base = path.parent
    return [load_case((base / r).resolve() if not Path(r).is_absolute() else Path(r)) for r in runs]
