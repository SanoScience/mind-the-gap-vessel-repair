"""Byte-exact regression against stored runs.

Every finished run records the repair parameters it used, so a run directory can be
replayed and its output mask compared with the one on disk. Repair is deterministic
(it loads a fitted mesh rather than fitting one), so the comparison is exact: any
difference means behaviour changed.

The fixtures are nnU-Net predictions derived from challenge datasets whose terms do
not allow redistribution, so they are not committed. Point the test at local runs
with a manifest, see ``tests/README.md``. Without one, these tests skip.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from golden_cases import load_manifest

MANIFEST = Path(__file__).parent / "golden_manifest.json"

pytestmark = pytest.mark.golden


def _cases():
    if not MANIFEST.exists():
        return []
    try:
        return load_manifest(MANIFEST)
    except Exception as exc:  # a broken manifest should be loud, not silently skipped
        pytest.fail(f"cannot read {MANIFEST}: {exc}")


CASES = _cases()


def _case_id(case) -> str:
    return f"{case.case_id}-{case.method}"


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """Run every manifest case once and hand back its fresh output directory."""
    if not CASES:
        pytest.skip(f"no {MANIFEST.name}; see tests/README.md")

    outputs = {}
    for case in CASES:
        missing = case.missing_inputs()
        if missing:
            pytest.skip(f"{case.case_id}: inputs not on this machine: {missing[0]}")
        out = tmp_path_factory.mktemp(f"replay_{case.case_id}")
        completed = subprocess.run(
            [sys.executable, "-m", "mask_mesh_fit.repair_case", *case.command(out)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.fail(f"{case.case_id}: repair_case failed\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
        outputs[_case_id(case)] = out
    return outputs


def read_mask(path: Path) -> np.ndarray:
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(bool)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_repaired_mask_is_unchanged(case, replayed):
    """The replayed mask must equal the stored one voxel for voxel."""
    produced = read_mask(replayed[_case_id(case)] / case.repaired_name)
    expected = read_mask(case.expected_mask)

    assert produced.shape == expected.shape
    differing = int((produced != expected).sum())
    assert differing == 0, (
        f"{case.case_id}: {differing} voxels differ from the stored run "
        f"({int(produced.sum())} vs {int(expected.sum())} foreground voxels)"
    )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_repair_metrics_are_unchanged(case, replayed):
    """Bridge accounting must match too, so a silent change in decisions is caught."""
    section = "mesh_path_repair" if case.method == "mesh_path_connect" else "endpoint_repair"
    produced = json.loads((replayed[_case_id(case)] / "metrics.json").read_text())[section]
    expected = case.metrics[section]

    watched = [
        "repair_status",
        "original_components",
        "repaired_components",
        "accepted_paths",
        "rejected_paths",
        "added_voxels",
    ]
    assert {k: produced[k] for k in watched} == {k: expected[k] for k in watched}


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_per_bridge_decisions_are_unchanged(case, replayed):
    """Every bridge's own record must match: which component, what radius, accepted or why not.

    The summary counts above would survive a change that swapped one accepted bridge
    for another. This compares the decisions themselves.
    """
    section = "mesh_path_repair" if case.method == "mesh_path_connect" else "endpoint_repair"
    produced = json.loads((replayed[_case_id(case)] / "metrics.json").read_text())[section]

    assert _comparable(produced) == _comparable(case.metrics[section])


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_cleanup_metrics_are_unchanged(case, replayed):
    """The mesh-supported cleanup must remove exactly the same components."""
    expected = case.metrics.get("post_repair_cleanup")
    if expected is None:
        pytest.skip("this run had no post-repair cleanup")
    produced = json.loads((replayed[_case_id(case)] / "metrics.json").read_text())["post_repair_cleanup"]

    assert _comparable(produced) == _comparable(expected)


def _comparable(value):
    """Round floats so a last-bit difference in a reported distance is not a failure."""
    if isinstance(value, dict):
        return {k: _comparable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_comparable(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value
