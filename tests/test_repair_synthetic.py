"""Behavioural tests for the repair stage on synthetic phantoms.

These pin the properties the method claims: repair reconnects components through
the mesh, it only ever adds voxels, and it declines a bridge that would exceed the
foreground-growth budget. They need no dataset and run in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

import synthetic as S
from mask_mesh_fit.artifact_filter import filter_mask_artifacts_by_component_distance
from mask_mesh_fit.endpoint_repair import repair_mask_with_endpoint_paths
from mask_mesh_fit.mesh_path_repair import repair_mask_with_mesh_paths
from mask_mesh_fit.post_repair_cleanup import remove_mesh_uncovered_components

SHAPE = (48, 48, 80)
FULL_CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)


def count_components(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=FULL_CONNECTIVITY)[1])


@pytest.fixture
def geom():
    return S.geometry(SHAPE)


@pytest.fixture
def broken():
    """Tube split in two by an axial gap."""
    mask = S.broken_tube(SHAPE)
    assert count_components(mask) == 2, "phantom should start disconnected"
    return mask


@pytest.fixture
def mesh():
    """Fitted-mesh stand-in spanning the whole tube, gap included."""
    return S.tube_mesh(12.0, 66.0)


def repair_via_mesh_path(mask, mesh, geom, **overrides):
    vertices, faces, edge_index = mesh
    kwargs = dict(
        anchor_mm=3.0,
        radius_mm=1.0,
        min_accept_radius_mm=2.0,
        max_radius_mm=5.0,
        radius_mode="adaptive_local",
        min_component_voxels=1,
        max_added_fraction=0.30,
        connectivity=26,
    )
    kwargs.update(overrides)
    return repair_mask_with_mesh_paths(mask, vertices, faces, edge_index, geom, **kwargs)


def repair_via_endpoints(mask, mesh, geom, **overrides):
    vertices, faces, _ = mesh
    kwargs = dict(
        max_gap_mm=14.0,
        mesh_support_mm=3.0,
        min_mesh_support_fraction=0.5,
        radius_mm=1.0,
        min_accept_radius_mm=2.0,
        max_radius_mm=5.0,
        radius_mode="adaptive_local",
        min_component_voxels=1,
        max_added_fraction=0.30,
        connectivity=26,
    )
    kwargs.update(overrides)
    return repair_mask_with_endpoint_paths(mask, vertices, faces, geom, **kwargs)


@pytest.mark.parametrize("strategy", [repair_via_mesh_path, repair_via_endpoints])
def test_repair_reconnects_a_broken_tube(strategy, broken, mesh, geom):
    result = strategy(broken, mesh, geom)

    assert result.metrics["repair_status"] == "connected"
    assert result.metrics["original_components"] == 2
    assert result.metrics["repaired_components"] == 1
    assert result.metrics["accepted_paths"] >= 1
    assert count_components(result.repaired) == 1


@pytest.mark.parametrize("strategy", [repair_via_mesh_path, repair_via_endpoints])
def test_repair_only_adds_voxels(strategy, broken, mesh, geom):
    """The repaired mask must be a superset of the prediction it post-processes."""
    result = strategy(broken, mesh, geom)

    assert np.all(result.repaired >= broken)
    assert result.metrics["added_voxels"] == int((result.repaired & ~broken).sum())
    assert result.metrics["added_voxels"] > 0


@pytest.mark.parametrize("strategy", [repair_via_mesh_path, repair_via_endpoints])
def test_bridge_is_rejected_when_it_exceeds_the_growth_budget(strategy, broken, mesh, geom):
    result = strategy(broken, mesh, geom, max_added_fraction=1e-6)

    assert result.metrics["accepted_paths"] == 0
    assert result.metrics["rejected_paths"] >= 1
    assert result.metrics["added_voxels"] == 0
    assert np.array_equal(result.repaired, broken), "a rejected bridge must leave the mask untouched"


@pytest.mark.parametrize("strategy", [repair_via_mesh_path, repair_via_endpoints])
def test_connected_mask_is_left_alone(strategy, mesh, geom):
    whole = S.x_cylinder(SHAPE, (14.0, 64.0), (24.0, 24.0), 4.0)

    result = strategy(whole, mesh, geom)

    assert result.metrics["repair_status"] == "noop_already_connected"
    assert np.array_equal(result.repaired, whole)


def test_bridge_mask_holds_exactly_the_added_voxels(broken, mesh, geom):
    result = repair_via_mesh_path(broken, mesh, geom)

    assert np.array_equal(result.bridge_mask & broken, np.zeros_like(broken))
    assert np.array_equal(result.repaired, broken | result.bridge_mask)


def test_artifact_filter_removes_far_islands_and_keeps_near_ones(geom):
    tube = S.x_cylinder(SHAPE, (14.0, 64.0), (24.0, 24.0), 4.0)
    near = S.sphere(SHAPE, (39.0, 32.0, 24.0), 2.0)
    far = S.sphere(SHAPE, (39.0, 24.0, 44.0), 2.0)
    mask = tube | near | far
    assert count_components(mask) == 3

    result = filter_mask_artifacts_by_component_distance(
        mask, geom, keep_near_main_mm=8.0, remove_distance_mm=12.0, max_remove_voxels=0, connectivity=26
    )

    assert np.all(result.cleaned_mask >= tube), "the main structure must survive"
    assert np.any(result.cleaned_mask & near), "a component close to the main one is kept"
    assert not np.any(result.cleaned_mask & far), "a distant island is removed"
    assert np.array_equal(result.cleaned_mask | result.removed_mask, mask)


def test_cleanup_removes_components_the_mesh_does_not_support(mesh, geom):
    vertices, faces, _ = mesh
    tube = S.x_cylinder(SHAPE, (14.0, 64.0), (24.0, 24.0), 4.0)
    unsupported = S.sphere(SHAPE, (39.0, 24.0, 44.0), 2.0)
    mask = tube | unsupported

    result = remove_mesh_uncovered_components(
        mask, vertices, faces, geom, mesh_distance_mm=2.0, min_close_fraction=0.01,
        max_remove_voxels=1000, connectivity=26,
    )

    assert np.all(result.cleaned_mask >= tube)
    assert not np.any(result.cleaned_mask & unsupported)
    assert result.metrics["removed_voxels"] == int(unsupported.sum())


def test_cleanup_keeps_a_component_that_sits_on_the_mesh(mesh, geom):
    """Size alone must not condemn a component: mesh support is what decides."""
    vertices, faces, _ = mesh
    tube = S.x_cylinder(SHAPE, (14.0, 48.0), (24.0, 24.0), 4.0)
    on_mesh = S.x_cylinder(SHAPE, (54.0, 62.0), (24.0, 24.0), 4.0)
    mask = tube | on_mesh
    assert count_components(mask) == 2

    result = remove_mesh_uncovered_components(
        mask, vertices, faces, geom, mesh_distance_mm=2.0, min_close_fraction=0.01,
        max_remove_voxels=1000, connectivity=26,
    )

    assert np.array_equal(result.cleaned_mask, mask)
    assert result.metrics["removed_voxels"] == 0
