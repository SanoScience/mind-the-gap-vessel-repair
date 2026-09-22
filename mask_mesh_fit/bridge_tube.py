"""Turning a proposed bridge into voxels, and deciding how wide it may be.

Both repair strategies end the same way. Each proposes a path between two
disconnected components, one by walking the mesh graph and one by joining nearby
component endpoints, and each then has to rasterise that path as a tube and decide
whether the result is acceptable. That shared ending lives here.

A bridge is widened from the base radius upward and accepted at the first width that
actually joins the two components, provided it stays within the foreground-growth
budget. Under the adaptive policy the minimum acceptable width follows the local
calibre of the two vessel ends, so a bridge is neither a hair across an aorta nor a
rod across a communicating artery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .io_utils import ImageGeometry
from .voxelize import physical_ball_structure

FIXED = "fixed"
ADAPTIVE_LOCAL = "adaptive_local"
RADIUS_MODES = (FIXED, ADAPTIVE_LOCAL)


def spacing_zyx(geometry: ImageGeometry) -> np.ndarray:
    """Voxel spacing reordered to match the (z, y, x) array layout."""
    return np.asarray(
        [geometry.spacing_xyz[2], geometry.spacing_xyz[1], geometry.spacing_xyz[0]], dtype=float
    )


def voxel_step_mm(geometry: ImageGeometry) -> float:
    """Smallest voxel dimension, used as the step when widening a bridge."""
    return float(max(np.min(spacing_zyx(geometry)), 1e-6))


@dataclass(frozen=True)
class RadiusPolicy:
    """How wide a bridge may be, and the narrowest width it may be accepted at.

    ``resolve`` turns the "zero means auto" conventions of the command line into real
    numbers once, so the rest of the code never repeats the defaulting rules.
    """

    radius_mm: float
    max_radius_mm: float
    min_accept_radius_mm: float
    step_mm: float
    mode: str
    local_window_mm: float
    percentile: float
    scale: float

    @property
    def adaptive(self) -> bool:
        return self.mode == ADAPTIVE_LOCAL

    @classmethod
    def resolve(
        cls,
        geometry: ImageGeometry,
        *,
        radius_mm: float,
        max_radius_mm: float | None,
        auto_max_radius_mm: float,
        min_accept_radius_mm: float | None,
        mode: str,
        local_window_mm: float,
        percentile: float,
        scale: float,
    ) -> "RadiusPolicy":
        """Validate the mode and settle the radius bounds.

        ``auto_max_radius_mm`` is what a non-positive ``max_radius_mm`` falls back to.
        The two strategies choose it differently: the mesh-graph one from the anchor
        distance, the endpoint one from the largest gap it is willing to cross.
        """
        mode = str(mode).strip().lower()
        if mode not in RADIUS_MODES:
            raise ValueError(f"radius_mode must be one of: {', '.join(RADIUS_MODES)}")

        if max_radius_mm is None or float(max_radius_mm) <= 0:
            max_radius_mm = auto_max_radius_mm
        max_radius_mm = float(max(float(radius_mm), float(max_radius_mm)))
        if min_accept_radius_mm is None or float(min_accept_radius_mm) <= 0:
            min_accept_radius_mm = float(radius_mm)
        min_accept_radius_mm = float(
            min(max(float(radius_mm), float(min_accept_radius_mm)), max_radius_mm)
        )

        return cls(
            radius_mm=float(radius_mm),
            max_radius_mm=max_radius_mm,
            min_accept_radius_mm=min_accept_radius_mm,
            step_mm=voxel_step_mm(geometry),
            mode=mode,
            local_window_mm=float(local_window_mm),
            percentile=float(percentile),
            scale=float(scale),
        )

    def accept_radius_mm(self, main_local_radius: float, candidate_local_radius: float) -> float:
        """Narrowest width this bridge may be accepted at.

        Under the fixed policy that is simply the configured minimum. Under the
        adaptive policy it follows the mean local calibre of the two ends, clamped to
        the configured bounds.
        """
        if not self.adaptive:
            return self.min_accept_radius_mm

        usable = [r for r in (main_local_radius, candidate_local_radius) if np.isfinite(r) and r > 0]
        accept = self.min_accept_radius_mm
        if usable:
            accept = float(np.mean(usable) * self.scale)
        return float(
            min(max(self.radius_mm, accept, self.min_accept_radius_mm), self.max_radius_mm)
        )

    def candidate_radii_mm(self, accept_radius_mm: float) -> list[float]:
        """Widths to try, from the base radius up to the maximum, in voxel-sized steps."""
        values = np.arange(self.radius_mm, self.max_radius_mm + 0.5 * self.step_mm, self.step_mm)
        extra = [self.max_radius_mm]
        if self.adaptive:
            extra.append(float(accept_radius_mm))
        values = np.concatenate([values, np.asarray(extra, dtype=float)])
        return sorted(float(x) for x in np.unique(np.round(values, decimals=6)) if x <= self.max_radius_mm)


def local_component_radius_mm(
    labels: np.ndarray,
    component_label: int,
    anchor_physical_xyz: np.ndarray,
    geometry: ImageGeometry,
    window_mm: float,
    percentile: float,
) -> float:
    """Local calibre of one component near a point, as a distance-transform percentile.

    Returns NaN when the anchor is outside the volume or the component is absent from
    the window, which callers treat as "no local estimate available".
    """
    if int(component_label) <= 0 or not np.all(np.isfinite(anchor_physical_xyz)):
        return float("nan")
    anchor_index_xyz = geometry.physical_to_continuous_index_xyz(anchor_physical_xyz[None, :])[0]
    anchor_zyx = np.rint(anchor_index_xyz[[2, 1, 0]]).astype(np.int64)
    shape = np.asarray(labels.shape, dtype=np.int64)
    if np.any(anchor_zyx < 0) or np.any(anchor_zyx >= shape):
        return float("nan")

    spacing = spacing_zyx(geometry)
    margin_zyx = np.maximum(np.ceil(float(window_mm) / spacing).astype(np.int64), 1)
    lo = np.maximum(anchor_zyx - margin_zyx, 0)
    hi = np.minimum(anchor_zyx + margin_zyx + 1, shape)
    crop_slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    component_crop = labels[crop_slices] == int(component_label)
    if not np.any(component_crop):
        return float("nan")

    dist = ndimage.distance_transform_edt(component_crop, sampling=tuple(float(x) for x in spacing))
    values = dist[component_crop]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, float(percentile)))


def rasterize_path_tube(
    path_vertices_physical_xyz: np.ndarray,
    geometry: ImageGeometry,
    radius_mm: float,
) -> np.ndarray:
    """Draw a path as a tube of the given physical radius, in image space.

    The path is walked in half-voxel steps so the centerline has no gaps, then
    dilated by a ball sized in millimetres. The dilation runs on a crop around the
    centerline rather than the whole volume.
    """
    output = np.zeros(geometry.shape_zyx, dtype=bool)
    if path_vertices_physical_xyz.shape[0] == 0:
        return output
    index_xyz = geometry.physical_to_continuous_index_xyz(path_vertices_physical_xyz)
    index_zyx = index_xyz[:, [2, 1, 0]]
    shape = np.asarray(geometry.shape_zyx, dtype=np.int64)
    spacing = spacing_zyx(geometry)
    min_spacing = float(max(np.min(spacing), 1e-6))
    voxel_chunks: list[np.ndarray] = []

    for start, end in zip(index_zyx[:-1], index_zyx[1:], strict=False):
        physical_len = float(np.linalg.norm((end - start) * spacing))
        steps = max(int(np.ceil(physical_len / (0.5 * min_spacing))), 1)
        ts = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)
        samples = start[None, :] * (1.0 - ts[:, None]) + end[None, :] * ts[:, None]
        voxels = np.rint(samples).astype(np.int64)
        in_bounds = np.all((voxels >= 0) & (voxels < shape[None, :]), axis=1)
        voxels = voxels[in_bounds]
        if voxels.size > 0:
            voxel_chunks.append(voxels)

    if path_vertices_physical_xyz.shape[0] == 1:
        voxel = np.rint(index_zyx[0]).astype(np.int64)
        if np.all((voxel >= 0) & (voxel < shape)):
            voxel_chunks.append(voxel[None, :])

    if not voxel_chunks:
        return output

    voxels = np.unique(np.concatenate(voxel_chunks, axis=0), axis=0)
    if float(radius_mm) <= 0:
        output[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
        return output

    radii_xyz = np.maximum(np.ceil(float(radius_mm) / np.asarray(geometry.spacing_xyz, dtype=float)).astype(int), 1)
    margin_zyx = np.asarray([radii_xyz[2], radii_xyz[1], radii_xyz[0]], dtype=np.int64) + 1
    lo = np.maximum(voxels.min(axis=0) - margin_zyx, 0)
    hi = np.minimum(voxels.max(axis=0) + margin_zyx + 1, shape)
    crop_slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    crop_shape = tuple(int(b - a) for a, b in zip(lo, hi, strict=True))
    centerline_crop = np.zeros(crop_shape, dtype=bool)
    crop_voxels = voxels - lo[None, :]
    centerline_crop[crop_voxels[:, 0], crop_voxels[:, 1], crop_voxels[:, 2]] = True
    structure = physical_ball_structure(geometry.spacing_xyz, float(radius_mm))
    output[crop_slices] = ndimage.binary_dilation(centerline_crop, structure=structure)
    return output


@dataclass
class TubeTrial:
    """Outcome of widening a bridge until it joins the two components.

    ``merges`` means the bridge was accepted. ``too_many_added`` means it did join
    them, but only by adding more foreground than the budget allows.
    """

    tube: np.ndarray
    added: np.ndarray
    added_voxels: int
    merges: bool
    selected_radius_mm: float
    too_many_added: bool


def grow_tube_until_merge(
    path_points_physical_xyz: np.ndarray,
    current: np.ndarray,
    main_seed: np.ndarray,
    candidate_seed: np.ndarray,
    geometry: ImageGeometry,
    policy: RadiusPolicy,
    accept_radius_mm: float,
    max_added_voxels: int,
) -> TubeTrial:
    """Widen the bridge until it touches both components, then judge what it cost.

    The tube is rasterised from a continuous path and is connected by construction,
    so it joins the components the moment it touches both seeds. Testing that is far
    cheaper than relabelling the volume at every trial width.
    """
    trial = TubeTrial(
        tube=np.zeros_like(current, dtype=bool),
        added=np.zeros_like(current, dtype=bool),
        added_voxels=0,
        merges=False,
        selected_radius_mm=policy.radius_mm,
        too_many_added=False,
    )

    for trial_radius in policy.candidate_radii_mm(accept_radius_mm):
        tube = rasterize_path_tube(path_points_physical_xyz, geometry, float(trial_radius))
        added = tube & ~current
        added_voxels = int(added.sum())
        merges = bool(np.any(tube & main_seed) and np.any(tube & candidate_seed))
        if not merges:
            continue
        if float(trial_radius) < float(accept_radius_mm):
            continue

        trial.tube = tube
        trial.added = added
        trial.added_voxels = added_voxels
        trial.selected_radius_mm = float(trial_radius)
        if added_voxels > max_added_voxels:
            trial.too_many_added = True
        else:
            trial.merges = True
        return trial

    return trial
