"""Small synthetic phantoms: a broken tube and a mesh that spans the break.

The repair stage only ever sees a binary mask, a fitted mesh and the image
geometry, so a handful of voxels and a few hundred triangles are enough to drive
every branch of it. Nothing here depends on the challenge datasets, so these
fixtures are safe to publish and fast enough to run on every commit.
"""

from __future__ import annotations

import numpy as np

from mask_mesh_fit.io_utils import ImageGeometry

# Voxel index (x, y, z) equals physical (x, y, z) under this geometry, which keeps
# the phantoms readable: a coordinate in millimetres is also a voxel index.
IDENTITY_SPACING_MM = 1.0


def geometry(shape_zyx: tuple[int, int, int], spacing_mm: float = IDENTITY_SPACING_MM) -> ImageGeometry:
    """Axis-aligned geometry with isotropic spacing and the origin at the corner."""
    return ImageGeometry(
        shape_zyx=tuple(int(v) for v in shape_zyx),
        spacing_xyz=np.full(3, float(spacing_mm), dtype=np.float64),
        origin_xyz=np.zeros(3, dtype=np.float64),
        direction_xyz=np.eye(3, dtype=np.float64),
    )


def x_cylinder(
    shape_zyx: tuple[int, int, int],
    x_range: tuple[float, float],
    centre_yz: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Solid cylinder along the x axis, as a boolean ``(z, y, x)`` mask."""
    nz, ny, nx = shape_zyx
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    cy, cz = centre_yz
    radial = (yy - cy) ** 2 + (zz - cz) ** 2 <= radius**2
    axial = (xx >= x_range[0]) & (xx <= x_range[1])
    return radial & axial


def sphere(shape_zyx: tuple[int, int, int], centre_xyz: tuple[float, float, float], radius: float) -> np.ndarray:
    """Solid ball, as a boolean ``(z, y, x)`` mask."""
    nz, ny, nx = shape_zyx
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    cx, cy, cz = centre_xyz
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius**2


def broken_tube(
    shape_zyx: tuple[int, int, int] = (48, 48, 80),
    gap: tuple[float, float] = (34.0, 44.0),
    x_range: tuple[float, float] = (14.0, 64.0),
    centre_yz: tuple[float, float] = (24.0, 24.0),
    radius: float = 4.0,
) -> np.ndarray:
    """A straight tube cut in two by an axial gap: the canonical failure mode."""
    mask = x_cylinder(shape_zyx, x_range, centre_yz, radius)
    nx = shape_zyx[2]
    xx = np.arange(nx)[None, None, :]
    inside_gap = (xx > gap[0]) & (xx < gap[1])
    return mask & ~inside_gap


def tube_mesh(
    x_start: float,
    x_end: float,
    centre_yz: tuple[float, float] = (24.0, 24.0),
    radius: float = 4.0,
    rings: int = 24,
    sides: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed triangulated tube along x, returned as ``(vertices_xyz, faces, edge_index)``.

    Stands in for a fitted mesh: it spans the whole structure, including any gap in
    the mask, so the mesh graph offers a path between the separated components.
    """
    cy, cz = centre_yz
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    xs = np.linspace(x_start, x_end, rings)
    ring = np.stack([np.zeros_like(angles), radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    vertices = np.concatenate([ring + np.array([x, cy, cz]) for x in xs], axis=0)

    faces = []
    for r in range(rings - 1):
        for s in range(sides):
            a = r * sides + s
            b = r * sides + (s + 1) % sides
            c = (r + 1) * sides + s
            d = (r + 1) * sides + (s + 1) % sides
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces_arr = np.asarray(faces, dtype=np.int64)
    return vertices.astype(np.float64), faces_arr, edge_index_from_faces(faces_arr)


def edge_index_from_faces(faces: np.ndarray) -> np.ndarray:
    """Undirected mesh graph as a ``(2, E)`` array, both directions present."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.concatenate([e, e[:, ::-1]], axis=0)
    e = np.unique(e, axis=0)
    return e.T.astype(np.int64)
