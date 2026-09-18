# Template meshes

Anatomy-specific template meshes used in the paper. Each `.npz` file contains at
least `vertices` (V, 3) and `faces` (F, 3); most also carry a precomputed
`edge_index` and per-stage face/edge tables (`stage_faces_k`, `stage_edge_index_k`)
consumed by the four-stage graph decoder. The template is recentred and
radius-normalised before fitting, so absolute size does not matter.

| File | Anatomy | Topology | Vertices | Faces |
| --- | --- | --- | --- | --- |
| `aorta_cylinder_10k.npz` | Aorta (SEGA, AortaSeg24) | capped cylinder | 9,986 | 19,968 |
| `topcow_torus_10k.npz` | Circle of Willis (TopCoW) | thin torus | 10,240 | 20,480 |
| `parse_sphere_20k.npz` | Pulmonary arteries (PARSE) | UV sphere | 19,970 | 39,936 |

The template choice matters: a topology that encourages a false bridge can make
the fitted mesh propose a wrong connection, while a template without a needed
loop or branch makes repair overly conservative. For a new vascular territory,
start from the template whose topology is closest to the expected anatomy and
tune the regularisation weights (see the main README).
