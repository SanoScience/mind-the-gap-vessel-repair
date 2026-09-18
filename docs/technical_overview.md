# Technical Overview of the Repair Pipeline

## One-Sentence Description

The current v24 system is a per-case, mesh-guided post-processing pipeline: it takes a hard nnUNet segmentation mask, overfits a deformable template mesh to that mask, and then uses the fitted mesh as a geometric prior to repair broken mask components while trying to avoid adding unnecessary voxels.

## What The System Is

v24 is not a full copy of the v23 hybrid training stack. It is a lightweight package under:

```text
frameworks/v24/mask_mesh_fit
```

The core entry points are:

```bash
python -m mask_mesh_fit.fit_case
python -m mask_mesh_fit.repair_case
```

`fit_case.py` performs mesh fitting and optional repair in one run. `repair_case.py` reuses an already-fitted mesh and applies a repair method to a mask. This is the common workflow when many parameter variants are being tested without refitting the mesh.

The main implementation files are:

```text
frameworks/v24/mask_mesh_fit/fit_case.py
frameworks/v24/mask_mesh_fit/repair_case.py
frameworks/v24/mask_mesh_fit/optimize.py
frameworks/v24/mask_mesh_fit/losses.py
frameworks/v24/mask_mesh_fit/mesh_path_repair.py
frameworks/v24/mask_mesh_fit/artifact_filter.py
frameworks/v24/mask_mesh_fit/post_repair_cleanup.py
frameworks/v24/mask_mesh_fit/evaluate_cleaned_vs_repaired.py
connectivity_evaluation.ipynb
```

## What The System Is Not

The default v24 mesh pipeline is not a CT-to-mask neural network. It does not infer directly from the CT image and it does not use nnUNet probability maps by default. The supervision for mesh fitting is the hard nnUNet binary mask.

It also does not train one general mesh model across all patients. It overfits one mesh to one case at a time. The fitted mesh is then used as a case-specific geometric prior for post-processing.

There is an optional learned refiner package under:

```text
frameworks/v24/mask_mesh_refine
```

but the current main repair experiments use geometric repair, especially `mesh_path_connect`, not the learned refiner.

## High-Level Pipeline

```text
nnUNet predicted binary mask (.nii.gz)
        |
        v
SimpleITK load with spacing, origin, direction
        |
        v
optional pre-fit artifact filtering
        |
        v
marching-cubes target surface from the mask
        |
        v
physical xyz coordinates and per-case normalization
        |
        v
template mesh: sphere, cylinder, torus, or tree-like prior
        |
        v
4-stage v23-style MeshDecoder overfitting
        |
        v
best-Chamfer fitted mesh saved as fitted_mesh_best_chamfer.npz
        |
        v
geometric repair of mask
        |
        v
optional post-repair cleanup
        |
        v
repaired NIfTI mask and metrics
```

## Input Geometry

Masks are loaded with SimpleITK so the original NIfTI metadata is preserved:

```text
array shape: z, y, x
spacing: x, y, z
origin: x, y, z
direction matrix: 3 x 3
```

The foreground mask is treated as binary. The target surface for fitting is extracted with marching cubes. Marching-cubes vertices are converted from voxel index coordinates into physical xyz coordinates before fitting. This is important: the mesh is optimized in real image geometry, not raw array index space.

The target point cloud is normalized per case:

```text
center = target bounding-box center
scale  = half of the largest bounding-box extent
normalized_point = (physical_point - center) / scale
```

The final fitted mesh is mapped back to physical coordinates before repair and export.

## Template Meshes

Templates are `.npz` files with at least:

```text
vertices
faces
```

They may also contain:

```text
edge_index
```

If `edge_index` is missing, v24 builds the mesh graph from triangular faces. The template is recentred and radius-normalized before fitting. The practical templates used so far include:

```text
aorta: sphere or cylinder templates
TOPCOW / Circle of Willis: torus or tree-like templates
Parse lungs/vessels: experiment-specific templates
```

The template choice matters strongly. If the template topology encourages a false bridge, the fitted mesh can propose a false connection. If the template lacks a needed branch or loop, repair can become too conservative.

## Mesh Fitting Model

The current active deformation mode is:

```bash
--deformation-mode decoder
```

The old direct per-vertex deformation path is not the intended current path. The decoder path uses the v23 `MeshDecoder` code from:

```text
frameworks/v23/Voxel_Mesh/Mesh_Deformation_Decoder.py
```

The typical configuration is a four-stage decoder:

```text
decoder_stages            = 4
decoder_latent_dim        = 128
decoder_local_feature_dim = 128
decoder_hidden_dim        = 128
decoder_num_blocks        = 3
decoder_graph_layer       = gcn
decoder_edge_features     = none
```

The usual stage offset schedule is:

```text
--decoder-stage-max-offsets 1.0 0.5 0.2 0.1
```

or, for more conservative cylinder/TOPCOW experiments:

```text
--decoder-stage-max-offsets 0.35 0.20 0.10 0.05
```

The key difference from v23 hybrid training is:

```text
v23: decoder deforms the mesh using features sampled from voxel feature maps.
v24: local per-vertex features are trainable parameters for this one case.
```

So the v24 mesh fit reuses the v23 graph-deformation machinery, but there is no CT encoder in the fitting loop.

## Forward Pass During Fitting

One optimization step does:

```text
1. Start from the template mesh after initial case alignment.
2. Initialize latent state and trainable local per-vertex features.
3. Run stage 1 of the MeshDecoder.
4. Run stage 2 using the previous stage output.
5. Run stage 3.
6. Run stage 4.
7. Compute the fitting loss on the final-stage mesh.
8. Optionally add auxiliary losses from intermediate stages.
9. If the final Chamfer is the best seen so far, save that mesh as the best candidate.
```

The final mesh used downstream is selected by lowest final Chamfer, not simply the last optimization step.

The Lightning loop is used mainly for progress bars, TensorBoard logging, and a clean optimization loop. A case run with `--fit-steps 6000` means 6000 optimization steps, not 6000 dataset epochs.

## What Is Optimized

The optimized parameters are:

```text
MeshDecoder weights
initial latent vector
one local-feature tensor per decoder stage
optional alignment parameters
```

Alignment parameters can be optimized with flags such as:

```bash
--train-alignment-in-decoder
--optimize-align-in-decoder
```

The reusable warm start is saved as:

```text
mesh_fit_state.pt
```

and can be reused with:

```bash
--init-fit-state path/to/mesh_fit_state.pt
```

## Mesh Fitting Losses

The main data term is Chamfer distance from sampled predicted mesh surface points to sampled target mask-surface points. v24 uses:

```text
chamferdist.ChamferDistance
```

Target sampling supports random sampling and FPS. FPS uses `pointnet2_utils` when available.

The regularization terms are v23 / MultiGeoMed style:

```text
edge length regularization
laplacian smoothing
normal consistency
face area regularization
normalized face-area variance
```

The practical loss is:

```text
loss =
  lambda_chamfer        * surface_chamfer
+ lambda_bbox           * bbox_loss
+ lambda_vertex_chamfer * vertex_chamfer
+ lambda_deform         * deformation_from_reference
+ lambda_edge           * edge_regularization
+ lambda_laplacian      * laplacian_smoothing
+ lambda_normal         * normal_consistency
+ lambda_face_area      * face_area_regularization
+ lambda_face_area_var  * normalized_face_area_variance
```

There is no currently active hard edge-length projection in the main workflow. Earlier edge-cap experiments were not kept as the main solution because they did not reliably prevent anatomical shortcut artifacts.

## Pre-Fit Artifact Filter

An optional mask-cleaning step can run before mesh fitting and before repair:

```bash
--mask-artifact-filter component_distance
```

It labels connected components in the nnUNet mask. The largest component is treated as the main component. Each non-main component is classified by physical distance to the main component.

The key parameters are:

```bash
--artifact-keep-near-main-mm
--artifact-remove-distance-mm
--artifact-max-remove-voxels
--artifact-connectivity
```

The intended logic is conservative:

```text
always keep components close to the main component
remove far components only if they satisfy the remove-distance rule
optionally apply a size cap
```

If `--artifact-max-remove-voxels 0` is used, the size cap is disabled and distance becomes the deciding rule.

When enabled, debug outputs include:

```text
artifact_cleaned_mask.nii.gz
artifact_removed_mask.nii.gz
artifact_component_labels.nii.gz
```

The cleaned mask is used as the repair input and, in `fit_case.py`, as the surface target for fitting.

## Geometric Repair Modes

v24 has several geometric repair methods:

```text
voxel_or
sdf_cc
component_sdf
both
mesh_path_connect
```

The older voxel/SDF modes require voxelizing the whole fitted mesh. These are useful baselines but can damage Dice if the mesh fills too much volume or contains a wrong bridge.

The current conservative repair method for aorta experiments is:

```bash
--geometric-repair-method mesh_path_connect
```

This method does not voxelize the entire mesh. It only uses the mesh graph to propose thin connector paths between disconnected mask components.

## Mesh-Path Repair

`mesh_path_connect` works as follows:

```text
1. Label connected components in the input mask.
2. If the mask already has one component, do nothing.
3. Build the mesh graph from edge_index or triangular faces.
4. Assign each mesh vertex to its nearest mask component if it is within anchor_mm.
5. Treat the largest mask component as the main component.
6. For each candidate component, find a mesh-graph path from main anchors to candidate anchors.
7. Rasterize only that path as a thin physical tube.
8. Accept the tube only if it actually touches/merges the two seed components and does not add too many voxels.
9. Repeat until connected or no valid bridge remains.
```

Important parameters:

```bash
--path-repair-anchor-mm
--path-repair-radius-mm
--path-repair-min-accept-radius-mm
--path-repair-max-radius-mm
--path-repair-radius-mode fixed|adaptive_local
--path-repair-local-radius-window-mm
--path-repair-radius-percentile
--path-repair-radius-scale
--path-repair-min-component-voxels
--path-repair-max-added-fraction
--path-repair-connectivity
```

The anchor radius is critical. A mesh vertex can only serve as evidence for a mask component if it lies close enough to that component. If a disconnected component has no mesh anchors, it cannot be repaired by this method.

The adaptive-radius mode estimates local endpoint thickness from the connected mask components:

```text
adaptive radius = mean(local radius near main endpoint, local radius near candidate endpoint) * scale
```

then clamps it between the requested minimum and maximum radii.

## Edge-Filtered Bridge Selection

For TOPCOW experiments, an optional path selector was added:

```bash
--path-repair-selection edge_filtered_shortest
```

It can reject suspicious one-edge or long-edge mesh shortcuts before falling back:

```bash
--path-repair-min-path-edges 2
--path-repair-max-mesh-edge-mm 4.0
--path-repair-edge-filter-fallback old_shortest|skip
```

This is opt-in. The default remains:

```bash
--path-repair-selection shortest
```

`edge_filtered_shortest` can help avoid some anatomically wrong direct shortcuts, but if it is too strict it can become overly conservative and fail to repair true gaps.

## Post-Repair Cleanup

After mesh-path repair, an optional final cleanup can remove leftover floating components not supported by the fitted mesh:

```bash
--post-repair-cleanup remove_mesh_uncovered_components
```

It runs after repair but before final masks are saved. It labels the repaired mask and keeps the largest component as the main component. For every other component, it checks whether component voxels are close to mesh support points.

Mesh support points are:

```text
fitted mesh vertices + fitted mesh face centroids
```

The key parameters are:

```bash
--post-cleanup-mesh-distance-mm 2.0
--post-cleanup-min-close-fraction 0.01
--post-cleanup-max-remove-voxels 1000
--post-cleanup-connectivity 26
```

A non-main component is removed only if:

```text
component_voxels <= max_remove_voxels
and fraction of voxels close to mesh support < min_close_fraction
```

This is intended for aorta-style single-object anatomy, where tiny floating repaired-mask components are almost always artifacts.

When cleanup is enabled, outputs include:

```text
repaired_mask_mesh_path_connect_precleanup.nii.gz
post_cleanup_removed_mask.nii.gz
post_cleanup_component_labels.nii.gz
repaired_mask_mesh_path_connect.nii.gz
```

Important interpretation:

```text
repaired_mask_mesh_path_connect.nii.gz is the final post-cleanup mask if cleanup was enabled.
```

## Main Outputs

Mesh fitting outputs:

```text
target_surface.npz
fitted_mesh_align.npz
fitted_mesh_stage1.npz
fitted_mesh_stage2.npz
fitted_mesh_stage3.npz
fitted_mesh_detail.npz
fitted_mesh_best_chamfer.npz
mesh_fit_state.pt
metrics.json
tensorboard/
```

Voxel/SDF repair outputs can include:

```text
mesh_voxelized.nii.gz
mesh_sdf.nii.gz
local_repair_region.nii.gz
repaired_mask.nii.gz
repaired_mask_component_sdf.nii.gz
component_sdf_candidate.nii.gz
component_sdf_kept_candidate.nii.gz
```

Mesh-path repair outputs include:

```text
artifact_cleaned_mask.nii.gz
artifact_removed_mask.nii.gz
artifact_component_labels.nii.gz
mesh_path_bridge_mask.nii.gz
mesh_path_component_labels.nii.gz
repaired_mask.nii.gz
repaired_mask_mesh_path_connect.nii.gz
metrics.json
```

If post-cleanup is active, the final `repaired_mask_mesh_path_connect.nii.gz` is the cleaned final output, and the pre-cleanup mask is saved separately.

## Metrics During Runs

Each run writes:

```text
metrics.json
```

Useful fields include:

```text
case_id
mask_path
template_path
target surface counts
template vertex/face counts
decoder configuration
normalization center/scale
stage metrics
best/final Chamfer
repair method
repair status
accepted/rejected mesh paths
added voxels
timing information
artifact filter metrics
post repair cleanup metrics
```

Timing keys include mesh fitting, training, voxelization, repair, post-cleanup, QA overlay, and total runtime when available.

## Evaluation Workflow

The current evaluation lives mainly in:

```text
connectivity_evaluation.ipynb
frameworks/v24/mask_mesh_fit/evaluate_cleaned_vs_repaired.py
```

For the newer aorta-style evaluation, the comparison is:

```text
raw nnUNet mask
artifact-cleaned nnUNet mask
final repaired mask
GT mask
```

The cleaned mask is usually:

```text
artifact_cleaned_mask.nii.gz
```

The final repaired mask is usually:

```text
repaired_mask_mesh_path_connect.nii.gz
```

The evaluation computes:

```text
Dice
largest-connected-component Dice
clDice
tprec
trec
false branch
length error
component count
largest-component IoU
fragmentation degree
normalized fragmentation
skeleton-graph beta0/beta1 estimates
foreground ccDice
Euler-derived beta0/beta1/beta2 and errors
```

The added BenchmarkTopoSegMetrics-inspired metrics are:

```text
ccDice
Euler beta0, beta1, beta2 counts
Euler beta0, beta1, beta2 errors
```

For aorta, some plots intentionally force the expected GT skeleton beta0 to one, because the aorta target is interpreted as one connected object. This convention is applied to the existing skeleton-beta0 comparison when requested. It should not be blindly applied to TOPCOW or to Euler-derived topology unless explicitly requested.

For TOPCOW / Circle of Willis, GT topology should generally be measured from the GT mask, because a complete one-component topology is not guaranteed in every case.

## Current Dataset-Specific Interpretation

### Aorta

The intended anatomical prior is one connected object. The best current repair strategy is usually:

```text
artifact filtering before repair
mesh_path_connect with adaptive local radius
optional post-repair cleanup
```

The goal is to reduce broken disconnected components while minimizing Dice loss from over-repair.

### TOPCOW / Circle of Willis

The topology is harder. Some cases should remain partially disconnected or should not close a full false loop. A torus prior can help, but it can also suggest anatomically wrong bridges.

For TOPCOW, bridge parameters often need to be thinner and more conservative than for aorta:

```text
smaller bridge radius
smaller max radius
smaller max added fraction
optional edge-filtered path selection
```

However, too much conservatism gives little topology improvement. TOPCOW is still an active tuning problem rather than a solved setting.

### Parse / Lung-Like Vessel Trees

Large branching structures can make mesh-path bridges too small or too large depending on radius settings and mesh fit quality. Interpret results visually and tune radius/anchor settings carefully.

## Example Aorta Repair Command

This applies repair to an already fitted mesh:

```bash
python -m mask_mesh_fit.repair_case \
  --case-id A080 \
  --mask-dir /path/to/nnunet/fold_0/validation \
  --mesh-npz /path/to/A080_voxelrepair/fitted_mesh_best_chamfer.npz \
  --output-dir /path/to/A080_voxelrepair_mesh_path_connect_adaptive_artifactfilter_10_v5_postcleanup \
  --geometric-repair-method mesh_path_connect \
  --mask-artifact-filter component_distance \
  --artifact-keep-near-main-mm 57.0 \
  --artifact-remove-distance-mm 57.5 \
  --artifact-max-remove-voxels 0 \
  --artifact-connectivity 26 \
  --path-repair-anchor-mm 3.0 \
  --path-repair-radius-mm 1.0 \
  --path-repair-min-accept-radius-mm 2.0 \
  --path-repair-max-radius-mm 5.0 \
  --path-repair-radius-mode adaptive_local \
  --path-repair-local-radius-window-mm 6.0 \
  --path-repair-radius-percentile 80 \
  --path-repair-radius-scale 1.0 \
  --path-repair-min-component-voxels 1 \
  --path-repair-max-added-fraction 0.03 \
  --path-repair-connectivity 26 \
  --post-repair-cleanup remove_mesh_uncovered_components \
  --post-cleanup-mesh-distance-mm 2.0 \
  --post-cleanup-min-close-fraction 0.01 \
  --post-cleanup-max-remove-voxels 1000 \
  --post-cleanup-connectivity 26 \
  --disable-qa
```

## Example TOPCOW Repair Command

This is the thinner TOPCOW-style adaptive repair used in several experiments:

```bash
python -m mask_mesh_fit.repair_case \
  --case-id topcow_ct_133 \
  --mask-dir /path/to/topcow/fold_0/validation \
  --mesh-npz /path/to/topcow_ct_133_voxelrepair/fitted_mesh_best_chamfer.npz \
  --output-dir /path/to/topcow_ct_133_mesh_path_connect_adaptive_postcleanup_v1 \
  --geometric-repair-method mesh_path_connect \
  --mask-artifact-filter component_distance \
  --artifact-keep-near-main-mm 57.0 \
  --artifact-remove-distance-mm 57.5 \
  --artifact-max-remove-voxels 0 \
  --artifact-connectivity 26 \
  --path-repair-anchor-mm 3.0 \
  --path-repair-radius-mm 0.25 \
  --path-repair-min-accept-radius-mm 0.35 \
  --path-repair-max-radius-mm 1.0 \
  --path-repair-radius-mode adaptive_local \
  --path-repair-local-radius-window-mm 6.0 \
  --path-repair-radius-percentile 80 \
  --path-repair-radius-scale 0.7 \
  --path-repair-min-component-voxels 1 \
  --path-repair-max-added-fraction 0.15 \
  --path-repair-connectivity 26 \
  --post-repair-cleanup remove_mesh_uncovered_components \
  --post-cleanup-mesh-distance-mm 2.0 \
  --post-cleanup-min-close-fraction 0.01 \
  --post-cleanup-max-remove-voxels 1000 \
  --post-cleanup-connectivity 26 \
  --disable-qa
```

## Known Limitations

The fitted mesh can follow nnUNet false positives if those components are not filtered before fitting. This is why the component-distance artifact filter was added.

The mesh can also propose anatomically wrong shortcuts, especially for TOPCOW. Edge-filtered path selection can reject some one-edge shortcuts, but it is not a perfect anatomical validator.

Voxelized mesh repair can fill too much volume and reduce Dice. The preferred mesh-path repair avoids voxelizing the whole mesh and adds only accepted bridge tubes.

For TOPCOW, deciding when the Circle of Willis should or should not be fully connected is not as simple as aorta. Evaluation should use GT topology rather than assuming beta0 equals one.

## File Checklist

To understand implementation:

```text
fit_case.py               end-to-end per-case fit and repair
repair_case.py            repair from existing fitted mesh
optimize.py               four-stage decoder optimization
losses.py                 Chamfer and regularizers
mesh_path_repair.py       current conservative path connector
artifact_filter.py        pre-fit / pre-repair component filtering
post_repair_cleanup.py    final mesh-uncovered component cleanup
evaluate_cleaned_vs_repaired.py
connectivity_evaluation.ipynb
```

To inspect one case:

```text
metrics.json
fitted_mesh_best_chamfer.npz
artifact_cleaned_mask.nii.gz
mesh_path_bridge_mask.nii.gz
repaired_mask_mesh_path_connect.nii.gz
post_cleanup_removed_mask.nii.gz
```

