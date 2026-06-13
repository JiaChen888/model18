# model18: Modular PolyQ Dynamic DSSP to 3D Ensemble

model18 upgrades the previous model12/model16/model17 workflow into a modular pipeline. The main DSSP classifier remains model12-centered, while model16/model17 provide calibrated probabilities, temporal/contact features, and MD/SMD-backed conformer retrieval.

## Core Idea

```text
PolyQ sequence length
+ model12/model16 DSSP probability profile
+ DSSP sequence sampling
+ MD/SMD trajectory library
+ geometry/contact ranking
-> traceable top-k 3D conformer PDB ensemble
```

This is designed for IDP systems such as PolyQ, where a single native 3D structure is not expected. model18 reconstructs a conformational ensemble rather than one fixed fold.

## Module Layout

```text
model18/dssp_classifier.py      model12-centered DSSP probability interface
model18/dssp_sampler.py         sample N DSSP sequences from residue probabilities
model18/trajectory_library.py   load MD/SMD frame-level DSSP, geometry, contact features
model18/ranker.py               rank frames by DSSP, geometry, contact consistency
model18/pdb_exporter.py         export true trajectory frames as PDB using MDAnalysis
model18/visualization.py        3D cartoon plot and PyMOL script generation
model18/pipeline.py             end-to-end orchestration
scripts/run_model18_ensemble.py command-line entry point
```

## Current Default Inputs

The default implementation uses model17 tables as the trajectory/probability backend:

```text
../model17_q46md_temporal_contact/tables/model17_multilength_temporal_contact_table.csv
../model17_q46md_temporal_contact/tables/independent_md_multilength_geometry_contact_features.csv
```

These tables contain Q36/Q46 independent MD features and model12/model16-derived SS8 soft probabilities. No mkdssp is used.







## Preprocessed Clean 300 Database

Training, epoch comparison, and ablation experiments should use the clean 300 preprocessed database:

```text
/media/zj/f6835876-8873-4931-8383-70fe383c9b741/IDPss-output/preprocess_data_current_clean_300_model10v2_threshold50_65_78_20260605
```

model18 provides three preprocessing/indexing steps that reuse this saved database instead of recomputing every trajectory each time:

```bash
./scripts/run_model18_preprocess_steps.sh
```

Detected ready samples:

```text
Q22 = 111
Q36 = 103
Q46 = 86
Total = 300
```

Generated manifest:

```text
outputs/preprocess_db_current_clean_300/model18_training_manifest.csv
```

See `docs/MODEL18_PREPROCESS_DATABASE.md`.



## Clean-300 Training Entry

Training now reads only the saved clean-300 preprocessing database through:

```text
outputs/preprocess_db_current_clean_300/model18_training_manifest.csv
```

Smoke-tested trainer:

```bash
python3 scripts/train_model18_clean300_baseline.py   --epochs 1   --sample-stride 1000   --max-train-rows 50000   --max-test-rows 20000   --outdir outputs/train_smoke_clean300_epoch1
```

Smoke result:

```text
accuracy = 0.7647
macro F1 = 0.4247
device = cuda
```

Full epoch commands and ablation commands are documented in `docs/MODEL18_CLEAN300_TRAINING_AND_ABLATION.md`.

Final clean-300 result summary:

```text
Best dense full-model run:
  training_outputs_clean300_epoch100
  sample_stride = 100
  top-1 accuracy = 0.8162
  macro F1 = 0.5387

Best unified stride500 model18 variant:
  training_outputs_clean300_ablation_no_position_epoch100
  top-1 accuracy = 0.8110
  macro F1 = 0.5613

Full stride500 baseline:
  training_outputs_clean300_epoch100_stride500
  top-1 accuracy = 0.8097
  macro F1 = 0.5567
```

Epoch 300 and 500 did not improve over epoch 100 on the fixed split, indicating mild overfitting with longer optimization. Contact and dynamic features were the most useful feature groups in ablation. Explicit residue position was slightly harmful under the current split and is therefore treated as an optional calibration feature rather than a required input.

Paper-ready result files:

```text
docs/MODEL18_CLEAN300_FINAL_RESULTS.md
docs/MODEL18_CLEAN300_EPOCH_AND_ABLATION_RESULTS.csv
outputs/sci_figures/model18_clean300_epoch_comparison.png
outputs/sci_figures/model18_clean300_ablation_comparison.png
outputs/sci_figures/model18_clean300_sample_distribution.png
```


## Model19 Raw Force-Window Preprocessing

To restore the model12-style physical force input, model19 now supports a precomputed raw force-window database. For every clean300 DSSP/contact frame, the preprocessing exports the previous 25 raw pull-force points using the original model12 rule:

```text
force_end = frame_index * 5 + 1
force_start = max(0, force_end - window_size * 5)
```

Generated database:

```text
outputs/preprocess_force_windows_clean300_model12_style/
  force_window_manifest.csv
  summary.json
  {sample_id}/force_window.npy
  {sample_id}/force_window_mask.npy
  {sample_id}/force_window_metadata.csv
```

Current clean300 export:

```text
samples = 300
total force windows = 2,117,264
window_size = 5
force_len = 25
frame_stride = 1
local size = about 946 MB
```

Run preprocessing:

```bash
python3 scripts/preprocess_model12_force_windows.py \
  --window-size 5 \
  --frame-stride 1 \
  --outdir outputs/preprocess_force_windows_clean300_model12_style
```

Train model19 with true raw force windows:

```bash
python3 scripts/train_model19_model12_plus_contact.py \
  --epochs 100 \
  --sample-stride 500 \
  --window-size 5 \
  --batch-size 128 \
  --force-window-manifest outputs/preprocess_force_windows_clean300_model12_style/force_window_manifest.csv \
  --outdir training_outputs_clean300_model19_raw_force_epoch100_stride500
```

Result at epoch100/stride500:

```text
final accuracy = 0.8411
final macro F1 = 0.5702
best epoch = 90
best accuracy = 0.8421
best macro F1 = 0.5843
```

The raw force-window branch improves over the previous model19 proxy temporal input and nearly matches model18 full-contact at stride500. See:

```text
docs/MODEL19_RAW_FORCE_WINDOW_PREPROCESSING.md
docs/MODEL19_RAW_FORCE_WINDOW_RESULTS.md
outputs/model19_raw_force_comparison/model19_raw_force_comparison.csv
outputs/model19_raw_force_comparison/model19_raw_force_vs_previous_models.png
```

## Ablation Switches

model18 reserves configurable ablation switches for later systematic experiments:

```text
use_dssp_position
use_dssp_composition
use_geometry
use_contact
use_stress_adjusted_dssp
use_early_smd_filter
use_learned_ranking
export_pdb
make_visualization
```

Fast ablation smoke test:

```bash
python3 scripts/run_model18_ensemble.py --config configs/ablation_no_geom_contact_q46.json --length 46 --samples 50 --top-k 5
```

Equivalent CLI override:

```bash
python3 scripts/run_model18_ensemble.py --length 46 --disable geometry --disable contact --disable pdb --disable visualization
```

Each run records:

```text
ablation_config_used.json
model18_run_summary.json
```

See `docs/MODEL18_ABLATION_SWITCHES.md`.

## SMD 300-Sample Library Status

The complete Q22/Q36/Q46 300-sample SMD library is not yet fully read into model18. A local audit found:

```text
Q22: 252 run directories, 239 ready_basic
Q36: 188 run directories, 174 ready_basic
Q46: 80 run directories, 56 ready_basic
```

Current model18 retrieval backend still uses the model17 independent MD subset:

```text
Q46-MD main/v2/v3/1000ns + Q36-MD/v2
runs = 5
frames = 629
Q22 not yet included in model18 retrieval backend
```

New modules have been added for the requested upgrades:

```text
model18/smd_audit.py
model18/early_smd_filter.py
model18/stress_adjusted_dssp.py
model18/learned_ranking.py
```

See:

```text
docs/POLYQ_SMD_TRAJECTORY_LIBRARY_AUDIT.csv
docs/POLYQ_SMD_TRAJECTORY_LIBRARY_AUDIT_SUMMARY.csv
docs/MODEL18_SMD_300_LIBRARY_AUDIT_AND_OPTIMIZATION.md
```

## Run Demo

```bash
python3 scripts/run_model18_ensemble.py --length 46 --samples 200 --top-k 5
```

For Q36:

```bash
python3 scripts/run_model18_ensemble.py --length 36 --samples 200 --top-k 5
```

## Outputs

Default output directory:

```text
outputs/demo_q46/
```

Main files:

```text
dssp_probability_profile.csv       residue-wise SS8 probability profile
sampled_dssp_frequency.csv         sampled DSSP frequency table
consensus_dssp.txt                 high-frequency consensus DSSP sequence
top_ranked_conformer_scores.csv    DSSP/geometry/contact ranking scores
pdb/*.pdb                          top-k full-protein PDB conformers
model18_top_conformer_cartoon3d.png 3D cartoon contact sheet
open_in_pymol.pml                  PyMOL script for direct visualization
model18_run_summary.json           run summary
```

## PyMOL Visualization

```bash
pymol outputs/demo_q46/open_in_pymol.pml
```

or directly:

```bash
pymol outputs/demo_q46/pdb/*.pdb
```

## Scientific Interpretation

model18 does not claim that DSSP labels alone uniquely determine 3D structure. It uses DSSP probabilities as constraints and retrieves traceable real conformers from MD/SMD trajectory frames. The resulting PDBs are therefore ensemble candidates consistent with dynamic DSSP and geometry/contact statistics.

## Best Current Model Stack

```text
model12: primary DSSP classifier framework
model16: calibrated probability layer
model17: temporal/contact MD expansion
model18: modular DSSP-to-3D ensemble reconstruction pipeline
```


## Full Test Status

A complete Q36/Q46 test was run after the modular split.

```text
python3 -m py_compile model18/*.py scripts/*.py
python3 scripts/run_model18_ensemble.py --length 46 --samples 200 --top-k 5
python3 scripts/run_model18_ensemble.py --config configs/q36_paths.json --length 36 --samples 200 --top-k 5
```

Validation results:

```text
Q46: 5 full-protein PDB files, each 785 ATOM records and 46 CA atoms
Q36: 5 full-protein PDB files, each 615 ATOM records and 36 CA atoms
PNG/PML/CSV outputs generated for both groups
```

Detailed reports:

```text
docs/MODEL18_FULL_TEST_REPORT.md
docs/MODEL18_FULL_TEST_PDB_CHECK.csv
docs/MODEL18_SCI_FIGURE_AND_WRITING_PLAN.md
```



## model17_v2 Backend Epoch Selection

model18 does not train by epoch. The epoch 100/300/500 comparison belongs to the model17_v2 temporal/contact backend.

```text
epoch100: accuracy = 0.8889, ECE = 0.1216, Brier = 0.2434  recommended
epoch300: accuracy = 0.8889, ECE = 0.2237, Brier = 0.2731
epoch500: accuracy = 0.8889, ECE = 0.3194, Brier = 0.3362
```

Conclusion: use epoch100 as the preferred model17_v2 backend because longer training does not improve top1 accuracy and worsens calibration.

See:

```text
docs/MODEL18_MODEL17_EPOCH_BACKEND_SELECTION.md
outputs/sci_figures/model17_v2_epoch100_300_500_backend_comparison.png
```

## SCI Figure Recommendation

Use these as the main manuscript candidates:

```text
1. docs/MODEL18_ARCHITECTURE.svg
2. Q36/Q46 DSSP probability and sampled-frequency profiles
3. outputs/demo_q46/model18_top_conformer_cartoon3d.png
4. outputs/demo_q36/model18_top_conformer_cartoon3d.png
5. outputs/sci_figures/demo_q36_score_decomposition.png
6. outputs/sci_figures/demo_q46_score_decomposition.png
7. PyMOL ray-traced images generated from open_in_pymol.pml
```

## Full Contact-Map CNN/GNN Branch

model18 now includes a full LxL contact/distance map branch derived from the older IDPss/model6/model11/model12 graph-contact design:

```text
model18/full_contact_dataset.py
model18/full_contact_map_model.py
scripts/train_model18_full_contact_map.py
docs/MODEL18_FULL_CONTACT_MAP_MODEL11_MODEL12_INTEGRATION.md
```

This branch reads `contact_maps.npy` as a complete residue-residue matrix, converts it to a sparse top-k contact-strength graph, encodes contact rows with CNN, applies DenseGraphSAGE message passing, and fuses the graph embedding with dynamic/contact/residue features for per-residue SS8 prediction. A CUDA smoke test completed successfully in `outputs/train_smoke_full_contact_epoch1/`.

Recommended formal comparison:

```bash
python3 scripts/train_model18_full_contact_map.py --epochs 100 --sample-stride 500 --batch-size 128 --outdir training_outputs_clean300_full_contact_epoch100
```

## Current Best Model18 Result

The formal full contact-map epoch100 run is now the best clean-300 model18 result:

```text
run = training_outputs_clean300_full_contact_epoch100
model = FullContactMapDSSPModel
sample_stride = 500
batch_size = 128
train frames = 3739
test frames = 797
test residue-frame labels = 28496
accuracy = 0.8430
macro F1 = 0.5871
```

This improves over the previous MLP full baseline (`accuracy = 0.8097`, `macro F1 = 0.5567`) and the previous MLP no-position ablation (`accuracy = 0.8110`, `macro F1 = 0.5613`). The current best route is therefore:

```text
model12-centered DSSP objective
+ model6/model11/model12 full contact-map graph encoder
+ dynamic/contact/residue feature fusion
+ model18 ensemble retrieval and 3D PDB visualization
```

Result files:

```text
docs/MODEL18_FULL_CONTACT_EPOCH100_RESULT.md
docs/MODEL18_FULL_CONTACT_COMPARISON_RESULTS.csv
outputs/sci_figures/model18_full_contact_vs_mlp_comparison.png
training_outputs_clean300_full_contact_epoch100/
```

## Model12 vs Model18 Full-Contact Near-Protocol Comparison

A stride100 model18 full-contact run was completed to better match the historical model12 clean300 settings:

```text
model18 full-contact stride100 epoch100:
  accuracy = 0.8736
  macro F1 = 0.6441
  macro precision = 0.6648
  macro recall = 0.6356
  targets = 135,136

model12 epoch100 historical clean300:
  accuracy = 0.8805
  macro precision = 0.6779
  macro recall = 0.7214
  targets = 149,924
```

The protocols are close but not identical. model12 used the old `window_size=5` sampler and force-window encoder; model18 full-contact uses frame-level manifest sampling and does not yet include the full model12 force-window branch. The best next version should merge model12's force-window/dynamic auxiliary heads with model18's full LxL contact-map CNN/GNN branch.

See:

```text
docs/MODEL12_VS_MODEL18_FULL_CONTACT_CLEAN300_COMPARISON.md
outputs/sci_figures/model12_vs_model18_full_contact_clean300_comparison.png
training_outputs_clean300_full_contact_epoch100_stride100/
```

## Model19 / Model12-Plus-Contact Status

model19 has been implemented as a model12-style temporal window branch fused with the model18 full LxL contact-map CNN/GNN branch:

```text
model18/model19_dataset.py
model18/model19_model.py
scripts/train_model19_model12_plus_contact.py
```

Current clean300 does not contain raw force-window tensors, so the first model19 test used a proxy window built from `dynamic_features + contact_features`.

Formal stride500 result:

```text
model18 full-contact:  accuracy = 0.8430, macro F1 = 0.5871
model19 proxy-window:  accuracy = 0.8315, macro F1 = 0.5472
```

Conclusion: the architecture is implemented and stable, but the proxy window does not recover model12's raw force-window advantage. The next required step is to export raw force-window tensors from preprocessing and rerun model19.

See:

```text
docs/MODEL19_MODEL12_PLUS_CONTACT_PROXY_RESULT.md
outputs/sci_figures/model19_proxy_window_vs_model18_full_contact.png
training_outputs_clean300_model19_epoch100_stride500/
```
