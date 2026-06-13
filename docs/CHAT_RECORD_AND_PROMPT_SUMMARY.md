# Chat Record and Prompt Summary for Model18 Merge

User objective:

```text
Build a modular model18 pipeline from the existing PolyQ IDP DSSP work.
Keep model12 as the main DSSP classifier framework.
Use model16 calibration and model17 temporal/contact MD features.
Convert high-probability dynamic DSSP and Q length into traceable 3D multi-conformer PDBs.
Generate PyMOL/VMD-compatible visualization outputs.
```

Key design decision:

```text
Do not claim DSSP + length uniquely reconstructs 3D.
Instead: sample DSSP states, retrieve matching MD/SMD frames, rank by DSSP/geometry/contact, export real-frame PDB ensemble.
```

Requested functional steps:

```text
1. sample N DSSP sequences from DSSP probabilities
2. retrieve the most similar frames from existing MD/SMD libraries
3. rank by DSSP match + Rg + end-to-end + contact score
4. keep top-k conformers
5. output multi-conformer PDB files
6. generate PyMOL/VMD visualization figures/scripts
```

Model naming:

```text
model18 = modular DSSP-to-3D ensemble framework
```

## Latest Completed Work: Clean-300 Training, Ablation, and SCI Figures

Date context: 2026-06-12.

User requested autonomous completion of model18 training and documentation from the fixed clean-300 preprocessing database. Completed tasks:

```text
1. Verified clean-300 manifest and training outputs.
2. Added scripts/run_model18_clean300_experiments.py for reproducible epoch/ablation runs.
3. Added scripts/aggregate_model18_clean300_results.py for metric aggregation and SCI figure generation.
4. Completed epoch100/300/500 comparison under sample_stride=500.
5. Completed ablations: no dynamic, no contact_features, no contact_map_stats, no all-contact, no residue_features, no position.
6. Generated 300 dpi SCI figures and final metric CSV/Markdown reports.
7. Updated README, training documentation, SCI figure/writing plan, and this chat/task summary.
```

Final result files:

```text
docs/MODEL18_CLEAN300_FINAL_RESULTS.md
docs/MODEL18_CLEAN300_EPOCH_AND_ABLATION_RESULTS.csv
outputs/sci_figures/model18_clean300_epoch_comparison.png
outputs/sci_figures/model18_clean300_ablation_comparison.png
outputs/sci_figures/model18_clean300_sample_distribution.png
```

Key result summary:

```text
Dense full-model reference, stride100:
  accuracy = 0.8162
  macro F1 = 0.5387

Unified full-model baseline, stride500 epoch100:
  accuracy = 0.8097
  macro F1 = 0.5567

Unified no-position variant, stride500 epoch100:
  accuracy = 0.8110
  macro F1 = 0.5613

Epoch300/500 did not improve over epoch100.
All-contact removal caused the largest ablation drop.
```

Current scientific conclusion:

```text
model18 should be presented as a modular dynamic DSSP/contact/geometry framework for PolyQ IDP ensemble reconstruction. It should not be described as deterministic 3D reconstruction from DSSP alone. The best current route is dynamic DSSP probability -> frame-level geometry/contact scoring -> top-k traceable MD/SMD conformer retrieval -> PDB/PyMOL/VMD visualization.
```

Prompt summary for another Codex merge task:

```text
Continue from model18_modular_polyq_ensemble. Preserve model12-centered DSSP classifier logic and model18 modular pipeline. Use the fixed clean-300 manifest at outputs/preprocess_db_current_clean_300/model18_training_manifest.csv. Do not switch to the incomplete 372-sample dataset. Current best training outputs and figures are already generated. Next high-value improvement is to replace row-level contact-map statistics with full LxL contact/distance map CNN/GNN or residue-level temporal TCN/GRU/Transformer, while keeping the model18 retrieval/PDB/export modules unchanged.
```

## Latest User Request: Restore Full Contact-Map Model6/11/12 Branch

The user clarified that the older IDPss/model6 or earlier framework already used contact-map style information and asked to add this back into the dynamic/physical DSSP prediction route, then fuse it toward model11/model12. Completed implementation:

```text
model18/full_contact_dataset.py
model18/full_contact_map_model.py
scripts/train_model18_full_contact_map.py
docs/MODEL18_FULL_CONTACT_MAP_MODEL11_MODEL12_INTEGRATION.md
outputs/train_smoke_full_contact_epoch1/
```

Smoke test result:

```text
CUDA full-contact branch ran successfully.
train frames = 300, test frames = 120
accuracy = 0.7271, macro F1 = 0.1203 after 1 epoch small subset
```

Interpretation: this is not yet the formal 100/300/500 result; it proves the full LxL contact/distance map CNN/GNN branch can read clean300 and train. Next formal comparison should run `scripts/train_model18_full_contact_map.py --epochs 100 --sample-stride 500`.

## Latest Formal Result: Full Contact-Map Epoch100

The user requested the formal full contact-map training command. Completed:

```bash
python3 scripts/train_model18_full_contact_map.py --epochs 100 --sample-stride 500 --batch-size 128 --outdir training_outputs_clean300_full_contact_epoch100
```

Result:

```text
accuracy = 0.8430
macro F1 = 0.5871
macro precision = 0.6311
macro recall = 0.5903
train frames = 3739
test frames = 797
test residue-frame labels = 28496
device = cuda
```

Conclusion: full LxL contact/distance map CNN/GNN is now the best model18 branch and outperforms the previous MLP baseline and no-position variant.

## Latest Near-Protocol Model12 vs Model18 Comparison

To better compare with historical model12, model18 full-contact was rerun with sample_stride=100:

```bash
python3 scripts/train_model18_full_contact_map.py --epochs 100 --sample-stride 100 --batch-size 128 --outdir training_outputs_clean300_full_contact_epoch100_stride100
```

Result:

```text
accuracy = 0.8736
macro F1 = 0.6441
macro precision = 0.6648
macro recall = 0.6356
test targets = 135,136
```

Historical model12 clean300 epoch100:

```text
accuracy = 0.8805
macro precision = 0.6779
macro recall = 0.7214
test targets = 149,924
```

Conclusion: model18 full-contact is close but still slightly below historical model12 accuracy. The remaining gap likely comes from model12's old window_size=5 force-window encoder, dynamic auxiliary heads, and regularization/early stopping. Next strongest route: model19/model12-plus-contact-map = model12 force-window/dynamic heads + model18 full LxL contact-map CNN/GNN + model18 3D retrieval.

## Latest Model19 Implementation Record

Implemented model19/model12-plus-contact:

```text
model18/model19_dataset.py
model18/model19_model.py
scripts/train_model19_model12_plus_contact.py
```

Design:

```text
model18 full LxL contact-map CNN/GNN
+ model12-style temporal window encoder
+ dynamic/contact/residue fusion
```

Important data finding:

```text
clean300 sample folders do not contain raw force-window or pullf arrays.
Available arrays are dynamic_features, contact_features, contact_maps, dssp_labels, residue_features, y_masks.
```

Therefore the first model19 version used a proxy temporal window built from dynamic_features + contact_features. Formal result:

```text
model19 proxy-window stride500 epoch100:
accuracy = 0.8315
macro F1 = 0.5472
```

This is below model18 full-contact stride500:

```text
accuracy = 0.8430
macro F1 = 0.5871
```

Conclusion: model19 framework is ready, but true improvement requires raw force-window preprocessing. Proxy windows are not enough.
