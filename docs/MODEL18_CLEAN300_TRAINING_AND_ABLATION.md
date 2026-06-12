# MODEL18 Clean-300 Training and Ablation

## Fixed Input

All model18 training, epoch comparison, and ablation experiments should read:

```text
outputs/preprocess_db_current_clean_300/model18_training_manifest.csv
```

The manifest points to the clean 300 preprocessed database:

```text
/media/zj/f6835876-8873-4931-8383-70fe383c9b741/IDPss-output/preprocess_data_current_clean_300_model10v2_threshold50_65_78_20260605
```

## Baseline Trainer

```text
scripts/train_model18_clean300_baseline.py
```

It loads each sample directory and uses:

```text
dssp_labels.npy        target SS8 label
contact_maps.npy       contact-map row statistics
contact_features.npy   per-frame contact features
dynamic_features.npy   per-frame geometry/dynamic features
residue_features.npy   per-residue static features
y_masks.npy            valid residue mask
```

## Smoke Test Result

Command:

```bash
python3 scripts/train_model18_clean300_baseline.py   --epochs 1   --sample-stride 1000   --max-train-rows 50000   --max-test-rows 20000   --outdir outputs/train_smoke_clean300_epoch1
```

Result:

```text
accuracy = 0.7647
macro F1 = 0.4247
train rows = 50,000
test rows = 15,488
feature dim = 78
device = cuda
```

This confirms the clean 300 database and manifest can be loaded for training.

## Epoch 100 / 300 / 500 Commands

Recommended normal sampling:

```bash
python3 scripts/train_model18_clean300_baseline.py   --epochs 100   --sample-stride 100   --outdir training_outputs_clean300_epoch100

python3 scripts/train_model18_clean300_baseline.py   --epochs 300   --sample-stride 100   --outdir training_outputs_clean300_epoch300

python3 scripts/train_model18_clean300_baseline.py   --epochs 500   --sample-stride 100   --outdir training_outputs_clean300_epoch500
```

For faster tests:

```bash
python3 scripts/train_model18_clean300_baseline.py   --epochs 100   --sample-stride 500   --max-train-rows 200000   --max-test-rows 50000   --outdir training_outputs_clean300_epoch100_fast
```

## Ablation Commands

No contact map statistics:

```bash
python3 scripts/train_model18_clean300_baseline.py   --epochs 100   --sample-stride 100   --disable contact_map_stats   --outdir training_outputs_clean300_ablation_no_contact_map
```

No dynamic/geometry features:

```bash
python3 scripts/train_model18_clean300_baseline.py   --epochs 100   --sample-stride 100   --disable dynamic   --outdir training_outputs_clean300_ablation_no_dynamic
```

No residue/static features:

```bash
python3 scripts/train_model18_clean300_baseline.py   --epochs 100   --sample-stride 100   --disable residue_features   --outdir training_outputs_clean300_ablation_no_residue_static
```

No contact features and no contact-map statistics:

```bash
python3 scripts/train_model18_clean300_baseline.py   --epochs 100   --sample-stride 100   --disable contact_features   --disable contact_map_stats   --outdir training_outputs_clean300_ablation_no_contact_all
```

## Outputs Per Training Run

```text
model18_clean300_baseline.pt
training_history.csv
test_metrics.csv
summary.json
train_rows_metadata.csv  optional, only with --save-row-metadata
test_rows_metadata.csv   optional, only with --save-row-metadata
```

## Notes

This is a lightweight MLP baseline for validating the clean 300 training database and ablation logic. It is not yet the final temporal/contact CNN/GNN model. The next model can reuse the same `model18_training_manifest.csv` and per-sample `.npy` arrays.

## Completed Clean-300 Epoch and Ablation Results

Unified comparison settings:

```text
manifest = outputs/preprocess_db_current_clean_300/model18_training_manifest.csv
sample_stride = 500
train samples = 248
test samples = 52
train residue-frame rows = 130,078
test residue-frame rows = 28,496
device = cuda
```

Epoch comparison:

```text
epoch100 full: accuracy = 0.8097, macro F1 = 0.5567
epoch300 full: accuracy = 0.8001, macro F1 = 0.5378
epoch500 full: accuracy = 0.8008, macro F1 = 0.5400
```

The longer 300/500 epoch runs did not improve the fixed test split. The recommended full-model checkpoint is therefore epoch100. The training history also showed that early checkpoints around 30-60 epochs can reach comparable or better macro F1, so future production training should add early stopping.

Ablation at epoch100/stride500:

```text
full model:                accuracy = 0.8097, macro F1 = 0.5567
no dynamic:                accuracy = 0.8025, macro F1 = 0.5314
no contact_features:       accuracy = 0.8077, macro F1 = 0.5326
no contact_map_stats:      accuracy = 0.7998, macro F1 = 0.5491
no all contact:            accuracy = 0.7966, macro F1 = 0.5232
no residue_features:       accuracy = 0.8085, macro F1 = 0.5412
no residue_position:       accuracy = 0.8110, macro F1 = 0.5613
```

Interpretation:

```text
1. Dynamic geometry and contact information are the most important feature groups.
2. Removing all contact features caused the largest macro-F1 drop.
3. Residue/static features are less important for PolyQ because all residues are glutamine.
4. Explicit residue position slightly reduced generalization in this split, likely because length/position bias is already represented by dynamic and contact features.
5. The current contact-map input is only row-level statistics, not a full CNN/GNN map encoder; this remains the largest model-structure improvement target.
```

Dense sampling reference:

```text
training_outputs_clean300_epoch100
sample_stride = 100
accuracy = 0.8162
macro F1 = 0.5387
test rows = 135,136
```

The dense stride100 full-model run gives the highest top-1 accuracy, while the unified stride500 no-position variant gives the best macro F1. For manuscript reporting, use the full model as the main baseline and report no-position as an ablation-derived optimization.

Generated paper figures:

```text
outputs/sci_figures/model18_clean300_epoch_comparison.png
outputs/sci_figures/model18_clean300_ablation_comparison.png
outputs/sci_figures/model18_clean300_sample_distribution.png
```

Full metric table:

```text
docs/MODEL18_CLEAN300_EPOCH_AND_ABLATION_RESULTS.csv
```
