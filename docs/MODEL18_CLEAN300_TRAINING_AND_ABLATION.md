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
train_rows_metadata.csv
test_rows_metadata.csv
```

## Notes

This is a lightweight MLP baseline for validating the clean 300 training database and ablation logic. It is not yet the final temporal/contact CNN/GNN model. The next model can reuse the same `model18_training_manifest.csv` and per-sample `.npy` arrays.
