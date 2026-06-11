# MODEL18 Preprocessed Database

## Fixed Training Database

Use the clean 300 database for training, epoch comparison, and ablation experiments:

```text
/media/zj/f6835876-8873-4931-8383-70fe383c9b741/IDPss-output/preprocess_data_current_clean_300_model10v2_threshold50_65_78_20260605
```

This database is preferred over the 372-sample `all_data_new_v2` source because the 372 set contains incomplete or warning trajectories. The clean 300 set has already passed the threshold/QC rule used by earlier model6/model8 runs.

## Per-Sample Files

Each sample directory contains per-frame and per-residue arrays:

```text
sequences.npy          force/time-series-like scalar input used by earlier models
dssp_labels.npy        per-frame SS8 labels, padded to 46 residues
contact_maps.npy       per-frame CA contact/distance map tensor
contact_features.npy   per-frame contact summary features
dynamic_features.npy   per-frame geometry/dynamic features
residue_features.npy   per-residue static features
seq_masks.npy          sequence/time mask
y_masks.npy            valid residue mask; used to infer true Q length
```

`sequence_times_ps.npy` exists in some older preprocess databases, but not in the clean 300 database. model18 treats it as optional.

## Length Distribution

After fixing true length inference from `y_masks.npy`:

| length | samples | ready |
|---:|---:|---:|
| Q22 | 111 | 111 |
| Q36 | 103 | 103 |
| Q46 | 86 | 86 |

Total ready samples: 300.

## Train/Test Split

The training manifest reuses old model8from6a split indices from:

```text
/media/zj/f6835876-8873-4931-8383-70fe383c9b741/IDPss-output/model_current_clean_300_epoch100_model8from6a_20260605
```

Current manifest split:

| length | train | test |
|---:|---:|---:|
| Q22 | 93 | 18 |
| Q36 | 86 | 17 |
| Q46 | 69 | 17 |

## Step Commands

```bash
cd /home/zj/jyxl/test_project/516/graph_idpss/model18_modular_polyq_ensemble
./scripts/run_model18_preprocess_steps.sh
```

Equivalent explicit commands:

```bash
python3 scripts/preprocess_step1_build_index.py   --process-dir /media/zj/f6835876-8873-4931-8383-70fe383c9b741/IDPss-output/preprocess_data_current_clean_300_model10v2_threshold50_65_78_20260605   --outdir outputs/preprocess_db_current_clean_300

python3 scripts/preprocess_step2_smoke_validate_arrays.py   --index outputs/preprocess_db_current_clean_300/preprocessed_sample_index.csv   --outdir outputs/preprocess_db_current_clean_300   --max-samples 12

python3 scripts/preprocess_step3_build_training_manifest.py   --index outputs/preprocess_db_current_clean_300/preprocessed_sample_index.csv   --outdir outputs/preprocess_db_current_clean_300   --split-source /media/zj/f6835876-8873-4931-8383-70fe383c9b741/IDPss-output/model_current_clean_300_epoch100_model8from6a_20260605
```

## Output Files

```text
outputs/preprocess_db_current_clean_300/preprocessed_sample_index.csv
outputs/preprocess_db_current_clean_300/preprocessed_sample_summary_by_length.csv
outputs/preprocess_db_current_clean_300/preprocessed_smoke_validation.csv
outputs/preprocess_db_current_clean_300/model18_training_manifest.csv
outputs/preprocess_db_current_clean_300/model18_training_manifest_summary.csv
```

## Usage Policy

All later model18 training, epoch 100/300/500 comparisons, and ablation experiments should read from `model18_training_manifest.csv` and the clean 300 per-sample database. The raw 372 sample set can remain an inference/candidate source, but should not replace the clean 300 training database without re-QC.
