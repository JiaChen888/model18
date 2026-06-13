# Chat/Task Record: Model19 Raw Force-Window Preprocessing

Date: 2026-06-13

## User Request

Preprocess and export raw force-window data following the model12 processing rule. Every original frame should be strictly preprocessed with dense sampling, saved as a reusable preprocessing database, and later training should read the saved database directly.

## Actions Completed

1. Audited the old model12/model8 force-window logic in `runner8from6a.py`.
2. Confirmed clean300 sample directories contain `sequences.npy`, `seq_masks.npy`, and frame-level `dssp_labels.npy`.
3. Implemented `scripts/preprocess_model12_force_windows.py`.
4. Exported dense raw force-window database for all 300 clean samples.
5. Updated `model18/model19_dataset.py` so model19 reads raw `force_window.npy` when a force-window manifest is provided, with fallback to the older proxy branch.
6. Updated `scripts/train_model19_model12_plus_contact.py` with `--force-window-manifest`.
7. Smoke-tested 1 epoch and 10 epoch training.
8. Ran formal model19 raw-force training at epoch100/stride500.
9. Generated model comparison CSV and 300 dpi figure.
10. Updated README, preprocessing docs, result docs, and `.gitignore`.

## Preprocessing Result

Output directory:

```text
outputs/preprocess_force_windows_clean300_model12_style/
```

Summary:

```text
samples       = 300
total_windows = 2,117,264
window_size   = 5
force_len     = 25
frame_stride  = 1
size          = about 946 MB
```

The large `.npy` database is generated locally and ignored by Git. It can be regenerated with the documented command.

## Formal Training Result

Output directory:

```text
training_outputs_clean300_model19_raw_force_epoch100_stride500/
```

Final epoch100:

```text
accuracy = 0.8411
macro F1 = 0.5702
```

Best checkpoint:

```text
epoch = 90
accuracy = 0.8421
macro F1 = 0.5843
```

## Interpretation

The true raw force-window input improves over the previous model19 proxy temporal branch and approaches the model18 full-contact stride500 baseline. It has not yet surpassed the historical model12 result, indicating that model12's remaining advantage likely comes from its original auxiliary heads, force-window regularization, split/stride details, and dynamic task formulation rather than raw force-window input alone.

## Next Recommended Step

Run the same raw force-window pipeline with stride100 and then add model12-style dynamic auxiliary heads/calibration loss under the same clean300 split.
