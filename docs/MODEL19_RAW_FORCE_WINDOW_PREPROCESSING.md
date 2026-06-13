# Model19 Raw Force-Window Preprocessing

## Purpose

This preprocessing step exports the model12-style raw pull-force window for every DSSP/contact frame in the clean300 PolyQ dataset. The saved database is then read directly by model19/model12-plus-contact-map training, avoiding repeated on-the-fly slicing and ensuring that epoch/ablation experiments use identical force-window inputs.

## Source Dataset

- Clean dataset: `outputs/preprocess_db_current_clean_300/model18_training_manifest.csv`
- External raw sample root recorded in the manifest: `/media/zj/f6835876-8873-4931-8383-70fe383c9b741/IDPss-output/preprocess_data_current_clean_300_model10v2_threshold50_65_78_20260605`
- Sample count: 300 complete SMD samples
- Length groups: Q22/Q36/Q46

Each sample directory must contain:

- `sequences.npy`: raw force sequence
- `seq_masks.npy`: valid force-point mask
- `dssp_labels.npy`: frame-level DSSP labels

## Model12-Compatible Window Rule

For DSSP frame `i`, model12 maps the frame to raw force points by:

```python
seq_len = window_size * 5
force_end = i * 5 + 1
force_start = max(0, force_end - seq_len)
window = sequences[force_start:force_end]
```

With the default `window_size=5`, each frame gets a 25-point raw force window. Short initial windows are left-padded with zeros and a matching zero mask.

## Generated Database

Main output directory:

```text
outputs/preprocess_force_windows_clean300_model12_style/
```

Per sample:

```text
{sample_id}/force_window.npy
{sample_id}/force_window_mask.npy
{sample_id}/force_window_metadata.csv
```

Global files:

```text
force_window_manifest.csv
summary.json
```

Current summary:

```text
samples       = 300
total_windows = 2,117,264
force_len     = 25
frame_stride  = 1
size          = about 946 MB
```

## Commands

Smoke test:

```bash
python3 scripts/preprocess_model12_force_windows.py \
  --window-size 5 \
  --frame-stride 1 \
  --max-samples 3 \
  --outdir outputs/preprocess_force_windows_clean300_model12_style_smoke
```

Full clean300 preprocessing:

```bash
python3 scripts/preprocess_model12_force_windows.py \
  --window-size 5 \
  --frame-stride 1 \
  --outdir outputs/preprocess_force_windows_clean300_model12_style
```

Train model19 with raw force-window input:

```bash
python3 scripts/train_model19_model12_plus_contact.py \
  --epochs 100 \
  --sample-stride 500 \
  --window-size 5 \
  --batch-size 128 \
  --force-window-manifest outputs/preprocess_force_windows_clean300_model12_style/force_window_manifest.csv \
  --outdir training_outputs_clean300_model19_raw_force_epoch100_stride500
```

## Notes

- This is raw model12-style force-window preprocessing, not proxy dynamic/contact windowing.
- `Clean300Model19WindowDataset` now prefers `force_window.npy` when a force-window manifest is provided. If absent, it falls back to the older dynamic/contact proxy window for compatibility.
- The `.npy` database is large and should be treated as generated data. Commit the scripts, manifests, summaries, and documentation; keep the full array database in local/project storage or external data storage.
