# Model19 Raw Force-Window Results

## Completed Run

Command:

```bash
python3 scripts/train_model19_model12_plus_contact.py \
  --epochs 100 \
  --sample-stride 500 \
  --window-size 5 \
  --batch-size 128 \
  --force-window-manifest outputs/preprocess_force_windows_clean300_model12_style/force_window_manifest.csv \
  --outdir training_outputs_clean300_model19_raw_force_epoch100_stride500
```

Dataset:

- clean300 complete SMD samples
- Train frames at stride500: 3739
- Test frames at stride500: 797
- Test residues: 28496
- Force window source: `outputs/preprocess_force_windows_clean300_model12_style/force_window_manifest.csv`
- Device: `cuda`

## Metrics

Final epoch100:

- Accuracy: 0.841065
- Macro F1: 0.570244
- Macro precision: 0.613437
- Macro recall: 0.591767
- Mean confidence: 0.807422

Best checkpoint by accuracy:

- Epoch: 90
- Accuracy: 0.842118
- Macro F1: 0.584260
- Macro precision: 0.622538
- Macro recall: 0.585558

## Comparison

Comparison files:

```text
outputs/model19_raw_force_comparison/model19_raw_force_comparison.csv
outputs/model19_raw_force_comparison/model19_raw_force_vs_previous_models.png
```

Reference values:

| Model | Accuracy | Macro F1 | Note |
|---|---:|---:|---|
| model12 historical epoch100 | 0.8805 | NA | old model12 summary |
| model12 historical epoch500 | 0.8875 | NA | old model12 summary |
| model18 full-contact stride500 | 0.8430 | 0.5871 | previous full contact-map CNN/GNN run |
| model18 full-contact stride100 | 0.8736 | 0.6441 | denser sampling |
| model19 proxy stride500 | 0.8315 | 0.5472 | dynamic/contact proxy temporal input |
| model19 raw-force stride500 | 0.8411 | 0.5702 | true model12-style raw force-window |

## Interpretation

The raw force-window branch improves over the previous model19 proxy branch, confirming that model12-style physical force windows carry useful temporal information. At stride500/epoch100, it nearly matches but does not clearly exceed the model18 full-contact baseline. This means the current integration is useful but incomplete: it restores the raw input but not all model12 advantages.

The next optimization should keep this raw force-window database and add model12-style dynamic auxiliary heads, force-event labels, calibration loss, and stride100 dense training under the same split.

## SCI Figure Candidate

Use:

```text
outputs/model19_raw_force_comparison/model19_raw_force_vs_previous_models.png
```

Suggested caption:

Model comparison on the clean300 PolyQ SMD dataset. The model19 raw-force variant combines full residue-residue contact maps with model12-style raw pull-force windows precomputed for every DSSP frame. Raw force-window input improves over the proxy temporal branch and approaches the full-contact baseline, supporting subsequent fusion with model12 auxiliary heads and denser frame sampling.
