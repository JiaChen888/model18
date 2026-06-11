# MODEL18 Backend Epoch Selection

model18 itself is a modular retrieval and reconstruction pipeline and does not train by epoch. The epoch 100/300/500 comparison refers to the model17_v2 temporal/contact backend that feeds model18 with dynamic DSSP/contact information.

## Q36 Holdout Results

| epoch | accuracy | macro F1 | ECE | Brier | decision |
|---:|---:|---:|---:|---:|---|
| 100 | 0.8889 | 0.3137 | 0.1216 | 0.2434 | recommended |
| 300 | 0.8889 | 0.3137 | 0.2237 | 0.2731 | over-confident |
| 500 | 0.8889 | 0.3137 | 0.3194 | 0.3362 | over-confident |

## Interpretation

All three epochs have the same top1 accuracy on the Q36 grouped holdout. Longer training does not improve classification but worsens calibration. Therefore, model18 should use epoch100 as the preferred model17_v2 backend when a trained temporal/contact backend is needed.

## Scientific Boundary

These metrics are evaluated against model12/model16-derived labels, not mkdssp or experimental DSSP labels. The result supports model18 backend selection, not a claim of experimentally validated DSSP improvement.

## Files

```text
docs/MODEL17_V2_EPOCH100_300_500_BACKEND_METRICS.csv
outputs/sci_figures/model17_v2_epoch100_300_500_backend_comparison.png
```
