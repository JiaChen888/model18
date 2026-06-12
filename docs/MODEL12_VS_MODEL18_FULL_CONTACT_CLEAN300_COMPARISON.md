# Model12 vs Model18 Full-Contact Clean-300 Comparison

## Purpose

This comparison addresses why the newest model18 full-contact branch still does not exactly match the highest historical model12 accuracy. The closest available comparison uses clean300 and sample_stride=100, but the protocols are not perfectly identical:

```text
model18 full-contact: frame-level manifest split, no force-window encoder, targets = 135136
model12 history: old sampler, window_size=5, stride=100, sample split seed42, targets = 149924
```

Therefore this is a near-protocol comparison, not a mathematically identical checkpoint evaluation.

## Results

| Model | Accuracy | Macro precision | Macro recall | Targets |
|---|---:|---:|---:|---:|
| model18 full-contact epoch100 stride100 | 0.8736 | 0.6648 | 0.6356 | 135136 |
| model12 epoch100 historical clean300 | 0.8805 | 0.6779 | 0.7214 | 149924 |
| model12 epoch500 historical clean300 | 0.8875 | 0.6896 | 0.7471 | 149924 |

## Interpretation

model18 full-contact stride100 substantially improves over the previous model18 stride500 run and reaches:

```text
accuracy = 0.8736
macro precision = 0.6648
macro recall = 0.6356
macro F1 = 0.6441
```

It remains slightly below historical model12 epoch100 accuracy by 0.0069, and below model12 epoch500 by 0.0139. The most likely causes are:

```text
1. model12 uses the old window_size=5 sampler and force-window encoder.
2. model12 includes dynamic auxiliary heads inherited from model11 plus model12 regularization/early stopping.
3. model18 full-contact currently uses frame-level contact/dynamic descriptors but not the full model12 force-window branch.
4. The target counts differ, so the evaluation pools are close but not identical.
```

## Conclusion

The next best model should merge both advantages:

```text
model12 force-window + dynamic auxiliary heads + model12 regularization
+ model18 full LxL contact-map row-CNN/DenseGraphSAGE branch
+ model18 retrieval/PDB/3D ensemble modules
```

This is the strongest route for a model19/model12-plus-contact-map version.

## Figure

```text
outputs/sci_figures/model12_vs_model18_full_contact_clean300_comparison.png
```
