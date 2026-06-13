# MODEL19 Model12-Plus-Contact Proxy Result

## Implemented Route

model19 was implemented as:

```text
model18 full LxL contact-map row-CNN/DenseGraphSAGE
+ model12-style temporal window encoder
+ dynamic/contact/residue feature fusion
+ model18 output format and training reports
```

Because the fixed clean300 preprocessing database does not contain raw force-window arrays, the first model19 implementation uses a proxy temporal window:

```text
force_window_proxy = window(dynamic_features + contact_features)
window_size = 5
```

The raw force-window interface is reserved and can replace this proxy once preprocessing exports force-window tensors.

## Formal Stride500 Result

```text
command = python3 scripts/train_model19_model12_plus_contact.py --epochs 100 --sample-stride 500 --window-size 5 --batch-size 128 --outdir training_outputs_clean300_model19_epoch100_stride500
accuracy = 0.8315
macro F1 = 0.5472
macro precision = 0.5573
macro recall = 0.5509
targets = 28496
```

## Comparison

```text
model18 full-contact stride500: accuracy = 0.8430, macro F1 = 0.5871
model19 proxy-window stride500: accuracy = 0.8315, macro F1 = 0.5472
```

The proxy-window version did not improve over the full contact-map baseline. This is an important negative result: compressed dynamic/contact descriptors are not equivalent to the old model12 raw force-window branch.

## Interpretation

```text
1. The model19 architecture is implemented and trains stably.
2. The available clean300 arrays lack raw force/pullf windows.
3. Using dynamic/contact feature windows as a proxy reduces performance relative to model18 full-contact.
4. To recover model12's advantage, preprocessing must export raw force-window tensors or the old sampler's force window input.
```

## Next Required Step

Add a preprocessing step that saves, for each DSSP/contact frame:

```text
force_window.npy          [frames, window_size, force_channels]
force_window_mask.npy     [frames, window_size]
optional_force_events.npy [frames, event labels]
```

Then rerun model19 with raw force windows instead of the current proxy.

## Figure

```text
outputs/sci_figures/model19_proxy_window_vs_model18_full_contact.png
```
