# MODEL18 Full Contact-Map Epoch100 Result

## Official Run

```text
command = python3 scripts/train_model18_full_contact_map.py --epochs 100 --sample-stride 500 --batch-size 128 --outdir training_outputs_clean300_full_contact_epoch100
model = FullContactMapDSSPModel
train frames = 3739
test frames = 797
test residue-frame labels = 28496
device = cuda
```

## Result

```text
accuracy = 0.8430
macro F1 = 0.5871
macro precision = 0.6311
macro recall = 0.5903
```

## Comparison

```text
MLP full stride500:       accuracy = 0.8097, macro F1 = 0.5567
MLP no-position variant:  accuracy = 0.8110, macro F1 = 0.5613
Full contact-map CNN/GNN: accuracy = 0.8430, macro F1 = 0.5871
```

The full contact-map branch is now the best model18 clean-300 model. Compared with the MLP full baseline, accuracy increased by 0.0333 and macro F1 increased by 0.0304. Compared with the best previous MLP ablation variant, accuracy increased by 0.0320 and macro F1 increased by 0.0258.

## Generated Figure

```text
outputs/sci_figures/model18_full_contact_vs_mlp_comparison.png
```

## SCI Caption

**Figure X. Full contact-map branch improves dynamic DSSP prediction in model18.** The full contact-map CNN/GNN branch, derived from the earlier IDPss/model6 and model11/model12 graph-contact framework, was compared with the frame-residue MLP baseline under the same clean-300 split and sample stride. Using the complete LxL contact/distance map improved both top-1 accuracy and macro F1, demonstrating that residue-residue topology provides information beyond scalar contact descriptors.

## Manuscript Text

Restoring the complete contact/distance-map representation substantially improved model18 performance. Under the fixed clean-300 split, the full contact-map CNN/GNN branch achieved an accuracy of 0.8430 and a macro F1 of 0.5871, outperforming both the full MLP baseline and the previous no-position MLP variant. This confirms that the earlier IDPss/model6-style graph-contact representation captures residue-residue topological information that is not retained by scalar contact row summaries.
