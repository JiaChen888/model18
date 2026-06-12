# MODEL18 Full Contact-Map Integration with Model6/11/12

## Why This Was Added

The previous clean-300 model18 MLP baseline used only four scalar row statistics from each contact-map row:

```text
mean, std, max, sum
```

That is useful for fast ablation, but it is not the same as the older IDPss/model6/model11/model12 framework. The older framework already used the complete contact/distance matrix as a graph input.

## Reused Older-Model Design

Local source files inspected:

```text
graph_idpss/model6.py
graph_idpss/model8from6a.py
graph_idpss/model11_dynamic_multitask.py
graph_idpss/model12_regularized_multitask.py
```

Key inherited ideas:

```text
1. contact_to_topk_strength(contact)
   - distance-like maps are converted by exp(-d/sigma)
   - contact-like maps are kept as strengths
   - diagonal is removed
   - top-k contacts per residue are retained

2. DenseGraphSAGE
   - graph message passing over the full LxL contact/distance matrix
   - residue/node features are updated by neighbor aggregation

3. Multimodal fusion
   - sequence/length/residue features
   - graph/contact features
   - dynamic/force/physical features
   - DSSP head plus auxiliary physical heads

4. model11/model12 objective
   - main output remains residue-level SS8 DSSP
   - auxiliary outputs regularize dynamic physical features
```

## New model18 Files

```text
model18/full_contact_dataset.py
model18/full_contact_map_model.py
scripts/train_model18_full_contact_map.py
```

The new branch reads the same fixed clean-300 manifest:

```text
outputs/preprocess_db_current_clean_300/model18_training_manifest.csv
```

and loads each sample directory's saved arrays:

```text
dssp_labels.npy        residue-level SS8 target
contact_maps.npy       full LxL contact/distance map
contact_features.npy   frame-level contact descriptors
dynamic_features.npy   frame-level dynamic/geometry descriptors
residue_features.npy   per-residue descriptors
y_masks.npy            valid residue mask
```

## New Architecture

```text
Full LxL contact/distance map
  -> contact_to_topk_strength
  -> row-CNN over each residue's full contact row
  -> DenseGraphSAGE message passing
  -> graph/contact residue embedding

Residue features + optional position
  -> node embedding

Dynamic features + contact_features
  -> frame scalar embedding

[node embedding, row-CNN embedding, graph embedding, scalar embedding]
  -> multimodal fusion MLP
  -> per-residue SS8 DSSP logits
  -> auxiliary contact-number/contact-density heads
```

This is the model18-compatible version of the model6/model11/model12 contact-map path. It is not a replacement for model12; it is a full-map contact branch that can be fused back into the model12-centered DSSP classifier.

## Smoke Test Completed

Command:

```bash
python3 scripts/train_model18_full_contact_map.py   --epochs 1   --sample-stride 1000   --max-train-frames 300   --max-test-frames 120   --batch-size 64   --outdir outputs/train_smoke_full_contact_epoch1
```

Result:

```text
device = cuda
train frames = 300
test frames = 120
test residue labels = 4214
accuracy = 0.7271
macro F1 = 0.1203
```

The 1-epoch small-subset result is only a smoke test. It confirms that clean-300 full contact maps can be read, moved to CUDA, trained, evaluated, and saved.

## Recommended Next Formal Runs

Fast comparison:

```bash
python3 scripts/train_model18_full_contact_map.py   --epochs 100   --sample-stride 500   --batch-size 128   --outdir training_outputs_clean300_full_contact_epoch100
```

Ablation without full contact map:

```bash
python3 scripts/train_model18_full_contact_map.py   --epochs 100   --sample-stride 500   --batch-size 128   --disable full_contact_map   --outdir training_outputs_clean300_full_contact_ablation_no_full_map_epoch100
```

Ablation without dynamic geometry:

```bash
python3 scripts/train_model18_full_contact_map.py   --epochs 100   --sample-stride 500   --batch-size 128   --disable dynamic   --outdir training_outputs_clean300_full_contact_ablation_no_dynamic_epoch100
```

## How It Upgrades Model11/12

Current recommended fusion route:

```text
IDPss/model6 full contact graph encoder
  + model11 dynamic physical auxiliary heads
  + model12 regularization/early stopping
  + model18 modular retrieval/PDB/visualization output
  -> next model: model12-centered full-contact dynamic DSSP ensemble model
```

Preserve model11/model12 outputs:

```text
1. residue-level DSSP logits/probabilities
2. dynamic DSSP labels
3. auxiliary contact/force/geometry descriptors
4. downstream top-k 3D conformer retrieval
5. PyMOL/VMD-compatible PDB ensemble
```

Add to model12/model18:

```text
1. full LxL contact/distance map CNN/GNN embedding
2. contact-number/contact-density auxiliary loss
3. no-position option, because model18 ablation found explicit position can slightly hurt generalization
4. early stopping, because epoch300/500 did not improve over epoch100
```

## SCI Wording

```text
To evaluate whether topological contact information contributed beyond scalar contact descriptors, we added a full contact-map branch derived from the original IDPss/model6 and model11/model12 architectures. The complete residue-residue contact/distance matrix was converted to a sparse top-k contact-strength graph and encoded by a row-wise CNN followed by DenseGraphSAGE message passing. The resulting graph embedding was fused with dynamic geometry and residue descriptors for residue-level SS8 prediction. This branch preserves the model12-centered DSSP objective while restoring the full contact-map representation used in the earlier IDPss framework.
```
