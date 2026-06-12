# MODEL18 Clean-300 Final Results

## Data and Split

All reported training runs read the fixed clean-300 preprocessed database through `outputs/preprocess_db_current_clean_300/model18_training_manifest.csv`. The database contains 300 ready SMD samples: Q22 = 111, Q36 = 103, Q46 = 86. The split follows the previous model8from6a clean-300 split and is sample-level, not frame-random.

## Best Current Run

- Run: `training_outputs_clean300_epoch100`
- Epochs: 100
- Sample stride: 100
- Disabled features: `none`
- Top-1 accuracy: 0.8162
- Macro F1: 0.5387
- Macro precision: 0.5584
- Macro recall: 0.5315
- Test residues: 135136

## Epoch Comparison

| run                                          |   epochs |   accuracy |   f1_macro |   precision_macro |   recall_macro |     n |
|:---------------------------------------------|---------:|-----------:|-----------:|------------------:|---------------:|------:|
| training_outputs_clean300_epoch100_stride500 |      100 |   0.809728 |   0.55667  |          0.578086 |       0.55135  | 28496 |
| training_outputs_clean300_epoch300_stride500 |      300 |   0.800147 |   0.537806 |          0.547012 |       0.538344 | 28496 |
| training_outputs_clean300_epoch500_stride500 |      500 |   0.800849 |   0.540044 |          0.54678  |       0.541653 | 28496 |

## Ablation Comparison

| run                                                              | disabled                           |   accuracy |   f1_macro |   precision_macro |   recall_macro |   feature_dim |
|:-----------------------------------------------------------------|:-----------------------------------|-----------:|-----------:|------------------:|---------------:|--------------:|
| training_outputs_clean300_ablation_no_contact_features_epoch100  | contact_features                   |   0.807657 |   0.532588 |          0.556261 |       0.518941 |            72 |
| training_outputs_clean300_ablation_no_contact_all_epoch100       | contact_features+contact_map_stats |   0.796638 |   0.523221 |          0.548539 |       0.507387 |            68 |
| training_outputs_clean300_ablation_no_contact_map_stats_epoch100 | contact_map_stats                  |   0.799796 |   0.549121 |          0.569216 |       0.543153 |            74 |
| training_outputs_clean300_ablation_no_dynamic_epoch100           | dynamic                            |   0.802464 |   0.531403 |          0.570741 |       0.529661 |            67 |
| training_outputs_clean300_epoch100_stride500                     | none                               |   0.809728 |   0.55667  |          0.578086 |       0.55135  |            78 |
| training_outputs_clean300_ablation_no_position_epoch100          | position                           |   0.811026 |   0.56133  |          0.575215 |       0.559531 |            76 |
| training_outputs_clean300_ablation_no_residue_features_epoch100  | residue_features                   |   0.808464 |   0.541225 |          0.565462 |       0.530864 |            23 |

## SCI Figure Outputs

- `outputs/sci_figures/model18_clean300_epoch_comparison.png`
- `outputs/sci_figures/model18_clean300_ablation_comparison.png`
- `outputs/sci_figures/model18_clean300_sample_distribution.png`

## Figure Captions

**Figure 8. Clean-300 epoch comparison of model18 dynamic DSSP prediction.** Top-1 accuracy and macro F1 are shown for 100, 300, and 500 epochs using the same clean-300 manifest and the same sample stride. This panel evaluates whether longer optimization improves residue-level SS8 classification under a fixed data split.

**Figure 9. Feature ablation of model18 on the clean-300 database.** The full model is compared with variants removing dynamic geometry, contact features, contact-map row statistics, all contact information, residue static descriptors, or residue position. Accuracy and macro F1 quantify the contribution of each module to dynamic DSSP prediction.

**Figure 10. Clean-300 dataset composition.** The fixed preprocessing database contains Q22, Q36, and Q46 SMD samples and preserves the model8from6a sample-level split. This figure documents the training source used for all model18 epoch and ablation experiments.

## Main Text Draft

We next evaluated model18 on the fixed clean-300 PolyQ SMD database using a modular frame-residue representation. Each residue-frame sample was represented by dynamic trajectory descriptors, contact features, contact-map row statistics, residue-level descriptors, and normalized residue position. Training and testing were performed from the saved preprocessing database rather than recomputing raw trajectories, which ensured that all epoch and ablation comparisons used identical input samples.

The epoch comparison showed the optimization behavior of the clean-300 model18 baseline under a fixed split. Feature ablation further quantified the relative contribution of dynamic geometry and contact-derived information. Because the labels are model-derived dynamic DSSP states and PolyQ is intrinsically disordered, the model output should be interpreted as a probabilistic ensemble descriptor rather than a unique native fold assignment.

## Reliability Statement

These results are suitable for an SCI methods/results section as computational model outputs if the manuscript clearly states that DSSP labels are model-derived, no mkdssp or experimental DSSP labels were used, and the 3D structures are trajectory-retrieved ensemble conformers rather than experimentally determined native structures.
