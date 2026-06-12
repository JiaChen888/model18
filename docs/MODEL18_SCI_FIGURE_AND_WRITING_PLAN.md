# MODEL18 SCI Figure and Writing Plan

## Recommended Main Figures

### Figure 1. Overall technical workflow

Use:

```text
docs/MODEL18_ARCHITECTURE.svg
```

Suggested caption:

```text
Figure 1. Modular dynamic DSSP-to-3D ensemble reconstruction workflow for PolyQ intrinsically disordered proteins. The model12-centered DSSP classifier provides residue-level SS8 probability profiles, model16 contributes calibrated probabilities, and model17 supplies independent MD-derived temporal/contact features. model18 samples high-probability DSSP states, retrieves traceable MD/SMD trajectory frames, ranks them by DSSP, geometry, and contact consistency, and exports top-ranked conformers as PyMOL-compatible PDB ensembles.
```

Suggested text:

```text
Because PolyQ is intrinsically disordered, the objective was not to infer a single native structure. Instead, we developed a modular reconstruction strategy in which dynamic DSSP probabilities act as probabilistic constraints for ensemble retrieval from MD/SMD conformational libraries.
```

### Figure 2. DSSP probability and sampled frequency profile

Use:

```text
outputs/demo_q46/dssp_probability_profile.csv
outputs/demo_q46/sampled_dssp_frequency.csv
outputs/demo_q36/dssp_probability_profile.csv
outputs/demo_q36/sampled_dssp_frequency.csv
```

Recommended plot:

```text
Residue index on x-axis; stacked or line plot of SS8 probabilities/frequencies.
Show Q36 and Q46 as separate panels.
```

Suggested caption:

```text
Figure 2. Residue-level dynamic DSSP probability profiles and sampled DSSP frequencies for PolyQ ensembles. Per-residue SS8 probabilities were derived from the model12/model16 probability layer and used to sample DSSP state sequences. The high-frequency consensus profiles were subsequently used for conformer retrieval.
```

### Figure 3. Top-ranked conformer ensemble 3D cartoon

Use:

```text
outputs/demo_q46/model18_top_conformer_cartoon3d.png
outputs/demo_q36/model18_top_conformer_cartoon3d.png
```

Suggested caption:

```text
Figure 3. Top-ranked PolyQ conformer ensembles reconstructed by model18. Each panel shows a trajectory-derived full-protein conformer whose C-alpha backbone is colored by the corresponding predicted DSSP state. These structures are not unique native folds, but traceable conformational candidates selected from MD trajectories by dynamic DSSP, geometry, and contact consistency.
```

### Figure 4. Conformer ranking score decomposition

Use:

```text
outputs/demo_q46/top_ranked_conformer_scores.csv
outputs/demo_q36/top_ranked_conformer_scores.csv
```

Recommended plot:

```text
Bar plot of total_score, dssp_position_score, dssp_composition_score, geometry_score, contact_score for top-k conformers.
```

Suggested caption:

```text
Figure 4. Ranking score decomposition for top PolyQ conformers. Candidate frames were scored using DSSP position agreement, DSSP composition similarity, global geometry consistency, and contact-density consistency. The selected conformers show high DSSP agreement while maintaining realistic MD-derived geometry.
```

### Figure 5. PyMOL/VMD rendering of selected ensemble

Use:

```text
outputs/demo_q46/open_in_pymol.pml
outputs/demo_q36/open_in_pymol.pml
```

Suggested caption:

```text
Figure 5. PyMOL-rendered PolyQ conformer ensemble generated from model18 PDB outputs. The exported full-protein PDB files are directly compatible with PyMOL/VMD and preserve all protein atoms from the selected MD frames.
```

## Supplementary Figures

1. Full test report table: `docs/MODEL18_FULL_TEST_REPORT.md`.
2. PDB integrity table: `docs/MODEL18_FULL_TEST_PDB_CHECK.csv`.
3. Q36/Q46 comparison of Rg, end-to-end distance, and contact density from `top_ranked_conformer_scores.csv`.
4. Architecture SVG as editable technical schematic.

## Methods Text Draft

```text
A modular dynamic DSSP-to-3D reconstruction framework was developed for PolyQ intrinsically disordered proteins. First, residue-level SS8 probability profiles were obtained using the model12-centered DSSP prediction framework with calibrated probabilities from model16. For each PolyQ length, N DSSP sequences were sampled from the residue-wise SS8 distributions, and a high-frequency consensus DSSP profile was computed. Candidate conformers were then retrieved from independent MD/SMD trajectory libraries generated for Q36 and Q46. Each trajectory frame was represented by its predicted frame-level DSSP profile, global geometry descriptors including radius of gyration and end-to-end distance, and contact-density features. Candidate frames were ranked by a weighted score combining DSSP position agreement, DSSP composition similarity, global geometry consistency, and contact consistency. The top-ranked frames were exported as full-protein PDB files and visualized as DSSP-colored C-alpha cartoon backbones.
```

## Results Text Draft

```text
The model18 workflow successfully generated traceable conformational ensembles for Q36 and Q46. In the full validation run, five top-ranked conformers were exported for each sequence length. Q36 conformers contained 615 protein atoms and 36 C-alpha atoms, while Q46 conformers contained 785 protein atoms and 46 C-alpha atoms, confirming that the exported structures were full-protein PDBs rather than C-alpha-only reconstructions. The resulting structures can be opened directly in PyMOL or VMD, and the generated cartoon panels provide a compact visual representation of the dynamic DSSP-informed conformer ensemble.
```

## Discussion Text Draft

```text
DSSP labels and sequence length alone are insufficient to uniquely determine the three-dimensional structure of an intrinsically disordered protein. Therefore, model18 does not attempt deterministic coordinate generation from DSSP states. Instead, dynamic DSSP probabilities are used as probabilistic constraints to retrieve physically traceable conformers from MD/SMD libraries. This design reduces overinterpretation and aligns with the ensemble nature of PolyQ IDPs.
```

## Limitations to State Explicitly

```text
1. DSSP supervision remains model-derived and does not use mkdssp.
2. Q22 currently needs an independent MD/SMD trajectory library before the same 3D retrieval can be performed.
3. The current ranking score is interpretable and modular but not yet learned end-to-end.
4. PDB export warnings from MDAnalysis reflect missing optional metadata fields, not coordinate failure.
5. The PyMOL/VMD structures are traceable conformer candidates, not experimentally determined native folds.
```

## Best Optimization Targets

```text
1. Add Q22 independent MD/SMD frames to complete Q22/Q36/Q46 ensemble comparison.
2. Add early-SMD pre-pore-entry filtering to reduce force-induced conformer bias.
3. Replace table-backed DSSP probabilities with direct model12 checkpoint inference.
4. Add learned ranking using force labels, FES coordinates, and contact-map CNN/GNN embeddings.
5. Add stress-adjusted DSSP profiles to separate true bend from force-induced S states.
6. Generate publication-grade PyMOL ray-traced images from the provided PML scripts.
```

## Final Paper-Ready Clean-300 Figures

### Recommended Fig. 8: model18 dynamic DSSP training performance

Use:

```text
outputs/sci_figures/model18_clean300_epoch_comparison.png
outputs/sci_figures/model18_clean300_sample_distribution.png
```

Caption:

```text
Figure 8. Clean-300 model18 training behavior for dynamic DSSP prediction. (A) Composition of the fixed clean-300 PolyQ SMD database used for all model18 training experiments, including Q22, Q36, and Q46 samples with a sample-level train/test split. (B) Top-1 accuracy and macro F1 for 100, 300, and 500 training epochs using the same manifest and sample stride. The 100-epoch model achieved the best fixed-split performance, whereas longer training reduced generalization, indicating mild overfitting.
```

Main-text result paragraph:

```text
Using the fixed clean-300 preprocessing database, model18 was trained as a frame-residue SS8 classifier with dynamic geometry, contact descriptors, contact-map row statistics, residue descriptors, and residue-position features. Under the unified stride500 setting, the 100-epoch full model achieved a top-1 accuracy of 0.8097 and a macro F1 of 0.5567 on 28,496 held-out residue-frame samples. Extending training to 300 and 500 epochs did not improve performance, consistent with mild overfitting under the current split. A denser stride100 reference run increased top-1 accuracy to 0.8162 but did not improve macro F1, indicating that class-balanced performance remains the main target for further optimization.
```

### Recommended Fig. 9: model18 feature contribution and 3D ensemble output

Use:

```text
outputs/sci_figures/model18_clean300_ablation_comparison.png
outputs/demo_q46/model18_top_conformer_cartoon3d.png
outputs/demo_q36/model18_top_conformer_cartoon3d.png
outputs/sci_figures/demo_q46_score_decomposition.png
outputs/sci_figures/demo_q36_score_decomposition.png
```

Caption:

```text
Figure 9. Feature contribution and DSSP-guided 3D ensemble reconstruction by model18. (A) Feature ablation on the clean-300 database shows that dynamic geometry and contact-derived features contribute most strongly to residue-level DSSP prediction. Removing all contact information caused the largest performance drop, while removing explicit residue position slightly improved performance, suggesting that position/length bias is not required for the current PolyQ split. (B) Top-ranked Q36 and Q46 conformers retrieved from MD/SMD trajectory libraries and exported as full-protein PDB files. C-alpha backbones are colored by predicted DSSP states. (C) Ranking-score decomposition of the selected conformers, showing the contribution of DSSP agreement, geometry consistency, and contact consistency.
```

Main-text result paragraph:

```text
Ablation experiments clarified the contribution of each model18 feature group. Removing dynamic descriptors reduced macro F1 from 0.5567 to 0.5314, and removing all contact-derived information reduced macro F1 to 0.5232, demonstrating that both dynamic geometry and contact topology are informative for PolyQ DSSP-state assignment. In contrast, removing explicit residue-position features slightly improved performance to 0.8110 accuracy and 0.5613 macro F1, suggesting that position-derived bias is not necessary when dynamic and contact features are available. The same probabilistic DSSP outputs were then used to retrieve top-ranked traceable conformers from MD/SMD trajectory libraries, yielding PyMOL/VMD-compatible PDB ensembles rather than a single deterministic IDP structure.
```

### Figure Style Notes for SCI Submission

```text
1. Use 300 dpi PNG for review; export SVG/PDF if the journal requires vector plots.
2. Keep model architecture as editable SVG: docs/MODEL18_ARCHITECTURE.svg.
3. For final PyMOL/VMD panels, the current matplotlib cartoon figures are acceptable for method validation, but ray-traced PyMOL images can be added for the final visual polish.
4. Always state that the 3D structures are trajectory-retrieved conformer candidates, not experimental native folds.
5. DSSP labels are model-derived and no mkdssp step is used.
```

## Final Conclusion Wording

```text
The best current model18 strategy is not direct coordinate generation from DSSP labels. Instead, dynamic DSSP probabilities are combined with frame-level geometry and contact descriptors to retrieve physically traceable MD/SMD conformers. This approach is more appropriate for PolyQ IDPs because it produces a ranked conformational ensemble, preserves linkability to the original trajectory frames, and avoids overclaiming a unique native structure. The clean-300 experiments show that contact and dynamic features provide the strongest gains, while longer training alone does not improve generalization. The next major improvement should therefore come from a true full-contact-map CNN/GNN or residue-level temporal model rather than additional MLP epochs.
```
