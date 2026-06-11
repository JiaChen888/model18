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
