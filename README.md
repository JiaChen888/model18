# model18: Modular PolyQ Dynamic DSSP to 3D Ensemble

model18 upgrades the previous model12/model16/model17 workflow into a modular pipeline. The main DSSP classifier remains model12-centered, while model16/model17 provide calibrated probabilities, temporal/contact features, and MD/SMD-backed conformer retrieval.

## Core Idea

```text
PolyQ sequence length
+ model12/model16 DSSP probability profile
+ DSSP sequence sampling
+ MD/SMD trajectory library
+ geometry/contact ranking
-> traceable top-k 3D conformer PDB ensemble
```

This is designed for IDP systems such as PolyQ, where a single native 3D structure is not expected. model18 reconstructs a conformational ensemble rather than one fixed fold.

## Module Layout

```text
model18/dssp_classifier.py      model12-centered DSSP probability interface
model18/dssp_sampler.py         sample N DSSP sequences from residue probabilities
model18/trajectory_library.py   load MD/SMD frame-level DSSP, geometry, contact features
model18/ranker.py               rank frames by DSSP, geometry, contact consistency
model18/pdb_exporter.py         export true trajectory frames as PDB using MDAnalysis
model18/visualization.py        3D cartoon plot and PyMOL script generation
model18/pipeline.py             end-to-end orchestration
scripts/run_model18_ensemble.py command-line entry point
```

## Current Default Inputs

The default implementation uses model17 tables as the trajectory/probability backend:

```text
../model17_q46md_temporal_contact/tables/model17_multilength_temporal_contact_table.csv
../model17_q46md_temporal_contact/tables/independent_md_multilength_geometry_contact_features.csv
```

These tables contain Q36/Q46 independent MD features and model12/model16-derived SS8 soft probabilities. No mkdssp is used.



## SMD 300-Sample Library Status

The complete Q22/Q36/Q46 300-sample SMD library is not yet fully read into model18. A local audit found:

```text
Q22: 252 run directories, 239 ready_basic
Q36: 188 run directories, 174 ready_basic
Q46: 80 run directories, 56 ready_basic
```

Current model18 retrieval backend still uses the model17 independent MD subset:

```text
Q46-MD main/v2/v3/1000ns + Q36-MD/v2
runs = 5
frames = 629
Q22 not yet included in model18 retrieval backend
```

New modules have been added for the requested upgrades:

```text
model18/smd_audit.py
model18/early_smd_filter.py
model18/stress_adjusted_dssp.py
model18/learned_ranking.py
```

See:

```text
docs/POLYQ_SMD_TRAJECTORY_LIBRARY_AUDIT.csv
docs/POLYQ_SMD_TRAJECTORY_LIBRARY_AUDIT_SUMMARY.csv
docs/MODEL18_SMD_300_LIBRARY_AUDIT_AND_OPTIMIZATION.md
```

## Run Demo

```bash
python3 scripts/run_model18_ensemble.py --length 46 --samples 200 --top-k 5
```

For Q36:

```bash
python3 scripts/run_model18_ensemble.py --length 36 --samples 200 --top-k 5
```

## Outputs

Default output directory:

```text
outputs/demo_q46/
```

Main files:

```text
dssp_probability_profile.csv       residue-wise SS8 probability profile
sampled_dssp_frequency.csv         sampled DSSP frequency table
consensus_dssp.txt                 high-frequency consensus DSSP sequence
top_ranked_conformer_scores.csv    DSSP/geometry/contact ranking scores
pdb/*.pdb                          top-k full-protein PDB conformers
model18_top_conformer_cartoon3d.png 3D cartoon contact sheet
open_in_pymol.pml                  PyMOL script for direct visualization
model18_run_summary.json           run summary
```

## PyMOL Visualization

```bash
pymol outputs/demo_q46/open_in_pymol.pml
```

or directly:

```bash
pymol outputs/demo_q46/pdb/*.pdb
```

## Scientific Interpretation

model18 does not claim that DSSP labels alone uniquely determine 3D structure. It uses DSSP probabilities as constraints and retrieves traceable real conformers from MD/SMD trajectory frames. The resulting PDBs are therefore ensemble candidates consistent with dynamic DSSP and geometry/contact statistics.

## Best Current Model Stack

```text
model12: primary DSSP classifier framework
model16: calibrated probability layer
model17: temporal/contact MD expansion
model18: modular DSSP-to-3D ensemble reconstruction pipeline
```


## Full Test Status

A complete Q36/Q46 test was run after the modular split.

```text
python3 -m py_compile model18/*.py scripts/*.py
python3 scripts/run_model18_ensemble.py --length 46 --samples 200 --top-k 5
python3 scripts/run_model18_ensemble.py --config configs/q36_paths.json --length 36 --samples 200 --top-k 5
```

Validation results:

```text
Q46: 5 full-protein PDB files, each 785 ATOM records and 46 CA atoms
Q36: 5 full-protein PDB files, each 615 ATOM records and 36 CA atoms
PNG/PML/CSV outputs generated for both groups
```

Detailed reports:

```text
docs/MODEL18_FULL_TEST_REPORT.md
docs/MODEL18_FULL_TEST_PDB_CHECK.csv
docs/MODEL18_SCI_FIGURE_AND_WRITING_PLAN.md
```



## model17_v2 Backend Epoch Selection

model18 does not train by epoch. The epoch 100/300/500 comparison belongs to the model17_v2 temporal/contact backend.

```text
epoch100: accuracy = 0.8889, ECE = 0.1216, Brier = 0.2434  recommended
epoch300: accuracy = 0.8889, ECE = 0.2237, Brier = 0.2731
epoch500: accuracy = 0.8889, ECE = 0.3194, Brier = 0.3362
```

Conclusion: use epoch100 as the preferred model17_v2 backend because longer training does not improve top1 accuracy and worsens calibration.

See:

```text
docs/MODEL18_MODEL17_EPOCH_BACKEND_SELECTION.md
outputs/sci_figures/model17_v2_epoch100_300_500_backend_comparison.png
```

## SCI Figure Recommendation

Use these as the main manuscript candidates:

```text
1. docs/MODEL18_ARCHITECTURE.svg
2. Q36/Q46 DSSP probability and sampled-frequency profiles
3. outputs/demo_q46/model18_top_conformer_cartoon3d.png
4. outputs/demo_q36/model18_top_conformer_cartoon3d.png
5. outputs/sci_figures/demo_q36_score_decomposition.png
6. outputs/sci_figures/demo_q46_score_decomposition.png
7. PyMOL ray-traced images generated from open_in_pymol.pml
```
