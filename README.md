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
