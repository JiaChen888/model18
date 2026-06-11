# Chat Record and Prompt Summary for Model18 Merge

User objective:

```text
Build a modular model18 pipeline from the existing PolyQ IDP DSSP work.
Keep model12 as the main DSSP classifier framework.
Use model16 calibration and model17 temporal/contact MD features.
Convert high-probability dynamic DSSP and Q length into traceable 3D multi-conformer PDBs.
Generate PyMOL/VMD-compatible visualization outputs.
```

Key design decision:

```text
Do not claim DSSP + length uniquely reconstructs 3D.
Instead: sample DSSP states, retrieve matching MD/SMD frames, rank by DSSP/geometry/contact, export real-frame PDB ensemble.
```

Requested functional steps:

```text
1. sample N DSSP sequences from DSSP probabilities
2. retrieve the most similar frames from existing MD/SMD libraries
3. rank by DSSP match + Rg + end-to-end + contact score
4. keep top-k conformers
5. output multi-conformer PDB files
6. generate PyMOL/VMD visualization figures/scripts
```

Model naming:

```text
model18 = modular DSSP-to-3D ensemble framework
```
