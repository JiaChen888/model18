# MODEL18 Technical Route

## Objective

Convert the PolyQ dynamic DSSP workflow into a modular system that can be reused, replaced, and extended.

## Pipeline

```text
1. data sampler
   MD/SMD trajectory frames, Q length, optional early-window filtering

2. DSSP classifier
   model12-centered SS8 probability prediction
   current default: model12/model16-derived probability table

3. DSSP sampler
   sample N DSSP sequences from per-residue SS8 probabilities
   summarize high-frequency consensus DSSP

4. trajectory retrieval
   compare sampled/consensus DSSP with frame-level DSSP profiles
   restrict by length Q22/Q36/Q46

5. ranking
   total score = 0.45 DSSP position score
               + 0.25 DSSP composition score
               + 0.20 geometry score
               + 0.10 contact score

6. 3D reconstruction
   export top-k real MD/SMD frames as PDB using MDAnalysis

7. visualization
   Matplotlib 3D DSSP-colored cartoon
   PyMOL script for publication-quality rendering
```

## Why Retrieval Instead of Direct Coordinate Generation

DSSP + sequence length is underdetermined for IDPs. Multiple 3D conformations can share similar DSSP labels. model18 therefore uses DSSP as a probabilistic constraint and retrieves actual trajectory frames with known coordinates.

## Replaceable Interfaces

- `DSSPClassifierBase`: replace table-backed probabilities with direct model12 checkpoint inference.
- `TrajectoryLibrary`: add SMD early-window frames, Q22, or external IDP trajectories.
- `rank_frames`: adjust score weights or add FES/contact-map CNN outputs.
- `export_frame_pdb`: switch from full protein to backbone-only or CA-only export.

## Current Limitation

The default DSSP labels are model-derived, not mkdssp or experimental labels. This is consistent with the project rule of not invoking mkdssp, but the limitation must be stated in manuscripts.
