# MODEL18 SMD 300-Library Audit and Optimization Status

## Direct Answer

The complete Q22/Q36/Q46 300-sample SMD trajectory library is **not yet fully read into model18**.

Current model18/model17 active trajectory backend:

```text
Q46-MD: main, v2, v3, 1000ns
Q36-MD: v2
runs = 5
frames = 629
Q36 residue-frame rows = 3,132
Q46 residue-frame rows = 24,932
Q22 = not included in current model18 retrieval backend
```

This means model18 currently validates the modular reconstruction pipeline on Q36/Q46 independent MD trajectories, not on the complete 300-sample SMD library.

## SMD Library Audit from Local Disk

Audited source root:

```text
/media/zj/5e3e7fef-d2f0-47aa-9148-5fca8b7098d41/chenj/IDP
```

Detected local run directories:

| group | run directories | ready_basic | protein.xtc | whole.xtc | noH.xtc | pullf |
|---|---:|---:|---:|---:|---:|---:|
| Q22 | 252 | 239 | 201 | 167 | 126 | 240 |
| Q36 | 188 | 174 | 142 | 172 | 142 | 177 |
| Q46 | 80 | 56 | 59 | 54 | 50 | 76 |

`ready_basic` means at least one XTC, at least one GRO, and at least one pullf XVG were found in the run directory.

Therefore, the visible local SMD files do **not** currently form a confirmed 300-run-per-length complete library. Additional paths may exist, but they have not been indexed into model18 yet.

## model6/model8 Data Check

model6 epoch200 outputs contain DSSP sample-level test data for all lengths:

```text
Q22 samples = 1463
Q36 samples = 1460
Q46 samples = 2127
```

model8from6a pilot metadata currently shows only Q22 window samples:

```text
train = 486 rows
test = 121 rows
length = Q22 only
```

These are sample/window metadata, not proof that the full SMD trajectory library was read as trajectories.

## New model18 Modules Added

```text
model18/smd_audit.py              index local SMD trajectory library
model18/early_smd_filter.py       pre-pore-entry / low-force filtering interface
model18/stress_adjusted_dssp.py   S -> C stress-adjusted DSSP interface
model18/learned_ranking.py        force/FES/contact-map learned ranking scaffold
```

## Optimization Items

### 1. Complete Q22/Q36/Q46 SMD trajectory ingestion

Required next step:

```text
Convert audited SMD runs into a normalized manifest with:
length, run_id, run_dir, selected_xtc, selected_gro, selected_tpr, pullf_xvg, ready flag
```

Then extract frame-level features using a consistent stride and join pullf force labels.

### 2. Early-SMD pre-pore-entry filtering

Purpose:

```text
Reduce force-induced structural bias before using frames for initial-conformer retrieval.
```

Current implemented interface:

```text
model18/early_smd_filter.py
```

Planned criteria:

```text
low-force quantile
pre-entry z-coordinate threshold
pre-peak-force window
optional time cutoff
```

### 3. Direct model12 checkpoint inference

Current model18 uses table-backed probabilities from model12/model16/model17 outputs.

Target:

```text
sequence -> model12 checkpoint -> residue-level SS8 probability -> model18 sampler
```

This needs a stable checkpoint loader for the model12 architecture and preprocessing code.

### 4. Learned ranking

Current ranking is interpretable weighted scoring:

```text
0.45 DSSP position + 0.25 DSSP composition + 0.20 geometry + 0.10 contact
```

New scaffold:

```text
model18/learned_ranking.py
```

Future targets:

```text
force labels
FES basin membership
entry/transition event labels
full contact-map CNN/GNN embeddings
```

### 5. Stress-adjusted DSSP

Implemented scaffold:

```text
model18/stress_adjusted_dssp.py
```

Purpose:

```text
Convert high-stress S states to C when S is likely a force-induced bend rather than intrinsic bend.
```

## Current Conclusion

model18 is structurally ready for the five requested upgrades, but the complete Q22/Q36/Q46 300-sample SMD library is not yet fully integrated. The immediate priority is to normalize the SMD manifest and extract early-window force/geometry/contact features from all ready SMD runs.
