# MODEL18 Ablation Switches

model18 now supports reserved ablation switches through config files and CLI overrides.

## Switches

```text
use_dssp_position          DSSP position-level agreement in rank score
use_dssp_composition       DSSP composition similarity in rank score
use_geometry               Rg/end-to-end/local geometry consistency
use_contact                contact-density consistency
use_stress_adjusted_dssp   use stress-adjusted S -> C sequence for ranking
use_early_smd_filter       apply early/pre-pore-entry frame filter
use_learned_ranking        reserved learned-ranker switch
export_pdb                 export top-k PDB files
make_visualization         generate 3D cartoon and PyMOL script
```

## CLI Examples

Full default:

```bash
python3 scripts/run_model18_ensemble.py --length 46 --samples 200 --top-k 5
```

No geometry/contact ranking, fast no-PDB test:

```bash
python3 scripts/run_model18_ensemble.py --config configs/ablation_no_geom_contact_q46.json --length 46 --samples 50 --top-k 5
```

CLI override:

```bash
python3 scripts/run_model18_ensemble.py --length 46 --disable geometry --disable contact --disable pdb --disable visualization
```

Enable stress-adjusted DSSP ranking:

```bash
python3 scripts/run_model18_ensemble.py --length 46 --enable stress_adjusted
```

## Output Recording

Each run writes:

```text
ablation_config_used.json
model18_run_summary.json
```

This makes each experiment traceable for later ablation tables.
