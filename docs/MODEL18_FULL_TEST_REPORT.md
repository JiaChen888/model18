# MODEL18 Full Test Report

Test date: 2026-06-11

## Summary

### demo_q36

- sequence length: 36
- consensus DSSP: `SSSSSSTTTSSSSSSSSSSSSSSSSSSSSSSSSSSS`
- top-k PDB files: 5
- score range: 0.9249 - 0.9349
- PNG: `/home/zj/jyxl/test_project/516/graph_idpss/model18_modular_polyq_ensemble/outputs/demo_q36/model18_top_conformer_cartoon3d.png` (226335 bytes)
- PyMOL script: `/home/zj/jyxl/test_project/516/graph_idpss/model18_modular_polyq_ensemble/outputs/demo_q36/open_in_pymol.pml`

Top-ranked frames:

|   rank | run_id   |   frame_index |   total_score |   dssp_position_score |   geometry_score |   contact_score |
|-------:|:---------|--------------:|--------------:|----------------------:|-----------------:|----------------:|
|      1 | Q36_v2   |         22000 |      0.934942 |              0.972222 |         0.838282 |          0.8673 |
|      2 | Q36_v2   |         39000 |      0.932288 |              0.972222 |         0.825015 |          0.8673 |
|      3 | Q36_v2   |         15000 |      0.930219 |              0.972222 |         0.814669 |          0.8673 |
|      4 | Q36_v2   |         40000 |      0.927171 |              0.972222 |         0.799427 |          0.8673 |
|      5 | Q36_v2   |         39500 |      0.924861 |              0.972222 |         0.787878 |          0.8673 |

### demo_q46

- sequence length: 46
- consensus DSSP: `SSIISSTTTSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS`
- top-k PDB files: 5
- score range: 0.9407 - 0.9489
- PNG: `/home/zj/jyxl/test_project/516/graph_idpss/model18_modular_polyq_ensemble/outputs/demo_q46/model18_top_conformer_cartoon3d.png` (232802 bytes)
- PyMOL script: `/home/zj/jyxl/test_project/516/graph_idpss/model18_modular_polyq_ensemble/outputs/demo_q46/open_in_pymol.pml`

Top-ranked frames:

|   rank | run_id     |   frame_index |   total_score |   dssp_position_score |   geometry_score |   contact_score |
|-------:|:-----------|--------------:|--------------:|----------------------:|-----------------:|----------------:|
|      1 | Q46_main   |         24500 |      0.948928 |                     1 |         0.859783 |        0.769711 |
|      2 | Q46_v3     |         36500 |      0.945272 |                     1 |         0.841506 |        0.769711 |
|      3 | Q46_main   |          8500 |      0.945016 |                     1 |         0.840224 |        0.769711 |
|      4 | Q46_1000ns |         88500 |      0.942635 |                     1 |         0.82832  |        0.769711 |
|      5 | Q46_v3     |         41500 |      0.940658 |                     1 |         0.818434 |        0.769711 |

## PDB Integrity

| demo     | pdb                                  |   atoms |   ca |   expected_atoms |   expected_ca | ok   |
|:---------|:-------------------------------------|--------:|-----:|-----------------:|--------------:|:-----|
| demo_q36 | Q36_rank01_Q36_v2_frame22000.pdb     |     615 |   36 |              615 |            36 | True |
| demo_q36 | Q36_rank02_Q36_v2_frame39000.pdb     |     615 |   36 |              615 |            36 | True |
| demo_q36 | Q36_rank03_Q36_v2_frame15000.pdb     |     615 |   36 |              615 |            36 | True |
| demo_q36 | Q36_rank04_Q36_v2_frame40000.pdb     |     615 |   36 |              615 |            36 | True |
| demo_q36 | Q36_rank05_Q36_v2_frame39500.pdb     |     615 |   36 |              615 |            36 | True |
| demo_q46 | Q46_rank01_Q46_main_frame24500.pdb   |     785 |   46 |              785 |            46 | True |
| demo_q46 | Q46_rank02_Q46_v3_frame36500.pdb     |     785 |   46 |              785 |            46 | True |
| demo_q46 | Q46_rank03_Q46_main_frame8500.pdb    |     785 |   46 |              785 |            46 | True |
| demo_q46 | Q46_rank04_Q46_1000ns_frame88500.pdb |     785 |   46 |              785 |            46 | True |
| demo_q46 | Q46_rank05_Q46_v3_frame41500.pdb     |     785 |   46 |              785 |            46 | True |

## Warnings

MDAnalysis reports missing optional PDB metadata such as chainIDs, occupancies, and elements. Coordinates, ATOM records, and CA counts are valid for PyMOL visualization.
