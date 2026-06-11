from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ablation import AblationConfig, load_ablation_config, write_ablation_config
from .dssp_classifier import Model12ProbabilityTableClassifier
from .dssp_sampler import sample_dssp_sequences
from .early_smd_filter import filter_pre_pore_entry
from .io_utils import ensure_dir, load_json, resolve_path
from .pdb_exporter import export_frame_pdb
from .ranker import rank_frames
from .stress_adjusted_dssp import add_stress_adjusted_column
from .trajectory_library import TrajectoryLibrary
from .visualization import plot_cartoon3d, write_pymol_script


def run_ensemble_pipeline(config_path: str | Path, sequence: str, n_samples: int = 200, top_k: int = 5,
                          seed: int = 18, ablation: AblationConfig | dict | None = None) -> dict:
    cfg_path = Path(config_path).resolve()
    cfg = load_json(cfg_path)
    base = cfg_path.parent.parent
    temporal_table = resolve_path(cfg["model17_temporal_table"], base)
    geometry_table = resolve_path(cfg["model17_geometry_table"], base)
    outdir = ensure_dir(resolve_path(cfg.get("output_dir", "outputs/demo"), base))
    ablation_cfg = ablation if isinstance(ablation, AblationConfig) else load_ablation_config({**cfg.get("ablation", {}), **(ablation or {})})
    write_ablation_config(ablation_cfg, outdir / "ablation_config_used.json")

    classifier = Model12ProbabilityTableClassifier(temporal_table)
    profile = classifier.predict_probabilities(sequence)
    sampled = sample_dssp_sequences(profile, n=n_samples, seed=seed)

    profile.probabilities.to_csv(outdir / "dssp_probability_profile.csv", index=False)
    sampled.frequency_table.to_csv(outdir / "sampled_dssp_frequency.csv", index=False)
    (outdir / "consensus_dssp.txt").write_text(sampled.consensus + "\n", encoding="utf-8")

    lib = TrajectoryLibrary(temporal_table, geometry_table)
    frames = lib.load_frame_profiles(len(sequence))
    if ablation_cfg.use_stress_adjusted_dssp:
        frames = add_stress_adjusted_column(frames, dssp_col="frame_dssp", extension_col="end_to_end_A")
    if ablation_cfg.use_early_smd_filter:
        frames = filter_pre_pore_entry(frames, force_col="force", max_time_ps=cfg.get("early_smd_max_time_ps"))
        if frames.empty:
            raise ValueError("Ablation early-SMD filter removed all frames; relax filter settings.")
    ranked = rank_frames(frames, sampled.consensus, top_k=top_k, ablation=ablation_cfg)
    ranked.to_csv(outdir / "top_ranked_conformer_scores.csv", index=False)

    pdb_paths = []
    dssp_sequences = []
    if ablation_cfg.export_pdb:
        for _, row in ranked.iterrows():
            src = lib.source_for_frame(row["run_id"], int(row["frame_index"]))
            pdb_name = f"Q{len(sequence)}_rank{int(row['rank']):02d}_{row['run_id']}_frame{int(row['frame_index'])}.pdb"
            pdb_path = outdir / "pdb" / pdb_name
            export_frame_pdb(src["source_gro"], src["source_xtc"], int(row["frame_index"]), pdb_path)
            pdb_paths.append(pdb_path)
            dssp_sequences.append(row.get("ranking_dssp_sequence_used", row["frame_dssp"]))

    plot_path = None
    pml_path = None
    if ablation_cfg.export_pdb and ablation_cfg.make_visualization and pdb_paths:
        plot_path = plot_cartoon3d(pdb_paths, dssp_sequences, outdir / "model18_top_conformer_cartoon3d.png", "model18 top-ranked PolyQ conformer ensemble")
        pml_path = write_pymol_script(pdb_paths, outdir / "open_in_pymol.pml")

    summary = {
        "sequence_length": len(sequence),
        "n_dssp_samples": n_samples,
        "top_k": top_k,
        "consensus_dssp": sampled.consensus,
        "ablation": ablation_cfg.to_dict(),
        "probability_profile": str(outdir / "dssp_probability_profile.csv"),
        "frequency_table": str(outdir / "sampled_dssp_frequency.csv"),
        "ranked_scores": str(outdir / "top_ranked_conformer_scores.csv"),
        "pdb_files": [str(p) for p in pdb_paths],
        "cartoon3d_png": str(plot_path) if plot_path else None,
        "pymol_script": str(pml_path) if pml_path else None,
    }
    pd.Series(summary, dtype="object").to_json(outdir / "model18_run_summary.json", force_ascii=False, indent=2)
    return summary
