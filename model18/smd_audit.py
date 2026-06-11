from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def infer_group(path: str) -> str:
    if "Q22" in path:
        return "Q22"
    if "Q36" in path:
        return "Q36"
    if "Q46" in path:
        return "Q46"
    return "unknown"


def audit_smd_library(base_dir: str | Path, output_csv: str | Path | None = None) -> pd.DataFrame:
    base = Path(base_dir)
    candidate_roots = []
    for p in base.rglob("*"):
        if p.is_dir() and re.search(r"(^|/)(Q22|Q36|Q46)[^/]*$", str(p), re.I):
            candidate_roots.append(p)
    rows = []
    seen = set()
    for root in candidate_roots:
        dirs = [root]
        dirs.extend([d for d in root.iterdir() if d.is_dir() and re.match(r"v\d+", d.name)])
        for d in dirs:
            if d in seen:
                continue
            seen.add(d)
            files = list(d.iterdir()) if d.exists() else []
            xtc = [f for f in files if f.suffix.lower() == ".xtc"]
            gro = [f for f in files if f.suffix.lower() == ".gro"]
            tpr = [f for f in files if f.suffix.lower() == ".tpr"]
            pullf = [f for f in files if "pullf" in f.name.lower() and f.suffix.lower() == ".xvg"]
            if not (xtc or gro or tpr or pullf):
                continue
            rows.append({
                "group": infer_group(str(d)),
                "root": str(root),
                "run_dir": str(d),
                "run_name": d.name,
                "n_xtc": len(xtc),
                "n_gro": len(gro),
                "n_tpr": len(tpr),
                "n_pullf": len(pullf),
                "has_protein_xtc": any(f.name == "protein.xtc" for f in xtc),
                "has_whole_xtc": any(f.name == "whole.xtc" for f in xtc),
                "has_noH_xtc": any("noh" in f.name.lower() for f in xtc),
                "has_pullf": bool(pullf),
                "ready_basic": bool(xtc and gro and pullf),
                "xtc_files": ";".join(f.name for f in xtc[:8]),
                "gro_files": ";".join(f.name for f in gro[:8]),
                "pullf_files": ";".join(f.name for f in pullf[:6]),
            })
    df = pd.DataFrame(rows).drop_duplicates("run_dir") if rows else pd.DataFrame()
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def summarize_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby("group").agg(
        run_dirs=("run_dir", "count"),
        ready_basic=("ready_basic", "sum"),
        protein_xtc=("has_protein_xtc", "sum"),
        whole_xtc=("has_whole_xtc", "sum"),
        noH_xtc=("has_noH_xtc", "sum"),
        pullf=("has_pullf", "sum"),
    ).reset_index()
