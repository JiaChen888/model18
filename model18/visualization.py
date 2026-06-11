from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import DSSP_COLORS


def read_ca_from_pdb(pdb_path: str | Path) -> np.ndarray:
    coords = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(coords, dtype=float)


def plot_cartoon3d(pdb_paths: list[str | Path], dssp_sequences: list[str], out_png: str | Path, title: str) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = len(pdb_paths)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4.2 * cols, 3.8 * rows))
    for idx, (pdb, dssp) in enumerate(zip(pdb_paths, dssp_sequences), start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        ca = read_ca_from_pdb(pdb)
        if len(ca) < 2:
            ax.set_title(Path(pdb).stem)
            continue
        for i in range(len(ca) - 1):
            label = dssp[i] if i < len(dssp) else "C"
            ax.plot(ca[i:i + 2, 0], ca[i:i + 2, 1], ca[i:i + 2, 2], color=DSSP_COLORS.get(label, "gray"), linewidth=2.4)
        colors = [DSSP_COLORS.get((dssp[i] if i < len(dssp) else "C"), "gray") for i in range(len(ca))]
        ax.scatter(ca[:, 0], ca[:, 1], ca[:, 2], c=colors, s=8)
        ax.set_title(Path(pdb).stem, fontsize=9)
        ax.set_axis_off()
        xyz_min = ca.min(axis=0)
        xyz_max = ca.max(axis=0)
        center = (xyz_min + xyz_max) / 2
        radius = max((xyz_max - xyz_min).max() / 2, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def write_pymol_script(pdb_paths: list[str | Path], out_pml: str | Path) -> Path:
    out_pml = Path(out_pml)
    out_pml.parent.mkdir(parents=True, exist_ok=True)
    lines = ["reinitialize", "bg_color white"]
    for pdb in pdb_paths:
        name = Path(pdb).stem.replace("-", "_")
        lines.append(f"load {Path(pdb).resolve()}, {name}")
        lines.append(f"show cartoon, {name}")
        lines.append(f"hide lines, {name}")
        lines.append(f"set cartoon_smooth_loops, on, {name}")
        lines.append(f"spectrum count, rainbow, {name}")
    lines.extend([
        "set ray_opaque_background, off",
        "zoom",
        "ray 1800,1200",
        "png model18_ensemble_pymol.png, dpi=300",
    ])
    out_pml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_pml
