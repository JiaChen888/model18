from __future__ import annotations

from pathlib import Path


def export_frame_pdb(source_gro: str, source_xtc: str, frame_index: int, output_pdb: str | Path, selection: str = "protein") -> Path:
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise RuntimeError("MDAnalysis is required for PDB export from trajectories") from exc
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    u = mda.Universe(source_gro, source_xtc)
    if frame_index >= len(u.trajectory):
        raise IndexError(f"frame_index {frame_index} outside trajectory with {len(u.trajectory)} frames")
    u.trajectory[frame_index]
    atoms = u.select_atoms(selection)
    if len(atoms) == 0:
        atoms = u.atoms
    atoms.write(str(output_pdb))
    return output_pdb
