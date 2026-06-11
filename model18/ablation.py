from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class AblationConfig:
    use_dssp_position: bool = True
    use_dssp_composition: bool = True
    use_geometry: bool = True
    use_contact: bool = True
    use_stress_adjusted_dssp: bool = False
    use_early_smd_filter: bool = False
    use_learned_ranking: bool = False
    export_pdb: bool = True
    make_visualization: bool = True

    dssp_position_weight: float = 0.45
    dssp_composition_weight: float = 0.25
    geometry_weight: float = 0.20
    contact_weight: float = 0.10

    def to_dict(self) -> dict:
        return asdict(self)


def load_ablation_config(raw: dict | None = None) -> AblationConfig:
    raw = raw or {}
    cfg = AblationConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def apply_cli_overrides(cfg: AblationConfig, disabled: list[str] | None = None, enabled: list[str] | None = None) -> AblationConfig:
    disabled = disabled or []
    enabled = enabled or []
    aliases = {
        "dssp_position": "use_dssp_position",
        "dssp_composition": "use_dssp_composition",
        "dssp": "use_dssp_position",
        "geometry": "use_geometry",
        "contact": "use_contact",
        "stress_adjusted": "use_stress_adjusted_dssp",
        "early_smd": "use_early_smd_filter",
        "learned_ranking": "use_learned_ranking",
        "pdb": "export_pdb",
        "visualization": "make_visualization",
    }
    for name in disabled:
        attr = aliases.get(name, name)
        if hasattr(cfg, attr):
            setattr(cfg, attr, False)
    for name in enabled:
        attr = aliases.get(name, name)
        if hasattr(cfg, attr):
            setattr(cfg, attr, True)
    return cfg


def write_ablation_config(cfg: AblationConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return path
