from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from model18.full_contact_map_model import FullContactMapDSSPModel


class TemporalWindowEncoder(nn.Module):
    """model12-style temporal/force-window encoder over frame feature windows."""

    def __init__(self, in_dim: int, out_dim: int = 64, hidden: int = 128, dropout: float = 0.20, encoder: str = "transformer"):
        super().__init__()
        self.encoder = encoder
        self.in_proj = nn.Linear(in_dim, hidden)
        if encoder == "gru":
            self.core = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout, bidirectional=False)
        else:
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 3, dropout=dropout, batch_first=True, activation="gelu")
            self.core = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Sequential(nn.Linear(hidden, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout))

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(window.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        x = torch.nan_to_num((x - mean) / std, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        h = self.in_proj(x)
        if self.encoder == "gru":
            _, last = self.core(h)
            h = last[-1]
        else:
            h = self.core(h).mean(dim=1)
        return self.out(self.norm(h))


class Model19DSSPModel(FullContactMapDSSPModel):
    """model19 = model12-style temporal window + model18 full contact map.

    It keeps the full LxL contact-map CNN/GNN path and adds a temporal window
    encoder inspired by model12's force-window branch. clean300 currently lacks
    raw force arrays, so the branch consumes `force_window_proxy` built from
    dynamic_features + contact_features windows.
    """

    def __init__(self, *args, n_window_features: int, window_encoder: str = "transformer", window_aux_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        graph_dim = self.scalar_proj[0].out_features
        self.temporal_window = TemporalWindowEncoder(n_window_features, out_dim=graph_dim, hidden=kwargs.get("hidden", 128), dropout=kwargs.get("dropout", 0.20), encoder=window_encoder)
        old_in = self.fusion[0].in_features
        hidden = self.fusion[0].out_features
        dropout = kwargs.get("dropout", 0.20)
        self.fusion = nn.Sequential(
            nn.Linear(old_in + graph_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
        )
        self.window_reconstruction_head = nn.Linear(hidden, n_window_features)
        self.window_aux_weight = float(window_aux_weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cmap = batch["contact_map"].float()
        mask = batch["mask"].float()
        parts = []
        if self.use_residue_features:
            parts.append(batch["residue_features"].float())
        if self.use_residue_position:
            parts.append(batch["position"].float())
        node_raw = torch.cat(parts, dim=-1) if parts else mask.unsqueeze(-1)
        node = self.node_proj(node_raw) * mask.unsqueeze(-1)
        fused = [node]
        if self.use_full_contact_map:
            row_feat = self.row_cnn(cmap, mask)
            adj = __import__("model18.full_contact_map_model", fromlist=["contact_to_topk_strength"]).contact_to_topk_strength(cmap, top_n=self.top_n)
            graph_feat = self.graph(adj, node + row_feat, mask)
            fused.extend([row_feat, graph_feat])
        scalar_parts = []
        if self.use_dynamic:
            scalar_parts.append(batch["dynamic"].float())
        if self.use_contact_features:
            scalar_parts.append(batch["contact_features"].float())
        scalar = torch.cat(scalar_parts, dim=-1) if scalar_parts else torch.ones(mask.shape[0], 1, device=mask.device, dtype=mask.dtype)
        scalar_feat = self.scalar_proj(scalar).unsqueeze(1).expand(-1, mask.shape[1], -1) * mask.unsqueeze(-1)
        window_feat = self.temporal_window(batch["force_window_proxy"]).unsqueeze(1).expand(-1, mask.shape[1], -1) * mask.unsqueeze(-1)
        fused.extend([scalar_feat, window_feat])
        h = self.fusion(torch.cat(fused, dim=-1)) * mask.unsqueeze(-1)
        self.last_aux = {
            "contact_number": self.contact_number_head(h).squeeze(-1) * mask,
            "contact_density": self.contact_density_head(h).squeeze(-1) * mask,
            "window_reconstruction": self.window_reconstruction_head(h.mean(dim=1)),
        }
        return self.dssp_head(h).masked_fill(~mask.bool().unsqueeze(-1), -1e4)


def window_auxiliary_loss(model: Model19DSSPModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if model.window_aux_weight <= 0 or not getattr(model, "last_aux", None):
        return batch["target"].float().sum() * 0.0
    target = batch["force_window_proxy"].float().mean(dim=1)
    pred = model.last_aux["window_reconstruction"]
    return model.window_aux_weight * F.smooth_l1_loss(pred, target)
