from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


DSSP_CLASSES = ["C", "H", "B", "E", "G", "I", "T", "S"]


def contact_to_topk_strength(contact: torch.Tensor, top_n: int = 12, sigma: float = 0.6) -> torch.Tensor:
    """Convert a full contact/distance map into a sparse symmetric strength matrix.

    This follows the model6/model8/model11 convention: if values look like
    distances, use exp(-d/sigma); if values are already contact strengths, keep
    them as non-negative weights. The diagonal is removed and only top-k contacts
    per residue are retained.
    """
    x = contact.float().clamp_min(0.0)
    valid = (x > 0).float()
    max_val = x.detach().amax() if x.numel() else torch.tensor(0.0, device=x.device)
    if max_val > 1.5:
        strength = torch.exp(-x / sigma) * valid
    else:
        strength = x * valid
    n = strength.shape[-1]
    eye = torch.eye(n, device=strength.device, dtype=torch.bool).unsqueeze(0)
    strength = strength.masked_fill(eye, 0.0)
    k = min(int(top_n), n)
    vals, idx = torch.topk(strength, k=k, dim=-1)
    out = torch.zeros_like(strength)
    out.scatter_(-1, idx, vals)
    return torch.maximum(out, out.transpose(-1, -2))


class DenseGraphSAGEBatch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 96, out_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.msg1 = nn.Linear(hidden_dim, hidden_dim)
        self.msg2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, adj: torch.Tensor, node: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = adj.shape
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        a = (adj + eye) * mask.unsqueeze(1) * mask.unsqueeze(2)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1.0)
        h = self.in_proj(node)
        z = torch.bmm(a, h)
        z = self.dropout(F.gelu(self.norm1(self.msg1(z))))
        h = h + z
        z = torch.bmm(a, h)
        z = self.dropout(F.gelu(self.norm2(self.msg2(z))))
        h = h + z
        return self.out_proj(h) * mask.unsqueeze(-1)


class ContactRowCNNEncoder(nn.Module):
    """Per-residue CNN over full contact-map rows.

    This branch is intentionally different from row summary statistics. It reads
    the complete L-dimensional row pattern for every residue and learns local
    contact motifs before graph message passing.
    """

    def __init__(self, max_len: int = 46, out_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.max_len = int(max_len)
        self.net = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(24, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Linear(32, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout))

    def forward(self, contact: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = contact.shape
        rows = contact.reshape(b * n, 1, n)
        feat = self.net(rows).squeeze(-1)
        feat = self.proj(feat).view(b, n, -1)
        return feat * mask.unsqueeze(-1)


class FullContactMapDSSPModel(nn.Module):
    """model18 full-contact branch inspired by model6/model11/model12.

    Inputs are frame-level tensors from the clean-300 preprocessing database.
    The model keeps the model12 goal of residue-level SS8 prediction but replaces
    scalar contact-map row summaries with a full LxL contact/distance-map encoder:
    row-CNN + top-k DenseGraphSAGE + multimodal fusion.
    """

    def __init__(
        self,
        n_residue_features: int,
        n_dynamic: int,
        n_contact_features: int,
        hidden: int = 128,
        graph_dim: int = 64,
        top_n: int = 12,
        dropout: float = 0.20,
        use_dynamic: bool = True,
        use_contact_features: bool = True,
        use_residue_features: bool = True,
        use_residue_position: bool = True,
        use_full_contact_map: bool = True,
    ):
        super().__init__()
        self.use_dynamic = use_dynamic
        self.use_contact_features = use_contact_features
        self.use_residue_features = use_residue_features
        self.use_residue_position = use_residue_position
        self.use_full_contact_map = use_full_contact_map
        self.top_n = int(top_n)
        node_in = 0
        if use_residue_features:
            node_in += n_residue_features
        if use_residue_position:
            node_in += 2
        if node_in == 0:
            node_in = 1
        self.node_proj = nn.Sequential(nn.Linear(node_in, graph_dim), nn.GELU(), nn.LayerNorm(graph_dim), nn.Dropout(dropout))
        self.row_cnn = ContactRowCNNEncoder(out_dim=graph_dim, dropout=dropout)
        self.graph = DenseGraphSAGEBatch(graph_dim, hidden_dim=hidden, out_dim=graph_dim, dropout=dropout)
        scalar_dim = 0
        if use_dynamic:
            scalar_dim += n_dynamic
        if use_contact_features:
            scalar_dim += n_contact_features
        self.scalar_proj = nn.Sequential(nn.Linear(max(scalar_dim, 1), graph_dim), nn.GELU(), nn.LayerNorm(graph_dim), nn.Dropout(dropout))
        fusion_dim = graph_dim
        if use_full_contact_map:
            fusion_dim += graph_dim * 2
        fusion_dim += graph_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
        )
        self.dssp_head = nn.Linear(hidden, 8)
        self.contact_number_head = nn.Linear(hidden, 1)
        self.contact_density_head = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cmap = batch["contact_map"].float()
        mask = batch["mask"].float()
        parts = []
        if self.use_residue_features:
            parts.append(batch["residue_features"].float())
        if self.use_residue_position:
            parts.append(batch["position"].float())
        if parts:
            node_raw = torch.cat(parts, dim=-1)
        else:
            node_raw = mask.unsqueeze(-1)
        node = self.node_proj(node_raw) * mask.unsqueeze(-1)
        fused = [node]
        if self.use_full_contact_map:
            row_feat = self.row_cnn(cmap, mask)
            adj = contact_to_topk_strength(cmap, top_n=self.top_n)
            graph_feat = self.graph(adj, node + row_feat, mask)
            fused.extend([row_feat, graph_feat])
        scalar_parts = []
        if self.use_dynamic:
            scalar_parts.append(batch["dynamic"].float())
        if self.use_contact_features:
            scalar_parts.append(batch["contact_features"].float())
        if scalar_parts:
            scalar = torch.cat(scalar_parts, dim=-1)
        else:
            scalar = torch.ones(mask.shape[0], 1, device=mask.device, dtype=mask.dtype)
        scalar_feat = self.scalar_proj(scalar).unsqueeze(1).expand(-1, mask.shape[1], -1) * mask.unsqueeze(-1)
        fused.append(scalar_feat)
        h = self.fusion(torch.cat(fused, dim=-1)) * mask.unsqueeze(-1)
        self.last_aux = {
            "contact_number": self.contact_number_head(h).squeeze(-1) * mask,
            "contact_density": self.contact_density_head(h).squeeze(-1) * mask,
        }
        return self.dssp_head(h).masked_fill(~mask.bool().unsqueeze(-1), -1e4)
