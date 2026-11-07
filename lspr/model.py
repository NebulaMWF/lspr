"""LSPR score network (paper Sec. 3.3, Appendix A.2).

Given the current pose set ``T_{1..N}`` (c2w, gauge-normalized), dense DINOv3
patch tokens ``F`` of shape ``(N, P, C_dino)``, the overlap graph and a noise
level ``sigma``, the network predicts a per-view tangent vector
``xi_pred in R^{N x 6}`` (rotation first, then translation). It is trained by
xi-prediction; the score used at inference is ``-xi_pred / sigma^2``
(see ``lspr/diffusion.py``).

* Token projection: DINOv3 tokens are linearly projected to ``hidden_dim``
  and a learnable per-view CLS token is prepended.
* Per-view self-attention: ``num_self_layers`` transformer blocks run on each
  view's tokens independently.
* Sparse cross-view attention: ``num_cross_layers`` blocks in which view i
  attends to the concatenated tokens of its top-K_e overlap neighbours only.
* Conditioning: every block is FiLM-modulated by
  ``MLP(sinusoid(log T_i) || sinusoid(sigma))``.
* Head: CLS token -> MLP -> separate rotation / translation linear heads.

The paper configuration is ``hidden_dim=320, num_heads=8, 4 + 4 layers``
(~14M parameters, see ``configs/``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .se3 import se3_log


# ----------------------------------------------------------------------
# Pose / sigma embedding
# ----------------------------------------------------------------------

def _sinusoidal_embed(values: torch.Tensor, num_freqs: int = 8, base: float = 10000.0) -> torch.Tensor:
    """Map a (..., D) tensor to (..., D * 2 * num_freqs) sinusoidal embedding.

    The frequencies are geometrically spaced from 1 to base.  Used to embed
    SE(3) Lie-algebra elements and the noise scalar.
    """
    if values.dim() == 0:
        values = values.unsqueeze(0)
    *prefix, D = values.shape
    freqs = torch.exp(
        torch.linspace(0.0, math.log(base), num_freqs, device=values.device)
    )
    angles = values.unsqueeze(-1) * freqs
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)
    return torch.cat([sin_part, cos_part], dim=-1).reshape(*prefix, D * 2 * num_freqs)


def pose_to_lie_log(poses_c2w: torch.Tensor) -> torch.Tensor:
    """SE(3) -> tangent-space (..., 6).  Used as part of pose conditioning."""
    R = poses_c2w[..., :3, :3]
    t = poses_c2w[..., :3, 3]
    return se3_log(R, t)


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: ``y = γ ⊙ x + β`` with γ, β from condition."""

    def __init__(self, condition_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(condition_dim, hidden_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gb = self.proj(condition)
        gamma, beta = gb.chunk(2, dim=-1)
        if x.dim() == 3 and condition.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with optional FiLM conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        condition_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.film1 = FiLM(condition_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.film2 = FiLM(condition_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        *,
        kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        if kv is None:
            attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        else:
            kv_norm = self.norm1(kv)
            attn_out, _ = self.attn(h, kv_norm, kv_norm, attn_mask=attn_mask, need_weights=False)
        x = x + self.film1(attn_out, condition)
        x = x + self.film2(self.mlp(self.norm2(x)), condition)
        return x


# ----------------------------------------------------------------------
# Cross-view attention along sparse edges
# ----------------------------------------------------------------------

def gather_neighbors(
    tokens: torch.Tensor,
    edge_index: torch.Tensor,
    num_views: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each view i, gather the concatenated tokens of its neighbors.

    Returns (kv_tokens, kv_padding_mask) where:
      - ``kv_tokens``: shape (N, P*K, hidden) — concatenation of K neighbors'
        token sequences (P tokens each).  K = max degree across views;
        views with fewer neighbors are zero-padded.
      - ``kv_padding_mask``: shape (N, P*K), True at padded positions.

    edge_index has shape (2, E) with edge_index[0] = src, edge_index[1] = dst.
    We treat (src=i, dst=j) as "i attends to j".
    """
    src = edge_index[0].long()
    dst = edge_index[1].long()
    # Group neighbors per src
    neighbors_per_view: list[list[int]] = [[] for _ in range(num_views)]
    for s, d in zip(src.tolist(), dst.tolist()):
        if s == d:
            continue
        neighbors_per_view[s].append(d)
    max_k = max(1, max(len(nbrs) for nbrs in neighbors_per_view))
    P = int(tokens.shape[1])
    H = int(tokens.shape[2])
    device = tokens.device
    kv = torch.zeros((num_views, max_k * P, H), dtype=tokens.dtype, device=device)
    mask = torch.ones((num_views, max_k * P), dtype=torch.bool, device=device)
    for i in range(num_views):
        nbrs = neighbors_per_view[i]
        for k, j in enumerate(nbrs):
            kv[i, k * P:(k + 1) * P] = tokens[j]
            mask[i, k * P:(k + 1) * P] = False
    return kv, mask


# ----------------------------------------------------------------------
# Main network
# ----------------------------------------------------------------------

@dataclass
class ScoreNetworkConfig:
    feature_dim: int = 1024  # DINOv3 ViT-L/16 token dim
    hidden_dim: int = 320
    num_heads: int = 8
    num_self_layers: int = 4
    num_cross_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    sigma_freqs: int = 8
    pose_freqs: int = 4
    rotation_score_scale: float = 1.0
    translation_score_scale: float = 1.0


class ScoreNetwork(nn.Module):
    """Cross-view score network for SE(3) DSM."""

    def __init__(self, cfg: ScoreNetworkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = int(cfg.hidden_dim)
        self.token_proj = nn.Linear(cfg.feature_dim, H)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, H))
        nn.init.normal_(self.cls_token, std=0.02)

        # Sigma embed: scalar -> sinusoidal -> MLP
        sigma_emb_dim = 1 * 2 * cfg.sigma_freqs
        # Pose embed: 6-DoF Lie -> sinusoidal -> MLP
        pose_emb_dim = 6 * 2 * cfg.pose_freqs
        cond_in = sigma_emb_dim + pose_emb_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, H),
            nn.GELU(),
            nn.Linear(H, H),
        )

        self.self_blocks = nn.ModuleList([
            TransformerBlock(H, cfg.num_heads, condition_dim=H, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout)
            for _ in range(cfg.num_self_layers)
        ])
        self.cross_blocks = nn.ModuleList([
            TransformerBlock(H, cfg.num_heads, condition_dim=H, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout)
            for _ in range(cfg.num_cross_layers)
        ])

        self.score_norm = nn.LayerNorm(H)
        self.score_mlp = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, H),
            nn.GELU(),
        )
        self.rot_head = nn.Linear(H, 3)
        self.trans_head = nn.Linear(H, 3)
        nn.init.zeros_(self.rot_head.weight)
        nn.init.zeros_(self.rot_head.bias)
        nn.init.zeros_(self.trans_head.weight)
        nn.init.zeros_(self.trans_head.bias)

    def encode_condition(self, poses: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Build per-view condition vector from pose Lie-algebra and noise level."""
        N = int(poses.shape[0])
        pose_lie = pose_to_lie_log(poses).to(torch.float32)
        pose_emb = _sinusoidal_embed(pose_lie, num_freqs=self.cfg.pose_freqs)
        if sigma.dim() == 0:
            sigma = sigma.expand(N)
        elif sigma.dim() == 1 and sigma.shape[0] == 1:
            sigma = sigma.expand(N)
        sigma_input = sigma.to(torch.float32).unsqueeze(-1)
        sigma_emb = _sinusoidal_embed(sigma_input, num_freqs=self.cfg.sigma_freqs)
        cond_in = torch.cat([sigma_emb, pose_emb], dim=-1)
        return self.cond_mlp(cond_in)  # (N, H)

    def forward(
        self,
        features: torch.Tensor,           # (N, P, C)
        poses: torch.Tensor,              # (N, 4, 4)
        edge_index: torch.Tensor,         # (2, E)
        sigma: torch.Tensor,              # scalar or (N,)
    ) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"features must be (N,P,C), got {tuple(features.shape)}")
        if poses.shape[-2:] != (4, 4):
            raise ValueError(f"poses must be (N,4,4), got {tuple(poses.shape)}")
        N = int(features.shape[0])
        if poses.shape[0] != N:
            raise ValueError(f"features and poses disagree on N: {features.shape[0]} vs {poses.shape[0]}")

        condition = self.encode_condition(poses, sigma)  # (N, H)

        x = self.token_proj(features.to(torch.float32))  # (N, P, H)
        cls = self.cls_token.expand(N, 1, -1)
        x = torch.cat([cls, x], dim=1)  # (N, 1+P, H)

        for blk in self.self_blocks:
            x = blk(x, condition)

        if int(edge_index.shape[1]) > 0:
            # The padding mask is intentionally not applied: views with fewer than
            # max-degree neighbours attend to zero-padded slots. This matches the
            # behaviour the released checkpoints were trained with.
            kv, _kv_mask = gather_neighbors(x, edge_index, num_views=N)
            for blk in self.cross_blocks:
                x = blk(x, condition, kv=kv, attn_mask=None)
        else:
            for blk in self.cross_blocks:
                x = blk(x, condition)

        cls_out = x[:, 0]
        h = self.score_mlp(self.score_norm(cls_out))
        rot = self.rot_head(h) * self.cfg.rotation_score_scale
        trans = self.trans_head(h) * self.cfg.translation_score_scale
        return torch.cat([rot, trans], dim=-1)  # (N, 6)
