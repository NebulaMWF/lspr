"""Multi-view geometry: gauge, overlap graph, and GT-free cross-view consistency.

All functions take c2w poses (V, 4, 4), pinhole intrinsics (V, 3, 3) and metric
z-depth maps (V, H, W) whose pixel grid matches the intrinsics.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .se3 import invert_T


def gauge_normalize(poses_c2w: torch.Tensor, pivot: int = 0) -> torch.Tensor:
    """Express all poses relative to view ``pivot`` (which becomes the identity)."""
    return invert_T(poses_c2w[pivot]).unsqueeze(0) @ poses_c2w


# ----------------------------------------------------------------------
# Pinhole helpers (pixel coordinates are (y, x))
# ----------------------------------------------------------------------

def unproject(pix_yx: torch.Tensor, z: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    y, x = pix_yx[:, 0].float(), pix_yx[:, 1].float()
    X = (x - K[0, 2]) * z / K[0, 0]
    Y = (y - K[1, 2]) * z / K[1, 1]
    return torch.stack([X, Y, z.float()], dim=-1)


def project(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    z = points[:, 2].clamp_min(1e-6)
    u = K[0, 0] * points[:, 0] / z + K[0, 2]
    v = K[1, 1] * points[:, 1] / z + K[1, 2]
    return torch.stack([v, u], dim=-1)


def sample_bilinear(depth: torch.Tensor, yx: torch.Tensor) -> torch.Tensor:
    H, W = depth.shape
    gx = yx[:, 1].float() / max(1.0, float(W - 1)) * 2.0 - 1.0
    gy = yx[:, 0].float() / max(1.0, float(H - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    out = F.grid_sample(depth.view(1, 1, H, W), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(-1)


def transfer_points(pix_i: torch.Tensor, z_i: torch.Tensor, K_i: torch.Tensor,
                    T_i: torch.Tensor, T_j: torch.Tensor) -> torch.Tensor:
    """Lift pixels of view i with depth and express the 3D points in camera j."""
    P_i = unproject(pix_i, z_i, K_i)
    P_h = torch.cat([P_i, torch.ones_like(P_i[:, :1])], dim=-1)
    return ((invert_T(T_j) @ T_i) @ P_h.unsqueeze(-1)).squeeze(-1)[:, :3]


def sample_valid_pixels(depths: Sequence[torch.Tensor], num_samples: int, seed: int) -> Dict[int, torch.Tensor]:
    """Fixed random subset of valid-depth pixels per view, used by every consistency evaluation."""
    rng = np.random.default_rng(seed)
    out: Dict[int, torch.Tensor] = {}
    for v, d in enumerate(depths):
        ys, xs = torch.nonzero(d > 1e-4, as_tuple=True)
        if ys.numel() == 0:
            continue
        idx = np.arange(int(ys.numel()))
        if idx.shape[0] > num_samples:
            idx = rng.choice(idx, size=num_samples, replace=False)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=d.device)
        out[v] = torch.stack([ys[idx_t], xs[idx_t]], dim=-1)
    return out


# ----------------------------------------------------------------------
# Overlap graph (Sec. 3.3)
# ----------------------------------------------------------------------

@torch.no_grad()
def overlap_matrix(poses_c2w: torch.Tensor, intrinsics: torch.Tensor, depths: torch.Tensor,
                   *, num_samples: int = 4096, rel_tol: float = 0.05, abs_tol: float = 0.02,
                   seed: int = 0) -> torch.Tensor:
    """``O[i, j]``: fraction of sampled view-i points that are visible in view j.

    Up to ``num_samples`` valid-depth pixels of view i are lifted to 3D, transferred
    into camera j, projected, and compared against view j's depth at the nearest
    pixel (tolerance ``max(rel_tol * z_obs, abs_tol)``). Computed once per bundle
    from the foundation-model initialization, so it uses no ground truth.
    Diagonal entries are 1.
    """
    poses_c2w = poses_c2w.to(torch.float32)
    V = int(poses_c2w.shape[0])
    rng = np.random.default_rng(seed)
    out = torch.eye(V, dtype=torch.float32)
    for i in range(V):
        ys, xs = torch.nonzero(depths[i] > 0, as_tuple=True)
        n_valid = int(ys.numel())
        if n_valid == 0:
            continue
        if n_valid > num_samples:
            keep = torch.as_tensor(rng.choice(n_valid, size=num_samples, replace=False), device=ys.device)
            ys, xs = ys[keep], xs[keep]
        pix_i = torch.stack([ys, xs], dim=-1)
        z_i = depths[i][ys, xs].float()
        n_total = float(pix_i.shape[0])
        for j in range(V):
            if i == j:
                continue
            P_j = transfer_points(pix_i, z_i, intrinsics[i].float(), poses_c2w[i], poses_c2w[j])
            P_j = P_j[P_j[:, 2] > 1e-3]
            if P_j.shape[0] == 0:
                continue
            yx = project(P_j, intrinsics[j].float())
            H, W = depths[j].shape
            inside = (yx[:, 0] >= 0) & (yx[:, 0] <= H - 1) & (yx[:, 1] >= 0) & (yx[:, 1] <= W - 1)
            if not bool(inside.any()):
                continue
            P_j, yx = P_j[inside], torch.round(yx[inside]).long()
            z_obs = depths[j][yx[:, 0], yx[:, 1]].float()
            ok = z_obs > 0
            tol = torch.clamp(rel_tol * z_obs[ok], min=abs_tol)
            visible = (z_obs[ok] - P_j[ok, 2]).abs() <= tol
            out[i, j] = float(visible.sum()) / n_total
    return out


def topk_edges(overlap: torch.Tensor, *, topk: int = 4, min_overlap: float = 0.02) -> torch.Tensor:
    """Directed K_e-nearest-neighbour graph: view i attends to its ``topk`` most overlapping views."""
    mat = torch.as_tensor(overlap, dtype=torch.float32)
    n = int(mat.shape[0])
    k = min(int(topk), max(1, n - 1))
    edges = []
    for i in range(n):
        row = mat[i].clone()
        row[i] = -1.0
        vals, idxs = torch.topk(row, k=k)
        edges += [[i, int(j)] for v, j in zip(vals.tolist(), idxs.tolist()) if v >= min_overlap]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ----------------------------------------------------------------------
# GT-free cross-view consistency (scale calibration, Appendix A.4)
# ----------------------------------------------------------------------

def _pair_consistency(pix_i, z_i, K_i, K_j, D_j, T_i, T_j, rel_tol=0.05, abs_tol=0.02) -> Tuple[float, float, int]:
    """Returns (valid fraction, depth agreement, #projected) for the edge i -> j."""
    keep = z_i > 1e-4
    if int(keep.sum()) == 0:
        return 0.0, 0.0, 0
    pix_i, z_i = pix_i[keep], z_i[keep]
    P_j = transfer_points(pix_i, z_i, K_i, T_i, T_j)
    yx = project(P_j, K_j)
    H, W = D_j.shape
    inside = (yx[:, 0] >= 0) & (yx[:, 0] <= H - 1) & (yx[:, 1] >= 0) & (yx[:, 1] <= W - 1)
    if int(inside.sum()) == 0:
        return 0.0, 0.0, 0
    z_pred = P_j[inside, 2]
    z_obs = sample_bilinear(D_j, yx[inside])
    vis = (z_pred > 1e-4) & ((z_pred - z_obs).abs() <= torch.clamp(rel_tol * z_obs.abs(), min=abs_tol))
    valid = float(vis.sum()) / float(max(1, pix_i.shape[0]))
    depth = 1.0 - float((z_pred - z_obs).abs().mean()) / float(torch.clamp(z_obs.abs().mean(), min=1.0))
    return min(1.0, max(0.0, valid)), min(1.0, max(0.0, depth)), int(inside.sum())


@torch.no_grad()
def consistency_score(poses_c2w: torch.Tensor, intrinsics: torch.Tensor, depths: torch.Tensor,
                      edge_index: torch.Tensor, pixels: Dict[int, torch.Tensor],
                      weights: Tuple[float, float, float] = (0.45, 0.40, 0.15)) -> float:
    """``alpha * S_reproj + beta * S_depth + gamma * S_cycle`` averaged over graph edges.

    ``S_cycle`` is the per-edge geometric mean of the first two terms. No GT is used.
    """
    poses = gauge_normalize(poses_c2w)
    vals, deps, cycs = [], [], []
    for e in range(int(edge_index.shape[1])):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if i not in pixels:
            vals.append(0.0); deps.append(0.0); cycs.append(0.0)
            continue
        z_i = depths[i][pixels[i][:, 0], pixels[i][:, 1]]
        v, d, n = _pair_consistency(pixels[i], z_i, intrinsics[i], intrinsics[j], depths[j], poses[i], poses[j])
        if n == 0:
            v = d = 0.0
        vals.append(v); deps.append(d); cycs.append(float(np.sqrt(v * d)))
    if not vals:
        return 0.0
    a, b, c = weights
    return a * float(np.mean(vals)) + b * float(np.mean(deps)) + c * float(np.mean(cycs))
