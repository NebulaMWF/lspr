"""SE(3) / SO(3) Lie-group utilities and Umeyama alignment."""

from __future__ import annotations

import torch


def hat(v: torch.Tensor) -> torch.Tensor:
    h = torch.zeros(v.shape[:-1] + (3, 3), device=v.device, dtype=v.dtype)
    h[..., 0, 1] = -v[..., 2]
    h[..., 0, 2] = v[..., 1]
    h[..., 1, 0] = v[..., 2]
    h[..., 1, 2] = -v[..., 0]
    h[..., 2, 0] = -v[..., 1]
    h[..., 2, 1] = v[..., 0]
    return h


def so3_log(R: torch.Tensor) -> torch.Tensor:
    orig_dtype = R.dtype
    R = R.to(torch.float32)

    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    sin_theta = torch.sin(theta)
    w_hat = 0.5 * (R - R.transpose(-1, -2))
    w = torch.stack([w_hat[..., 2, 1], w_hat[..., 0, 2], w_hat[..., 1, 0]], dim=-1)

    small = theta < 1e-4
    large = ~small

    if torch.any(large):
        w_large = (theta[large] / (2.0 * sin_theta[large])).unsqueeze(-1) * w[large]
        w = w.clone()
        w[large] = w_large

    return w.to(orig_dtype)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    orig_dtype = w.dtype
    w = w.to(torch.float32)
    theta = torch.norm(w, dim=-1, keepdim=True)
    w_hat = hat(w)
    I = torch.eye(3, device=w.device, dtype=torch.float32)

    small = theta.squeeze(-1) < 1e-4
    R = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=torch.float32)

    if torch.any(small):
        w_hat_s = w_hat[small]
        R_s = I + w_hat_s + 0.5 * (w_hat_s @ w_hat_s)
        R[small] = R_s

    if torch.any(~small):
        w_hat_ns = w_hat[~small]
        k = theta[~small].squeeze(-1)
        R_ns = (
            I
            + torch.sin(k).view(-1, 1, 1) / k.view(-1, 1, 1) * w_hat_ns
            + (1 - torch.cos(k)).view(-1, 1, 1) / (k * k).view(-1, 1, 1) * (w_hat_ns @ w_hat_ns)
        )
        R[~small] = R_ns

    return R.to(orig_dtype)


def se3_log(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    orig_dtype = R.dtype
    R = R.to(torch.float32)
    t = t.to(torch.float32)

    w = so3_log(R).to(torch.float32)
    theta = torch.norm(w, dim=-1)
    theta_sq = theta * theta

    I = torch.eye(3, device=R.device, dtype=torch.float32)
    w_hat = hat(w)

    small = theta < 1e-4
    V_inv = torch.zeros_like(R)

    if torch.any(small):
        V_inv_s = I - 0.5 * w_hat + (1.0 / 12.0) * (w_hat @ w_hat)
        V_inv[small] = V_inv_s[small]

    if torch.any(~small):
        theta_ns = theta[~small]
        w_hat_ns = w_hat[~small]
        theta_sq_ns = theta_sq[~small]
        term1 = (1.0 / theta_sq_ns) * (1 - (0.5 * theta_ns / torch.tan(0.5 * theta_ns)))
        V_inv_ns = I - 0.5 * w_hat_ns + term1.view(-1, 1, 1) * (w_hat_ns @ w_hat_ns)
        V_inv[~small] = V_inv_ns

    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)
    xi = torch.cat([w, v], dim=-1)
    return xi.to(orig_dtype)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    orig_dtype = xi.dtype
    xi = xi.to(torch.float32)
    w = xi[..., :3]
    v = xi[..., 3:]
    theta = torch.norm(w, dim=-1)
    I = torch.eye(3, device=xi.device, dtype=torch.float32)
    w_hat = hat(w)

    small = theta < 1e-4
    V = torch.zeros(xi.shape[:-1] + (3, 3), device=xi.device, dtype=torch.float32)
    R = so3_exp(w).to(torch.float32)

    if torch.any(small):
        V_s = I + 0.5 * w_hat + (1.0 / 6.0) * (w_hat @ w_hat)
        V[small] = V_s[small]

    if torch.any(~small):
        w_hat_ns = w_hat[~small]
        theta_ns = theta[~small]
        theta_sq_ns = theta_ns * theta_ns
        term1 = (1 - torch.cos(theta_ns)).view(-1, 1, 1) / theta_sq_ns.view(-1, 1, 1)
        term2 = (theta_ns - torch.sin(theta_ns)).view(-1, 1, 1) / (theta_sq_ns * theta_ns).view(-1, 1, 1)
        V_ns = I + term1 * w_hat_ns + term2 * (w_hat_ns @ w_hat_ns)
        V[~small] = V_ns

    t = (V @ v.unsqueeze(-1)).squeeze(-1)
    T = torch.eye(4, device=xi.device, dtype=torch.float32).expand(xi.shape[:-1] + (4, 4)).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T.to(orig_dtype)


def invert_T(T: torch.Tensor) -> torch.Tensor:
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (...,4,4), got {tuple(T.shape)}")
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    tinv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    out = torch.eye(4, device=T.device, dtype=T.dtype).expand(T.shape).clone()
    out[..., :3, :3] = Rt
    out[..., :3, 3] = tinv
    return out


def compose_T(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.shape[-2:] != (4, 4) or B.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (...,4,4), got {tuple(A.shape)} and {tuple(B.shape)}")
    return A @ B


def umeyama_sim3(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    with_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Umeyama Sim(3) alignment: find scale s, rotation R, translation t such that
    s * R @ src_i + t ≈ dst_i.

    Returns (s, R, t) with s scalar, R 3x3, t (3,). If ``with_scale`` is False, returns
    rigid SE(3) alignment (s == 1).
    """
    if src.shape != dst.shape or src.dim() != 2 or src.shape[1] != 3:
        raise ValueError(f"umeyama_sim3 expects (N,3) src/dst, got {tuple(src.shape)}, {tuple(dst.shape)}")
    src = src.to(torch.float64)
    dst = dst.to(torch.float64)
    mu_src = src.mean(dim=0)
    mu_dst = dst.mean(dim=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    H = src_c.transpose(0, 1) @ dst_c / float(max(1, src.shape[0]))
    U, D, Vt = torch.linalg.svd(H)
    S = torch.eye(3, dtype=torch.float64, device=src.device)
    if torch.det(U) * torch.det(Vt) < 0:
        S[2, 2] = -1.0
    R = (Vt.transpose(0, 1) @ S @ U.transpose(0, 1))
    if with_scale:
        var_src = src_c.pow(2).sum() / float(max(1, src.shape[0]))
        s = (torch.diag(S) * D).sum() / var_src.clamp_min(1e-12)
    else:
        s = torch.tensor(1.0, dtype=torch.float64, device=src.device)
    t = mu_dst - s * (R @ mu_src)
    return s.to(torch.float32), R.to(torch.float32), t.to(torch.float32)

