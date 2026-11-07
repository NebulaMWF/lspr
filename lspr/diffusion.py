"""Tangent-space noise schedule, DSM training target, and the DDIM-style solver.

Conventions (see the paper, Sec. 3.2 and 3.4):

* Poses are camera-to-world (c2w) 4x4 matrices, one per view.
* The gauge is fixed by mapping view 0 to the identity (``gauge_normalize``).
* Training perturbs *every* view, ``T_v^sigma = exp(xi_v) T_v^gt`` with
  ``xi_v ~ N(0, sigma^2 I_6)``, re-gauges, and regresses the effective residual
  ``xi_v^eff = log(T_v^sigma (T_v^gt)^-1)`` (the anchor target is zero).
* The network predicts ``xi`` ("xi-prediction"); the score used by the solver is
  ``-xi_pred / sigma^2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from .geometry import gauge_normalize
from .se3 import invert_T, se3_exp, se3_log


@dataclass
class NoiseSchedule:
    """Cosine schedule ``sigma(t) = s_min + 0.5 (s_max - s_min)(1 - cos(pi t))``."""

    sigma_min: float
    sigma_max: float
    num_steps: int

    def sigma_at(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        u = 0.5 * (1.0 - torch.cos(math.pi * t))
        return self.sigma_min + (self.sigma_max - self.sigma_min) * u

    def inference_grid(self, device: torch.device) -> torch.Tensor:
        """Decreasing sigma sequence of length ``num_steps + 1`` (t = 1, ..., 0)."""
        ts = torch.linspace(1.0, 0.0, int(self.num_steps) + 1, device=device)
        return self.sigma_at(ts)


def sample_train_sigma(
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    rng: Optional[torch.Generator] = None,
    distribution: str = "uniform",
) -> torch.Tensor:
    """Draw one training noise level (shared by all views of a bundle).

    * ``uniform``:  sigma = s_min + u (s_max - s_min)
    * ``shifted``:  sigma = s_min + u_s (s_max - s_min), u_s = 1 - (1 - u)^(1/2).
      The density of u_s is 2 (1 - u_s), i.e. biased toward *small* noise; the
      mean is s_min + (s_max - s_min) / 3.
    * ``log_uniform``: sigma = exp(U(log s_min, log s_max)).
    """
    u = torch.rand((1,), device=device, generator=rng)
    dist = str(distribution).lower()
    if dist == "uniform":
        return sigma_min + u * (sigma_max - sigma_min)
    if dist == "shifted":
        u_s = 1.0 - (1.0 - u).pow(0.5)
        return sigma_min + u_s * (sigma_max - sigma_min)
    if dist == "log_uniform":
        lo, hi = math.log(max(sigma_min, 1e-8)), math.log(sigma_max)
        return torch.exp(lo + u * (hi - lo))
    raise ValueError(f"unknown sigma distribution {distribution!r}")


def perturb_poses(
    poses: torch.Tensor, sigma: torch.Tensor, *, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    """``exp(xi_v) T_v`` with ``xi_v ~ N(0, sigma^2 I_6)`` for every view."""
    eps = torch.randn((int(poses.shape[0]), 6), device=poses.device, dtype=torch.float32, generator=rng)
    xi = eps * sigma.reshape(-1, 1).to(torch.float32)
    return se3_exp(xi) @ poses.to(torch.float32)


def dsm_training_pair(
    poses_gt: torch.Tensor, sigma: torch.Tensor, *, rng: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(T^sigma, xi^eff)``: the gauge-normalized noisy pose set and its target."""
    anchor = gauge_normalize(poses_gt)
    noisy = gauge_normalize(perturb_poses(anchor, sigma, rng=rng))
    rel = noisy @ invert_T(anchor)
    xi_eff = se3_log(rel[:, :3, :3], rel[:, :3, 3]).to(torch.float32)
    return noisy, xi_eff


def xi_prediction_loss(xi_pred: torch.Tensor, xi_true: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Weighted xi regression, ``mean(w(sigma) (xi_pred - xi_true)^2)``, averaged over all views.

    ``w(sigma) = min(1 / sigma^2, 5)``. For the training ranges used in the paper
    (sigma <= 0.2) the clamp is always active, so the weight is the constant 5.
    """
    sigma_view = sigma.reshape(-1).expand(xi_true.shape[0]).unsqueeze(-1).clamp_min(1e-3)
    weight = (1.0 / sigma_view.pow(2)).clamp_max(5.0)
    return (weight * (xi_pred - xi_true).pow(2)).mean()


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""``xi_fn(poses, sigma) -> (V, 6)`` predicted tangent noise."""


@dataclass
class SolverOutput:
    poses: torch.Tensor
    update_norms: List[float]


@torch.no_grad()
def ddim_solve(poses_init: torch.Tensor, xi_fn: ScoreFn, schedule: NoiseSchedule) -> SolverOutput:
    """Deterministic K-step DDIM-style ODE on se(3)^V, started from the initialization.

    ``delta_k = (sigma_k - sigma_{k-1}) sigma_k * score``, ``score = -xi_pred / sigma_k^2``.
    The update is applied to all views and the pose set is re-gauged after every
    step, so the anchor's update is absorbed into the other views.
    ``update_norms[k]`` is the Frobenius norm of the V x 6 update (used by the gate).
    """
    grid = schedule.inference_grid(poses_init.device)
    poses = gauge_normalize(poses_init.to(torch.float32))
    norms: List[float] = []
    for k in range(int(schedule.num_steps)):
        sigma_k, sigma_next = grid[k], grid[k + 1]
        xi_pred = xi_fn(poses, sigma_k)
        score = -xi_pred / sigma_k.clamp_min(1e-3).pow(2)
        delta = (sigma_k - sigma_next) * sigma_k * score
        poses = gauge_normalize(se3_exp(delta) @ poses)
        norms.append(float(torch.linalg.norm(delta).item()))
    return SolverOutput(poses=poses, update_norms=norms)
