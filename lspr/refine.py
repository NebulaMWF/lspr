"""Init-anchored ensemble refinement (paper Sec. 3.4, Algorithm 1).

Starting from the foundation-model initialization, the K-step DDIM solver
follows the averaged xi-prediction of all ensemble members. The optional
confidence gate returns the initialization when the accumulated update norm
exceeds ``tau`` (``tau = 0`` disables it, the paper default), and the optional
post-hoc scale calibration rescales translations (disabled by default).
No ground truth is used anywhere in this module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from .data import PreparedBundle
from .diffusion import NoiseSchedule, ddim_solve
from .geometry import gauge_normalize
from .model import ScoreNetwork, ScoreNetworkConfig
from .scale_calibration import calibrate_global_scale


def load_checkpoint(path: Path) -> Dict:
    """Load a training checkpoint. Falls back to full unpickling for checkpoints
    whose ``args`` contain non-tensor objects; only load files you trust."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        warnings.warn(f"{path}: weights_only load failed, falling back to full unpickling")
        return torch.load(path, map_location="cpu", weights_only=False)


def load_score_network(path: Path, device: torch.device) -> ScoreNetwork:
    payload = load_checkpoint(Path(path))
    net = ScoreNetwork(ScoreNetworkConfig(**payload["score_network_config"])).to(device)
    net.load_state_dict(payload["score_network"])
    return net.eval()


def load_ensemble(paths: Sequence[Path], device: torch.device) -> List[ScoreNetwork]:
    nets = [load_score_network(p, device) for p in paths]
    dims = {n.cfg.feature_dim for n in nets}
    if len(dims) != 1:
        raise ValueError(f"ensemble members disagree on feature_dim: {dims}")
    return nets


@dataclass
class RefineConfig:
    num_steps: int = 10
    sigma_min: float = 0.003
    sigma_max: float = 0.25
    gate_threshold: float = 0.0      # tau; 0 disables the confidence gate
    scale_calibration: bool = False  # optional post-hoc scale calibration


@dataclass
class RefineResult:
    poses: torch.Tensor          # (V, 4, 4) gauge-normalized c2w
    total_update_norm: float     # sum_k ||delta_k||_F
    gate_active: bool
    scale: float                 # post-hoc scale (1.0 when disabled)


@torch.no_grad()
def refine(nets: Sequence[ScoreNetwork], prepared: PreparedBundle, cfg: RefineConfig) -> RefineResult:
    poses_init = gauge_normalize(prepared.poses_init)
    features, edge_index = prepared.features, prepared.edge_index
    weight = 1.0 / float(len(nets))

    def xi_fn(poses: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return sum(weight * net(features, poses, edge_index, sigma) for net in nets)

    schedule = NoiseSchedule(cfg.sigma_min, cfg.sigma_max, cfg.num_steps)
    out = ddim_solve(poses_init, xi_fn, schedule)
    total = float(sum(out.update_norms))
    gate_active = cfg.gate_threshold > 0.0 and total > cfg.gate_threshold
    poses = poses_init if gate_active else out.poses

    scale = 1.0
    if cfg.scale_calibration:
        if prepared.pixels is None:
            raise ValueError("scale calibration needs prepared.pixels (prepare_bundle(with_pixels=True))")
        cal = calibrate_global_scale(poses, prepared.intrinsics, prepared.depths, edge_index, prepared.pixels)
        poses, scale = cal.poses, cal.scale
    return RefineResult(poses=gauge_normalize(poses), total_update_norm=total,
                        gate_active=bool(gate_active), scale=float(scale))
