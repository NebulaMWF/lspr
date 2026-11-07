"""Optional GT-free global scale calibration (paper Appendix A.4).

Searches a scalar ``s`` that rescales all camera translations so as to maximise
the cross-view consistency score of the pose graph (sensor depth re-projection).
Three successive 1-D grids are used (coarse, fine +-0.08, micro +-0.01) and the
arg-max is returned. The paper's main configuration does not use this step; it
is provided for initializers whose metric scale drifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from .geometry import consistency_score, gauge_normalize

COARSE_GRID = (0.70, 0.80, 0.85, 0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10, 1.15, 1.25, 1.40)


def scale_translations(poses: torch.Tensor, s: float) -> torch.Tensor:
    out = poses.clone()
    out[:, :3, 3] = out[:, :3, 3] * float(s)
    return gauge_normalize(out)


@dataclass
class ScaleCalibrationResult:
    scale: float
    poses: torch.Tensor
    score_init: float
    score_final: float


def _linspace(center: float, radius: float, steps: int) -> List[float]:
    lo, hi = max(1e-3, center - radius), center + radius
    return [lo + (hi - lo) * i / max(1, steps - 1) for i in range(steps)]


@torch.no_grad()
def calibrate_global_scale(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    depths: torch.Tensor,
    edge_index: torch.Tensor,
    pixels: Dict[int, torch.Tensor],
    *,
    coarse_grid: Optional[Iterable[float]] = None,
    fine_radius: float = 0.08,
    fine_steps: int = 17,
    micro_radius: float = 0.01,
    micro_steps: int = 11,
) -> ScaleCalibrationResult:
    def score(s: float) -> float:
        return consistency_score(scale_translations(poses, s), intrinsics, depths, edge_index, pixels)

    def argmax(values: List[float]) -> tuple:
        scores = [score(s) for s in values]
        k = max(range(len(scores)), key=lambda i: scores[i])
        return float(values[k]), float(scores[k])

    s_coarse, _ = argmax(list(coarse_grid or COARSE_GRID))
    s_fine, _ = argmax(_linspace(s_coarse, fine_radius, fine_steps))
    s_best, score_best = argmax(_linspace(s_fine, micro_radius, micro_steps))
    return ScaleCalibrationResult(
        scale=s_best,
        poses=scale_translations(poses, s_best),
        score_init=score(1.0),
        score_final=score_best,
    )
