"""Pose-error metrics and summary statistics (paper Sec. 4, Appendix C.1).

For a bundle with predicted and ground-truth c2w poses we report, per bundle,
the RMSE over views of the translation error (metres) and of the rotation
angle error (degrees) under three alignments:

* ``raw``:  both pose sets are expressed relative to view 0 (no fitting).
* ``se3``:  rigid Umeyama alignment of predicted to GT camera centres.
* ``sim3``: Umeyama alignment with a global scale ``s*`` (the paper's main metric).

An alignment ``x -> s R x + t`` acts on a c2w pose as
``R_v <- R R_v`` and ``t_v <- s R t_v + t``; the residual is ``T_gt^-1 T_aligned``.
Bundles with a single view fall back to the raw metric.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from .se3 import invert_T, so3_log, umeyama_sim3


def _rmse_errors(T_err: torch.Tensor) -> Dict[str, float]:
    trans = torch.linalg.norm(T_err[:, :3, 3], dim=-1)
    rot = torch.linalg.norm(so3_log(T_err[:, :3, :3]), dim=-1)
    return {
        "t": float(torch.sqrt(trans.pow(2).mean())),
        "r": float(torch.rad2deg(torch.sqrt(rot.pow(2).mean()))),
    }


def _aligned(poses: torch.Tensor, s: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = torch.eye(4, dtype=torch.float32, device=poses.device).repeat(poses.shape[0], 1, 1)
    out[:, :3, :3] = R.unsqueeze(0) @ poses[:, :3, :3]
    out[:, :3, 3] = s * (poses[:, :3, 3] @ R.transpose(0, 1)) + t
    return out


@torch.no_grad()
def pose_errors(poses_pred: torch.Tensor, poses_gt: torch.Tensor) -> Dict[str, float]:
    """Per-bundle errors: ``t_{raw,se3,sim3}`` (m), ``r_{raw,se3,sim3}`` (deg), ``scale_sim3``."""
    pred = poses_pred.detach().to(torch.float32).cpu()
    gt = poses_gt.detach().to(torch.float32).cpu()

    pred_rel = invert_T(pred[0]) @ pred
    gt_rel = invert_T(gt[0]) @ gt
    raw = _rmse_errors(invert_T(gt_rel) @ pred_rel)
    out = {"t_raw": raw["t"], "r_raw": raw["r"]}

    if pred.shape[0] < 2:
        out.update(t_se3=raw["t"], r_se3=raw["r"], t_sim3=raw["t"], r_sim3=raw["r"], scale_sim3=1.0)
        return out

    c_pred, c_gt = pred[:, :3, 3], gt[:, :3, 3]
    for name, with_scale in (("se3", False), ("sim3", True)):
        s, R, t = umeyama_sim3(c_pred, c_gt, with_scale=with_scale)
        err = _rmse_errors(invert_T(gt) @ _aligned(pred, s, R, t))
        out[f"t_{name}"] = err["t"]
        out[f"r_{name}"] = err["r"]
        if with_scale:
            out["scale_sim3"] = float(s)
    return out


def _stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def summarize(records: List[Dict], *, keys: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Aggregate per-bundle records with ``init`` / ``final`` error dicts.

    Translation statistics are converted to centimetres. ``choose`` is the
    fraction of bundles whose final Sim(3) translation error is strictly lower
    than that of the (uncalibrated) initialization.
    """
    keys = keys or ["t_raw", "r_raw", "t_se3", "r_se3", "t_sim3", "r_sim3", "scale_sim3"]
    summary: Dict[str, Dict] = {}
    for stage in ("init", "final"):
        stage_recs = [r[stage] for r in records if r.get(stage)]
        summary[stage] = {}
        for k in keys:
            factor = 100.0 if k.startswith("t_") else 1.0
            summary[stage][k + ("_cm" if factor != 1.0 else "")] = _stats(
                factor * float(r[k]) for r in stage_recs if k in r
            )
    paired = [r for r in records if r.get("init") and r.get("final")]
    if paired:
        summary["choose"] = float(np.mean([r["final"]["t_sim3"] < r["init"]["t_sim3"] for r in paired]))
    return summary
