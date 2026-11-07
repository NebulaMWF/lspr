"""Run the refinement over a list of bundles and collect per-bundle records.

Ground truth, when present, is used only *after* refinement to compute errors.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from tqdm import tqdm

from .data import BundleDataset, prepare_bundle
from .metrics import pose_errors
from .model import ScoreNetwork
from .refine import RefineConfig, refine


def refine_bundles(
    nets: Sequence[ScoreNetwork],
    dataset: BundleDataset,
    positions: Sequence[int],
    cfg: RefineConfig,
    *,
    device: torch.device,
    edge_topk: int = 4,
    edge_min_overlap: float = 0.02,
    num_pixel_samples: int = 128,
    seed: int = 0,
    keep_poses: bool = False,
    progress: bool = True,
) -> List[Dict]:
    """One record per bundle: ``name``, ``num_views``, solver statistics and, with GT,
    ``init`` / ``final`` error dicts (``lspr.metrics.pose_errors``). A bundle that
    fails gets ``{"name", "error"}`` instead. ``positions`` index into ``dataset``;
    the fixed pixel samples of a bundle are seeded with ``seed + position``."""
    records: List[Dict] = []
    for pos in tqdm(list(positions), desc="refine", disable=not progress):
        name = dataset.names[int(pos)]
        try:
            b = prepare_bundle(dataset[int(pos)], device=device, edge_topk=edge_topk,
                               edge_min_overlap=edge_min_overlap, num_pixel_samples=num_pixel_samples,
                               seed=seed + int(pos), with_pixels=cfg.scale_calibration)
            res = refine(nets, b, cfg)
        except Exception as exc:  # keep going; failures are reported in the summary
            records.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rec: Dict = {
            "name": name,
            "num_views": b.num_views,
            "total_update_norm": res.total_update_norm,
            "gate_active": res.gate_active,
            "scale": res.scale,
        }
        if b.poses_gt is not None:
            rec["init"] = pose_errors(b.poses_init, b.poses_gt)
            rec["final"] = pose_errors(res.poses, b.poses_gt)
        if keep_poses:
            rec["poses_init"] = b.poses_init.cpu()
            rec["poses_final"] = res.poses.cpu()
        records.append(rec)
    return records
