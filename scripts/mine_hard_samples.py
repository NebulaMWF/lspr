"""Tier training bundles by the error of the current ensemble (hard-sample re-weighting, paper Sec. 3.3).

    python scripts/mine_hard_samples.py --config configs/mine_r1.yaml
    torchrun --nproc_per_node 8 scripts/mine_hard_samples.py --config configs/mine_r1.yaml

The refinement itself is GT-free; the GT of the *training* bundles is used only
afterwards to measure the final Sim(3) translation error t and assign

    easy: t < 0.8 cm      mid: 0.8 cm <= t < 1.5 cm      hard: t >= 1.5 cm

The tier file is consumed by ``scripts/train.py --tier-file`` (``--tier-weights``
sets the sampling mass of each tier). Defaults reproduce the mining configuration
used for the released members (K = 5, sigma in [0.003, 0.15], gate 0.40, scale
calibration on).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lspr import runtime  # noqa: E402
from lspr.config import parse_with_config  # noqa: E402
from lspr.data import BundleDataset, load_split  # noqa: E402
from lspr.evaluation import refine_bundles  # noqa: E402
from lspr.refine import RefineConfig, load_ensemble  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--split-file", type=Path, default=None, help="default: <data-dir>/split.json")
    p.add_argument("--subset", default="train")
    p.add_argument("--max-samples", type=int, default=0, help="mine a random subset of this size (0 = all)")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--checkpoints", type=Path, nargs="+", default=None, help="current ensemble members")
    p.add_argument("--num-steps", type=int, default=5)
    p.add_argument("--sigma-min", type=float, default=0.003)
    p.add_argument("--sigma-max", type=float, default=0.15)
    p.add_argument("--gate-threshold", type=float, default=0.40)
    p.add_argument("--scale-calibration", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--easy-threshold", type=float, default=0.008, help="metres")
    p.add_argument("--hard-threshold", type=float, default=0.015, help="metres")
    p.add_argument("--edge-topk", type=int, default=4)
    p.add_argument("--edge-min-overlap", type=float, default=0.02)
    p.add_argument("--fixed-samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/mining/tiers.json"))
    p.add_argument("--device", default="cuda")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_with_config(build_parser(), argv)
    if not args.checkpoints:
        raise SystemExit("--checkpoints (or `checkpoints:` in the config) is required")
    env = runtime.setup(args.device)
    log = runtime.Logger(None, enabled=env.is_main)

    names = sorted(load_split(args.split_file or args.data_dir / "split.json", args.subset))
    if 0 < args.max_samples < len(names):
        keep = np.random.default_rng(args.sample_seed).permutation(len(names))[:args.max_samples]
        names = sorted(names[int(i)] for i in keep)
    dataset = BundleDataset(args.data_dir, names)
    nets = load_ensemble(args.checkpoints, env.device)
    cfg = RefineConfig(num_steps=args.num_steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                       gate_threshold=args.gate_threshold, scale_calibration=args.scale_calibration)
    log(f"mining {len(dataset)} '{args.subset}' bundles with a {len(nets)}-member ensemble, {cfg}")

    local = refine_bundles(nets, dataset, env.shard(range(len(dataset))), cfg, device=env.device,
                           edge_topk=args.edge_topk, edge_min_overlap=args.edge_min_overlap,
                           num_pixel_samples=args.fixed_samples, seed=args.seed, progress=env.is_main)
    records = runtime.gather(env, local)

    if env.is_main:
        tiers = {"easy": [], "mid": [], "hard": []}
        errors = []
        for r in records:
            if "final" not in r:
                continue
            t = float(r["final"]["t_sim3"])
            errors.append(t)
            tier = "easy" if t < args.easy_threshold else ("hard" if t >= args.hard_threshold else "mid")
            tiers[tier].append(r["name"])
        tiers = {k: sorted(v) for k, v in tiers.items()}
        stats = {
            "num_bundles": len(records),
            "num_scored": len(errors),
            **{f"num_{k}": len(v) for k, v in tiers.items()},
            "t_sim3_median_cm": float(np.median(errors) * 100.0) if errors else None,
        }
        runtime.write_json(args.output, {
            "tiers": tiers,
            "thresholds": {"easy_below_m": args.easy_threshold, "hard_from_m": args.hard_threshold},
            "stats": stats,
            "config": vars(args),
        })
        log(f"{stats} -> {args.output}")
    runtime.cleanup(env)


if __name__ == "__main__":
    main()
