"""Evaluate an LSPR ensemble with the paper's main protocol (Sec. 4).

    python scripts/evaluate.py --config configs/eval.yaml
    torchrun --nproc_per_node 8 scripts/evaluate.py --config configs/eval.yaml

Defaults: K = 10 DDIM steps on a cosine sigma grid over [0.003, 0.25], uniform
average of the members' xi-predictions, confidence gate off (tau = 0), and no
post-hoc scale calibration. Inference is GT-free; when bundles carry ``T_gt``
the script reports raw / SE(3) / Sim(3) errors of the initialization and of the
refined poses. Output: ``--output`` (JSON with config, summary, per-bundle records)
and, with ``--save-poses``, ``<output>.poses.pt`` holding gauge-normalized
(view 0 = identity) c2w poses per bundle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lspr import runtime  # noqa: E402
from lspr.config import parse_with_config  # noqa: E402
from lspr.data import BundleDataset, load_split, read_index  # noqa: E402
from lspr.evaluation import refine_bundles  # noqa: E402
from lspr.metrics import summarize  # noqa: E402
from lspr.refine import RefineConfig, load_ensemble  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--split-file", type=Path, default=None, help="default: <data-dir>/split.json")
    p.add_argument("--subset", default="test", help="split subset, or 'all' for every indexed bundle")
    p.add_argument("--limit", type=int, default=0, help="evaluate a fixed random subset of this size (0 = all)")
    p.add_argument("--limit-seed", type=int, default=42)
    p.add_argument("--checkpoints", type=Path, nargs="+", default=None, help="ensemble members")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--sigma-min", type=float, default=0.003)
    p.add_argument("--sigma-max", type=float, default=0.25)
    p.add_argument("--gate-threshold", type=float, default=0.0, help="tau; 0 disables the confidence gate")
    p.add_argument("--scale-calibration", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--edge-topk", type=int, default=4)
    p.add_argument("--edge-min-overlap", type=float, default=0.02)
    p.add_argument("--fixed-samples", type=int, default=128, help="pixels per view for consistency scoring")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/eval/results.json"))
    p.add_argument("--save-poses", action="store_true")
    p.add_argument("--device", default="cuda")
    return p


def select_names(args: argparse.Namespace) -> List[str]:
    if args.subset == "all":
        names = [e["name"] for e in read_index(args.data_dir) if int(e.get("num_views", 0)) >= 2]
    else:
        names = load_split(args.split_file or args.data_dir / "split.json", args.subset)
    names = sorted(names)
    if 0 < args.limit < len(names):
        keep = np.random.default_rng(args.limit_seed).permutation(len(names))[:args.limit]
        names = sorted(names[int(i)] for i in keep)
    return names


def format_summary(summary: Dict) -> str:
    rows = []
    for stage in ("init", "final"):
        s = summary.get(stage, {})
        t, r, t_se3 = s.get("t_sim3_cm", {}), s.get("r_sim3", {}), s.get("t_se3_cm", {})
        if not t.get("n"):
            continue
        rows.append(f"{stage:>5}  Sim(3) t_med {t['median']:.3f} cm  t_mean {t['mean']:.3f} cm  "
                    f"t_p90 {t['p90']:.3f} cm  r_med {r['median']:.3f} deg  |  SE(3) t_med {t_se3['median']:.3f} cm")
    if "choose" in summary:
        rows.append(f"choose (final Sim(3) t < init) = {summary['choose']:.3f}")
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_with_config(build_parser(), argv)
    if not args.checkpoints:
        raise SystemExit("--checkpoints (or `checkpoints:` in the config) is required")
    env = runtime.setup(args.device)
    log = runtime.Logger(None, enabled=env.is_main)

    dataset = BundleDataset(args.data_dir, select_names(args))
    nets = load_ensemble(args.checkpoints, env.device)
    cfg = RefineConfig(num_steps=args.num_steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                       gate_threshold=args.gate_threshold, scale_calibration=args.scale_calibration)
    log(f"{len(nets)}-member ensemble, {len(dataset)} bundles ({args.subset}), {cfg}")

    local = refine_bundles(nets, dataset, env.shard(range(len(dataset))), cfg, device=env.device,
                           edge_topk=args.edge_topk, edge_min_overlap=args.edge_min_overlap,
                           num_pixel_samples=args.fixed_samples, seed=args.seed,
                           keep_poses=args.save_poses, progress=env.is_main)
    records = sorted(runtime.gather(env, local), key=lambda r: r["name"])

    if env.is_main:
        poses = {r["name"]: {"init": r.pop("poses_init"), "final": r.pop("poses_final")}
                 for r in records if "poses_final" in r}
        ok = [r for r in records if "error" not in r]
        summary = summarize(ok)
        runtime.write_json(args.output, {
            "config": vars(args),
            "num_bundles": len(records),
            "num_failed": len(records) - len(ok),
            "summary": summary,
            "records": records,
        })
        if args.save_poses:
            torch.save(poses, args.output.with_suffix(".poses.pt"))
        log(format_summary(summary) or "no ground truth in these bundles (use --save-poses to keep the poses)")
        log(f"{len(ok)}/{len(records)} bundles refined -> {args.output}")
    runtime.cleanup(env)


if __name__ == "__main__":
    main()
