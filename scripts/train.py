"""Train one LSPR score network with GT-anchored denoising score matching (paper Sec. 3.2-3.3).

Single GPU:
    python scripts/train.py --config configs/m1_scratch.yaml
Multi-GPU (one bundle per step per rank, gradients averaged by DDP):
    torchrun --nproc_per_node 8 scripts/train.py --config configs/m1_scratch.yaml

Each step draws one bundle and one noise level sigma (shared by its views),
perturbs every gauge-normalized GT pose by xi ~ N(0, sigma^2 I_6), re-gauges,
and regresses the effective tangent residual (``lspr.diffusion.dsm_training_pair``).
Early stopping monitors the same xi-loss on the validation split.
``checkpoint_best.pt`` / ``checkpoint_last.pt`` are written to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lspr import runtime  # noqa: E402
from lspr.config import parse_with_config  # noqa: E402
from lspr.data import BundleDataset, PreparedBundle, load_split, prepare_bundle  # noqa: E402
from lspr.diffusion import dsm_training_pair, sample_train_sigma, xi_prediction_loss  # noqa: E402
from lspr.geometry import gauge_normalize  # noqa: E402
from lspr.model import ScoreNetwork, ScoreNetworkConfig  # noqa: E402
from lspr.refine import load_checkpoint  # noqa: E402
from lspr.se3 import invert_T, se3_exp, se3_log  # noqa: E402

TIERS = ("easy", "mid", "hard")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    g.add_argument("--split-file", type=Path, default=None, help="default: <data-dir>/split.json")
    g.add_argument("--train-subset", default="train")
    g.add_argument("--val-subset", default="val")
    g.add_argument("--edge-topk", type=int, default=4)
    g.add_argument("--edge-min-overlap", type=float, default=0.02)

    g = p.add_argument_group("model")
    g.add_argument("--hidden-dim", type=int, default=320)
    g.add_argument("--num-heads", type=int, default=8)
    g.add_argument("--num-self-layers", type=int, default=4)
    g.add_argument("--num-cross-layers", type=int, default=4)
    g.add_argument("--sigma-freqs", type=int, default=8)
    g.add_argument("--pose-freqs", type=int, default=4)
    g.add_argument("--init-checkpoint", type=Path, default=None,
                   help="warm start from another member's weights (optimizer state is not restored)")

    g = p.add_argument_group("denoising score matching")
    g.add_argument("--sigma-min", type=float, default=0.005)
    g.add_argument("--sigma-max", type=float, default=0.2)
    g.add_argument("--sigma-sampling", choices=["uniform", "shifted", "log_uniform"], default="shifted")
    g.add_argument("--feature-aug-std", type=float, default=0.03,
                   help="multiplicative Gaussian jitter on DINOv3 tokens (training steps only)")

    g = p.add_argument_group("optimization")
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--samples-per-epoch", type=int, default=2048, help="training bundles per rank per epoch")
    g.add_argument("--val-samples-per-epoch", type=int, default=128, help="validation bundles per rank")
    g.add_argument("--early-stop-patience", type=int, default=6, help="0 disables early stopping")
    g.add_argument("--early-stop-min-delta", type=float, default=5e-4)

    g = p.add_argument_group("hard-sample re-weighting")
    g.add_argument("--tier-file", type=Path, default=None, help="output of scripts/mine_hard_samples.py")
    g.add_argument("--tier-weights", type=float, nargs=4, default=[1.0, 2.0, 3.0, 0.0],
                   metavar=("EASY", "MID", "HARD", "UNTIERED"),
                   help="sampling mass per tier; UNTIERED covers training bundles absent from the tier file")

    g = p.add_argument_group("runtime")
    g.add_argument("--output-dir", type=Path, default=Path("runs/lspr"))
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="cuda")
    return p


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------

def draw_uniform(pool_size: int, num: int, rng: np.random.Generator) -> np.ndarray:
    """Without replacement when the pool is large enough, otherwise with replacement."""
    if num <= pool_size:
        return rng.permutation(pool_size)[:num]
    return rng.integers(0, pool_size, size=num)


def tier_probabilities(train_names: Sequence[str], tier_file: Path, weights: Sequence[float]) -> np.ndarray:
    """Per-bundle sampling probability: each tier's mass is spread uniformly over its
    training members, so the tier total is independent of the tier size."""
    tiers = json.loads(Path(tier_file).read_text())["tiers"]
    pos = {n: i for i, n in enumerate(train_names)}
    prob = np.zeros(len(train_names), dtype=np.float64)
    tiered = np.zeros(len(train_names), dtype=bool)
    for tier, w in zip(TIERS, weights[:3]):
        members = [pos[n] for n in tiers.get(tier, []) if n in pos]
        if w > 0 and members:
            prob[members] += w / len(members)
            tiered[members] = True
    untiered = np.flatnonzero(~tiered)
    if weights[3] > 0 and untiered.size:
        prob[untiered] += weights[3] / untiered.size
    if prob.sum() <= 0:
        raise ValueError(f"{tier_file}: the tier weights select no training bundle")
    return prob / prob.sum()


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------

def mean_tangent_residual(poses: torch.Tensor, ref: torch.Tensor) -> float:
    rel = poses @ invert_T(ref)
    return float(se3_log(rel[:, :3, :3], rel[:, :3, 3]).norm(dim=-1).mean())


def train_step(net: torch.nn.Module, b: PreparedBundle, args: argparse.Namespace,
               rng: torch.Generator) -> Tuple[torch.Tensor, Dict[str, float]]:
    if b.poses_gt is None:
        raise ValueError(f"bundle {b.name} has no T_gt; training needs ground-truth poses")
    features = b.features
    if args.feature_aug_std > 0:
        noise = torch.randn(features.shape, generator=rng, device=features.device, dtype=features.dtype)
        features = features * (1.0 + args.feature_aug_std * noise)
    sigma = sample_train_sigma(args.sigma_min, args.sigma_max, device=features.device, rng=rng,
                               distribution=args.sigma_sampling)
    noisy, xi_eff = dsm_training_pair(b.poses_gt, sigma, rng=rng)
    xi_pred = net(features, noisy, b.edge_index, sigma)
    loss = xi_prediction_loss(xi_pred, xi_eff, sigma)
    return loss, {"loss": float(loss.item()), "sigma": float(sigma.item()),
                  "xi_eff_norm": float(xi_eff.norm(dim=-1).mean())}


@torch.no_grad()
def val_step(net: ScoreNetwork, b: PreparedBundle, args: argparse.Namespace,
             rng: torch.Generator) -> Dict[str, float]:
    """Validation xi-loss plus a one-step denoising diagnostic from the initialization."""
    if b.poses_gt is None:
        raise ValueError(f"bundle {b.name} has no T_gt; validation needs ground-truth poses")
    sigma = sample_train_sigma(args.sigma_min, args.sigma_max, device=b.features.device, rng=rng,
                               distribution=args.sigma_sampling)
    noisy, xi_eff = dsm_training_pair(b.poses_gt, sigma, rng=rng)
    loss = xi_prediction_loss(net(b.features, noisy, b.edge_index, sigma), xi_eff, sigma)
    denoised = gauge_normalize(se3_exp(-net(b.features, b.poses_init, b.edge_index, sigma)) @ b.poses_init)
    return {
        "val_loss": float(loss.item()),
        "val_init_to_gt": mean_tangent_residual(b.poses_init, b.poses_gt),
        "val_denoised_to_gt": mean_tangent_residual(denoised, b.poses_gt),
    }


def save_checkpoint(path: Path, net: ScoreNetwork, opt: torch.optim.Optimizer,
                    args: argparse.Namespace, epoch: int, best_val_loss: float) -> None:
    torch.save({
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "args": runtime.to_jsonable(vars(args)),
        "score_network": net.state_dict(),
        "optimizer": opt.state_dict(),
        "score_network_config": asdict(net.cfg),
    }, path)


def build_network(args: argparse.Namespace, feature_dim: int, log: runtime.Logger) -> ScoreNetwork:
    cfg = ScoreNetworkConfig(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_self_layers=args.num_self_layers,
        num_cross_layers=args.num_cross_layers,
        sigma_freqs=args.sigma_freqs,
        pose_freqs=args.pose_freqs,
    )
    net = ScoreNetwork(cfg)
    if args.init_checkpoint is not None:
        payload = load_checkpoint(args.init_checkpoint)
        src = payload["score_network_config"]
        for key in ("feature_dim", "hidden_dim", "num_heads", "num_self_layers", "num_cross_layers",
                    "sigma_freqs", "pose_freqs"):
            if src.get(key) != getattr(cfg, key):
                raise ValueError(f"--init-checkpoint mismatch on {key}: {src.get(key)} vs {getattr(cfg, key)}")
        missing, unexpected = net.load_state_dict(payload["score_network"], strict=False)
        log(f"warm start from {args.init_checkpoint} (missing={len(missing)}, unexpected={len(unexpected)})")
    return net


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_with_config(build_parser(), argv)
    env = runtime.setup(args.device)
    device = env.device
    torch.manual_seed(args.seed + env.rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = runtime.Logger(args.output_dir / "train.log", enabled=env.is_main)

    split_file = args.split_file or args.data_dir / "split.json"
    train_set = BundleDataset(args.data_dir, load_split(split_file, args.train_subset))
    val_set = BundleDataset(args.data_dir, load_split(split_file, args.val_subset))
    if len(train_set) == 0 or len(val_set) == 0:
        raise SystemExit(f"empty split: train={len(train_set)} val={len(val_set)}")
    feature_dim = int(train_set[0]["tokens"].shape[-1])

    module = build_network(args, feature_dim, log).to(device)
    net: torch.nn.Module = module
    if env.enabled:
        net = DDP(module, device_ids=[env.local_rank] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log(f"ScoreNetwork {sum(p.numel() for p in module.parameters()) / 1e6:.2f}M params, "
        f"train={len(train_set)} val={len(val_set)} world_size={env.world_size}")
    if env.is_main:
        runtime.write_json(args.output_dir / "args.json", vars(args))

    # Epoch indices come from one generator shared (same seed) by all ranks and are
    # strided by rank; the noise generator differs per rank.
    np_rng = np.random.default_rng(args.seed)
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(args.seed + env.rank * 7919)
    prob = None
    if args.tier_file is not None:
        prob = tier_probabilities(train_set.names, args.tier_file, args.tier_weights)
        log(f"tier-weighted sampling from {args.tier_file}: weights={args.tier_weights}, "
            f"support={int((prob > 0).sum())}/{len(prob)}")

    def prepare(ds: BundleDataset, i: int) -> PreparedBundle:
        return prepare_bundle(ds[i], device=device, edge_topk=args.edge_topk,
                              edge_min_overlap=args.edge_min_overlap, with_pixels=False)

    best_val, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        n_global = args.samples_per_epoch * env.world_size
        if prob is None:
            global_idx = draw_uniform(len(train_set), n_global, np_rng)
        else:
            global_idx = np_rng.choice(len(train_set), size=n_global, replace=True, p=prob)

        net.train()
        acc: Dict[str, List[float]] = {}
        t0 = time.time()
        for i in tqdm(env.shard(global_idx.tolist()), desc=f"epoch {epoch:03d}", leave=False,
                      disable=not env.is_main):
            loss, metrics = train_step(net, prepare(train_set, int(i)), args, torch_rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_grad_norm)
            opt.step()
            for k, v in metrics.items():
                acc.setdefault(k, []).append(v)
        train_means = {k: runtime.all_mean(env, float(np.mean(v))) for k, v in acc.items()}

        module.eval()
        val_idx = draw_uniform(len(val_set), args.val_samples_per_epoch * env.world_size, np_rng)
        vacc: Dict[str, List[float]] = {}
        for i in env.shard(val_idx.tolist()):
            for k, v in val_step(module, prepare(val_set, int(i)), args, torch_rng).items():
                vacc.setdefault(k, []).append(v)
        val_means = {k: runtime.all_mean(env, float(np.mean(v))) for k, v in vacc.items()}
        elapsed = time.time() - t0

        log(f"[epoch {epoch:03d}] loss={train_means['loss']:.5f} sigma={train_means['sigma']:.4f} "
            f"val_loss={val_means['val_loss']:.5f} init->gt={val_means['val_init_to_gt']:.4f} "
            f"one-step->gt={val_means['val_denoised_to_gt']:.4f} ({elapsed:.0f}s)")
        history.append({"epoch": epoch, "train": train_means, "val": val_means, "elapsed_s": elapsed})
        if env.is_main:
            runtime.write_json(args.output_dir / "history.json", history)

        if val_means["val_loss"] < best_val - args.early_stop_min_delta:
            best_val, stale = val_means["val_loss"], 0
            if env.is_main:
                save_checkpoint(args.output_dir / "checkpoint_best.pt", module, opt, args, epoch, best_val)
                log(f"[best] epoch {epoch} val_loss={best_val:.5f}")
        else:
            stale += 1
            if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
                log(f"[early stop] no improvement for {stale} epochs")
                break
        if env.is_main:
            save_checkpoint(args.output_dir / "checkpoint_last.pt", module, opt, args, epoch, best_val)

    runtime.cleanup(env)


if __name__ == "__main__":
    main()
