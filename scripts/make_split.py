"""Write a scene-disjoint train/val/test split for a processed data directory.

    python scripts/make_split.py --data-dir data/processed --output data/processed/split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lspr.data import read_index, scene_disjoint_split  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="default: <data-dir>/split.json")
    p.add_argument("--train-frac", type=float, default=0.85)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    entries = read_index(args.data_dir)
    if not entries:
        raise SystemExit(f"no index*.jsonl entries under {args.data_dir}")
    split = scene_disjoint_split(entries, train=args.train_frac, val=args.val_frac,
                                 seed=args.seed, min_views=args.min_views)
    out = args.output or args.data_dir / "split.json"
    out.write_text(json.dumps(split, indent=1))
    print(" ".join(f"{k}={len(v)}" for k, v in split.items()), "->", out)


if __name__ == "__main__":
    main()
