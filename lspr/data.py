"""Processed-bundle format, dataset, splits, and per-bundle preparation.

A *bundle* is one synchronized multi-camera capture. ``scripts/preprocess.py``
writes one ``<name>.pt`` file per bundle into ``<data_dir>/bundles/`` with::

    tokens      (V, P, C)  float16   DINOv3 patch tokens (registers / CLS removed)
    init_c2w    (V, 4, 4)  float32   foundation-model initialization (c2w)
    T_gt        (V, 4, 4)  float32   ground-truth c2w (metres, OpenCV); optional
    depths      (V, H, W)  float32   metric z-depth, 0 = invalid
    intrinsics  (V, 3, 3)  float32   pinhole intrinsics matching ``depths``
    overlap     (V, V)     float32   init-based co-visibility (see geometry.overlap_matrix)
    meta        dict                 name, scene_id, num_views, feature_model, init_source

and appends one JSON line per bundle to ``<data_dir>/index*.jsonl``.
A split file (``scripts/make_split.py``) is ``{"train": [names], "val": [...], "test": [...]}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .geometry import gauge_normalize, sample_valid_pixels, topk_edges


def read_index(data_dir: Path) -> List[Dict]:
    """All index entries (``name``, ``scene_id``, ``num_views``) sorted by name."""
    entries: Dict[str, Dict] = {}
    for path in sorted(Path(data_dir).glob("index*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                entries[rec["name"]] = rec
    return [entries[k] for k in sorted(entries)]


def load_split(split_file: Path, subset: str) -> List[str]:
    split = json.loads(Path(split_file).read_text())
    if subset not in split:
        raise KeyError(f"subset {subset!r} not in {split_file} (has {sorted(split)})")
    return list(split[subset])


class BundleDataset:
    """Lazy list of processed bundles stored as ``<data_dir>/bundles/<name>.pt``."""

    def __init__(self, data_dir: Path, names: Optional[Sequence[str]] = None) -> None:
        self.data_dir = Path(data_dir)
        if names is None:
            names = [e["name"] for e in read_index(self.data_dir)]
            if not names:
                names = sorted(p.stem for p in (self.data_dir / "bundles").glob("*.pt"))
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    def path(self, idx: int) -> Path:
        return self.data_dir / "bundles" / f"{self.names[idx]}.pt"

    def __getitem__(self, idx: int) -> Dict:
        return torch.load(self.path(idx), map_location="cpu", weights_only=True)


@dataclass
class PreparedBundle:
    name: str
    features: torch.Tensor                 # (V, P, C) float32
    poses_init: torch.Tensor               # (V, 4, 4) gauge-normalized
    poses_gt: Optional[torch.Tensor]       # (V, 4, 4) gauge-normalized, or None
    intrinsics: torch.Tensor               # (V, 3, 3)
    depths: torch.Tensor                   # (V, H, W)
    edge_index: torch.Tensor               # (2, E)
    pixels: Optional[Dict[int, torch.Tensor]]  # fixed pixel samples for consistency scoring

    @property
    def num_views(self) -> int:
        return int(self.poses_init.shape[0])


def prepare_bundle(
    bundle: Dict,
    *,
    device: torch.device,
    edge_topk: int = 4,
    edge_min_overlap: float = 0.02,
    num_pixel_samples: int = 128,
    seed: int = 0,
    with_pixels: bool = True,
) -> PreparedBundle:
    """Move a bundle to ``device``, fix the gauge (view 0), and build the overlap graph."""
    f32 = dict(device=device, dtype=torch.float32)
    depths = torch.as_tensor(bundle["depths"]).to(**f32)
    T_gt = bundle.get("T_gt")
    return PreparedBundle(
        name=str(bundle.get("meta", {}).get("name", "")),
        features=torch.as_tensor(bundle["tokens"]).to(**f32),
        poses_init=gauge_normalize(torch.as_tensor(bundle["init_c2w"]).to(**f32)),
        poses_gt=None if T_gt is None else gauge_normalize(torch.as_tensor(T_gt).to(**f32)),
        intrinsics=torch.as_tensor(bundle["intrinsics"]).to(**f32),
        depths=depths,
        edge_index=topk_edges(bundle["overlap"], topk=edge_topk, min_overlap=edge_min_overlap).to(device),
        pixels=sample_valid_pixels(list(depths), num_pixel_samples, seed) if with_pixels else None,
    )


def scene_disjoint_split(
    entries: Sequence[Dict], *, train: float = 0.85, val: float = 0.10, seed: int = 42, min_views: int = 2
) -> Dict[str, List[str]]:
    """Partition bundles by ``scene_id`` so no scene appears in two subsets."""
    by_scene: Dict[str, List[str]] = {}
    for e in entries:
        if int(e.get("num_views", 0)) < min_views:
            continue
        by_scene.setdefault(str(e.get("scene_id", e["name"])), []).append(e["name"])
    scenes = sorted(by_scene)
    order = np.random.default_rng(seed).permutation(len(scenes))
    n_train = int(round(train * len(scenes)))
    n_val = int(round(val * len(scenes)))
    groups = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }
    return {k: sorted(n for i in idx for n in by_scene[scenes[int(i)]]) for k, idx in groups.items()}
