"""Convert raw multi-camera RGB-D bundles into LSPR training/evaluation bundles.

Input: a directory of ``*.npz`` files, one per synchronized capture, with

    images      (V, H, W, 3) uint8     RGB
    depths      (V, H, W)    float32   metric z-depth aligned with ``images`` (0 = invalid)
    intrinsics  (V, 3, 3)    float32   pinhole intrinsics of ``images``
    T_gt        (V, 4, 4)    float32   ground-truth camera-to-world (optional, needed for training)
    init_c2w    (V, 4, 4)    float32   initialization (optional if --init worldmirror2)
    scene_id    str                    scene identifier for scene-disjoint splits (optional)

For each bundle this script
  1. extracts DINOv3 patch tokens (square 518 px canvas, aspect-preserving resize, black padding),
  2. takes the initialization from the file or runs WorldMirror-v2 (``--init worldmirror2``),
  3. resizes depth / intrinsics to ``--depth-max-side`` and computes the init-based overlap matrix,
and writes ``<output>/bundles/<name>.pt`` plus one line per bundle in ``<output>/index*.jsonl``.

Run several processes with ``--num-shards N --shard-index k`` to parallelize over GPUs.

Pretrained weights (DINOv3, WorldMirror-v2) are downloaded from the Hugging Face Hub on
first use into ``--cache-dir`` (default ``.cache/huggingface`` in the repository root).
DINOv3 is a gated model: request access on its model page and log in with
``huggingface-cli login`` (or set ``HF_TOKEN``) before the first run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lspr.geometry import overlap_matrix  # noqa: E402

DEFAULT_DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_WORLDMIRROR = "tencent/HY-World-2.0"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"

_GATED_HINT = ("If this is a gated model (e.g. DINOv3), request access on its Hugging Face page, "
               "then run `huggingface-cli login` or set HF_TOKEN. Alternatively pass a local path.")


# ----------------------------------------------------------------------
# DINOv3 patch tokens
# ----------------------------------------------------------------------

class DINOv3Extractor:
    def __init__(self, model_name: str, device: torch.device, image_size: int = 518,
                 cache_dir: Optional[Path] = None) -> None:
        from transformers import AutoImageProcessor, AutoModel

        cache = str(cache_dir) if cache_dir is not None else None
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache)
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache).to(device).eval()
        except OSError as exc:
            raise OSError(f"could not load '{model_name}'. {_GATED_HINT}") from exc
        self.device = device
        self.image_size = int(image_size)
        self.num_special = 1 + int(getattr(self.model.config, "num_register_tokens", 0))

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> torch.Tensor:
        """``(V, H, W, 3)`` uint8 -> ``(V, P, C)`` float16 patch tokens."""
        size = (self.image_size, self.image_size)
        canvases = []
        for img in images:
            pil = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB")
            resized = ImageOps.contain(pil, size)
            canvas = Image.new("RGB", size, (0, 0, 0))
            canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
            canvases.append(canvas)
        inputs = self.processor(images=canvases, return_tensors="pt",
                                size={"height": self.image_size, "width": self.image_size})
        hidden = self.model(pixel_values=inputs["pixel_values"].to(self.device)).last_hidden_state
        return hidden[:, self.num_special:, :].to(torch.float16).cpu()


# ----------------------------------------------------------------------
# Optional WorldMirror-v2 initialization (requires HY-World-2.0 installed)
# ----------------------------------------------------------------------

class WorldMirrorInit:
    """Wraps HY-WorldMirror-2.0 inference (518 px input, bf16 autocast)."""

    def __init__(self, model_name: str, subfolder: str, device: torch.device, target_size: int = 518,
                 cache_dir: Optional[Path] = None) -> None:
        try:
            from safetensors.torch import load_file
            from hyworld2.worldrecon.pipeline import _load_model_config, _load_state_dict_selective, _resolve_model_dir
            from hyworld2.worldrecon.hyworldmirror.models.models.worldmirror import WorldMirror
        except ImportError as exc:
            raise ImportError(
                "--init worldmirror2 needs the HY-World-2.0 package (module `hyworld2`) on PYTHONPATH"
            ) from exc
        root = model_name
        if not Path(model_name).is_dir():
            # HF repo id: fetch only the WorldMirror subfolder into the local cache
            from huggingface_hub import snapshot_download
            root = snapshot_download(repo_id=model_name, allow_patterns=[f"{subfolder}/*"],
                                     cache_dir=str(cache_dir) if cache_dir is not None else None)
        model_dir = _resolve_model_dir(root, subfolder)
        self.model = WorldMirror(**_load_model_config(model_dir)).to(device).eval()
        _load_state_dict_selective(self.model, load_file(str(Path(model_dir) / "model.safetensors")),
                                   source_name=model_dir)
        self.device = device
        self.target_size = int(target_size)

    def _prepare(self, images: np.ndarray) -> torch.Tensor:
        S = self.target_size
        views = []
        for img in images:
            pil = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB")
            w, h = pil.size
            new_w, new_h = S, round(h * (S / w) / 14) * 14
            arr = np.asarray(pil.resize((new_w, new_h), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            if t.shape[1] > S:
                top = (t.shape[1] - S) // 2
                t = t[:, top:top + S, :]
            views.append(t)
        return torch.stack(views).unsqueeze(0)

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> torch.Tensor:
        batch = self._prepare(images).to(self.device)
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
            pred = self.model(views={"img": batch}, cond_flags=[0, 0, 0])
        poses = pred["camera_poses"].float().cpu()
        if poses.dim() == 4:
            poses = poses[0]
        if poses.shape[-2:] == (3, 4):
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(poses.shape[0], 1, 4)
            poses = torch.cat([poses, bottom], dim=1)
        if not torch.isfinite(poses).all():
            raise ValueError("non-finite WorldMirror poses")
        return poses


# ----------------------------------------------------------------------
# Depth / intrinsics resizing
# ----------------------------------------------------------------------

def resize_depth(depths: np.ndarray, intrinsics: np.ndarray, max_side: int):
    """Nearest-neighbour resize so that max(H, W) <= max_side; intrinsics follow."""
    depths_t = torch.as_tensor(depths, dtype=torch.float32)
    K = torch.as_tensor(intrinsics, dtype=torch.float32).clone()
    H, W = depths_t.shape[-2:]
    scale = min(1.0, float(max_side) / float(max(H, W))) if max_side > 0 else 1.0
    if scale >= 1.0:
        return depths_t, K
    h, w = int(round(H * scale)), int(round(W * scale))
    sy, sx = h / H, w / W
    depths_t = F.interpolate(depths_t.unsqueeze(1), size=(h, w), mode="nearest-exact").squeeze(1)
    K[:, 0, 0] *= sx
    K[:, 1, 1] *= sy
    K[:, 0, 2] = (K[:, 0, 2] + 0.5) * sx - 0.5
    K[:, 1, 2] = (K[:, 1, 2] + 0.5) * sy - 0.5
    return depths_t, K


def _optional(npz, key: str) -> Optional[np.ndarray]:
    return np.asarray(npz[key]) if key in npz.files else None


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init", choices=["bundle", "worldmirror2"], default="bundle",
                   help="use init_c2w stored in each npz, or run WorldMirror-v2")
    p.add_argument("--dino-model", default=DEFAULT_DINO, help="HF id or local path of the DINOv3 ViT-L/16 model")
    p.add_argument("--dino-image-size", type=int, default=518)
    p.add_argument("--worldmirror-model", default=DEFAULT_WORLDMIRROR,
                   help="HF id or local path of the HY-World-2.0 repo")
    p.add_argument("--worldmirror-subfolder", default="HY-WorldMirror-2.0")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help="download cache for pretrained weights (default: <repo>/.cache/huggingface)")
    p.add_argument("--depth-max-side", type=int, default=518, help="0 keeps the input resolution")
    p.add_argument("--overlap-samples", type=int, default=4096)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    files = sorted(args.input_dir.glob("*.npz"))[args.shard_index::args.num_shards]
    out_dir = args.output_dir / "bundles"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / f"index_{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"

    dino = DINOv3Extractor(args.dino_model, device, args.dino_image_size, cache_dir=args.cache_dir)
    wm = (WorldMirrorInit(args.worldmirror_model, args.worldmirror_subfolder, device, cache_dir=args.cache_dir)
          if args.init == "worldmirror2" else None)

    n_written, n_skipped = 0, 0
    with index_path.open("a") as index_file:
        for path in tqdm(files, desc=f"shard {args.shard_index}/{args.num_shards}"):
            name = path.stem
            target = out_dir / f"{name}.pt"
            if target.exists() and not args.overwrite:
                continue
            with np.load(path, allow_pickle=False) as npz:
                images = np.asarray(npz["images"])
                depths = np.asarray(npz["depths"], dtype=np.float32)
                intrinsics = np.asarray(npz["intrinsics"], dtype=np.float32)
                T_gt = _optional(npz, "T_gt")
                init = _optional(npz, "init_c2w")
                scene_id = str(npz["scene_id"]) if "scene_id" in npz.files else name
            V = int(images.shape[0])
            if V < args.min_views or depths.shape[:3] != images.shape[:3]:
                tqdm.write(f"[skip] {name}: {V} views / depth-image shape mismatch")
                n_skipped += 1
                continue

            if wm is not None:
                init_t = wm(images)
            elif init is not None:
                init_t = torch.as_tensor(init, dtype=torch.float32)
            else:
                raise ValueError(f"{name}: no init_c2w in the bundle; use --init worldmirror2")

            depths_t, K_t = resize_depth(depths, intrinsics, args.depth_max_side)
            overlap = overlap_matrix(init_t, K_t, depths_t, num_samples=args.overlap_samples, seed=0)
            if float((overlap - torch.eye(V)).max()) <= 0.0:
                tqdm.write(f"[skip] {name}: no overlapping view pair under the initialization")
                n_skipped += 1
                continue

            record = {
                "tokens": dino(images),
                "init_c2w": init_t,
                "T_gt": None if T_gt is None else torch.as_tensor(T_gt, dtype=torch.float32),
                "depths": depths_t,
                "intrinsics": K_t,
                "overlap": overlap,
                "meta": {
                    "name": name,
                    "scene_id": scene_id,
                    "num_views": V,
                    "feature_model": str(args.dino_model),
                    "init_source": args.init,
                },
            }
            torch.save(record, target)
            index_file.write(json.dumps({"name": name, "scene_id": scene_id, "num_views": V}) + "\n")
            index_file.flush()
            n_written += 1
    print(f"wrote {n_written} bundles, skipped {n_skipped} -> {args.output_dir}")


if __name__ == "__main__":
    main()
