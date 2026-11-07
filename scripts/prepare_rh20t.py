"""Convert the RH20T dataset into raw multi-camera bundles for ``scripts/preprocess.py``.

Each RH20T scene is one robot trajectory recorded by several fixed RGB-D cameras (plus
in-hand cameras, which are dropped). For every scene this script

  1. selects the largest set of ``--min-views``..``--max-views`` fixed cameras in which every
     camera overlaps with at least ``--overlap-min-neighbors`` others (ground-truth
     co-visibility >= ``--overlap-threshold`` at the middle frame),
  2. samples ``--frames-per-trajectory`` frame indices: the first frame plus random others,
  3. keeps a frame if the same overlap condition holds at that frame,

and writes one ``<scene>_f<frame>.npz`` per kept frame with ``images``, ``depths`` (metres),
``intrinsics``, ``T_gt`` (camera-to-world, relative to the first selected camera) and
``scene_id`` (the trajectory).

Expected layout (official release, unpacked)::

    <rh20t-root>/RH20T_cfg{1..7}/calib/<timestamp>/{intrinsics,extrinsics}.npy
    <rh20t-root>/RH20T_cfg{1..7}/<scene>/metadata.json
    <rh20t-root>/RH20T_cfg{1..7}/<scene>/cam_<serial>/{color.mp4,depth.mp4,timestamps.npy}
    <rh20t-root>/rh20t_api/configs/configs.json        (from the RH20T API repository)

The scene list is built once and stored in ``<output-dir>/rh20t_scenes.json``. A scene's
position in that list seeds its random choices, so ``--num-shards N --shard-index k`` runs
give the same output as a single process. Finished scenes are logged in
``<output-dir>/progress_*.jsonl`` and skipped when the script is restarted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

OVERLAP_POINTS = 8192      # depth points sampled per camera for the co-visibility test
OVERLAP_DEPTH_TOL = 0.05   # relative depth tolerance (at least 2 cm)


# ----------------------------------------------------------------------
# Scene list
# ----------------------------------------------------------------------

def _load_dict(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True).item()  # RH20T stores calibration / timestamps as dicts
    return data if isinstance(data, dict) else None


def _in_hand_serials(config_path: Path) -> Dict[int, List[str]]:
    if not config_path.exists():
        return {}
    return {int(e["conf_num"]): [str(s) for s in e.get("in_hand", [])]
            for e in json.loads(config_path.read_text()) if isinstance(e, dict) and "conf_num" in e}


def _calib_dir(cfg_dir: Path, calib_ts) -> Optional[Path]:
    """``calib/<timestamp>``, searched recursively; folders marked "wrong" are ignored."""
    root, name = cfg_dir / "calib", str(calib_ts).strip()
    if not root.exists() or not name:
        return None

    def ok(p: Path) -> bool:
        return p.is_dir() and not any("wrong" in part.lower() for part in p.relative_to(root).parts)

    if (root / name).is_dir():
        return root / name if ok(root / name) else None
    matches = sorted((p for p in root.rglob(name) if ok(p)), key=lambda p: (len(p.parts), str(p)))
    return matches[0] if matches else None


def _timestamps(cam_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        ts = _load_dict(cam_dir / "timestamps.npy")
        color, depth = (np.asarray(ts.get(k, []), dtype=np.int64) for k in ("color", "depth"))
    except Exception:
        return None
    return (color, depth) if color.size and depth.size else None


def _scan_scene(scene_dir: Path, in_hand: Sequence[str]) -> Optional[Dict]:
    """Calibrated fixed cameras of one scene, each with its frame closest to the common mid time."""
    try:
        calib_ts = json.loads((scene_dir / "metadata.json").read_text()).get("calib")
    except Exception:
        return None
    calib = _calib_dir(scene_dir.parent, calib_ts) if calib_ts is not None else None
    if calib is None:
        return None
    intr, extr = _load_dict(calib / "intrinsics.npy"), _load_dict(calib / "extrinsics.npy")
    if intr is None or extr is None:
        return None

    cams = []
    for serial, K in intr.items():
        cam_dir = scene_dir / f"cam_{serial}"
        if serial in in_hand or not cam_dir.exists():
            continue
        ts = _timestamps(cam_dir)
        K = np.asarray(K, dtype=np.float32)
        w2c = extr.get(serial)
        if isinstance(w2c, list):
            w2c = w2c[0] if w2c else None
        w2c = None if w2c is None else np.asarray(w2c, dtype=np.float32)
        if ts is None or K.shape[0] < 3 or K.shape[1] < 3 or w2c is None or w2c.shape != (4, 4):
            continue
        try:
            c2w = np.linalg.inv(w2c)  # RH20T extrinsics are world-to-camera
        except np.linalg.LinAlgError:
            continue
        cams.append((str(serial), cam_dir, ts, K[:3, :3], c2w))
    if len(cams) < 2:
        return None

    lo, hi = max(int(c[2][0].min()) for c in cams), min(int(c[2][0].max()) for c in cams)
    mid = (lo + hi) // 2 if lo <= hi else int(np.median([float(np.median(c[2][0])) for c in cams]))
    cameras = []
    for serial, cam_dir, (color_ts, depth_ts), K, c2w in cams:
        if not ((cam_dir / "color.mp4").exists() and (cam_dir / "depth.mp4").exists()):
            continue
        color_idx = int(np.argmin(np.abs(color_ts - mid)))
        depth_idx = int(np.argmin(np.abs(depth_ts - int(color_ts[color_idx]))))
        cameras.append({"serial": serial, "K": K.tolist(), "c2w": c2w.tolist(),
                        "mid_color": color_idx, "mid_depth": depth_idx})
    if len(cameras) < 2:
        return None
    return {"cfg": scene_dir.parent.name, "name": scene_dir.name, "cameras": cameras}


def scan_scenes(root: Path, config_path: Path, workers: int = 16) -> List[Dict]:
    in_hand = _in_hand_serials(config_path)
    jobs = []
    for cfg_dir in sorted(root.iterdir()):
        if not (cfg_dir.is_dir() and cfg_dir.name.startswith("RH20T_cfg") and not cfg_dir.name.endswith("_depth")):
            continue
        m = re.search(r"cfg_?(\d+)", cfg_dir.name)
        serials = in_hand.get(int(m.group(1)) if m else -1, [])
        jobs += [(d, serials) for d in sorted(cfg_dir.iterdir())
                 if d.is_dir() and d.name != "calib" and not d.name.endswith("_human")]
    with ThreadPoolExecutor(workers) as pool:
        scenes = list(tqdm(pool.map(lambda job: _scan_scene(*job), jobs), total=len(jobs), desc="scan RH20T"))
    return [s for s in scenes if s is not None]


# ----------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------

def _read_frame(video: Path, index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok and frame is not None else None


def _decode_depth(frame: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Depth videos store 16-bit millimetres as two stacked 8-bit planes (low byte on top)."""
    h, w = hw
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] >= 2 * h:
        depth = gray[h:2 * h, :w].astype(np.int32) * 256 + gray[:h, :w].astype(np.int32)
    else:
        depth = gray.astype(np.int32)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32) / 1000.0


def _scale_intrinsics(K: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Calibration refers to the full sensor resolution (principal point near the centre); rescale to ``hw``."""
    K = K.astype(np.float32, copy=True)
    sx = float(hw[1]) / max(1.0, 2.0 * float(K[0, 2]))
    sy = float(hw[0]) / max(1.0, 2.0 * float(K[1, 2]))
    if abs(sx - 1.0) >= 1e-3 or abs(sy - 1.0) >= 1e-3:
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
    return K


def _load_view(cam_dir: Path, color_idx: int, depth_idx: int, K: np.ndarray):
    """RGB image, metric depth and matching intrinsics of one camera, or None if a frame is unreadable."""
    frame = _read_frame(cam_dir / "color.mp4", color_idx)
    if frame is None:
        return None
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = _read_frame(cam_dir / "depth.mp4", depth_idx)
    if frame is None:
        return None
    return image, _decode_depth(frame, image.shape[:2]), _scale_intrinsics(K, image.shape[:2])


# ----------------------------------------------------------------------
# Ground-truth co-visibility and camera selection
# ----------------------------------------------------------------------

def _visible_fraction(points: np.ndarray, c2w: np.ndarray, K: np.ndarray, depth: np.ndarray) -> float:
    """Fraction of world ``points`` that project into ``depth`` with a consistent depth."""
    cam = (np.linalg.inv(c2w) @ np.hstack([points, np.ones((len(points), 1), np.float32)]).T).T[:, :3]
    cam = cam[cam[:, 2] > 1e-3]
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    H, W = depth.shape
    inside = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    cam = cam[inside]
    d = depth[np.rint(v[inside]).astype(int), np.rint(u[inside]).astype(int)]
    cam, d = cam[d > 0], d[d > 0]
    visible = np.abs(d - cam[:, 2]) <= np.maximum(OVERLAP_DEPTH_TOL * d, 0.02)
    return float(np.count_nonzero(visible)) / float(len(points))


def gt_overlap(depths: Sequence[np.ndarray], intrinsics: Sequence[np.ndarray], poses: Sequence[np.ndarray],
               rng: np.random.Generator, max_points: int = OVERLAP_POINTS) -> Optional[np.ndarray]:
    """``overlap[i, j]``: fraction of sampled view-i depth points seen with consistent depth by view j."""
    points = []
    for depth, K, c2w in zip(depths, intrinsics, poses):
        valid = np.flatnonzero(depth > 0)
        if valid.size == 0:
            return None
        if valid.size > max_points:
            valid = rng.choice(valid, size=max_points, replace=False)
        v, u = valid // depth.shape[1], valid % depth.shape[1]
        z = depth[v, u]
        cam = np.stack([(u - K[0, 2]) / max(K[0, 0], 1e-8) * z, (v - K[1, 2]) / max(K[1, 1], 1e-8) * z, z], axis=1)
        points.append((c2w @ np.concatenate([cam, np.ones((len(cam), 1), np.float32)], axis=1).T).T[:, :3])
    V = len(points)
    overlap = np.eye(V, dtype=np.float32)
    for i, j in itertools.permutations(range(V), 2):
        overlap[i, j] = _visible_fraction(points[i], poses[j], intrinsics[j], depths[j])
    return overlap if (overlap[~np.eye(V, dtype=bool)] > 0).any() else None


def _offdiag(overlap: np.ndarray) -> np.ndarray:
    mat = overlap.astype(np.float32, copy=True)
    np.fill_diagonal(mat, 0.0)
    return mat


def _passes(overlap: np.ndarray, threshold: float, min_neighbors: int) -> bool:
    """Every view overlaps with at least ``min_neighbors`` others."""
    if overlap.shape[0] < 2:
        return False
    adj = _offdiag(overlap) >= threshold
    return int(adj.sum(axis=1).min()) >= min_neighbors and bool(adj.any())


def _mean_overlap(overlap: np.ndarray) -> float:
    V = overlap.shape[0]
    return float(_offdiag(overlap).sum() / float(max(1, V * (V - 1)))) if V >= 2 else 0.0


def _components(adj: np.ndarray) -> List[List[int]]:
    seen = np.zeros(len(adj), dtype=bool)
    comps = []
    for start in range(len(adj)):
        if seen[start]:
            continue
        seen[start], stack, comp = True, [start], []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in np.flatnonzero(adj[node] & ~seen):
                seen[nb] = True
                stack.append(int(nb))
        comps.append(sorted(comp))
    return comps


def choose_cameras(overlap: np.ndarray, min_views: int, max_views: int, threshold: float,
                   min_neighbors: int) -> Optional[List[int]]:
    """Largest valid camera subset (see ``_passes``); ties go to the highest mean overlap."""
    adj = _offdiag(overlap) >= threshold
    best, best_key = None, None
    for comp in _components(adj):
        if len(comp) < min_views:
            continue
        chosen = None
        for size in range(min(max_views, len(comp)), max(2, min_views) - 1, -1):
            chosen, chosen_score = None, None
            for combo in itertools.combinations(comp, size):
                sub = overlap[np.ix_(combo, combo)]
                if not _passes(sub, threshold, min_neighbors):
                    continue
                score = (_mean_overlap(sub), float((_offdiag(sub) >= threshold).sum(axis=1).min()))
                if chosen_score is None or score > chosen_score:
                    chosen, chosen_score = list(combo), score
            if chosen is not None:
                break
        if chosen is None:
            continue
        key = (len(chosen), _mean_overlap(overlap[np.ix_(chosen, chosen)]), tuple(chosen))
        if best_key is None or key > best_key:
            best, best_key = chosen, key
    return best


# ----------------------------------------------------------------------
# Per-scene bundles
# ----------------------------------------------------------------------

def scene_bundles(scene: Dict, root: Path, rng: np.random.Generator,
                  args: argparse.Namespace) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
    """Yield ``(frame_index, arrays)`` for every kept frame of one scene."""
    scene_dir = root / scene["cfg"] / scene["name"]

    # 1. camera subset from the ground-truth overlap at the middle frame
    cams, depths, Ks, poses = [], [], [], []
    for cam in scene["cameras"]:
        view = _load_view(scene_dir / f"cam_{cam['serial']}", cam["mid_color"], cam["mid_depth"],
                          np.asarray(cam["K"], dtype=np.float32))
        if view is not None:
            cams.append(cam)
            depths.append(view[1])
            Ks.append(view[2])
            poses.append(np.asarray(cam["c2w"], dtype=np.float32))
    if len(cams) < args.min_views:
        return
    overlap = gt_overlap(depths, Ks, poses, rng)
    chosen = None if overlap is None else choose_cameras(
        overlap, args.min_views, args.max_views, args.overlap_threshold, args.overlap_min_neighbors)
    if chosen is None:
        return
    cams = [cams[i] for i in chosen]

    # 2. frame indices: the first frame plus random others
    stamps = [_timestamps(scene_dir / f"cam_{c['serial']}") for c in cams]
    if any(s is None for s in stamps):
        return
    n_frames, total = min(len(s[0]) for s in stamps), max(1, args.frames_per_trajectory)
    if n_frames < total:
        return
    extra = rng.choice(np.arange(1, n_frames, dtype=np.int64), size=total - 1, replace=False) if total > 1 else []

    # 3. keep a frame if every camera still has an overlapping neighbour
    poses = [np.asarray(c["c2w"], dtype=np.float32) for c in cams]
    T0_inv = np.linalg.inv(poses[0])
    T_rel = [T0_inv @ T for T in poses]
    for frame in [0, *sorted(int(f) for f in extra)]:
        views = []
        for cam, (color_ts, depth_ts) in zip(cams, stamps):
            depth_idx = int(np.argmin(np.abs(depth_ts - int(color_ts[frame]))))
            view = _load_view(scene_dir / f"cam_{cam['serial']}", frame, depth_idx,
                              np.asarray(cam["K"], dtype=np.float32))
            if view is None:
                break
            views.append(view)
        if len(views) < len(cams):
            continue
        depths, Ks = [v[1] for v in views], [v[2] for v in views]
        overlap = gt_overlap(depths, Ks, T_rel, rng)
        if overlap is None or not _passes(overlap, args.overlap_threshold, args.overlap_min_neighbors):
            continue
        yield frame, {
            "images": np.stack([v[0] for v in views]),
            "depths": np.stack(depths),
            "intrinsics": np.stack(Ks),
            "T_gt": np.stack(T_rel),
            "scene_id": np.str_(scene["name"]),
            "camera_serials": np.array([c["serial"] for c in cams]),
        }


def _write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rh20t-root", type=Path, required=True, help="directory containing RH20T_cfg1 ... RH20T_cfg7")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config-json", type=Path, default=None,
                   help="RH20T configs.json (default: <rh20t-root>/rh20t_api/configs/configs.json)")
    p.add_argument("--frames-per-trajectory", type=int, default=16)
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--max-views", type=int, default=8)
    p.add_argument("--overlap-threshold", type=float, default=0.05)
    p.add_argument("--overlap-min-neighbors", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--scan-workers", type=int, default=16, help="threads for building the scene list")
    p.add_argument("--scan-only", action="store_true", help="only build <output-dir>/rh20t_scenes.json")
    args = p.parse_args(argv)

    root, out = args.rh20t_root, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    scene_list = out / "rh20t_scenes.json"
    if scene_list.exists():
        scenes = json.loads(scene_list.read_text())["scenes"]
    else:
        config = args.config_json or root / "rh20t_api" / "configs" / "configs.json"
        scenes = scan_scenes(root, config, args.scan_workers)
        tmp = scene_list.with_name(f"{scene_list.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"scenes": scenes}))
        os.replace(tmp, scene_list)
    print(f"{len(scenes)} scenes in {scene_list}")
    if args.scan_only:
        return

    done = set()
    for path in out.glob("progress_*.jsonl"):
        done.update(json.loads(line)["scene"] for line in path.read_text().splitlines() if line.strip())
    todo = [(g, s) for g, s in enumerate(scenes)
            if g % args.num_shards == args.shard_index and s["name"] not in done]
    progress = out / f"progress_{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    n_written = 0
    with progress.open("a") as log:
        for g, scene in tqdm(todo, desc=f"shard {args.shard_index}/{args.num_shards}"):
            rng = np.random.default_rng(args.seed + g * 1000003)
            frames = []
            for frame, arrays in scene_bundles(scene, root, rng, args):
                _write_npz(out / f"{scene['name']}_f{frame:05d}.npz", arrays)
                frames.append(frame)
            log.write(json.dumps({"scene": scene["name"], "index": g, "frames": frames}) + "\n")
            log.flush()
            n_written += len(frames)
    print(f"wrote {n_written} bundles from {len(todo)} scenes -> {out}")


if __name__ == "__main__":
    main()
