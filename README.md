# LSPR: Init-Anchored Score-Based Multi-Camera Calibration for Indoor Robot Manipulation

LSPR refines the camera extrinsics of a multi-camera
RGB-D rig, starting from a foundation-model initialization (WorldMirror-v2 in the paper).
It uses a score network on frozen DINOv3 tokens with a few DDIM steps on SE(3).

## Installation

```bash
conda create -n lspr python=3.11 -y && conda activate lspr
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA build
pip install -r requirements.txt
```

`--init worldmirror2` also needs [HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0).
Install its requirements and add it to `PYTHONPATH`.

Pretrained weights are downloaded on first use into `.cache/huggingface/`. Use
`--cache-dir` to change the location, or `--dino-model` / `--worldmirror-model` to
load local copies.

| model | Hub id | access |
|-------|--------|--------|
| DINOv3 ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | gated |
| WorldMirror-v2 | `tencent/HY-World-2.0` (subfolder `HY-WorldMirror-2.0`) | public |

All commands run from the repository root.

## 1. Data

The paper uses [RH20T](https://rh20t.github.io). Unpack the RGB-D release
into `data/RH20T/` and put the [RH20T API](https://github.com/rh20t/rh20t_api) repository at
`data/RH20T/rh20t_api/`.

```bash
# 1) select cameras and frames, write one .npz per synchronized capture to data/raw
python scripts/prepare_rh20t.py --rh20t-root data/RH20T --output-dir data/raw --scan-only
for k in $(seq 0 15); do
  python scripts/prepare_rh20t.py --rh20t-root data/RH20T --output-dir data/raw --num-shards 16 --shard-index $k &
done; wait

# 2) DINOv3 tokens + WorldMirror-v2 initialization, sharded over 8 GPUs
for k in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$k python scripts/preprocess.py --input-dir data/raw --output-dir data/processed \
      --init worldmirror2 --num-shards 8 --shard-index $k &
done; wait

# 3) train / val / test split
python scripts/make_split.py --data-dir data/processed
```

Other datasets use the same pipeline from step 2. Write one `.npz` per synchronized capture to
`data/raw/`:

| key          | shape        | dtype   | notes                                       |
|--------------|--------------|---------|---------------------------------------------|
| `images`     | (V, H, W, 3) | uint8   | RGB                                         |
| `depths`     | (V, H, W)    | float32 | metric depth, 0 = invalid                   |
| `intrinsics` | (V, 3, 3)    | float32 |                                             |
| `T_gt`       | (V, 4, 4)    | float32 | GT camera-to-world; optional at test time   |
| `init_c2w`   | (V, 4, 4)    | float32 | own initialization, used with `--init bundle` |
| `scene_id`   | str          |         | optional, groups captures for the split     |

## 2. Training

The four ensemble members are trained in sequence:

```bash
python scripts/train.py             --config configs/m1_scratch.yaml
python scripts/train.py             --config configs/m2_warm_uniform.yaml
python scripts/mine_hard_samples.py --config configs/mine_r1.yaml
python scripts/train.py             --config configs/m3_hardmine_r1.yaml
python scripts/mine_hard_samples.py --config configs/mine_r2.yaml
python scripts/train.py             --config configs/m4_hardmine_r2.yaml
```

The paper trains on 8 GPUs: replace `python` with `torchrun --nproc_per_node 8`.

## 3. Evaluation

```bash
python scripts/evaluate.py --config configs/eval.yaml                 # test split, paper setting
python scripts/evaluate.py --config configs/eval.yaml --subset all --save-poses \
    --data-dir data/my_rig --output runs/eval/my_rig.json            # own data, no GT needed
```

The results go to a JSON file. When `T_gt` is available, it includes raw / SE(3) / Sim(3)
pose errors. `--save-poses` also writes the refined camera-to-world poses, with view 0
as the identity.

From Python:

```python
from lspr import BundleDataset, RefineConfig, load_ensemble, prepare_bundle, refine

nets = load_ensemble(["runs/m1_scratch/checkpoint_last.pt", "runs/m2_warm_uniform/checkpoint_best.pt",
                      "runs/m3_hardmine_r1/checkpoint_best.pt", "runs/m4_hardmine_r2/checkpoint_last.pt"], "cuda")
bundle = prepare_bundle(BundleDataset("data/processed")[0], device="cuda", with_pixels=False)
poses = refine(nets, bundle, RefineConfig()).poses     # (V, 4, 4)
```
