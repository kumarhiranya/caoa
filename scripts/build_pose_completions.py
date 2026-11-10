"""Complete every object–CAD pair with a CAPCM checkpoint: the OCAM's (and the retrieval's) completion input.

    uv run python scripts/build_pose_completions.py --name capcm_concat --run outputs/capcm_concat

The model runs on each instance's world-frame points (`caoa/data/pose_completion.py`); cache `<name>__world`.
"""

import argparse
import pickle

import numpy as np
import torch
import yaml
from tqdm import tqdm

from caoa.data.pairs import load_pairs
from caoa.data.pose_completion import COMPLETION_FRAMES, complete_pairs, pose_completion_dir
from caoa.models.odgnet import UpTrans, load_checkpoint
from caoa.paths import REPO_ROOT, load_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="completion model name, used as the cache folder")
    parser.add_argument("--run", required=True, help="CAPCM run folder (ckpt-best.pth, config.yaml), relative to the repo or absolute")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame", choices=COMPLETION_FRAMES, default="world")
    args = parser.parse_args()

    paths = load_paths()
    run = REPO_ROOT / args.run
    device = torch.device("cuda")
    model = UpTrans.from_config(yaml.safe_load(open(run / "config.yaml"))["model"]).to(device).eval()
    info = load_checkpoint(model, run / "ckpt-best.pth")
    print(f"{args.name}: {run} (epoch {info['epoch']}), frame {args.frame}")

    np.random.seed(args.seed)
    pairs = load_pairs(paths.cache)
    by_scene = {}
    for p in pairs:
        by_scene.setdefault(p.scene_id, []).append(p)
    out = pose_completion_dir(paths.cache, args.name, args.frame)
    out.mkdir(parents=True, exist_ok=True)
    for scene_id, ps in tqdm(sorted(by_scene.items()), desc="scenes"):
        if (out / f"{scene_id}.pkl").exists():
            continue
        completed = complete_pairs(model, ps, device, args.frame)
        tmp = out / f"{scene_id}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump({p.model_index: c for p, c in zip(ps, completed)}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(out / f"{scene_id}.pkl")
    size = sum(f.stat().st_size for f in out.glob("*.pkl")) / 2**30
    print(f"{len(pairs)} pairs completed -> {out} ({size:.2f} GiB)")


if __name__ == "__main__":
    main()
