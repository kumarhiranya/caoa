"""Complete every pose pair with a CAPCM checkpoint, as the legacy pose dataset did (P8).

    uv run python scripts/build_pose_completions.py --name odg_s2ccompletion_upTransb1 \
        --ckpt experiments/UpTrans/S2CCompletion/s2ccompletion_upTransb1/ckpt-best.pth \
        --config experiments/UpTrans/S2CCompletion/s2ccompletion_upTransb1/config.yaml

See `caoa/data/pose_completion.py` for what is reproduced (incl. the legacy GT-box-frame input).
`--frame world` runs the model in the world frame instead (the leak-free fix; cache `<name>__world`).
"""

import argparse
import pickle

import numpy as np
import torch
import yaml
from tqdm import tqdm

from caoa.data.pairs import load_pairs
from caoa.data.pose_completion import COMPLETION_FRAMES, complete_pairs, pose_completion_dir
from caoa.models.odgnet import UpTrans, load_legacy_checkpoint
from caoa.paths import load_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="completion model name, used as the cache folder")
    parser.add_argument("--ckpt", required=True, help="checkpoint, relative to the legacy ODGNet folder")
    parser.add_argument("--config", required=True, help="legacy run config (for the model section), relative to ODGNet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame", choices=COMPLETION_FRAMES, default="legacy", help="frame the model runs in")
    args = parser.parse_args()

    paths = load_paths()
    odgnet = paths.legacy_repo / "ODGNet"
    device = torch.device("cuda")
    model = UpTrans.from_config(yaml.safe_load(open(odgnet / args.config))["model"]).to(device).eval()
    info = load_legacy_checkpoint(model, odgnet / args.ckpt)
    print(f"{args.name}: {args.ckpt} (epoch {info['epoch']}), frame {args.frame}")

    np.random.seed(args.seed)
    pairs = load_pairs(paths.cache, class_rule="relpose_cache")
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
