"""CAD surface points for the Siamese OCAM: 8,192 uniform samples per Scan2CAD CAD, in the CAD's own
bounding-box frame (centred, longest side 1; no annotation). One .npz {catid_idcad: (8192, 3) float32} in
`<cache>/cad_points_v1/` (~300 MB for the 3,049 CADs).

    uv run python scripts/build_cad_points.py
"""

import argparse
import pickle

import numpy as np
from tqdm import tqdm

from caoa.data.pose_dataset import cad_points_path
from caoa.data.shapenet import load_cad_mesh, sample_uniform
from caoa.paths import load_paths

NUM_POINTS = 8192


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = load_paths()
    with open(paths.s2c_cad_splits, "rb") as f:
        splits = pickle.load(f)
    keys = sorted({"_".join(path.split("ShapeNetV2/")[1].split("/")[:2]) for phase in splits.values() for path, _ in phase})
    rng = np.random.default_rng(args.seed)
    out = {}
    for k in tqdm(keys, desc="CADs"):
        pts = sample_uniform(load_cad_mesh(paths.shapenet, *k.split("_", 1)), NUM_POINTS, rng=rng)
        lo, hi = pts.min(0), pts.max(0)
        out[k] = ((pts - (lo + hi) / 2) / (hi - lo).max()).astype(np.float32)
    dst = cad_points_path(paths.cache)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    print(f"{len(out)} CADs -> {dst} ({dst.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
