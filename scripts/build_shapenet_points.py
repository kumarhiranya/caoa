"""CAD retrieval database: 2,048 uniform surface samples for every ShapeNetV2 model (52,491), in the CAD's
own bounding-box frame (centred, longest side 1; as `build_cad_points.py`). One .npz in `<cache>/shapenet_points_v1/`
(~1.3 GB) with `keys`, `catids`, `points`, `bbox_center`, `bbox_extent`, plus `failed.txt`.
Each model's samples are seeded from its key, so the cache does not depend on the worker count.

    uv run python scripts/build_shapenet_points.py --workers 20
"""

import argparse
import hashlib
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from caoa.data.shapenet import (
    SHAPENET_POINTS,
    list_models,
    load_cad_mesh,
    sample_uniform,
    shapenet_points_path,
)
from caoa.paths import load_paths

SHAPENET = None


def sample(key: str):
    try:
        mesh = load_cad_mesh(SHAPENET, *key.split("_", 1))
        if len(mesh.faces) == 0 or mesh.area <= 0:
            return key, None, "empty mesh"
        seed = int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "little")
        pts = sample_uniform(mesh, SHAPENET_POINTS, rng=np.random.default_rng(seed))
        lo, hi = pts.min(0), pts.max(0)
        center, extent = (lo + hi) / 2, hi - lo
        return key, (((pts - center) / extent.max()).astype(np.float32), center, extent), None
    except Exception as e:  # broken OBJ files exist in ShapeNetV2
        return key, None, f"{type(e).__name__}: {e}"


def init(root):
    global SHAPENET
    SHAPENET = root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="first N models only (smoke test; not saved)")
    args = parser.parse_args()
    paths = load_paths()
    keys = list_models(paths.shapenet)
    if args.limit:
        keys = keys[:: max(1, len(keys) // args.limit)][: args.limit]
    print(f"{len(keys)} models")
    done, failed = {}, {}
    with Pool(args.workers, initializer=init, initargs=(paths.shapenet,)) as pool:
        for key, res, err in tqdm(pool.imap_unordered(sample, keys, chunksize=8), total=len(keys), mininterval=30):
            if res is None:
                failed[key] = err
            else:
                done[key] = res
    print(f"sampled {len(done)}, failed {len(failed)}")
    for k, e in sorted(failed.items())[:20]:
        print(f"  {k}: {e}")
    if args.limit:
        return
    order = sorted(done)
    dst = shapenet_points_path(paths.cache)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst,
        keys=np.array(order),
        catids=np.array([k.split("_", 1)[0] for k in order]),
        points=np.stack([done[k][0] for k in order]),
        bbox_center=np.stack([done[k][1] for k in order]).astype(np.float32),
        bbox_extent=np.stack([done[k][2] for k in order]).astype(np.float32),
    )
    (dst.parent / "failed.txt").write_text("".join(f"{k}\t{e}\n" for k, e in sorted(failed.items())))
    print(f"{len(order)} models -> {dst} ({dst.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
