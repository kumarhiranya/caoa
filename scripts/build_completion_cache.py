"""Build the S2C-Completion (CAPCM) sample cache from the pair cache, the scenes and the ShapeNet meshes.

    uv run python scripts/build_completion_cache.py --radius 0 --estimate 40   # size estimate only
    uv run python scripts/build_completion_cache.py --radius 0 [--jobs 8]      # build

One cache per context radius, as legacy kept one file per radius
(`scannet_inst_completion_context_spatial{r}cm_{phase}.npy`); each pair's CAD is
sampled once with `resample_mesh(density=2000)` when its cache entry is built.
Points are stored as float32 (user decision 2026-10-21; legacy kept float64), ~1.1 GiB.
"""

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from caoa.data import context, scan2cad, shapenet
from caoa.data.completion_dataset import CompletionSample, completion_cache_dir, keep_for_completion
from caoa.data.pairs import load_pairs
from caoa.data.scannet import load_scene
from caoa.geometry import transform_points
from caoa.paths import load_paths


def build_scene(args):
    scene_id, pairs, radius, paths, seed = args
    np.random.seed(seed)  # resample_mesh uses the global NumPy RNG, as legacy did
    world = None
    if radius > 0:
        scene = load_scene(paths.scannet_scans, scene_id)
        anno = scan2cad.load_annotations(paths.s2c_annotations)[scene_id]
        world = transform_points(scene.points, scan2cad.trs_matrix(anno["trs"]))
    samples = []
    for p in pairs:
        mesh = shapenet.load_cad_mesh(paths.shapenet, p.catid_cad, p.id_cad)
        ctx = (context.context_points("completion", world, scene.instance, p.instance_id, radius)
               if radius > 0 else np.zeros((0, 3)))
        samples.append(CompletionSample(
            scene_id=p.scene_id, model_index=p.model_index, catid_cad=p.catid_cad, id_cad=p.id_cad, sem_name=p.sem_name,
            instance_points=p.points.astype(np.float32), context_points=ctx.astype(np.float32),
            cad_points=shapenet.resample_mesh(mesh, density=shapenet.COMPLETION_CAD_DENSITY).astype(np.float32),
            cad_matrix=p.cad_matrix,
        ))
    return scene_id, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, required=True, help="context radius in metres (legacy: 0 or 1)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--estimate", type=int, default=0, help="only build N pairs and extrapolate the cache size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-scenes", type=int, default=0, help="only the first N train and N val scenes (smoke test)")
    parser.add_argument("--split", choices=("train", "val"), default=None, help="only this split (e.g. val for evaluation)")
    args = parser.parse_args()

    paths = load_paths()
    pairs = [p for p in load_pairs(paths.cache, args.split, class_rule="relpose_cache") if keep_for_completion(p.sem_name)]
    by_scene = {}
    for p in pairs:
        by_scene.setdefault(p.scene_id, []).append(p)
    print(f"{len(pairs)} completion pairs in {len(by_scene)} scenes "
          f"(train {sum(p.split == 'train' for p in pairs)}, val {sum(p.split == 'val' for p in pairs)})")

    if args.limit_scenes:
        from caoa.data.labels import scannet_split

        keep = []
        for split in ("train", "val"):
            keep += [s for s in sorted(by_scene) if scannet_split(s) == split][: args.limit_scenes]
        by_scene = {s: by_scene[s] for s in keep}

    if args.estimate:
        rng = np.random.default_rng(args.seed)
        some = [pairs[i] for i in rng.choice(len(pairs), args.estimate, replace=False)]
        groups = {}
        for p in some:
            groups.setdefault(p.scene_id, []).append(p)
        samples = [s for sid, ps in groups.items() for s in build_scene((sid, ps, args.radius, paths, args.seed))[1]]
        per_pair = np.mean([len(pickle.dumps(s, protocol=pickle.HIGHEST_PROTOCOL)) for s in samples])
        n_cad = np.mean([len(s.cad_points) for s in samples])
        n_ctx = np.mean([len(s.context_points) for s in samples])
        print(f"radius {args.radius}: ~{per_pair / 2**10:.0f} KiB per pair (mean {n_cad:.0f} CAD points, {n_ctx:.0f} context points) "
              f"-> ~{per_pair * len(pairs) / 2**30:.2f} GiB for all {len(pairs)} pairs")
        return

    out = completion_cache_dir(paths.cache, args.radius)
    out.mkdir(parents=True, exist_ok=True)
    todo = [(s, ps, args.radius, paths, args.seed + i) for i, (s, ps) in enumerate(sorted(by_scene.items()))
            if not (out / f"{s}.pkl").exists()]
    with ProcessPoolExecutor(args.jobs) as pool:
        for scene_id, samples in tqdm(pool.map(build_scene, todo), total=len(todo), desc="scenes"):
            tmp = out / f"{scene_id}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.rename(out / f"{scene_id}.pkl")
    size = sum(f.stat().st_size for f in out.glob("*.pkl")) / 2**30
    print(f"cache: {out} ({size:.2f} GiB)")


if __name__ == "__main__":
    main()
