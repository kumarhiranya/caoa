"""Build the object-CAD pair cache for every extracted ScanNet scene that Scan2CAD annotates.

    uv run python scripts/build_pairs.py [--jobs 8] [--limit N] [--overwrite]

Prints a summary, the pair counts (OCAM and SEM sets), and a sanity check:
the share of each instance's points that lie inside its annotated box (padded
by 10%). A wrong instance numbering shows up as a low share.
"""

import argparse
import collections
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from caoa.data import pairs as pairs_mod
from caoa.data import scan2cad
from caoa.data.labels import scannet_split
from caoa.data.scannet import available_scenes
from caoa.paths import load_paths


def inside_box_fraction(pair, pad=1.1) -> float:
    local = (pair.points - pair.box_center) @ pair.box_rotation
    return float(np.all(np.abs(local) <= pad * pair.box_extents / 2, axis=1).mean())


def build_one(args):
    scans_dir, cache_root, scene_id, anno = args
    pairs, stats = pairs_mod.build_scene_pairs(scans_dir, scene_id, anno)
    pairs_mod.save_scene_pairs(cache_root, scene_id, pairs, stats)
    return scene_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N scenes (smoke test)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    annos = scan2cad.load_annotations(paths.s2c_annotations)
    scenes = [s for s in available_scenes(paths.scannet_scans) if s in annos and scannet_split(s) is not None]
    print(f"{len(annos)} annotated scenes, {len(scenes)} of them extracted locally")
    if args.limit:
        scenes = scenes[: args.limit]

    out_dir = pairs_mod.cache_dir(paths.cache)
    todo = [s for s in scenes if args.overwrite or not (out_dir / f"{s}.pkl").exists()]
    jobs = [(paths.scannet_scans, paths.cache, s, annos[s]) for s in todo]
    with ProcessPoolExecutor(args.jobs) as pool:
        list(tqdm(pool.map(build_one, jobs), total=len(jobs), desc="scenes"))

    # Summarise from the cache so reruns report everything, not just new scenes.
    totals = collections.Counter()
    all_pairs = []
    for s in scenes:
        with open(out_dir / f"{s}.pkl", "rb") as f:
            blob = pickle.load(f)
        st = blob["stats"]
        totals.update(matched=st.matched, kept=st.kept, missing_instance=st.missing_instance)
        all_pairs += blob["pairs"]

    n_scenes = collections.Counter(scannet_split(s) for s in scenes)
    inside = np.array([inside_box_fraction(p) for p in all_pairs])
    n_points = np.array([len(p.points) for p in all_pairs])
    size_mb = sum(p.stat().st_size for p in out_dir.glob("*.pkl")) / 2**20

    print(f"\nscenes: {len(scenes)} {dict(n_scenes)}   pairs matched: {totals['matched']}   cached: {totals['kept']}")
    print(f"dropped: {totals['missing_instance']} instance id not in scene")
    for name, relabelled in (("OCAM / CAPCM", pairs_mod.RELABELLED), ("SEM", pairs_mod.SEM_RELABELLED)):
        kept = pairs_mod.apply_class_names(all_pairs, relabelled)
        print(f"{name:<14} train {sum(p.split == 'train' for p in kept):>5}  val {sum(p.split == 'val' for p in kept):>5}")
    print("instance classes:", dict(collections.Counter(p.scannet_class for p in all_pairs).most_common()))
    print(f"points per instance: median {np.median(n_points):.0f}, min {n_points.min()}, max {n_points.max()}")
    print(
        f"share of instance points inside the padded box: median {np.median(inside):.3f}, "
        f"<0.5 for {(inside < 0.5).sum()} pairs, <0.9 for {(inside < 0.9).sum()} pairs"
    )
    print(f"cache: {out_dir} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
