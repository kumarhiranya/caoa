"""Run the release (`scripts/make_release.py`) on one scan: instances in, a retrieved ShapeNetV2 CAD and its alignment out.

    uv run python scripts/align_scene.py --ply <scene>_vh_clean_2.ply --instances <scene>_sg_preds.npz --out <scene>.json \
        [--release "<caoa_data>/release_v1"] [--conf 0.3] [--k 10] [--search class|all] [--scan-trs scan.json]

The steps are P11's (`scripts/run_p11.py`), for a single scene and without evaluation (`caoa.inference`, shared with
`notebooks/single_scene_pipeline.ipynb`):
1. instances: SoftGroup / SphericalMask format (`caoa.data.softgroup`: run-length masks over the `.ply` vertices,
   1-based `label_id` over the 18 instance classes, `conf`); kept if conf >= `--conf` and of a class a CAD can be (not
   door, window, picture, curtain, shower curtain);
2. completion: the release CAPCM on the instance points (world frame);
3. retrieval: stage 1 = the embedding (class-mapped synsets, or all with `--search all`), top `--k`; stage 2 = the
   release Siamese OCAM aligns every candidate and the best fit (`sum`: completion + instance Chamfer) is kept;
4. output: per instance, the CAD (`catid_cad`, `id_cad`) and its pose: `cad_to_scan` (4 x 4, the CAD's
   `model_normalized.obj` into the `.ply` coordinates), `cad_to_world` (into the Scan2CAD world frame) and `trs` (that
   world pose as translation, rotation quaternion w x y z and scale, the format of a Scan2CAD `aligned_models` entry),
   plus the predicted oriented box in the scan's coordinates.

The models work in the Scan2CAD world frame (y up). `--scan-trs` gives the scan's Scan2CAD `trs` (a JSON dict
{translation, rotation, scale}, e.g. from the scene's Scan2CAD annotation); by default ScanNet's z-up axes are rotated to
y-up (Scan2CAD's rotation for ScanNet scans; the OCAM normalises each instance, so the translation does not matter).
Needs ShapeNetCore v2 (`shapenet` in `configs/paths*.yaml`) for the candidate meshes.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from caoa.inference import Release, run_scene
from caoa.paths import load_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--release", default=None)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--search", choices=("class", "all"), default="class")
    parser.add_argument("--scan-trs", default=None, help="JSON file with the scan's Scan2CAD trs")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = load_paths()
    release = Release(Path(args.release) if args.release else paths.cache.parent / "release_v1")
    trs = json.loads(Path(args.scan_trs).read_text()) if args.scan_trs else None
    run = run_scene(release, args.ply, args.instances, paths, args.conf, args.k, args.search, trs, args.workers)
    scan = run["scan"]
    print(f"{scan.scene_id}: {len(run['objects'])} objects (conf >= {args.conf}); "
          + ", ".join(f"{k} {v:.1f} s" for k, v in run["timing"].items()))
    out = {"scene": scan.scene_id, "trs": scan.trs, "release": str(release.root), "search": args.search, "k": args.k,
           "timing_s": run["timing"], "objects": run["objects"]}
    Path(args.out).write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
