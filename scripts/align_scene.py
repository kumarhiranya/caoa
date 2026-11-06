"""Run the release (`scripts/make_release.py`) on one scan: instances in, a retrieved ShapeNetV2 CAD and its alignment out.

    uv run python scripts/align_scene.py --ply <scene>_vh_clean_2.ply --instances <scene>_sg_preds.npz --out <scene>.json \
        [--release "<caoa_data>/release_v1"] [--conf 0.3] [--k 10] [--search class|all] [--scan-trs scan.json]

The steps are P11's (`scripts/run_p11.py`), for a single scene and without evaluation:
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
import torch
import yaml
from plyfile import PlyData

from caoa.data import scan2cad
from caoa.data.legacy_sampling import tile_to_min
from caoa.data.pairs import Pair
from caoa.data.pose_completion import MIN_POINTS, complete, to_canonical
from caoa.data.softgroup import EXCLUDED_CLASSES, INSTANCE_CLASSES, rle_decode, scannet20_label
from caoa.eval.pose_eval import to_world
from caoa.eval.scan2cad_benchmark import alignment_from_box
from caoa.geometry import compose, decompose, matrix_to_quat, quat_to_matrix, transform_points
from caoa.models.odgnet import UpTrans, load_legacy_checkpoint
from caoa.paths import load_paths
from caoa.retrieval import pipeline
from caoa.retrieval.data import fixed_queries
from caoa.retrieval.model import CADRetrieval

SCANNET_TRS = {"translation": [0.0, 0.0, 0.0], "rotation": [-0.7071067811865476, 0.7071067811865475, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


def instance_pairs(points_world: np.ndarray, preds_file: Path, conf: float, scene_id: str):
    """Kept instances as `Pair`s (GT fields are placeholders: identity box, so the completion storage frame is the world)."""
    preds = np.load(preds_file, allow_pickle=True)["arr_0"].item()
    pairs, meta = [], []
    for i, inst in enumerate(preds["pred_instances"]):
        label = INSTANCE_CLASSES[int(inst["label_id"]) - 1]
        if inst["conf"] < conf or label in EXCLUDED_CLASSES:
            continue
        mask = rle_decode(inst["pred_mask"])
        if len(mask) != len(points_world):
            raise ValueError(f"{preds_file}: masks over {len(mask)} vertices, the .ply has {len(points_world)}")
        if not mask.any():
            continue
        pts = points_world[mask].astype(np.float32)
        pairs.append(Pair(scene_id, "val", len(pairs), -1, scannet20_label(label), "", "", "__SYM_NONE", pts,
                          np.zeros((len(pts), 3), np.uint8), np.zeros(3), np.eye(3), np.ones(3), np.eye(4), np.zeros((0, 3))))
        meta.append({"instance": i, "label": label, "conf": float(inst["conf"]), "n_points": int(mask.sum())})
    return pairs, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--release", default=None)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--yaws", type=int, default=8)
    parser.add_argument("--search", choices=("class", "all"), default="class")
    parser.add_argument("--scan-trs", default=None, help="JSON file with the scan's Scan2CAD trs")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = load_paths()
    rel = Path(args.release) if args.release else paths.cache.parent / "release_v1"
    device = torch.device("cuda")
    trs = json.loads(Path(args.scan_trs).read_text()) if args.scan_trs else SCANNET_TRS

    v = PlyData.read(args.ply)["vertex"]
    world = transform_points(np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), scan2cad.trs_matrix(trs))
    scene_id = Path(args.ply).name.split("_vh_clean")[0]
    pairs, meta = instance_pairs(world, Path(args.instances), args.conf, scene_id)
    print(f"{scene_id}: {len(pairs)} instances kept (conf >= {args.conf})")
    if not pairs:
        Path(args.out).write_text(json.dumps({"scene": scene_id, "trs": trs, "objects": []}, indent=1))
        return

    # completion
    capcm_cfg = yaml.safe_load((rel / "capcm/config.yaml").read_text())
    capcm = UpTrans.from_config(capcm_cfg["model"]).to(device).eval()
    load_legacy_checkpoint(capcm, rel / "capcm/ckpt-best.pth")
    np.random.seed(0)
    done = complete(capcm, [tile_to_min(p.points.astype(np.float64), MIN_POINTS) for p in pairs], device)
    comps = {(p.scene_id, p.model_index): to_canonical(p, c) for p, c in zip(pairs, done)}
    del capcm

    # retrieval, stage 1
    rcfg = yaml.safe_load((rel / "retrieval/config.yaml").read_text())
    rmodel = CADRetrieval(**rcfg["model"]).to(device)
    rmodel.load_state_dict(torch.load(rel / "retrieval/model_final.pth", map_location=device)["model"])
    db = np.load(rel / "retrieval/db_embeddings.npz")
    keys, db_emb = db["keys"].tolist(), torch.from_numpy(db["embeddings"]).to(device)
    queries = fixed_queries(pairs, comps, rcfg["data"]["num_points"], rcfg["data"]["inst_fraction"], rcfg["seed"])
    synsets = json.loads((rel / "class_synsets.json").read_text())
    allowed = pipeline.class_allowed([p.scannet_class for p in pairs], synsets, db["catids"]).to(device) if args.search == "class" else None
    top = pipeline.stage1(rmodel, queries, db_emb, allowed, device, args.k, args.yaws)
    del rmodel, db_emb
    cols = [[keys[j] for j in row] + [keys[row[0]]] for row in top]  # align() expects a last (GT) column: repeat the top-1

    # stage 2: the OCAM on every candidate, best fit
    pcfg = yaml.safe_load((rel / "ocam/config.yaml").read_text())
    pcfg["model"]["sym_embedding"]["checkpoint"] = str(rel / pcfg["model"]["sym_embedding"]["checkpoint"])  # absolute
    cad_points, sym_embeddings, cad_box = pipeline.prepare_candidates({k for row in cols for k in row}, paths, device, args.workers,
                                                                      pcfg["model"]["sym_embedding"])
    ocam = pipeline.load_ocam(pcfg, rel / "ocam/model_final.pth", device)
    pred, _, nscale, noff, scores = pipeline.align(pairs, comps, cols, pcfg, ocam, cad_points, sym_embeddings, device)
    best = scores["sum"].argmin(1).numpy()
    q = np.arange(len(pairs))
    centers, rotations, extents = to_world(pred[q, best].numpy(), nscale[q, best].numpy(), noff[q, best].numpy())
    to_scan = np.linalg.inv(scan2cad.trs_matrix(trs))

    objects = []
    for i, m in enumerate(meta):
        key = cols[i][best[i]]
        catid, id_cad = key.split("_", 1)
        center, half = cad_box[key]
        a = alignment_from_box(trs, catid, id_cad, half, center, centers[i], rotations[i], extents[i], "plain")  # scan frame
        cad_to_scan = compose(a.t, quat_to_matrix(a.q), a.s)
        cad_to_world = scan2cad.trs_matrix(trs) @ cad_to_scan
        t_w, R_w, s_w = decompose(cad_to_world)
        objects.append({**m, "catid_cad": catid, "id_cad": id_cad, "fit_score": float(scores["sum"][i, best[i]]),
                        "candidates": cols[i][: args.k],
                        "cad_box": {"center": np.asarray(center).tolist(), "half_extents": np.asarray(half).tolist()},  # CAD coordinates
                        "cad_to_scan": cad_to_scan.tolist(), "cad_to_world": cad_to_world.tolist(),
                        "trs": {"translation": t_w.tolist(), "rotation": matrix_to_quat(R_w).tolist(), "scale": s_w.tolist()},
                        "box": {"center": transform_points(centers[i][None], to_scan)[0].tolist(),
                                "rotation": (to_scan[:3, :3] @ rotations[i] / np.linalg.norm(to_scan[:3, 0])).tolist(),
                                "extents": (extents[i] / np.linalg.norm(scan2cad.trs_matrix(trs)[:3, 0])).tolist()}})
    out = {"scene": scene_id, "trs": trs, "release": str(rel), "search": args.search, "k": args.k, "objects": objects}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(objects)} objects -> {args.out}")


if __name__ == "__main__":
    main()
