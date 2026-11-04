"""P11: the full pipeline on the cached SoftGroup val predictions (user decisions 2026-11-04; docs/PLAN.md § P11).

1. Instances (`caoa.data.softgroup`): every predicted instance of a kept class (not door / window / picture / curtain /
   shower curtain), points from our ScanNet loader in the Scan2CAD world frame.
2. Matching for evaluation: IoU >= 0.5 with the S2C-C val objects (`relpose_cache`, the 1,864), one-to-one, highest
   first. Capture rate (matched / objects) and false positives (unmatched predictions) are reported per confidence
   threshold; pose is evaluated on the matched instances at `--conf`.
3. Completion: the leak-free concat CAPCM in the world frame (as `build_pose_completions.py --frame world`).
4. Retrieval + pose (`caoa.retrieval.pipeline`): stage 1 within the synsets of SoftGroup's predicted class
   (`class_synsets`, from the train pairs; "otherfurniture" and unmapped classes search all) and over all synsets;
   stage 2 with the P10b Siamese OCAM.
5. Rows: GT instances + GT CAD on the same objects; SoftGroup + GT CAD; SoftGroup + retrieval (class-mapped / all).
Output: `outputs/p11_softgroup/` (`results.json`, `matching.json`, `class_synsets.json`).

    uv run python scripts/run_p11.py [--conf 0.3] [--k 10]
"""

import argparse
import dataclasses
import json
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import torch
import yaml

from caoa.data import scan2cad
from caoa.data.labels import scannet_split
from caoa.data.legacy_sampling import tile_to_min
from caoa.data.pairs import load_pairs
from caoa.data.pose_completion import MIN_POINTS, complete, load_pose_completions, to_canonical
from caoa.data.shapenet import load_shapenet_points
from caoa.data.softgroup import EXCLUDED_CLASSES, load_scene_instances, match_instances, sg_pred_path
from caoa.eval.scan2cad_benchmark import category_name
from caoa.models.odgnet import UpTrans, load_legacy_checkpoint
from caoa.paths import REPO_ROOT, load_paths
from caoa.retrieval import pipeline, search
from caoa.retrieval.data import cad_key, fixed_queries
from caoa.retrieval.model import CADRetrieval

CAPCM = "experiments/UpTrans_comp/Concat_SN_S2CC/concat_snaug_s2cc_upTransb_comp"
CONF_THRESHOLDS = (0.0, 0.1, 0.3, 0.5)
SEARCH_ALL = ("otherfurniture",)
SG_TO_SCANNET20 = {"refrigerator": "refridgerator"}  # SoftGroup's spelling vs ours
CTX = {}


def class_synsets(train_pairs, coverage: float = 0.95) -> dict[str, list[str]]:
    """Per ScanNet class, the fewest CAD synsets covering `coverage` of its train pairs (no val information)."""
    counts = defaultdict(Counter)
    for p in train_pairs:
        counts[p.scannet_class][p.catid_cad] += 1
    out = {}
    for cls, c in counts.items():
        total, acc, keep = sum(c.values()), 0, []
        for syn, n in c.most_common():
            keep.append(syn)
            acc += n
            if acc / total >= coverage:
                break
        out[cls] = keep
    return out


def scene_job(scene_id):
    paths, annos, pairs_by_scene = CTX["paths"], CTX["annos"], CTX["pairs_by_scene"]
    instances, scene = load_scene_instances(paths.scannet_scans, paths.legacy_repo, scene_id, annos[scene_id])
    instances = [i for i in instances if i.label not in EXCLUDED_CLASSES]
    objects = pairs_by_scene.get(scene_id, [])
    gt_masks = [scene.instance == p.instance_id for p in objects]
    per_conf = {}
    for t in CONF_THRESHOLDS:
        keep = [k for k, i in enumerate(instances) if i.conf >= t]
        m, _ = match_instances([instances[k] for k in keep], gt_masks) if gt_masks else ({}, None)
        per_conf[t] = {keep[i]: v for i, v in m.items()}
    return scene_id, instances, [(p.scene_id, p.model_index) for p in objects], per_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--yaws", type=int, default=8)
    parser.add_argument("--retrieval", default="retrieval_embed_comp_inst")
    parser.add_argument("--pose", default="siamunshared_world_comp_concat_only_emb1024")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke test)")
    args = parser.parse_args()
    assert args.conf in CONF_THRESHOLDS
    paths = load_paths()
    device = torch.device("cuda")
    out_dir = REPO_ROOT / "outputs" / "p11_softgroup"
    out_dir.mkdir(parents=True, exist_ok=True)
    rcfg = yaml.safe_load(open(REPO_ROOT / "outputs" / args.retrieval / "config.yaml"))
    pcfg = yaml.safe_load(open(REPO_ROOT / "outputs" / args.pose / "config.yaml"))

    # 1-2: instances and matching
    annos = scan2cad.load_annotations(paths.s2c_annotations)
    val_pairs = load_pairs(paths.cache, "val", class_rule=pcfg["data"]["class_rule"])
    pair_of = {(p.scene_id, p.model_index): p for p in val_pairs}
    pairs_by_scene = defaultdict(list)
    for p in val_pairs:
        pairs_by_scene[p.scene_id].append(p)
    scenes = sorted(s for s in annos if scannet_split(s) == "val" and sg_pred_path(paths.legacy_repo, s).exists())
    if args.limit:
        scenes = scenes[: args.limit]
    CTX.update(paths=paths, annos=annos, pairs_by_scene=pairs_by_scene)
    with Pool(args.workers) as pool:
        per_scene = pool.map(scene_job, scenes, chunksize=1)
    n_objects = sum(len(objs) for _, _, objs, _ in per_scene)
    print(f"{len(scenes)} scenes, {sum(len(i) for _, i, _, _ in per_scene)} kept-class predictions, {n_objects} S2C-C objects")

    matching = {}
    for t in CONF_THRESHOLDS:
        cap, tot, fp, pred_n, ious = Counter(), Counter(), Counter(), Counter(), []
        for _, insts, objs, per_conf in per_scene:
            matched_objs = {objs[j] for j, _ in per_conf[t].values()}
            for key in objs:
                cat = category_name(pair_of[key].catid_cad)
                tot[cat] += 1
                cap[cat] += key in matched_objs
            for k, inst in enumerate(insts):
                if inst.conf >= t:
                    pred_n[inst.label] += 1
                    fp[inst.label] += k not in per_conf[t]
            ious += [v for _, v in per_conf[t].values()]
        matching[str(t)] = {
            "capture": sum(cap.values()) / max(sum(tot.values()), 1),
            "capture_per_class": {c: cap[c] / tot[c] for c in sorted(tot)},
            "n_matched": sum(cap.values()), "n_objects": sum(tot.values()),
            "n_predictions": sum(pred_n.values()), "false_positives": sum(fp.values()),
            "false_positives_per_class": dict(fp), "predictions_per_class": dict(pred_n),
            "iou_mean": float(np.mean(ious)) if ious else None,
        }
        m = matching[str(t)]
        print(f"conf >= {t}: capture {m['capture']:.4f} ({m['n_matched']}/{m['n_objects']}), predictions {m['n_predictions']}, "
              f"false positives {m['false_positives']}, mean IoU {m['iou_mean']:.3f}")
    (out_dir / "matching.json").write_text(json.dumps(matching, indent=1))

    # matched instances at --conf, as pairs: the instance's points, evaluated against the matched object
    sg_pairs, labels, gt_pairs = [], [], []
    for _, insts, objs, per_conf in per_scene:
        for k, (j, _) in sorted(per_conf[args.conf].items()):
            gt = pair_of[objs[j]]
            inst = insts[k]
            sg_pairs.append(dataclasses.replace(gt, points=inst.points, colors=np.zeros((len(inst.points), 3), np.uint8)))
            labels.append(SG_TO_SCANNET20.get(inst.label, inst.label))
            gt_pairs.append(gt)
    print(f"evaluating {len(sg_pairs)} matched instances (conf >= {args.conf})")

    # 3: completion, leak-free (world frame), stored in the pair's storage frame as the pose caches
    odgnet = paths.legacy_repo / "ODGNet"
    capcm = UpTrans.from_config(yaml.safe_load(open(odgnet / CAPCM / "config.yaml"))["model"]).to(device).eval()
    load_legacy_checkpoint(capcm, odgnet / CAPCM / "ckpt-best.pth")
    np.random.seed(0)
    world = complete(capcm, [tile_to_min(p.points.astype(np.float64), MIN_POINTS) for p in sg_pairs], device)
    sg_comp = {(p.scene_id, p.model_index): to_canonical(p, c) for p, c in zip(sg_pairs, world)}
    del capcm
    comp_cfg = pcfg["data"]["completion"]
    gt_comp = load_pose_completions(paths.cache, comp_cfg["name"], comp_cfg["frame"])

    # 4: retrieval (class-mapped and all), then alignment
    synsets = class_synsets(load_pairs(paths.cache, "train", class_rule=pcfg["data"]["class_rule"]))
    (out_dir / "class_synsets.json").write_text(json.dumps(synsets, indent=1))
    db = load_shapenet_points(paths.cache)
    keys = db["keys"].tolist()
    db_index = {k: i for i, k in enumerate(keys)}
    db_points = torch.from_numpy(db["points"]).to(device)
    rmodel = CADRetrieval(**rcfg["model"]).to(device)
    rmodel.load_state_dict(torch.load(REPO_ROOT / "outputs" / args.retrieval / "model_final.pth", map_location=device)["model"])
    db_emb = search.embed_points(rmodel.cad, db_points, device)
    queries = fixed_queries(sg_pairs, sg_comp, rcfg["data"]["num_points"], rcfg["data"]["inst_fraction"], rcfg["seed"])
    db_cat = np.asarray(db["catids"])
    allowed = torch.stack([torch.from_numpy(np.isin(db_cat, synsets[c]) if c in synsets and c not in SEARCH_ALL else np.ones(len(db_cat), bool))
                           for c in labels]).to(device)
    tops = {"class": pipeline.stage1(rmodel, queries, db_emb, allowed, device, args.k, args.yaws),
            "all": pipeline.stage1(rmodel, queries, db_emb, None, device, args.k, args.yaws)}
    del rmodel, db_emb, allowed
    cols = {mode: [[keys[j] for j in row] + [cad_key(p)] for row, p in zip(top, sg_pairs)] for mode, top in tops.items()}
    cand_keys = {key for c in cols.values() for row in c for key in row}
    print(f"{len(cand_keys)} unique candidate CADs")
    cad_points, sym_embeddings, cad_box = pipeline.prepare_candidates(cand_keys, paths, device, args.workers)
    ocam = pipeline.load_ocam(pcfg, REPO_ROOT / "outputs" / args.pose / "model_final.pth", device)

    results = {"conf": args.conf, "k": args.k, "n_matched": len(sg_pairs), "matching": matching}
    gt_cols = [[cad_key(p), cad_key(p)] for p in gt_pairs]  # GT instances, GT CAD only
    out = pipeline.align(gt_pairs, gt_comp, gt_cols, pcfg, ocam, cad_points, sym_embeddings, device)
    results["gt_instances"] = pipeline.evaluate_selections(gt_pairs, gt_cols, *out, annos, cad_box, db_points, db_index, ["gt_cad"])
    for mode in ("class", "all"):
        out = pipeline.align(sg_pairs, sg_comp, cols[mode], pcfg, ocam, cad_points, sym_embeddings, device)
        results[f"softgroup_{mode}"] = pipeline.evaluate_selections(sg_pairs, cols[mode], *out, annos, cad_box, db_points, db_index)
    for row, res in results.items():
        if isinstance(res, dict) and row not in ("matching",):
            for name, r in res.items():
                print(pipeline.summary_line(f"{row}/{name}", r))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
