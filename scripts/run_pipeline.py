"""The full pipeline on predicted instances, evaluated on the S2C-C val objects: SoftGroup (`--instances softgroup`) or
SphericalMask (`--instances sphericalmask`) val predictions.

1. Instances (`caoa.data.instances`): every predicted instance of a kept class (not door / window / picture / curtain /
   shower curtain), points from our ScanNet loader in the Scan2CAD world frame.
2. Matching for evaluation: IoU >= 0.5 with the 1,864 S2C-C val objects, one-to-one, highest first. Capture rate
   (matched / objects) per confidence threshold; poses are evaluated on the objects matched at `--conf` (0.3). Only
   matched objects are evaluated: predictions without an S2C-C object (mostly real objects Scan2CAD did not align) and
   objects without a prediction are not failures. Detection precision / recall are reported separately, against all
   ScanNet GT instances of the predicted class (same matching rule).
3. Completion: the OCAM's CAPCM (its config's `data.completion.checkpoint`; for an OCAM without completion input, the
   retrieval config's), on the instance points in the world frame.
4. Retrieval + alignment (`caoa.retrieval.pipeline`): stage 1 within the synsets of the predicted class (`class_synsets`,
   from the train pairs; "otherfurniture" and unmapped classes search all) and over all synsets; stage 2 re-ranks the
   top k with the OCAM.
5. Rows: GT instances + GT CAD on the same objects (the reference); predicted instances + GT CAD; predicted instances +
   retrieval (class-mapped / all).
`--restrict-to <run folder>` evaluates only the objects matched there too (the common object set of two instance
sources). Output: `outputs/pipeline_<instances>__<pose><tag>/` (`results.json`, `matching.json`, `class_synsets.json`).

    uv run python scripts/run_pipeline.py --instances sphericalmask [--pose ocam_final] [--conf 0.3] [--k 10]
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
from caoa.data.labels import scannet20_name, scannet_split
from caoa.data.sampling import tile_to_min
from caoa.data.pairs import load_pairs
from caoa.data.pose_completion import MIN_POINTS, complete, load_pose_completions, to_canonical
from caoa.data.scannet import dominant_label
from caoa.data.shapenet import load_shapenet_points
from caoa.data.instances import EXCLUDED_CLASSES, INSTANCE_SOURCES, KEPT_SCANNET20, SG_TO_SCANNET20, load_scene_instances, match_instances, pred_dir, sg_pred_path
from caoa.eval.scan2cad_benchmark import category_name
from caoa.models.odgnet import UpTrans, load_checkpoint
from caoa.paths import REPO_ROOT, load_paths
from caoa.retrieval import pipeline, search
from caoa.retrieval.data import cad_key, fixed_queries
from caoa.retrieval.model import CADRetrieval

CONF_THRESHOLDS = (0.0, 0.1, 0.3, 0.5)
CTX = {}


def scene_job(scene_id):
    paths, annos, pairs_by_scene = CTX["paths"], CTX["annos"], CTX["pairs_by_scene"]
    instances, scene = load_scene_instances(paths.scannet_scans, CTX["preds"], scene_id, annos[scene_id])
    instances = [i for i in instances if i.label not in EXCLUDED_CLASSES]
    objects = pairs_by_scene.get(scene_id, [])
    gt_masks = [scene.instance == p.instance_id for p in objects]
    ids = [i for i in np.unique(scene.instance) if i >= 0]
    sn_masks = [scene.instance == i for i in ids]
    sn_cls = [scannet20_name(dominant_label(scene.semantic[m])) for m in sn_masks]
    per_conf, detection = {}, {}
    for t in CONF_THRESHOLDS:
        keep = [k for k, i in enumerate(instances) if i.conf >= t]
        m, _ = match_instances([instances[k] for k in keep], gt_masks) if gt_masks else ({}, None)
        per_conf[t] = {keep[i]: v for i, v in m.items()}
        # detection against every ScanNet instance of the predicted class (IoU of other classes zeroed)
        tp = 0
        for cls in {SG_TO_SCANNET20.get(instances[k].label, instances[k].label) for k in keep}:
            ks = [k for k in keep if SG_TO_SCANNET20.get(instances[k].label, instances[k].label) == cls]
            gs = [g for g, c in zip(sn_masks, sn_cls) if c == cls]
            tp += len(match_instances([instances[k] for k in ks], gs)[0]) if gs else 0
        detection[t] = (len(keep), tp)
    n_sn = sum(c in KEPT_SCANNET20 for c in sn_cls)
    return scene_id, instances, [(p.scene_id, p.model_index) for p in objects], per_conf, detection, n_sn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", choices=INSTANCE_SOURCES, default="softgroup")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--yaws", type=int, default=8)
    parser.add_argument("--retrieval", default="retrieval")
    parser.add_argument("--pose", default="ocam_final")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke test)")
    parser.add_argument("--tag", default="", help="output folder suffix: outputs/pipeline_<instances>__<pose><tag>")
    parser.add_argument("--restrict-to", default=None, help="another pipeline output folder: evaluate only the S2C-C objects matched "
                        "there too (common object set for comparing instance sources)")
    parser.add_argument("--matching-only", action="store_true", help="stop after the capture / detection statistics")
    args = parser.parse_args()
    assert args.conf in CONF_THRESHOLDS
    paths = load_paths()
    device = torch.device("cuda")
    out_dir = REPO_ROOT / "outputs" / f"pipeline_{args.instances}__{args.pose}{args.tag}"
    preds = pred_dir(paths, args.instances)
    out_dir.mkdir(parents=True, exist_ok=True)
    rcfg = yaml.safe_load(open(REPO_ROOT / "outputs" / args.retrieval / "config.yaml"))
    pcfg = yaml.safe_load(open(REPO_ROOT / "outputs" / args.pose / "config.yaml"))

    # 1-2: instances and matching
    annos = scan2cad.load_annotations(paths.s2c_annotations)
    val_pairs = load_pairs(paths.cache, "val")
    pair_of = {(p.scene_id, p.model_index): p for p in val_pairs}
    pairs_by_scene = defaultdict(list)
    for p in val_pairs:
        pairs_by_scene[p.scene_id].append(p)
    scenes = sorted(s for s in annos if scannet_split(s) == "val" and sg_pred_path(preds, s).exists())
    if args.limit:
        scenes = scenes[: args.limit]
    CTX.update(paths=paths, annos=annos, pairs_by_scene=pairs_by_scene, preds=preds)
    with Pool(args.workers) as pool:
        per_scene = pool.map(scene_job, scenes, chunksize=1)
    n_objects = sum(len(r[2]) for r in per_scene)
    n_sn = sum(r[5] for r in per_scene)
    print(f"{len(scenes)} scenes, {sum(len(r[1]) for r in per_scene)} kept-class predictions, {n_objects} S2C-C objects, "
          f"{n_sn} ScanNet instances of the kept classes")

    matching = {}
    for t in CONF_THRESHOLDS:
        cap, tot, fp, pred_n, ious = Counter(), Counter(), Counter(), Counter(), []
        for _, insts, objs, per_conf, _, _ in per_scene:
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
        kept, tp = (sum(r[4][t][i] for r in per_scene) for i in (0, 1))
        prec, rec = tp / max(kept, 1), tp / max(n_sn, 1)
        matching[str(t)]["detection"] = {"predictions": kept, "true_positives": tp, "scannet_instances": n_sn, "precision": prec,
                                         "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9)}
        m = matching[str(t)]
        print(f"conf >= {t}: capture {m['capture']:.4f} ({m['n_matched']}/{m['n_objects']}), predictions {m['n_predictions']}, "
              f"not S2C-C {m['false_positives']}, mean IoU {m['iou_mean']:.3f} | detection vs ScanNet: precision {prec:.3f} "
              f"recall {rec:.3f} F1 {m['detection']['f1']:.3f}")
    matched_at_conf = sorted([list(objs[j]) for _, _, objs, per_conf, _, _ in per_scene for j, _ in per_conf[args.conf].values()])
    matching["matched_objects"] = {"conf": args.conf, "objects": matched_at_conf}  # [scene_id, model_index]
    (out_dir / "matching.json").write_text(json.dumps(matching, indent=1))
    if args.matching_only:
        return

    # matched instances at --conf, as pairs: the instance's points, evaluated against the matched object
    keep = None
    if args.restrict_to:
        other = json.loads((REPO_ROOT / args.restrict_to / "matching.json").read_text())["matched_objects"]
        assert other["conf"] == args.conf, f"{args.restrict_to} matched at conf {other['conf']}"
        keep = {tuple(o) for o in other["objects"]}
    sg_pairs, labels, gt_pairs = [], [], []
    for _, insts, objs, per_conf, _, _ in per_scene:
        for k, (j, _) in sorted(per_conf[args.conf].items()):
            if keep is not None and tuple(objs[j]) not in keep:
                continue
            gt = pair_of[objs[j]]
            inst = insts[k]
            sg_pairs.append(dataclasses.replace(gt, points=inst.points, colors=np.zeros((len(inst.points), 3), np.uint8)))
            labels.append(SG_TO_SCANNET20.get(inst.label, inst.label))
            gt_pairs.append(gt)
    print(f"evaluating {len(sg_pairs)} matched instances (conf >= {args.conf})")

    # 3: completion (world-frame input), stored in the pair's storage frame as the pose caches. The CAPCM: the pose
    # config's, or, for an OCAM without completion input, the retrieval config's (retrieval queries always use completions).
    comp_cfg = pcfg["data"].get("completion") or rcfg["data"]["completion"]
    capcm_dir = REPO_ROOT / comp_cfg["checkpoint"]
    capcm = UpTrans.from_config(yaml.safe_load(open(capcm_dir / "config.yaml"))["model"]).to(device).eval()
    print(f"CAPCM {capcm_dir} (epoch {load_checkpoint(capcm, capcm_dir / 'ckpt-best.pth')['epoch']})")
    np.random.seed(0)
    world = complete(capcm, [tile_to_min(p.points.astype(np.float64), MIN_POINTS) for p in sg_pairs], device)
    sg_comp = {(p.scene_id, p.model_index): to_canonical(p, c) for p, c in zip(sg_pairs, world)}
    del capcm
    gt_comp = load_pose_completions(paths.cache, comp_cfg["name"], comp_cfg.get("frame", "world"))

    # 4: retrieval (class-mapped and all), then alignment
    synsets = pipeline.class_synsets(load_pairs(paths.cache, "train"))
    (out_dir / "class_synsets.json").write_text(json.dumps(synsets, indent=1))
    db = load_shapenet_points(paths.cache)
    keys = db["keys"].tolist()
    db_index = {k: i for i, k in enumerate(keys)}
    db_points = torch.from_numpy(db["points"]).to(device)
    rmodel = CADRetrieval(**rcfg["model"]).to(device)
    rmodel.load_state_dict(torch.load(REPO_ROOT / "outputs" / args.retrieval / "model_final.pth", map_location=device)["model"])
    db_emb = search.embed_points(rmodel.cad, db_points, device)
    queries = fixed_queries(sg_pairs, sg_comp, rcfg["data"]["num_points"], rcfg["data"]["inst_fraction"], rcfg["seed"])
    allowed = pipeline.class_allowed(labels, synsets, db["catids"]).to(device)
    tops = {"class": pipeline.stage1(rmodel, queries, db_emb, allowed, device, args.k, args.yaws),
            "all": pipeline.stage1(rmodel, queries, db_emb, None, device, args.k, args.yaws)}
    del rmodel, db_emb, allowed
    cols = {mode: [[keys[j] for j in row] + [cad_key(p)] for row, p in zip(top, sg_pairs)] for mode, top in tops.items()}
    cand_keys = {key for c in cols.values() for row in c for key in row}
    print(f"{len(cand_keys)} unique candidate CADs")
    cad_points, sym_embeddings, cad_box = pipeline.prepare_candidates(cand_keys, paths, device, pcfg["model"].get("sym_embedding"), args.workers)
    ocam = pipeline.load_ocam(pcfg, REPO_ROOT / "outputs" / args.pose / "model_final.pth", device)

    results = {"conf": args.conf, "k": args.k, "n_matched": len(sg_pairs), "matching": matching, "restrict_to": args.restrict_to}
    gt_cols = [[cad_key(p), cad_key(p)] for p in gt_pairs]  # GT instances, GT CAD only
    out = pipeline.align(gt_pairs, gt_comp, gt_cols, pcfg, ocam, cad_points, sym_embeddings, device)
    results["gt_instances"] = pipeline.evaluate_selections(gt_pairs, gt_cols, *out, annos, cad_box, db_points, db_index, ["gt_cad"])
    for mode in ("class", "all"):
        out = pipeline.align(sg_pairs, sg_comp, cols[mode], pcfg, ocam, cad_points, sym_embeddings, device)
        results[f"{args.instances}_{mode}"] = pipeline.evaluate_selections(sg_pairs, cols[mode], *out, annos, cad_box, db_points, db_index)
    for row, res in results.items():
        if isinstance(res, dict) and row not in ("matching",):
            for name, r in res.items():
                print(pipeline.summary_line(f"{row}/{name}", r))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
