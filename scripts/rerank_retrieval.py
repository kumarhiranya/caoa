"""CAD retrieval, stage 2 (P11d): re-rank the stage-1 top-k with the Siamese OCAM, choosing CAD and pose together.

For every val object, each of its top-k same-synset candidates (stage-1 embedding, `scripts/train_retrieval.py`) is fed
to the P10b Siamese OCAM (separate trunks) as the CAD input, sampled as that model was trained: 8,192 uniform points in
the CAD's own bounding-box frame and its SEM `cad_bbox` embedding (5,000 points tiled to 8,192). The candidate is then
placed in the predicted box (unit-box points R (s * u) + t, as the Chamfer loss) and scored against the object in the
OCAM test frame:
* `comp`: two-sided CD-L1 between the placed CAD and the completion;
* `inst`: one-sided mean distance from the instance points to the placed CAD (observed points should lie on it);
* `sum`: comp + inst.
Selections reported: `stage1` (top-1 of the embedding), `rerank_<score>`, `oracle_topk` (the candidate closest to the GT
CAD: what a perfect re-ranker could reach), and `gt_cad` (the GT CAD through this same pipeline; reproduces P10b).
Per selection: retrieval metrics (exact hit, CD to the GT CAD) and pose accuracy (legacy metric on the box; S2C criteria
with the chosen CAD's own bounding box and centre). Output: `outputs/<name>/results.json`.

    uv run python scripts/rerank_retrieval.py [--k 10]
"""

import argparse
import dataclasses
import json
from multiprocessing import Pool

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from caoa.data import scan2cad
from caoa.data.legacy_sampling import sample_points
from caoa.data.pairs import load_pairs
from caoa.data.pose_completion import load_pose_completions
from caoa.data.pose_dataset import ObjectPoseDataset, PoseDataConfig, collate
from caoa.data.shapenet import load_cad_mesh, load_shapenet_points, sample_uniform
from caoa.eval import groupings, legacy_report, scan2cad_benchmark
from caoa.eval.pose_eval import to_world
from caoa.models.chamfer import nn_squared_distances
from caoa.models.ocam import OCAMSiamese
from caoa.models.pose_losses import box_points
from caoa.models.sem import SEMClassifier, load_legacy_sem, make_batch
from caoa.paths import REPO_ROOT, load_paths
from caoa.retrieval import search
from caoa.retrieval.data import cad_key, fixed_queries, world_completion
from caoa.retrieval.model import CADRetrieval

SEM_RUN = "S2Csym_emb128_bce_vox1cm_wd1e-03_b64_lr1e-04_bbcls-ptv3-v1m1-0-shapenet_v2_mini_ptv3cls_softmax_symbinary_cad_bce_adam_cosineannealing"
CAD_POINTS, SEM_POINTS, SEM_INPUT = 8192, 5000, 8192
SCORE_POINTS = 2048
SHAPENET = None


def init(root):
    global SHAPENET
    SHAPENET = root


def bbox_frame(pts):
    lo, hi = pts.min(0), pts.max(0)
    return (pts - (lo + hi) / 2) / (hi - lo).max()


def sample_candidate(key: str):
    """As `build_cad_points.py` (OCAM CAD input) and `build_sem_embeddings.py` (SEM input), plus the mesh's own box
    (centre, half extents) for the S2C criteria."""
    rng = np.random.default_rng(int.from_bytes(key.encode()[-8:], "little"))
    mesh = load_cad_mesh(SHAPENET, *key.split("_", 1))
    cad = bbox_frame(sample_uniform(mesh, CAD_POINTS, rng=rng)).astype(np.float32)
    np.random.seed(rng.integers(2**31))
    sem = sample_points(bbox_frame(sample_uniform(mesh, SEM_POINTS, rng=rng)), SEM_INPUT).astype(np.float32)
    lo, hi = mesh.bounds
    return key, cad, sem, ((lo + hi) / 2), ((hi - lo) / 2)


class Seeded(Dataset):
    """The same object sample for every candidate of a query: the RNG is seeded with the query index."""

    def __init__(self, inner, query_of):
        self.inner, self.query_of = inner, query_of

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        np.random.seed(int(self.query_of[i]))
        return self.inner[i]


def load_cfg(run):
    return yaml.safe_load(open(REPO_ROOT / "outputs" / run / "config.yaml"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="retrieval_embed_comp_inst")
    parser.add_argument("--pose", default="siamunshared_world_comp_concat_only_emb1024")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--yaws", type=int, default=8)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="first N val objects only (smoke test)")
    args = parser.parse_args()
    paths = load_paths()
    device = torch.device("cuda")
    rcfg, pcfg = load_cfg(args.retrieval), load_cfg(args.pose)
    out_dir = REPO_ROOT / "outputs" / f"rerank_{args.retrieval}__{args.pose}__k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # stage 1
    db = load_shapenet_points(paths.cache)
    keys = db["keys"].tolist()
    db_index = {k: i for i, k in enumerate(keys)}
    db_points = torch.from_numpy(db["points"]).to(device)
    comp_cfg = pcfg["data"]["completion"]
    completions = load_pose_completions(paths.cache, comp_cfg["name"], comp_cfg["frame"])
    pairs = load_pairs(paths.cache, "val", class_rule=pcfg["data"]["class_rule"])
    if args.limit:
        pairs = pairs[: args.limit]
    targets = torch.tensor([db_index[cad_key(p)] for p in pairs])
    rmodel = CADRetrieval(**rcfg["model"]).to(device)
    rmodel.load_state_dict(torch.load(REPO_ROOT / "outputs" / args.retrieval / "model_final.pth", map_location=device)["model"])
    queries = fixed_queries(pairs, completions, rcfg["data"]["num_points"], rcfg["data"]["inst_fraction"], rcfg["seed"])
    db_emb = search.embed_points(rmodel.cad, db_points, device)
    q_emb = search.embed_queries(rmodel.query, queries, device, yaws=args.yaws)
    mask = search.allowed_mask(db["catids"][targets.numpy()], db["catids"], "same", device)
    top, _ = search.search(q_emb, db_emb, mask, targets.to(device), k=args.k)
    top = top.cpu().numpy()
    del rmodel, db_emb

    # candidates: the top-k of every object, plus the GT CADs (the gt_cad reference)
    cand_keys = sorted({keys[j] for j in top.ravel()} | {cad_key(p) for p in pairs})
    print(f"{len(pairs)} objects, k={args.k}: {len(cand_keys)} unique candidate CADs")
    cad_points, sem_clouds, cad_box = {}, {}, {}
    with Pool(args.workers, initializer=init, initargs=(paths.shapenet,)) as pool:
        for key, cad, sem, center, half in pool.imap_unordered(sample_candidate, cand_keys, chunksize=8):
            cad_points[key], sem_clouds[key], cad_box[key] = cad, sem, (center, half)
    sem_model = SEMClassifier()
    load_legacy_sem(sem_model, paths.legacy_repo / "checkpoints/symmetry_ptv3" / SEM_RUN / f"{SEM_RUN}.pth")
    sem_model = sem_model.to(device).eval()
    sym_embeddings = {}
    with torch.no_grad():
        for s in range(0, len(cand_keys), 32):
            chunk = cand_keys[s : s + 32]
            emb = sem_model(make_batch([sem_clouds[k] for k in chunk], device, grid="train"))["emb"].cpu().numpy()
            sym_embeddings.update(zip(chunk, emb.astype(np.float32)))
    del sem_model, sem_clouds

    # stage 2: one OCAM forward per (object, candidate); column k is the GT CAD
    cols = [[keys[j] for j in row] + [cad_key(p)] for row, p in zip(top, pairs)]
    flat_pairs, query_of = [], []
    for qi, (p, row) in enumerate(zip(pairs, cols)):
        for key in row:
            catid, id_cad = key.split("_", 1)
            flat_pairs.append(dataclasses.replace(p, catid_cad=catid, id_cad=id_cad))
            query_of.append(qi)
    data = {k: tuple(v) if isinstance(v, list) else v for k, v in pcfg["data"].items() if k not in ("class_rule", "completion")}
    ds = ObjectPoseDataset(flat_pairs, False, PoseDataConfig(**data), completions, comp_cfg["mode"], sym_embeddings, cad_points)
    loader = DataLoader(Seeded(ds, query_of), batch_size=64, num_workers=8, collate_fn=collate)
    ocam = OCAMSiamese(embedding=pcfg["model"]["embedding"], ext_emb_size=128, shared=pcfg["model"]["siamese"]["shared"]).to(device)
    ocam.load_state_dict(torch.load(REPO_ROOT / "outputs" / args.pose / "model_final.pth", map_location=device)["model"])
    ocam.eval()
    preds, tgts, nscale, noff = [], [], [], []
    with torch.no_grad():
        for b in loader:
            preds.append(ocam(b["coords"].to(device), b["feats"].to(device), b["batch"].to(device), b["batch_size"],
                              b["cad_coords"].to(device), b["cad_feats"].to(device), b["cad_batch"].to(device), b["sym_emb"].to(device)).cpu())
            tgts.append(b["target"])
            nscale.append(b["norm_scale"])
            noff.append(b["norm_offset"])
    pred, target = torch.cat(preds).double(), torch.cat(tgts).double()
    nscale, noff = torch.cat(nscale), torch.cat(noff)
    K = args.k + 1

    # fit scores, in the OCAM test frame
    rng = np.random.default_rng(0)
    unit = torch.stack([torch.from_numpy(cad_points[p.catid_cad + "_" + p.id_cad][rng.choice(CAD_POINTS, SCORE_POINTS, replace=False)]
                                         / np.maximum(np.ptp(cad_points[p.catid_cad + "_" + p.id_cad], axis=0), 1e-3)) for p in flat_pairs])
    placed = box_points(pred.float(), unit.float())
    obj_comp = torch.stack([torch.from_numpy(sample_points(world_completion(p, completions[(p.scene_id, p.model_index)]), SCORE_POINTS)) for p in pairs])
    obj_inst = torch.stack([torch.from_numpy(sample_points(p.points.astype(np.float64), SCORE_POINTS)) for p in pairs])
    per_q = torch.arange(len(flat_pairs)) // K
    comp_t = obj_comp[per_q] * nscale[:, None, None] - noff[:, None, :]
    inst_t = obj_inst[per_q] * nscale[:, None, None] - noff[:, None, :]
    s_comp, s_inst = [], []
    for s in range(0, len(flat_pairs), 512):
        a = placed[s : s + 512].to(device)
        d1, d2 = nn_squared_distances(a, comp_t[s : s + 512].float().to(device))
        s_comp.append(((d1.sqrt().mean(1) + d2.sqrt().mean(1)) / 2).cpu())
        _, di = nn_squared_distances(a, inst_t[s : s + 512].float().to(device))
        s_inst.append(di.sqrt().mean(1).cpu())
    scores = {"comp": torch.cat(s_comp).view(-1, K)[:, : args.k], "inst": torch.cat(s_inst).view(-1, K)[:, : args.k]}
    scores["sum"] = scores["comp"] + scores["inst"]

    cd_gt, _ = search.chamfer(db_points[torch.from_numpy(top.ravel()).to(device)], db_points[targets.to(device).repeat_interleave(args.k)])
    cd_gt = cd_gt.reshape(-1, args.k)
    choice = {"stage1": np.zeros(len(pairs), dtype=int), "oracle_topk": cd_gt.argmin(1)}
    choice.update({f"rerank_{k}": v.argmin(1).numpy() for k, v in scores.items()})
    choice["gt_cad"] = np.full(len(pairs), args.k)

    # metrics per selection
    annos = scan2cad.load_annotations(paths.s2c_annotations)
    labels = groupings.class_labels(pairs, "scannet_class")
    syms = [p.sym for p in pairs]
    pred_q, tgt_q = pred.view(-1, K, 15), target.view(-1, K, 15)
    ns_q, no_q = nscale.view(-1, K), noff.view(-1, K, 3)
    results = {"k": args.k, "n_objects": len(pairs), "n_candidates": len(cand_keys)}
    for name, c in choice.items():
        qi = np.arange(len(pairs))
        P, T = pred_q[qi, c].numpy(), tgt_q[qi, 0].numpy()
        legacy = legacy_report.eval_pose_report(P, T, syms, labels)["avg"]
        centers, rotations, extents = to_world(P, ns_q[qi, c].numpy(), no_q[qi, c].numpy())
        chosen = [cols[i][c[i]] for i in qi]
        ok = {conv: [] for conv in scan2cad_benchmark.CONVENTIONS}
        for i, p in enumerate(pairs):
            anno = annos[p.scene_id]
            model = anno["aligned_models"][p.model_index]
            catid, id_cad = chosen[i].split("_", 1)
            if name == "gt_cad":  # the annotation's own box, as pose_eval
                center, half = model["center"], model["bbox"]
            else:
                center, half = cad_box[chosen[i]]
            gt = scan2cad_benchmark.ground_truth_pose(anno["trs"], model)
            for conv in ok:
                a = scan2cad_benchmark.alignment_from_box(anno["trs"], catid, id_cad, half, center, centers[i], rotations[i], extents[i], conv)
                ok[conv].append(scan2cad_benchmark.is_correct(a, *gt, model["sym"]))
        cats = np.array([scan2cad_benchmark.category_name(p.catid_cad) for p in pairs])
        crit = {}
        for conv, v in ok.items():
            v = np.array(v)
            crit[conv] = {"instance_mean": float(v.mean()), "class_mean": float(np.mean([v[cats == x].mean() for x in sorted(set(cats))]))}
        cd = cd_gt[qi, np.minimum(c, args.k - 1)] if name != "gt_cad" else np.zeros(len(pairs))
        results[name] = {
            "exact_hit": float(np.mean([chosen[i] == cad_key(p) for i, p in enumerate(pairs)])),
            "cd_l1_to_gt": float(cd.mean()),
            "legacy_total_acc": legacy["total_acc"], "legacy_q_acc": legacy["q_acc"], "legacy_s_acc": legacy["s_acc"],
            "s2c_criteria": crit,
        }
        r = results[name]
        print(f"{name:12s} exact {r['exact_hit']:.3f}  CD-L1 to GT {r['cd_l1_to_gt']:5.1f}  legacy {r['legacy_total_acc']:.4f} "
              f"(rot {r['legacy_q_acc']:.3f} scale {r['legacy_s_acc']:.3f})  S2C −center {crit['minus_center']['instance_mean']:.4f} / "
              f"{crit['minus_center']['class_mean']:.4f}  plain {crit['plain']['instance_mean']:.4f}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
