"""Retrieval + alignment for a list of objects (P11d stage 2, shared by `scripts/rerank_retrieval.py` and P11).

An object is a `Pair`: its `points` are the observed instance (GT or predicted), its completion is given in the pair's
storage frame (`pose_completion.to_canonical`), and its CAD fields name the GT CAD of the S2C-C object it is evaluated
against (the target box is that object's GT box).

1. `stage1`: the retrieval embedding's top-k CADs per object, within an allowed set (a boolean mask over the database).
2. `prepare_candidates`: every candidate sampled as the P10b Siamese OCAM was trained (8,192 uniform points in the CAD's
   own bounding-box frame) plus its SEM `cad_bbox` embedding and the mesh's own box (centre, half extents).
3. `align`: one OCAM forward per (object, candidate), the candidate placed in the predicted box (unit-box points
   R (s * u) + t) and scored against the object in the OCAM test frame: `comp` = two-sided CD-L1 to the completion,
   `inst` = one-sided mean distance from the instance points, `sum` = both. The last candidate column is the GT CAD.
4. `evaluate_selections`: per selection (`stage1`, `rerank_<score>`, `oracle_topk` = the candidate closest to the GT
   CAD, `gt_cad`): exact hit, CD-L1 to the GT CAD, legacy metric on the box, S2C criteria with the chosen CAD's own box
   (unchanged code) and the CAD-independent `s2c_box`.
"""

import dataclasses
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from caoa.data.legacy_sampling import sample_points
from caoa.data.pose_dataset import ObjectPoseDataset, PoseDataConfig, collate
from caoa.data.shapenet import load_cad_mesh, sample_uniform
from caoa.eval import box_criteria, groupings, legacy_report, scan2cad_benchmark
from caoa.eval.pose_eval import to_world
from caoa.models.chamfer import nn_squared_distances
from caoa.models.ocam import OCAMSiamese
from caoa.models.pose_losses import box_points
from caoa.models.sem import SEMClassifier, load_legacy_sem, make_batch
from caoa.retrieval import search
from caoa.retrieval.data import cad_key, world_completion

SEM_RUN = "S2Csym_emb128_bce_vox1cm_wd1e-03_b64_lr1e-04_bbcls-ptv3-v1m1-0-shapenet_v2_mini_ptv3cls_softmax_symbinary_cad_bce_adam_cosineannealing"
CAD_POINTS, SEM_POINTS, SEM_INPUT = 8192, 5000, 8192
SCORE_POINTS = 2048
SHAPENET = None


def _init(root):
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


@torch.no_grad()
def stage1(rmodel, queries: np.ndarray, db_emb: torch.Tensor, allowed: torch.Tensor | None, device, k: int, yaws: int = 8):
    """Top-k database indices (Q, k) of the embedding, restricted to `allowed` (Q, N) when given."""
    q_emb = search.embed_queries(rmodel.query, queries, device, yaws=yaws)
    sim = q_emb @ db_emb.T
    if allowed is not None:
        sim = sim.masked_fill(~allowed, float("-inf"))
    return sim.topk(k, dim=1).indices.cpu().numpy()


def prepare_candidates(keys, paths, device, workers: int = 20):
    """{key: 8,192 bbox-frame points}, {key: SEM embedding}, {key: (mesh box centre, half extents)}."""
    cad_points, sem_clouds, cad_box = {}, {}, {}
    with Pool(workers, initializer=_init, initargs=(paths.shapenet,)) as pool:
        for key, cad, sem, center, half in pool.imap_unordered(sample_candidate, sorted(keys), chunksize=8):
            cad_points[key], sem_clouds[key], cad_box[key] = cad, sem, (center, half)
    sem_model = SEMClassifier()
    load_legacy_sem(sem_model, paths.legacy_repo / "checkpoints/symmetry_ptv3" / SEM_RUN / f"{SEM_RUN}.pth")
    sem_model = sem_model.to(device).eval()
    order, sym_embeddings = sorted(keys), {}
    with torch.no_grad():
        for s in range(0, len(order), 32):
            chunk = order[s : s + 32]
            emb = sem_model(make_batch([sem_clouds[k] for k in chunk], device, grid="train"))["emb"].cpu().numpy()
            sym_embeddings.update(zip(chunk, emb.astype(np.float32)))
    return cad_points, sym_embeddings, cad_box


def load_ocam(pcfg: dict, checkpoint, device) -> OCAMSiamese:
    ocam = OCAMSiamese(embedding=pcfg["model"]["embedding"], ext_emb_size=128, shared=pcfg["model"]["siamese"]["shared"]).to(device)
    ocam.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    return ocam.eval()


@torch.no_grad()
def align(pairs, completions: dict, cols: list[list[str]], pcfg: dict, ocam, cad_points, sym_embeddings, device):
    """OCAM predictions for every (object, candidate) and the fit scores. `cols[i]`: the object's candidates (k), then
    its GT CAD. Returns pred, target, norm_scale, norm_offset (each (Q, k + 1, ...)) and the scores (Q, k)."""
    K = len(cols[0])
    flat_pairs, query_of = [], []
    for qi, (p, row) in enumerate(zip(pairs, cols)):
        for key in row:
            catid, id_cad = key.split("_", 1)
            flat_pairs.append(dataclasses.replace(p, catid_cad=catid, id_cad=id_cad))
            query_of.append(qi)
    data = {k: tuple(v) if isinstance(v, list) else v for k, v in pcfg["data"].items() if k not in ("class_rule", "completion")}
    ds = ObjectPoseDataset(flat_pairs, False, PoseDataConfig(**data), completions, pcfg["data"]["completion"]["mode"], sym_embeddings, cad_points)
    loader = DataLoader(Seeded(ds, query_of), batch_size=64, num_workers=8, collate_fn=collate)
    preds, tgts, nscale, noff = [], [], [], []
    for b in loader:
        preds.append(ocam(b["coords"].to(device), b["feats"].to(device), b["batch"].to(device), b["batch_size"],
                          b["cad_coords"].to(device), b["cad_feats"].to(device), b["cad_batch"].to(device), b["sym_emb"].to(device)).cpu())
        tgts.append(b["target"])
        nscale.append(b["norm_scale"])
        noff.append(b["norm_offset"])
    pred, target = torch.cat(preds).double(), torch.cat(tgts).double()
    nscale, noff = torch.cat(nscale), torch.cat(noff)

    rng = np.random.default_rng(0)
    unit = torch.stack([torch.from_numpy(cad_points[cad_key(p)][rng.choice(CAD_POINTS, SCORE_POINTS, replace=False)]
                                         / np.maximum(np.ptp(cad_points[cad_key(p)], axis=0), 1e-3)) for p in flat_pairs])
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
    scores = {"comp": torch.cat(s_comp).view(-1, K)[:, : K - 1], "inst": torch.cat(s_inst).view(-1, K)[:, : K - 1]}
    scores["sum"] = scores["comp"] + scores["inst"]
    return pred.view(-1, K, 15), target.view(-1, K, 15), nscale.view(-1, K), noff.view(-1, K, 3), scores


def evaluate_selections(pairs, cols, pred, target, nscale, noff, scores, annos, cad_box, db_points, db_index, selections=None):
    """Metrics per selection (module docstring). `db_points`: (N, P, 3) database points on the GPU, for CD to the GT CAD."""
    k = len(cols[0]) - 1
    device = db_points.device
    cand = torch.tensor([[db_index[key] for key in row[:k]] for row in cols], device=device)
    gt = torch.tensor([db_index[cad_key(p)] for p in pairs], device=device)
    cd_gt, _ = search.chamfer(db_points[cand.reshape(-1)], db_points[gt.repeat_interleave(k)])
    cd_gt = cd_gt.reshape(-1, k)
    choice = {"stage1": np.zeros(len(pairs), dtype=int), "oracle_topk": cd_gt.argmin(1)}
    choice.update({f"rerank_{name}": v.argmin(1).numpy() for name, v in scores.items()})
    choice["gt_cad"] = np.full(len(pairs), k)
    if selections is not None:
        choice = {n: c for n, c in choice.items() if n in selections}

    labels = groupings.class_labels(pairs, "scannet_class")
    syms = [p.sym for p in pairs]
    cats = np.array([scan2cad_benchmark.category_name(p.catid_cad) for p in pairs])
    qi = np.arange(len(pairs))
    results = {}
    for name, c in choice.items():
        P, T = pred[qi, c].numpy(), target[qi, 0].numpy()
        legacy = legacy_report.eval_pose_report(P, T, syms, labels)["avg"]
        table1 = legacy_report.eval_pose_report(P, T, syms, groupings.class_labels(pairs, "table1"))["per_class"]["total"]
        centers, rotations, extents = to_world(P, nscale[qi, c].numpy(), noff[qi, c].numpy())
        chosen = [cols[i][c[i]] for i in qi]
        ok = {conv: [] for conv in scan2cad_benchmark.CONVENTIONS}
        for i, p in enumerate(pairs):
            anno = annos[p.scene_id]
            model = anno["aligned_models"][p.model_index]
            catid, id_cad = chosen[i].split("_", 1)
            center, half = (model["center"], model["bbox"]) if name == "gt_cad" else cad_box[chosen[i]]  # gt_cad: as pose_eval
            gt_pose = scan2cad_benchmark.ground_truth_pose(anno["trs"], model)
            for conv in ok:
                a = scan2cad_benchmark.alignment_from_box(anno["trs"], catid, id_cad, half, center, centers[i], rotations[i], extents[i], conv)
                ok[conv].append(scan2cad_benchmark.is_correct(a, *gt_pose, model["sym"]))
        crit = {}
        for conv, v in ok.items():
            v = np.array(v)
            crit[conv] = {"instance_mean": float(v.mean()), "class_mean": float(np.mean([v[cats == x].mean() for x in sorted(set(cats))]))}
        cd = cd_gt[qi, np.minimum(c, k - 1)] if name != "gt_cad" else np.zeros(len(pairs))
        results[name] = {
            "exact_hit": float(np.mean([chosen[i] == cad_key(p) for i, p in enumerate(pairs)])),
            "category_acc": float(np.mean([chosen[i].split("_", 1)[0] == p.catid_cad for i, p in enumerate(pairs)])),
            "cd_l1_to_gt": float(cd.mean()),
            "legacy_total_acc": legacy["total_acc"], "legacy_q_acc": legacy["q_acc"], "legacy_s_acc": legacy["s_acc"],
            "legacy_table1": {"micro": table1["micro_avg"], "macro": table1["macro_avg"]},  # Paper Table 1's grouping
            "s2c_criteria": crit,
            "s2c_box": box_criteria.evaluate_boxes(pairs, centers, rotations, extents, annos),
        }
    return results


def summary_line(name: str, r: dict) -> str:
    c = r["s2c_criteria"]
    return (f"{name:22s} exact {r['exact_hit']:.3f}  cat {r['category_acc']:.3f}  CD-L1 to GT {r['cd_l1_to_gt']:5.1f}  legacy {r['legacy_total_acc']:.4f} "
            f"(rot {r['legacy_q_acc']:.3f} scale {r['legacy_s_acc']:.3f})  S2C −center {c['minus_center']['instance_mean']:.4f} / "
            f"{c['minus_center']['class_mean']:.4f}  plain {c['plain']['instance_mean']:.4f}  "
            f"box {r['s2c_box']['instance_mean']:.4f} / {r['s2c_box']['class_mean']:.4f}")
