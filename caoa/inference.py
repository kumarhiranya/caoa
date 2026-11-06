"""The full pipeline on one scan, stage by stage, with the release models (`scripts/make_release.py`).

Used by `scripts/align_scene.py` (all stages, JSON out) and `notebooks/single_scene_pipeline.ipynb` (each stage shown).
The stages are P11's (`scripts/run_p11.py`) without the dataset-wide evaluation:

1. `load_scan`: the `.ply` vertices, into the Scan2CAD world frame (y up) the models work in;
2. `load_instances`: SoftGroup / SphericalMask predictions (`caoa.data.softgroup` format), kept if conf >= `conf` and of a
   class a CAD can be; each becomes a `Pair` whose GT fields are placeholders (identity box: the completion storage frame
   is then the world frame);
3. `complete_instances`: the release CAPCM on the instance points (world frame);
4. `retrieve`: stage 1, the retrieval embedding against the ShapeNetV2 database (class-mapped synsets or all), top k;
5. `align_candidates`: stage 2, SEM on every candidate CAD, the Siamese OCAM aligns the object to each, fit scores;
6. `assemble`: the best fit per instance (`sum` score), its CAD pose in the scan and world frames.
`evaluate_scene` scores the result against the S2C-C annotation of a ScanNet val scene, per object (legacy metric,
Scan2CAD criteria with the chosen CAD, box criteria), for objects matched by an instance (IoU >= 0.5), as P11.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from plyfile import PlyData

from caoa.data import scan2cad
from caoa.data.legacy_sampling import tile_to_min
from caoa.data.pairs import Pair
from caoa.data.pose_completion import MIN_POINTS, complete, to_canonical
from caoa.data.softgroup import (
    EXCLUDED_CLASSES,
    INSTANCE_CLASSES,
    rle_decode,
    scannet20_label,
)
from caoa.eval.pose_eval import to_world
from caoa.eval.scan2cad_benchmark import alignment_from_box
from caoa.geometry import (
    compose,
    decompose,
    matrix_to_quat,
    quat_to_matrix,
    transform_points,
)
from caoa.models.odgnet import UpTrans, load_legacy_checkpoint
from caoa.retrieval import pipeline
from caoa.retrieval import search as rsearch
from caoa.retrieval.data import fixed_queries
from caoa.retrieval.model import CADRetrieval

# Scan2CAD's rotation for ScanNet scans (z up -> y up); the OCAM normalises each instance, so no translation is needed.
SCANNET_TRS = {"translation": [0.0, 0.0, 0.0], "rotation": [-0.7071067811865476, 0.7071067811865475, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class Scan:
    scene_id: str
    trs: dict  # scan -> Scan2CAD world
    points: np.ndarray  # (V, 3) float64, the .ply coordinates
    colors: np.ndarray  # (V, 3) uint8
    world: np.ndarray  # (V, 3) float64, Scan2CAD world frame


@dataclass
class Instances:
    pairs: list  # Pair per kept instance (world-frame points, placeholder GT fields)
    meta: list  # dicts: instance (index in the prediction file), label, conf, n_points
    masks: list  # (V,) bool per kept instance
    dropped: list = field(default_factory=list)  # (label, conf) of the predictions not kept


@dataclass
class Candidates:
    keys: list  # (Q, k) database keys "catid_idcad", stage-1 order
    similarity: np.ndarray  # (Q, k) embedding similarity


@dataclass
class Alignment:
    pred: np.ndarray  # (Q, k, 15) OCAM output per candidate, normalised frame
    nscale: np.ndarray  # (Q, k)
    noff: np.ndarray  # (Q, k, 3)
    scores: dict  # name -> (Q, k) fit scores (lower = better)
    centers: np.ndarray  # (Q, k, 3) predicted box per candidate, world frame
    rotations: np.ndarray  # (Q, k, 3, 3)
    extents: np.ndarray  # (Q, k, 3)
    cad_points: dict  # key -> (8192, 3) CAD points in its own bbox frame (the OCAM's CAD input)
    cad_box: dict  # key -> (mesh box centre, half extents), CAD coordinates
    sym_probs: dict  # key -> SEM (not symmetric, symmetric) probabilities


class Release:
    """The release folder: configs and checkpoints, models loaded on demand."""

    def __init__(self, root, device="cuda"):
        self.root = Path(root)
        self.device = torch.device(device)
        self.capcm_cfg = yaml.safe_load((self.root / "capcm/config.yaml").read_text())
        self.rcfg = yaml.safe_load((self.root / "retrieval/config.yaml").read_text())
        self.pcfg = yaml.safe_load((self.root / "ocam/config.yaml").read_text())
        sym = self.pcfg["model"]["sym_embedding"]
        sym["checkpoint"] = str(self.root / sym["checkpoint"])  # absolute (`sem_checkpoint` joins it to the repo root)
        self.synsets = json.loads((self.root / "class_synsets.json").read_text())
        self.manifest = json.loads((self.root / "manifest.json").read_text())

    def capcm(self):
        model = UpTrans.from_config(self.capcm_cfg["model"]).to(self.device).eval()
        load_legacy_checkpoint(model, self.root / "capcm/ckpt-best.pth")
        return model

    def retrieval(self):
        model = CADRetrieval(**self.rcfg["model"]).to(self.device)
        model.load_state_dict(torch.load(self.root / "retrieval/model_final.pth", map_location=self.device)["model"])
        return model.eval()

    def database(self):
        db = np.load(self.root / "retrieval/db_embeddings.npz")
        return db["keys"].tolist(), db["catids"], torch.from_numpy(db["embeddings"]).to(self.device)

    def ocam(self):
        return pipeline.load_ocam(self.pcfg, self.root / "ocam/model_final.pth", self.device)


def load_scan(ply, trs: dict | None = None) -> Scan:
    v = PlyData.read(ply)["vertex"]
    points = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    colors = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    trs = trs or SCANNET_TRS
    return Scan(Path(ply).name.split("_vh_clean")[0], trs, points, colors, transform_points(points, scan2cad.trs_matrix(trs)))


def load_instances(scan: Scan, preds_file, conf: float = 0.3) -> Instances:
    preds = np.load(preds_file, allow_pickle=True)["arr_0"].item()
    out = Instances([], [], [], [])
    for i, inst in enumerate(preds["pred_instances"]):
        label = INSTANCE_CLASSES[int(inst["label_id"]) - 1]
        mask = rle_decode(inst["pred_mask"])
        if len(mask) != len(scan.points):
            raise ValueError(f"{preds_file}: masks over {len(mask)} vertices, the .ply has {len(scan.points)}")
        if inst["conf"] < conf or label in EXCLUDED_CLASSES or not mask.any():
            out.dropped.append((label, float(inst["conf"])))
            continue
        pts = scan.world[mask].astype(np.float32)
        out.pairs.append(Pair(scan.scene_id, "val", len(out.pairs), -1, scannet20_label(label), "", "", "__SYM_NONE", pts,
                              np.zeros((len(pts), 3), np.uint8), np.zeros(3), np.eye(3), np.ones(3), np.eye(4), np.zeros((0, 3))))
        out.meta.append({"instance": i, "label": label, "conf": float(inst["conf"]), "n_points": int(mask.sum())})
        out.masks.append(mask)
    return out


def complete_instances(release: Release, inst: Instances, seed: int = 0) -> dict:
    """{(scene_id, model_index): completion}, world frame (the placeholder box makes storage = world)."""
    model = release.capcm()
    np.random.seed(seed)
    done = complete(model, [tile_to_min(p.points.astype(np.float64), MIN_POINTS) for p in inst.pairs], release.device)
    del model
    return {(p.scene_id, p.model_index): to_canonical(p, c) for p, c in zip(inst.pairs, done)}


@torch.no_grad()
def retrieve(release: Release, inst: Instances, comps: dict, k: int = 10, search: str = "class", yaws: int = 8) -> Candidates:
    model = release.retrieval()
    keys, catids, db_emb = release.database()
    rd = release.rcfg["data"]
    queries = fixed_queries(inst.pairs, comps, rd["num_points"], rd["inst_fraction"], release.rcfg["seed"])
    allowed = pipeline.class_allowed([p.scannet_class for p in inst.pairs], release.synsets, catids).to(release.device) if search == "class" else None
    top = pipeline.stage1(model, queries, db_emb, allowed, release.device, k, yaws)
    q_emb = rsearch.embed_queries(model.query, queries, release.device, yaws=yaws)  # as in `pipeline.stage1`
    sim = (q_emb[:, None, :] * db_emb[torch.from_numpy(top).to(release.device)]).sum(-1)
    return Candidates([[keys[j] for j in row] for row in top], sim.cpu().numpy())


def align_candidates(release: Release, inst: Instances, comps: dict, cands: Candidates, paths, workers: int = 8) -> Alignment:
    cols = [row + [row[0]] for row in cands.keys]  # `pipeline.align` expects a last (GT) column: repeat the top-1
    cad_points, sym_emb, cad_box, sym_probs = pipeline.prepare_candidates({c for row in cols for c in row}, paths, release.device,
                                                                          workers, release.pcfg["model"]["sym_embedding"], return_probs=True)
    ocam = release.ocam()
    pred, _, nscale, noff, scores = pipeline.align(inst.pairs, comps, cols, release.pcfg, ocam, cad_points, sym_emb, release.device)
    k = len(cands.keys[0])
    pred, nscale, noff = pred[:, :k].numpy(), nscale[:, :k].numpy(), noff[:, :k].numpy()
    Q = len(inst.pairs)
    c, R, e = to_world(pred.reshape(Q * k, 15), nscale.reshape(-1), noff.reshape(-1, 3))
    return Alignment(pred, nscale, noff, {n: v.numpy() for n, v in scores.items()}, c.reshape(Q, k, 3), R.reshape(Q, k, 3, 3),
                     e.reshape(Q, k, 3), cad_points, cad_box, sym_probs)


def candidate_pose(scan: Scan, cands: Candidates, al: Alignment, i: int, b: int) -> dict:
    """Pose of instance `i`'s candidate `b` placed on its predicted box: `cad_to_scan` / `cad_to_world` (4 x 4, the CAD's
    `model_normalized.obj` into the scan / Scan2CAD world), `trs` (the world pose as a Scan2CAD `aligned_models` entry),
    and the box in scan coordinates."""
    W = scan2cad.trs_matrix(scan.trs)
    to_scan, unit = np.linalg.inv(W), np.linalg.norm(W[:3, 0])
    key = cands.keys[i][b]
    catid, id_cad = key.split("_", 1)
    center, half = al.cad_box[key]
    a = alignment_from_box(scan.trs, catid, id_cad, half, center, al.centers[i, b], al.rotations[i, b], al.extents[i, b], "plain")
    cad_to_scan = compose(a.t, quat_to_matrix(a.q), a.s)
    cad_to_world = W @ cad_to_scan
    t_w, R_w, s_w = decompose(cad_to_world)
    return {"catid_cad": catid, "id_cad": id_cad,
            "cad_box": {"center": np.asarray(center).tolist(), "half_extents": np.asarray(half).tolist()},  # CAD coordinates
            "cad_to_scan": cad_to_scan.tolist(), "cad_to_world": cad_to_world.tolist(),
            "trs": {"translation": t_w.tolist(), "rotation": matrix_to_quat(R_w).tolist(), "scale": s_w.tolist()},
            "box": {"center": transform_points(al.centers[i, b][None], to_scan)[0].tolist(),
                    "rotation": (to_scan[:3, :3] @ al.rotations[i, b] * unit).tolist(), "extents": (al.extents[i, b] / unit).tolist()}}


def assemble(scan: Scan, inst: Instances, cands: Candidates, al: Alignment, score: str = "sum") -> list[dict]:
    """Per instance: the best-fitting candidate (lowest `score`) and its pose."""
    objects = []
    for i, m in enumerate(inst.meta):
        b = int(al.scores[score][i].argmin())
        key = cands.keys[i][b]
        objects.append({**m, **candidate_pose(scan, cands, al, i, b), "rank_stage1": b, "fit_score": float(al.scores[score][i, b]),
                        "symmetric_prob": float(al.sym_probs[key][1]), "candidates": cands.keys[i]})
    return objects


def run_scene(release: Release, ply, preds_file, paths, conf: float = 0.3, k: int = 10, search: str = "class",
              trs: dict | None = None, workers: int = 8) -> dict:
    """All stages; returns {scan, instances, completions, candidates, alignment, objects, timing (s)}."""
    timing, t0 = {}, time.perf_counter()

    def lap(name):
        nonlocal t0
        _sync()
        timing[name] = time.perf_counter() - t0
        t0 = time.perf_counter()

    scan = load_scan(ply, trs)
    inst = load_instances(scan, preds_file, conf)
    lap("load")
    out = {"scan": scan, "instances": inst, "timing": timing}
    if not inst.pairs:
        return {**out, "completions": {}, "candidates": None, "alignment": None, "objects": []}
    comps = complete_instances(release, inst)
    lap("completion")
    cands = retrieve(release, inst, comps, k, search)
    lap("retrieval")
    al = align_candidates(release, inst, comps, cands, paths, workers)
    lap("alignment")
    return {**out, "completions": comps, "candidates": cands, "alignment": al, "objects": assemble(scan, inst, cands, al)}


def evaluate_scene(run: dict, scans_dir, s2c_annotations: dict, val_pairs, iou_threshold: float = 0.5) -> list[dict]:
    """Per S2C-C object of the scene: the matched instance (IoU >= threshold, one-to-one) and whether the final pose
    passes the legacy metric, Scan2CAD's criteria (−center, with the chosen CAD) and the box criteria. The scan must be
    loaded with the scene's Scan2CAD `trs`."""
    from caoa.data.scannet import load_scene
    from caoa.data.softgroup import Instance, match_instances
    from caoa.eval import box_criteria, metrics
    from caoa.eval import scan2cad_benchmark as s2c

    scan, inst, al = run["scan"], run["instances"], run["alignment"]
    anno = s2c_annotations[scan.scene_id]
    objects = [p for p in val_pairs if p.scene_id == scan.scene_id]
    scene = load_scene(scans_dir, scan.scene_id)
    gt_masks = [scene.instance == p.instance_id for p in objects]
    preds = [Instance(scan.scene_id, i, m["label"], m["label"], m["conf"], np.flatnonzero(mask), None)
             for i, (m, mask) in enumerate(zip(inst.meta, inst.masks))]
    matches, _ = match_instances(preds, gt_masks, iou_threshold) if objects and preds else ({}, None)
    by_gt = {j: (i, iou) for i, (j, iou) in matches.items()}
    rows = []
    for j, p in enumerate(objects):
        row = {"model_index": p.model_index, "scannet_class": p.scannet_class, "category": s2c.category_name(p.catid_cad),
               "gt_cad": f"{p.catid_cad}_{p.id_cad}", "sym": p.sym, "matched": j in by_gt}
        if j in by_gt:
            i, iou = by_gt[j]
            o = run["objects"][i]
            b = o["rank_stage1"]
            ns, no = al.nscale[i, b], al.noff[i, b]
            target = np.concatenate([p.box_center * ns - no, p.box_rotation.reshape(9), p.box_extents * ns])
            legacy = metrics.alignment_errors(*(x[None] for x in (al.pred[i, b, :3], al.pred[i, b, 3:12].reshape(3, 3), al.pred[i, b, 12:])),
                                              target[None, :3], target[None, 3:12].reshape(1, 3, 3), target[None, 12:], [p.sym],
                                              metrics.MetricConfig.legacy())
            model = anno["aligned_models"][p.model_index]
            a = alignment_from_box(anno["trs"], o["catid_cad"], o["id_cad"], o["cad_box"]["half_extents"], o["cad_box"]["center"],
                                   al.centers[i, b], al.rotations[i, b], al.extents[i, b], "minus_center")
            et, er, es = box_criteria.box_errors((al.centers[i, b], al.rotations[i, b], al.extents[i, b]),
                                                 (p.box_center, p.box_rotation, p.box_extents), p.sym, anno["trs"])
            row.update({"instance": i, "iou": iou, "label": o["label"], "retrieved_cad": f"{o['catid_cad']}_{o['id_cad']}",
                        "exact_cad": f"{o['catid_cad']}_{o['id_cad']}" == row["gt_cad"],
                        "gt_in_topk": row["gt_cad"] in o["candidates"], "legacy": bool(legacy["ok"][0]),
                        "s2c_criteria": bool(s2c.is_correct(a, *s2c.ground_truth_pose(anno["trs"], model), model["sym"])),
                        "box_ok": bool(et <= s2c.THRESHOLD_TRANSLATION and er <= s2c.THRESHOLD_ROTATION and es <= s2c.THRESHOLD_SCALE),
                        "err_t_m": et, "err_r_deg": er, "err_s_pct": es})
        rows.append(row)
    return rows
