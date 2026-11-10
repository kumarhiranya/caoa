"""Predicted instances, the input of the full pipeline: SoftGroup or SphericalMask predictions on the ScanNet val scans.

One `<scene>_sg_preds.npz` per scene (SoftGroup's output format; `scripts/external/sphericalmask_infer.py` writes
SphericalMask's in the same format): a dict with `pred_instances` (each: `pred_mask` as SoftGroup's run-length encoding
over the scene's vertices, `label_id`, `conf`). The masks index the vertices of the ScanNet `.ply` in file order, so the
instance points come from our scene loader, in the Scan2CAD world frame like the object–CAD pairs
(`caoa.data.pairs.build_scene_pairs`).

`label_id` is the instance class, 1-based over the 18 ScanNet instance classes (ScanNet20 without wall and floor). The
dominant per-point semantic label inside the mask is recorded too.

Matching for evaluation: point-mask IoU with each S2C-C object's ScanNet instance, one-to-one, highest IoU first,
IoU >= `iou_threshold` (0.5). Only matched objects are evaluated; unmatched objects and predictions are not failures
(the capture rate reports how many objects were matched).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from caoa.data import scan2cad
from caoa.data.labels import scannet20_name
from caoa.data.scannet import dominant_label, load_scene
from caoa.geometry import transform_points

INSTANCE_CLASSES = (
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
)
# Classes kept for alignment: everything a Scan2CAD CAD can be, not the architectural / flat ones.
EXCLUDED_CLASSES = ("door", "window", "picture", "curtain", "shower curtain")


INSTANCE_SOURCES = ("softgroup", "sphericalmask")
SG_TO_SCANNET20 = {"refrigerator": "refridgerator"}  # the instance classes' spelling vs the ScanNet20 labels'
KEPT_SCANNET20 = {SG_TO_SCANNET20.get(c, c) for c in INSTANCE_CLASSES if c not in EXCLUDED_CLASSES}


def scannet20_label(instance_label: str) -> str:
    """ScanNet20 name (our spelling) of an instance class."""
    return SG_TO_SCANNET20.get(instance_label, instance_label)


def pred_dir(paths, source: str = "softgroup") -> Path:
    """Folder of the per-scene prediction files of `source` (`paths.softgroup_preds` / `paths.sphericalmask_preds`)."""
    if source in INSTANCE_SOURCES:
        return Path(getattr(paths, f"{source}_preds"))
    raise ValueError(f"instance source {source!r}; one of {INSTANCE_SOURCES}")


def sg_pred_path(preds: Path, scene_id: str) -> Path:
    return Path(preds) / f"{scene_id}_sg_preds.npz"


def rle_decode(rle: dict) -> np.ndarray:
    """SoftGroup `rle_encode` inverse: `counts` = "start length start length ...", 1-based starts. SphericalMask's
    `rle_encode_gpu_batch` stores the same runs as an integer array instead of a string."""
    counts = rle["counts"]
    s = np.asarray(counts.split() if isinstance(counts, str) else counts, dtype=np.int64)
    starts, lengths = s[0::2] - 1, s[1::2]
    mask = np.zeros(rle["length"], dtype=bool)
    for lo, n in zip(starts, lengths):
        mask[lo : lo + n] = True
    return mask


@dataclass
class Instance:
    scene_id: str
    index: int  # position in the scene's `pred_instances`
    label: str  # SoftGroup's instance class (`label_id`)
    semantic_label: str  # dominant per-point semantic prediction in the mask
    conf: float
    vertex_ids: np.ndarray  # indices into the scene's vertices
    points: np.ndarray = field(repr=False)  # (N, 3) float32, Scan2CAD world frame


def load_scene_instances(scans_dir: Path, preds_dir: Path, scene_id: str, anno: dict, conf_threshold: float = 0.0):
    """(instances, scene): predicted instances above `conf_threshold`, all classes, points in the Scan2CAD world frame.
    `preds_dir`: see `pred_dir`."""
    preds = np.load(sg_pred_path(preds_dir, scene_id), allow_pickle=True)["arr_0"].item()
    scene = load_scene(scans_dir, scene_id)
    if len(scene.points) != len(preds["semantic_preds"]):
        raise ValueError(f"{scene_id}: {len(scene.points)} vertices vs {len(preds['semantic_preds'])} in the SoftGroup file")
    world = transform_points(scene.points, scan2cad.trs_matrix(anno["trs"])).astype(np.float32)
    out = []
    for i, inst in enumerate(preds["pred_instances"]):
        if inst["conf"] < conf_threshold:
            continue
        mask = rle_decode(inst["pred_mask"])
        if not mask.any():
            continue
        sem = np.asarray(preds["semantic_preds"])
        sem = sem.argmax(1) if sem.ndim == 2 else sem  # SphericalMask stores per-point class scores
        out.append(Instance(scene_id, i, INSTANCE_CLASSES[int(inst["label_id"]) - 1],
                            scannet20_name(dominant_label(sem[mask])), float(inst["conf"]),
                            np.flatnonzero(mask), world[mask]))
    return out, scene


def match_instances(instances: list[Instance], gt_masks: list[np.ndarray], iou_threshold: float = 0.5):
    """One-to-one, highest IoU first. Returns {instance position: (gt position, IoU)} and the full IoU matrix."""
    n_vertices = len(gt_masks[0]) if gt_masks else 0
    iou = np.zeros((len(instances), len(gt_masks)))
    for i, inst in enumerate(instances):
        pred = np.zeros(n_vertices, dtype=bool)
        pred[inst.vertex_ids] = True
        for j, gt in enumerate(gt_masks):
            inter = np.count_nonzero(pred & gt)
            if inter:
                iou[i, j] = inter / np.count_nonzero(pred | gt)
    matches, used_pred, used_gt = {}, set(), set()
    for flat in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] < iou_threshold:
            break
        if i in used_pred or j in used_gt:
            continue
        matches[int(i)] = (int(j), float(iou[i, j]))
        used_pred.add(i)
        used_gt.add(j)
    return matches, iou
