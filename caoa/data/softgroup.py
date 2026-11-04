"""Cached SoftGroup val predictions as the instance source of the full pipeline (P11; user decisions 2026-11-04).

The legacy run's raw outputs (`legacy/data/objects/softgroup_preds/sg_raw/val/<scene>_sg_preds.npz`, 313 scenes): per
scene a dict with `pred_instances` (each: `pred_mask` as SoftGroup's run-length encoding over the scene's vertices,
`label_id`, `conf`) and SoftGroup's own scene arrays. The masks index the vertices of the ScanNet `.ply` in file order
(checked: same count and order, and SoftGroup's `instance_labels` equal our instance ids), so instance points come
from our loader in the Scan2CAD world frame, as the pairs (`caoa.data.pairs.build_scene_pairs`).

`label_id` is SoftGroup's instance class, 1-based over its 18 instance classes (ScanNet20 without wall and floor).
Legacy instead took the dominant per-point semantic prediction inside the mask; both are recorded.

Matching for evaluation (P11b): point-mask IoU with each S2C-C object's ScanNet instance, one-to-one, highest IoU first,
IoU >= `iou_threshold` (0.5). Unmatched objects are reported as capture rate, not failures (user, 2026-10-29).
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


def sg_pred_path(legacy_repo: Path, scene_id: str) -> Path:
    return Path(legacy_repo) / "data/objects/softgroup_preds/sg_raw/val" / f"{scene_id}_sg_preds.npz"


def rle_decode(rle: dict) -> np.ndarray:
    """SoftGroup `rle_encode` inverse: `counts` = "start length start length ...", 1-based starts."""
    s = np.asarray(rle["counts"].split(), dtype=np.int64)
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
    semantic_label: str  # legacy: dominant per-point semantic prediction in the mask
    conf: float
    vertex_ids: np.ndarray  # indices into the scene's vertices
    points: np.ndarray = field(repr=False)  # (N, 3) float32, Scan2CAD world frame


def load_scene_instances(scans_dir: Path, legacy_repo: Path, scene_id: str, anno: dict, conf_threshold: float = 0.0):
    """(instances, scene): predicted instances above `conf_threshold`, all classes, points in the Scan2CAD world frame."""
    preds = np.load(sg_pred_path(legacy_repo, scene_id), allow_pickle=True)["arr_0"].item()
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
        out.append(Instance(scene_id, i, INSTANCE_CLASSES[int(inst["label_id"]) - 1],
                            scannet20_name(dominant_label(preds["semantic_preds"][mask])), float(inst["conf"]),
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
