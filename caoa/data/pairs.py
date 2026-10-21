"""Object-CAD pairs: S2C-Completion's matched (ScanNet instance, Scan2CAD model) pairs.

Pairs are built per scene from the raw ScanNet files and cached as one pickle
per scene. Points are stored in the Scan2CAD world frame (y up, metres); any
normalisation for a particular model or metric happens in the dataset classes.

The cache keeps every matched pair with the instance's own ScanNet class. Which
pairs a run uses, and the class name each one gets, is decided at load time by
a class rule (`CLASS_RULES`), so the legacy variants share one cache.
"""

import dataclasses
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from caoa.data import scan2cad
from caoa.data.labels import scannet20_name, scannet_split, synset_to_eval_class
from caoa.data.scannet import dominant_label, load_scene
from caoa.geometry import transform_points

CACHE_VERSION = "pairs_v2"

# Class rules (legacy `load_s2c_data_pose_v2`, `custom_datasets_pose.py`):
#   "v2":            S2CCompletionPoseV2. An unannotated instance takes its CAD's eval
#                    class, or is dropped if the CAD is not one of the 8 eval classes.
#   "relpose_cache": S2CCompletionRelPose, which wrote the cache file the reported runs
#                    read (same file name as V2). Same as "v2" for unannotated *and*
#                    otherfurniture instances. Gives the 1,864-sample legacy test set.
#   "all_matched":   keep every matched pair; unannotated instances take their CAD's
#                    eval class when there is one. Not legacy; a candidate fix.
CLASS_RULES = ("v2", "relpose_cache", "all_matched")
_RELABELLED = {"v2": ("unannotated",), "relpose_cache": ("unannotated", "otherfurniture"), "all_matched": ("unannotated",)}


@dataclass
class Pair:
    scene_id: str
    split: str  # "train" / "val" (the official ScanNet split = S2C-C train / test)
    model_index: int  # index into the scene's `aligned_models`
    instance_id: int
    scannet_class: str  # the instance's dominant ScanNet20 class, "unannotated" if none
    catid_cad: str
    id_cad: str
    sym: str
    points: np.ndarray  # (N, 3) float32, all instance points, world frame
    colors: np.ndarray  # (N, 3) uint8
    box_center: np.ndarray  # (3,) OBB target, world frame
    box_rotation: np.ndarray  # (3, 3)
    box_extents: np.ndarray  # (3,) full side lengths
    cad_matrix: np.ndarray  # (4, 4) CAD trs, for the official Scan2CAD target
    keypoints_scan: np.ndarray = field(repr=False)  # (K, 3) world frame
    sem_name: str | None = None  # class name under the class rule; set by `apply_class_rule`


@dataclass
class SceneStats:
    """Bookkeeping for the pairs a scene contributes or drops."""

    matched: int = 0  # models with scannet_instance_id >= 0
    kept: int = 0
    missing_instance: int = 0  # annotation points at an instance id the scene doesn't have


def class_rule_name(scannet_class: str, catid_cad: str, rule: str) -> str | None:
    """Class name of a pair under `rule`, or None if the rule drops it."""
    if rule not in CLASS_RULES:
        raise ValueError(f"unknown class rule {rule!r}, expected one of {CLASS_RULES}")
    if scannet_class in _RELABELLED[rule]:
        name = synset_to_eval_class().get(catid_cad)
        return name if name is not None or rule != "all_matched" else scannet_class
    return scannet_class


def apply_class_rule(pairs: list[Pair], rule: str) -> list[Pair]:
    out = []
    for p in pairs:
        name = class_rule_name(p.scannet_class, p.catid_cad, rule)
        if name is not None:
            out.append(dataclasses.replace(p, sem_name=name))
    return out


def build_scene_pairs(scans_dir: Path, scene_id: str, anno: dict) -> tuple[list[Pair], SceneStats]:
    split = scannet_split(scene_id)
    scene = load_scene(scans_dir, scene_id)
    points = transform_points(scene.points, scan2cad.trs_matrix(anno["trs"])).astype(np.float32)

    stats = SceneStats()
    pairs = []
    for model_index, model in enumerate(anno["aligned_models"]):
        inst_id = model["scannet_instance_id"]
        if inst_id < 0:
            continue
        stats.matched += 1
        mask = scene.instance == inst_id
        if not mask.any():
            stats.missing_instance += 1
            continue

        center, rotation, extents = scan2cad.oriented_box(model)
        pairs.append(
            Pair(
                scene_id=scene_id,
                split=split,
                model_index=model_index,
                instance_id=inst_id,
                scannet_class=scannet20_name(dominant_label(scene.semantic[mask])),
                catid_cad=model["catid_cad"],
                id_cad=model["id_cad"],
                sym=model["sym"],
                points=points[mask],
                colors=scene.colors[mask],
                box_center=center,
                box_rotation=rotation,
                box_extents=extents,
                cad_matrix=scan2cad.trs_matrix(model["trs"]),
                keypoints_scan=np.asarray(model["keypoints_scan"]["position"], dtype=np.float64).reshape(-1, 3),
            )
        )
        stats.kept += 1
    return pairs, stats


def cache_dir(cache_root: Path) -> Path:
    return Path(cache_root) / CACHE_VERSION


def save_scene_pairs(cache_root: Path, scene_id: str, pairs: list[Pair], stats: SceneStats):
    out = cache_dir(cache_root) / f"{scene_id}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"pairs": pairs, "stats": stats}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(out)


def load_pairs(cache_root: Path, split: str | None = None, *, class_rule: str) -> list[Pair]:
    """Cached pairs of `split` (all splits if None), filtered and named by `class_rule`."""
    pairs = []
    for path in sorted(cache_dir(cache_root).glob("scene*.pkl")):
        with open(path, "rb") as f:
            pairs.extend(p for p in pickle.load(f)["pairs"] if split is None or p.split == split)
    return apply_class_rule(pairs, class_rule)
