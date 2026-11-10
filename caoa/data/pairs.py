"""Object-CAD pairs: S2C-Completion's matched (ScanNet instance, Scan2CAD model) pairs.

Pairs are built per scene from the raw ScanNet files and cached as one pickle
per scene. Points are stored in the Scan2CAD world frame (y up, metres); any
normalisation for a particular model or metric happens in the dataset classes.

The cache keeps every matched pair with the instance's own ScanNet class; the class
name a pair is used with (`sem_name`) is set at load time (`class_name`).
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

# Instances of these classes take their CAD's eval class, and are dropped if the CAD is not one of the 8 eval classes.
# RELABELLED gives the 6,663 train / 1,864 val pairs of the CAPCM, OCAM, retrieval and every table; the SEM's training
# set keeps "otherfurniture" instances (SEM_RELABELLED).
RELABELLED = ("unannotated", "otherfurniture")
SEM_RELABELLED = ("unannotated",)


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
    sem_name: str | None = None  # class name the pair is used with; set by `load_pairs`


@dataclass
class SceneStats:
    """Bookkeeping for the pairs a scene contributes or drops."""

    matched: int = 0  # models with scannet_instance_id >= 0
    kept: int = 0
    missing_instance: int = 0  # annotation points at an instance id the scene doesn't have


def class_name(scannet_class: str, catid_cad: str, relabelled=RELABELLED) -> str | None:
    """Class name of a pair, or None if the pair is dropped."""
    if scannet_class in relabelled:
        return synset_to_eval_class().get(catid_cad)
    return scannet_class


def apply_class_names(pairs: list[Pair], relabelled=RELABELLED) -> list[Pair]:
    out = []
    for p in pairs:
        name = class_name(p.scannet_class, p.catid_cad, relabelled)
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


def load_pairs(cache_root: Path, split: str | None = None, relabelled=RELABELLED) -> list[Pair]:
    """Cached pairs of `split` (all splits if None), filtered and named (`class_name`)."""
    pairs = []
    for path in sorted(cache_dir(cache_root).glob("scene*.pkl")):
        with open(path, "rb") as f:
            pairs.extend(p for p in pickle.load(f)["pairs"] if split is None or p.split == split)
    return apply_class_names(pairs, relabelled)
