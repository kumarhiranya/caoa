"""ScanNet / ShapeNet label maps."""

import csv
import json
from functools import cache

from caoa.paths import RESOURCES

# NYU40 ids of the 20 ScanNet benchmark classes, in SoftGroup's index order.
SCANNET20_NYU40_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
SCANNET20_NAMES = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refridgerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)
IGNORE = -100

# The eight Scan2CAD classes the paper reports per-class results for.
EVAL_CLASSES = ("bathtub", "bed", "bookshelf", "cabinet", "chair", "desk", "sofa", "table")


def scannet20_name(label: int) -> str:
    return "unannotated" if label == IGNORE else SCANNET20_NAMES[label]


@cache
def nyu40_to_scannet20():
    """Lookup table: NYU40 id -> ScanNet20 index, IGNORE for everything else."""
    import numpy as np

    lut = np.full(150, IGNORE, dtype=np.int64)
    lut[list(SCANNET20_NYU40_IDS)] = np.arange(len(SCANNET20_NYU40_IDS))
    return lut


@cache
def raw_to_nyu40_name() -> dict[str, str]:
    """Raw ScanNet category -> NYU40 class name, or "unannotated" outside ScanNet20.

    Only used to drop wall/floor segment groups when numbering instances.
    """
    valid = set(SCANNET20_NAMES)
    with open(RESOURCES / "scannetv2-labels.combined.tsv", newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        return {r["raw_category"]: (r["nyu40class"] if r["nyu40class"] in valid else "unannotated") for r in rows}


@cache
def scannet_split(scene_id: str) -> str | None:
    """'train' / 'val' from the official ScanNet v2 lists (= S2C-Completion train / test)."""
    return _split_lookup().get(scene_id)


@cache
def _split_lookup() -> dict[str, str]:
    lookup = {}
    for split in ("train", "val"):
        for line in (RESOURCES / f"scannetv2_{split}.txt").read_text().split():
            lookup[line] = split
    return lookup


@cache
def shapenet_synset_names() -> dict[str, list[str]]:
    with open(RESOURCES / "shapenet_taxonomy.json") as f:
        return {s["synsetId"]: s["name"].split(",") for s in json.load(f)}


@cache
def synset_to_eval_class() -> dict[str, str]:
    """ShapeNet synset -> first of its names that is an EVAL_CLASSES name (legacy `get_cadid_valid_names`)."""
    out = {}
    for synset, names in shapenet_synset_names().items():
        match = next((n for n in names if n in EVAL_CLASSES), None)
        if match is not None:
            out[synset] = match
    return out
