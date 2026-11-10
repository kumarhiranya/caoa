"""Per-class groupings for the CAOA metric's tables.

* "scannet_class": the pair's class (`Pair.sem_name`).
* "cad_taxonomy_name": the CAD's ShapeNet category name: the category's ScanNet name if it has one, else its
  alphabetically first name.
* "table1": the paper's Table 1 columns: "cad_taxonomy_name" folded to bath / bookshelf / cabinet / chair / sofa /
  table, every other category -> "other".
"""

import json
from functools import cache

from caoa.data.labels import SCANNET20_NAMES
from caoa.paths import RESOURCES

GROUPINGS = ("scannet_class", "cad_taxonomy_name", "table1")
TABLE1_COLUMNS = {"bathtub": "bath", "bookshelf": "bookshelf", "cabinet": "cabinet", "chair": "chair", "sofa": "sofa", "table": "table"}
_SCANNET_NAMES = set(SCANNET20_NAMES) | {"unannotated"}


@cache
def _synset_names() -> dict[str, set[str]]:
    """Synset -> set of its comma-separated names."""
    with open(RESOURCES / "shapenet_taxonomy.json") as f:
        return {s["synsetId"]: set(s["name"].split(",")) for s in json.load(f)}


@cache
def cad_taxonomy_name(catid: str) -> str:
    names = _synset_names()[catid]
    scannet = sorted(n for n in names if n in _SCANNET_NAMES)
    if len(scannet) > 1:
        raise ValueError(f"synset {catid} has several ScanNet names {scannet}")
    return scannet[0] if scannet else sorted(names)[0]


def table1_name(catid: str) -> str:
    return TABLE1_COLUMNS.get(cad_taxonomy_name(catid), "other")


def class_labels(pairs, grouping: str) -> list[str]:
    if grouping == "scannet_class":
        return [p.sem_name for p in pairs]
    if grouping == "cad_taxonomy_name":
        return [cad_taxonomy_name(p.catid_cad) for p in pairs]
    if grouping == "table1":
        return [table1_name(p.catid_cad) for p in pairs]
    raise ValueError(f"unknown grouping {grouping!r}, expected one of {GROUPINGS}")
