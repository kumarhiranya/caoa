"""Per-class groupings used by the legacy reports and the paper's Table 1.

* "scannet_class": the pair's class under the class rule (legacy GT-instance
  reports, `dataset.names` in `eval_pose_s2c_v2`).
* "cad_taxonomy_name": the CAD's ShapeNet category name as the legacy SoftGroup
  evaluation names it (`SGPredsS2CCompletionPoseVal`, custom_datasets_pose.py:6561):
  the category's ScanNet name if it has one, else its alphabetically first name.
* "table1": the paper's Table 1 columns, decided by the user (2026-10-20):
  "cad_taxonomy_name" folded to bath / bookshelf / cabinet / chair / sofa / table,
  every other category -> "other".
"""

import json
from functools import cache

from caoa.data.labels import SCANNET20_NAMES
from caoa.paths import RESOURCES

GROUPINGS = ("scannet_class", "cad_taxonomy_name", "table1")
TABLE1_COLUMNS = {"bathtub": "bath", "bookshelf": "bookshelf", "cabinet": "cabinet", "chair": "chair", "sofa": "sofa", "table": "table"}
# Legacy SCANNETV2_NAMES (general_utils.py), which `shapenet2scannet_name` matches against.
_LEGACY_SCANNET_NAMES = set(SCANNET20_NAMES) | {"unannotated"}


@cache
def _synset_names() -> dict[str, set[str]]:
    """Legacy SHAPENET_SYNSET_NAME: synset -> set of its comma-separated names."""
    with open(RESOURCES / "shapenet_taxonomy.json") as f:
        return {s["synsetId"]: set(s["name"].split(",")) for s in json.load(f)}


@cache
def cad_taxonomy_name(catid: str) -> str:
    names = _synset_names()[catid]
    scannet = sorted(n for n in names if n in _LEGACY_SCANNET_NAMES)
    # The legacy code returns the first match in set order; with one match that is deterministic.
    if len(scannet) > 1:
        raise ValueError(f"synset {catid} has several ScanNet names {scannet}; legacy naming would be hash-order dependent")
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
