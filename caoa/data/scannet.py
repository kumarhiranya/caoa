"""ScanNet v2 scene loading.

Reproduces SoftGroup's `prepare_data_inst.py` (the `*_inst_nostuff_v2.pth`
files the legacy code and the S2C-Completion annotator read) directly from the
raw scene files, so the instance numbering matches `scannet_instance_id`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

from caoa.data.labels import IGNORE, nyu40_to_scannet20, raw_to_nyu40_name


@dataclass
class Scene:
    scene_id: str
    points: np.ndarray  # (N, 3) float64, raw ScanNet coordinates (z up)
    colors: np.ndarray  # (N, 3) uint8
    semantic: np.ndarray  # (N,) int64, ScanNet20 index or IGNORE
    instance: np.ndarray  # (N,) int64, instance index or IGNORE

    def instance_ids(self) -> np.ndarray:
        ids = np.unique(self.instance)
        return ids[ids >= 0]


def scene_files(scans_dir: Path, scene_id: str) -> dict[str, Path]:
    d = Path(scans_dir) / scene_id
    return {
        "mesh": d / f"{scene_id}_vh_clean_2.ply",
        "labels": d / f"{scene_id}_vh_clean_2.labels.ply",
        "segs": d / f"{scene_id}_vh_clean_2.0.010000.segs.json",
        "aggregation": d / f"{scene_id}.aggregation.json",
    }


def available_scenes(scans_dir: Path) -> list[str]:
    scans_dir = Path(scans_dir)
    return sorted(
        d.name for d in scans_dir.glob("scene*_*") if d.is_dir() and all(p.exists() for p in scene_files(scans_dir, d.name).values())
    )


def load_scene(scans_dir: Path, scene_id: str) -> Scene:
    files = scene_files(scans_dir, scene_id)

    vertices = PlyData.read(files["mesh"])["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)
    colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.uint8)

    nyu40 = np.asarray(PlyData.read(files["labels"])["vertex"]["label"], dtype=np.int64)
    semantic = nyu40_to_scannet20()[nyu40]

    with open(files["segs"]) as f:
        seg_indices = np.asarray(json.load(f)["segIndices"])
    with open(files["aggregation"]) as f:
        seg_groups = json.load(f)["segGroups"]

    raw2nyu = raw_to_nyu40_name()
    instance_segments = [g["segments"] for g in seg_groups if raw2nyu[g["label"]] not in ("wall", "floor")]
    # SoftGroup's fix for scene0217_00, whose aggregation lists every group twice.
    if scene_id == "scene0217_00" and instance_segments[0] == instance_segments[len(instance_segments) // 2]:
        instance_segments = instance_segments[: len(instance_segments) // 2]

    instance = np.full(len(points), IGNORE, dtype=np.int64)
    for inst_id, segments in enumerate(instance_segments):
        instance[np.isin(seg_indices, segments)] = inst_id

    return Scene(scene_id, points, colors, semantic, instance)


def dominant_label(semantic: np.ndarray) -> int:
    """Most frequent non-wall/floor label of an instance (legacy `get_dominant_label`)."""
    valid = semantic[semantic > 1]
    return int(np.bincount(valid).argmax()) if len(valid) else IGNORE
