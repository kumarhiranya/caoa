"""Completed points for the pose pairs, as the legacy pose dataset built them (P8).

Reproduces `S2CCompletionPoseV2.infer_completion_odg` on the GT-instance path
(`custom_datasets_pose.py:4608`, called at `:5063` from `load_s2c_data_pose_v2`):
* input: the instance points (tiled to >= 2048, `SamplePointsFrom2Array`) in the canonical
  frame of their **GT box** (`dp_points`, `:4990`). This is the legacy GT-pose leak, reproduced
  as-is (docs/PLAN.md P8);
* per instance: `sample_points(4096)`, centre on the mean, divide by the max radius;
* the model (in eval mode) runs on batches of 8 instances (`max_bs=8`), no context;
* output: `ret[1]` (dense, `num_pred` points) mapped back with the same centroid / radius.
Completions stay in that canonical frame (float32), one pickle per scene: {model_index: (num_pred, 3)}.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

from caoa.data.legacy_sampling import sample_points, tile_to_min

MAX_BS = 8
INPUT_POINTS = 4096
MIN_POINTS = 2048
CACHE_VERSION = "pose_completions_v1"


def canonical_points(p) -> np.ndarray:
    """Legacy `dp_points`: tiled instance points in the GT box frame (largest side 1)."""
    pts = tile_to_min(p.points.astype(np.float64), MIN_POINTS)
    return (pts - p.box_center) @ p.box_rotation / p.box_extents.max()


def pose_completion_dir(cache_root, name: str) -> Path:
    return Path(cache_root) / CACHE_VERSION / name


@torch.no_grad()
def complete(model, point_sets: list[np.ndarray], device) -> list[np.ndarray]:
    out = []
    for start in range(0, len(point_sets), MAX_BS):
        batch, centroids, radii = [], [], []
        for pts in point_sets[start : start + MAX_BS]:
            x = torch.from_numpy(sample_points(pts, INPUT_POINTS).astype(np.float32)).to(device)
            c = x.mean(0)
            x = x - c
            m = torch.sqrt((x ** 2).sum(1)).max()
            batch.append((x / m)[None])
            centroids.append(c)
            radii.append(m)
        dense = model(torch.cat(batch))[1]
        out += [(d * m + c).cpu().numpy().astype(np.float32) for d, c, m in zip(dense, centroids, radii)]
    return out


def load_pose_completions(cache_root, name: str) -> dict:
    """{(scene_id, model_index): canonical-frame completed points}."""
    out = {}
    for f in sorted(pose_completion_dir(cache_root, name).glob("scene*.pkl")):
        with open(f, "rb") as fh:
            out.update({(f.stem, k): v for k, v in pickle.load(fh).items()})
    return out
