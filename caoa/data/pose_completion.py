"""CAPCM completions of the object–CAD pairs, the OCAM's input (`scripts/build_pose_completions.py`).

* input: the instance points as they are in the Scan2CAD world frame (tiled to >= 2048 points), so no part of the
  GT pose reaches the completion;
* per instance: `sample_points(4096)`, centre on the mean, divide by the max radius;
* the model (in eval mode) runs on batches of 8 instances, no context;
* output: the dense completion (`num_pred` points) mapped back with the same centroid / radius.

Completions are stored (float32, one pickle per scene: {model_index: (num_pred, 3)}) in the canonical frame of the
pair's GT box (`to_canonical`); the pose dataset maps them back to the world exactly, so this is only a storage
convention. Cache: `<name>__world`.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

from caoa.data.sampling import sample_points, tile_to_min

MAX_BS = 8
INPUT_POINTS = 4096
MIN_POINTS = 2048
CACHE_VERSION = "pose_completions_v1"
COMPLETION_FRAMES = ("world",)


def world_points(p) -> np.ndarray:
    """The tiled instance points in the world frame (the model's input)."""
    return tile_to_min(p.points.astype(np.float64), MIN_POINTS)


def to_canonical(p, world_completion: np.ndarray) -> np.ndarray:
    """Storage frame of the cache (inverse of the pose dataset's mapping back to the world)."""
    return ((world_completion - p.box_center) @ p.box_rotation / p.box_extents.max()).astype(np.float32)


def complete_pairs(model, pairs, device, frame: str = "world") -> list[np.ndarray]:
    """Canonical-frame (storage) completions of `pairs`, the model run on the world-frame instance points."""
    if frame == "world":
        return [to_canonical(p, c) for p, c in zip(pairs, complete(model, [world_points(p) for p in pairs], device))]
    raise ValueError(f"completion frame {frame!r}; one of {COMPLETION_FRAMES}")


def pose_completion_dir(cache_root, name: str, frame: str = "world") -> Path:
    if frame not in COMPLETION_FRAMES:
        raise ValueError(f"completion frame {frame!r}; one of {COMPLETION_FRAMES}")
    return Path(cache_root) / CACHE_VERSION / f"{name}__{frame}"


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


def load_pose_completions(cache_root, name: str, frame: str = "world") -> dict:
    """{(scene_id, model_index): canonical-frame completed points}."""
    out = {}
    folder = pose_completion_dir(cache_root, name, frame)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder}: run scripts/build_pose_completions.py --frame {frame}")
    for f in sorted(folder.glob("scene*.pkl")):
        with open(f, "rb") as fh:
            out.update({(f.stem, k): v for k, v in pickle.load(fh).items()})
    return out
