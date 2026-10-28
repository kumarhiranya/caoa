"""Completed points for the pose pairs, as the legacy pose dataset built them (P8).

Reproduces `S2CCompletionPoseV2.infer_completion_odg` on the GT-instance path
(`custom_datasets_pose.py:4608`, called at `:5063` from `load_s2c_data_pose_v2`):
* input: the instance points (tiled to >= 2048, `SamplePointsFrom2Array`) in the canonical
  frame of their **GT box** (`dp_points`, `:4990`). This is the legacy GT-pose leak, reproduced
  as-is (docs/PLAN.md P8);
* per instance: `sample_points(4096)`, centre on the mean, divide by the max radius;
* the model (in eval mode) runs on batches of 8 instances (`max_bs=8`), no context;
* output: `ret[1]` (dense, `num_pred` points) mapped back with the same centroid / radius.
Completions are stored in that canonical frame (float32), one pickle per scene: {model_index: (num_pred, 3)}.

Fix (switch `completion_frame`, user-approved 2026-10-27; docs/results/P8.md): `world` runs the model on the
instance points as they are in the Scan2CAD world frame, so no GT pose reaches it. Only the GT rotation ever
reached it (centre and scale cancel in the normalisation above), but the completion-only OCAM learned to read
that rotation from the completion's point layout. The `world` completions are converted to the same canonical
storage frame afterwards; the pose dataset maps them back to the world exactly, so this is only a storage
convention. Caches: `<name>` (legacy) and `<name>__world`.
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
COMPLETION_FRAMES = ("legacy", "world")


def canonical_points(p) -> np.ndarray:
    """Legacy `dp_points`: tiled instance points in the GT box frame (largest side 1)."""
    pts = tile_to_min(p.points.astype(np.float64), MIN_POINTS)
    return (pts - p.box_center) @ p.box_rotation / p.box_extents.max()


def world_points(p) -> np.ndarray:
    """`world` frame input: the tiled instance points in the world frame (no GT)."""
    return tile_to_min(p.points.astype(np.float64), MIN_POINTS)


def to_canonical(p, world_completion: np.ndarray) -> np.ndarray:
    """Storage frame of the cache (inverse of the pose dataset's mapping back to the world)."""
    return ((world_completion - p.box_center) @ p.box_rotation / p.box_extents.max()).astype(np.float32)


def complete_pairs(model, pairs, device, frame: str = "legacy") -> list[np.ndarray]:
    """Canonical-frame completions of `pairs`, with the model run in the legacy GT box frame or the world frame."""
    if frame == "legacy":
        return complete(model, [canonical_points(p) for p in pairs], device)
    if frame == "world":
        return [to_canonical(p, c) for p, c in zip(pairs, complete(model, [world_points(p) for p in pairs], device))]
    raise ValueError(f"completion frame {frame!r}; one of {COMPLETION_FRAMES}")


def pose_completion_dir(cache_root, name: str, frame: str = "legacy") -> Path:
    if frame not in COMPLETION_FRAMES:
        raise ValueError(f"completion frame {frame!r}; one of {COMPLETION_FRAMES}")
    return Path(cache_root) / CACHE_VERSION / (name if frame == "legacy" else f"{name}__{frame}")


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


def load_pose_completions(cache_root, name: str, frame: str = "legacy") -> dict:
    """{(scene_id, model_index): canonical-frame completed points}."""
    out = {}
    folder = pose_completion_dir(cache_root, name, frame)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder}: run scripts/build_pose_completions.py --frame {frame}")
    for f in sorted(folder.glob("scene*.pkl")):
        with open(f, "rb") as fh:
            out.update({(f.stem, k): v for k, v in pickle.load(fh).items()})
    return out
