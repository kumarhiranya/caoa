"""The legacy point (re)sampling helpers, shared by the pose and completion datasets.

Three variants existed in the legacy code; each port uses the one on its call path:

* `sample_points` (`S2CCompletionPoseV2.sample_points`, `S2CCompletion.sample_points`):
  exactly n points; tile and keep the first n if fewer, random subset if more;
* `tile_to_min` (`SamplePointsFrom2Array(..., downsample_if_higher=False)`): tile a small
  set (num // len + 1) times without truncating; larger sets unchanged;
* `tile_or_sample` (`SamplePointsFrom1Array(..., resample_if_lower=True)`): exactly n
  points; tile first if fewer, then a random subset; empty stays empty.
"""

import numpy as np


def sample_points(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) < n:
        return np.tile(points, (n // len(points) + 1, 1))[:n]
    if len(points) > n:
        return points[np.random.choice(len(points), n, replace=False)]
    return points


def tile_to_min(points: np.ndarray, n: int) -> np.ndarray:
    return np.tile(points, (n // len(points) + 1, 1)) if len(points) < n else points


def tile_or_sample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        return points
    if len(points) < n:
        points = np.tile(points, (n // len(points) + 1, 1))
    return points[rng.choice(len(points), n, replace=False)]
