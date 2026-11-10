"""Point (re)sampling helpers shared by the datasets.

* `sample_points`: exactly n points; tile and keep the first n if fewer, a random subset if more;
* `tile_to_min`: tile a small set (n // len + 1) times without truncating; larger sets unchanged.
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
