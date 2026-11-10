"""Scene context around an object, for the CAPCM's context variant.

The instance's axis-aligned box is grown by `radius` in each side length (0.5 * radius per side), and the scene's
layout points (instance label < 0: walls, floors and unannotated surfaces) inside it are kept (inclusive bounds).
"""

import numpy as np


def grown_box(inst_points: np.ndarray, radius: float):
    lo, hi = inst_points.min(0), inst_points.max(0)
    center, extent = (lo + hi) / 2, (hi - lo) + radius
    return center - extent / 2, center + extent / 2


def crop(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return points[np.all((points >= lo) & (points <= hi), axis=1)]


def context_points(scene_points: np.ndarray, instance: np.ndarray, inst_id: int, radius: float) -> np.ndarray:
    """Layout points around instance `inst_id` (`instance` = per-point instance labels, < 0 for layout)."""
    if radius <= 0:
        return np.zeros((0, 3), dtype=scene_points.dtype)
    lo, hi = grown_box(scene_points[instance == inst_id], radius)
    return crop(scene_points[instance < 0], lo, hi)
