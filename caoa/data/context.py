"""Scene context around an object: the three legacy crop rules (P4b in docs/PLAN.md).

All three take the instance's axis-aligned box and grow each side length by
`radius` (so `radius` is added to the full extent, 0.5 * radius per side), then
keep scene points inside it (open3d `AxisAlignedBoundingBox.crop`, inclusive).
They differ in which scene points are candidates:

* "pose_v2" (`S2CCompletionPoseV2.extract_context_pcd`): every scene point except the instance's;
* "pose_relpose" (`S2CCompletionRelPose`): every scene point, instance included; when more
  than `context_pointnum` points fall inside, a random `num_points // 2` of them, otherwise
  no context at all (as legacy);
* "completion" (`ODGNet/datasets/S2CCompletionDataset.py`): layout points only
  (instance label < 0, i.e. walls, floors and unannotated stuff).
"""

import numpy as np

from caoa.data.legacy_sampling import tile_or_sample

CONTEXT_RULES = ("pose_v2", "pose_relpose", "completion")


def grown_box(inst_points: np.ndarray, radius: float):
    lo, hi = inst_points.min(0), inst_points.max(0)
    center, extent = (lo + hi) / 2, (hi - lo) + radius
    return center - extent / 2, center + extent / 2


def crop(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return points[np.all((points >= lo) & (points <= hi), axis=1)]


def context_points(rule: str, scene_points: np.ndarray, instance: np.ndarray, inst_id: int, radius: float,
                   num_points: int | None = None, context_pointnum: int | None = None,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Context points for instance `inst_id` (`instance` = per-point instance labels, < 0 for layout)."""
    if radius <= 0:
        return np.zeros((0, 3), dtype=scene_points.dtype)
    lo, hi = grown_box(scene_points[instance == inst_id], radius)
    if rule == "pose_v2":
        return crop(scene_points[instance != inst_id], lo, hi)
    if rule == "completion":
        return crop(scene_points[instance < 0], lo, hi)
    if rule == "pose_relpose":
        if num_points is None or context_pointnum is None:
            raise ValueError("pose_relpose needs num_points and context_pointnum (from the RelPose config)")
        pts = crop(scene_points, lo, hi)
        if len(pts) > context_pointnum:
            return tile_or_sample(pts, num_points // 2, rng or np.random.default_rng())
        return np.zeros((0, 3), dtype=scene_points.dtype)  # legacy leaves the context empty here
    raise ValueError(f"unknown context rule {rule!r}, expected one of {CONTEXT_RULES}")
