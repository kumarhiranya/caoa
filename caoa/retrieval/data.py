"""CAD retrieval queries (P11d; user decisions 2026-11-03).

A query is a scanned object: its leak-free CAPCM completion (`completion.frame: world`) plus its instance points, in
the Scan2CAD world frame (y up), half of the points from each. It is centred on its axis-aligned bounding box and
scaled so the longest side is 1, the normalisation the database CADs get in their own frame
(`scripts/build_shapenet_points.py`). The object's yaw is unknown, so training rotates each query by a random yaw
(before the normalisation) and evaluation can average the embedding over yaws (`search.embed_queries`).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from caoa.data.legacy_sampling import sample_points
from caoa.data.pairs import Pair
from caoa.geometry import yaw_matrix

QUERY_POINTS = 2048


def cad_key(p: Pair) -> str:
    return f"{p.catid_cad}_{p.id_cad}"


def world_completion(p: Pair, completion: np.ndarray) -> np.ndarray:
    """Cached completions are stored in the GT box frame (`pose_completion.to_canonical`); this maps them back exactly."""
    return (completion.astype(np.float64) * p.box_extents.max()) @ p.box_rotation.T + p.box_center


def normalize_bbox(points: np.ndarray) -> np.ndarray:
    lo, hi = points.min(0), points.max(0)
    return (points - (lo + hi) / 2) / max(float((hi - lo).max()), 1e-6)


def query_points(p: Pair, completion: np.ndarray, num_points: int = QUERY_POINTS, inst_fraction: float = 0.5,
                 yaw: float = 0.0) -> np.ndarray:
    """(num_points, 3) float32: instance and completion samples, rotated by `yaw` about +y, bbox-normalised."""
    n_inst = int(round(num_points * inst_fraction))
    parts = [sample_points(p.points.astype(np.float64), n_inst), sample_points(world_completion(p, completion), num_points - n_inst)]
    pts = np.concatenate([x for x in parts if len(x)])
    if yaw:
        pts = pts @ yaw_matrix(yaw).T
    return normalize_bbox(pts).astype(np.float32)


class RetrievalQueries(Dataset):
    """Queries for `pairs`; `target` is the index of the pair's GT CAD in the database (`db_index`: key -> index)."""

    def __init__(self, pairs: list[Pair], completions: dict, db_index: dict[str, int], train: bool,
                 num_points: int = QUERY_POINTS, inst_fraction: float = 0.5, yaw: float = 0.0):
        missing = [cad_key(p) for p in pairs if cad_key(p) not in db_index]
        if missing:
            raise KeyError(f"{len(missing)} GT CADs not in the database, e.g. {missing[0]}")
        self.pairs, self.completions, self.db_index = pairs, completions, db_index
        self.train, self.num_points, self.inst_fraction, self.yaw = train, num_points, inst_fraction, yaw

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        p = self.pairs[i]
        yaw = np.random.uniform(-np.pi, np.pi) if self.train else self.yaw
        pts = query_points(p, self.completions[(p.scene_id, p.model_index)], self.num_points, self.inst_fraction, yaw)
        return {"points": torch.from_numpy(pts), "target": self.db_index[cad_key(p)], "index": i}


def collate(samples: list[dict]) -> dict:
    return {
        "points": torch.stack([s["points"] for s in samples]),
        "target": torch.tensor([s["target"] for s in samples]),
        "index": torch.tensor([s["index"] for s in samples]),
    }
