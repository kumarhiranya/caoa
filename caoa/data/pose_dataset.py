"""Object-only pose samples, as the legacy `S2CCompletionPoseV2` built them.

Train: points in the canonical frame of the target box (axis-aligned,
centred, largest side 1), then a random yaw / isotropic scale / translation;
the target is the transformed box.
Test: points in the Scan2CAD world frame, scaled so the object's largest
extent is 1 and shifted so the point mean sits at (0.1, 0.1, 0.1); the target
is the GT box in that frame. `norm_scale` / `norm_offset` map predictions back
to metres.

With completion (P8), the per-pair completed points (`scripts/build_pose_completions.py`,
in the same canonical frame; `caoa.data.pose_completion`) replace the instance points (`completion_mode="only"`,
legacy `completion_points_only`) or are appended to them ("inst+comp"); at test time
they are mapped back to the world frame first, so the normalisation covers them, as legacy.

With `sym_embeddings` (P10a), each sample also carries the SEM embedding of its GT CAD (`sym_emb`), looked up
by `catid_idcad` as legacy `S2CCompletionPoseV2` did (`sym_map[key][1]`).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from caoa.data.legacy_sampling import sample_points, tile_to_min
from caoa.data.pairs import Pair
from caoa.geometry import yaw_matrix


@dataclass(frozen=True)
class PoseDataConfig:
    num_points: int = 8192
    voxel_size: float = 0.01
    min_points: int = 2048  # instances are tiled up to this when cached (legacy SamplePointsFrom2Array)
    rot_range_deg: tuple[int, int] = (-180, 180)
    scale_range: tuple[float, float] = (0.4, 1.4)
    scale_aspect_range: tuple[float, float] = (0.0, 0.0)
    trans_range: tuple[float, float] = (-0.5, 0.5)
    test_center: float = 0.1


COMPLETION_MODES = ("only", "inst+comp")


def load_sym_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Legacy `data/s2c_symmetry/<model>/embeddings.npy`: {catid_idcad: (class probabilities, embedding)} -> embeddings."""
    return {k: np.asarray(v[1], dtype=np.float32) for k, v in np.load(path, allow_pickle=True).item().items()}


class ObjectPoseDataset(Dataset):
    def __init__(self, pairs: list[Pair], train: bool, cfg: PoseDataConfig = PoseDataConfig(),
                 completions: dict | None = None, completion_mode: str | None = None, sym_embeddings: dict | None = None):
        """`completions`: {(scene_id, model_index): (N, 3) canonical-frame points};
        `sym_embeddings`: {catid_idcad: (E,) embedding} (see `load_sym_embeddings`)."""
        if (completions is None) != (completion_mode is None) or completion_mode not in (None, *COMPLETION_MODES):
            raise ValueError(f"completions and completion_mode {completion_mode!r} go together; modes: {COMPLETION_MODES}")
        self.pairs = pairs
        self.train = train
        self.cfg = cfg
        self.completions = completions
        self.completion_mode = completion_mode
        self.sym_embeddings = sym_embeddings

    def __len__(self):
        return len(self.pairs)

    def _augment(self):
        cfg = self.cfg
        yaw = np.radians(np.random.randint(cfg.rot_range_deg[0], cfg.rot_range_deg[1] + 1))
        scale = np.random.uniform(*cfg.scale_range) + np.random.uniform(*cfg.scale_aspect_range, size=3)
        t = np.random.uniform(*cfg.trans_range, size=3)
        return t, yaw_matrix(yaw), scale

    def __getitem__(self, i: int) -> dict:
        p = self.pairs[i]
        points = tile_to_min(p.points.astype(np.float64), self.cfg.min_points)
        longest = p.box_extents.max()
        canonical = (points - p.box_center) @ p.box_rotation / longest
        if self.completion_mode is not None:
            completed = self.completions[(p.scene_id, p.model_index)].astype(np.float64)
            canonical = completed if self.completion_mode == "only" else np.concatenate([canonical, completed])
            points = (canonical * longest) @ p.box_rotation.T + p.box_center  # back to world (legacy denorm_tmat)

        if self.train:
            t, R, scale = self._augment()
            points = (canonical * scale) @ R.T + t
            target_t, target_R, target_s = t, R, p.box_extents / longest * scale
            norm_scale, norm_offset = 1.0, np.zeros(3)
        else:
            norm_scale = 1.0 / np.ptp(points, axis=0).max()
            points = points * norm_scale
            norm_offset = points.mean(0) - self.cfg.test_center
            points = points - norm_offset
            target_t, target_R, target_s = p.box_center * norm_scale - norm_offset, p.box_rotation, p.box_extents * norm_scale

        points = sample_points(points, self.cfg.num_points).astype(np.float32)
        target = np.concatenate([target_t, target_R.reshape(9), target_s]).astype(np.float32)
        sample = {
            "coords": torch.from_numpy(points / self.cfg.voxel_size),
            "feats": torch.from_numpy(points),
            "target": torch.from_numpy(target),
            "index": i,
            "norm_scale": norm_scale,
            "norm_offset": norm_offset,
        }
        if self.sym_embeddings is not None:
            sample["sym_emb"] = torch.from_numpy(self.sym_embeddings[f"{p.catid_cad}_{p.id_cad}"])
        return sample


def collate(samples: list[dict]) -> dict:
    out = {
        "coords": torch.cat([s["coords"] for s in samples]),
        "feats": torch.cat([s["feats"] for s in samples]),
        "batch": torch.cat([torch.full((len(s["coords"]),), b, dtype=torch.int32) for b, s in enumerate(samples)]),
        "batch_size": len(samples),
        "target": torch.stack([s["target"] for s in samples]),
        "index": torch.tensor([s["index"] for s in samples]),
        "norm_scale": torch.tensor([s["norm_scale"] for s in samples], dtype=torch.float64),
        "norm_offset": torch.tensor(np.stack([s["norm_offset"] for s in samples])),
    }
    if "sym_emb" in samples[0]:
        out["sym_emb"] = torch.stack([s["sym_emb"] for s in samples])
    return out

