"""S2C-Completion for CAPCM: port of `ODGNet/datasets/S2CCompletionDataset.py` (current file).

Samples (`build_samples`, legacy `load_s2c_completion_context`):
* matched pairs under the legacy relabel ('unannotated', 'otherfurniture' -> the
  CAD's eval class), then only the 8 VALID_CLASSES are kept;
* GT = the CAD surface sampled once per pair with `resample_mesh(density=2000)`, in
  CAD coordinates; instance and context are mapped into the CAD frame with
  inv(CAD trs) (train) or kept in the world frame and normalised (val, as legacy);
* context = layout points (instance label < 0) in the instance's AABB grown by
  `context_radius` (context rule "completion").

Items (`__getitem__`): the legacy 6-tuple `(catid, id_cad, partial, gt, sem_emb, idx)`
that `ODGNet/tools/runner.py` unpacks.

Fixes to the legacy file, reviewed with the user (docs/PLAN.md P5a):
1. the cache entries are unpacked as the 9 fields they have (legacy line 466 unpacked 10);
3. `sem_emb` comes from `semantic_embeddings.npy` (key `catid_idcad`), as in SNAugDataset;
   the file covers only the 7 SN-Indoor categories, so the 91 pairs whose CAD is from
   another category get a zero vector (user decision 2026-10-21; UpTrans ignores sem_emb);
4. with context, the input mirrors the pose-time inference (custom_datasets_pose.py:4661):
   instance + context with a flag, the same yaw for instance / context / target,
   normalised by the combined centroid and max radius, then sampled. The flag values
   are a parameter, decided empirically in P6.
"""

import random
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from caoa.data.legacy_sampling import sample_points
from caoa.geometry import transform_points, yaw_matrix

VALID_CLASSES = ("bathtub", "bed", "bookshelf", "cabinet", "chair", "desk", "sofa", "table")
SEM_EMB_DIM = 512


@dataclass
class CompletionSample:
    scene_id: str
    model_index: int
    catid_cad: str
    id_cad: str
    sem_name: str
    instance_points: np.ndarray  # world frame (stored float32; user decision 2026-10-21)
    context_points: np.ndarray  # world frame, (0, 3) when none
    cad_points: np.ndarray  # CAD frame (`resample_mesh`)
    cad_matrix: np.ndarray  # CAD trs (legacy calc_Mcad), CAD -> world


def legacy_val_frame(inst, cad, ctx):
    """Legacy val/test normalisation: centre on the instance mean, scale by a scalar range, re-centre."""
    inst, cad, ctx = inst.copy(), cad.copy(), ctx.copy()
    mean = inst.mean(0)
    inst, cad, ctx = inst - mean, cad - mean, ctx - mean
    p_min = inst.min()
    p_range = (inst.max() - inst.min()) * 2
    inst, cad, ctx = (inst - p_min) / p_range, (cad - p_min) / p_range, (ctx - p_min) / p_range
    mean = inst.mean(0)
    return inst - mean, cad - mean, ctx - mean


def pc_norm(pc, pc2):
    """Legacy `pc_norm`: centre on pc's centroid, divide by pc's max radius (applied to both)."""
    centroid = np.mean(pc, axis=0)
    pc, pc2 = pc - centroid, pc2 - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / m, pc2 / m


class S2CCompletionDataset(Dataset):
    def __init__(self, samples: list[CompletionSample], phase: str, sem_embeddings: dict, num_points: int = 4096,
                 rotate_z: bool = True, context_radius: float = 0.0, context_flag: tuple[float, float] = (1.0, -1.0)):
        self.phase = "val" if phase in ("val", "test") else "train"
        self.num_points = num_points
        self.rotate_z = rotate_z
        self.context_radius = context_radius
        self.context_flag = context_flag
        self.samples = samples
        self.sem_embeddings = sem_embeddings
        self.data, self.context_data, self.targets = [], [], []
        for s in samples:
            s_inst, s_ctx, s_cad = (np.asarray(a, dtype=np.float64) for a in (s.instance_points, s.context_points, s.cad_points))
            if self.phase == "train":
                to_cad = np.linalg.inv(s.cad_matrix)
                inst, ctx, cad = transform_points(s_inst, to_cad), transform_points(s_ctx, to_cad), s_cad
            else:
                inst, cad, ctx = legacy_val_frame(s_inst, transform_points(s_cad, s.cad_matrix), s_ctx)
            self.data.append(inst)
            self.context_data.append(ctx)
            self.targets.append(cad)

    def __len__(self):
        return len(self.data)

    def _random_yaw(self):
        # legacy random_rotate_around_center_z: np.random.random_integers(0, 360) degrees about y, about the origin
        return yaw_matrix(np.radians(np.random.randint(0, 361)))

    def __getitem__(self, idx):
        s = self.samples[idx]
        xyz, target, ctx = self.data[idx], self.targets[idx], self.context_data[idx]
        sem_emb = self.sem_embeddings.get(f"{s.catid_cad}_{s.id_cad}")
        sem_emb = np.zeros(SEM_EMB_DIM, np.float32) if sem_emb is None else sem_emb.astype(np.float32)
        R = self._random_yaw() if self.rotate_z and self.phase == "train" else None

        if self.context_radius > 0:  # fix 4
            if R is not None:
                xyz, ctx, target = xyz @ R.T, ctx @ R.T, target @ R.T
            pos, neg = self.context_flag
            pts = np.concatenate([xyz, ctx]) if len(ctx) else xyz
            flag = np.concatenate([np.full(len(xyz), pos), np.full(len(ctx), neg)])[:, None]
            pts, target = pc_norm(pts, target)
            partial = sample_points(np.concatenate([pts, flag], axis=1), self.num_points).astype(np.float32)
            coords_target = sample_points(target, self.num_points).astype(np.float32)
        else:
            if R is not None:
                xyz, target = xyz @ R.T, target @ R.T
            xyz = sample_points(xyz, self.num_points)
            target = sample_points(target, self.num_points)
            partial, coords_target = pc_norm(xyz.astype(np.float32), target.astype(np.float32))
        return s.catid_cad, s.id_cad, partial.astype(np.float32), coords_target.astype(np.float32), sem_emb, idx


def keep_for_completion(sem_name: str | None) -> bool:
    return sem_name in VALID_CLASSES


def seed_legacy_rngs(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def completion_cache_dir(cache_root, radius: float):
    """One cache per context radius, as legacy (`..._spatial{r}cm_{phase}.npy`)."""
    from pathlib import Path

    return Path(cache_root) / f"completion_v1_spatial{int(radius * 100)}cm"


def load_completion_samples(cache_root, radius: float, split: str) -> list[CompletionSample]:
    """Cached samples of one split ("train" / "val" = official ScanNet split), 8 VALID_CLASSES only."""
    import pickle

    from caoa.data.labels import scannet_split

    out = []
    for f in sorted(completion_cache_dir(cache_root, radius).glob("scene*.pkl")):
        if scannet_split(f.stem) != split:
            continue
        with open(f, "rb") as fh:
            out.extend(s for s in pickle.load(fh) if keep_for_completion(s.sem_name))
    return out
