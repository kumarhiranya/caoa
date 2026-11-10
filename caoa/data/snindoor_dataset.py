"""SN-Indoor samples for the CAPCM: a port of ODGNet's `SNAugDataset`.

Per item:
* GT: the CAD mesh (`trimesh.load(force="mesh", skip_materials=True)`) sampled afresh with `mesh.sample(10000)`;
* input: the pre-generated partial scan (`ShapeNetV2-augmented/<cat>/<id>/<k>.ply`), cropped by a random non-linear
  surface (the larger side is kept; "hard" = no shift) and jittered with Gaussian noise (sigma 0.005);
* both sampled to `num_points`, rotated by the same random yaw in [-180, 180] degrees (train and val), then `pc_norm`
  on the input; returns the 6-tuple `(catid, id_cad, partial, gt, sem_emb, idx)`.

The file list is `aug_train_test_splits.pkl[phase]` (7 categories; its paths start with `data/S2CCompletion/ShapeNetV2`
and are mapped onto `shapenet_root`), drawn with `random.choices(k=len // data_downscale_factor)` (with replacement).
"""

import math
import pickle
import random
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData
from torch.utils.data import Dataset

from caoa.data.completion_dataset import pc_norm
from caoa.data.labels import synset_to_eval_class
from caoa.data.sampling import sample_points
from caoa.geometry import yaw_matrix

_SPLIT_PREFIX = "data/S2CCompletion/ShapeNetV2"
SHIFT_RANGE = {"easy": 0.2, "medium": 0.1}  # anything else ("hard") -> 0


def load_split(shapenet_root: Path, phase: str) -> list[tuple[Path, Path]]:
    """`aug_train_test_splits.pkl` pairs with their relative paths mapped onto `shapenet_root`."""
    with open(Path(shapenet_root) / "aug_train_test_splits.pkl", "rb") as f:
        pairs = pickle.load(f)["val" if phase in ("val", "test") else "train"]
    root = str(shapenet_root)

    def remap(p: str) -> Path:
        assert p.startswith(_SPLIT_PREFIX), p
        return Path(root + p[len(_SPLIT_PREFIX):])

    return [(remap(mesh), remap(pcd)) for mesh, pcd in pairs]


def crop_random_nonlinear(points: np.ndarray, difficulty: str = "easy") -> np.ndarray:
    """ODGNet's random non-linear crop (same `random` calls in the same order), vectorised."""
    a = random.random() * 7
    b = random.random() * 3
    m = random.random() / 2
    n = random.random()
    p = random.random() * 5
    q = random.random() * 5
    u = random.choice([-3, -2, 2, 3])
    v = random.choice([-3, -2, 2, 3])
    shift_range = SHIFT_RANGE.get(difficulty, 0)
    shift = random.uniform(-shift_range, shift_range)
    k = -(m * math.sin(p) + n * math.cos(q)) + shift  # -nonlinear_plane(0, 0) + shift, with k = 0 inside
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    below = z < m * np.sin(a * x + p) + n * np.cos(b * y + q) + u * x + v * y + k
    side1, side2 = points[below], points[~below]
    return side1 if len(side1) > len(side2) else side2


def add_gaussian_noise(points: np.ndarray, degree: float = 0.005) -> np.ndarray:
    return points + np.random.normal(0, degree, points.shape)


def read_ply_points(path: Path) -> np.ndarray:
    v = PlyData.read(str(path))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


class SNIndoorDataset(Dataset):
    def __init__(self, shapenet_root: Path, phase: str, sem_embeddings: dict, num_points: int = 4096, difficulty: str = "hard",
                 add_noise_to_target: bool = False, rotate_z: bool = True, data_downscale_factor: int = 1,
                 filter_classes: bool = False, add_dummy_feat: bool = False):
        self.phase = "val" if phase in ("val", "test") else "train"
        self.num_points = num_points
        self.difficulty = difficulty
        self.add_noise_to_target = add_noise_to_target
        self.rotate_z = rotate_z
        self.add_dummy_feat = add_dummy_feat
        self.sem_embeddings = sem_embeddings
        pairs = load_split(shapenet_root, self.phase)
        self.file_pair_list = random.choices(pairs, k=len(pairs) // data_downscale_factor)
        if filter_classes:
            valid = synset_to_eval_class()
            self.file_pair_list = [fp for fp in self.file_pair_list if self.cat_id(fp[0]) in valid]

    @staticmethod
    def cat_id(mesh_path: Path) -> str:
        return mesh_path.parts[-4]

    @staticmethod
    def model_id(mesh_path: Path) -> str:
        return mesh_path.parts[-3]

    def __len__(self):
        return len(self.file_pair_list)

    def __getitem__(self, idx):
        mesh_path, pcd_path = self.file_pair_list[idx]
        cat_id, model_id = self.cat_id(mesh_path), self.model_id(mesh_path)
        sem_emb = self.sem_embeddings[f"{cat_id}_{model_id}"].astype(np.float32)

        target = np.asarray(trimesh.load(mesh_path, force="mesh", skip_materials=True).sample(10000))
        partial = add_gaussian_noise(crop_random_nonlinear(read_ply_points(pcd_path), self.difficulty), degree=0.005)
        target = sample_points(target, self.num_points)
        partial = sample_points(partial, self.num_points)
        if self.add_noise_to_target:
            target = add_gaussian_noise(target, degree=0.005)
        if self.rotate_z:
            R = yaw_matrix(np.radians(np.random.randint(-180, 181)))
            target, partial = target @ R.T, partial @ R.T
        coords, coords_target = pc_norm(partial.astype(np.float32), target.astype(np.float32))
        if self.add_dummy_feat:
            coords = np.concatenate([coords, np.ones((len(coords), 1), dtype=np.float32)], axis=1)
        return cat_id, model_id, coords.astype(np.float32), coords_target.astype(np.float32), sem_emb, idx
