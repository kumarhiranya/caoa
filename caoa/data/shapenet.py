"""ShapeNet CAD models: loading and the two legacy point samplers.

Each legacy module sampled CAD surfaces its own way, and each port follows its
own call path:

* pose pipeline (`S2CCompletionPoseV2`): `trimesh.load_mesh(force="mesh")` ->
  open3d `sample_points_uniformly(5000)`, i.e. area-weighted uniform sampling;
  `sample_uniform` reproduces the distribution (not open3d's RNG stream).
* completion loader (`ODGNet/datasets/S2CCompletionDataset.py`):
  `trimesh.load(force="mesh", skip_materials=True)` -> `resample_mesh(density=2000)`
  (Chris Choy's barycentric sampler; point count ~ surface area x density).
"""

from pathlib import Path

import numpy as np
import trimesh

POSE_CAD_POINTS = 5000
COMPLETION_CAD_DENSITY = 2000


def cad_path(shapenet_root: Path, catid: str, id_cad: str) -> Path:
    return Path(shapenet_root) / catid / id_cad / "models" / "model_normalized.obj"


def load_cad_mesh(shapenet_root: Path, catid: str, id_cad: str) -> trimesh.Trimesh:
    return trimesh.load(cad_path(shapenet_root, catid, id_cad), force="mesh", skip_materials=True)


def sample_uniform(mesh: trimesh.Trimesh, n: int = POSE_CAD_POINTS, rng: np.random.Generator | None = None) -> np.ndarray:
    """Area-weighted uniform surface samples (the distribution of open3d `sample_points_uniformly`)."""
    rng = rng or np.random.default_rng()
    tri = mesh.triangles
    areas = mesh.area_faces
    faces = rng.choice(len(tri), size=n, p=areas / areas.sum())
    r1, r2 = rng.random((n, 1)), rng.random((n, 1))
    s = np.sqrt(r1)
    a, b, c = tri[faces, 0], tri[faces, 1], tri[faces, 2]
    return (1 - s) * a + s * (1 - r2) * b + s * r2 * c


def resample_mesh(mesh_cad, density: float = 1) -> np.ndarray:
    """Verbatim port of the legacy `resample_mesh` (uses the global NumPy RNG, as the original)."""
    faces = np.array(mesh_cad.faces if isinstance(mesh_cad, trimesh.Trimesh) else mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :], vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1
    n_samples = np.sum(n_samples_per_face)

    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc : acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    return (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
