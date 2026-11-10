"""ShapeNet CAD models: loading and the two surface samplers.

* `sample_uniform`: a fixed number of area-weighted uniform surface samples (pose pipeline, 5,000 points);
* `resample_mesh`: Chris Choy's barycentric sampler, point count ~ surface area x density (completion targets,
  density 2,000).
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
    """Barycentric sampler with about `density` points per unit area (uses the global NumPy RNG)."""
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


# CAD retrieval database: every ShapeNetV2 model, sampled once (`scripts/build_shapenet_points.py`).
SHAPENET_POINTS_VERSION = "shapenet_points_v1"
SHAPENET_POINTS = 2048


def shapenet_points_path(cache_root: Path) -> Path:
    return Path(cache_root) / SHAPENET_POINTS_VERSION / f"bbox_{SHAPENET_POINTS}.npz"


def list_models(shapenet_root: Path) -> list[str]:
    """Every model in the ShapeNetV2 copy as `catid_idcad`, sorted (55 synsets; loose files are skipped)."""
    root = Path(shapenet_root)
    return sorted(
        f"{c.name}_{m.name}"
        for c in root.iterdir() if c.is_dir() and c.name.isdigit() and len(c.name) == 8
        for m in c.iterdir() if m.is_dir()
    )


def load_shapenet_points(cache_root: Path) -> dict:
    """{keys (N,) str, catids (N,) str, points (N, 2048, 3) float32 in the CAD's own bbox frame (centred, longest side 1),
    bbox_center / bbox_extent (N, 3) of the samples in `model_normalized` coordinates}."""
    with np.load(shapenet_points_path(cache_root)) as f:
        return {k: f[k] for k in f.files}
