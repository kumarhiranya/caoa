"""CAD samplers and context rules (P4a/P4b)."""

import numpy as np
import pytest
import trimesh

from caoa.data import context, shapenet
from caoa.paths import load_paths

PATHS = load_paths()


def legacy_resample_mesh():
    """Verbatim legacy resample_mesh, read from ODGNet/datasets/S2CCompletionDataset.py."""
    src = (PATHS.legacy_repo / "ODGNet/datasets/S2CCompletionDataset.py").read_text().replace("\r", "")
    start = src.index("def resample_mesh(")
    end = src.index("\ndef load_model(", start)
    scope = {"np": np, "trimesh": trimesh}
    exec(src[start:end], scope)
    return scope["resample_mesh"]


@pytest.mark.skipif(not (PATHS.legacy_repo / "ODGNet").exists(), reason="legacy ODGNet not found")
def test_resample_mesh_matches_legacy_with_same_rng():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    np.random.seed(3)
    ref = legacy_resample_mesh()(mesh, density=2000)
    np.random.seed(3)
    ours = shapenet.resample_mesh(mesh, density=2000)
    np.testing.assert_array_equal(ours, ref)


def test_uniform_sampler_is_area_weighted_on_the_surface():
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.1))
    pts = shapenet.sample_uniform(box, 20000, np.random.default_rng(0))
    assert np.all(np.abs(pts) <= np.array([1.0, 0.5, 0.05]) + 1e-9)
    on_big_faces = np.isclose(np.abs(pts[:, 2]), 0.05).mean()
    big, total = 2 * 2.0 * 1.0, box.area
    assert on_big_faces == pytest.approx(big / total, abs=0.02)


def test_context_rules():
    rng = np.random.default_rng(0)
    scene = rng.uniform(-3, 3, (5000, 3))
    instance = np.full(5000, -100)
    instance[np.linalg.norm(scene, axis=1) < 0.5] = 0          # the object
    instance[(scene[:, 0] > 0.6) & (scene[:, 0] < 0.9)] = 1     # a neighbouring object
    lo, hi = context.grown_box(scene[instance == 0], 1.0)
    inside = np.all((scene >= lo) & (scene <= hi), axis=1)

    v2 = context.context_points("pose_v2", scene, instance, 0, 1.0)
    comp = context.context_points("completion", scene, instance, 0, 1.0)
    assert len(v2) == (inside & (instance != 0)).sum()
    assert len(comp) == (inside & (instance < 0)).sum()
    rel = context.context_points("pose_relpose", scene, instance, 0, 1.0, num_points=1000, context_pointnum=10, rng=rng)
    assert len(rel) == 500
    assert len(context.context_points("pose_relpose", scene, instance, 0, 1.0, num_points=1000, context_pointnum=10**6)) == 0
