"""CAD samplers and the context crop."""

import numpy as np
import pytest
import trimesh

from caoa.data import context, shapenet


def test_uniform_sampler_is_area_weighted_on_the_surface():
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.1))
    pts = shapenet.sample_uniform(box, 20000, np.random.default_rng(0))
    assert np.all(np.abs(pts) <= np.array([1.0, 0.5, 0.05]) + 1e-9)
    on_big_faces = np.isclose(np.abs(pts[:, 2]), 0.05).mean()
    big, total = 2 * 2.0 * 1.0, box.area
    assert on_big_faces == pytest.approx(big / total, abs=0.02)


def test_resample_mesh_density():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    np.random.seed(3)
    pts = shapenet.resample_mesh(mesh, density=2000)
    assert len(pts) > 0 and np.allclose(np.linalg.norm(pts, axis=1), 0.4, atol=0.02)


def test_context_keeps_layout_points_in_the_grown_box():
    rng = np.random.default_rng(0)
    scene = rng.uniform(-3, 3, (5000, 3))
    instance = np.full(5000, -100)
    instance[np.linalg.norm(scene, axis=1) < 0.5] = 0          # the object
    instance[(scene[:, 0] > 0.6) & (scene[:, 0] < 0.9)] = 1     # a neighbouring object
    lo, hi = context.grown_box(scene[instance == 0], 1.0)
    inside = np.all((scene >= lo) & (scene <= hi), axis=1)
    assert len(context.context_points(scene, instance, 0, 1.0)) == (inside & (instance < 0)).sum()
    assert len(context.context_points(scene, instance, 0, 0.0)) == 0
