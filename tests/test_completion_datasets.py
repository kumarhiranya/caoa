"""CAPCM datasets: the SN-Indoor crop, frames and item formats."""

import random

import numpy as np
import pytest

from caoa.data import shapenet, snindoor_dataset
from caoa.data.completion_dataset import CompletionSample, S2CCompletionDataset, keep_for_completion
from caoa.data.pairs import load_pairs
from caoa.paths import load_paths

PATHS = load_paths()
SEM_EMB = PATHS.semantic_embeddings
needs_snindoor = pytest.mark.skipif(not (PATHS.shapenet.parent / "ShapeNetV2-augmented").exists(), reason="SN-Indoor partials not available")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_snindoor_crop_keeps_a_part_and_is_seeded(difficulty):
    pts = np.random.default_rng(1).uniform(-0.5, 0.5, (3000, 3))
    random.seed(0)
    a = snindoor_dataset.crop_random_nonlinear(pts, difficulty)
    random.seed(0)
    b = snindoor_dataset.crop_random_nonlinear(pts, difficulty)
    np.testing.assert_array_equal(a, b)
    assert 0 < len(a) <= len(pts)


@needs_snindoor
@pytest.mark.data
def test_snindoor_items():
    sem = np.load(SEM_EMB, allow_pickle=True).item()
    ds = snindoor_dataset.SNIndoorDataset(PATHS.shapenet, "val", sem, data_downscale_factor=10000)
    cat, model, partial, gt, emb, idx = ds[0]
    assert partial.shape == (4096, 3) and gt.shape == (4096, 3) and emb.shape == (512,)
    assert np.abs(partial.mean(0)).max() < 1e-4 and np.isclose(np.linalg.norm(partial, axis=1).max(), 1.0, atol=1e-5)


@pytest.mark.data
def test_s2c_completion_frames():
    """The instance must lie on its CAD in both frames (train: CAD frame; val: normalised world frame)."""
    pairs = [p for p in load_pairs(PATHS.cache, "val") if keep_for_completion(p.sem_name)][:4]
    if not all((PATHS.shapenet / p.catid_cad / p.id_cad).exists() for p in pairs):
        pytest.skip("CAD meshes not available")
    np.random.seed(0)
    samples = [
        CompletionSample(p.scene_id, p.model_index, p.catid_cad, p.id_cad, p.sem_name, p.points.astype(float), np.zeros((0, 3)),
                         shapenet.resample_mesh(shapenet.load_cad_mesh(PATHS.shapenet, p.catid_cad, p.id_cad), 2000), p.cad_matrix)
        for p in pairs
    ]
    sem = np.load(SEM_EMB, allow_pickle=True).item()
    for phase in ("train", "val"):
        ds = S2CCompletionDataset(samples, phase, sem)
        for i in range(len(ds)):
            inst, cad = ds.data[i], ds.targets[i]
            d = np.sqrt(((inst[:, None, :] - cad[None, ::7, :]) ** 2).sum(-1)).min(1)
            scale = np.ptp(cad, axis=0).max()
            assert np.median(d) < 0.05 * scale, (phase, i, np.median(d), scale)
            cat, model, partial, gt, emb, idx = ds[i]
            assert partial.shape == (4096, 3) and gt.shape == (4096, 3)
