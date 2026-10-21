"""P8: completion inference for the pose pairs and its use in the pose dataset."""

import numpy as np
import pytest
import torch

from caoa.data.pairs import cache_dir, load_pairs
from caoa.data.pose_completion import canonical_points, complete
from caoa.data.pose_dataset import ObjectPoseDataset
from caoa.paths import load_paths
from tests import legacy_ref

PATHS = load_paths()
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_pairs = pytest.mark.skipif(not cache_dir(PATHS.cache).exists(), reason="run scripts/build_pairs.py first")


@needs_cuda
def test_completion_inference_matches_legacy():
    """Same model, same inputs -> same completions. Instance sizes in [2048, 4096) make the
    4096-point input deterministic (tiling) in both; 19 instances cover a partial last batch."""
    from caoa.models.odgnet import UpTrans

    torch.manual_seed(0)
    model = UpTrans().cuda().eval()
    rng = np.random.default_rng(0)
    sets = [rng.normal(size=(rng.integers(2048, 4096), 3)) * rng.uniform(0.2, 1.0, 3) for _ in range(19)]
    ours = complete(model, sets, torch.device("cuda"))
    with torch.no_grad():
        theirs = legacy_ref.infer_completion_odg(model, sets)
    assert len(ours) == len(theirs) == len(sets)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)


@pytest.fixture(scope="module")
def pairs():
    return load_pairs(PATHS.cache, "val", class_rule="relpose_cache")[:60]


def _identity_completions(pairs):
    """'Completions' equal to the (tiled) canonical instance points."""
    return {(p.scene_id, p.model_index): canonical_points(p).astype(np.float32) for p in pairs}


@needs_pairs
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("mode", ["only", "inst+comp"])
def test_completion_frames_match_the_instance_frames(pairs, train, mode):
    """Legacy maps completions from the canonical frame back to the world (test) or augments them
    in the canonical frame (train), exactly like the instance points. With completions equal to the
    instance points, targets and normalisation must not change ("inst+comp" only duplicates points,
    which leaves the bounding box and mean unchanged)."""
    plain = ObjectPoseDataset(pairs, train)
    comp = ObjectPoseDataset(pairs, train, completions=_identity_completions(pairs), completion_mode=mode)
    for i in range(len(pairs)):
        np.random.seed(i)
        a = plain[i]
        np.random.seed(i)
        b = comp[i]
        torch.testing.assert_close(b["target"], a["target"], rtol=0, atol=1e-5)
        assert b["norm_scale"] == pytest.approx(a["norm_scale"], rel=1e-6)
        np.testing.assert_allclose(b["norm_offset"], a["norm_offset"], atol=1e-6)
        if mode == "only":
            torch.testing.assert_close(b["feats"], a["feats"], rtol=0, atol=1e-5)


def test_completion_arguments_go_together(pairs):
    with pytest.raises(ValueError):
        ObjectPoseDataset(pairs, False, completions={})
    with pytest.raises(ValueError):
        ObjectPoseDataset(pairs, False, completions={}, completion_mode="instance")
