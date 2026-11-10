"""Completion inference for the object–CAD pairs and its use in the pose dataset."""

import dataclasses

import numpy as np
import pytest
import torch

from caoa.data.pairs import cache_dir, load_pairs
from caoa.data.pose_completion import complete, complete_pairs, to_canonical, world_points
from caoa.data.pose_dataset import ObjectPoseDataset
from caoa.geometry import yaw_matrix
from caoa.paths import load_paths

PATHS = load_paths()
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_pairs = pytest.mark.skipif(not cache_dir(PATHS.cache).exists(), reason="run scripts/build_pairs.py first")


@needs_cuda
def test_completion_inference_outputs():
    """One completion per input set (19 sets: a partial last batch), in the input's coordinates: the model works on the
    centred, radius-normalised input and the output is mapped back."""
    from caoa.models.odgnet import UpTrans

    torch.manual_seed(0)
    model = UpTrans().cuda().eval()
    rng = np.random.default_rng(0)
    sets = [rng.normal(size=(rng.integers(2048, 4096), 3)) * rng.uniform(0.2, 1.0, 3) + rng.uniform(-5, 5, 3) for _ in range(19)]
    out = complete(model, sets, torch.device("cuda"))
    assert len(out) == len(sets)
    for x, c in zip(sets, out):
        assert c.ndim == 2 and c.shape[1] == 3 and len(c) > len(x) and np.isfinite(c).all()
        radius = np.linalg.norm(x - x.mean(0), axis=1).max()
        assert np.linalg.norm(c.mean(0) - x.mean(0)) < radius  # mapped back next to the input


@pytest.fixture(scope="module")
def pairs():
    return load_pairs(PATHS.cache, "val")[:60]


def _identity_completions(pairs):
    """'Completions' equal to the (tiled) canonical instance points."""
    return {(p.scene_id, p.model_index): to_canonical(p, world_points(p)) for p in pairs}


@needs_pairs
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("mode", ["only", "inst+comp"])
def test_completion_frames_match_the_instance_frames(pairs, train, mode):
    """The dataset maps completions from the canonical frame back to the world (test) or augments them
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


def _to_world(p, canonical):
    return canonical.astype(np.float64) * p.box_extents.max() @ p.box_rotation.T + p.box_center


@needs_cuda
@needs_pairs
def test_world_frame_completion_uses_no_gt_pose(pairs):
    """Changing the GT box leaves the completion, mapped back to the world as the pose dataset does, unchanged: no part
    of the GT pose reaches the completion."""
    from caoa.models.odgnet import UpTrans

    torch.manual_seed(0)
    model = UpTrans().cuda().eval()
    device = torch.device("cuda")
    ps = pairs[:10]
    moved = [dataclasses.replace(p, box_center=p.box_center + 1.0, box_rotation=p.box_rotation @ yaw_matrix(1.0),
                                 box_extents=p.box_extents * 1.7) for p in ps]
    np.random.seed(0)
    a = complete_pairs(model, ps, device, "world")
    np.random.seed(0)
    b = complete_pairs(model, moved, device, "world")
    for p, q, ca, cb in zip(ps, moved, a, b):
        np.testing.assert_allclose(_to_world(p, ca), _to_world(q, cb), rtol=0, atol=1e-4)
