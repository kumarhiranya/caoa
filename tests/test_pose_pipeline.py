import json

import numpy as np
import pytest
import torch

from caoa.data import scan2cad
from caoa.data.pairs import cache_dir, load_pairs
from caoa.data.pose_dataset import ObjectPoseDataset, collate
from caoa.eval import pose_eval
from caoa.paths import load_paths

PATHS = load_paths()
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_pairs = pytest.mark.skipif(not cache_dir(PATHS.cache).exists(), reason="run scripts/build_pairs.py first")


@pytest.fixture(scope="module")
def val_pairs():
    return load_pairs(PATHS.cache, "val", class_rule="relpose_cache")


@needs_cuda
def test_strided_ops_follow_minkowski_coordinates():
    import spconv.pytorch as spconv

    from caoa.models import sparse

    torch.manual_seed(0)
    idx = torch.cat([torch.randint(0, 2, (500, 1)), torch.randint(0, 64, (500, 3))], 1).unique(dim=0).int().cuda()
    x = spconv.SparseConvTensor(torch.randn(len(idx), 4, device="cuda"), idx, [64] * 3, 2)
    for op in (sparse.strided_conv(4, 8).cuda(), sparse.strided_max_pool()):
        y = op(x)
        expected = idx.clone()
        expected[:, 1:] //= 2
        assert set(map(tuple, y.indices.tolist())) == set(map(tuple, expected.unique(dim=0).tolist()))

    # Max pool at a site = max over input sites within +-1 of 2 * site (ME kernel window),
    # including negative maxima (spconv alone clamps them to 0).
    y = sparse.strided_max_pool()(x)
    for site, feat in zip(y.indices, y.features):
        near = (x.indices[:, 0] == site[0]) & ((x.indices[:, 1:] - 2 * site[1:]).abs() <= 1).all(1)
        torch.testing.assert_close(feat, x.features[near].max(0).values)


@needs_pairs
def test_test_frame_round_trip(val_pairs):
    ds = ObjectPoseDataset(val_pairs[:200], train=False)
    batch = collate([ds[i] for i in range(len(ds))])
    centers, rotations, extents = pose_eval.to_world(batch["target"].numpy(), batch["norm_scale"].numpy(), batch["norm_offset"].numpy())
    for p, c, R, e in zip(ds.pairs, centers, rotations, extents):
        np.testing.assert_allclose(c, p.box_center, atol=1e-5)
        np.testing.assert_allclose(R, p.box_rotation, atol=1e-5)
        np.testing.assert_allclose(e, p.box_extents, atol=1e-5)


@needs_pairs
def test_train_targets_describe_the_points(val_pairs):
    """Undoing the sampled target pose must put the points back inside the canonical box."""
    ds = ObjectPoseDataset(val_pairs[:100], train=True)
    for i in range(len(ds)):
        s = ds[i]
        t, R, ext = s["target"][:3].numpy(), s["target"][3:12].numpy().reshape(3, 3), s["target"][12:].numpy()
        local = (s["feats"].numpy() - t) @ R
        inside = np.all(np.abs(local) <= 1.1 * ext / 2, axis=1).mean()
        assert inside > 0.3 or ds.pairs[i].box_extents.min() < 0.05


@needs_pairs
def test_ground_truth_predictions_score_perfectly(val_pairs):
    pairs = val_pairs[:150]
    ds = ObjectPoseDataset(pairs, train=False)
    batch = collate([ds[i] for i in range(len(ds))])
    target = batch["target"].numpy()
    with open(PATHS.scan2cad_full_annotations) as f:
        full = json.load(f)
    with open(PATHS.scan2cad_cad_appearances) as f:
        appearances = json.load(f)
    scenes = sorted({p.scene_id for p in pairs})
    res = pose_eval.evaluate(
        pairs, target, target, batch["norm_scale"].numpy(), batch["norm_offset"].numpy(),
        scan2cad.load_annotations(PATHS.s2c_annotations), full, appearances, scenes,
    )
    assert res["legacy"]["scannet_class"]["avg"]["total_acc"] == 1.0
    assert res["legacy"]["table1"]["per_class"]["total"]["micro_avg"] == 1.0
    assert res["matched/minus_center"]["instance_mean"] == 1.0
    n_gt = sum(len(a["aligned_models"]) for a in full if a["id_scan"] in set(scenes))
    assert res["benchmark/minus_center"]["instance_mean"] == pytest.approx(len(pairs) / n_gt)


@needs_cuda
@needs_pairs
def test_model_forward_backward(val_pairs):
    from caoa.models.ocam import OCAMObjectOnly

    ds = ObjectPoseDataset(val_pairs[:8], train=True)
    batch = collate([ds[i] for i in range(len(ds))])
    model = OCAMObjectOnly().cuda()
    out = model(batch["coords"].cuda(), batch["feats"].cuda(), batch["batch"].cuda(), batch["batch_size"])
    assert out.shape == (8, 15)
    out.abs().sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


SYM_MODEL = "S2Csym_emb128_bce_vox1cm_wd1e-03_b64_lr1e-04_bbcls-ptv3-v1m1-0-shapenet_v2_mini_ptv3cls_softmax_symbinary_cad_bce_adam_cosineannealing"
SYM_PATH = PATHS.legacy_repo / "data/s2c_symmetry" / SYM_MODEL / "embeddings.npy"
needs_sym = pytest.mark.skipif(not SYM_PATH.exists(), reason="legacy SEM embeddings missing")


@needs_pairs
@needs_sym
def test_sym_embeddings_cover_every_pair():
    """P10a: legacy looks up `sym_map[catid_idcad][1]` for every pose pair, so every CAD needs an embedding."""
    from caoa.data.pose_dataset import load_sym_embeddings

    emb = load_sym_embeddings(SYM_PATH)
    assert {e.shape for e in emb.values()} == {(128,)}
    pairs = load_pairs(PATHS.cache, class_rule="relpose_cache")
    assert all(f"{p.catid_cad}_{p.id_cad}" in emb for p in pairs)


@needs_cuda
@needs_pairs
@needs_sym
def test_model_with_sym_embedding(val_pairs):
    """Legacy "mnkpv3_emb": the embedding is concatenated to the pooled feature, so the heads see it."""
    from caoa.data.pose_dataset import load_sym_embeddings
    from caoa.models.ocam import OCAMObjectOnly

    emb = load_sym_embeddings(SYM_PATH)
    ds = ObjectPoseDataset(val_pairs[:8], train=False, sym_embeddings=emb)
    batch = collate([ds[i] for i in range(len(ds))])
    p = val_pairs[3]
    np.testing.assert_array_equal(batch["sym_emb"][3].numpy(), emb[f"{p.catid_cad}_{p.id_cad}"])
    model = OCAMObjectOnly(ext_emb_size=128).cuda()
    args = (batch["coords"].cuda(), batch["feats"].cuda(), batch["batch"].cuda(), batch["batch_size"])
    with pytest.raises(ValueError):
        model(*args)
    out = model(*args, batch["sym_emb"].cuda())
    assert out.shape == (8, 15)
    out.abs().sum().backward()
    assert all(q.grad is not None for q in model.parameters() if q.requires_grad)
    model.eval()
    with torch.no_grad():
        a = model(*args, batch["sym_emb"].cuda())
        b = model(*args, torch.zeros_like(batch["sym_emb"]).cuda())
    assert not torch.allclose(a, b)
    assert model.translation[0][0].in_features == 2 * 1024 + 128
