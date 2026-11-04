
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
    res = pose_eval.evaluate(
        pairs, target, target, batch["norm_scale"].numpy(), batch["norm_offset"].numpy(), scan2cad.load_annotations(PATHS.s2c_annotations)
    )
    assert res["legacy"]["scannet_class"]["avg"]["total_acc"] == 1.0
    assert res["legacy"]["table1"]["per_class"]["total"]["micro_avg"] == 1.0
    assert res["s2c_criteria/minus_center"]["instance_mean"] == 1.0
    assert res["s2c_criteria/minus_center"]["class_mean"] == 1.0
    assert res["s2c_box"]["instance_mean"] == 1.0


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
@pytest.mark.parametrize("frame", ["legacy_export", "cad_bbox"])
def test_sym_embeddings_cover_every_pair(frame):
    """P10a: legacy looks up `sym_map[catid_idcad][1]` for every pose pair, so every CAD needs an embedding."""
    from caoa.data.pose_dataset import load_sym_embeddings, sem_embedding_path

    path = sem_embedding_path(PATHS, SYM_MODEL, frame)
    if not path.exists():
        pytest.skip(f"{path} missing (scripts/build_sem_embeddings.py)")
    emb = load_sym_embeddings(path)
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


def _fake_cad_points(pairs, n=4096, seed=0):
    """CAD stand-ins: points filling each pair's box, in the CAD bounding-box frame (centred, longest side 1)."""
    rng = np.random.default_rng(seed)
    out = {}
    for p in pairs:
        ext = p.box_extents / p.box_extents.max()
        pts = rng.uniform(-0.5, 0.5, (n, 3)) * ext
        pts[:8] = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * ext  # exact bbox
        out[f"{p.catid_cad}_{p.id_cad}"] = pts.astype(np.float32)
    return out


@needs_pairs
def test_chamfer_box_points_follow_the_target(val_pairs):
    """P10c: unit-box CAD points placed by the GT box pose give the GT box (test frame: metres x norm_scale - offset),
    and the Chamfer pose loss is zero at the GT pose and grows with a pose error."""
    from caoa.models.pose_losses import box_points, chamfer_pose_loss

    pairs = val_pairs[:6]
    ds = ObjectPoseDataset(pairs, train=False, cad_points=_fake_cad_points(pairs))
    batch = collate([ds[i] for i in range(len(ds))])
    corners = torch.tensor([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], dtype=torch.float32)
    got = box_points(batch["target"].double(), corners[None].expand(len(pairs), -1, -1).double())
    for i, p in enumerate(pairs):
        world = p.box_center + (corners.numpy() * p.box_extents) @ p.box_rotation.T
        want = world * batch["norm_scale"][i].item() - batch["norm_offset"][i].numpy()
        np.testing.assert_allclose(got[i].numpy(), want, atol=1e-5)
    unit = batch["cad_unit"]
    assert unit.abs().max() <= 0.5 + 1e-5
    assert chamfer_pose_loss(batch["target"], batch["target"], unit).item() == pytest.approx(0, abs=1e-10)
    off = batch["target"].clone()
    off[:, :3] += 0.05
    assert chamfer_pose_loss(off, batch["target"], unit).item() > 1e-3


@needs_cuda
@needs_pairs
@pytest.mark.parametrize("shared", [True, False])
def test_siamese_forward_backward(val_pairs, shared):
    """P10b: shared = one trunk (paper); unshared = two trunks, each with its own conv4 (legacy RelPose, bug fixed)."""
    from caoa.models.ocam import OCAMSiamese

    pairs = val_pairs[:4]
    ds = ObjectPoseDataset(pairs, train=True, cad_points=_fake_cad_points(pairs))
    batch = collate([ds[i] for i in range(len(ds))])
    model = OCAMSiamese(shared=shared).cuda()
    assert (model.cad is model.obj) == shared
    obj = [batch[k].cuda() for k in ("coords", "feats", "batch")]
    cad = [batch[k].cuda() for k in ("cad_coords", "cad_feats", "cad_batch")]
    out = model(*obj, batch["batch_size"], *cad)
    assert out.shape == (4, 15)
    out.abs().sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    if not shared:
        for trunk in (model.obj, model.cad):
            assert next(trunk.conv4.parameters()).grad.abs().sum() > 0


@needs_cuda
@needs_pairs
@pytest.mark.parametrize("shared", [True, False])
def test_siamese_overfits_32_samples(val_pairs, shared):
    """P10 gate: the Siamese OCAM with the paper's loss (t2 / r3 / s2 L1 + 5 x Chamfer) fits 32 fixed samples
    (legacy metric ~100%; >= 90% after 800 steps, a margin over spconv's run-to-run GPU nondeterminism)."""
    from caoa.eval import legacy_report
    from caoa.models.ocam import OCAMSiamese
    from caoa.models.pose_losses import chamfer_pose_loss

    torch.manual_seed(0)
    np.random.seed(0)
    pairs = val_pairs[:32]
    ds = ObjectPoseDataset(pairs, train=False, cad_points=_fake_cad_points(pairs))
    batch = collate([ds[i] for i in range(len(ds))])
    model = OCAMSiamese(embedding=256, shared=shared).cuda()
    inputs = [batch[k].cuda() for k in ("coords", "feats", "batch")] + [batch["batch_size"]] + [batch[k].cuda() for k in ("cad_coords", "cad_feats", "cad_batch")]
    target, unit = batch["target"].cuda(), batch["cad_unit"].cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    w = torch.tensor([2.0] * 3 + [3.0] * 9 + [2.0] * 3, device="cuda")
    for _ in range(800):
        pred = model(*inputs)
        loss = ((pred - target).abs() * w).sum() / len(pairs) + 5 * chamfer_pose_loss(pred, target, unit)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(*inputs).cpu().numpy()
    report = legacy_report.eval_pose_report(pred, batch["target"].numpy(), [p.sym for p in pairs], [p.sem_name for p in pairs])
    assert report["avg"]["total_acc"] >= 0.9, report["avg"]["total_acc"]
