"""Parity of the CAPCM (ODGNet UpTrans) port: CUDA-op replacements, and the model vs the legacy code."""

import importlib
import sys
import types

import numpy as np
import pytest
import torch
import yaml

from caoa.models import chamfer, pointops
from caoa.models.odgnet import UpTrans, load_legacy_checkpoint
from caoa.paths import load_paths

PATHS = load_paths()
ODGNET = PATHS.legacy_repo / "ODGNet"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_legacy = pytest.mark.skipif(not (ODGNET / "models" / "UpTrans.py").exists(), reason="legacy ODGNet not found")


def fps_reference(points: np.ndarray, m: int) -> list[int]:
    """`furthest_point_sampling_kernel` (pointnet2_ops sampling_gpu.cu), sequential, float32."""
    pts = points.astype(np.float32)
    temp = np.full(len(pts), 1e10, dtype=np.float32)
    skip = (pts * pts).sum(1) <= np.float32(1e-3)
    idx, old = [0], 0
    for _ in range(1, m):
        diff = pts - pts[old]
        d = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
        temp = np.where(skip, temp, np.minimum(temp, d))
        cand = np.where(skip, np.float32(-1), temp)
        old = int(cand.argmax())
        idx.append(old)
    return idx


def test_furthest_point_sample_matches_cuda_kernel_semantics():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(3, 700, 3)).astype(np.float32)
    pts[:, 5:40] *= 0.01  # points the kernel never selects (|p|^2 <= 1e-3)
    ours = pointops.furthest_point_sample(torch.from_numpy(pts), 128).numpy()
    for b in range(3):
        assert ours[b].tolist() == fps_reference(pts[b], 128)
        assert not set(range(5, 40)) & set(ours[b, 1:].tolist())


def test_gather_and_grouping():
    feats = torch.randn(2, 5, 30)
    idx = torch.randint(0, 30, (2, 7, 4))
    grouped = pointops.grouping_operation(feats, idx)
    for b in range(2):
        np.testing.assert_allclose(grouped[b].numpy(), feats[b][:, idx[b]].numpy())
    sel = idx[:, :, 0]
    np.testing.assert_allclose(pointops.gather_operation(feats, sel)[1].numpy(), feats[1][:, sel[1]].numpy())


def test_chamfer_matches_brute_force_values_and_gradients():
    torch.manual_seed(0)
    a = torch.randn(2, 300, 3, requires_grad=True)
    b = torch.randn(2, 450, 3)
    d = torch.cdist(a, b) ** 2
    ref_l2 = d.min(2).values.mean() + d.min(1).values.mean()
    ref_l1 = (d.min(2).values.sqrt().mean() + d.min(1).values.sqrt().mean()) / 2
    l2, l1 = chamfer.ChamferDistanceL2()(a, b), chamfer.ChamferDistanceL1()(a, b)
    assert l2.item() == pytest.approx(ref_l2.item(), rel=1e-5)
    assert l1.item() == pytest.approx(ref_l1.item(), rel=1e-5)
    (g_ref,) = torch.autograd.grad(ref_l1, a)
    (g,) = torch.autograd.grad(l1, a)
    torch.testing.assert_close(g, g_ref, rtol=1e-4, atol=1e-6)


class _Cfg(dict):
    """EasyDict-like config: attribute access plus keys(), as legacy UpTrans expects."""

    __getattr__ = dict.__getitem__


@pytest.fixture
def legacy_uptrans():
    """Import the legacy ODGNet model code with our ops standing in for the CUDA extensions."""
    saved = {k: sys.modules.get(k) for k in ("pointnet2_ops", "pointnet2_ops.pointnet2_utils", "extensions",
                                             "extensions.chamfer_dist", "models", "models.build", "models.pointnet", "models.UpTrans")}
    utils = types.ModuleType("pointnet2_ops.pointnet2_utils")
    for name in ("furthest_point_sample", "gather_operation", "grouping_operation"):
        setattr(utils, name, getattr(pointops, name))
    for name in ("ball_query", "three_nn", "three_interpolate"):  # imported but not used by UpTrans
        setattr(utils, name, None)
    pkg = types.ModuleType("pointnet2_ops")
    pkg.pointnet2_utils = utils
    ext = types.ModuleType("extensions")
    cd = types.ModuleType("extensions.chamfer_dist")
    cd.ChamferDistanceL1, cd.ChamferDistanceL2 = chamfer.ChamferDistanceL1, chamfer.ChamferDistanceL2
    models = types.ModuleType("models")
    models.__path__ = [str(ODGNET / "models")]
    build = types.ModuleType("models.build")
    build.MODELS = types.SimpleNamespace(register_module=lambda *a, **k: (lambda cls: cls))
    sys.modules.update({"pointnet2_ops": pkg, "pointnet2_ops.pointnet2_utils": utils, "extensions": ext,
                        "extensions.chamfer_dist": cd, "models": models, "models.build": build})
    try:
        yield importlib.import_module("models.UpTrans").UpTrans
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def experiment_dirs():
    """Legacy runs that saved checkpoints (test-only runs keep a config but no ckpt)."""
    if not ODGNET.exists():
        return []
    return sorted(p.parent for p in (ODGNET / "experiments").glob("*/*/*/config.yaml") if any(p.parent.glob("ckpt-*.pth")))


@needs_cuda
@needs_legacy
@pytest.mark.parametrize("with_context", [False, True])
def test_port_matches_legacy_model_on_same_weights(legacy_uptrans, with_context):
    run = ODGNET / "experiments" / ("UpTrans_comp/S2CCompletion/s2ccompletion_upTransb1_comp" if with_context else "UpTrans/S2CCompletion/s2ccompletion_upTransb1")
    model_cfg = yaml.safe_load(open(run / "config.yaml"))["model"]
    legacy = legacy_uptrans(_Cfg(model_cfg)).cuda().eval()
    ours = UpTrans.from_config(model_cfg).cuda().eval()
    load_legacy_checkpoint(legacy, run / "ckpt-best.pth")
    load_legacy_checkpoint(ours, run / "ckpt-best.pth")

    torch.manual_seed(0)
    xyz = torch.randn(2, 4096, 3, device="cuda") * 0.3
    feats = torch.sign(torch.randn(2, 4096, 1, device="cuda")) if with_context else None
    with torch.no_grad():
        out_legacy = legacy(xyz, feats)
        out_ours = ours(xyz, feats)
    for a, b in zip(out_legacy, out_ours):
        torch.testing.assert_close(a, b)
    assert out_ours[1].shape == (2, model_cfg["num_pred"], 3)


@needs_legacy
@pytest.mark.parametrize("run", experiment_dirs(), ids=lambda p: "/".join(p.parts[-3:]))
def test_legacy_checkpoints_load_strictly(run):
    model_cfg = yaml.safe_load(open(run / "config.yaml"))["model"]
    if model_cfg.get("NAME") != "UpTrans":
        pytest.skip(f"{model_cfg.get('NAME')} is not ported")
    ckpts = sorted(run.glob("ckpt-*.pth"))
    info = load_legacy_checkpoint(UpTrans.from_config(model_cfg), ckpts[0])
    assert info["epoch"] is not None
