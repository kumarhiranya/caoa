"""The CAPCM (ODGNet UpTrans) port: CUDA-op replacements, the model and its checkpoints."""

import numpy as np
import pytest
import torch
import yaml

from caoa.models import chamfer, pointops
from caoa.models.odgnet import UpTrans, load_checkpoint
from caoa.paths import REPO_ROOT

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


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


def test_uptrans_forward_shapes():
    model = UpTrans().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 2048, 3) * 0.3)
    assert out[1].shape[0] == 2 and out[1].shape[2] == 3 and out[1].shape[1] > 2048


@pytest.mark.parametrize("run", sorted((REPO_ROOT / "outputs").glob("capcm_*")), ids=lambda p: p.name)
def test_trained_checkpoints_load_strictly(run):
    model_cfg = yaml.safe_load(open(run / "config.yaml"))["model"]
    info = load_checkpoint(UpTrans.from_config(model_cfg), run / "ckpt-best.pth")
    assert info["epoch"] is not None
