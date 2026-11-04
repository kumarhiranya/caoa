"""P9: vendored PTv3 / SEM."""

import pickle

import numpy as np
import pytest
import torch

from caoa.models.ptv3._compat import segment_csr
from caoa.paths import load_paths

PATHS = load_paths()
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
RUN = "S2Csym_emb128_bce_vox1cm_wd1e-03_b64_lr1e-04_bbcls-ptv3-v1m1-0-shapenet_v2_mini_ptv3cls_softmax_symbinary_cad_bce_adam_cosineannealing"
CKPT = PATHS.legacy_repo / "checkpoints/symmetry_ptv3" / RUN / f"{RUN}.pth"
needs_sem = pytest.mark.skipif(not CKPT.exists(), reason="legacy SEM checkpoint missing")


@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min"])
def test_segment_csr_matches_a_loop(reduce):
    """Stand-in for torch_scatter.segment_csr: values and gradients per CSR segment."""
    torch.manual_seed(0)
    src = torch.randn(40, 5, dtype=torch.float64, requires_grad=True)
    indptr = torch.tensor([0, 3, 4, 17, 40])
    out = segment_csr(src, indptr, reduce)
    fn = {"sum": lambda x: x.sum(0), "mean": lambda x: x.mean(0), "max": lambda x: x.max(0)[0], "min": lambda x: x.min(0)[0]}[reduce]
    ref = torch.stack([fn(src[a:b]) for a, b in zip(indptr[:-1], indptr[1:])])
    torch.testing.assert_close(out, ref)
    g = torch.randn_like(out)
    (ga,) = torch.autograd.grad(out, src, g)
    (gb,) = torch.autograd.grad(ref, src, g)
    torch.testing.assert_close(ga, gb)


@needs_cuda
@needs_sem
def test_legacy_sem_checkpoint_reproduces_legacy_embeddings():
    """The legacy checkpoint loads strictly into the vendored PTv3 and, fed as legacy `build_dataset_sym.py` fed it
    (`S2CSymmetry`, test split: no augmentation), reproduces `embeddings.npy` up to the random surface sampling."""
    from caoa.data.shapenet import load_cad_mesh, sample_uniform
    from caoa.models.sem import SEMClassifier, load_legacy_sem, make_batch

    legacy = np.load(PATHS.legacy_repo / "data/s2c_symmetry" / RUN / "embeddings.npy", allow_pickle=True).item()
    with open(PATHS.legacy_repo / "scan2cad_annotator/s2c_cad_splits.pkl", "rb") as f:
        items = pickle.load(f)["test"][:16]
    keys = ["_".join(path.split("ShapeNetV2/")[1].split("/")[:2]) for path, _ in items]
    rng = np.random.default_rng(0)
    clouds = [sample_uniform(load_cad_mesh(PATHS.shapenet, *k.split("_")), 8192, rng=rng) for k in keys]
    model = SEMClassifier()
    load_legacy_sem(model, CKPT)
    with torch.no_grad():
        out = model.cuda().eval()(make_batch(clouds, "cuda"))
    ours = out["emb"].cpu().numpy()
    ref = np.stack([legacy[k][1] for k in keys])
    cos = (ours * ref).sum(1) / np.linalg.norm(ours, axis=1) / np.linalg.norm(ref, axis=1)
    assert cos.min() > 0.99
    np.testing.assert_array_equal(out["probs"].argmax(1).cpu().numpy(), np.stack([legacy[k][0] for k in keys]).argmax(1))


def test_recalibrate_batchnorm_matches_eval_mode_activations():
    """The recalibrated statistics are the plain mean / variance of what eval mode feeds each BatchNorm (dropout off)."""
    from torch import nn

    from caoa.models.sem import recalibrate_batchnorm

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop, self.bn = nn.Dropout(0.5), nn.BatchNorm1d(4)

        def forward(self, b):
            return self.bn(self.drop(b["x"]))

    torch.manual_seed(0)
    model = Toy()
    batches = [{"x": torch.randn(32, 4) * 3 + 1} for _ in range(5)]
    model.bn.running_var.fill_(100.0)
    recalibrate_batchnorm(model, batches)
    x = torch.cat([b["x"] for b in batches])
    torch.testing.assert_close(model.bn.running_mean, x.mean(0), atol=1e-5, rtol=1e-5)
    per_batch_var = torch.stack([b["x"].var(0) for b in batches]).mean(0)  # momentum None: mean of per-batch unbiased variances
    torch.testing.assert_close(model.bn.running_var, per_batch_var, atol=1e-4, rtol=1e-4)
    assert not model.training and model.bn.momentum == 0.1
