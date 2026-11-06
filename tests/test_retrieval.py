import numpy as np
import pytest
import torch

from caoa.retrieval import search
from caoa.retrieval.data import normalize_bbox
from caoa.retrieval.model import info_nce

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_normalize_bbox_unit_longest_side():
    pts = np.random.default_rng(0).normal(size=(500, 3)) * [3.0, 1.0, 0.5] + 7.0
    n = normalize_bbox(pts)
    np.testing.assert_allclose((n.max(0) + n.min(0)) / 2, 0, atol=1e-12)
    assert np.ptp(n, axis=0).max() == pytest.approx(1.0)


def test_info_nce_masks_duplicate_positives():
    q = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    c = torch.cat([q, q[:1]])  # candidate 3 is another copy of query 0's GT CAD
    keys, targets = torch.tensor([10, 11, 12, 10]), torch.tensor([10, 11, 12])
    masked = info_nce(q, c, keys, targets, torch.tensor(50.0))
    unmasked = info_nce(q, c, torch.tensor([10, 11, 12, 99]), targets, torch.tensor(50.0))
    assert masked < unmasked  # the duplicate no longer competes with the positive


def test_search_rank_and_category_mask():
    db = torch.nn.functional.normalize(torch.randn(20, 4), dim=1)
    catids = np.array(["a"] * 10 + ["b"] * 10)
    targets = torch.tensor([3, 15])
    q = db[targets] + 0.01
    mask = search.allowed_mask(catids[targets.numpy()], catids, "same", "cpu")
    top, rank = search.search(torch.nn.functional.normalize(q, dim=1), db, mask, targets, k=5)
    assert top[:, 0].tolist() == [3, 15] and rank.tolist() == [1, 1]
    assert np.all(catids[top[0].numpy()] == "a") and np.all(catids[top[1].numpy()] == "b")


@needs_cuda
def test_identical_shapes_have_zero_chamfer_and_perfect_metrics():
    pts = torch.rand(6, 256, 3, device="cuda") - 0.5
    l1, l2 = search.chamfer(pts, pts)
    np.testing.assert_allclose(l1, 0, atol=1e-6)
    targets = torch.arange(6)
    catids = np.array(["a", "a", "a", "b", "b", "b"])
    res = search.retrieval_metrics(targets.cuda()[:, None].repeat(1, 50), torch.ones(6, dtype=torch.long), targets, pts, catids, None)
    assert res["recall@1"] == 1.0 and res["cd_l1"] == pytest.approx(0, abs=1e-6) and res["category_acc"] == 1.0


@needs_cuda
def test_retrieval_model_forward_backward():
    from caoa.retrieval.model import CADRetrieval

    model = CADRetrieval(embedding=64, out_dim=32).cuda()
    q = model.query(torch.rand(4, 512, 3, device="cuda") - 0.5)
    c = model.cad(torch.rand(6, 512, 3, device="cuda") - 0.5)
    assert q.shape == (4, 32) and torch.allclose(q.norm(dim=1), torch.ones(4, device="cuda"))
    loss = info_nce(q, c, torch.arange(6, device="cuda"), torch.arange(4, device="cuda"), model.logit_scale())
    loss.backward()
    assert model.log_scale.grad is not None and next(model.cad.trunk.parameters()).grad is not None


def test_class_synsets_and_allowed():
    from types import SimpleNamespace

    from caoa.retrieval.pipeline import class_allowed, class_synsets

    pairs = [SimpleNamespace(scannet_class="chair", catid_cad=c) for c in ["a"] * 96 + ["b"] * 4]
    pairs += [SimpleNamespace(scannet_class="table", catid_cad=c) for c in ["t"] * 50 + ["u"] * 50]
    syn = class_synsets(pairs)
    assert syn == {"chair": ["a"], "table": ["t", "u"]}  # fewest synsets covering 95%
    allowed = class_allowed(["chair", "otherfurniture", "sofa"], syn, ["a", "b", "t"]).numpy()
    assert allowed.tolist() == [[True, False, False], [True, True, True], [True, True, True]]  # search-all / unmapped
