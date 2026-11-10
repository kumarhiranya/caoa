"""Completion metrics: definitions and aggregation."""

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from caoa.eval import completion_metrics as cm


def test_aggregation():
    """validate(): mean of per-category means; test(): (sum of samples + sum of category means) / (n + n_categories)."""
    rng = np.random.default_rng(0)
    samples = [(str(rng.choice(["03001627", "04379243", "02871439"])), rng.uniform(0, 50, 3).tolist()) for _ in range(57)]
    ev = cm.CompletionEvaluator()
    for tax, values in samples:
        ev.update(tax, values)
    s = ev.summary()
    cats = sorted({t for t, _ in samples})
    cat_mean = {c: np.mean([v for t, v in samples if t == c], axis=0) for c in cats}
    values = np.array([v for _, v in samples])
    np.testing.assert_allclose([s["validate"][k] for k in cm.METRIC_NAMES], np.mean([cat_mean[c] for c in cats], axis=0), rtol=1e-12)
    np.testing.assert_allclose([s["test"][k] for k in cm.METRIC_NAMES],
                               (values.sum(0) + sum(cat_mean.values())) / (len(samples) + len(cats)), rtol=1e-12)
    for c in cats:
        assert s["per_category"][c]["count"] == sum(t == c for t, _ in samples)
        np.testing.assert_allclose([s["per_category"][c][k] for k in cm.METRIC_NAMES], cat_mean[c], rtol=1e-12)


def test_sample_metrics_match_reference_definitions():
    torch.manual_seed(0)
    pred = torch.rand(1, 2000, 3) * 0.2
    gt = torch.rand(1, 1500, 3) * 0.2
    f, cdl1, cdl2 = cm.sample_metrics(pred, gt)

    p, g = pred[0].double().numpy(), gt[0].double().numpy()
    d_pg, _ = cKDTree(g).query(p)  # pred -> gt
    d_gp, _ = cKDTree(p).query(g)  # gt -> pred
    precision, recall = np.mean(d_pg < 0.01), np.mean(d_gp < 0.01)
    assert f == pytest.approx(2 * precision * recall / (precision + recall), abs=1e-12)
    assert cdl1 == pytest.approx((d_pg.mean() + d_gp.mean()) / 2 * 1000, rel=1e-5)
    assert cdl2 == pytest.approx(((d_pg ** 2).mean() + (d_gp ** 2).mean()) * 1000, rel=1e-5)


def test_ignore_zeros_drops_zero_sum_points_at_batch_size_1():
    gt = torch.tensor([[[0.1, -0.1, 0.0], [0.5, 0.5, 0.5]]])
    pred = torch.tensor([[[0.5, 0.5, 0.5]]])
    # The (0.1, -0.1, 0) point sums to 0, so ignore_zeros drops it: perfect match.
    assert cm.sample_metrics(pred, gt)[1] == pytest.approx(0.0, abs=1e-9)
