"""CAPCM (point completion) evaluation: ODGNet's metrics (`utils/metrics.py`) and its runner's aggregation.

* per sample (batch size 1):
  F-Score at 0.01 (Euclidean NN distances, strict <), CDL1 x 1000 and
  CDL2 x 1000 with `ignore_zeros=True`, computed on the dense output vs the GT;
* `validate()`: overall = mean over categories of per-category means (this
  picked `ckpt-best`);
* `test()`: overall = (sum of per-sample values + sum of per-category means) /
  (n_samples + n_categories), the numbers Table 2 reports;
* per-category table: count and means, keyed by taxonomy id;
* `per_sample_mean`: the plain mean over samples.
"""

from collections import defaultdict

import numpy as np
import torch

from caoa.models.chamfer import ChamferDistanceL1, ChamferDistanceL2, _gather, _nearest

METRIC_NAMES = ("F-Score", "CDL1", "CDL2")
F_SCORE_THRESHOLD = 0.01
_CDL1 = ChamferDistanceL1(ignore_zeros=True)
_CDL2 = ChamferDistanceL2(ignore_zeros=True)


def _nn_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Euclidean distance (float64, like open3d) from each src point to its nearest dst point. (1, N, 3) inputs."""
    nearest = _gather(dst, _nearest(src.float(), dst.float()))
    return ((src.double() - nearest.double()) ** 2).sum(-1).sqrt()[0]


def f_score(pred: torch.Tensor, gt: torch.Tensor, th: float = F_SCORE_THRESHOLD) -> float:
    """`Metrics._get_f_score`: per-sample F-score, averaged over the batch when batch size > 1."""
    if pred.size(0) != 1:
        return sum(f_score(pred[i : i + 1], gt[i : i + 1], th) for i in range(pred.size(0))) / pred.size(0)
    recall = float((_nn_distance(gt, pred) < th).double().mean())
    precision = float((_nn_distance(pred, gt) < th).double().mean())
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0


def sample_metrics(pred: torch.Tensor, gt: torch.Tensor) -> list[float]:
    """`Metrics.get(dense_points, gt)` -> [F-Score, CDL1 x 1000, CDL2 x 1000]."""
    with torch.no_grad():
        return [f_score(pred, gt), _CDL1(pred, gt).item() * 1000, _CDL2(pred, gt).item() * 1000]


class CompletionEvaluator:
    """Accumulates per-sample metrics by taxonomy id and reports the aggregates above."""

    def __init__(self):
        self.per_sample: list[list[float]] = []
        self.by_category: dict[str, list[list[float]]] = defaultdict(list)

    def update(self, taxonomy_id: str, values: list[float]):
        self.per_sample.append(list(values))
        self.by_category[taxonomy_id].append(list(values))

    def category_means(self) -> dict[str, list[float]]:
        return {k: np.mean(v, axis=0).tolist() for k, v in self.by_category.items()}

    def summary(self) -> dict:
        cat = self.category_means()
        cat_values = list(cat.values())
        as_dict = lambda v: dict(zip(METRIC_NAMES, (float(x) for x in v)))  # noqa: E731
        return {
            "validate": as_dict(np.mean(cat_values, axis=0)),
            "test": as_dict(np.mean(self.per_sample + cat_values, axis=0)),
            "per_sample_mean": as_dict(np.mean(self.per_sample, axis=0)),
            "per_category": {k: {"count": len(self.by_category[k]), **as_dict(v)} for k, v in cat.items()},
            "n_samples": len(self.per_sample),
        }
