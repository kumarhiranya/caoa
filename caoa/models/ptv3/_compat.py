"""Stand-ins for the two third-party helpers the vendored PTv3 `model.py` imports.

* `segment_csr`: `torch_scatter.segment_csr` (no wheel for torch 2.9), via `torch.segment_reduce` on CSR offsets.
  PTv3 calls it with non-empty segments only (pooling clusters, per-cloud means).
* `DropPath`: `timm.models.layers.DropPath` (timm 0.9), copied to avoid the dependency.
"""

import torch
from torch import nn


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    return torch.segment_reduce(src, reduce, offsets=indptr, axis=0)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
