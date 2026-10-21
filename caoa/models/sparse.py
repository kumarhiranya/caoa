"""MinkowskiEngine-style sparse ops on top of spconv.

The legacy OCAM network was written for MinkowskiEngine (ME). These helpers
reproduce the ME semantics it relied on:

* `voxelize`: floor continuous coordinates to voxels and average the point
  features in each voxel (ME TensorField.sparse(), UNWEIGHTED_AVERAGE).
* `strided_conv` / `strided_max_pool`: stride-2 ops whose output sites are the input
  sites floored to the new stride, as in ME. spconv's own stride-2 output
  also activates neighbouring sites, so those are dropped; the kernel windows
  are identical (offsets {-1, 0, 1} x input stride).
* `slice_to`: give each voxel of a fine tensor the features of the coarse
  voxel that contains it (ME slice()).

Tensors are spconv SparseConvTensors with indices (batch, x, y, z) in units of
their own stride; `stride` is tracked by the caller.
"""

import spconv.pytorch as spconv
import torch

# Coordinates are shifted by OFFSET voxels so they are non-negative. OFFSET is a
# multiple of every stride used, so flooring to a stride is unaffected.
OFFSET = 1024
SPATIAL_SIZE = 2 * OFFSET
_KEY_BASE = 1 << 16


def _keys(indices: torch.Tensor) -> torch.Tensor:
    idx = indices.long()
    return ((idx[:, 0] * _KEY_BASE + idx[:, 1]) * _KEY_BASE + idx[:, 2]) * _KEY_BASE + idx[:, 3]


def voxelize(coords: torch.Tensor, feats: torch.Tensor, batch: torch.Tensor, batch_size: int):
    """coords (N, 3) float, already divided by the voxel size; batch (N,) int.

    Returns the voxel tensor and, for every point, the index of its voxel.
    """
    grid = torch.floor(coords).long() + OFFSET
    if grid.min() < 0 or grid.max() >= SPATIAL_SIZE:
        raise ValueError(f"coordinates outside +-{OFFSET} voxels")
    indices = torch.cat([batch.long()[:, None], grid], dim=1)
    unique, point2voxel = torch.unique(indices, dim=0, return_inverse=True)
    voxel_feats = feats.new_zeros(len(unique), feats.shape[1]).index_reduce_(0, point2voxel, feats, "mean", include_self=False)
    tensor = spconv.SparseConvTensor(voxel_feats, unique.int(), [SPATIAL_SIZE] * 3, batch_size)
    return tensor, point2voxel


def _restrict(out: spconv.SparseConvTensor, inp: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
    wanted = inp.indices.clone()
    wanted[:, 1:] = torch.div(wanted[:, 1:], 2, rounding_mode="floor")
    keep = torch.isin(_keys(out.indices), _keys(wanted))
    return spconv.SparseConvTensor(out.features[keep], out.indices[keep], out.spatial_shape, out.batch_size)


class Strided(spconv.SparseModule):
    """Wraps a stride-2, kernel-3 spconv op so its output sites follow ME."""

    def __init__(self, op: torch.nn.Module):
        super().__init__()
        self.op = op

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return _restrict(self.op(x), x)


def strided_conv(in_ch: int, out_ch: int) -> Strided:
    return Strided(spconv.SparseConv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False))


class StridedMaxPool(spconv.SparseModule):
    """Kernel-3 stride-2 max pool with ME output sites.

    spconv's max pool returns max(0, max(inputs)); ME's returns max(inputs), and
    the legacy network pools LeakyReLU outputs, which can be negative. Shifting
    the features to be positive first makes spconv's result the true max.
    """

    def __init__(self):
        super().__init__()
        self.op = spconv.SparseMaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        shift = (-x.features.detach().min()).clamp(min=0) + 1
        y = _restrict(self.op(x.replace_feature(x.features + shift)), x)
        return y.replace_feature(y.features - shift)


def strided_max_pool() -> StridedMaxPool:
    return StridedMaxPool()


def slice_to(coarse: spconv.SparseConvTensor, fine_indices: torch.Tensor, factor: int) -> torch.Tensor:
    """Features of `coarse` at each fine site; `factor` = coarse stride / fine stride."""
    target = fine_indices.clone()
    target[:, 1:] = torch.div(target[:, 1:], factor, rounding_mode="floor")
    coarse_keys = _keys(coarse.indices)
    order = torch.argsort(coarse_keys)
    pos = torch.searchsorted(coarse_keys[order], _keys(target))
    return coarse.features[order[pos]]


def global_max_avg_pool(x: spconv.SparseConvTensor) -> torch.Tensor:
    """(batch_size, 2C): per-sample max then mean over all sites (ME GlobalMax + GlobalAvg)."""
    b = x.indices[:, 0].long()
    feats = x.features
    shape = (x.batch_size, feats.shape[1])
    index = b[:, None].expand_as(feats)
    mx = feats.new_zeros(shape).scatter_reduce(0, index, feats, "amax", include_self=False)
    avg = feats.new_zeros(shape).scatter_reduce(0, index, feats, "mean", include_self=False)
    return torch.cat([mx, avg], dim=1)
