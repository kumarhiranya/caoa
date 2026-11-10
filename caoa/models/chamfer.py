"""Chamfer distances as defined by ODGNet's `extensions/chamfer_dist` (pure PyTorch).

`dist1` / `dist2` are squared nearest-neighbour distances (xyz1 -> xyz2 and
xyz2 -> xyz1), as the CUDA `chamfer.forward` returns them. Like the CUDA
backward, gradients flow only through each point's nearest neighbour.

* L2: mean(dist1) + mean(dist2)
* L1: (mean(sqrt(dist1)) + mean(sqrt(dist2))) / 2
`ignore_zeros` drops all-zero points, only for batch size 1 (as ODGNet's).
"""

import torch


@torch.no_grad()
def _nearest(src: torch.Tensor, dst: torch.Tensor, max_elements: int = 2 ** 28) -> torch.Tensor:
    """Index of each src point's nearest dst point.

    The search uses |s|^2 + |d|^2 - 2 s.d (matrix multiply), chunked over src; callers then
    recompute the exact distance of the chosen pair, so values differ from an exhaustive exact
    search only when two neighbours are equidistant to within float32 rounding.
    """
    B, N, _ = src.shape
    chunk = max(1, max_elements // (B * dst.shape[1]))
    dst_sq = (dst * dst).sum(-1)[:, None, :]
    out = []
    for s in range(0, N, chunk):
        part = src[:, s : s + chunk]
        d = torch.baddbmm(dst_sq + (part * part).sum(-1)[..., None], part, dst.transpose(1, 2), alpha=-2.0)
        out.append(d.argmin(dim=2))
    return torch.cat(out, dim=1)


def _gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return points.gather(1, idx[..., None].expand(*idx.shape, points.shape[-1]))


def nn_squared_distances(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """(B, N, 3), (B, M, 3) -> squared NN distances (B, N), (B, M)."""
    xyz1, xyz2 = xyz1.float(), xyz2.float()
    dist1 = ((xyz1 - _gather(xyz2, _nearest(xyz1, xyz2))) ** 2).sum(-1)
    dist2 = ((xyz2 - _gather(xyz1, _nearest(xyz2, xyz1))) ** 2).sum(-1)
    return dist1, dist2


def _drop_zeros(xyz1, xyz2, ignore_zeros):
    if xyz1.size(0) == 1 and ignore_zeros:
        xyz1 = xyz1[torch.sum(xyz1, dim=2).ne(0)].unsqueeze(0)
        xyz2 = xyz2[torch.sum(xyz2, dim=2).ne(0)].unsqueeze(0)
    return xyz1, xyz2


class ChamferDistanceL2(torch.nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        dist1, dist2 = nn_squared_distances(*_drop_zeros(xyz1, xyz2, self.ignore_zeros))
        return torch.mean(dist1) + torch.mean(dist2)


class ChamferDistanceL1(torch.nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        dist1, dist2 = nn_squared_distances(*_drop_zeros(xyz1, xyz2, self.ignore_zeros))
        return (torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))) / 2
