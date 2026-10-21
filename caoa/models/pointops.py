"""Pure-PyTorch versions of the `pointnet2_ops` CUDA ops that ODGNet uses.

Semantics follow `ODGNet/extensions/pointnet2_ops_lib` (the build the legacy runs used):

* `furthest_point_sample` starts at index 0 and, like the CUDA kernel, never
  selects a point whose squared norm is <= 1e-3 (`if (mag <= 1e-3) continue;`).
  Exact ties (duplicated points) may resolve to a different index than the CUDA
  reduction, which yields the same coordinates.
* `gather_operation` / `grouping_operation` index features along the point axis
  and are differentiable w.r.t. the features, like the CUDA autograd functions.
"""

import torch


@torch.no_grad()
def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """xyz: (B, N, 3) -> indices (B, npoint), int32."""
    B, N, _ = xyz.shape
    xyz = xyz.float()
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    if npoint <= 1:
        return idx.int()
    selectable = (xyz * xyz).sum(-1) > 1e-3
    temp = torch.full((B, N), 1e10, device=xyz.device)
    rows = torch.arange(B, device=xyz.device)
    old = torch.zeros(B, dtype=torch.long, device=xyz.device)
    for j in range(1, npoint):
        last = xyz[rows, old]  # (B, 3)
        diff = xyz - last[:, None, :]
        d = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1] + diff[..., 2] * diff[..., 2]
        temp = torch.where(selectable, torch.minimum(temp, d), temp)
        best = torch.where(selectable, temp, torch.full_like(temp, -1.0))
        old = best.argmax(dim=1)  # all -1 (nothing selectable) -> index 0, as in CUDA
        idx[:, j] = old
    return idx.int()


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features (B, C, N), idx (B, M) -> (B, C, M)."""
    B, C, _ = features.shape
    return features.gather(2, idx.long()[:, None, :].expand(B, C, idx.shape[1]))


def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features (B, C, N), idx (B, M, K) -> (B, C, M, K)."""
    B, C, _ = features.shape
    _, M, K = idx.shape
    flat = idx.long().reshape(B, 1, M * K).expand(B, C, M * K)
    return features.gather(2, flat).reshape(B, C, M, K)
