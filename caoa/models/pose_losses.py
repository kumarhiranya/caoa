"""Pose losses beyond legacy's per-component L1 (P10c)."""

import torch

from caoa.models.chamfer import nn_squared_distances


def box_points(pose: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    """(B, 15) box poses (t, row-major R, s) applied to (B, M, 3) unit-box points: R (s * u) + t."""
    R = pose[:, 3:12].reshape(-1, 3, 3)
    return (unit * pose[:, None, 12:]) @ R.transpose(1, 2) + pose[:, None, :3]


def chamfer_pose_loss(pred: torch.Tensor, target: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    """P10c (user, 2026-10-28): Chamfer (legacy chamfer_3DDist form, mean(d1) + mean(d2) of squared distances) between
    the CAD placed by the predicted and by the GT box pose."""
    d1, d2 = nn_squared_distances(box_points(pred, unit), box_points(target, unit))
    return d1.mean() + d2.mean()
