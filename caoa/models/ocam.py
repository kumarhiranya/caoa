"""OCAM pose network: the legacy `MinkowskiFCNNP_v3` (object points only), ported to spconv.

Layer layout, widths, activations and initialisation follow the legacy model
(`utils/model_utils.py`); the ME semantics it depends on are in `sparse.py`.
Outputs translation (3), rotation as a tanh-squashed 3x3 matrix (9), scale (3).
"""

import math

import spconv.pytorch as spconv
import torch
from torch import nn

from caoa.models import sparse


def mlp_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_ch, out_ch, bias=False), nn.BatchNorm1d(out_ch), nn.LeakyReLU())


def _bn_act(ch: int) -> list[nn.Module]:
    return [nn.BatchNorm1d(ch), nn.LeakyReLU()]


def conv_block(in_ch: int, out_ch: int, stride: int) -> spconv.SparseSequential:
    conv = spconv.SubMConv3d(in_ch, out_ch, kernel_size=3, bias=False) if stride == 1 else sparse.strided_conv(in_ch, out_ch)
    return spconv.SparseSequential(conv, *_bn_act(out_ch))


def head(in_ch: int, hidden: int, out_ch: int, tanh: bool = False) -> nn.Sequential:
    layers = [mlp_block(in_ch, 512), nn.Dropout(0.5), mlp_block(512, hidden), nn.Linear(hidden, out_ch)]
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class OCAMObjectOnly(nn.Module):
    """Legacy "mnkpv3": completed (or raw) object points -> 9-DoF box pose."""

    def __init__(self, in_channels=3, embedding=1024, channels=(32, 48, 64, 96, 128), rotation_dims=9):
        super().__init__()
        c = channels
        self.mlp1 = mlp_block(in_channels, c[0])
        self.conv1 = conv_block(c[0], c[1], stride=1)
        self.conv2 = conv_block(c[1], c[2], stride=2)
        self.conv3 = conv_block(c[2], c[3], stride=2)
        self.conv4 = conv_block(c[3], c[4], stride=2)
        self.pool = sparse.strided_max_pool()
        self.conv5 = spconv.SparseSequential(
            conv_block(sum(c[1:]), embedding // 4, stride=2),
            conv_block(embedding // 4, embedding // 2, stride=2),
            conv_block(embedding // 2, embedding, stride=2),
        )
        self.translation = head(2 * embedding, 64, 3)
        self.rotation = head(2 * embedding, 256, rotation_dims, tanh=True)
        self.scale = head(2 * embedding, 64, 3)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d)):
                # ME.utils.kaiming_normal_(mode="fan_out"): fan_out = out_channels * kernel volume
                fan_out = m.out_channels * math.prod(m.kernel_size)
                nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        """coords: (N, 3) point coordinates divided by the voxel size; feats: (N, C); batch: (N,)."""
        point_feats = self.mlp1(feats)
        x, _ = sparse.voxelize(coords, point_feats, batch, batch_size)

        y1 = self.pool(self.conv1(x))  # stride 2
        y2 = self.pool(self.conv2(y1))  # stride 8
        y3 = self.pool(self.conv3(y2))  # stride 32
        y4 = self.pool(self.conv4(y3))  # stride 128

        multi_scale = torch.cat(
            [sparse.slice_to(y, x.indices, factor) for y, factor in ((y1, 2), (y2, 8), (y3, 32), (y4, 128))], dim=1
        )
        y = self.conv5(x.replace_feature(multi_scale))
        pooled = sparse.global_max_avg_pool(y)
        return torch.cat([self.translation(pooled), self.rotation(pooled), self.scale(pooled)], dim=1)
