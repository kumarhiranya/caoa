"""OCAM pose network: the legacy `MinkowskiFCNNP_v3` (object points only), ported to spconv.

Layer layout, widths, activations and initialisation follow the legacy model
(`utils/model_utils.py`); the ME semantics it depends on are in `sparse.py`.
Outputs translation (3), rotation as a tanh-squashed 3x3 matrix (9), scale (3).
With `ext_emb_size > 0` it is the legacy `MinkowskiFCNNP_v3_emb` ("mnkpv3_emb", P10a): an external
embedding (the SEM symmetry embedding of the CAD) is concatenated to the pooled feature before the heads.
`OCAMSiamese` (P10b) adds a CAD trunk, shared (paper) or separate (legacy RelPose).
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


def init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d)):
            # ME.utils.kaiming_normal_(mode="fan_out"): fan_out = out_channels * kernel volume
            fan_out = m.out_channels * math.prod(m.kernel_size)
            nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def check_ext_emb(ext_emb, ext_emb_size):
    if (ext_emb is None) != (ext_emb_size == 0):
        raise ValueError(f"model built with ext_emb_size={ext_emb_size}, got ext_emb={'None' if ext_emb is None else tuple(ext_emb.shape)}")


class OCAMEncoder(nn.Module):
    """The legacy `mnkpv3` trunk: points -> multi-scale sparse convs -> global max+avg pool (2 x embedding)."""

    def __init__(self, in_channels=3, embedding=1024, channels=(32, 48, 64, 96, 128)):
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

    def encode(self, coords: torch.Tensor, feats: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
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
        return sparse.global_max_avg_pool(y)


class OCAMObjectOnly(OCAMEncoder):
    """Legacy "mnkpv3" (and "mnkpv3_emb"): completed (or raw) object points [+ external embedding] -> 9-DoF box pose."""

    def __init__(self, in_channels=3, embedding=1024, channels=(32, 48, 64, 96, 128), rotation_dims=9, ext_emb_size=0):
        super().__init__(in_channels, embedding, channels)
        self.ext_emb_size = ext_emb_size
        feat = 2 * embedding + ext_emb_size
        self.translation = head(feat, 64, 3)
        self.rotation = head(feat, 256, rotation_dims, tanh=True)
        self.scale = head(feat, 64, 3)
        init_weights(self)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor, batch: torch.Tensor, batch_size: int,
                ext_emb: torch.Tensor | None = None) -> torch.Tensor:
        """ext_emb: (batch_size, ext_emb_size), required iff the model was built with ext_emb_size > 0."""
        check_ext_emb(ext_emb, self.ext_emb_size)
        pooled = self.encode(coords, feats, batch, batch_size)
        if ext_emb is not None:
            pooled = torch.cat([pooled, ext_emb], dim=1)
        return torch.cat([self.translation(pooled), self.rotation(pooled), self.scale(pooled)], dim=1)


def relpose_head(in_ch: int, hidden: tuple[int, ...], out_ch: int, tanh: bool = False) -> nn.Sequential:
    """Legacy `MinkowskiFCNNRelPose_v1` branch: MLP blocks with dropout (0.5) between them, then a linear layer."""
    layers, prev = [], in_ch
    for i, h in enumerate(hidden):
        if i:
            layers.append(nn.Dropout(0.5))
        layers.append(mlp_block(prev, h))
        prev = h
    layers.append(nn.Linear(prev, out_ch))
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class OCAMSiamese(nn.Module):
    """P10b: object points + CAD points -> 9-DoF box pose of the object.

    `shared=False` is the legacy `MinkowskiFCNNRelPose_v1` layout (separate object and CAD trunks), with its bug fixed:
    legacy's CAD trunk calls the object trunk's `conv4` (its own `conv4_cad` is never used). `shared=True` is the
    paper's Siamese network (one trunk for both). The heads are legacy RelPose's in both cases (input: object and CAD
    features, 4 x embedding, + an optional external embedding such as SEM's).
    """

    def __init__(self, in_channels=3, embedding=1024, channels=(32, 48, 64, 96, 128), rotation_dims=9, ext_emb_size=0,
                 shared=True):
        super().__init__()
        self.shared = shared
        self.obj = OCAMEncoder(in_channels, embedding, channels)
        self.cad = self.obj if shared else OCAMEncoder(in_channels, embedding, channels)
        self.ext_emb_size = ext_emb_size
        feat = 4 * embedding + ext_emb_size
        self.translation = relpose_head(feat, (1024, 512, 64), 3)
        self.rotation = relpose_head(feat, (1024, 512, 256, 128), rotation_dims, tanh=True)
        self.scale = relpose_head(feat, (1024, 512, 64), 3)
        init_weights(self)

    def forward(self, coords, feats, batch, batch_size, cad_coords, cad_feats, cad_batch, ext_emb=None) -> torch.Tensor:
        check_ext_emb(ext_emb, self.ext_emb_size)
        pooled = [self.obj.encode(coords, feats, batch, batch_size), self.cad.encode(cad_coords, cad_feats, cad_batch, batch_size)]
        if ext_emb is not None:
            pooled.append(ext_emb)
        pooled = torch.cat(pooled, dim=1)
        return torch.cat([self.translation(pooled), self.rotation(pooled), self.scale(pooled)], dim=1)
