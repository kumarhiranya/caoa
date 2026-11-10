"""SEM: the symmetry embedding module, a PTv3 classifier of CAD symmetry.

* Backbone: the vendored PTv3 (`caoa.models.ptv3`) with the `cls-ptv3-v1m1-0-shapenet_v2_mini` configuration, except
  `enable_flash`: flash-attn is not used; the non-flash path pads the same patches, so only the attention precision
  differs (fp16 with flash).
* Head: mean over each cloud's points (`segment_csr`), then an MLP 128-256-128-2 with BN, ReLU, dropout 0.5 and a
  softmax. The model outputs probabilities in train and eval mode (trained with `BCELoss` on them). `emb` is the
  pooled 128-d feature, the SEM embedding the OCAM consumes (`scripts/build_sem_embeddings.py`).
* Input: `coord` = `feat` = the points, clouds concatenated with CSR `offset`: the CAD (5,000 uniform surface points,
  tiled to 8,192) in its own bounding-box frame (longest side 1), `grid_coord = coord // voxel_size`.
"""

import numpy as np
import torch
from torch import nn

from caoa.models.ptv3._compat import segment_csr
from caoa.models.ptv3.model import Point, PointTransformerV3

EMBED_DIM = 128
VOXEL_SIZE = 0.01
CLASSES = ("__SYM_NONE", "__SYM")  # index 0 = not symmetric, 1 = any rotational symmetry
BACKBONE = dict(
    in_channels=3,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2),
    enc_depths=(2, 4, 2),
    enc_channels=(32, 64, 128),
    enc_num_head=(2, 4, 8),
    enc_patch_size=(1024, 1024, 1024),
    dec_depths=(2, 2),
    dec_channels=(64, 64),
    dec_num_head=(4, 4),
    dec_patch_size=(1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=False,  # flash-attn not used
    upcast_attention=False,
    upcast_softmax=False,
    cls_mode=True,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
)


class SEMClassifier(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), embed_dim: int = EMBED_DIM, backbone: dict = BACKBONE):
        super().__init__()
        self.backbone = PointTransformerV3(**backbone)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, input_dict: dict) -> dict:
        point = self.backbone(Point(input_dict))
        emb = segment_csr(point.feat, nn.functional.pad(point.offset, (1, 0)), reduce="mean")
        return {"probs": self.cls_head(emb), "emb": emb}


def make_batch(clouds: list[np.ndarray], device, voxel_size: float = VOXEL_SIZE) -> dict:
    """A batch of point clouds (each (N, 3), CAD bounding-box frame) as the PTv3 input dict."""
    grids = [torch.div(torch.as_tensor(c, dtype=torch.float32), voxel_size, rounding_mode="floor").long() for c in clouds]
    coord = torch.cat([torch.as_tensor(c, dtype=torch.float32) for c in clouds]).to(device)
    return {
        "coord": coord,
        "feat": coord.clone(),
        "grid_coord": torch.cat(grids).to(device),
        "offset": torch.as_tensor(np.cumsum([len(c) for c in clouds]), device=device),
    }


CAD_POINTS, INPUT_POINTS = 5000, 8192


def cad_bbox_points(mesh, rng) -> np.ndarray:
    """The SEM's input for a CAD mesh: 5,000 uniform surface points in its own bounding-box frame (centred, longest side
    1), tiled to 8,192."""
    from caoa.data.sampling import sample_points
    from caoa.data.shapenet import sample_uniform

    pts = sample_uniform(mesh, CAD_POINTS, rng=rng)
    lo, hi = pts.min(0), pts.max(0)
    return sample_points((pts - (lo + hi) / 2) / (hi - lo).max(), INPUT_POINTS)


def sem_checkpoint(sym_cfg: dict):
    """The SEM weights behind a pose config's `sym_embedding`: its `checkpoint` (a plain state_dict), relative to the
    repository root or absolute."""
    from caoa.paths import REPO_ROOT

    return REPO_ROOT / sym_cfg["checkpoint"]


def load_sem(model: SEMClassifier, path) -> None:
    """SEM checkpoints are a plain `state_dict`."""
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)


BN_RECALIBRATION = ("none", "train")


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, batches) -> None:
    """Recompute every BatchNorm's running statistics as the plain average over `batches` (input dicts), with everything
    else in eval mode (no dropout, no drop-path, fixed serialization orders), so they match the activations eval mode
    produces. The statistics kept from training were accumulated with drop-path on, and with them the SEM collapses in
    eval mode (66% accuracy instead of 98%)."""
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
        m.train()
    for b in batches:
        model(b)
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    model.eval()
