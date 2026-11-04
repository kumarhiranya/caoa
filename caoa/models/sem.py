"""SEM: the PTv3 symmetry classifier (P9), as legacy `train_sym_ptv3.py` built it.

* Backbone: the vendored PTv3 (`caoa.models.ptv3`) with the legacy `cls-ptv3-v1m1-0-shapenet_v2_mini` config
  (= the run's `model_config.yaml`), except `enable_flash`: flash-attn is not installed, and the non-flash path
  pads the same patches, so only the attention precision differs (fp16 in flash).
* Head: legacy `PT3ClassifierSoftmax` (`PointTransformerV3/model_utils.py`): mean over each cloud's points
  (`segment_csr`), then MLP 128-256-128-2 with BN, ReLU, dropout 0.5 and a softmax. The model outputs
  probabilities in train and eval mode (legacy trains it with `BCELoss` on them). `emb` is the pooled 128-d
  feature, the SEM embedding that OCAM consumes (`data/s2c_symmetry/<run>/embeddings.npy`).
* Input: `coord` = `feat` = the points, clouds concatenated with CSR `offset`. Two legacy pipelines differ:
  - "train" (`S2CCompletionSymmetry`, data_source cad, which trained the checkpoint): the CAD (5,000 uniform
    points, tiled to 8,192) in its GT box frame (longest side 1), `grid_coord = coord // voxel_size`;
  - "export" (`S2CSymmetry`, which `build_dataset_sym.py` used for `embeddings.npy`): raw ShapeNet
    coordinates, `grid_coord = (coord - coord.min(0)) // voxel_size`. The checkpoint collapses on these inputs
    (every CAD predicted not symmetric; docs/results/P9.md).
"""

import numpy as np
import torch
from torch import nn

from caoa.models.ptv3._compat import segment_csr
from caoa.models.ptv3.model import Point, PointTransformerV3

EMBED_DIM = 128
VOXEL_SIZE = 0.01
CLASSES = ("__SYM_NONE", "__SYM")  # legacy task "symbinary": index 0 = not symmetric, 1 = any rotational symmetry
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
    enable_flash=False,  # legacy: True
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
            nn.Softmax(dim=1),  # legacy nn.Softmax() picks dim 1 for 2-D input
        )

    def forward(self, input_dict: dict) -> dict:
        point = self.backbone(Point(input_dict))
        emb = segment_csr(point.feat, nn.functional.pad(point.offset, (1, 0)), reduce="mean")
        return {"probs": self.cls_head(emb), "emb": emb}


GRID_MODES = ("train", "export")


def make_batch(clouds: list[np.ndarray], device, voxel_size: float = VOXEL_SIZE, grid: str = "export") -> dict:
    """Legacy `__getitem__` (minus its random point shuffle) + `point_collate_fn`; `grid` picks the pipeline (above)."""
    if grid not in GRID_MODES:
        raise ValueError(f"grid {grid!r}; one of {GRID_MODES}")
    if grid == "export":  # float64, as legacy (numpy input)
        grids = [torch.div(c - c.min(0)[0], voxel_size, rounding_mode="floor").long()
                 for c in (torch.as_tensor(np.asarray(c, dtype=np.float64)) for c in clouds)]
    else:  # float32 tensor, as legacy
        grids = [torch.div(torch.as_tensor(c, dtype=torch.float32), voxel_size, rounding_mode="floor").long() for c in clouds]
    coord = torch.cat([torch.as_tensor(c, dtype=torch.float32) for c in clouds]).to(device)
    return {
        "coord": coord,
        "feat": coord.clone(),
        "grid_coord": torch.cat(grids).to(device),
        "offset": torch.as_tensor(np.cumsum([len(c) for c in clouds]), device=device),
    }


def load_legacy_sem(model: SEMClassifier, path) -> None:
    """Legacy checkpoints are a plain `state_dict` (`torch.save(model.state_dict())`)."""
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)


BN_RECALIBRATION = ("none", "train")


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, batches) -> None:
    """Fix (P9, user 2026-11-03; switch `bn_recalibration: train`): recompute every BatchNorm's running statistics as the
    plain average over `batches` (input dicts), with everything else in eval mode (no dropout, no drop-path, fixed
    serialization orders), so they match the activations eval mode produces. Legacy keeps the statistics from training,
    which were accumulated with drop-path on; a SEM trained on CAD-frame inputs then fails in eval mode (docs/results/P9.md)."""
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
