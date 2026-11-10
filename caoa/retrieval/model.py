"""CAD retrieval model: a joint embedding of scanned objects and CADs (after Dahnert et al., "Joint Embedding
of 3D Scan and CAD Objects", ICCV 2019), with separate query and CAD encoders, since the domains differ.

Each encoder is the OCAM sparse-conv trunk (`OCAMEncoder`: multi-scale convs, global max + avg pool) followed by a
projection MLP; embeddings are L2-normalised, and similarity is their dot product. Trained with InfoNCE against the
GT CAD (`scripts/train_retrieval.py`).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from caoa.models.ocam import OCAMEncoder, init_weights


class PointEncoder(nn.Module):
    def __init__(self, embedding: int = 512, out_dim: int = 256, voxel_size: float = 0.01):
        super().__init__()
        self.voxel_size = voxel_size
        self.trunk = OCAMEncoder(embedding=embedding)
        self.head = nn.Sequential(
            nn.Linear(2 * embedding, 2 * embedding), nn.BatchNorm1d(2 * embedding), nn.ReLU(inplace=True), nn.Linear(2 * embedding, out_dim)
        )
        init_weights(self)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, 3), bbox-normalised (longest side 1) -> (B, out_dim) unit vectors."""
        b, n, _ = points.shape
        flat = points.reshape(-1, 3)
        batch = torch.arange(b, device=points.device, dtype=torch.int32).repeat_interleave(n)
        return F.normalize(self.head(self.trunk.encode(flat / self.voxel_size, flat, batch, b)), dim=1)


class CADRetrieval(nn.Module):
    def __init__(self, embedding: int = 512, out_dim: int = 256, voxel_size: float = 0.01, temperature: float = 0.07):
        super().__init__()
        self.query = PointEncoder(embedding, out_dim, voxel_size)
        self.cad = PointEncoder(embedding, out_dim, voxel_size)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))  # learnable, as CLIP

    def logit_scale(self) -> torch.Tensor:
        return self.log_scale.clamp(max=math.log(100.0)).exp()


def info_nce(q: torch.Tensor, c: torch.Tensor, keys: torch.Tensor, targets: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """q: (B, D) queries; c: (K, D) candidate CADs whose first B are the queries' GT CADs (`keys`: their database
    indices, `targets`: the queries' GT indices). Other copies of a query's GT CAD among the candidates are masked out."""
    logits = scale * q @ c.T
    duplicate = keys[None, :] == targets[:, None]
    duplicate[torch.arange(len(q)), torch.arange(len(q))] = False
    return F.cross_entropy(logits.masked_fill(duplicate, float("-inf")), torch.arange(len(q), device=q.device))
