"""CAD retrieval: embed the database and the queries, search, and score (P11d).

Search modes (`category`):
* "same" (default, user 2026-11-03): only CADs of the query's ShapeNet synset; for now the GT CAD's synset (the
  GT-instance setting), a predicted one in P11;
* "all": every database CAD; the retrieval then also decides the category.

Metrics, per query, against its GT CAD (always in the database):
* `recall@k`: the GT CAD among the top k (exact model);
* `rank`: the GT CAD's rank among the allowed CADs (1 = top), median and mean;
* `category_acc`: the top-1 CAD's synset is the GT's (always 1 in "same" mode);
* `cd_l1` / `cd_l2` (x 1000): Chamfer distance between the top-1 CAD and the GT CAD, both in their own bounding-box frame
  (centred, longest side 1), 2,048 points each: L1 = (mean d(a,b) + mean d(b,a)) / 2 with Euclidean d, L2 = mean d^2 +
  mean d^2 (the CDL1 / CDL2 conventions of the completion metrics). Shape similarity, which counts near-duplicates
  of the GT model as good retrievals; `random_*` is the same for a random CAD of the GT synset (the chance level).
"""

import numpy as np
import torch

from caoa.eval.scan2cad_benchmark import category_name
from caoa.geometry import yaw_matrix
from caoa.models.chamfer import nn_squared_distances
from caoa.retrieval.data import normalize_bbox

CATEGORY_MODES = ("same", "all")
RECALL_K = (1, 5, 10, 50)


@torch.no_grad()
def embed_points(encoder, points, device, batch_size: int = 256) -> torch.Tensor:
    """points: (N, P, 3) array or tensor -> (N, D) embeddings on `device`."""
    encoder.eval()
    out = []
    for start in range(0, len(points), batch_size):
        out.append(encoder(torch.as_tensor(points[start : start + batch_size], dtype=torch.float32).to(device)))
    return torch.cat(out)


@torch.no_grad()
def embed_queries(encoder, queries: np.ndarray, device, yaws: int = 1, batch_size: int = 256) -> torch.Tensor:
    """queries: (Q, P, 3) bbox-normalised world-frame queries. With `yaws` > 1, the mean embedding over that many yaw
    rotations (each re-normalised), renormalised: test-time augmentation for the unknown yaw."""
    total = 0
    for i in range(yaws):
        R = yaw_matrix(2 * np.pi * i / yaws)
        rotated = np.stack([normalize_bbox(q.astype(np.float64) @ R.T) for q in queries]).astype(np.float32) if i else queries
        total = total + embed_points(encoder, rotated, device, batch_size)
    return torch.nn.functional.normalize(total, dim=1)


def allowed_mask(query_catids, db_catids, category: str, device) -> torch.Tensor | None:
    if category == "all":
        return None
    if category != "same":
        raise ValueError(f"category mode {category!r}; one of {CATEGORY_MODES}")
    synsets = sorted(set(db_catids) | set(query_catids))
    code = {s: i for i, s in enumerate(synsets)}
    q = torch.tensor([code[c] for c in query_catids], device=device)
    d = torch.tensor([code[c] for c in db_catids], device=device)
    return q[:, None] == d[None, :]


@torch.no_grad()
def search(q_emb: torch.Tensor, db_emb: torch.Tensor, mask: torch.Tensor | None, targets: torch.Tensor, k: int = max(RECALL_K)):
    """Top-k database indices (Q, k) and the GT CAD's rank (Q,) among the allowed CADs."""
    sim = q_emb @ db_emb.T
    if mask is not None:
        sim = sim.masked_fill(~mask, float("-inf"))
    gt_sim = sim.gather(1, targets[:, None])
    rank = (sim > gt_sim).sum(1) + 1
    return sim.topk(k, dim=1).indices, rank


@torch.no_grad()
def chamfer(a: torch.Tensor, b: torch.Tensor, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(N, P, 3) x (N, P, 3) -> CD-L1, CD-L2 (x 1000) per pair (conventions in the module docstring)."""
    l1, l2 = [], []
    for s in range(0, len(a), batch_size):
        d1, d2 = nn_squared_distances(a[s : s + batch_size], b[s : s + batch_size])
        l1.append((d1.sqrt().mean(1) + d2.sqrt().mean(1)) / 2)
        l2.append(d1.mean(1) + d2.mean(1))
    return 1000 * torch.cat(l1).cpu().numpy(), 1000 * torch.cat(l2).cpu().numpy()


def retrieval_metrics(top: torch.Tensor, rank: torch.Tensor, targets: torch.Tensor, db_points: torch.Tensor,
                      db_catids: np.ndarray, mask: torch.Tensor | None, seed: int = 0) -> dict:
    """`db_points`: (N, P, 3) database points on the GPU; `mask`: the same-category mask, for the random baseline."""
    device = db_points.device
    targets = targets.to(device)
    top1 = top[:, 0]
    cd_l1, cd_l2 = chamfer(db_points[top1], db_points[targets])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if mask is None:  # random baseline: always a random CAD of the GT synset
        mask = allowed_mask(db_catids[targets.cpu().numpy()], db_catids, "same", device)
    random_pick = torch.multinomial(mask.float().cpu(), 1, generator=gen)[:, 0].to(device)
    rnd_l1, rnd_l2 = chamfer(db_points[random_pick], db_points[targets])

    gt_cat = db_catids[targets.cpu().numpy()]
    cats = np.array([category_name(c) for c in gt_cat])
    hit = {k: (top[:, :k] == targets[:, None]).any(1).cpu().numpy() for k in RECALL_K}
    cat_ok = db_catids[top1.cpu().numpy()] == gt_cat
    rank = rank.cpu().numpy()

    def summary(sel):
        return {
            "n": int(sel.sum()),
            **{f"recall@{k}": float(hit[k][sel].mean()) for k in RECALL_K},
            "rank_median": float(np.median(rank[sel])),
            "rank_mean": float(rank[sel].mean()),
            "category_acc": float(cat_ok[sel].mean()),
            "cd_l1": float(cd_l1[sel].mean()),
            "cd_l2": float(cd_l2[sel].mean()),
            "random_cd_l1": float(rnd_l1[sel].mean()),
            "random_cd_l2": float(rnd_l2[sel].mean()),
        }

    per_class = {c: summary(cats == c) for c in sorted(set(cats))}
    out = summary(np.ones(len(cats), dtype=bool))
    out["class_mean"] = {k: float(np.mean([v[k] for v in per_class.values()])) for k in ("recall@1", "recall@5", "cd_l1", "random_cd_l1")}
    out["per_class"] = per_class
    return out
