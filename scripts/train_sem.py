"""Retrain SEM, the PTv3 symmetry classifier (P9; user decision 2026-11-03: CAD bounding-box frame inputs).

Legacy recipe (`train_sym_ptv3.py` + `training_utils.train_sym_ptv3`, run `S2Csym_emb128_bce_..._cosineannealing`):
`PT3ClassifierSoftmax` from scratch, BCE on the softmax probabilities against the one-hot label (index 0 = not
symmetric, 1 = any rotational symmetry), AdamW lr 1e-4 / wd 1e-3, cosine annealing to lr / 100 over 10k steps,
batch 64, shuffled, `drop_last`, seed 777; `eval_sym` over the whole val loader (shuffled, batch 64, `drop_last`) every
500 steps, logged as accuracy and macro-F1.
Data: one sample per pose pair (the pair's GT CAD and its Scan2CAD `sym`), V2 class rule, as the logged run
(docs/results/P9.md: 2,067 val pairs -> 2,048 evaluated). Input (the change): the CAD in its own bounding-box frame
(centred, longest side 1), 5,000 uniform surface points tiled to 8,192, `grid = coord // 1 cm`, the input OCAM and the
retrieval feed SEM (`scripts/build_sem_embeddings.py`). Legacy trained on the CAD in its GT box frame.
Outputs in `outputs/<name>/`: `model.pth` (plain state_dict, the legacy checkpoint format), `train_log.json`, `results.json`.

    uv run python scripts/train_sem.py [--name sem_cad_bbox] [--max-steps 10000]
"""

import argparse
import json
import random
import time
from multiprocessing import Pool

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from caoa.data.legacy_sampling import sample_points
from caoa.data.pairs import load_pairs
from caoa.data.shapenet import load_cad_mesh, sample_uniform
from caoa.models.sem import SEMClassifier, make_batch
from caoa.paths import REPO_ROOT, load_paths

CAD_POINTS, NUM_POINTS = 5000, 8192
SHAPENET = None


def init(root):
    global SHAPENET
    SHAPENET = root


def cad_bbox_points(key: str):
    rng = np.random.default_rng(int.from_bytes(key.encode()[-8:], "little"))
    pts = sample_uniform(load_cad_mesh(SHAPENET, *key.split("_", 1)), CAD_POINTS, rng=rng)
    lo, hi = pts.min(0), pts.max(0)
    return key, ((pts - (lo + hi) / 2) / (hi - lo).max()).astype(np.float32)


class SymmetrySamples(Dataset):
    def __init__(self, pairs, cad_points):
        self.keys = [f"{p.catid_cad}_{p.id_cad}" for p in pairs]
        self.labels = np.array([0 if p.sym == "__SYM_NONE" else 1 for p in pairs])
        self.cad_points = cad_points

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return sample_points(self.cad_points[self.keys[i]], NUM_POINTS), int(self.labels[i])


def collate(items):
    return [c for c, _ in items], torch.tensor([y for _, y in items])


def loader(ds, batch_size, workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers, collate_fn=collate,
                      persistent_workers=True)


@torch.no_grad()
def evaluate(model, val_loader, val_ds, device):
    """Legacy `eval_sym` (shuffled, drop_last) plus the whole val set (batch 64, no drop), and inference time."""
    model.eval()
    preds, labels = [], []
    for clouds, y in val_loader:
        preds.append(model(make_batch(clouds, device, grid="train"))["probs"].argmax(1).cpu())
        labels.append(y)
    p, y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    all_p, times = [], []
    for s in range(0, len(val_ds), 64):
        clouds = [val_ds[i][0] for i in range(s, min(s + 64, len(val_ds)))]
        torch.cuda.synchronize()
        t0 = time.time()
        all_p.append(model(make_batch(clouds, device, grid="train"))["probs"].argmax(1).cpu())
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000 / len(clouds))
    all_p = torch.cat(all_p).numpy()
    model.train()
    return {
        "accuracy": float((p == y).mean()), "macro_f1": float(f1_score(y, p, average="macro")), "n": len(y),
        "accuracy_all": float((all_p == val_ds.labels).mean()), "macro_f1_all": float(f1_score(val_ds.labels, all_p, average="macro")),
        "n_all": len(val_ds), "ms_per_cad": float(np.mean(times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="sem_cad_bbox")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--val-freq", type=int, default=500)
    parser.add_argument("--class-rule", default="v2")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    paths = load_paths()
    device = torch.device("cuda")
    out_dir = REPO_ROOT / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = {s: load_pairs(paths.cache, s, class_rule=args.class_rule) for s in ("train", "val")}
    keys = sorted({f"{p.catid_cad}_{p.id_cad}" for v in pairs.values() for p in v})
    with Pool(20, initializer=init, initargs=(paths.shapenet,)) as pool:
        cad_points = dict(pool.imap_unordered(cad_bbox_points, keys, chunksize=8))
    train_ds, val_ds = SymmetrySamples(pairs["train"], cad_points), SymmetrySamples(pairs["val"], cad_points)
    print(f"class rule {args.class_rule}: train {len(train_ds)} (symmetric {train_ds.labels.sum()}), "
          f"val {len(val_ds)} (symmetric {val_ds.labels.sum()}); {len(keys)} CADs", flush=True)
    train_loader, val_loader = loader(train_ds, 64, args.workers), loader(val_ds, 64, 4)

    model = SEMClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=1e-6)
    bce = torch.nn.BCELoss()
    log = {"step": [], "loss": [], "val": []}
    step, t0, losses = 0, time.time(), []
    model.train()
    while step < args.max_steps:
        for clouds, y in train_loader:
            if step >= args.max_steps:
                break
            probs = model(make_batch(clouds, device, grid="train"))["probs"]
            loss = bce(probs, torch.nn.functional.one_hot(y, 2).float().to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 150 == 0:
                print(f"step {step}  loss {np.mean(losses):.4f}  {(time.time() - t0) / step:.3f} s/step", flush=True)
                losses = []
            if step % args.val_freq == 0:
                res = evaluate(model, val_loader, val_ds, device)
                log["step"].append(step)
                log["val"].append(res)
                print(f"[val] step {step}  acc {res['accuracy']:.4f} F1 {res['macro_f1']:.4f} (n {res['n']})  "
                      f"all: acc {res['accuracy_all']:.4f} F1 {res['macro_f1_all']:.4f}", flush=True)
                (out_dir / "train_log.json").write_text(json.dumps(log))
    torch.save(model.state_dict(), out_dir / "model.pth")
    (out_dir / "results.json").write_text(json.dumps({"final": log["val"][-1], "class_rule": args.class_rule}, indent=1))


if __name__ == "__main__":
    main()
