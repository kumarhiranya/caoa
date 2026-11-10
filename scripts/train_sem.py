"""Train the SEM, the PTv3 symmetry classifier (`caoa.models.sem`).

PTv3 classifier from scratch, BCE on the softmax probabilities against the one-hot label (index 0 = not symmetric,
1 = any rotational symmetry), AdamW lr 1e-4 / wd 1e-3, cosine annealing to 1e-6 over 10k steps, batch 64, shuffled,
`drop_last`, seed 777; validation every 500 steps (accuracy and macro-F1, on the shuffled val loader with `drop_last`
and on the whole val set).
Data: one sample per object–CAD pair (the pair's CAD and its Scan2CAD `sym`; `pairs.SEM_RELABELLED`). Input: the CAD
in its own bounding-box frame (centred, longest side 1), 5,000 uniform surface points tiled to 8,192,
`grid = coord // 1 cm`, as the OCAM and the retrieval feed the SEM (`scripts/build_sem_embeddings.py`).
After training, the BatchNorm running statistics are recomputed on training data in eval mode
(`caoa.models.sem.recalibrate_batchnorm`; `--bn-recalibration none` skips it): without it the model collapses in eval
mode, because its statistics were accumulated with drop-path on.

Outputs in `outputs/<name>/`: `model.pth` (plain state_dict, before recalibration), `model_recalibrated.pth` (the SEM
the OCAM uses), `train_log.json`, `results.json` (`final`: the last validation; `final_recalibrated`). `--eval-only`
evaluates (and recalibrates) a checkpoint, writing `results_eval_only.json`.

    uv run python scripts/train_sem.py [--name sem] [--max-steps 10000]
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

from caoa.data.sampling import sample_points
from caoa.data.pairs import SEM_RELABELLED, load_pairs
from caoa.data.shapenet import load_cad_mesh, sample_uniform
from caoa.models.sem import BN_RECALIBRATION, SEMClassifier, make_batch, recalibrate_batchnorm
from caoa.paths import REPO_ROOT, load_paths

CAD_POINTS, NUM_POINTS = 5000, 8192
RECALIBRATION_BATCHES = 60  # x 64 training samples
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
    """The shuffled val loader (drop_last) plus the whole val set (batch 64, no drop), and inference time."""
    model.eval()
    preds, labels = [], []
    for clouds, y in val_loader:
        preds.append(model(make_batch(clouds, device))["probs"].argmax(1).cpu())
        labels.append(y)
    p, y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    all_p, times = [], []
    for s in range(0, len(val_ds), 64):
        clouds = [val_ds[i][0] for i in range(s, min(s + 64, len(val_ds)))]
        torch.cuda.synchronize()
        t0 = time.time()
        all_p.append(model(make_batch(clouds, device))["probs"].argmax(1).cpu())
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000 / len(clouds))
    all_p = torch.cat(all_p).numpy()
    model.train()
    return {
        "accuracy": float((p == y).mean()), "macro_f1": float(f1_score(y, p, average="macro")), "n": len(y),
        "accuracy_all": float((all_p == val_ds.labels).mean()), "macro_f1_all": float(f1_score(val_ds.labels, all_p, average="macro")),
        "n_all": len(val_ds), "ms_per_cad": float(np.mean(times)),
    }


def recalibrate(model, train_ds, device, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(len(train_ds))[: 64 * RECALIBRATION_BATCHES]
    batches = (make_batch([train_ds[i][0] for i in idx[s : s + 64]], device) for s in range(0, len(idx), 64))
    recalibrate_batchnorm(model, batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="sem")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--val-freq", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bn-recalibration", choices=BN_RECALIBRATION, default="train")
    parser.add_argument("--eval-only", default=None, help="checkpoint (plain state_dict) to evaluate instead of training")
    args = parser.parse_args()
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    paths = load_paths()
    device = torch.device("cuda")
    out_dir = REPO_ROOT / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = {s: load_pairs(paths.cache, s, SEM_RELABELLED) for s in ("train", "val")}
    keys = sorted({f"{p.catid_cad}_{p.id_cad}" for v in pairs.values() for p in v})
    with Pool(20, initializer=init, initargs=(paths.shapenet,)) as pool:
        cad_points = dict(pool.imap_unordered(cad_bbox_points, keys, chunksize=8))
    train_ds, val_ds = SymmetrySamples(pairs["train"], cad_points), SymmetrySamples(pairs["val"], cad_points)
    print(f"train {len(train_ds)} (symmetric {train_ds.labels.sum()}), "
          f"val {len(val_ds)} (symmetric {val_ds.labels.sum()}); {len(keys)} CADs", flush=True)
    train_loader, val_loader = loader(train_ds, 64, args.workers), loader(val_ds, 64, 4)

    model = SEMClassifier().to(device)
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=device))
        results = {"final": evaluate(model, val_loader, val_ds, device), "checkpoint": args.eval_only}
        if args.bn_recalibration == "train":
            recalibrate(model, train_ds, device)
            results["final_recalibrated"] = evaluate(model, val_loader, val_ds, device)
            torch.save(model.state_dict(), out_dir / "model_recalibrated.pth")
        print(json.dumps({k: v for k, v in results.items() if k.startswith("final")}, indent=1))
        (out_dir / "results_eval_only.json").write_text(json.dumps(results, indent=1))
        return
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
            probs = model(make_batch(clouds, device))["probs"]
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
    results = {"final": log["val"][-1], "bn_recalibration": args.bn_recalibration}
    torch.save(model.state_dict(), out_dir / "model.pth")
    if args.bn_recalibration == "train":
        recalibrate(model, train_ds, device)
        results["final_recalibrated"] = evaluate(model, val_loader, val_ds, device)
        print(f"[val] recalibrated  acc {results['final_recalibrated']['accuracy']:.4f} F1 {results['final_recalibrated']['macro_f1']:.4f}", flush=True)
        torch.save(model.state_dict(), out_dir / "model_recalibrated.pth")
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
