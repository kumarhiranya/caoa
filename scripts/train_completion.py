"""Train / evaluate CAPCM (ODGNet UpTrans), reproducing legacy `ODGNet/tools/runner.py`.

    uv run python scripts/train_completion.py configs/completion/legacy_s2cc.yaml [--max-epoch N] [--test-only CKPT]

Legacy behaviour reproduced:
* epochs 0..max_epoch inclusive; train loader batch `total_bs`, shuffled, drop_last;
  val/test loaders batch 1;
* loss = UpTrans.get_loss (4 x CD-L1, x 1e3) + orth_weight x (dictionary orthogonality);
* AdamW; LambdaLR max(lr_decay ** (e / decay_step), lowest_decay) and the BN-momentum
  scheduler max(bn_momentum * bn_decay ** (e / decay_step), lowest_decay), both stepped
  with the finished epoch's index;
* every `val_freq` epochs (not epoch 0): `validate()` (class-averaged); ckpt-best by
  `consider_metric`; ckpt-last every epoch; per-epoch checkpoints for the last 10;
* at the end: `test()` aggregation on the test split, with per-category tables.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from caoa.data.completion_dataset import S2CCompletionDataset, load_completion_samples
from caoa.data.snindoor_dataset import SNIndoorDataset
from caoa.eval.completion_metrics import CompletionEvaluator, sample_metrics
from caoa.models.odgnet import UpTrans, load_legacy_checkpoint
from caoa.paths import REPO_ROOT, load_paths

LOWER_IS_BETTER = {"CDL1": True, "CDL2": True, "F-Score": False}


def build_dataset(spec: dict, paths, sem):
    name = spec["NAME"]
    if name == "ConcatDataset":
        return ConcatDataset([build_dataset(d, paths, sem) for d in spec["datasets"]])
    phase = "val" if spec["phase"] in ("val", "test") else "train"
    if name == "S2CCompletion":
        samples = load_completion_samples(paths.cache, spec["context_radius"], phase)
        return S2CCompletionDataset(samples, phase, sem, num_points=spec["num_points"], rotate_z=spec["rotate_z"],
                                    context_radius=spec["context_radius"], context_flag=tuple(spec.get("context_flag", (1.0, -1.0))))
    if name == "SNAugDataset":
        return SNIndoorDataset(paths.shapenet, phase, sem, num_points=spec["num_points"], difficulty=spec["difficulty"],
                               add_noise_to_target=spec["add_noise_to_target"], rotate_z=spec["rotate_z"],
                               data_downscale_factor=spec["data_downscale_factor"], filter_classes=spec["filter_classes"],
                               add_dummy_feat=spec["add_dummy_feat"])
    raise ValueError(f"unknown dataset {name}")


def forward(model, partial):
    return model(partial[:, :, :3], partial[:, :, 3:]) if partial.shape[-1] > 3 else model(partial)


@torch.no_grad()
def evaluate(model, loader, device):
    """Per-sample legacy metrics on the dense output (batch size 1)."""
    model.eval()
    ev = CompletionEvaluator()
    for catids, _, partial, gt, _, _ in loader:
        dense = forward(model, partial.to(device))[1]
        ev.update(catids[0], sample_metrics(dense, gt.to(device)))
    model.train()
    return ev.summary()


def set_bn_momentum(model, momentum):
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.momentum = momentum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--max-epoch", type=int, default=None, help="override, e.g. for a smoke test")
    parser.add_argument("--test-only", default=None, help="legacy or new checkpoint to evaluate")
    parser.add_argument("--val-freq", type=int, default=None, help="override, e.g. for a smoke test")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.max_epoch is not None:
        cfg["max_epoch"] = args.max_epoch
    if args.val_freq is not None:
        cfg["val_freq"] = args.val_freq
    out_dir = REPO_ROOT / "outputs" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    seed = cfg.get("seed", 2023)  # legacy --seed default
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    paths = load_paths()
    sem = np.load(paths.legacy_repo / "data/ShapeNetSemantic/classification_ptv3/semantic_embeddings.npy", allow_pickle=True).item()

    model = UpTrans.from_config(cfg["model"]).to(device)
    test_set = build_dataset(cfg["dataset"]["test"], paths, sem)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 8))

    if not args.test_only:
        train(model, cfg, paths, sem, out_dir, device)
    ckpt = args.test_only or out_dir / "ckpt-best.pth"  # legacy test() was run on a chosen checkpoint, usually ckpt-best
    info = load_legacy_checkpoint(model, ckpt)
    print(f"testing {ckpt}: epoch {info['epoch']}, stored metrics {info['metrics']}")
    results = evaluate(model, test_loader, device)
    (out_dir / "test_results.json").write_text(json.dumps(results, indent=1))
    print(f"[TEST] Metrics (legacy test() aggregation) = {results['test']}")
    print(f"validate() aggregation = {results['validate']}  | per-sample mean = {results['per_sample_mean']}  (n={results['n_samples']})")


def train(model, cfg, paths, sem, out_dir, device):
    train_set = build_dataset(cfg["dataset"]["train"], paths, sem)
    val_set = build_dataset(cfg["dataset"]["val"], paths, sem)
    train_loader = DataLoader(train_set, batch_size=cfg["total_bs"], shuffle=True, drop_last=True,
                              num_workers=cfg.get("num_workers", 8), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 8))
    print(f"train {len(train_set)}, val {len(val_set)}")

    opt_cfg, sch, bn = cfg["optimizer"], cfg["scheduler"]["kwargs"], cfg["bnmscheduler"]["kwargs"]
    optimizer = torch.optim.AdamW(model.parameters(), **opt_cfg["kwargs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: max(sch["lr_decay"] ** (e / sch["decay_step"]), sch["lowest_decay"]))
    bn_momentum = lambda e: max(bn["bn_momentum"] * bn["bn_decay"] ** (e / bn["decay_step"]), bn["lowest_decay"])  # noqa: E731
    set_bn_momentum(model, bn_momentum(0))
    metric = cfg["consider_metric"]
    best, log = None, []

    model.zero_grad()
    for epoch in range(cfg["max_epoch"] + 1):
        model.train()
        start, totals, n = time.time(), np.zeros(4), 0
        for it, (_, _, partial, gt, _, _) in enumerate(train_loader):
            partial, gt = partial.to(device), gt.to(device)
            ret = forward(model, partial)
            loss_pcd, sparse, dense = model.get_loss(ret, gt)
            orth1, orth2 = model.get_constrain(ret)
            (loss_pcd + cfg["loss"]["orth_weight"] * (orth1 + orth2)).backward()
            optimizer.step()  # step_per_update = 1 in every legacy config
            model.zero_grad()
            totals += [sparse.item() * 1000, dense.item() * 1000, orth1.item() * 1000, orth2.item() * 1000]
            n += 1
            if it % 100 == 0:
                print(f"[Epoch {epoch}/{cfg['max_epoch']}][Batch {it + 1}/{len(train_loader)}] lr {optimizer.param_groups[0]['lr']:.6f}", flush=True)
        scheduler.step(epoch)
        set_bn_momentum(model, bn_momentum(epoch))
        entry = {"epoch": epoch, "time": time.time() - start, "losses": (totals / max(n, 1)).tolist()}

        if epoch % cfg.get("val_freq", 10) == 0 and epoch != 0:
            res = evaluate(model, val_loader, device)
            entry["validate"] = res["validate"]
            value = res["validate"][metric]
            if best is None or (value < best if LOWER_IS_BETTER[metric] else value > best):
                best = value
                torch.save({"base_model": model.state_dict(), "epoch": epoch, "metrics": res["validate"]}, out_dir / "ckpt-best.pth")
            print(f"[Validation] EPOCH {epoch}: {res['validate']}  (best {metric} {best:.4f})", flush=True)
        torch.save({"base_model": model.state_dict(), "epoch": epoch}, out_dir / "ckpt-last.pth")
        if cfg["max_epoch"] - epoch < 10:
            torch.save({"base_model": model.state_dict(), "epoch": epoch}, out_dir / f"ckpt-epoch-{epoch:03d}.pth")
        log.append(entry)
        (out_dir / "train_log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
