"""Train and evaluate the object-only OCAM pose network.

    uv run python scripts/train_pose.py configs/pose/legacy_nocomp.yaml [--max-steps N] [--eval-only CKPT]

The training loop reproduces legacy `train_pose9dof` (utils/training_utils.py):
checks run on the step counter before it is incremented (stats every
`stat_freq`, checkpoint every `save_freq`, validation every `val_freq`), the
loop only stops at the end of the epoch in which `max_steps` is reached, and the
cosine schedule keeps stepping past `max_steps`. The final model is then
evaluated as legacy eval mode did (`eval_pose_s2c_v2` with per-class results and
timing), plus the labelled official/matched additions.

Writes to outputs/<name>/: config, TensorBoard logs, train_log.json (legacy
keys), model.pth / model_final.pth, results.json and per-class CSVs.
"""

import argparse
import csv
import json
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from caoa.data import scan2cad
from caoa.data.labels import scannet_split
from caoa.data.pairs import cache_dir, load_pairs
from caoa.data.pose_completion import load_pose_completions
from caoa.data.pose_dataset import ObjectPoseDataset, PoseDataConfig, collate, load_sym_embeddings
from caoa.eval import groupings, pose_eval
from caoa.models.ocam import OCAMObjectOnly
from caoa.paths import REPO_ROOT, load_paths


def make_loader(pairs, train: bool, cfg: dict, data_cfg: PoseDataConfig, shuffle: bool, completions=None, completion_mode=None,
                sym_embeddings=None):
    return DataLoader(
        ObjectPoseDataset(pairs, train, data_cfg, completions, completion_mode, sym_embeddings),
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
        persistent_workers=train,
    )


def model_inputs(batch, device):
    ext = batch["sym_emb"].to(device) if "sym_emb" in batch else None  # legacy "mnkpv3_emb" (P10a)
    return batch["coords"].to(device), batch["feats"].to(device), batch["batch"].to(device), batch["batch_size"], ext


def forward(model, batch, device):
    return model(*model_inputs(batch, device))


@torch.no_grad()
def predict(model, loader, device, timed=False):
    """Predictions for a whole loader. With `timed`, also inference ms/sample:
    "legacy" times the forward call without synchronising (as legacy did), "synced" waits for the GPU."""
    model.eval()
    out = {"pred": [], "target": [], "norm_scale": [], "norm_offset": [], "index": []}
    legacy_ms, synced_ms = [], []
    for batch in loader:
        inputs = model_inputs(batch, device)
        if timed:
            torch.cuda.synchronize()
        start = time.time()
        pred = model(*inputs)
        legacy_ms.append((time.time() - start) * 1000 / batch["batch_size"])
        if timed:
            torch.cuda.synchronize()
            synced_ms.append((time.time() - start) * 1000 / batch["batch_size"])
        out["pred"].append(pred.cpu().numpy())
        for k in ("target", "norm_scale", "norm_offset", "index"):
            out[k].append(batch[k].numpy())
    model.train()
    result = {k: np.concatenate(v) for k, v in out.items()}
    if timed:
        result["timing_ms_per_sample"] = {"legacy": float(np.mean(legacy_ms)), "synced": float(np.mean(synced_ms))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--max-steps", type=int, default=None, help="override, e.g. for a smoke test")
    parser.add_argument("--eval-only", default=None, help="checkpoint to evaluate instead of training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    tcfg = cfg["train"]
    out_dir = REPO_ROOT / "outputs" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # as legacy (ODGNet main.py:18, train_pose.py)

    paths = load_paths()
    data = dict(cfg["data"])
    class_rule = data.pop("class_rule")
    completion = data.pop("completion", None)  # {name: <pose completion cache>, mode: only | inst+comp, frame: legacy | world}
    completions = load_pose_completions(paths.cache, completion["name"], completion.get("frame", "legacy")) if completion else None
    completion_mode = completion["mode"] if completion else None
    # P10a: SEM symmetry embedding of the GT CAD (legacy `symmetry_model_name`, data/s2c_symmetry/<name>/embeddings.npy)
    sym_model = cfg["model"].get("sym_embedding")
    sym_embeddings = load_sym_embeddings(paths.legacy_repo / "data/s2c_symmetry" / sym_model / "embeddings.npy") if sym_model else None
    train_pairs = load_pairs(paths.cache, "train", class_rule=class_rule)
    val_pairs = load_pairs(paths.cache, "val", class_rule=class_rule)
    data_cfg = PoseDataConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
    train_loader = make_loader(train_pairs, True, cfg, data_cfg, shuffle=tcfg["shuffle"], completions=completions, completion_mode=completion_mode,
                               sym_embeddings=sym_embeddings)
    val_loader = make_loader(val_pairs, False, cfg, data_cfg, shuffle=False, completions=completions, completion_mode=completion_mode,
                             sym_embeddings=sym_embeddings)
    print(f"class rule {class_rule}: train pairs {len(train_pairs)}, val pairs {len(val_pairs)}"
          + (f"; completion {completion['name']} ({completion_mode}, frame {completion.get('frame', 'legacy')})" if completion else ""))

    ext_emb_size = len(next(iter(sym_embeddings.values()))) if sym_embeddings else 0
    model = OCAMObjectOnly(embedding=cfg["model"]["embedding"], ext_emb_size=ext_emb_size).to(device)
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=device)["model"])
    else:
        train(model, train_loader, val_loader, val_pairs, tcfg, out_dir, device)

    final = predict(model, val_loader, device, timed=True)
    results = evaluate(final, val_pairs, paths)
    results["class_rule"] = class_rule
    results["timing_ms_per_sample"] = final["timing_ms_per_sample"]
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    write_per_class_csvs(results, out_dir)
    print_summary(results)


def train(model, train_loader, val_loader, val_pairs, tcfg, out_dir, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["max_steps"], eta_min=tcfg["lr"] * tcfg["min_lr_ratio"])
    w = tcfg["loss_weights"]
    l1 = torch.nn.L1Loss(reduction="sum")
    writer = SummaryWriter(out_dir)
    log = {k: [] for k in ("loss", "loss_pose", "loss_t", "loss_q", "loss_s", "iter_loss", "epochs",
                           "inst_pose_mae", "iter_acc", "s2c_t", "s2c_q", "s2c_s", "s2c_total")}

    i, epoch, start = 0, 0, time.time()
    model.train()
    while i < tcfg["max_steps"]:
        for batch in train_loader:
            pred = forward(model, batch, device)
            target = batch["target"].to(device)
            n = batch["batch_size"]
            loss_t = l1(pred[:, :3], target[:, :3]) / n
            loss_q = l1(pred[:, 3:12], target[:, 3:12]) / n
            loss_s = l1(pred[:, 12:], target[:, 12:]) / n
            loss = w["t"] * loss_t + w["r"] * loss_q + w["s"] * loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % tcfg["stat_freq"] == 0 and i > 0:
                values = {"loss": loss, "loss_pose": loss, "loss_t": loss_t, "loss_q": loss_q, "loss_s": loss_s}
                for k, v in values.items():
                    log[k].append(v.item())
                log["iter_loss"].append(i)
                log["epochs"].append(epoch)
                rate = (i + 1) / (time.time() - start)
                print(f"Iter: {i}, Loss: {loss.item():.4f}, t:{loss_t.item():.4f}, q:{loss_q.item():.4f}, s:{loss_s.item():.4f}  ({rate:.1f} it/s)", flush=True)
                for k in ("loss", "loss_t", "loss_q", "loss_s"):
                    writer.add_scalar(f"loss/{k.removeprefix('loss_') if k != 'loss' else 'pose'}", log[k][-1], i)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], i)
                (out_dir / "train_log.json").write_text(json.dumps(log))

            if i % tcfg["save_freq"] == 0:
                torch.save({"model": model.state_dict(), "step": i}, out_dir / "model.pth")

            if i % tcfg["val_freq"] == 0 and i > 0:
                val = predict(model, val_loader, device)
                rep = pose_eval.evaluate_legacy_only(val_pairs, val["pred"], val["target"])
                avg, mae = rep["avg"], rep["mae"]
                log["inst_pose_mae"].append(mae["pose"])
                log["iter_acc"].append(i)
                for k in ("t", "q", "s"):
                    log[f"s2c_{k}"].append(avg[f"{k}_acc"])
                log["s2c_total"].append(avg["total_acc"])
                for k in ("pose", "t", "q", "s"):
                    writer.add_scalar(f"val_mae/{k}", mae[k], i)
                for k in ("t_acc", "q_acc", "s_acc", "total_acc", "rotation_error_deg"):
                    writer.add_scalar(f"test_acc/{k}", avg[k], i)
                print(
                    f"Epoch: {epoch}, Validation Pose MAE: Inst:{mae['pose']:.4f} t {mae['t']:.4f} q {mae['q']:.4f} s {mae['s']:.4f}, "
                    f"S2C Avg: total {avg['total_acc']:.4f} t {avg['t_acc']:.4f} q {avg['q_acc']:.4f} s {avg['s_acc']:.4f}",
                    flush=True,
                )
                (out_dir / "train_log.json").write_text(json.dumps(log))

            i += 1
        epoch += 1

    print(f"stopped after {i} steps ({epoch} epochs)")
    torch.save({"model": model.state_dict(), "step": i}, out_dir / "model_final.pth")
    writer.close()


def evaluate(final, val_pairs, paths):
    with open(paths.scan2cad_full_annotations) as f:
        full = json.load(f)
    with open(paths.scan2cad_cad_appearances) as f:
        appearances = json.load(f)
    # Benchmark scenes: every locally available val scene, including those with no matched pairs.
    scenes = sorted(p.stem for p in cache_dir(paths.cache).glob("scene*.pkl") if scannet_split(p.stem) == "val")
    pairs = [val_pairs[i] for i in final["index"]]
    results = pose_eval.evaluate(
        pairs, final["pred"], final["target"], final["norm_scale"], final["norm_offset"],
        scan2cad.load_annotations(paths.s2c_annotations), full, appearances, scenes,
    )
    results["n_val_pairs"] = len(pairs)
    results["n_benchmark_scenes"] = len(scenes)
    results["n_benchmark_gt"] = sum(len(a["aligned_models"]) for a in full if a["id_scan"] in set(scenes))
    return results


def write_per_class_csvs(results, out_dir):
    """One table per grouping: rows = classes + macro/micro averages; columns as legacy eval mode printed."""
    for grouping in groupings.GROUPINGS:
        pc = results["legacy"][grouping]["per_class"]
        rows = [c for c in pc["total"] if c not in ("macro_avg", "micro_avg")] + ["macro_avg", "micro_avg"]
        with open(out_dir / f"legacy_per_class_{grouping}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", "translation", "rotation", "scale", "total", "support"])
            for r in rows:
                w.writerow([r, pc["t"][r], pc["q"][r], pc["s"][r], pc["total"][r], pc["support"].get(r, "")])


def print_summary(results):
    print(f"\nval pairs {results['n_val_pairs']} (class rule {results['class_rule']}), "
          f"benchmark scenes {results['n_benchmark_scenes']} ({results['n_benchmark_gt']} GT objects)")
    for g in groupings.GROUPINGS:
        r = results["legacy"][g]
        print(f"legacy [{g:<17}] micro {r['per_class']['total']['micro_avg']:.4f}  macro {r['per_class']['total']['macro_avg']:.4f}")
    a = results["legacy"]["scannet_class"]
    print(f"legacy MAE {a['mae']}  mean rot err {a['avg']['rotation_error_deg']:.2f} deg  mean scale err {a['avg']['mean_scale_error_pct']:.2f}%")
    e5 = results["legacy_valid_classes"]
    print(f"legacy calc_pose_accuracy_s2c (VALID_CLASSES): avg {e5['avg']['total_acc']:.4f}  cls_avg {e5['cls_avg']['total_acc']:.4f}")
    for key in ("benchmark/plain", "benchmark/minus_center", "matched/plain", "matched/minus_center"):
        r = results[key]
        print(f"{key:<24} instance {r['instance_mean']:.4f}  class {r['class_mean']:.4f}")
    print(f"inference ms/sample: {results['timing_ms_per_sample']}")


if __name__ == "__main__":
    main()
