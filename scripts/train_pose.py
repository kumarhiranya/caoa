"""Train and evaluate an OCAM (object–CAD alignment network).

    uv run python scripts/train_pose.py configs/ocam/final.yaml [--seed N] [--max-steps N] [--eval-only CKPT]

Training loop: checks run on the step counter before it is incremented (stats every `stat_freq`, checkpoint every
`save_freq`, validation every `val_freq` with the CAOA metric), the loop stops at the end of the epoch in which
`max_steps` is reached, and the cosine schedule keeps stepping past `max_steps`. The final model is then evaluated on
the 1,864 val objects (`caoa.eval.pose_eval`: CAOA metric per grouping, Scan2CAD criteria, box criteria) with timing.

`--seed N` (other than the config's) trains `<name>_seed<N>`. Writes to outputs/<name>/: config, TensorBoard logs,
train_log.json, model.pth (last periodic checkpoint) / model_final.pth, results.json and per-class CSVs.
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
from caoa.data.pairs import load_pairs
from caoa.data.pose_completion import load_pose_completions
from caoa.data.pose_dataset import ObjectPoseDataset, PoseDataConfig, collate, load_cad_points, load_sym_embeddings, sem_embedding_path
from caoa.eval import groupings, pose_eval
from caoa.models.ocam import OCAMObjectOnly, OCAMSiamese
from caoa.models.pose_losses import chamfer_pose_loss
from caoa.paths import REPO_ROOT, load_paths


def make_loader(pairs, train: bool, cfg: dict, data_cfg: PoseDataConfig, shuffle: bool, completions=None, completion_mode=None,
                sym_embeddings=None, cad_points=None):
    return DataLoader(
        ObjectPoseDataset(pairs, train, data_cfg, completions, completion_mode, sym_embeddings, cad_points),
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
        persistent_workers=train,
    )


def model_inputs(batch, device):
    ext = batch["sym_emb"].to(device) if "sym_emb" in batch else None  # SEM embedding
    obj = (batch["coords"].to(device), batch["feats"].to(device), batch["batch"].to(device), batch["batch_size"])
    if "cad_coords" in batch:  # Siamese
        return *obj, batch["cad_coords"].to(device), batch["cad_feats"].to(device), batch["cad_batch"].to(device), ext
    return *obj, ext


def forward(model, batch, device):
    return model(*model_inputs(batch, device))


@torch.no_grad()
def predict(model, loader, device, timed=False):
    """Predictions for a whole loader. With `timed`, also inference ms/sample: "unsynced" times the forward call
    without synchronising, "synced" waits for the GPU."""
    model.eval()
    out = {"pred": [], "target": [], "norm_scale": [], "norm_offset": [], "index": []}
    unsynced_ms, synced_ms = [], []
    for batch in loader:
        inputs = model_inputs(batch, device)
        if timed:
            torch.cuda.synchronize()
        start = time.time()
        pred = model(*inputs)
        unsynced_ms.append((time.time() - start) * 1000 / batch["batch_size"])
        if timed:
            torch.cuda.synchronize()
            synced_ms.append((time.time() - start) * 1000 / batch["batch_size"])
        out["pred"].append(pred.cpu().numpy())
        for k in ("target", "norm_scale", "norm_offset", "index"):
            out[k].append(batch[k].numpy())
    model.train()
    result = {k: np.concatenate(v) for k, v in out.items()}
    if timed:
        result["timing_ms_per_sample"] = {"unsynced": float(np.mean(unsynced_ms)), "synced": float(np.mean(synced_ms))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--max-steps", type=int, default=None, help="override, e.g. for a smoke test")
    parser.add_argument("--eval-only", default=None, help="checkpoint to evaluate instead of training")
    parser.add_argument("--seed", type=int, default=None, help="another seed: trains <name>_seed<N>")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None and args.seed != cfg["seed"]:
        cfg["name"], cfg["seed"] = f"{cfg['name']}_seed{args.seed}", args.seed
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
    torch.backends.cudnn.benchmark = True

    paths = load_paths()
    data = dict(cfg["data"])
    completion = data.pop("completion", None)  # {name: <pose completion cache>, mode: only | inst+comp, frame: world, checkpoint}
    completions = load_pose_completions(paths.cache, completion["name"], completion.get("frame", "world")) if completion else None
    completion_mode = completion["mode"] if completion else None
    # Siamese OCAM: CAD points in the CAD's bounding-box frame (scripts/build_cad_points.py)
    siamese = cfg["model"].get("siamese")  # {shared: true (one trunk, the paper's) | false (separate trunks)}
    cad_points = load_cad_points(paths.cache) if siamese else None
    # SEM symmetry embedding of the CAD; sym_embedding: {name: <embeddings cache>, frame: cad_bbox, checkpoint}
    sym = cfg["model"].get("sym_embedding")
    sym_embeddings = load_sym_embeddings(sem_embedding_path(paths, sym["name"], sym.get("frame", "cad_bbox"))) if sym else None
    train_pairs = load_pairs(paths.cache, "train")
    val_pairs = load_pairs(paths.cache, "val")
    data_cfg = PoseDataConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
    train_loader = make_loader(train_pairs, True, cfg, data_cfg, shuffle=tcfg["shuffle"], completions=completions, completion_mode=completion_mode,
                               sym_embeddings=sym_embeddings, cad_points=cad_points)
    val_loader = make_loader(val_pairs, False, cfg, data_cfg, shuffle=False, completions=completions, completion_mode=completion_mode,
                             sym_embeddings=sym_embeddings, cad_points=cad_points)
    print(f"train pairs {len(train_pairs)}, val pairs {len(val_pairs)}"
          + (f"; completion {completion['name']} ({completion_mode})" if completion else ""))

    ext_emb_size = len(next(iter(sym_embeddings.values()))) if sym_embeddings else 0
    if siamese:
        model = OCAMSiamese(embedding=cfg["model"]["embedding"], ext_emb_size=ext_emb_size, shared=siamese["shared"]).to(device)
    else:
        model = OCAMObjectOnly(embedding=cfg["model"]["embedding"], ext_emb_size=ext_emb_size).to(device)
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=device)["model"])
    else:
        train(model, train_loader, val_loader, val_pairs, tcfg, out_dir, device)

    final = predict(model, val_loader, device, timed=True)
    results = evaluate(final, val_pairs, paths)
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
    chamfer_weight = tcfg.get("chamfer_weight", 0.0)  # the paper's Chamfer term
    log = {k: [] for k in ("loss", "loss_pose", "loss_t", "loss_q", "loss_s", "loss_chamfer", "iter_loss", "epochs",
                           "val_pose_mae", "iter_acc", "val_caoa_t", "val_caoa_q", "val_caoa_s", "val_caoa_total")}

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
            loss_pose = w["t"] * loss_t + w["r"] * loss_q + w["s"] * loss_s
            loss_cd = chamfer_pose_loss(pred, target, batch["cad_unit"].to(device)) if chamfer_weight else torch.zeros((), device=device)
            loss = loss_pose + chamfer_weight * loss_cd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % tcfg["stat_freq"] == 0 and i > 0:
                values = {"loss": loss, "loss_pose": loss_pose, "loss_t": loss_t, "loss_q": loss_q, "loss_s": loss_s, "loss_chamfer": loss_cd}
                for k, v in values.items():
                    log[k].append(v.item())
                log["iter_loss"].append(i)
                log["epochs"].append(epoch)
                rate = (i + 1) / (time.time() - start)
                print(f"Iter: {i}, Loss: {loss.item():.4f}, t:{loss_t.item():.4f}, q:{loss_q.item():.4f}, s:{loss_s.item():.4f}, chamfer: {loss_cd.item():.4f}  ({rate:.1f} it/s)", flush=True)
                for k in ("loss", "loss_t", "loss_q", "loss_s") + (("loss_chamfer",) if chamfer_weight else ()):
                    writer.add_scalar(f"loss/{k.removeprefix('loss_') if k != 'loss' else 'pose'}", log[k][-1], i)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], i)
                (out_dir / "train_log.json").write_text(json.dumps(log))

            if i % tcfg["save_freq"] == 0:
                torch.save({"model": model.state_dict(), "step": i}, out_dir / "model.pth")

            if i % tcfg["val_freq"] == 0 and i > 0:
                val = predict(model, val_loader, device)
                rep = pose_eval.evaluate_caoa_only(val_pairs, val["pred"], val["target"])
                avg, mae = rep["avg"], rep["mae"]
                log["val_pose_mae"].append(mae["pose"])
                log["iter_acc"].append(i)
                for k in ("t", "q", "s"):
                    log[f"val_caoa_{k}"].append(avg[f"{k}_acc"])
                log["val_caoa_total"].append(avg["total_acc"])
                for k in ("pose", "t", "q", "s"):
                    writer.add_scalar(f"val_mae/{k}", mae[k], i)
                for k in ("t_acc", "q_acc", "s_acc", "total_acc", "rotation_error_deg"):
                    writer.add_scalar(f"test_acc/{k}", avg[k], i)
                print(
                    f"Epoch: {epoch}, Validation Pose MAE: Inst:{mae['pose']:.4f} t {mae['t']:.4f} q {mae['q']:.4f} s {mae['s']:.4f}, "
                    f"CAOA metric: total {avg['total_acc']:.4f} t {avg['t_acc']:.4f} q {avg['q_acc']:.4f} s {avg['s_acc']:.4f}",
                    flush=True,
                )
                (out_dir / "train_log.json").write_text(json.dumps(log))

            i += 1
        epoch += 1

    print(f"stopped after {i} steps ({epoch} epochs)")
    torch.save({"model": model.state_dict(), "step": i}, out_dir / "model_final.pth")
    writer.close()


def evaluate(final, val_pairs, paths):
    pairs = [val_pairs[i] for i in final["index"]]
    results = pose_eval.evaluate(
        pairs, final["pred"], final["target"], final["norm_scale"], final["norm_offset"], scan2cad.load_annotations(paths.s2c_annotations)
    )
    results["n_val_pairs"] = len(pairs)
    return results


def write_per_class_csvs(results, out_dir):
    """CAOA metric, one table per grouping: rows = classes + macro/micro averages."""
    for grouping in groupings.GROUPINGS:
        pc = results["caoa"][grouping]["per_class"]
        rows = [c for c in pc["total"] if c not in ("macro_avg", "micro_avg")] + ["macro_avg", "micro_avg"]
        with open(out_dir / f"caoa_per_class_{grouping}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", "translation", "rotation", "scale", "total", "support"])
            for r in rows:
                w.writerow([r, pc["t"][r], pc["q"][r], pc["s"][r], pc["total"][r], pc["support"].get(r, "")])


def print_summary(results):
    print(f"\nval pairs {results['n_val_pairs']}")
    for key in ("s2c_box", "s2c_criteria/minus_center", "s2c_criteria/plain"):
        r = results[key]
        print(f"{key:<24} instance {r['instance_mean']:.4f}  class {r['class_mean']:.4f}")
    for g in groupings.GROUPINGS:
        r = results["caoa"][g]
        print(f"CAOA [{g:<17}] micro {r['per_class']['total']['micro_avg']:.4f}  macro {r['per_class']['total']['macro_avg']:.4f}")
    a = results["caoa"]["scannet_class"]
    print(f"CAOA MAE {a['mae']}  mean rot err {a['avg']['rotation_error_deg']:.2f} deg  mean scale err {a['avg']['mean_scale_error_pct']:.2f}%")
    print(f"inference ms/sample: {results['timing_ms_per_sample']}")


if __name__ == "__main__":
    main()
