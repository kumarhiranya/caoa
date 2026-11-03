"""Train and evaluate the CAD retrieval embedding (P11d; caoa/retrieval).

Queries: S2C-C pairs (completion + instance, world frame); database: every ShapeNetV2 model
(`scripts/build_shapenet_points.py`). Each step: B queries against their GT CADs, `neg_same_category` random CADs of
each query's synset and `neg_random` random CADs from the whole database (InfoNCE over all of them). Validation on
the val pairs every `val_freq` steps (fixed queries, yaw as stored); the final evaluation also reports yaw test-time
augmentation. Outputs in `outputs/<name>/`: `config.yaml`, `train_log.json`, `model_final.pth`, `results.json`.

    uv run python scripts/train_retrieval.py configs/retrieval/embed_comp_inst.yaml [--max-steps N]
"""

import argparse
import json
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from caoa.data.pairs import load_pairs
from caoa.data.pose_completion import load_pose_completions
from caoa.data.shapenet import load_shapenet_points
from caoa.paths import REPO_ROOT, load_paths
from caoa.retrieval import search
from caoa.retrieval.data import RetrievalQueries, cad_key, collate, fixed_queries
from caoa.retrieval.model import CADRetrieval, info_nce

CAD_CHUNK = 256


def evaluate(model, queries, targets, db_points_gpu, db_catids, device, yaws=(1,)) -> dict:
    db_emb = search.embed_points(model.cad, db_points_gpu, device)
    q_catids = db_catids[targets.numpy()]
    out = {}
    for yaw in yaws:
        q_emb = search.embed_queries(model.query, queries, device, yaws=yaw)
        for mode in search.CATEGORY_MODES:
            mask = search.allowed_mask(q_catids, db_catids, mode, device)
            top, rank = search.search(q_emb, db_emb, mask, targets.to(device))
            out[f"{mode}/yaw{yaw}"] = search.retrieval_metrics(top, rank, targets, db_points_gpu, db_catids, mask)
    model.train()
    return out


def brief(res: dict) -> str:
    return "  ".join(f"{k}: R@1 {v['recall@1']:.3f} R@5 {v['recall@5']:.3f} cat {v['category_acc']:.3f} CD-L1 {v['cd_l1']:.1f} "
                     f"(random {v['random_cd_l1']:.1f})" for k, v in res.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-freq", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dcfg, tcfg = cfg["data"], cfg["train"]
    if args.max_steps is not None:
        tcfg["max_steps"] = args.max_steps
    if args.val_freq is not None:
        tcfg["val_freq"] = args.val_freq
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    paths = load_paths()
    device = torch.device("cuda")
    out_dir = REPO_ROOT / "outputs" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    db = load_shapenet_points(paths.cache)
    db_index = {k: i for i, k in enumerate(db["keys"].tolist())}
    db_catids = db["catids"]
    db_points = torch.from_numpy(db["points"])
    db_points_gpu = db_points.to(device)
    members = {c: torch.from_numpy(np.flatnonzero(db_catids == c)) for c in np.unique(db_catids)}
    print(f"database: {len(db_index)} ShapeNetV2 models, {len(members)} synsets")

    completions = load_pose_completions(paths.cache, dcfg["completion"]["name"], dcfg["completion"]["frame"])
    pairs = {s: [p for p in load_pairs(paths.cache, s, class_rule=dcfg["class_rule"]) if (p.scene_id, p.model_index) in completions]
             for s in ("train", "val")}
    print(f"pairs: train {len(pairs['train'])} / val {len(pairs['val'])}")
    train_ds = RetrievalQueries(pairs["train"], completions, db_index, train=True, num_points=dcfg["num_points"], inst_fraction=dcfg["inst_fraction"])
    loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, num_workers=tcfg["num_workers"],
                        collate_fn=collate, persistent_workers=True)
    val_queries = fixed_queries(pairs["val"], completions, dcfg["num_points"], dcfg["inst_fraction"], cfg["seed"])
    val_targets = torch.tensor([db_index[cad_key(p)] for p in pairs["val"]])

    model = CADRetrieval(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["max_steps"], eta_min=tcfg["lr"] * tcfg["min_lr_ratio"])
    target_catids = np.asarray(db_catids)
    log = {"step": [], "loss": [], "val_step": [], "val": []}
    step, t0, losses = 0, time.time(), []
    model.train()
    while step < tcfg["max_steps"]:
        for batch in loader:
            if step >= tcfg["max_steps"]:
                break
            targets = batch["target"]
            same = torch.cat([members[c][torch.randint(len(members[c]), (tcfg["neg_same_category"],))] for c in target_catids[targets.numpy()]])
            keys = torch.cat([targets, same, torch.randint(len(db_points), (tcfg["neg_random"],))])
            q = model.query(batch["points"].to(device, non_blocking=True))
            # in chunks: one spconv forward over > ~300 clouds of 2,048 points exceeds spconv's int32 buffer limit
            c = torch.cat([model.cad(x) for x in db_points_gpu[keys.to(device)].split(CAD_CHUNK)])
            loss = info_nce(q, c, keys.to(device), targets.to(device), model.logit_scale())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % tcfg["stat_freq"] == 0:
                log["step"].append(step)
                log["loss"].append(float(np.mean(losses)))
                print(f"step {step}  loss {np.mean(losses):.4f}  scale {model.logit_scale().item():.1f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {(time.time() - t0) / step:.3f} s/step", flush=True)
                losses = []
            if step % tcfg["val_freq"] == 0 or step == tcfg["max_steps"]:
                res = evaluate(model, val_queries, val_targets, db_points_gpu, db_catids, device)
                log["val_step"].append(step)
                log["val"].append({k: {m: v[m] for m in ("recall@1", "recall@5", "recall@10", "rank_median", "category_acc", "cd_l1", "cd_l2")}
                                   for k, v in res.items()})
                print(f"[val] step {step}  {brief(res)}", flush=True)
                (out_dir / "train_log.json").write_text(json.dumps(log))
    torch.save({"model": model.state_dict(), "step": step}, out_dir / "model_final.pth")

    results = evaluate(model, val_queries, val_targets, db_points_gpu, db_catids, device, yaws=tuple(cfg["eval"]["yaws"]))
    results["n_val_queries"] = len(val_targets)
    results["n_database"] = len(db_index)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    print(f"[final] {brief({k: v for k, v in results.items() if isinstance(v, dict)})}")


if __name__ == "__main__":
    main()
