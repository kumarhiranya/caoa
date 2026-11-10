"""SphericalMask inference on ScanNet val, saving per-scene predictions in SoftGroup's file layout.

Runs in SphericalMask's own environment (not this project's): the upstream repo, venv, data and checkpoint live in
their own folder (setup: docs/sphericalmask.md). Mirrors upstream
`tools/test.py` (model, dataset, loader, `ScanNetEval`; the checkpoint is loaded strictly after one key rename, see below; the AP printout is the sanity check against
the published 62.6 AP / 81.9 AP50 on val) and additionally writes `<out>/<scene>_sg_preds.npz` = {"arr_0": {"scan_id",
"pred_instances", "semantic_preds"}}, the format `caoa.data.instances` reads (`pred_instances`: run-length masks over the
scene's vertices in `.ply` order, 1-based `label_id` over the 18 instance classes, `conf`).

    cd <sphericalmask>/repo && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 ../venv/bin/python \
        <caoa>/scripts/external/sphericalmask_infer.py ../configs/spherical_mask_val.yaml ../ckpt/spherical_mask_155.pth ../preds/val
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from munch import Munch

sys.path.insert(0, os.getcwd())  # the upstream repo
# NumPy 2 removed names the upstream evaluation still uses (`ScanNetEval`): restore them as the old aliases.
for _name, _alias in (("in1d", np.isin), ("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)
from spherical_mask.data import build_dataloader, build_dataset  # noqa: E402
from spherical_mask.evaluation import ScanNetEval  # noqa: E402
from spherical_mask.model import SphericalMask  # noqa: E402
from spherical_mask.model.criterion_spherical_mask import Criterion_SphericalMask  # noqa: E402
from spherical_mask.util import get_root_logger  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("ckpt")
    parser.add_argument("out_dir")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config)))
    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_root_logger(log_file=os.path.join(args.out_dir, "infer.log"))

    criterion = Criterion_SphericalMask(cfg.model.semantic_classes, cfg.model.instance_classes, cfg.model.semantic_weight,
                                        cfg.model.ignore_label, semantic_only=cfg.model.semantic_only, total_epoch=cfg.epochs,
                                        voxel_scale=cfg.data.train.voxel_cfg.scale)
    model = SphericalMask(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=False).cuda()
    # The released checkpoint names one module `inst_ray_head`; the released code calls it `rid_module` (same 14 tensors,
    # same shapes). Upstream `load_checkpoint` is non-strict and would leave it randomly initialised: rename, load strictly.
    state = torch.load(args.ckpt, map_location="cpu")["net"]
    state = {k.replace("inst_ray_head.", "rid_module.", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    cfg.data.test["debug"] = False
    val_set = build_dataset(cfg.data.test, logger)
    loader = build_dataloader(val_set, training=False, dist=False, **cfg.dataloader.test)

    scannet_eval = ScanNetEval(val_set.CLASSES, dataset_name=cfg.data.train.type)
    preds, sems, insts = [], [], []
    model.iterative_sampling = False
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.limit and i >= args.limit:
                break
            res = model(batch)
            scan_id = res["scan_id"] if "scan_id" in res else batch["scan_ids"][0]
            pred_instances = [{k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in p.items()} for p in res["pred_instances"]]
            sem = res.get("semantic_preds")
            n = len(res["semantic_labels"])
            out = {"scan_id": scan_id, "pred_instances": pred_instances,
                   "semantic_preds": np.asarray(sem) if sem is not None else np.full(n, -100, dtype=np.int64)}
            np.savez(os.path.join(args.out_dir, f"{scan_id}_sg_preds.npz"), out)
            preds.append(res["pred_instances"])
            sems.append(res["semantic_labels"])
            insts.append(res["instance_labels"])
            if i % 20 == 0:
                logger.info(f"{i + 1}/{len(val_set)} {scan_id}: {len(pred_instances)} instances, {n} points")
    res = scannet_eval.evaluate(preds, sems, insts)
    logger.info(f"AP {res['all_ap']:.4f}  AP50 {res['all_ap_50%']:.4f}  AP25 {res['all_ap_25%']:.4f}")


if __name__ == "__main__":
    main()
