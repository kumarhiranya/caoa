"""SoftGroup inference on ScanNet val with the official checkpoint, saving per-scene predictions in the format
`caoa.data.instances` reads.

Runs in an environment with the official SoftGroup code (github.com/thangvubk/SoftGroup, its CUDA ops built) and its
ScanNet val preprocessing (`dataset/scannetv2/prepare_data_inst.py`; setup: docs/instance_predictions.md). Mirrors
upstream `tools/test.py` (model, dataset, loader, `ScanNetEval`; the AP printout is the check against the published
45.8 AP / 67.4 AP50 / 79.1 AP25 on val) and writes `<out>/<scene>_sg_preds.npz` = {"arr_0": {"scan_id",
"pred_instances", "semantic_preds"}} (`pred_instances`: run-length masks over the scene's vertices in `.ply` order,
1-based `label_id` over the 18 instance classes, `conf`).

    cd <softgroup>/repo && CUDA_VISIBLE_DEVICES=0 <python> <caoa>/scripts/external/softgroup_infer.py \
        configs/softgroup/softgroup_scannet.yaml ../ckpt/softgroup_scannet_spconv2.pth ../preds/val --data-root <scannet val>/..
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
from softgroup.data import build_dataloader, build_dataset  # noqa: E402
from softgroup.evaluation import ScanNetEval  # noqa: E402
from softgroup.model import SoftGroup  # noqa: E402
from softgroup.util import get_root_logger, load_checkpoint  # noqa: E402

# PyTorch >= 2 no longer indexes a CPU tensor with CUDA indices, which `get_instances` does with `proposals_idx`: move
# it to the GPU first (same values, same result).
_get_instances = SoftGroup.get_instances
SoftGroup.get_instances = lambda self, scan_id, proposals_idx, *a, **k: _get_instances(self, scan_id, proposals_idx.cuda(), *a, **k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("ckpt")
    parser.add_argument("out_dir")
    parser.add_argument("--data-root", default=None, help="folder that holds val/ (default: the config's data_root)")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config)))
    if args.data_root:
        cfg.data.test.data_root = args.data_root
    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_root_logger(log_file=os.path.join(args.out_dir, "infer.log"))

    model = SoftGroup(**cfg.model).cuda()
    load_checkpoint(args.ckpt, logger, model)
    val_set = build_dataset(cfg.data.test, logger)
    loader = build_dataloader(val_set, training=False, dist=False, **cfg.dataloader.test)

    scannet_eval = ScanNetEval(val_set.CLASSES)
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.limit and i >= args.limit:
                break
            res = model(batch)
            scan_id = res["scan_id"]
            out = {"scan_id": scan_id, "pred_instances": res["pred_instances"],
                   "semantic_preds": np.asarray(res["semantic_preds"]).astype(np.uint8)}
            np.savez_compressed(os.path.join(args.out_dir, f"{scan_id}_sg_preds.npz"), arr_0=np.array(out, dtype=object))
            preds.append(res["pred_instances"])
            gts.append(res["gt_instances"])
            if i % 20 == 0:
                logger.info(f"{i + 1}/{len(val_set)} {scan_id}: {len(res['pred_instances'])} instances, {len(out['semantic_preds'])} points")
    scannet_eval.evaluate(preds, gts)


if __name__ == "__main__":
    main()
