"""Re-score legacy CAPCM checkpoints on the S2C-C test set (P6 gate).

    uv run python scripts/rescore_completion.py [--runs S2C-C PCN ...] [--repeats 1]

Each run is the checkpoint + model config a legacy `test_S2CC_benchmark` run used
(Mar 5 2025). Targets: the logged `[TEST] Metrics` line (legacy `test()` aggregation)
and the per-category table in `<legacy_results>/ODG_*_S2CC.csv` (server copy; `paths.yaml`). The test set is
the S2C-C val split without context (`legacy val == test`), batch size 1, random
subsampling to 4,096 points per item as in legacy (so repeats differ slightly).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from caoa.data.completion_dataset import S2CCompletionDataset, load_completion_samples
from caoa.eval.completion_metrics import CompletionEvaluator, sample_metrics
from caoa.models.odgnet import UpTrans, load_legacy_checkpoint
from caoa.paths import REPO_ROOT, load_paths

RUNS = {  # name: (checkpoint, test-run config, logged [TEST] F / CDL1 / CDL2, per-category csv)
    "S2C-C": ("experiments/UpTrans/S2CCompletion/s2ccompletion_upTransb1/ckpt-best.pth",
              "experiments/config/s2ccompletion_upTransb1/test_S2CC_benchmark/config.yaml", (0.0602, 46.8023, 25.6921), "ODG_S2CC_S2CC.csv"),
    "PCN": ("PCN_models/ckpt-best.pth", "experiments/UpTransS2CC/PCN_models/test_S2CC_benchmark/config.yaml",
            (0.0261, 161.5209, 137.1238), "ODG_PCN_S2CC.csv"),
    "ShapeNet-34": ("ShapeNet34_models/512ckpt-best1.pth", "experiments/UpTransS2CC/ShapeNet34_models/test_S2CC_benchmark/config.yaml",
                    (0.0485, 94.0136, 77.8934), "ODG_SN34_S2CC.csv"),
    "ShapeNet-55": ("ShapeNet55_models/ckpt-best.pth", "experiments/UpTransS2CC/ShapeNet55_models/test_S2CC_benchmark/config.yaml",
                    (0.0483, 94.1031, 78.0289), "ODG_SN55_S2CC.csv"),
    "Concat SN-Indoor+S2C-C": ("experiments/UpTrans_comp/Concat_SN_S2CC/concat_snaug_s2cc_upTransb_comp/ckpt-last.pth",
                               "experiments/UpTrans_comp/Concat_SN_S2CC/test_S2CC_benchmark/config.yaml", (0.0624, 45.3712, 23.7441), "ODG_Concat_S2CC.csv"),
    "SN-Indoor": ("experiments/UpTrans_comp/SNAug/snaug_upTransb_comp/ckpt-last.pth",
                  "experiments/UpTrans_compS2CC/SNAug/test_S2CC_benchmark/config.yaml", (0.0513, 66.1926, 45.6618), None),
}


def read_category_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.reader(f))
    # per-taxonomy rows; the trailing "Overall" row has no count
    return {r[0]: {"count": int(r[1]), "F-Score": float(r[2]), "CDL1": float(r[3]), "CDL2": float(r[4])} for r in rows[1:] if len(r) >= 5 and r[1].strip().isdigit()}


@torch.no_grad()
def score(model, dataset, device):
    model.eval()
    ev = CompletionEvaluator()
    for catids, _, partial, gt, _, _ in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8):
        ev.update(catids[0], sample_metrics(model(partial.to(device))[1], gt.to(device)))
    return ev.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=list(RUNS))
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    paths = load_paths()
    odgnet = paths.legacy_repo / "ODGNet"
    np.random.seed(0)
    sem = np.load(paths.semantic_embeddings, allow_pickle=True).item()
    dataset = S2CCompletionDataset(load_completion_samples(paths.cache, 0.0, "val"), "val", sem, num_points=4096)
    print(f"S2C-C test set: {len(dataset)} samples")
    device = torch.device("cuda")
    out_dir = REPO_ROOT / "outputs" / "rescore_completion"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((out_dir / "results.json").read_text()) if (out_dir / "results.json").exists() else {}

    for name in args.runs:
        ckpt, cfg_path, logged, cat_csv = RUNS[name]
        ckpt = odgnet / ckpt
        if not ckpt.exists():
            print(f"{name}: checkpoint {ckpt} not found, skipped")
            report[name] = {"skipped": f"missing {ckpt}"}
            continue
        model = UpTrans.from_config(yaml.safe_load(open(odgnet / cfg_path))["model"]).to(device)
        load_legacy_checkpoint(model, ckpt)
        runs = [score(model, dataset, device) for _ in range(args.repeats)]
        test = {k: float(np.mean([r["test"][k] for r in runs])) for k in ("F-Score", "CDL1", "CDL2")}
        legacy_cats = read_category_csv(paths.legacy_results / cat_csv) if cat_csv else {}
        counts_match = all(runs[0]["per_category"].get(c, {}).get("count") == v["count"] for c, v in legacy_cats.items()) if legacy_cats else None
        report[name] = {"ours_test": test, "logged_test": dict(zip(("F-Score", "CDL1", "CDL2"), logged)),
                        "per_category": runs[0]["per_category"], "legacy_per_category": legacy_cats,
                        "category_counts_match": counts_match, "validate": runs[0]["validate"]}
        print(f"{name:<24} ours F {test['F-Score']:.4f} CDL1 {test['CDL1']:.4f} CDL2 {test['CDL2']:.4f} | "
              f"logged F {logged[0]:.4f} CDL1 {logged[1]:.4f} CDL2 {logged[2]:.4f} | category counts match: {counts_match}", flush=True)
        (out_dir / "results.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
