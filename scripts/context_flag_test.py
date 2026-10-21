"""Empirical context-flag test (P6; user decision 2026-10-20).

    uv run python scripts/context_flag_test.py [--repeats 2]

Scores the legacy context checkpoint (`UpTrans_comp/S2CCompletion/s2ccompletion_upTransb1_comp`,
in_channels 4, best-val CDL1 22.471 / CDL2 2.300 / F 0.2152 at epoch 170, from legacy
`validate()`) on the S2C-C val split with 1 m context, once per flag encoding:
+1/-1 (pose-time inference and the paper) and 1/0 (the ODGNet loader's context code).
The encoding that reproduces the stored numbers is the one the checkpoint was trained with.
"""

import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from caoa.data.completion_dataset import S2CCompletionDataset, load_completion_samples
from caoa.eval.completion_metrics import CompletionEvaluator, sample_metrics
from caoa.models.odgnet import UpTrans, load_legacy_checkpoint
from caoa.paths import REPO_ROOT, load_paths

RUN = "experiments/UpTrans_comp/S2CCompletion/s2ccompletion_upTransb1_comp"
ENCODINGS = {"+1/-1": (1.0, -1.0), "1/0": (1.0, 0.0)}


@torch.no_grad()
def score(model, dataset, device):
    ev = CompletionEvaluator()
    for catids, _, partial, gt, _, _ in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8):
        partial = partial.to(device)
        ev.update(catids[0], sample_metrics(model(partial[:, :, :3], partial[:, :, 3:])[1], gt.to(device)))
    return ev.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    paths = load_paths()
    run = paths.legacy_repo / "ODGNet" / RUN
    device = torch.device("cuda")
    model = UpTrans.from_config(yaml.safe_load(open(run / "config.yaml"))["model"]).to(device).eval()
    stored = load_legacy_checkpoint(model, run / "ckpt-best.pth")
    print(f"checkpoint epoch {stored['epoch']}, stored validate() metrics {stored['metrics']}")

    sem = np.load(paths.semantic_embeddings, allow_pickle=True).item()
    samples = load_completion_samples(paths.cache, 1.0, "val")
    print(f"val samples with 1 m context: {len(samples)} (with any context points: {sum(len(s.context_points) > 0 for s in samples)})")

    report = {"stored": stored["metrics"], "epoch": stored["epoch"]}
    np.random.seed(0)
    for name, flag in ENCODINGS.items():
        ds = S2CCompletionDataset(samples, "val", sem, num_points=4096, context_radius=1.0, context_flag=flag)
        runs = [score(model, ds, device) for _ in range(args.repeats)]
        mean = {agg: {k: float(np.mean([r[agg][k] for r in runs])) for k in ("F-Score", "CDL1", "CDL2")} for agg in ("validate", "test")}
        report[name] = mean
        print(f"flag {name:>6}: validate() F {mean['validate']['F-Score']:.4f} CDL1 {mean['validate']['CDL1']:.3f} CDL2 {mean['validate']['CDL2']:.3f}"
              f" | test() CDL1 {mean['test']['CDL1']:.3f}", flush=True)

    out = REPO_ROOT / "outputs" / "context_flag_test"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
