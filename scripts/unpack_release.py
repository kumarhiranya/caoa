"""Make a downloaded release usable by the training / evaluation scripts: link its models into `outputs/<run>/` under
the run names the configs use (`capcm_*`, `sem`, `ocam_<variant>[_seed<N>]`, `retrieval`), with `checkpoint` entries
pointing at those runs, and copy the released results, so `scripts/report.py` reproduces the tables and the evaluation
scripts can re-score any model.

    uv run python scripts/unpack_release.py [--release release_v1] [--outputs outputs]

Files are symlinked (configs and results are copied). Existing run folders are left alone.
"""

import argparse
import shutil
from pathlib import Path

import yaml

from caoa.paths import REPO_ROOT

CAPCM = {"s2cc": "capcm_s2cc", "concat": "capcm_concat", "s2cc_context100": "capcm_s2cc_context100"}
SYM = {"name": "sem", "frame": "cad_bbox", "checkpoint": "outputs/sem/model_recalibrated.pth"}


def run_paths(cfg: dict) -> dict:
    if cfg.get("model", {}).get("sym_embedding"):
        cfg["model"]["sym_embedding"] = dict(SYM)
    if cfg.get("data", {}).get("completion"):
        cfg["data"]["completion"]["checkpoint"] = "outputs/capcm_concat"
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default=str(REPO_ROOT / "release_v1"))
    parser.add_argument("--outputs", default=str(REPO_ROOT / "outputs"))
    args = parser.parse_args()
    rel, out = Path(args.release).resolve(), Path(args.outputs)
    made = []

    def run(name, files: dict, config: dict | None = None):
        d = out / name
        if d.exists():
            print(f"skip {d} (exists)")
            return
        d.mkdir(parents=True)
        for dst, src in files.items():
            if not src.exists():
                continue
            if src.suffix in (".json", ".yaml"):
                shutil.copy2(src, d / dst)
            else:
                (d / dst).symlink_to(src)
        if config is not None:
            (d / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        made.append(name)

    for folder, name in CAPCM.items():
        s = rel / "capcm" / folder
        run(name, {f: s / f for f in ("ckpt-best.pth", "config.yaml", "test_results.json", "train_log.json")})
    run("sem", {"model_recalibrated.pth": rel / "sem/model.pth", **{f: rel / "sem" / f for f in ("results.json", "results_eval_only.json", "train_log.json")}})
    for vdir in sorted((rel / "ocam").iterdir()):
        for sdir in sorted(vdir.glob("seed*")):
            seed = int(sdir.name[4:])
            name = f"ocam_{vdir.name}" + ("" if seed == 777 else f"_seed{seed}")
            cfg = run_paths(yaml.safe_load((sdir / "config.yaml").read_text()))
            run(name, {f: sdir / f for f in ("model_final.pth", "results.json", "train_log.json")}, cfg)
    rcfg = run_paths(yaml.safe_load((rel / "retrieval/config.yaml").read_text()))
    run("retrieval", {f: rel / "retrieval" / f for f in ("model_final.pth", "results.json", "train_log.json")}, rcfg)
    for d in sorted((rel / "results").iterdir()) if (rel / "results").exists() else []:
        run(d.name, {f.name: f for f in d.iterdir()})
    print(f"{len(made)} runs -> {out}")


if __name__ == "__main__":
    main()
