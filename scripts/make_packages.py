"""Build the downloadable archives: models and the redistributable data.

    uv run python scripts/make_packages.py [--release <caoa_data>/release_v1] [--dst <caoa_data>/packages]

Archives (zstd-compressed tar), each to be extracted in the repository root, where the default `configs/paths.yaml`
finds them:
* `caoa_models_v1.tar.zst`: the whole release folder (`release_v1/`: every checkpoint with its config and results);
* `caoa_models_v1_pipeline.tar.zst`: only what `scripts/align_scene.py` and the notebook need (CAPCM concat, SEM,
  final OCAM seed 777, retrieval with its database embeddings);
* `caoa_data_v1.tar.zst`: the data this project can redistribute, under `data/` and `cache/`:
  - `data/s2cc/s2cc_instance_ids.json`: the S2C-C annotation (ScanNet instance per Scan2CAD model); combine it with
    Scan2CAD's annotations with `scripts/build_s2cc_annotations.py`;
  - `data/s2cc/s2c_cad_splits.pkl` (CAD train / test split for the SEM) and `semantic_embeddings.npy` (CAPCM input);
  - `data/instances/{softgroup,sphericalmask}/val/`: instance predictions on the ScanNet val scans (masks over vertex
    indices, labels, confidences, per-point semantic labels);
  - `cache/sem_embeddings_v1/sem__cad_bbox/`: SEM embeddings of the Scan2CAD CADs.
Nothing derived from ScanNet or ShapeNet geometry is included; the README explains how to build those caches.
`packages.json` lists each archive with its size and SHA-256.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from caoa.paths import REPO_ROOT, load_paths
from caoa.data.pose_dataset import sem_embedding_path

PIPELINE = ["capcm/concat", "sem", "ocam/final/seed777", "retrieval", "class_synsets.json", "release.md", "manifest.json"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slim_predictions(src: Path, dst: Path) -> int:
    """Keep what `caoa.data.instances` reads: scan_id, pred_instances, semantic_preds (uint8)."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*_sg_preds.npz")):
        d = np.load(f, allow_pickle=True)["arr_0"].item()
        keep = {"scan_id": d["scan_id"], "semantic_preds": np.asarray(d["semantic_preds"]).astype(np.uint8),
                "pred_instances": [{k: p[k] for k in ("scan_id", "label_id", "conf", "pred_mask")} for p in d["pred_instances"]]}
        np.savez_compressed(dst / f.name, arr_0=np.array(keep, dtype=object))
        n += 1
    return n


def tar_zst(dst: Path, root: Path, members: list[str]):
    """tar of `members` (relative to `root`), compressed with zstd -19 (multithreaded)."""
    with tempfile.NamedTemporaryFile(suffix=".tar", dir=dst.parent, delete=False) as tmp:
        with tarfile.open(tmp.name, "w") as tar:
            for m in members:
                tar.add(root / m, arcname=m)
        subprocess.run(["zstd", "-19", "-T0", "-q", "-f", tmp.name, "-o", str(dst)], check=True)
        Path(tmp.name).unlink()
    dst.chmod(0o644)


def main():
    paths = load_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default=str(paths.cache.parent / "release_v1"))
    parser.add_argument("--dst", default=str(paths.cache.parent / "packages"))
    parser.add_argument("--instance-ids", required=True, help="s2cc_instance_ids.json (scripts/build_s2cc_annotations.py --extract)")
    args = parser.parse_args()
    release, dst = Path(args.release), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    tar_zst(dst / "caoa_models_v1.tar.zst", release.parent, [release.name])
    tar_zst(dst / "caoa_models_v1_pipeline.tar.zst", release.parent, [f"{release.name}/{m}" for m in PIPELINE])

    with tempfile.TemporaryDirectory(dir=dst) as tmp:
        root = Path(tmp)
        s2cc = root / "data/s2cc"
        s2cc.mkdir(parents=True)
        shutil.copy2(args.instance_ids, s2cc / "s2cc_instance_ids.json")
        shutil.copy2(paths.s2c_cad_splits, s2cc / "s2c_cad_splits.pkl")
        shutil.copy2(paths.semantic_embeddings, s2cc / "semantic_embeddings.npy")
        for source in ("softgroup", "sphericalmask"):
            n = slim_predictions(Path(getattr(paths, f"{source}_preds")), root / f"data/instances/{source}/val")
            print(f"{source}: {n} scenes")
        sem = root / "cache/sem_embeddings_v1/sem__cad_bbox"
        sem.mkdir(parents=True)
        shutil.copy2(sem_embedding_path(paths, "sem"), sem / "embeddings.npy")
        tar_zst(dst / "caoa_data_v1.tar.zst", root, ["data", "cache"])

    out = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(dst.glob("*.tar.zst"))}
    (dst / "packages.json").write_text(json.dumps(out, indent=1))
    for name, v in out.items():
        print(f"{name}: {v['bytes'] / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
