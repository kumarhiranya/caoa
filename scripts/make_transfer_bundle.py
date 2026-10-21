"""Stage the data needed to continue this project on another machine (e.g. the DGX).

    uv run python scripts/make_transfer_bundle.py [--out /home/data/caoa_dgx_bundle] [--drive ...]

Writes two trees, `<out>/now/` (P7, P8 and the test suite) and `<out>/later/` (P9–P12 and
cache rebuilds). Both mirror the target layout under one <DATA_ROOT>, matching
`configs/paths.dgx.template.yaml`, so they can be sent one after the other into the same folder.
Entries are symlinks to the local data, not copies. The exception is the few files that live on
the external drive, which are copied (~0.46 GB) so the transfer doesn't need the drive mounted.
Send with `rsync -aL` (follows the links); the commands are in `<out>/MANIFEST.md`.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from caoa.paths import load_paths

DRIVE = Path("/media/hiranya/KHG Elements/mmlab-server-data/data/hiranya/Github/vr_reconstruction")
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".cu", ".cpp", ".h", ".txt", ".sh", ".md"}
CODE_MAX_BYTES = 5 * 2**20
SKIP_DIRS = {".git", "__pycache__", "experiments", "checkpoints"}
ODGNET_CKPTS = [  # ckpt-best of the CAPCM runs used downstream: S2C-C, concat (paper's final), S2C-C + context
    "ODGNet/experiments/UpTrans/S2CCompletion/s2ccompletion_upTransb1",
    "ODGNet/experiments/UpTrans_comp/Concat_SN_S2CC/concat_snaug_s2cc_upTransb_comp",
    "ODGNet/experiments/UpTrans_comp/S2CCompletion/s2ccompletion_upTransb1_comp",
]
SCANNET_FILES = ("_vh_clean_2.ply", "_vh_clean_2.labels.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json")
SYMMETRY_RUN = "S2Csym_emb128_bce_vox1cm_wd1e-03_b64_lr1e-04_bbcls-ptv3-v1m1-0-shapenet_v2_mini_ptv3cls_softmax_symbinary_cad_bce_adam_cosineannealing"


class Tree:
    """A staging tree: `link` adds a symlink, `copy` a real copy; both record sizes per item."""

    def __init__(self, root: Path):
        self.root = root
        self.items: dict[str, list[int]] = {}  # item -> [files, bytes]

    def _add(self, item: str, src: Path):
        files, size = (1, src.stat().st_size) if src.is_file() else _du(src)
        entry = self.items.setdefault(item, [0, 0])
        entry[0] += files
        entry[1] += size

    def link(self, item: str, src: Path, rel: str):
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_symlink():
            dst.symlink_to(src)
        self._add(item, src)

    def copy(self, item: str, src: Path, rel: str):
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif not dst.exists():
            shutil.copy2(src, dst)
        self._add(item, src)


def _du(d: Path) -> tuple[int, int]:
    files = size = 0
    for root, _, names in os.walk(d, followlinks=True):
        for n in names:
            files += 1
            size += os.stat(os.path.join(root, n)).st_size
    return files, size


def legacy_code(legacy: Path):
    for root, dirs, names in os.walk(legacy):
        rel_root = Path(root).relative_to(legacy)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (rel_root == Path(".") and d == "data")]
        for n in names:
            p = Path(root) / n
            if p.suffix in CODE_SUFFIXES and p.stat().st_size < CODE_MAX_BYTES:
                yield p


def build_now(t: Tree, paths, drive: Path):
    for name in ("pairs_v2", "completion_v1_spatial0cm", "completion_v1_spatial100cm", "pose_completions_v1"):
        t.link("caoa cache (pairs, completion r=0 / r=1 m, pose completions)", paths.cache / name, f"caoa_cache/{name}")

    legacy = paths.legacy_repo
    for p in legacy_code(legacy):
        t.link("legacy code and configs", p, f"legacy/{p.relative_to(legacy)}")
    for p in (legacy / "ODGNet/experiments").rglob("*"):
        if p.suffix in (".yaml", ".log") and p.is_file():
            t.link("legacy ODGNet experiment configs and logs", p, f"legacy/{p.relative_to(legacy)}")
    for run in ODGNET_CKPTS:
        t.link("legacy ODGNet ckpt-best x3 (S2C-C, concat, S2C-C+context)", legacy / run / "ckpt-best.pth", f"legacy/{run}/ckpt-best.pth")

    for key in ("s2c_annotations", "scan2cad_full_annotations", "scan2cad_cad_appearances", "semantic_embeddings"):
        src = getattr(paths, key)
        t.link("annotations and SEM embeddings", src, f"legacy/{src.relative_to(legacy)}")
    t.link("SEM splits and legacy symmetry embeddings", legacy / "scan2cad_annotator/s2c_cad_splits.pkl", "legacy/scan2cad_annotator/s2c_cad_splits.pkl")
    t.link("SEM splits and legacy symmetry embeddings", legacy / "data/s2c_symmetry", "legacy/data/s2c_symmetry")
    for f in ("aug_train_test_splits.pkl", "splits.json", "taxonomy.json"):
        t.link("ShapeNet split and taxonomy files", paths.shapenet / f, f"shapenet/ShapeNetV2/{f}")

    t.copy("legacy target tables (results/*.csv, from the drive)", drive / "results", "legacy/results")
    for run in sorted((drive / "checkpoints/pose_mnk").iterdir()):
        for f in run.iterdir():
            if f.suffix != ".pth":
                t.copy("legacy pose_mnk run logs and configs (from the drive, no weights)", f, f"legacy/checkpoints/pose_mnk/{run.name}/{f.name}")


def build_later(t: Tree, paths, drive: Path):
    for scene in sorted(paths.scannet_scans.iterdir()):
        files = [scene / f"{scene.name}{s}" for s in SCANNET_FILES]
        if scene.name.startswith("scene") and all(f.exists() for f in files):
            for f in files:
                t.link("ScanNet scenes, 4 files each", f, f"scannet/scans/{scene.name}/{f.name}")

    cads = {(m["catid_cad"], m["id_cad"]) for s in json.load(open(paths.scan2cad_full_annotations)) for m in s["aligned_models"]}
    for catid, idcad in sorted(cads):
        t.link("Scan2CAD CAD models (3,049)", paths.shapenet / catid / idcad, f"shapenet/ShapeNetV2/{catid}/{idcad}")

    t.copy("SEM checkpoints: symmetry (used by the pose configs) and PTv3 classification (from the drive)",
           drive / "checkpoints/symmetry_ptv3" / SYMMETRY_RUN, f"legacy/checkpoints/symmetry_ptv3/{SYMMETRY_RUN}")
    for run in (drive / "checkpoints/classification_ptv3").iterdir():
        t.copy("SEM checkpoints: symmetry (used by the pose configs) and PTv3 classification (from the drive)",
               run, f"legacy/checkpoints/classification_ptv3/{run.name}")

    sg = paths.legacy_repo / "data/objects/softgroup_preds/sg_raw"
    t.link("SoftGroup predictions, val (+ legacy val caches)", sg / "val", "legacy/data/objects/softgroup_preds/sg_raw/val")
    for f in sg.iterdir():
        if f.is_file() and ("_val" in f.name or f.suffix == ".json"):
            t.link("SoftGroup predictions, val (+ legacy val caches)", f, f"legacy/data/objects/softgroup_preds/sg_raw/{f.name}")


def manifest(out: Path, trees: dict[str, Tree]) -> str:
    lines = ["# CAOA transfer bundle", "",
             "Two trees with the same layout under one `<DATA_ROOT>` on the target machine. Symlinks are",
             "followed by `rsync -L`, so the actual files are sent.", ""]
    for tier, t in trees.items():
        total = sum(b for _, b in t.items.values())
        lines += [f"## `{tier}/`: {total / 1e9:.1f} GB", "", "| Item | Files | Size |", "|---|---:|---:|"]
        lines += [f"| {k} | {n:,} | {b / 1e9:.2f} GB |" for k, (n, b) in t.items.items()]
        lines.append("")
    lines += [
        "## Transfer", "",
        "```bash",
        "# on this machine (set DGX and DATA_ROOT)",
        f"rsync -aL --info=progress2 {out}/now/   $DGX:$DATA_ROOT/",
        f"rsync -aL --info=progress2 {out}/later/ $DGX:$DATA_ROOT/",
        "",
        "# on the target, in the caoa repo",
        'sed "s|<DATA_ROOT>|$DATA_ROOT|" configs/paths.dgx.template.yaml > configs/paths.local.yaml',
        "uv sync --extra train && uv run pytest -q",
        "```", "",
        "Not included (only needed if the scope grows): SN-Indoor partials (`ShapeNetV2-augmented`, 16 GB) and their",
        "CAD targets; SoftGroup train predictions (18 GB) and `sg_processed` (67 GB); ScanNet `.sens` / 2D zips.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/data/caoa_dgx_bundle")
    parser.add_argument("--drive", default=str(DRIVE), help="legacy repo copy on the external drive")
    args = parser.parse_args()

    out, drive, paths = Path(args.out), Path(args.drive), load_paths()
    trees = {"now": Tree(out / "now"), "later": Tree(out / "later")}
    build_now(trees["now"], paths, drive)
    build_later(trees["later"], paths, drive)
    text = manifest(out, trees)
    (out / "MANIFEST.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
