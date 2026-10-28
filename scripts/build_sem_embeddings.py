"""SEM embeddings of every Scan2CAD CAD with the legacy symmetry checkpoint (P9 fix, user-approved 2026-10-28).

    uv run python scripts/build_sem_embeddings.py [--frame cad_bbox]

Legacy `embeddings.npy` was exported with inputs the checkpoint was not trained on and predicts every CAD as
not symmetric (docs/results/P9.md). `cad_bbox` feeds the CAD as the checkpoint's training pipeline did
(5,000 uniform surface points tiled to 8,192, grid = coord // 1 cm), but normalised with the CAD's own bounding
box (centred, longest side 1) instead of its GT box, so no annotation is used. One embedding per CAD, in the
legacy format {catid_idcad: (class probabilities, embedding)}: `<cache>/sem_embeddings_v1/<run>__cad_bbox/`.
"""

import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from caoa.data.legacy_sampling import sample_points
from caoa.data.pose_dataset import SEM_FRAMES, sem_embedding_path
from caoa.data.shapenet import load_cad_mesh, sample_uniform
from caoa.models.sem import SEMClassifier, load_legacy_sem, make_batch
from caoa.paths import load_paths

RUN = "S2Csym_emb128_bce_vox1cm_wd1e-03_b64_lr1e-04_bbcls-ptv3-v1m1-0-shapenet_v2_mini_ptv3cls_softmax_symbinary_cad_bce_adam_cosineannealing"
CAD_POINTS, NUM_POINTS, BATCH = 5000, 8192, 32


def cad_bbox_points(mesh, rng) -> np.ndarray:
    pts = sample_uniform(mesh, CAD_POINTS, rng=rng)
    lo, hi = pts.min(0), pts.max(0)
    return sample_points((pts - (lo + hi) / 2) / (hi - lo).max(), NUM_POINTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=RUN, help="legacy checkpoints/symmetry_ptv3 run")
    parser.add_argument("--frame", choices=[f for f in SEM_FRAMES if f != "legacy_export"], default="cad_bbox")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = load_paths()
    model = SEMClassifier()
    load_legacy_sem(model, paths.legacy_repo / "checkpoints/symmetry_ptv3" / args.run / f"{args.run}.pth")
    model = model.cuda().eval()
    with open(paths.legacy_repo / "scan2cad_annotator/s2c_cad_splits.pkl", "rb") as f:
        splits = pickle.load(f)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    out = {}
    for phase in ("train", "test"):
        items = splits[phase]
        correct = 0
        for start in tqdm(range(0, len(items), BATCH), desc=phase):
            chunk = items[start : start + BATCH]
            keys = ["_".join(path.split("ShapeNetV2/")[1].split("/")[:2]) for path, _ in chunk]
            clouds = [cad_bbox_points(load_cad_mesh(paths.shapenet, *k.split("_", 1)), rng) for k in keys]
            with torch.no_grad():
                r = model(make_batch(clouds, "cuda", grid="train"))
            probs, emb = r["probs"].cpu().numpy(), r["emb"].cpu().numpy()
            out.update({k: (p, e) for k, p, e in zip(keys, probs, emb)})
            correct += sum(int(p.argmax() == (sym != "__SYM_NONE")) for p, (_, sym) in zip(probs, chunk))
        print(f"{phase}: {len(items)} CADs, symmetry accuracy {correct / len(items):.4f}")

    dst = sem_embedding_path(paths, args.run, args.frame)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out)
    print(f"{len(out)} embeddings -> {dst}")


if __name__ == "__main__":
    main()
