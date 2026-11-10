"""SEM embeddings of every Scan2CAD CAD, the OCAM's symmetry input.

    uv run python scripts/build_sem_embeddings.py --checkpoint outputs/sem/model_recalibrated.pth [--name sem]

The CAD as the SEM is trained: 5,000 uniform surface points tiled to 8,192, in its own bounding-box frame (centred,
longest side 1), grid = coord // 1 cm. One entry per CAD, {catid_idcad: (class probabilities, embedding)}, in
`<cache>/sem_embeddings_v1/<name>__cad_bbox/embeddings.npy`.
"""

import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from caoa.data.pose_dataset import SEM_FRAMES, sem_embedding_path
from caoa.data.shapenet import load_cad_mesh
from caoa.models.sem import SEMClassifier, cad_bbox_points, load_sem, make_batch, sem_checkpoint
from caoa.paths import load_paths

BATCH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="sem", help="cache name (a pose config's `sym_embedding.name`)")
    parser.add_argument("--checkpoint", required=True, help="SEM weights (plain state_dict), relative to the repo or absolute")
    parser.add_argument("--frame", choices=SEM_FRAMES, default="cad_bbox")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = load_paths()
    model = SEMClassifier()
    load_sem(model, sem_checkpoint({"checkpoint": args.checkpoint}))
    model = model.cuda().eval()
    with open(paths.s2c_cad_splits, "rb") as f:
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
                r = model(make_batch(clouds, "cuda"))
            probs, emb = r["probs"].cpu().numpy(), r["emb"].cpu().numpy()
            out.update({k: (p, e) for k, p, e in zip(keys, probs, emb)})
            correct += sum(int(p.argmax() == (sym != "__SYM_NONE")) for p, (_, sym) in zip(probs, chunk))
        print(f"{phase}: {len(items)} CADs, symmetry accuracy {correct / len(items):.4f}")

    dst = sem_embedding_path(paths, args.name, args.frame)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out)
    print(f"{len(out)} embeddings -> {dst}")


if __name__ == "__main__":
    main()
