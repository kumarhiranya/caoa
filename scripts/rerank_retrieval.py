"""CAD retrieval, stage 2 (P11d), on the GT instances: re-rank the stage-1 top-k with the Siamese OCAM, choosing CAD
and pose together. The method and the reported selections are in `caoa/retrieval/pipeline.py`; this runs it on the 1,864
val pairs, with stage 1 restricted to the GT CAD's synset. Output: `outputs/rerank_<retrieval>__<pose>__k<k>/results.json`.

    uv run python scripts/rerank_retrieval.py [--k 10]
"""

import argparse
import json

import torch
import yaml

from caoa.data import scan2cad
from caoa.data.pairs import load_pairs
from caoa.data.pose_completion import load_pose_completions
from caoa.data.shapenet import load_shapenet_points
from caoa.paths import REPO_ROOT, load_paths
from caoa.retrieval import pipeline, search
from caoa.retrieval.data import cad_key, fixed_queries
from caoa.retrieval.model import CADRetrieval


def load_cfg(run):
    return yaml.safe_load(open(REPO_ROOT / "outputs" / run / "config.yaml"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="retrieval_embed_comp_inst")
    parser.add_argument("--pose", default="siamunshared_world_comp_concat_only_emb1024")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--yaws", type=int, default=8)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="first N val objects only (smoke test)")
    args = parser.parse_args()
    paths = load_paths()
    device = torch.device("cuda")
    rcfg, pcfg = load_cfg(args.retrieval), load_cfg(args.pose)
    out_dir = REPO_ROOT / "outputs" / f"rerank_{args.retrieval}__{args.pose}__k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    db = load_shapenet_points(paths.cache)
    keys = db["keys"].tolist()
    db_index = {k: i for i, k in enumerate(keys)}
    db_points = torch.from_numpy(db["points"]).to(device)
    comp_cfg = pcfg["data"]["completion"]
    completions = load_pose_completions(paths.cache, comp_cfg["name"], comp_cfg["frame"])
    pairs = load_pairs(paths.cache, "val", class_rule=pcfg["data"]["class_rule"])
    if args.limit:
        pairs = pairs[: args.limit]
    targets = torch.tensor([db_index[cad_key(p)] for p in pairs])

    rmodel = CADRetrieval(**rcfg["model"]).to(device)
    rmodel.load_state_dict(torch.load(REPO_ROOT / "outputs" / args.retrieval / "model_final.pth", map_location=device)["model"])
    db_emb = search.embed_points(rmodel.cad, db_points, device)
    queries = fixed_queries(pairs, completions, rcfg["data"]["num_points"], rcfg["data"]["inst_fraction"], rcfg["seed"])
    allowed = search.allowed_mask(db["catids"][targets.numpy()], db["catids"], "same", device)
    top = pipeline.stage1(rmodel, queries, db_emb, allowed, device, args.k, args.yaws)
    del rmodel, db_emb

    cols = [[keys[j] for j in row] + [cad_key(p)] for row, p in zip(top, pairs)]
    cand_keys = {key for row in cols for key in row}
    print(f"{len(pairs)} objects, k={args.k}: {len(cand_keys)} unique candidate CADs")
    cad_points, sym_embeddings, cad_box = pipeline.prepare_candidates(cand_keys, paths, device, args.workers)
    ocam = pipeline.load_ocam(pcfg, REPO_ROOT / "outputs" / args.pose / "model_final.pth", device)
    pred, target, nscale, noff, scores = pipeline.align(pairs, completions, cols, pcfg, ocam, cad_points, sym_embeddings, device)
    results = pipeline.evaluate_selections(pairs, cols, pred, target, nscale, noff, scores, scan2cad.load_annotations(paths.s2c_annotations),
                                           cad_box, db_points, db_index)
    for name, r in results.items():
        print(pipeline.summary_line(name, r))
    results.update({"k": args.k, "n_objects": len(pairs), "n_candidates": len(cand_keys)})
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
