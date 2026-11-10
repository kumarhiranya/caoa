"""Build the Scan2CAD-Completion (S2C-C) annotations from Scan2CAD's own annotations.

    uv run python scripts/build_s2cc_annotations.py --scan2cad <Scan2CAD>/full_annotations.json \
        --instance-ids <data package>/s2cc/s2cc_instance_ids.json --out <path in configs/paths.local.yaml: s2c_annotations>

S2C-C = Scan2CAD's `full_annotations.json` with one added field per aligned model, `scannet_instance_id`: the ScanNet
instance (in the scene's `aggregation.json` numbering) the CAD is aligned to, -1 if none. The data package ships only
those ids ({scene: [id per aligned model, in Scan2CAD's order]}); Scan2CAD's annotations come from Scan2CAD. With
`--extract`, writes the ids file from an existing S2C-C file instead.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan2cad", help="Scan2CAD full_annotations.json")
    parser.add_argument("--instance-ids", required=True)
    parser.add_argument("--out", help="S2C-C annotations to write")
    parser.add_argument("--extract", default=None, help="an S2C-C annotations file: write --instance-ids from it")
    args = parser.parse_args()

    if args.extract:
        s2cc = json.load(open(args.extract))
        ids = {scene: [m["scannet_instance_id"] for m in a["aligned_models"]] for scene, a in s2cc.items()}
        json.dump(ids, open(args.instance_ids, "w"), separators=(",", ":"))
        print(f"{len(ids)} scenes, {sum(map(len, ids.values()))} models -> {args.instance_ids}")
        return

    ids = json.load(open(args.instance_ids))
    scans = {a["id_scan"]: a for a in json.load(open(args.scan2cad))}
    out = {}
    for scene, scene_ids in ids.items():
        a = scans[scene]
        if len(a["aligned_models"]) != len(scene_ids):
            raise SystemExit(f"{scene}: {len(a['aligned_models'])} aligned models in Scan2CAD, {len(scene_ids)} ids")
        for m, i in zip(a["aligned_models"], scene_ids):
            m["scannet_instance_id"] = i
        out[scene] = a
    json.dump(out, open(args.out, "w"))
    print(f"{len(out)} scenes, {sum(map(len, ids.values()))} models -> {args.out}")


if __name__ == "__main__":
    main()
