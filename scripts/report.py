"""Regenerate every result table from `outputs/` into `docs/results/tables.md` (generated; commit it).

    uv run python scripts/report.py

Sources (missing ones are skipped and listed at the end):
* completion (Table 2): `outputs/capcm_*/test_results.json` and `train_log.json`;
* SEM: `outputs/sem/results*.json`;
* OCAM on GT instances: `outputs/ocam_<variant>[_seed778 | _seed779]/results.json` (mean ± sample std over the seeds);
* CAD retrieval: `outputs/retrieval/results.json`, `outputs/rerank_retrieval__ocam_final__k10/results.json`;
* full pipeline: `outputs/pipeline_<source>__ocam_<variant>[_seedN][_common]/{results,matching}.json`.
Metrics: S2C box = Scan2CAD's per-object criteria on the predicted box (the headline); CAOA = the paper's metric.
"""

import datetime
import json
import subprocess

import numpy as np

from caoa.paths import REPO_ROOT

OUT = REPO_ROOT / "outputs"
DST = REPO_ROOT / "docs" / "results" / "tables.md"
SEEDS = (777, 778, 779)
MISSING = []

PAPER_TABLE2 = {"S2C-C": (0.0602, 46.802, 25.692), "SN-Indoor + S2C-C (concat)": (0.0624, 45.371, 23.744),
                "S2C-C + context 100 cm": (0.2152, 22.471, 2.300)}
CAPCM = [("S2C-C", "capcm_s2cc"), ("SN-Indoor + S2C-C (concat)", "capcm_concat"), ("S2C-C + context 100 cm", "capcm_s2cc_context100")]
PAPER_TABLE1 = {"CAPCM+SEM-GT": (86.54, 78.15), "CAPCM+SEM-SG": (76.04, 66.84), "CAPCM+SEM-Sph": (77.51, 69.17),
                "CAPCM-Sph": (75.83, 67.45), "No-completion-Sph": (60.92, 48.63)}
ABLATION = [
    ("Completion only, no SEM (baseline)", "baseline"),
    ("+ SEM", "sem"),
    ("+ SEM + Siamese CAD branch, shared trunk (the paper's design)", "siamese_shared"),
    ("**+ SEM + Siamese CAD branch, separate trunks (final)**", "final"),
    ("final + the paper's loss (t2 / r3 / s2 + 5 x Chamfer)", "paper_loss"),
    ("final + the paper's loss and schedule", "paper_schedule"),
    ("No completion (instance points only, no SEM)", "no_completion"),
]
SOURCES = (("softgroup", "SoftGroup"), ("sphericalmask", "SphericalMask"))


def load(path):
    if not path.exists():
        MISSING.append(str(path.relative_to(REPO_ROOT)))
        return None
    return json.loads(path.read_text())


def seed_runs(name: str) -> list[str]:
    return [name] + [f"{name}_seed{s}" for s in SEEDS[1:]]


def mean_std(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan), len(v)


def fmt(ms, scale=100, digits=2):
    if ms is None:
        return "—"
    m, s, _ = ms
    return f"{m * scale:.{digits}f}" + (f" ± {s * scale:.{digits}f}" if not np.isnan(s) else "")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --- completion -----------------------------------------------------------------------------------------------------

def completion_section():
    rows = []
    for name, run in CAPCM:
        d = load(OUT / run / "test_results.json")
        if d is None:
            continue
        t = d["test"]
        log = load(OUT / run / "train_log.json") or []
        vals = [(e["epoch"], e["validate"]) for e in log if "validate" in e]
        best = min(vals, key=lambda x: x[1]["CDL1"]) if vals else None
        paper = PAPER_TABLE2[name]
        rows.append([name, f"{t['F-Score']:.4f} / {t['CDL1']:.2f} / {t['CDL2']:.2f}",
                     f"{best[1]['F-Score']:.4f} / {best[1]['CDL1']:.2f} / {best[1]['CDL2']:.2f} (epoch {best[0]})" if best else "—",
                     f"{paper[0]:.4f} / {paper[1]:.2f} / {paper[2]:.2f}"])
    return ("## Completion (paper Table 2)\n\nF-Score / CDL1 / CDL2 (x 1000) on the S2C-C val split (1,744 samples). Test = the "
            "`test()` aggregation on ckpt-best; validation = `validate()` (class-averaged), best epoch. The paper's context "
            "row is a best-validation value.\n\n"
            + table(["Training data", "Ours: test", "Ours: best validation", "Paper"], rows))


def sem_section():
    rows = []
    for f, label in (("results.json", "training run"), ("results_eval_only.json", "released weights")):
        d = load(OUT / "sem" / f)
        if not d:
            continue
        for key, what in (("final", "statistics from training"), ("final_recalibrated", "BatchNorm recalibrated")):
            if key in d:
                r = d[key]
                rows.append([f"{label}, {what}", f"{r['accuracy_all'] * 100:.1f}", f"{r['macro_f1_all'] * 100:.1f}", r["n_all"]])
    return ("## Symmetry embedding module (SEM)\n\nSymmetric / not symmetric on the val pairs' CADs (whole val set).\n\n"
            + table(["SEM", "Accuracy", "Macro-F1", "Samples"], rows))


# --- OCAM on GT instances -------------------------------------------------------------------------------------------

def ocam_metrics(run: str) -> dict | None:
    d = load(OUT / run / "results.json")
    if d is None:
        return None
    t1 = d["caoa"]["table1"]["per_class"]["total"]
    return {"box_inst": d["s2c_box"]["instance_mean"], "box_class": d["s2c_box"]["class_mean"],
            "caoa": d["caoa"]["scannet_class"]["avg"]["total_acc"], "t1_micro": t1["micro_avg"], "t1_macro": t1["macro_avg"],
            "crit": d["s2c_criteria/minus_center"]["instance_mean"]}


def ocam_seeds(variant: str):
    runs = [m for m in (ocam_metrics(r) for r in seed_runs(f"ocam_{variant}") if (OUT / r).exists() or r == f"ocam_{variant}") if m]
    if not runs:
        return None, 0
    return {k: mean_std([r[k] for r in runs]) for k in runs[0]}, len(runs)


def ablation_section():
    rows = []
    for name, variant in ABLATION:
        m, n = ocam_seeds(variant)
        if m is None:
            continue
        rows.append([name, n, fmt(m["box_inst"]) + " / " + fmt(m["box_class"]), fmt(m["caoa"]),
                     fmt(m["t1_micro"]) + " / " + fmt(m["t1_macro"]), fmt(m["crit"])])
    return ("## OCAM design (GT instances, GT CAD)\n\n1,864 val objects; mean ± std over seeds 777–779. S2C box: instance / "
            "class mean (Scan2CAD categories). CAOA: all objects, and in the paper's Table 1 grouping (avg / class avg). "
            "S2C criteria: on the GT CAD placed by the predicted box (−center convention).\n\n"
            + table(["OCAM", "Seeds", "S2C box", "CAOA", "CAOA, Table 1 grouping", "S2C criteria"], rows))


# --- full pipeline --------------------------------------------------------------------------------------------------

def pipeline_dir(source, variant, seed=777, common=False):
    return OUT / (f"pipeline_{source}__ocam_{variant}" + ("" if seed == 777 else f"_seed{seed}") + ("_common" if common else ""))


def pipeline_values(source, variant, common=False):
    """Over the seeds present: {final, gt_cad, ref} x {box inst/class, CAOA Table 1 micro/macro}, plus n."""
    out = []
    for seed in SEEDS:
        d = pipeline_dir(source, variant, seed, common)
        r = load(d / "results.json") if d.exists() or seed == 777 else None
        if not r:
            continue
        sel = {"final": r[f"{source}_class"]["rerank_sum"], "gt_cad": r[f"{source}_class"]["gt_cad"], "ref": r["gt_instances"]["gt_cad"]}
        out.append({"n": r["n_matched"], **{f"{k}_{m}": v for k, s in sel.items() for m, v in (
            ("box_inst", s["s2c_box"]["instance_mean"]), ("box_class", s["s2c_box"]["class_mean"]),
            ("t1_micro", s["caoa_table1"]["micro"]), ("t1_macro", s["caoa_table1"]["macro"]))}})
    if not out:
        return None, 0
    return {k: mean_std([o[k] for o in out]) for k in out[0]}, len(out)


def table1_section():
    m, n = ocam_seeds("final")
    rows = []
    if m:
        a, c = PAPER_TABLE1["CAPCM+SEM-GT"]
        rows.append(["CAPCM+SEM-GT", f"{a:.2f} / {c:.2f}", fmt(m["box_inst"]) + " / " + fmt(m["box_class"]),
                     fmt(m["t1_micro"]) + " / " + fmt(m["t1_macro"]), "—", "1,864 (all)", f"final, {n}"])
    for paper_row, source, variant in (("CAPCM+SEM-SG", "softgroup", "final"), ("CAPCM+SEM-Sph", "sphericalmask", "final"),
                                       ("CAPCM-Sph", "sphericalmask", "baseline"), ("No-completion-Sph", "sphericalmask", "no_completion")):
        v, n = pipeline_values(source, variant)
        if v is None:
            continue
        a, c = PAPER_TABLE1[paper_row]
        rows.append([paper_row, f"{a:.2f} / {c:.2f}", fmt(v["final_box_inst"]) + " / " + fmt(v["final_box_class"]),
                     fmt(v["final_t1_micro"]) + " / " + fmt(v["final_t1_macro"]),
                     fmt(v["ref_box_inst"]) + " / " + fmt(v["ref_box_class"]), f"{int(v['n'][0]):,}", f"{variant}, {n}"])
    return ("## Paper Table 1: alignment\n\nOurs: S2C box (instance / class mean) and the CAOA metric in the paper's Table 1 "
            "grouping (avg / class avg; the paper's column uses this metric). Predicted-instance rows: the S2C-C objects the "
            "source matches (IoU >= 0.5, conf >= 0.3); class-mapped CAD retrieval, re-ranked by the OCAM; the reference is the "
            "same OCAM on GT instances with the GT CAD, on the same objects. Mean ± std over the seeds.\n\n"
            + table(["Paper row", "Paper", "Ours: S2C box", "Ours: CAOA", "Reference: S2C box", "Objects", "OCAM, seeds"], rows))


def head_to_head_section():
    rows = []
    for source, name in SOURCES:
        for common, objects in ((False, "own matches"), (True, "matched by both")):
            v, n = pipeline_values(source, "final", common)
            if v is None:
                continue
            rows.append([name, objects, f"{int(v['n'][0]):,}", fmt(v["final_box_inst"]) + " / " + fmt(v["final_box_class"]),
                         fmt(v["gt_cad_box_inst"]), fmt(v["ref_box_inst"]), fmt(v["final_t1_micro"]) + " / " + fmt(v["final_t1_macro"]), n])
    return ("## SoftGroup vs SphericalMask\n\nFinal OCAM. Each source on its own matched objects, and both on the objects both "
            "match. S2C box (instance / class mean) with retrieval (final), with the GT CAD, and with GT instances (reference); "
            "CAOA in the Table 1 grouping. Mean ± std over the seeds.\n\n"
            + table(["Instances", "Objects", "N", "S2C box: final", "S2C box: GT CAD", "S2C box: GT instances", "CAOA: final", "Seeds"], rows))


def pipeline_detail_section(source, title):
    d = pipeline_dir(source, "final")
    r, m = load(d / "results.json"), load(d / "matching.json")
    lines = [f"## Full pipeline on {title} instances (final OCAM, seed 777)", ""]
    if m:
        rows = []
        for t, v in m.items():
            if t == "matched_objects":
                continue
            det = v.get("detection", {})
            rows.append([t, f"{v['capture'] * 100:.1f}", f"{v['n_matched']:,}", f"{v['n_predictions']:,}",
                         f"{det['precision'] * 100:.1f} / {det['recall'] * 100:.1f} / {det['f1']:.3f}" if det else "—"])
        lines += [table(["Confidence >=", "Capture of S2C-C objects (%)", "Matched", "Predictions",
                         "Detection vs ScanNet: precision / recall / F1"], rows), ""]
    if r:
        rows = []
        for row, label in (("gt_instances", "GT instances"), (f"{source}_class", f"{title}, class-mapped retrieval"),
                           (f"{source}_all", f"{title}, all-category retrieval")):
            for name, v in r.get(row, {}).items():
                if row != "gt_instances" and name not in ("gt_cad", "stage1", "rerank_sum", "oracle_topk"):
                    continue
                rows.append([label, name, f"{v['s2c_box']['instance_mean'] * 100:.2f} / {v['s2c_box']['class_mean'] * 100:.2f}",
                             f"{v['caoa_table1']['micro'] * 100:.2f} / {v['caoa_table1']['macro'] * 100:.2f}",
                             f"{v['s2c_criteria']['minus_center']['instance_mean'] * 100:.2f}",
                             f"{v['category_acc'] * 100:.1f}", f"{v['cd_l1_to_gt']:.1f}"])
        lines += [f"Alignment on the {r['n_matched']:,} matched objects (conf >= {r['conf']}). `rerank_sum`: the final selection; "
                  "`oracle_topk`: the top-k candidate closest to the GT CAD.", "",
                  table(["Instances", "CAD", "S2C box", "CAOA, Table 1 grouping", "S2C criteria", "Category correct (%)", "CD-L1 to GT CAD"], rows)]
    return "\n".join(lines)


# --- retrieval ------------------------------------------------------------------------------------------------------

def retrieval_section():
    d = load(OUT / "retrieval" / "results.json")
    lines = ["## CAD retrieval (GT instances)", ""]
    if d:
        rows = [[mode, f"{v['recall@1'] * 100:.1f} / {v['recall@10'] * 100:.1f} / {v['recall@50'] * 100:.1f}", f"{v['rank_median']:.0f}",
                 f"{v['category_acc'] * 100:.1f}", f"{v['cd_l1']:.1f} ({v['random_cd_l1']:.1f})"]
                for mode, v in d.items() if isinstance(v, dict) and "recall@1" in v]
        lines += ["Stage 1 (embedding): 1,864 val objects against 52,472 ShapeNetV2 models. Search: the GT CAD's synset (same) or all; "
                  "yaw1 / yaw8: query embedding from 1 or averaged over 8 yaw rotations.", "",
                  table(["Search / yaw", "Exact recall @1 / @10 / @50", "Median GT rank", "Category acc.", "CD-L1 top-1 to GT (random)"], rows), ""]
    r = load(OUT / "rerank_retrieval__ocam_final__k10" / "results.json")
    if r:
        rows = [[name, f"{v['exact_hit'] * 100:.1f}", f"{v['cd_l1_to_gt']:.1f}",
                 f"{v['s2c_box']['instance_mean'] * 100:.2f} / {v['s2c_box']['class_mean'] * 100:.2f}",
                 f"{v['caoa_total_acc'] * 100:.2f}", f"{v['s2c_criteria']['minus_center']['instance_mean'] * 100:.2f}"]
                for name, v in r.items() if isinstance(v, dict)]
        lines += [f"Stage 2: the final OCAM aligns the top {r['k']} (same synset, yaw8) and the best fit is kept (`rerank_sum`).", "",
                  table(["Selection", "Exact hit", "CD-L1 to GT CAD", "S2C box", "CAOA", "S2C criteria"], rows)]
    return "\n".join(lines)


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    sections = [
        f"# Result tables\n\nGenerated by `scripts/report.py` from `outputs/` on {datetime.date.today()} (commit {commit}). "
        "Do not edit by hand. Accuracies in %; mean ± sample std over the seeds present.",
        table1_section(),
        head_to_head_section(),
        ablation_section(),
        completion_section(),
        sem_section(),
        retrieval_section(),
        pipeline_detail_section("softgroup", "SoftGroup"),
        pipeline_detail_section("sphericalmask", "SphericalMask"),
    ]
    if MISSING:
        sections.append("## Missing sources\n\n" + "\n".join(f"- `{m}`" for m in sorted(set(MISSING))))
    DST.write_text("\n\n".join(sections) + "\n")
    print(f"wrote {DST.relative_to(REPO_ROOT)}; missing: {len(set(MISSING))}")


if __name__ == "__main__":
    main()
