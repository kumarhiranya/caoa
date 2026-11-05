"""P12: regenerate every result table from `outputs/` into `docs/results/tables.md` (generated; commit it).

Sources (missing ones are skipped and listed at the end):
* completion (Table 2): `outputs/<run>/test_results.json` of the P7 retrains;
* alignment on GT instances (Table 1 GT row, P8 / P10 ablation): `outputs/ab_<run>/results.json` when it exists (the
  `--eval-only` re-evaluation with every current report key), else `outputs/<run>/results.json`; seeds `<run>_seed778`,
  `<run>_seed779` are averaged with the original (seed 777) run: mean ± sample std over the seeds present;
* CAD retrieval (P11d): `outputs/retrieval_embed_comp_inst/results.json`, `outputs/rerank_*/results.json`;
* full pipeline on SoftGroup instances (P11): `outputs/p11_softgroup/{results,matching}.json`.
Leaky (GT-frame completion) runs are not reported (user, 2026-11-03).

    uv run python scripts/report.py
"""

import datetime
import json
import subprocess

import numpy as np

from caoa.paths import REPO_ROOT

OUT = REPO_ROOT / "outputs"
DST = REPO_ROOT / "docs" / "results" / "tables.md"
SEEDS = ("", "_seed778", "_seed779")
MISSING = []

# Paper values (docs/PLAN.md § Reproduction targets)
PAPER_TABLE2 = {"S2C-C": (0.0602, 46.802, 25.692), "Concat SN-Indoor + S2C-C": (0.0624, 45.371, 23.744),
                "S2C-C + context 100 cm": (0.2152, 22.471, 2.300)}
PAPER_TABLE1 = {"CAPCM+SEM-GT": (86.54, 78.15), "CAPCM+SEM-SG": (76.04, 66.84), "CAPCM+SEM-Sph": (77.51, 69.17),
                "CAPCM-Sph": (75.83, 67.45), "No-completion-Sph": (60.92, 48.63)}

COMPLETION = [("S2C-C", "legacy_s2cc"), ("Concat SN-Indoor + S2C-C", "legacy_concat"), ("S2C-C + context 100 cm", "legacy_s2cc_ctxt1")]
ABLATION = [
    ("Baseline: completion only, no SEM", "world_comp_concat_only_emb1024"),
    ("+ SEM", "semcad_world_comp_concat_only_emb1024"),
    ("+ SEM + Siamese, shared trunk (paper)", "siamshared_world_comp_concat_only_emb1024"),
    ("**+ SEM + Siamese, separate trunks (final)**", "siamunshared_world_comp_concat_only_emb1024"),
    ("+ paper loss (t2/r3/s2 + 5 × Chamfer)", "chamfer_siamunshared_world_comp_concat_only_emb1024"),
    ("+ paper schedule", "paperschedule_chamfer_siamunshared_world_comp_concat_only_emb1024"),
]
P8_VARIANTS = [
    ("S2C-C completion, completion only, emb 1024", "world_comp_s2cc_only_emb1024"),
    ("S2C-C completion, completion only, emb 512", "world_comp_s2cc_only_emb512"),
    ("S2C-C completion, completion only, emb 256", "world_comp_s2cc_only_emb256"),
    ("S2C-C completion, instance + completion, emb 1024", "world_comp_s2cc_inst_emb1024"),
    ("Concat completion, completion only, emb 1024", "world_comp_concat_only_emb1024"),
    ("Concat completion, instance + completion, emb 1024", "world_comp_concat_inst_emb1024"),
]


def load(path):
    if not path.exists():
        MISSING.append(str(path.relative_to(REPO_ROOT)))
        return None
    return json.loads(path.read_text())


def pose_metrics(run: str) -> dict | None:
    d = load(OUT / f"ab_{run}" / "results.json") if (OUT / f"ab_{run}" / "results.json").exists() else load(OUT / run / "results.json")
    if d is None:
        return None
    log = json.loads((OUT / run / "train_log.json").read_text()) if (OUT / run / "train_log.json").exists() else {}
    t1 = d["legacy"]["table1"]["per_class"]["total"]
    crit = d.get("s2c_criteria/minus_center") or d.get("matched/minus_center")
    plain = d.get("s2c_criteria/plain") or d.get("matched/plain")
    m = {
        "best": max(log["s2c_total"]) if log.get("s2c_total") else np.nan,
        "legacy": d["legacy"]["scannet_class"]["avg"]["total_acc"],
        "t1_micro": t1["micro_avg"], "t1_macro": t1["macro_avg"],
        "c_inst": crit["instance_mean"], "c_class": crit["class_mean"], "plain": plain["instance_mean"],
    }
    box = d.get("s2c_box")
    m["box_inst"], m["box_class"] = (box["instance_mean"], box["class_mean"]) if box else (np.nan, np.nan)
    return m


def seeded(run: str):
    runs = [m for m in (pose_metrics(run + s) if (OUT / (run + s)).exists() or s == "" else None for s in SEEDS) if m]
    if not runs:
        return None, 0
    return {k: (np.nanmean([r[k] for r in runs]), np.nanstd([r[k] for r in runs], ddof=1) if len(runs) > 1 else np.nan) for k in runs[0]}, len(runs)


def fmt(mean_std, scale=100, digits=2):
    m, s = mean_std
    if np.isnan(m):
        return "—"
    return f"{m * scale:.{digits}f}" + (f" ± {s * scale:.{digits}f}" if not np.isnan(s) else "")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def completion_section():
    rows = []
    for name, run in COMPLETION:
        d = load(OUT / run / "test_results.json")
        if d is None:
            continue
        t, v = d["test"], d["validate"]
        paper = PAPER_TABLE2[name]
        rows.append([name, f"{t['F-Score']:.4f} / {t['CDL1']:.2f} / {t['CDL2']:.2f}", f"{v['F-Score']:.4f} / {v['CDL1']:.2f} / {v['CDL2']:.2f}",
                     f"{paper[0]:.4f} / {paper[1]:.2f} / {paper[2]:.2f}"])
    return ("## Table 2: completion (CAPCM retrains, P7)\n\nF-Score / CDL1 / CDL2 (×1000) on the S2C-C val split (1,744 samples). "
            "Test = legacy `test()` aggregation on ckpt-best; validation = legacy `validate()` (class-averaged), best epoch. The paper's "
            "context row is a best-validation value.\n\n"
            + table(["Training data", "Ours: test", "Ours: best validation", "Paper"], rows))


def ablation_section(entries, title, note):
    rows = []
    for name, run in entries:
        m, n = seeded(run)
        if m is None:
            continue
        rows.append([name, n, fmt(m["best"], digits=1), fmt(m["legacy"]), fmt(m["t1_micro"]) + " / " + fmt(m["t1_macro"]),
                     fmt(m["c_inst"]) + " / " + fmt(m["c_class"]), fmt(m["plain"]), fmt(m["box_inst"]) + " / " + fmt(m["box_class"])])
    return f"## {title}\n\n{note}\n\n" + table(
        ["Run", "Seeds", "Legacy best (training)", "Legacy final", "Table 1 grouping: avg / class avg", "S2C criteria −center",
         "S2C criteria plain", "S2C box"], rows)


def retrieval_section():
    d = load(OUT / "retrieval_embed_comp_inst" / "results.json")
    lines = ["## CAD retrieval over ShapeNetV2 (P11d), GT instances", ""]
    if d:
        rows = [[mode, f"{v['recall@1'] * 100:.1f} / {v['recall@10'] * 100:.1f} / {v['recall@50'] * 100:.1f}", f"{v['rank_median']:.0f}",
                 f"{v['category_acc'] * 100:.1f}", f"{v['cd_l1']:.1f} ({v['random_cd_l1']:.1f})"]
                for mode, v in d.items() if isinstance(v, dict) and "recall@1" in v]
        lines += ["Stage 1 (embedding), 1,864 val objects, 52,472 database CADs.", "",
                  table(["Search / yaw averaging", "Exact recall @1 / @10 / @50", "Median GT rank", "Category acc.", "CD-L1 top-1 vs GT (random)"], rows), ""]
    for f in sorted(OUT.glob("rerank_*k10/results.json")):
        r = json.loads(f.read_text())
        rows = [[name, f"{v['exact_hit'] * 100:.1f}", f"{v['cd_l1_to_gt']:.1f}", f"{v['legacy_total_acc'] * 100:.2f}",
                 f"{v['s2c_criteria']['minus_center']['instance_mean'] * 100:.2f}",
                 f"{v['s2c_box']['instance_mean'] * 100:.2f} / {v['s2c_box']['class_mean'] * 100:.2f}"]
                for name, v in r.items() if isinstance(v, dict)]
        lines += [f"Stage 2 (`{f.parent.name}`), k = {r['k']}.", "",
                  table(["Selection", "Exact hit", "CD-L1 to GT CAD", "Legacy", "S2C criteria −center", "S2C box"], rows), ""]
    return "\n".join(lines)


def p11_section():
    r, m = load(OUT / "p11_softgroup" / "results.json"), load(OUT / "p11_softgroup" / "matching.json")
    lines = ["## Full pipeline on SoftGroup instances (P11)", ""]
    if m:
        rows = []
        for t, v in m.items():
            det = v.get("detection", {})
            rows.append([t, f"{v['capture'] * 100:.1f}", v["n_predictions"],
                         (f"{det['precision'] * 100:.1f} / {det['recall'] * 100:.1f} / {det['f1']:.3f}" if det else "—")])
        lines += [table(["Confidence ≥", "Capture of S2C-C objects (%)", "Predictions", "Detection vs ScanNet: precision / recall / F1"], rows), ""]
    if r:
        rows = []
        for row, label in (("gt_instances", "GT instances"), ("softgroup_class", "SoftGroup, class-mapped retrieval"), ("softgroup_all", "SoftGroup, all-category retrieval")):
            for name, v in r.get(row, {}).items():
                if row != "gt_instances" and name not in ("gt_cad", "stage1", "rerank_sum"):
                    continue
                t1 = v.get("legacy_table1")
                rows.append([label, name, f"{v['legacy_total_acc'] * 100:.2f}", f"{t1['micro'] * 100:.2f} / {t1['macro'] * 100:.2f}" if t1 else "—",
                             f"{v['s2c_criteria']['minus_center']['instance_mean'] * 100:.2f}",
                             f"{v['s2c_box']['instance_mean'] * 100:.2f} / {v['s2c_box']['class_mean'] * 100:.2f}"])
        lines += [f"Alignment on the {r['n_matched']} matched instances (conf ≥ {r['conf']}).", "",
                  table(["Instances", "CAD", "Legacy", "Table 1 grouping: avg / class avg", "S2C criteria −center", "S2C box"], rows)]
    return "\n".join(lines)


def paper_table1_section():
    rows = [[k, f"{a:.2f} / {c:.2f}"] for k, (a, c) in PAPER_TABLE1.items()]
    return ("## Paper Table 1 (for reference)\n\nAvg / class avg. Our counterparts: CAPCM+SEM-GT → the GT-instance ablation above "
            "(Table 1 grouping column); CAPCM+SEM-SG → the SoftGroup rows of P11; the Sph rows are **not reproduced (pending: "
            "SphericalMask, to be run in a fresh job)**.\n\n" + table(["Paper row", "Avg / class avg"], rows))


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    sections = [
        f"# Result tables\n\nGenerated by `scripts/report.py` from `outputs/` on {datetime.date.today()} (commit {commit}). Do not edit by hand. "
        "Accuracies in %. All alignment numbers use leak-free completions; mean ± sample std over the seeds present.",
        completion_section(),
        ablation_section(ABLATION, "Alignment on GT instances: P10 ablation (Table 1 GT row)",
                         "Concat completions (leak-free), completion only, emb 1024; 1,864 val objects. Legacy = `scannet_class` micro of "
                         "`model_final`; S2C criteria / box: instance / class mean."),
        ablation_section(P8_VARIANTS, "Alignment on GT instances: completion variants without SEM (P8, leak-free)",
                         "Same report; single runs unless seeds were added."),
        paper_table1_section(),
        retrieval_section(),
        p11_section(),
    ]
    if MISSING:
        sections.append("## Missing sources\n\n" + "\n".join(f"- `{m}`" for m in sorted(set(MISSING))))
    DST.write_text("\n\n".join(sections) + "\n")
    print(f"wrote {DST.relative_to(REPO_ROOT)}; missing: {len(set(MISSING))}")


if __name__ == "__main__":
    main()
