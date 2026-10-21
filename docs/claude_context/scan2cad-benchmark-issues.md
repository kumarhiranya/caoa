---
name: scan2cad-benchmark-issues
description: "Problems in the official Scan2CAD EvaluateBenchmark.py and CAOA Table 1, and how the user decided to report them (2026-10-20)"
metadata:
  type: project
---

Found 2026-10-20. The local `Scan2CAD/Routines/Script/EvaluateBenchmark.py` is byte-identical to upstream skanti/Scan2CAD. Per [[scan2cad-metric-fidelity]], it's replicated as-is; the user decides on any deviation.

- **GT reference point:** GT pose = inv(Mscan)·T·R·S·T(−center). Submissions, Scan2CAD's own aligner (DFGen grid spans [−0.5, 0.5]³, no centering) and Alignment2Mesh all use the plain CAD pose T·R·S, so a perfect submission has translation error |R·S·center|. On val GT that is median 10.8 cm, and >20 cm for 22.9% of objects. `center` = CAD bbox center (verified on meshes); `bbox` = half extents.
- **Scale error** |mean(s_pred/s_gt) − 1| lets anisotropic errors cancel. Infinite symmetry is approximated by 36 discrete steps. Matching is greedy in prediction order against the first same-category GT within thresholds. Missed objects count as failures. Class-mean is over the top 8 + "other" (display, table, bathtub, trashbin, sofa, chair, cabinet, bookshelf, other).
- **CAOA Table 1:** it has no display or trash-bin columns. The baseline rows (verified for Scan2CAD and FPFH vs. Scan2CAD paper Table 2) drop display and shift the later values one column left (CAOA's "other" = Scan2CAD's sofa 70.63, etc.; real other = 20.60). CAOA's own class-avg ≠ mean of its listed columns, e.g. the Sph row's columns average 74.99 vs. the reported 69.17.
- S2C-C annotation poses are identical to official `full_annotations.json` (14,225 models); only `scannet_instance_id` is added. Val has 3,184 GT models in total.

- Measured with the official protocol on val (312 scenes, 3,184 GT): perfect poses in the plain convention score 77.17% instance / 76.15% class (cabinet 29.2%); in the −center convention, 100%. Submitting only S2C-C-matched objects (the GT-instance setting) caps it at 76.54% / 74.07% (−center) or 61.56% / 58.34% (plain). The paper's GT row (86.54%) exceeds that cap, so it must have used legacy matched-only scoring.
- Implemented in `caoa/eval/scan2cad_benchmark.py`, with exact parity to upstream in `tests/test_scan2cad_benchmark.py` (verbatim copy in `tests/scan2cad_ref.py`).

**Decided by the user on 2026-10-20:**
1. The script stays unchanged. Report BOTH submission conventions, clearly labelled: plain CAD pose (faithful; how Scan2CAD's own 31.68% was scored) and −center.
2. Keep the official scale-mean and the 36-step infinite symmetry.
3. GT-instance results: report the official protocol (all val GT, misses counted) as the main number, plus the matched-only Scan2CAD-formula accuracy, labelled, for comparison with the paper's row.
4. Report all 9 official categories, with baselines copied from the original papers (not CAOA's shifted Table 1).

**How to apply:** any further deviation from the official script still needs the user's OK.
