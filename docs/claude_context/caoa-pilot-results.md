---
name: caoa-pilot-results
description: "Results of the 2026-10-20 legacy no-completion OCAM pilot (spconv port) on the 605-scene subset, under the legacy and official metrics"
metadata:
  type: project
---

Pilot run on 2026-10-20 (`outputs/legacy_nocomp/` in caoa). Settings mirror server run pose_mnk ..._mnkpv3_rotmat_t2q1s2. It trained on 3,080 pairs and was scored on 731 val pairs / 106 local val scenes (1,088 GT). Part of [[caoa-replication-project]].

- **Legacy metric:** 72.8% (t 0.941 / r 0.888 / s 0.822). The legacy server run on the full data got 71.8% (t 0.944 / r 0.884 / s 0.801), and the curves match step by step (ours 41.9% at 1k vs 43.3%). So the spconv port reproduces legacy.
- **Official protocol (misses counted):** plain 36.8% instance / 28.1% class; −center 48.2% / 37.8%. Matched-only with Scan2CAD formulas: plain 54.6% / 45.4%; −center 71.7% / 62.1%.
- The legacy pair filter (no ScanNet20 label and CAD outside the 8 classes → dropped) removes all 51 matched displays, 45 "other" and 11 trash bins in local val. Official ceiling: 67.2% with kept pairs vs 77.0% with all matched. Including all matched pairs would be a paper-mode switch, not yet done.
- 15k steps take ~1 h at ~4.4 it/s on the A6000.

**P6 re-scoring (2026-10-21, `outputs/rescore_completion/results.json`)**, S2C-C test set of 1,744 samples, legacy test() aggregation; per-category counts match legacy exactly.
- S2C-C: 46.835 / 25.533 vs logged 46.802 / 25.692.
- PCN: 161.537 / 137.363 vs 161.521 / 137.124.
- ShapeNet-34: 93.981 / 77.719 vs 94.014 / 77.893.
- **These three reproduce, so the port is faithful.**
- ShapeNet-55: checkpoint `ODGNet/ShapeNet55_models/ckpt-best.pth` is missing.
- Concat and SN-Indoor: the tests used ckpt-last at epochs 171 / 42, but the files on disk are now epochs 227 / 88 (training continued and overwrote them). So the tested weights are gone. On the current files we get concat 45.97 / 24.48 (logged 45.37 / 23.74) and SN-Indoor 62.95 / 39.24 (logged 66.19 / 45.66).
- **Context flag decided empirically (2026-10-21): +1/−1.** The legacy ctx checkpoint gives validate() CDL1 22.275 (stored 22.471) with +1/−1 vs 30.781 with 1/0. Fix 4's context input reproduces the lost Nov 2024 loader closely. `docs/results/P6.md` has the details.
