# CAOA reproduction: results

Clean re-implementation of *CAOA: Completion-Assisted Object-CAD Alignment* (arXiv 2606.18429), checked against the
legacy research code. This page puts every number in the paper next to ours and explains each difference. Details are in
the per-phase notes (`docs/results/P*.md`). All tables are regenerated from the run outputs by `scripts/report.py`
into [`results/tables.md`](results/tables.md).

> **Draft (2026-11-06).** Rows marked ⏳ wait for the release runs (job 135765, expected 2026-11-06 evening).

## How we evaluate

- **Data:** the Scan2CAD-Completion (S2C-C) annotations only (user, 2026-10-29). The val set is legacy's 1,864 objects
  (reproduced exactly by the RelPose-cache class rule).
- **Objects that are not evaluated never count as failures.** On predicted instances, only S2C-C objects matched by a
  prediction (point IoU ≥ 0.5, one-to-one, confidence ≥ 0.3) are scored. Predictions without an S2C-C object and objects
  without a prediction are left out; the capture rate says how many objects were matched.
- **Metrics,** in % of objects:
  - **legacy:** the legacy CAOA metric (normalised frame, legacy rotation and scale tests). The paper's numbers use it.
    Table 1 uses its grouping (avg / class avg).
  - **S2C criteria:** Scan2CAD's per-object criteria (translation ≤ 20 cm, rotation ≤ 20°, scale ≤ 20%, symmetry-aware)
    on the same objects, unchanged code. They compare the CAD's own proportions and origin, so any retrieved CAD that is
    not the GT model scores ~35% at best.
  - **S2C box:** the same criteria on the predicted box instead of the CAD (CAD-independent), so GT-CAD and retrieved-CAD
    rows can be compared.
- **Leak-free inputs throughout.** Legacy's completions saw the GT pose (the instance was completed in its GT box frame);
  every number here uses completions made in the world frame (P8). Leaky numbers are not reported (user, 2026-11-03).
- **Release models only** for the headline rows (user, 2026-11-05/06): the completion model, SEM, OCAM and retrieval are
  this project's own trainings; no legacy checkpoint is used.

## Completion (paper Table 2)

F-Score / CDL1 / CDL2 (×1000) on the S2C-C val split; legacy `test()` aggregation on the best checkpoint.

| Training data | Paper | Ours |
|---|---|---|
| S2C-C | 0.0602 / 46.80 / 25.69 | 0.0594 / **45.00** / **23.06** |
| SN-Indoor + S2C-C (concat; the model used downstream) | 0.0624 / 45.37 / 23.74 | 0.0623 / **44.42** / **23.20** |
| S2C-C + context 100 cm (best validation, as the paper) | 0.2152 / 22.47 / 2.30 | 0.2161 / 22.45 / 2.31 |

Reproduced: the same or slightly better (single runs). Not retrained (re-scored legacy checkpoints within 0.07% in P6):
the PCN / ShapeNet-55 / ShapeNet-34 rows. The paper's SN-Indoor-only rows (51.19, 34.99) are in no legacy log.

## Symmetry embedding (SEM)

| | Accuracy | Macro-F1 |
|---|---|---|
| Paper | 95.8 | 94.1 |
| Ours: retrained, BatchNorm recalibrated (the release SEM), val | 98.3 | 97.6 |
| Ours: unseen CADs only (425) | 91.1 | 86.3 |

The paper's numbers are legacy's training-log values (no evaluation code computes them, P9). Our higher overall number
is memorisation of CADs seen in training; on unseen CADs ours and the legacy checkpoint are equal. Retraining needed one
fix (switch `--bn-recalibration train`): without it the model collapses in eval mode (66%).

## Alignment (paper Table 1)

Legacy metric, Table 1 grouping, avg / class avg. Ours: release models, class-mapped CAD retrieval with re-ranking, on
the matched objects; the GT-instance reference is on the same objects.

| Paper row | Paper | Ours | Objects evaluated |
|---|---|---|---|
| CAPCM+SEM-GT (GT instances + GT CAD, all objects) | 86.54 / 78.15 | ⏳ (current inputs: 83.53 ± 0.34 / 74.98 ± 1.23) | 1,864 |
| CAPCM+SEM-SG (SoftGroup) | 76.04 / 66.84 | ⏳ | ⏳ (78% captured) |
| CAPCM+SEM-Sph (SphericalMask) | 77.51 / 69.17 | ⏳ | ⏳ (94.5% captured) |
| CAPCM-Sph (no SEM, single-trunk OCAM) | 75.83 / 67.45 | ⏳ | ⏳ |
| No-completion-Sph | 60.92 / 48.63 | ⏳ | ⏳ |

SoftGroup vs SphericalMask on the objects both match (head-to-head): ⏳.

Why our GT row is lower than the paper's:
- **The GT-pose leak.** With legacy's leaky completions the same model scored ~4 points higher (P8); 86.54 used them.
- **The paper's GT row is not attainable on the 1,864 set** (its avg is not k/1,864 for any k, and no grouping of the
  set gives its columns), so it was computed on another sample set or edited (PLAN.md, paper vs codebase).
- The paper's predicted-instance rows are *below* ours. How legacy chose the evaluated objects for them is unknown
  (its SoftGroup evaluation code is lost); ours scores matched objects only, as decided above.

## OCAM design (paper §3; our P10 ablation, 3 seeds)

GT instances, GT CAD, 1,864 objects; mean ± std over seeds 777–779. Legacy / S2C box (instance mean).

| OCAM | Legacy | S2C box |
|---|---|---|
| Completion only, no SEM | 83.15 ± 0.28 | 82.65 ± 0.73 |
| + SEM | 84.23 ± 0.62 | 83.35 ± 0.73 |
| + Siamese CAD branch, shared trunk (the paper's design) | 80.35 ± 0.62 | 79.60 ± 1.74 |
| **+ Siamese CAD branch, separate trunks (ours, final)** | 83.53 ± 0.34 | 84.50 ± 0.54 |
| + the paper's loss (t2/r3/s2, 5 × CAD–object Chamfer) | 83.14 ± 0.63 | 84.80 ± 0.14 |
| + the paper's schedule (wd 1e-4, 150 epochs) | 81.46 ± 0.28 | 82.85 ± 0.14 |

- The paper's shared-weight Siamese trunk is clearly worse (−3.2 vs separate trunks, ~5 std); so is its schedule (−2.1).
- SEM, separate trunks and the paper's loss are within ~1 point of each other. We keep separate trunks + SEM with
  legacy's loss and schedule: best on the Scan2CAD and box criteria, lowest spread.
- The legacy code had no SEM-using or Siamese alignment checkpoint; these are new trainings of the paper's description.

## CAD retrieval (not in the paper's tables)

The legacy CAD-matching module is lost; the paper describes PointNet-descriptor retrieval. Ours (framework completion,
not a contribution; user, 2026-10-29): an InfoNCE embedding of the completion + instance against all 52,472 ShapeNetV2
models, then the OCAM aligns the top 10 and the best fit is kept. On GT instances: exact model in the top 10 for 35.7%,
CD-L1 of the top-1 to the GT CAD 53.5 (random 91.1); after re-ranking, the S2C box criteria equal the GT CAD's
(⏳ release numbers).

## Not reproduced / out of scope

- Legacy's own SoftGroup run and evaluation (E8): the cached SoftGroup val predictions are used instead (user,
  2026-11-04); SphericalMask is run from its official checkpoint (val AP 62.55 / AP50 82.36 vs the published 62.6 / 81.9).
- Inference timing (E10, the paper's 0.58 s per scene for SEM): measured per CAD (4.9 ms), not per scene.
- CIS2VR-era evaluations (scope rule).

## Release

`caoa_data/release_v1/` on the lab share (⏳): the CAPCM, SEM, OCAM (3 seeds) and retrieval checkpoints with their
configs, results and a SHA-256 manifest, plus the ShapeNetV2 database embeddings. `scripts/align_scene.py` runs it on a
scan with instance predictions; see the README.
