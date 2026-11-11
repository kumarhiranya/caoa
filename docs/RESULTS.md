# CAOA: results

The results of this implementation next to every number in the paper (*CAOA – Completion-Assisted Object-CAD
Alignment*, 3DV 2026, [arXiv 2606.18429](https://arxiv.org/abs/2606.18429)). All tables are regenerated from the run
outputs by `scripts/report.py` into [`results/tables.md`](results/tables.md), which has more detail. The trained models
are on Zenodo ([doi:10.5281/zenodo.24121532](https://doi.org/10.5281/zenodo.24121532)).

## How we evaluate

- **Data:** the Scan2CAD-Completion (S2C-C) annotations. Val set: 1,864 objects in the 312 ScanNet val scans.
- **Predicted instances:** an S2C-C object is evaluated when a predicted instance matches it (point IoU ≥ 0.5,
  one-to-one, confidence ≥ 0.3). Unmatched objects and predictions without an S2C-C object are left out, not counted
  as failures. The capture rate reports how many objects were matched (SoftGroup 78.0%, SphericalMask 94.5%).
  Each predicted-instance row is shown with a reference: the same OCAM on the GT instances of the same objects,
  with the GT CAD.
- **Metrics** (% of objects passing):
  - **S2C box (headline):** Scan2CAD's per-object criteria (translation ≤ 20 cm, rotation ≤ 20°, scale ≤ 20%,
    symmetry-aware; the benchmark's own formulas) applied to the predicted oriented box. Doesn't depend on which CAD
    was retrieved. Instance mean / class mean over Scan2CAD's categories.
  - **CAOA:** the metric the paper's alignment numbers use. It is computed in each object's normalised frame, so the
    translation limit is 20% of the object's size. Its rotation test is lenient for symmetric objects: the smallest
    Euler-angle error over the symmetric poses. Reported in the paper's Table 1 grouping: avg / class avg.
  - **S2C criteria:** Scan2CAD's criteria on the placed CAD. They compare the CAD's own proportions and origin, so a
    retrieved CAD that is not the exact GT model rarely passes (about 35%). Reported in `results/tables.md`.
- **Instance predictions:** SoftGroup and SphericalMask, each run from its official code and released ScanNet
  checkpoint (val AP50 67.3 and 82.4, as published); see [`instance_predictions.md`](instance_predictions.md).
- **Seeds:** every alignment number is the mean ± std over three OCAM seeds (777–779).
- **Completions** come from the instance points as they are in the scan (world frame): no part of the GT pose reaches
  the completion network.

## Alignment (paper Table 1)

Final OCAM; predicted-instance rows use class-mapped CAD retrieval with re-ranking.

| Paper row | Paper (CAOA metric) | Ours: S2C box | Ours: CAOA | Reference (GT instances): S2C box | Objects |
|---|---|---|---|---|---|
| CAPCM+SEM-GT (GT instances, GT CAD) | 86.54 / 78.15 | **84.21 ± 0.31** / 79.84 ± 0.58 | 83.64 ± 0.28 / 75.48 ± 1.05 | — | 1,864 |
| CAPCM+SEM-SG (SoftGroup) | 76.04 / 66.84 | **84.26 ± 0.38** / 75.42 ± 0.64 | 82.91 ± 0.48 / 73.81 ± 1.33 | 86.53 ± 0.28 | 1,453 |
| CAPCM+SEM-Sph (SphericalMask) | 77.51 / 69.17 | **83.73 ± 0.26** / 78.06 ± 0.50 | 82.65 ± 0.80 / 72.95 ± 2.00 | 85.47 ± 0.28 | 1,762 |
| CAPCM-Sph (no SEM, no CAD input) | 75.83 / 67.45 | **81.54 ± 0.90** / 75.24 ± 2.03 | 82.69 ± 0.72 / 72.32 ± 0.91 | 83.30 ± 0.31 | 1,762 |
| No-completion-Sph | 60.92 / 48.63 | **74.16 ± 2.05** / 69.39 ± 3.06 | 73.42 ± 1.11 / 62.57 ± 1.38 | 74.42 ± 1.75 | 1,762 |

- **Every component helps, as in the paper.**
  - Completion: +7.4 box (74.2 → 81.5).
  - SEM with the Siamese CAD branch: +2.2 (81.5 → 83.7).
  - On the CAOA metric the last step is a tie (82.69 vs 82.65): its lenient rotation test hides the orientation
    errors the CAD branch fixes.
- **From GT to predicted instances** costs 1.7–2.3 box points with either instance source (reference vs final).
- **Our GT row is below the paper's; our predicted-instance rows are above it.**
  - The paper's GT value can't be obtained on the 1,864-object val set: its average isn't k / 1,864 for any k.
  - How the paper selected the objects for its predicted-instance rows isn't described. Ours scores matched objects
    only, as described above.

**SoftGroup vs SphericalMask** on the 1,443 objects both match (S2C box): SoftGroup 84.25 ± 0.32 vs SphericalMask
**85.68 ± 0.11** (GT-instance reference 86.5 for both). SphericalMask also matches 309 more objects (1,762 vs 1,453).
Those extra objects are harder, so its score on its own matches is lower (83.73 vs 84.26).

## OCAM design (paper §3)

GT instances, GT CAD, 1,864 objects, 3 seeds. The paper describes a Siamese network (one trunk shared by object and
CAD) with SEM, trained with a Chamfer term in the loss and its own schedule. Each step is tested separately:

| OCAM | S2C box | CAOA |
|---|---|---|
| Completion only, no SEM | 82.56 ± 0.35 / 77.20 ± 1.30 | 83.42 ± 0.14 |
| + SEM | 83.57 ± 0.66 / 78.92 ± 0.34 | **84.48 ± 0.65** |
| + SEM + Siamese CAD branch, shared trunk (the paper's design) | 75.91 ± 1.07 / 69.67 ± 0.93 | 78.17 ± 1.61 |
| **+ SEM + Siamese CAD branch, separate trunks (final)** | 84.21 ± 0.31 / 79.84 ± 0.58 | 83.64 ± 0.28 |
| final + the paper's loss (t2 / r3 / s2 + 5 × Chamfer) | **84.57 ± 0.45** / 80.30 ± 0.99 | 83.32 ± 0.11 |
| final + the paper's loss and schedule (wd 1e-4, 150 epochs) | 82.65 ± 1.24 / 78.21 ± 1.24 | 81.65 ± 0.21 |
| No completion (instance points only) | 73.30 ± 1.50 / 69.03 ± 1.96 | 72.57 ± 1.19 |

- **Separate trunks:** a CAD branch with its own trunk helps (+1.7 box over the baseline, +0.6 over SEM alone).
  Sharing one trunk between object and CAD loses 8.3 points.
- **SEM:** alone, it adds +1.0 box and +1.1 CAOA.
- **The paper's loss** is level with the final model (within one std). **The paper's schedule** on top of it is
  worse (−1.9).
- We keep separate trunks + SEM with the default loss and schedule: the lowest spread and the simpler loss.

## Completion (paper Table 2)

F-Score / CDL1 / CDL2 (× 1000) on the S2C-C val split.

| Training data | Paper | Ours |
|---|---|---|
| S2C-C | 0.0602 / 46.80 / 25.69 | 0.0594 / **45.00** / **23.06** |
| SN-Indoor + S2C-C (concat; the pipeline's model) | 0.0624 / 45.37 / 23.74 | 0.0623 / **44.42** / **23.20** |
| S2C-C + 100 cm context (best validation, as the paper) | 0.2152 / 22.47 / 2.30 | **0.2196 / 22.18 / 2.25** |

Single runs. Ours matches or beats the paper on CDL1 and CDL2. The paper's other Table 2 rows (PCN, ShapeNet-55 / 34,
SN-Indoor only) were not trained here.

## Symmetry Encoder Module (SEM)

| | Accuracy | Macro-F1 |
|---|---|---|
| Paper | 95.8 | 94.1 |
| Ours (val CADs) | **98.5** | **97.8** |

Training needs one addition: BatchNorm statistics recomputed in eval mode after training
(`caoa.models.sem.recalibrate_batchnorm`). Without it the model collapses in eval mode (66.5%), because the stored
statistics were collected with drop-path on. Many val CADs also occur in training, so on CADs never seen in
training the accuracy is lower.

## CAD retrieval (not in the paper's tables)

The paper retrieves candidates with PointNet++ features. This implementation learns its own embedding instead: the
completed object and every ShapeNetV2 CAD (52,472) are embedded in a joint space (InfoNCE), then the OCAM aligns the
top 10 and keeps the best fit.
- **Stage 1** (GT instances, same synset): exact model in the top 10 for 35.7%, top-1 Chamfer distance to the GT CAD
  53.5 (a random CAD of the synset: 91.1).
- **After re-ranking**, the S2C box score equals the GT CAD's: 84.87 for both. So retrieval costs no alignment
  accuracy, even though the exact model is found for only 6.6% of objects.
- **On predicted instances**, re-ranked retrieval scores within 0.2 box points of the GT CAD on average over the
  seeds: SoftGroup 84.26 vs 83.96, SphericalMask 83.73 vs 83.90 (`results/tables.md`).

## Release

`release_v1` (`scripts/make_release.py`; 144 files, 3.4 GB) contains:
- the three CAPCMs;
- the SEM;
- every OCAM of the tables above, for all three seeds;
- the retrieval model with its ShapeNetV2 database embeddings;
- the evaluation outputs;
- a SHA-256 manifest.

`scripts/align_scene.py` and `notebooks/single_scene_pipeline.ipynb` run it on a scan (see the README).
