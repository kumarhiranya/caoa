# CAOA rework plan

Clean re-implementation of *CAOA: Completion-Assisted Object-CAD Alignment*
(arXiv 2606.18429) in this repo (`caoa`; `caoa-repro` until 2026-10-21), using the legacy repo
(`~/research/Github/vr_reconstruction`) only as a reference.

Living document: update the status column and the phase sections as work lands.
Last updated: 2026-10-28 (P8 leak quantified and fixed; P9 port done, legacy SEM embeddings found degenerate and regenerated; P10a running). The project moved to the DGX (the workstation GPU is shared). Data comes from the transfer bundle (`scripts/make_transfer_bundle.py`; see `CLAUDE.md`). The repo is https://github.com/kumarhiranya/caoa.

## Ground rules

1. **Recreate first, fix later.** Every evaluation the legacy CAOA code used is
   preserved and reproduced as it was. That includes its metric, its report
   format and the sample set behind each reported number (see the
   [legacy evaluation inventory](#legacy-evaluation-inventory)). Each phase's
   first gate is reproducing the legacy numbers with the legacy evaluation.
   Fixes come afterwards, as config switches (see the
   [switch register](#switch-register)), and only with the user's agreement.
   **The codebase wins over the paper:** where they disagree, we reproduce what
   the code did. Paper mismatches are recorded under
   [paper vs codebase mismatches](#paper-vs-codebase-mismatches) but not chased,
   since the paper may include post-processing or corrections.
2. **The official Scan2CAD benchmark is reported in addition.** It is
   replicated as-is: both submission conventions, the scale mean, 36-step
   infinite symmetry. Any deviation from the official script needs the user's
   approval.
3. **Scope is CAOA's evaluations only.** CIS2VR-era evaluations are out of
   scope: `tools/eval/evaluate_alignment_scan2cad.py`, `mnk_eval.py`, and the
   retrieval evals in `eval_utils.py` / `db_utils.py`. They are ported only if a
   CAOA number turns out to depend on them.
4. **Every phase ends with a validation gate.** A phase is done only when its
   checks pass, or when the user accepts a documented deviation. Check types:
   - **parity**: tests against verbatim copies of legacy or upstream code, kept in `tests/*_ref.py`;
   - **sanity**: numeric or visual checks on the data;
   - **reproduction**: matching a [legacy target](#reproduction-targets) within a stated tolerance;
   - **regression**: the whole test suite still passes.
5. **Disk is tight.** No bulk extraction, copy or cache without a size estimate
   and the user's OK.
6. **Results are recorded.** Each run writes `outputs/<name>/results.json` plus
   the legacy report files. Each phase adds a note to `docs/results/` and
   updates this plan.

## Overview

| Phase | Scope | Status | Gate (summary) |
|---|---|---|---|
| P0 | Audit of paper and legacy code, data inventory, decisions | **Done** | Findings recorded |
| P1 | Repo, environment, ScanNet / Scan2CAD data layer, object–CAD pairs | **Done** | Parity with SoftGroup prep and legacy box maths |
| P2 | Evaluation core: legacy per-object metric, official Scan2CAD benchmark | **Done** | Exact parity with legacy and upstream scripts |
| P3 | OCAM object-only (legacy `mnkpv3`) on spconv: pilot | **Done** | Superseded by the P3b re-run |
| P3b | Legacy evaluation parity: legacy sample set, legacy reports, Table 1 grouping | **Done** ([results](results/P3b.md)) | Exact 1,864 test set; parity tests; pilot 73.0% vs server 71.2–71.8% |
| P4 | Data completion: CAD samplers, scene context; full ScanNet | **Code done; data local, caches built** (2026-10-21; `scene0212_00` unavailable) | CAD↔scan sanity; context parity (both variants) |
| P5 | Completion datasets: S2C-Completion (completion view), SN-Indoor | **Code done; caches built** (r = 0: 2.0 GB, r = 1 m: 2.8 GB; train 6,220 / val 1,744) | Parity with recovered legacy loaders |
| P6 | CAPCM model: ODGNet port + legacy completion evaluation | **Done** ([results](results/P6.md)) | Op/metric parity; re-scored `ODG_*_S2CC.csv` |
| P7 | CAPCM training: **S2C-C only** (user, 2026-10-21), **on the DGX** | **Done** ([results](results/P7.md)): test CDL1 45.00 vs 46.80; accepted by the user 2026-10-27 | Val curve vs `results/legacy_curves/s2ccompletion_upTransb1_val.csv`; ckpt-best test CDL1 vs 46.80 |
| P8 | OCAM + completion: legacy reproduction, then the leak-free mode | **Legacy reproduction done; leak quantified and fixed** ([results](results/P8.md)): 87.8 / 81.9 / 85.6 / 82.7 vs 87.8 / 81.3 / 85.0 / 81.3; leak-free completion-only 82.6–83.1 (−4), inst+comp 80.9–81.8 (≤ −1.3); emb 512 / 256 leak-free running | `pose_mnk` logs, `comp.csv` / `compinst.csv` |
| P9 | SEM: PTv3 symmetry classifier + legacy `eval_sym` | **Port done** ([results](results/P9.md)); legacy `embeddings.npy` degenerate (wrong export inputs), regenerated (`cad_bbox`, user 2026-10-28); retraining open | 95.8% accuracy; legacy embeddings reproduced |
| P10 | OCAM paper model: symmetry embedding, Siamese CAD+object, Eq. 5 loss | **P10a running** (2026-10-28: SEM `cad_bbox` + concat completion-only, legacy and world completion frames); P10b–d planned | Ablation deltas under legacy + official evaluation |
| P11 | Predicted instances: legacy SoftGroup evaluation, then the official pipeline | Planned | `sg_mnk_comp.csv`, `perclass_results.csv` |
| P12 | Full-scale runs, Table 1 / Table 2 report, release | Planned | All rows reproduced or explained |

Dependencies:
- P3b is next. It does not need new data, except its last validation step, which needs P4d-val.
- P4 → P5 → P6 → P7 → P8.
- P9 needs only P4a and can run in parallel with P5–P8.
- P10 needs P8 and P9. P11 needs P10.
- P12 needs everything, plus the full ScanNet extraction (P4d).

Data scope (after P4d-val, 2026-10-20): all 312 val scenes and 499 of 1,201
train scenes are extracted. Under the legacy class rule (`relpose_cache`) that
is the **complete legacy test set (1,864, verified sample-for-sample against
`inst.csv`)** and a 2,733-pair train subset (legacy: 6,671). Evaluation numbers
are comparable to legacy; training-data size is not.

---

## Legacy evaluation inventory

Every evaluation CAOA's legacy code used, where it lives, what it outputs, and
which phase reproduces it.

| # | Legacy evaluation | Code | Output / report format | Phase | Status |
|---|---|---|---|---|---|
| E1 | Per-object pose accuracy (20% / 20° / 20%, normalised frame, legacy rotation and scale errors) | `utils/eval_utils.py`: `calc_pose_accuracy_s2c_nocls`, `calc_rotation_accuracy`, `calc_rotation_diff` | per-sample pass flags, mean rotation error | P2 | ✅ exact parity |
| E2 | Pose eval driver | `utils/training_utils.py:2020` `eval_pose_s2c_v2` (`per_class_results=True`, `time_inference=True` in eval mode) | MAE (pose / t / q / s, sklearn), "Mean scale % error" print, per-class table **translation, rotation, scale, total, support + `macro_avg` / `micro_avg` rows** → CSV, inference ms/sample | P3b | ❌ partial |
| E3 | Per-class accuracy helper | `eval_utils.py:149` `per_class_accuracy` | per-class, macro and micro average | P3b | ⚠️ macro/micro of the total only |
| E4 | Training loop + training-time validation | `training_utils.py:2448` `train_pose9dof` | checks on the counter *before* increment (stats every 150, checkpoint every 1k, validation every 1k incl. step 15,000); no break inside the epoch, so training runs to the end of the epoch containing step 15,000, with the cosine LR stepping past `T_max`; logs MAE + `s2c_t/q/s/total` (`m_train_log.json` keys; JSON written only at stat steps, so the step-15,000 validation never reached the file) | P3b | ⚠️ accuracies only |
| E5 | Older per-class metric | `eval_utils.py:91` `calc_pose_accuracy_s2c` | 8 `VALID_CLASSES`, support-weighted avg + class avg, each term rounded to 4 decimals → `perclass_results.csv` | P3b (port); used in P11 | ❌ |
| E6 | GT-instance sample set behind reported numbers | cache shared by `S2CCompletionPoseV2` and `S2CCompletionRelPose` (`custom_datasets_pose.py:4831` / `:5555`) | 1,864 test / 6,671 train; RelPose class rule | P3b | ❌ V2 rule used |
| E7 | Table 1 per-class grouping | no table-building code. **Decided (C):** E8's CAD-category name (`shapenet2scannet_name(SHAPENET_SYNSET_NAME[catid])`, else `sorted(names)[0]`), folded to bathtub→bath, bookshelf, cabinet, chair, sofa, table; everything else → other | bath, bookshelf, cabinet, chair, other, sofa, table, class avg, avg | P3b | decided |
| E8 | SoftGroup-prediction evaluation | `SGPredsS2CCompletionPoseVal` (`custom_datasets_pose.py:6236`) + `eval_pose_s2c_v2` | instance↔CAD overlap matching, "instance capture accuracy", per-class table grouped by **CAD taxonomy name** → `sg_mnk_comp.csv` | P11 | ❌ |
| E9 | Completion evaluation | `ODGNet/utils/metrics.py` (`Metrics`: F-Score th=0.01, CD-L1, CD-L2 ×1e3); ODGNet `runner.validate()` (class-averaged) and `test()` (mixed averaging) | per-taxonomy CSV `Taxonomy,#Sample,F-Score,CDL1,CDL2,#ModelName` → `ODG_*_S2CC.csv`; `[TEST]` log lines | P6 | ❌ |
| E10 | Completion inference timing | `S2CCompletionPoseV2.infer_completion_odg` (`time_inference`) | average inference time per sample | P8 | ❌ |
| E11 | Symmetry evaluation | `training_utils.py:2307` `eval_sym` | argmax accuracy on string labels (`num2label`); returns gt / pred. Its test loader drops the last partial batch (shuffled, batch 64: 95.8 = 552/576). The paper's F1 (94.1) is the training log's macro-F1 (94.05). | P9 | port done; legacy sample set open |

Additions, not legacy, always labelled as such: the official Scan2CAD protocol
(both conventions), matched-only Scan2CAD-formula accuracy, and the
normalised-frame Scan2CAD-formula proxy (`caoa/eval/pose_eval.py`).

## Reproduction targets

Legacy numbers the new code must reproduce under the legacy evaluation. Files
are in the server copy (`mmlab-server-data/.../vr_reconstruction`).

**Pose with GT instances (E1–E4, E6)**
- `checkpoints/pose_mnk/*/m_train_log.json` (`s2c_total`, val every 1k steps):
  - no completion: 71.8% best (13k), 71.2% last logged (14k);
  - inst+completion: 81.3%;
  - completion only: 87.8% (emb 1024), 85.0% (512), 81.3% (256).
  Recheck best vs last for the latter runs in P8.
- `results/inst.csv`: micro 70.98 / macro 55.93 on 1,864 samples. `results/comp.csv`: 85.68 / 71.68. `results/compinst.csv`: 79.99 / 61.75. All per class.
  - Dated Feb 21 2025, before the `pose_mnk` runs; the producing checkpoints are unknown.
  - `mnk_comp.csv` and `sg_mnk_inst.csv` are byte copies of `comp.csv`.

**SoftGroup predictions (E8, E5)**
- `results/sg_mnk_comp.csv`: micro 66.75 / macro 26.26, 1,606 samples, grouped by CAD taxonomy name.
- `results/perclass_results.csv`: `calc_pose_accuracy_s2c` format, 1,602 samples, 7 classes.

**Completion (E9)**
- `results/ODG_{PCN,SN55,SN34,S2CC,Concat,SNAug_epoch40}_S2CC.csv`: per taxonomy F-Score / CDL1 / CDL2.
- `[TEST]` log lines behind Table 2: PCN 161.521 / 137.124; SN-55 94.103 / 78.029; SN-34 94.014 / 77.893; S2C-C 46.802 / 25.692. The "+Ctxt_100" 22.471 / 2.30 is a best-validation value.

**Symmetry (E11):** 95.8% accuracy, 94.1 F1 (paper) = the training log's final accuracy / macro-F1 (95.80 / 94.05); the checkpoint gives 94.8 / 90.6 on the 1,864 val pose pairs (P9).

**Paper Table 1:** No-completion-Sph 60.92 / 48.63; CAPCM-Sph 75.83 / 67.45; CAPCM+SEM-SG 76.04 / 66.84; CAPCM+SEM-Sph 77.51 / 69.17; CAPCM+SEM-GT 86.54 / 78.15 (avg / class avg). Which legacy runs produced these rows is unknown; to establish in P3b / P8 / P11.

---

## P0 — Audit and decisions (Done)

- Paper summarised; legacy code mapped to the paper's modules; mismatches catalogued ([issue register](#legacy-and-benchmark-issue-register)).
- Data located on the external drive and `/home/data`. Server copy found, with checkpoints, logs, result CSVs and `.git`.
- User decisions:
  - pilot on the local scenes, with no bulk extraction;
  - legacy first, then the paper;
  - spconv for OCAM, PTv3 for SEM;
  - a separate torch environment;
  - official metrics faithful to Scan2CAD, both conventions;
  - GT instances: official protocol + matched-only;
  - all 9 official categories;
  - **all legacy CAOA evaluations preserved and reproduced first** (2026-10-20);
  - CIS2VR-era evaluations out of scope.

## P1 — Foundation and data layer (Done)

Delivered:
- uv project (Python 3.12, torch 2.9.1+cu126, spconv-cu126 2.3.8), `configs/paths.yaml`, resources.
- `caoa/{geometry,paths}.py`, `caoa/data/{labels,scannet,scan2cad,pairs}.py`, `scripts/build_pairs.py`.
- Pair cache `cache/pairs_v1/` (199 MB): 605 scenes, 4,358 matched → 3,811 kept under the V2 class rule. Pairs keep `sem_name` and `catid_cad`, so the legacy rule can be applied at load time (P3b).

Validation (passed):
- `load_scene` is identical to SoftGroup's `prepare_data_inst.py` (3 scenes, including `scene0217_00`).
- Scan, CAD and box matrices are identical to legacy (200 scenes).
- Pair sanity: a median 95.5% of points are inside the padded box.
- S2C-C poses are identical to the official `full_annotations.json`.

Follow-up (moved to P3b): the sample-set rule.

## P2 — Evaluation core (Done)

Delivered:
- `caoa/eval/metrics.py`: E1 (legacy) and the per-object Scan2CAD formulas.
- `caoa/eval/scan2cad_benchmark.py`: the official protocol, `alignment_from_box` (both conventions).
- `caoa/eval/pose_eval.py`: `legacy` / `benchmark/<conv>` / `matched/<conv>` / `proxy`.

Validation (passed):
- E1 matches the legacy code exactly.
- The benchmark matches upstream exactly.
- A perfect `minus_center` submission scores 100%; a perfect `plain` one scores 77.17%.

Follow-ups (moved to P3b): E2–E5 reports.

## P3 — OCAM object-only, legacy mode (Done)

Delivered:
- `caoa/models/{sparse,ocam}.py`: `MinkowskiFCNNP_v3` ported to spconv, with ME downsampling sites and a true max-pool.
- `caoa/data/pose_dataset.py`, `scripts/train_pose.py`, `configs/pose/legacy_nocomp.yaml`.

Validation (passed):
- ME-semantics unit tests; frame round trip; GT predictions score perfectly.
- Legacy metric 72.8% vs the server's 71.8% (best) / 71.2% (last). The curve matches from 1k on.

**Caveat (found in the review):** the pilot used the V2 sample set (3,080 / 731, including `otherfurniture`), not the legacy one. The comparison is therefore not like-for-like; it is re-run in P3b.

Optional (P3+): convert legacy ME checkpoints to spconv for an exact port check. This needs ME's kernel-offset order.

## P3b — Legacy evaluation parity (Done, 2026-10-20; see [results](results/P3b.md))

Goal: everything the legacy GT-instance pose evaluation produced is reproduced, on the same sample set and in the same formats, before any further modules.

**P3b.1 Sample-set rule (E6).**
- New switch `class_rule`:
  - `v2`: the `S2CCompletionPoseV2` code;
  - `relpose_cache`: what built the reported datasets. `unannotated` *or* `otherfurniture` instances take the CAD's eval class (among the 8 `VALID_CLASSES`) or are dropped. **Default for legacy reproduction.**
  - `all_matched`: candidate paper/clean mode (also listed in P4).
- Applied when pairs are loaded; no cache rebuild.
- Expected on the subset: train 2,733 / val 659.

**P3b.2 Legacy report (E2, E3).** A module reproducing `eval_pose_s2c_v2`'s outputs:
- MAE over the whole 15-value pose vector and per block (t / q / s), sklearn semantics;
- mean legacy rotation error and mean scale % error;
- the per-class table (translation, rotation, scale, total, support, plus `macro_avg` / `micro_avg` rows) written as CSV in the legacy layout;
- inference time per sample.

**P3b.3 Older metric (E5).** Port `calc_pose_accuracy_s2c`, including its `VALID_CLASSES` restriction and per-term rounding; output in the `perclass_results.csv` layout.

**P3b.4 Training-time validation (E4).**
- Reproduce the legacy loop: counter semantics, the epoch overrun past `max_steps`, validation every 1k; log the legacy quantities under the legacy JSON keys.
- Any extra evaluation of the final model goes in a separately labelled file.

**P3b.5 Per-class groupings (E7).**
- One grouping argument for every report:
  - `scannet_class`: legacy GT-instance CSVs (E2);
  - `cad_taxonomy_name`: legacy SoftGroup CSV (E8). Name = `shapenet2scannet_name(SHAPENET_SYNSET_NAME[catid])`, else `sorted(names)[0]`;
  - `table1`: **decided by the user on 2026-10-20 (candidate C).** `cad_taxonomy_name` folded to Table 1's columns: bathtub → "bath", bookshelf, cabinet, chair, sofa, table; every other CAD category (bed, display, trash bin, file cabinet, …) → "other";
  - `valid_classes`: E5.
- Table 1's "avg" and "class avg" are computed as in E2/E3 (micro / macro) over the `table1` grouping. Mismatches with the paper's values are recorded, not chased.
- Rejected candidates, for the record: A (Scan2CAD top-8 with display / trash bin dropped) and B (`VALID_SYNSETS`, which has a bed column).
- The check script `table1_mapping_check.py` (scratchpad) is re-run on full val only to record the supports and the paper mismatches.

**P3b.6 Re-run the pilot** with `relpose_cache` (~1 h). Train on the local train subset (2,733) and evaluate on the full legacy val set (1,864, once P4d-val is in). Compare against:
- the server no-completion log (1k…14k);
- `inst.csv`'s per-class table (same evaluation, different checkpoint: compare the pattern, not exact numbers).

Validation gate:
- Parity tests vs verbatim legacy code: `eval_pose_s2c_v2`'s summary path (MAE, per-class CSV) on synthetic predictions; `per_class_accuracy`; `calc_pose_accuracy_s2c`.
- Sample-set check: with all 312 val scenes, `relpose_cache` gives exactly 1,864 samples with `inst.csv`'s per-class supports. On the old subset, 659 val already fits inside every support.
- Pilot: legacy metric within ±2 points of the server log at matching steps; CSV in the legacy layout.
- Official protocol and matched-only re-reported on the new sample set.

## P4 — Data completion

**P4a. CAD point cache.**
- Area-uniform samples of `model_normalized.obj` (legacy: open3d `sample_points_uniformly(5000)`), float32, plus bbox/center.
- ~0.3 GB at 8,192 pts/CAD, ~0.6 GB at 16,384. Local-pair CADs first (~1.5k), then all 3,049.

**P4b. Scene context, two legacy variants (switch).**
- `v2`: scene minus the instance, inside the instance AABB with extent + r (r total), not subsampled.
- `relpose_cache`: crop of the whole scene including the instance, subsampled to `num_points // 2` when above `context_pointnum`.
- Compute on demand; store only after a size estimate (raw: est. 1–3 GB).

**P4c. `all_matched` class rule.** Keep every matched pair (recovers displays and trash bins; local official ceiling 67.2% → 77.0%). Candidate fix, reported only after the legacy reproduction.

**P4d. Full ScanNet extraction (gated).**
- **P4d-val** (**done** 2026-10-20): the 206 missing val scenes, ~3 GB, ~1 h. All 312 val scenes are now local.
- **P4d-train** (**done** 2026-10-21): 694 of the 695 missing annotated train scenes, 9.2 GiB. **`scene0212_00` is unavailable**: its archive is an empty 32-byte 7z on every copy (8 matched Scan2CAD models). Re-downloading it from ScanNet needs the user's access.
  - With it missing, the legacy rule gives **6,663 train pairs = legacy 6,671 − its 8**, and 11,144 of 11,152 matched pairs. Together with the exact 1,864 test set, the legacy pose dataset is reproduced except for that scene.
- **Local ShapeNet mirror** (user, 2026-10-21: no reading from the external drive). `/home/data/datasets/ShapeNetV2_caoa/`, legacy layout (`ShapeNetV2/`, `ShapeNetV2-augmented/`), only the files the pipeline uses:
  - 22,213 `model_normalized.obj` (21,499 SN-Indoor + 3,049 S2C CADs, 2,335 shared), 16.9 GiB;
  - 171,992 SN-Indoor partials, 15.4 GiB;
  - `taxonomy.json`, `splits.json`, `aug_train_test_splits.pkl`.
  - Copy verified (194,208 files); `configs/paths.yaml` `shapenet:` points to it.

Findings (2026-10-20) that fix P4's details, all by call path:
- **Two CAD samplers.**
  - Pose cache (`S2CCompletionPoseV2`): `trimesh.load(force="mesh")` → open3d `sample_points_uniformly(5000)`.
  - Completion GT (`S2CCompletion` loader): `trimesh.load(force="mesh", skip_materials=True)` → Chris Choy barycentric `resample_mesh(density=2000)`, so the point count scales with the CAD's surface area.
  - Legacy sampled once per cached instance; we mirror that per pair (completion GT est. ~0.3 GB; pose 5,000 pts ≈ 0.35 GB for 5,957 pairs).
- **Three context rules, one per module.**
  - Pose V2: scene minus instance.
  - Pose RelPose: whole scene including the instance, subsampled.
  - Completion loader: **layout points only** (instance label < 0).
  - All three use the instance AABB with extent + r.

Validation gate:
- CAD↔scan sanity (median scan→CAD distance; 10 visual spot checks per class).
- CAD bbox/center equal to the annotation.
- Context parity for both variants vs verbatim-logic references on 50 pairs.
- After P4d-val: the val sample set under `relpose_cache` is exactly 1,864, with the per-class supports of `inst.csv`.
- Disk usage logged against the estimates.

Decisions needed: CAD points per model; context storage; P4d-val now?

## P5 — Completion datasets

- **P5a. Loaders.** The current files are complete; the earlier "truncated" reading was a tooling error (see item 2 below). An older compiled snapshot (Python 3.8 `.pyc`, Feb 28 / Mar 1 2025) exists and was decompiled to scratchpad `decompiled/`, but is reference only.
  - **User decision (2026-10-20): build from the CURRENT files** (`ODGNet/datasets/S2CCompletionDataset.py`, `SNAugDataset.py`), not the older snapshot. Anything that looks broken is listed and **reviewed with the user before any fix**. The snapshot serves only as reference evidence. Fixes go into the port; the legacy repo stays untouched unless the user says otherwise.
  - Not to be confused with `datasets/custom_datasets_completion.py`, the older MinkowskiEngine completion pipeline (`train_completion.py`, "mnk" models), which CAPCM does not use.
  - **Review of the current files (2026-10-20); fixes apply to the port only:**
    1. `S2CCompletionDataset.py:466` unpacks 10 values from a 9-key dict, so it crashes on every load. **Approved fix:** unpack the 9 fields.
    2. ~~`__getitem__` has no `return`~~. **Withdrawn.** Line 666 returns `(catid, id_cad, coords, coords_target, sem_embedding, idx)`, the 6-tuple `runner.py:137` expects. The file's last line has no trailing newline (and CRLF endings), so `wc -l` and line-range reads missed it. The same error was behind the earlier "loaders truncated" claim: **both loaders are complete.**
    3. `sem_embedding` has no source once the 10th field is gone. **Approved fix:** look it up in the same `data/ShapeNetSemantic/classification_ptv3/semantic_embeddings.npy` that SN-Indoor uses (key `catid_idcad`); only `UpTransv2` reads it.
    4. The context branch in `__getitem__` is unfinished (`np.ones(n, 1)` TypeError; raw unsampled points replace the normalised input; context not rotated). **Decided: mirror the pose-time inference** (`custom_datasets_pose.py:4661-4697`). Concatenate instance + context with the flag (encoding decided empirically in P6), apply the same yaw to instance, context and target, normalise by the combined centroid and max radius, then sample `num_points`.
    5. The semantic-model block (lines 158–187, 436–438) is inactive in every config and not ported.
    6. SN-Indoor's `random.choices` (sampling with replacement, even with downscale factor 1). **Decided: keep as legacy**; a plain-shuffle switch can come later as a fix.
- **Known from the recovered Feb 28 loader:**
  - **Train inputs are in the GT CAD frame:** `inv(calc_Mcad)`, then a random yaw. **Test inputs are world-oriented**, re-centred and scaled, then `pc_norm`.
  - **Classes:** `('unannotated', 'otherfurniture')` relabel, then **only the 8 VALID_CLASSES are kept** (unlike the pose dataset).
  - **Sizes:** N_POINTS 4,096 for input and GT.
  - **Concat run:** `concat_snaug_s2cc_upTransb_comp` validated and tested on **training data**. `others.subset` only changes the DataLoader; the inner datasets' `phase: train` decides the samples.
- **P5b.** S2C-Completion, completion view. Input: instance (± context with the ±1 flag). GT: CAD points placed by the GT pose. Legacy normalisation (centroid + max radius) and point counts. Record the input frame legacy used (GT-canonical or world), which matters for P8.
- **P5c.** SN-Indoor: the existing partial scans on the external drive (lazy reads, or a compact cache after a size estimate). Port the generator for reproducibility. Legacy difficulty = crop-plane shift (switch: `legacy` / `paper` %-removal).
- **P5d.** Splits: legacy validation = test. A clean held-out validation split exists only as a switch.

Validation gate:
- Samples equal those of the recovered legacy loaders (fixed seed / indices).
- Per-class counts.
- SN-Indoor generator statistics vs the existing files.

Decision needed: SN-Indoor storage.

## P6 — CAPCM model + legacy completion evaluation (Done, 2026-10-21; see [results](results/P6.md))

- **Verified (2026-10-20):** current `models/UpTrans.py` compiles to bytecode identical to the legacy Python 3.8 `.pyc`. `utils/metrics.py`, `tools/runner.py`, `tools/builder.py`, `datasets/data_transforms.py`, `extensions/chamfer_dist`, `utils/misc.py` match their 3.8 `.pyc` source size/date. Single, verified sources.
- **Legacy training recipe from the run configs:** AdamW lr 5e-4, **wd 5e-4**; LambdaLR decay_step 20, ×0.8, lowest_decay 0.01; batch 32; **400 epochs** (the paper says 200); `consider_metric: CDL1`; UpTrans num_pred 8192, num_seeds 1024, seed_fea 128, dim_feat 512, num_dicts 512, upscales [1, 2, 4].
- **P6a.** Port `ODGNet/models/UpTrans.py` (with the context-flag channel). Pure-torch FPS (`pointnet2_ops` semantics, start index 0), kNN, Chamfer.
- **P6b.** Losses: multi-level CD-L1 + dictionary orthogonality.
- **Context flag: decided empirically on 2026-10-21 as +1/−1.** +1/−1 gives `validate()` CDL1 22.275 vs the stored 22.471; 1/0 gives 30.781.
- **P6c. Legacy evaluation (E9), reproduced first.**
  - `Metrics` (F-Score at threshold 0.01, CD-L1, CD-L2 ×1e3) ported with a parity test against a verbatim copy.
  - `validate()` (class-averaged) and `test()` (mixed per-sample / per-class averaging) both reproduced.
  - Per-taxonomy CSV in the `ODG_*_S2CC.csv` layout.
  - A clean per-sample + class-mean protocol is added later as a labelled extra.
- **P6d.** Strict loading of the legacy checkpoints (local `ODGNet/experiments/...`; PCN / SN-34 / SN-55 from the server copy). Re-score them on S2C-C test.

Validation gate:
- Ops and metrics vs brute-force and verbatim references.
- Every checkpoint loads strictly.
- The re-scored per-taxonomy tables match `ODG_*_S2CC.csv`, and the `[TEST]` aggregates match the logs. Needs the full test set (P4d-val); on the subset, compare per-taxonomy patterns.
- **Outcome (2026-10-21):**
  - Passed: S2C-C 46.835 vs 46.802, PCN 161.537 vs 161.521, SN-34 93.981 vs 94.014; per-category counts exact.
  - Not reproducible: SN-55 (checkpoint missing); concat and SN-Indoor (`ckpt-last` overwritten after the tests: epochs 227 / 88 vs tested 171 / 42).

### Implementation status (2026-10-21)

Code written and tested (not yet run at scale):
- **P4a/P4b:**
  - `caoa/data/shapenet.py`: both CAD samplers; `resample_mesh` is bit-identical to legacy given the same RNG.
  - `caoa/data/context.py`: all three context rules.
  - `caoa/data/legacy_sampling.py`: the three legacy samplers.
- **P5:**
  - `caoa/data/completion_dataset.py` (S2C-C, approved fixes 1/3/4) and `scripts/build_completion_cache.py`.
    - Size estimate: **~2.1 GiB as float64 (legacy), ~1.1 GiB as float32**, for 4,308 pairs (2,564 train / 1,744 val = 1,864 minus the non-VALID classes).
  - `caoa/data/snindoor_dataset.py`: the crop matches legacy exactly under the same seed.
  - Open: 91 of the 4,308 S2C-C pairs use CADs that are missing from `semantic_embeddings.npy` (fix 3), asked 2026-10-21.
- **P6:**
  - `caoa/models/{pointops,chamfer,odgnet}.py`: the UpTrans port reproduces the legacy model's outputs exactly on the same weights (with and without context); every local UpTrans checkpoint loads strictly.
  - `caoa/eval/completion_metrics.py`: F-Score / CDL1 / CDL2 and the `validate()` / `test()` aggregations match legacy exactly.
- **P7:** `scripts/train_completion.py` (legacy runner loop, LR + BN-momentum schedulers, val every 10 epochs, ckpt-best by CDL1); configs `configs/completion/legacy_s2cc{,_ctxt1}.yaml`.

## P7 — CAPCM training (Table 2)

- Recipe (paper): AdamW 5e-4; LambdaLR step 20, γ 0.8, min 5e-6; CD-L1 + orthogonality; 200 epochs; batch 32. ~1.5 days per run on full data.
- **Scope (user, 2026-10-21): retrain S2C-C only**, to check that our training loop reproduces legacy (`s2ccompletion_upTransb1`: best epoch 180, test CDL1 46.80). The other rows are covered by the re-scored legacy checkpoints (P6). Runs after P8. **Extended (user, 2026-10-27/28):** concat (SN-Indoor + S2C-C, `configs/completion/legacy_concat.yaml`, validated on the S2C-C val split instead of legacy's training data; started 2026-10-27 22:13, ~6.3 min/epoch on a shared node) and S2C-C + context 1 m (`legacy_s2cc_ctxt1.yaml`, queued after P10a).
- Measured cost (after the Chamfer / k-NN / cudnn.benchmark speed-ups): 0.89 s per batch-32 step, about 19 GPU-hours for 401 epochs. For reference: concat about 38 h, SN-Indoor about 370 h (~15 days); legacy logged 0.83 s per step.
- **Runs on the user's DGX** (2026-10-21). Runbook and targets are in `CLAUDE.md`; its data (r = 0 completion cache, 2.0 GB, and `semantic_embeddings.npy`, 46 MB; a `paths.yaml` entry) is on the DGX since 2026-10-24 (`docs/HANDOFF_DGX.md`). The legacy validation curve (39 points, epochs 10–390) is in `results/legacy_curves/s2ccompletion_upTransb1_val.csv`.
- Batch 32 needs about 40 GB of GPU memory (legacy logged 39.99 GB). `train_completion.py` has no resume; the DGX's 7-day job limit makes it unnecessary.
- (Originally planned runs: S2C-C; SN-Indoor; SN-Indoor + S2C-C; + context 100 cm.)
- Legacy selection (best on test) and legacy averaging first.
- The paper's "+Ctxt_100" row is the best *validation* score of an **S2C-C-only** run with context. Reproduce what was run first; the paper's description second.

Validation gate:
- Curves vs the legacy ODGNet logs.
- Table 2 rows within the agreed tolerance under the matching legacy protocol.
- GPU budget agreed beforehand.

**Outcome (2026-10-27, [results](results/P7.md)): passed, accepted by the user.** DGX A100, 0.60 s/step, ~15.7 h with validation. Best validation CDL1 42.93 @ epoch 70 (legacy 43.60 @ 180); test on ckpt-best F 0.0594 / CDL1 45.00 / CDL2 23.06 (legacy 0.0602 / 46.80 / 25.69). The validation curve tracks legacy within ~1–2 CDL1 at every epoch.

## P8 — OCAM + completion

**Legacy reproduction first.**
- **Completion model (user, 2026-10-21): the paper's final numbers come from the CONCAT checkpoint** (`UpTrans_comp/Concat_SN_S2CC/concat_snaug_s2cc_upTransb_comp`, ckpt-best = epoch 219), as `build_completion_pose_datasets.py:132` loads it.
- **The saved `pose_mnk` runs used S2C-C completions**, not concat: their TensorBoard events are from 2025-02-27 20:47 to 02-28 04:00 UTC, before the first concat training log (2025-03-01). Their cache name `odg_s2ccompletion_upTransb1` = `odg_` + the S2C-C run name, which is how the builder names caches.
- **User decision (2026-10-21): run both.** (1) S2C-C completions (`s2ccompletion_upTransb1/ckpt-best`, epoch 180), checked against the `pose_mnk` logs; (2) concat completions, the paper's final setup, compared with Table 1 (no saved run uses them).
- Legacy's completed-pose caches (`data/objects/ScanNet_v2/dataset/instances/scannet_inst_pose_completion_*.npy`) are not on the drive or local, so completions can't be compared file to file; parity rests on the verbatim-code test below and on the alignment results.
- Code: `caoa/data/pose_completion.py` (inference, verbatim-tested against `infer_completion_odg`), `scripts/build_pose_completions.py` (cache `cache/pose_completions_v1/<name>/`), `ObjectPoseDataset(completions=, completion_mode="only" | "inst+comp")` (legacy `completion_points_only`; frames tested), `data.completion` in the pose configs.
- Configs `configs/pose/legacy_comp_*.yaml`:

  | Config | Completion | Points | Emb | Steps | Legacy target (best / last) |
  |---|---|---|---|---|---|
  | `legacy_comp_s2cc_inst_emb1024` | S2C-C | inst+comp | 1024 | 15k | 81.3 @13k / 80.7 |
  | `legacy_comp_s2cc_only_emb1024` | S2C-C | only | 1024 | 25k | 87.8 @23k / 87.3 |
  | `legacy_comp_s2cc_only_emb512` | S2C-C | only | 512 | 25k | 85.0 @22k / 84.8 |
  | `legacy_comp_s2cc_only_emb256` | S2C-C | only | 256 | 25k | 81.3 @22k / 80.9 |
  | `legacy_comp_concat_only_emb1024` | concat | only | 1024 | 25k | Table 1 |
  | `legacy_comp_concat_inst_emb1024` | concat | inst+comp | 1024 | 15k | Table 1 |

  All other settings equal `legacy_nocomp` (server configs compared key by key; `add_embedding_as_feat` is False or the default False in every run).
- First started 2026-10-21 15:33 on the workstation in two lanes, then **stopped at ~5% (user) to move to the DGX**. They are GPU-bound: the completed points fill many more sparse voxels than the partial scans, so each run trains at 0.6–0.7 it/s with two sharing the GPU, against ~4 it/s for `legacy_nocomp`. The total is about 28 GPU-hours. The legacy componly run took about 7 h for 25k steps (from its event files).
- Completion inputs exactly as legacy: GT-instance path in the **GT box frame** (`custom_datasets_pose.py:5000`, `:5024`, `:5063`).
- Completion inference timing (E10).
- Inst+completion and completion-only inputs; class rule `relpose_cache`.
- Targets: `pose_mnk` logs (81.3% / 87.8% / 85.0% / 81.3%), `comp.csv`, `compinst.csv` (pattern).

**Outcome of the legacy reproduction (2026-10-27, [results](results/P8.md)): passed.** All six runs on the DGX (one A100, ~2.5 it/s, 11 h in total). Best legacy metric: S2C-C only emb 1024 / 512 / 256 = 87.8 / 85.6 / 82.7 (legacy 87.8 / 85.0 / 81.3), S2C-C inst+comp 81.9 (81.3); concat only 87.4, concat inst+comp 82.5. Final models: official protocol 35–39% (plain) / 46–51% (−center).

**Then the fix (switch `completion_frame`):** `world`-frame completion, which uses no GT. Quantify the leak: the same OCAM evaluated with both completion frames.

**Outcome of the leak analysis and fix (2026-10-27/28, [results](results/P8.md)).** Only the GT rotation reaches the
CAPCM (centre and scale cancel in its normalisation), and the completed geometry barely depends on it, but completion-only
OCAM reads the target rotation from the completion's frame-tied point layout: with the completion frame turned 180°,
95% of its predictions turn 180° (accuracy 86.5 → 12.8), while inst+comp is unaffected. Retrained on `world`-frame
completions (user-approved; caches `<name>__world`, 0.78 GiB each): completion-only 86.75 → 82.62 (S2C-C) / 83.05 (concat),
inst+comp 81.49 → 80.85 / 83.10 → 81.81; official −center −2.2 to −2.7 for completion-only. Completion-only stays the best
leak-free setup. emb 512 / 256 leak-free running.

Validation gate:
- Legacy targets within tolerance (full data), or the same ordering and gains on the subset.
- Leak quantified.
- All reports: legacy (E2/E3), official, matched-only.

Decision needed: how leaky legacy numbers are presented.

## P9 — SEM + legacy symmetry evaluation

- Vendor PTv3 (single file + needed Pointcept bits). `torch_scatter` wheel if available for torch 2.9, else pure-torch segment ops. flash-attn off.
- Data: CAD points (P4a); binary symmetric/not from Scan2CAD `sym`; splits from `s2c_cad_splits.pkl` (2,440 / 609).
- Recipe (paper): emb 128, softmax head, BCE, AdamW 1e-4, wd 1e-3, cosine to 1e-6, 150 epochs.
- **Legacy evaluation (E11):** `eval_sym` (argmax accuracy on `num2label` strings), timing.
- F1 has no code path. Compute it from `eval_sym`'s returned gt/pred, with the definition (binary on "symmetric" vs macro) confirmed against 94.1.

Validation gate:
- The legacy `symmetry_ptv3` checkpoint loads strictly and reproduces `data/s2c_symmetry/.../embeddings.npy` (cosine ≈ 1).
- Retrained: 95.8% ± 1 accuracy; F1 definition settled; runtime vs 0.58 s/scene.

**Status (2026-10-28, [results](results/P9.md)).** PTv3 vendored (`caoa/models/ptv3`, imports only; `torch.segment_reduce`
for `torch_scatter`, flash-attn off) with the SEM head (`caoa/models/sem.py`). The checkpoint loads strictly and reproduces
`embeddings.npy` (test cosine median 0.9997, same argmax for all 3,049 CADs). **But that file predicts every CAD as not
symmetric:** the checkpoint was trained on `S2CCompletionSymmetry` (CAD in the GT box frame, `grid = coord // 1 cm`), and the
file was exported two days later through `S2CSymmetry` (raw ShapeNet coordinates, min-subtracted grid). Fed as trained, the
checkpoint gives 94.8% / macro-F1 90.6 on the val pose pairs (log 95.8 / 94.05; F1 = macro-F1). **Fix (user, 2026-10-28):**
`scripts/build_sem_embeddings.py`, each CAD in its own bounding-box frame (no annotation): 94.1% per pose pair, 87% per
unique CAD; switch `sym_embedding.frame: legacy_export | cad_bbox`. Open: retraining (recipe: legacy `train_sym_ptv3.py` on
which pipeline, to decide), E11 on legacy's exact sample set, timing.

## P10 — OCAM paper model

- **P10a.** Symmetry embedding concatenated (legacy `mnkpv3_emb`; no legacy run used it). Code done 2026-10-28 (`OCAMObjectOnly(ext_emb_size=128)`, `sym_embedding` config). Runs (user: `cad_bbox` embeddings only): concat completion-only emb 1024, legacy and world completion frames (`semcad_*`).
- **P10b.** Siamese CAD+object model with truly shared weights. The legacy `MinkowskiFCNNRelPose_v1` is not shared, and its CAD branch calls an object-branch layer: reproduce it as `legacy`, fix it as `paper`.
- **P10c.** Loss.
  - Legacy: object vs itself under GT/pred pose; t2/q1/s2 or t3/q1/s3.
  - Paper: 5 × Chamfer(CAD at the predicted pose, object) + L1, t2/r3/s2.
- **P10d.** Optimiser.
  - Legacy: wd 1e-3, 15k steps, min lr/100, no shuffle.
  - Paper: wd 1e-4, 150 epochs, min 1e-6.

Validation gate:
- Unit tests: Chamfer 0 at the GT pose; weights shared; a 32-sample overfit reaches ~100%.
- Ablation table on a fixed split under the legacy report (E2) and the official protocol.

## P11 — Predicted instances

**P11a. Legacy SoftGroup evaluation (E8), reproduced first.**
- SoftGroup predictions (local, 96 GB, `data/objects/softgroup_preds/sg_raw`) loaded as legacy.
- Confidence threshold: class default 0.5, `sgpreds_s2c_completion_pose.yaml` 0.3. Record which one the target CSVs used.
- Greedy instance↔CAD overlap matching with legacy conflict resolution.
- Legacy normalisation: ×1/(max bound + 0.5) if the bound > 0.8; mean-centred.
- `filter_classes`; classes = CAD taxonomy names.
- "Instance capture accuracy" (matched / total aligned CADs per catid).
- Reports: E2 layout (`sg_mnk_comp.csv`) and E5 layout (`perclass_results.csv`).
- Targets: 66.75 / 26.26 on 1,606; 1,602-sample E5 table; the paper SG row (76.04) once its run is identified.

**P11b. SphericalMask (gated).** Official code/checkpoint on ScanNet val; separate environment; needs P4d-val.

**P11c. Official pipeline.** Predicted instances → completion (`world` frame) → CAD choice → OCAM → submission (both conventions) → official protocol, reported next to the legacy protocol.

**P11d. CAD choice (switch).**
- legacy: GT CAD via overlap matching;
- pool: retrieval from the scene's CAD pool;
- paper: retrieval from ShapeNet (ambiguous).

Validation gate:
- Legacy SG reproduction within tolerance.
- The GT-instance pipeline equals P8/P10.
- Capture rates match the legacy printout.

Decisions needed: retrieval protocol; SphericalMask installation.

## P12 — Full-scale experiments and report

- Requires P4d (all 1,513 scenes). Rebuild the caches; 3 seeds per configuration.
- `scripts/report.py` produces:
  - every legacy layout (E2/E3/E5/E8/E9/E11);
  - Table 1 in the paper's grouping (validated in P3b.5);
  - the official protocol with all 9 categories in both conventions;
  - Table 2;
  - baselines from the original papers.
- `docs/RESULTS.md`: every paper number next to ours; each deviation traced to the issue register.
- README and reproduction steps.

Validation gate: every paper row is reproduced within tolerance, or has a user-accepted explanation.

---

## Legacy duplicate choices

The legacy code often has several near-duplicate implementations. Rule (user, 2026-10-20): when the
call path doesn't settle which one CAOA used, ask the user before choosing. Choices made so far,
with the evidence. **Status: D1–D13 confirmed by the user on 2026-10-20.** New choices are added here as they come up.

| # | What | Candidates | Used | Evidence |
|---|---|---|---|---|
| D1 | ScanNet instance preprocessing | `prepare_data_inst.py` in the ScanNet dataset dir (= external-drive copy) vs SoftGroup repo copy | dataset-dir copy | Writes `*_inst_nostuff_v2.pth`, the file legacy reads; the SoftGroup copy writes `*_inst_nostuff.pth`. Instance numbering is identical in both. |
| D2 | S2C annotation file | `scan2cad_v2_completion_annotations.json` (11,152 matched) vs `scan2cad_v2_annotations.json` (13,628 matched; 2,592 `scannet_instance_id`s differ) | completion file | Server `config.json` of every `pose_mnk` run; cache name `…_v2c_…`; the 1,864 set reproduces exactly. Other modules (ODGNet loaders, SEM, SG) are checked per phase. |
| D3 | Sample-set builder | `S2CCompletionPoseV2` vs `S2CCompletionRelPose` (shared cache file) | RelPose rule | Agreed in P3b; exact 1,864 reproduction |
| D4 | Box matrix `calc_Mbbox` | dataset class method vs `general_utils` | either | Identical code |
| D5 | `decompose_mat4`, `TransMatFromTrs` | `dataset_utils` vs two `general_utils` copies | `dataset_utils` | `from utils.dataset_utils import *` is imported last in `custom_datasets_pose.py`. The `general_utils` `TransMatFromTrs` copies crash on quaternion input (`Quat2RMat` returns 4×4). |
| D6 | Dominant instance label | dataset class method vs `dataset_utils` vs `general_utils.GetDominantLabel` | class method | Called as `self.get_dominant_label`; all three are identical |
| D7 | Point (re)sampling | `SamplePointsFrom2Array` (cache build), `S2CCompletionPoseV2.sample_points` (`__getitem__`) | as called | The only functions on the call path |
| D8 | ShapeNet taxonomy loaders | class `load_taxonomy` (name lists) vs `general_utils.LoadShapenetTaxonomy` (name sets → `SHAPENET_SYNSET_NAME`) vs `LoadShapeNetTaxonomy` | per call path: class loader for CAD → eval class; `SHAPENET_SYNSET_NAME` for SG / Table 1 names | Where each is called |
| D9 | Pose evaluation driver | `eval_pose_s2c_v2` vs `eval_pose_s2c` (×3: `training_utils.py:1759`, `:1852`, `eval_utils.py:868`), `eval_cls_pose` | `eval_pose_s2c_v2` | Called by `train_pose.py` (training and eval mode). The older drivers are not ported (confirmed: no CAOA number came from them). |
| D10 | Pose metric | `calc_pose_accuracy_s2c_nocls` vs `calc_pose_accuracy_s2c` | both | The first is on the call path; the second produced `perclass_results.csv` |
| D11 | Rotation difference | `eval_utils.calc_rotation_diff` (scipy) vs `tools/eval` version (quaternion dot) | `eval_utils` | Called by the CAOA metric; the `tools/eval` one is CIS2VR-era |
| D12 | Pose network | `MinkowskiFCNNP_v3` vs `_v3_emb`, `emb_attn_v1`, `Pose_v3`, `RelPose_v1`, … | `MinkowskiFCNNP_v3` | `type: mnkpv3` in the server config → `train_pose.py` mapping. Others come in later phases. |
| D13 | Training loop | `train_pose9dof` vs `train_pose9dof_cls` | `train_pose9dof` | Called by `train_pose.py` |

## Paper vs codebase mismatches

Recorded, not chased (ground rule 1). Reproduction follows the code.

**Table 1**
- No code builds the table. The per-class grouping is the user's choice C (P3b.5).
- No display or trash-bin columns. For every CAOA row, the class avg ≠ the mean of the shown columns (GT row 78.15 vs 83.47; SG row 66.84 vs 71.30), so hidden categories or a different averaging were involved.
- GT row avg 86.54 is not k/1,864 for any integer k. Under grouping C, the legacy test set's column supports are bath 72, bookshelf 113, cabinet 159, chair 965, other 117, sofa 94, table 344. **None** of the GT-row values is attainable with them; with the A/B groupings the six shared columns have the same supports, so no CAD-category grouping of the 1,864 set produces the row. It was computed on another sample set (possibly the SoftGroup-style pipeline fed GT instances) or edited afterwards.
- Baseline rows: the values after "chair" are shifted one column left relative to the original papers (verified for Scan2CAD and FPFH against Scan2CAD's Table 2).
- SG row: not attainable with `sg_mnk_comp.csv`'s per-class supports (only chair is), so it comes from a different SG run.
- GT row (86.54) exceeds the official Scan2CAD ceiling for GT instances (76.5%) and uses GT-pose-leaking completion inputs (P8).
- The paper says Scan2CAD-benchmark evaluation; the code uses the legacy normalised, matched-only metric (E1).

**Table 2**
- "+Ctxt_100" (22.471 / 2.30) is the best validation score of an S2C-C-only run with context, not SN-Indoor + S2C-C + context. It is also a `validate()` (class-averaged) number; with the `test()` aggregation the other rows use, the same checkpoint scores ~20.6.
- The SN-Indoor (51.193) and SN-Indoor + S2C-C (34.986) values are not in any log found.
- Rows mix ODGNet `validate()` and `test()` averaging; the validation set equals the test set.

**Datasets and method**
- S2C-C counts 8,535 = 6,671 / 1,864: the 1,864 test set is reproduced exactly by the RelPose-cache class rule (all per-class supports equal `inst.csv`). Train 6,671 not yet checked (needs P4d-train).
- SN-Indoor difficulty: paper = 25/50/75% of points removed; code = shift of the crop plane.
- SN-Indoor categories: the paper describes 35; the training split (`aug_train_test_splits.pkl`) has **7** (table, chair, sofa, cabinet, bathtub, bookshelf, bed); 120,432 train / 51,560 val pairs.
- Context "enlarged by 100 cm": the code adds 1 m to the total extent (0.5 m per side).
- OCAM.
  - Paper: Siamese CAD+object with shared weights, SEM input, Chamfer CAD-vs-object ×5, weights t2/r3/s2, AdamW wd 1e-4, 150 epochs.
  - Code: object-only `mnkpv3` (no checkpoint uses SEM), Chamfer off (or object-vs-itself), t2q1s2 / t3q1s3, wd 1e-3, 15k steps.
- CAD choice: the paper retrieves with PointNet descriptors; the code uses the GT CAD.
- SEM F1 (94.1): no code path computes it; it is the training log's macro-F1 (94.05, P9).

## Switch register

| Area | Legacy (reproduce first) | Fix / paper / official | Phase |
|---|---|---|---|
| Sample-set class rule | `relpose_cache` (reported numbers); `v2` (the V2 code) | `all_matched` | P3b, P4c |
| Per-class grouping | `scannet_class` (GT CSVs), `cad_taxonomy_name` (SG CSV), `valid_classes` (E5), `cad_top8` (Table 1, hypothesis) | official 9 categories | P3b |
| Pose eval frame and metric | normalised frame, E1 | official protocol (metres, both conventions), matched-only | P2 ✅ |
| Training loop / validation schedule | legacy counter semantics, epoch overrun, validation every 1k | + labelled final evaluation of `model_final` | P3b |
| Train data order | `shuffle=False` | shuffled | P3 ✅ (switch exists) |
| Scene context | `v2` (scene minus instance) / `relpose_cache` (incl. instance, subsampled) | paper "enlarged by 100 cm" (ambiguous) | P4b |
| Completion input frame | GT canonical (leaks GT pose) | world-oriented | P8 ✅ (`completion_frame`) |
| SEM embeddings for OCAM | legacy export (degenerate: all "not symmetric") | `cad_bbox` (CAD's own bbox, as trained) | P9 ✅ (`sym_embedding.frame`) |
| Concat CAPCM validation data | training data (inner datasets' `phase: train`) | S2C-C val split (user, 2026-10-27) | P7 (config) |
| Completion eval averaging | `validate()` class-avg / `test()` mixed | per-sample + class mean, separate | P6 |
| CAPCM model selection | best on test | held-out validation | P7 |
| SN-Indoor difficulty | crop-plane shift | 25/50/75% removed | P5 |
| OCAM model | object-only `mnkpv3`; RelPose (unshared) | Siamese shared + SEM | P10 |
| Chamfer loss / weights | object-self; t2q1s2 | CAD-object ×5; t2r3s2 | P10 |
| OCAM optimiser | wd 1e-3, 15k steps, lr/100 | wd 1e-4, 150 epochs, 1e-6 | P10 |
| SG evaluation | matched predictions only | official protocol | P11 |
| CAD for alignment | GT CAD | retrieval | P11 |

## Legacy and benchmark issue register

| Issue | Effect | Handling |
|---|---|---|
| Legacy rotation metric (scrambled quaternions, Euler max, `min` over symmetric candidates) | Symmetric objects pass almost always | Reproduced (E1); official reported alongside |
| Legacy scale error mean(\|ratio − 1\|) | Different pass set | Reproduced (E1) |
| Translation threshold in normalised units | "20 cm" = 20% of object size | Reproduced; official in metres |
| V2 and RelPose datasets share one cache file | Reported sample sets were built by RelPose's rule (1,864 / 6,671) | `relpose_cache` rule (P3b) |
| GT-pose leak in GT-instance completion | Inflates 81.3 / 87.8 / 86.54; completion-only OCAM reads the rotation from the completion's point layout (~4 points) | Reproduced, quantified, fixed (`world`, P8) |
| SEM `embeddings.npy` exported through a different input pipeline than the checkpoint was trained on | Every CAD predicted not symmetric; embeddings carry no symmetry | Regenerated (`cad_bbox`, P9) |
| `filter_classes: False` → category −1 for every sample | Labels unusable; per-class uses names | Names used, as legacy |
| Legacy loop overruns `max_steps` to the epoch end (cosine LR rising again after `T_max`); the step-15,000 validation is computed but never written to `m_train_log.json` | Logged "final" numbers are the step-14,000 values | Loop reproduced (P3b); all validations logged |
| Result CSVs: producing checkpoints unknown; `mnk_comp.csv` and `sg_mnk_inst.csv` are copies of `comp.csv` | Targets are patterns, not exact | Noted per target |
| Table 1 grouping not in code; avg 86.54 ≠ k/1,864; class avg ≠ mean of the shown columns | Table 1 provenance unclear | P3b.5 hypothesis + validation |
| Table 1 baseline rows shifted (display dropped) | Misleading comparison | Report from the original papers |
| SEM F1 (94.1) has no code path | Definition unknown | Settled: training-log macro-F1 (P9) |
| ODGNet S2C loader: 10-from-9 unpack at line 466 (crash); unfinished context branch | Can't run as-is | Fixes reviewed with the user (P5a) |
| ODGNet best-on-test, mixed averaging, "+Ctxt" row from an S2C-C-only run | Table 2 provenance | Reproduce as run (P6/P7) |
| Concat / SN-Indoor `ckpt-last` overwritten after their Mar 5 tests; SN-55 checkpoint missing | Those Table 2 / log rows can't be re-scored | Retrain in P7 |
| Official benchmark −center GT reference | Perfect plain poses score 77.2% | Kept; both conventions |
| Official scale mean / 36-step INF symmetry | Lenient scale; ≤5° slack | Kept |
| Paper GT row exceeds the official ceiling (76.5%) | Not the official protocol | Legacy + official + matched-only |
| No alignment checkpoint uses SEM; no SphericalMask outputs | "CAPCM+SEM" / Sph rows unverifiable as-is | Retrain (P10); run SphericalMask (P11b) |

## Disk budget

| Item | Size | Status |
|---|---|---|
| uv environment (torch, spconv) | 6.4 GB | done |
| Pair cache v2 (v1 deleted) | 552 MB | done (1,505 scenes) |
| CAD point cache | 0.3–0.6 GB | P4a, to approve |
| Context (if stored) | est. 1–3 GB | P4b, to approve |
| Completed-points caches | ~0.8 GB per model (8,527 pairs × 8,192 float32 points) | P8, done |
| World-frame completed-points caches (`__world`, S2C-C + concat) | 2 × 0.78 GiB, on the DGX lab share | P8 fix, done (user OK 2026-10-27) |
| SEM `cad_bbox` embeddings | 1.9 MB | P9 fix, done |
| DGX transfer bundle (`/home/data/caoa_dgx_bundle`) | symlinks + 0.5 GB copied from the drive; 7.8 GB (`now/`) + 29.6 GB (`later/`) | done; sent 2026-10-24 without ScanNet/ShapeNet (already on the DGX, MD5-identical): 15.2 GB on the DGX's lab storage (`docs/HANDOFF_DGX.md`) |
| SN-Indoor local cache (optional) | unknown, measure first | P5c, to approve |
| ScanNet val scenes (P4d-val) | ~3 GB | done |
| ScanNet train scenes (P4d-train) | 9.2 GiB | done (scene0212_00 unavailable) |
| Local ShapeNet mirror (CAD .obj + SN-Indoor partials) | 32.3 GiB | done (194,208 files) |
| SphericalMask env + outputs | est. 6–10 GB | P11b, gated |
| Checkpoints | ~0.1–0.3 GB per OCAM run | ongoing |

Free space at the last check: `/home` 212 GB, `/home/data` 147 GB.

## Working conventions

- **Configs:** one YAML per experiment; every switch is explicit. Legacy-reproduction configs are named `legacy_*`.
- **Outputs:** `outputs/<name>/` holds config, logs, checkpoints, `results.json` and the legacy report CSVs. `cache/` is derived and rebuildable.
- **Tests:** parity tests keep verbatim copies of the reference code in `tests/*_ref.py`. Data-dependent tests skip when the data is missing.
- **Commits:** one per phase or sub-phase once its gate passes, if the user agrees. Nothing is committed yet.
- **Notes:** findings go to the issue register here and to the project memory.
