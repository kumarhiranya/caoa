# P3b — Legacy evaluation parity (2026-10-20)

Run: `configs/pose/legacy_nocomp.yaml`, i.e. the server run
`pose_mnk/S2Cpose_emb1024_vox1cm_wd1e-03_b64_lr1e-04_mnkpv3_rotmat_t2q1s2_adam_cosineannealing`,
reproduced on spconv. Outputs are in `outputs/legacy_nocomp/` (not tracked).

## Gate checks

| Check | Result |
|---|---|
| Legacy test set: `relpose_cache` rule on all 312 val scenes | **1,864 samples, every per-class support equal to `results/inst.csv`** (exact) |
| Parity: `eval_pose_s2c_v2` summary (MAE, accuracies, per-class tables, macro/micro) vs verbatim copy | exact (`tests/test_legacy_report.py`) |
| Parity: `calc_pose_accuracy_s2c` (E5) vs verbatim copy | exact |
| Parity: class rules vs legacy `load_s2c_data_pose_v2` logic; CAD names vs legacy SG naming | exact |
| Training loop | legacy counter semantics; epoch overrun (15,007 steps, 349 epochs); validation every 1k incl. 15k |
| Full test suite | 30 passed |

## Pilot vs the server run (legacy metric, `s2c_total`)

| Step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k | 14k | 15k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours | 39.2 | 53.5 | 57.6 | 59.0 | 62.4 | 67.3 | 67.9 | 70.4 | 70.6 | 71.6 | 71.9 | 72.9 | 73.2 | 73.3 | 73.0 |
| Server | 43.3 | 45.2 | 51.2 | 57.3 | 59.8 | 63.9 | 66.1 | 66.4 | 70.1 | 70.1 | 69.8 | 70.6 | 71.8 | 71.2 | (not logged) |

- At 14k: translation 94.1 / rotation 89.9 / scale 81.3, vs the server's 94.2 / 88.8 / 79.5.
- Test set: identical (1,864). Training data: ours 2,733 pairs (499 of 1,201 train scenes extracted) vs legacy 6,671.
- From 9k on, ours is within +0.5 to +2.1 points of the server run.

## Final model (step 15,007), legacy evaluation

- Legacy metric: micro 73.18%, macro 55.62% (`scannet_class`). `inst.csv`, from an unknown checkpoint of the same configuration family: 70.98 / 55.93.
- `cad_taxonomy_name`: macro 40.17. `table1` (grouping C): macro 59.74.
- MAE: pose 0.0892, t 0.0426, q 0.1010, s 0.1006. Mean legacy rotation error 14.4°; mean scale error 14.4%.
- `calc_pose_accuracy_s2c` (VALID_CLASSES): avg 75.42, cls_avg 76.62.
- Inference: 1.37 ms/sample (legacy-style timing and synced timing agree).

## Additions (labelled; not legacy)

| | instance mean | class mean |
|---|---|---|
| Official Scan2CAD protocol, plain convention (312 scenes, 3,184 GT) | 33.29 | 24.10 |
| Official protocol, −center convention | 42.87 | 31.85 |
| Matched-only, Scan2CAD formulas, plain | 56.87 | 39.87 |
| Matched-only, −center | 73.23 | 66.81 |

## Notes

- The earlier pilot (`outputs/legacy_nocomp_v2rule`, V2 class rule, 731 local val pairs) is superseded by this run.
- Table 1 supports under grouping C, and the fact that the paper's GT row is not attainable on this set, are recorded in `docs/PLAN.md`.
