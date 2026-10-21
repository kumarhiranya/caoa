---
name: consult-on-legacy-duplicates
description: "When legacy CAOA code has several near-duplicate functions/variants and it's unclear which one applies, ask the user before choosing"
metadata:
  type: feedback
---

The legacy codebase has many redundant and near-duplicate functions: several taxonomy loaders, `eval_pose_s2c` ×3, `decompose_mat4` / `TransMatFromTrs` copies, several dataset classes that share caches, two `prepare_data_inst.py` copies, two S2C annotation files, and so on. If there is any doubt about which one the CAOA results used, check with the user first. Don't assume or decide alone.

**Why:** the user said so on 2026-10-20: "Its important we are on the same page."

**How to apply:** the default evidence is the actual call path of the CAOA entry scripts (e.g. `train_pose.py` → `S2CCompletionPoseV2` → `eval_pose_s2c_v2`) with the server run's `config.json`. When the call path doesn't settle it (shared caches, multiple copies on disk, config overrides, commented alternatives), present the candidates and the evidence, then ask. Record agreed choices in PLAN.md. Related: [[preserve-legacy-evaluation]], [[caoa-replication-project]].

The user confirmed choices D1–D13 (PLAN.md "Legacy duplicate choices") on 2026-10-20. Among them: the completion annotation file (D2); `eval_pose_s2c_v2` only, with the older `eval_pose_s2c` copies not needed (D9); the `dataset_utils` TransMatFromTrs/decompose_mat4 (D5).
