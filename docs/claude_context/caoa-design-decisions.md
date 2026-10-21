---
name: caoa-design-decisions
description: "User's 2026-10-19 choices for the CAOA rebuild: ScanNet subset strategy, legacy-vs-paper target, spconv backbone"
metadata:
  type: project
---

Decided by the user on 2026-10-19 (part of [[caoa-replication-project]]):

1. **ScanNet:** pilot on the 609 scenes already extracted in `/home/data/datasets/ScanNet_v2/dataset/scans/`. Update 2026-10-20: the user approved and we extracted the 206 missing VAL scenes (all 312 val are local now). Do NOT extract the remaining ~700 train scenes without asking; see [[disk-space-caution]]. The archives are solid LZMA2, so getting the 4 needed files means decompressing the whole ~2.5 GB `.sens` stream (~77 s per scene). See [[caoa-data-locations]].
2. **Target:** reproduce both, legacy behaviour first. Each paper↔code mismatch from [[caoa-legacy-code-findings]] becomes a config switch: normalized vs. metric threshold, OBB vs. CAD target, object-only vs. Siamese model, loss weights, scoring of missed objects. First match the legacy numbers, then switch to paper/official settings.
3. **Backbone:** port OCAM from MinkowskiEngine to spconv v2, mirroring the Minkowski layer layout. Keep PTv3 (Pointcept) for SEM.

**Why:** machine is an RTX A6000 with CUDA 12.6/12.8 and no working MinkowskiEngine build. `~/research/Github/caoa` is an older (Apr 2026) uncommitted copy of the legacy repo, not the new codebase, so don't overwrite it.

**How to apply:** don't reopen these choices. The new codebase still needs a home and a module plan agreed with the user.

**2026-10-21 (user):**
- Do NOT keep reading data from the external drive. Mirror locally everything the pipeline needs: the remaining 695 ScanNet train scenes (9.2 GiB), the SN-Indoor partials referenced by `aug_train_test_splits.pkl` (15.4 GiB), and the `model_normalized.obj` of SN-Indoor + S2C CADs (22,213 files, 16.9 GiB); ~52 GiB total incl. caches and checkpoints. Then build the full caches.
- **The paper's final numbers came from the CONCAT completion checkpoint** (`concat_snaug_s2cc_upTransb_comp`), as hard-coded in `build_completion_pose_datasets.py:132`. Use it for P8; the cache name `odg_s2ccompletion_upTransb1` in the pose configs is not the deciding evidence.
- **Order and scope (user, 2026-10-21): P8 first** (completions from the legacy concat ckpt-best, epoch 219), **then P7 = retrain S2C-C ONLY** (~18 h) to check that training reproduces. No context / concat / SN-Indoor retraining; the reproduced legacy checkpoints cover those.
