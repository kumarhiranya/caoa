---
name: caoa-replication-project
description: Goal and ground rules for rebuilding the CAOA paper (arXiv 2606.18429) experiments as a new clean codebase
metadata:
  node_type: memory
  type: project
  originSessionId: be0601d6-11b5-43be-b0f8-dc7f68e3618a
  modified: 2026-09-26T21:59:59.619Z
---

The user wants to functionally replicate the experiments/findings of "CAOA – Completion-Assisted Object-CAD Alignment" (arXiv 2606.18429v2, June 2026) as a NEW codebase, using the legacy repo at /home/hiranya/research/Github/vr_reconstruction only as reference. Paper PDF saved at `paper_2606.18429.pdf` in the project root (downloaded 2026-09-26).

**Why:** the legacy repo is messy (7k-line dataset files, duplicated utils, hardcoded paths, unmaintained libs like MinkowskiEngine, Python 3.8).

**How to apply:** work step-by-step with the user. Step 1 (paper + codebase summary) was done 2026-09-26. Step 2 (data layer + metric) was done 2026-10-19 in the NEW repo `~/research/Github/caoa`: uv, Python 3.12, package `caoa`, git-initialised but nothing committed yet. It has a pair cache of 605 scenes / 3,811 pairs in `cache/pairs_v1` (199 MB), and 12 parity tests against verbatim legacy code in `tests/legacy_ref.py`. On 2026-10-20 I added: torch 2.9.1+cu126 + spconv-cu126 2.3.8 (`uv sync --extra train`, venv 6.4 GB); the official Scan2CAD benchmark (`caoa/eval/scan2cad_benchmark.py`, see [[scan2cad-benchmark-issues]]); the spconv port of legacy `MinkowskiFCNNP_v3` (`caoa/models/ocam.py` + `sparse.py`, mimicking ME downsampling sites and true max-pool, since spconv's max pool clamps at 0); the legacy-mode dataset (`caoa/data/pose_dataset.py`); and `scripts/train_pose.py` with `configs/pose/legacy_nocomp.yaml` (writes to `outputs/<name>/`). The pilot (legacy no-completion, 15k steps, ~5 it/s) was trained that day. Legacy reference curve on the full data: 43% at 1k steps → 71.8% at 13k (legacy metric). Decisions: [[caoa-design-decisions]]. See [[caoa-legacy-code-findings]] for paper↔code discrepancies found during the audit.

The full phased plan (P0–P12, gates, switch register, issue register, disk budget) is at `~/research/Github/caoa/docs/PLAN.md` (written 2026-10-20). Revised the same day after the legacy-eval review: it now has a legacy evaluation inventory (E1–E11), reproduction targets, and a new phase P3b (legacy evaluation parity) that is next. The user said NOT to apply fixes until asked. P0–P3 are done.

Status 2026-10-20: **P0–P3b done and committed** (caoa `9a4e81f`, the first commit on master). The pilot reproduced legacy: 73.0% vs 71.2–71.8% on the exact 1,864 test set. Next is P4. The user decided that P5 builds from the CURRENT ODGNet loader files (not the Feb 28 bytecode snapshot); review any proposed fixes with the user before applying them (in the port; the legacy repo stays untouched). The context flag is decided empirically in P6.

Status 2026-10-21: P4–P7 code is written and tested (uncommitted): the UpTrans port is exact vs legacy on the same weights, completion metrics and aggregation are exact, both completion datasets are ported, plus the train_completion script. Waiting on: approval of the completion cache size (~1.1 GiB float32 / 2.1 GiB float64), the sem_emb fallback for 91 CADs missing from semantic_embeddings.npy, and permission to commit. A tiny smoke cache (8 scenes, 40 MB) exists in cache/completion_v1_spatial0cm.

**Moved 2026-10-21:** the new repo is now `~/research/Github/caoa`. The old 190 GB `caoa` folder was a btrfs copy of the legacy repo and was deleted at the user's request. A `caoa-repro -> caoa` symlink stays until the P8 runs that started there finish.
- Pushed to the private GitHub repo https://github.com/kumarhiranya/caoa.git (branch master).
- **The whole project moved to the DGX** (user, 2026-10-21; the workstation GPU is shared with ollama and steam). P8 runs were stopped at ~5%. P7 and P8 both run on the DGX.
- The data goes over as a transfer bundle. `scripts/make_transfer_bundle.py` staged it at `/home/data/caoa_dgx_bundle/{now,later}`: symlink trees plus 0.5 GB of drive copies; `now/` is 7.8 GB, `later/` 29.6 GB. On the DGX it's set up with `configs/paths.dgx.template.yaml`.
- The repo's `CLAUDE.md` and `docs/claude_context/` (copies of these memory notes) carry the context. Pull before pushing, since two machines commit.
- **Data sent 2026-10-24** (workstation on the GlobalProtect VPN, ~16 MB/s): 15.2 GB to `/network/rit/lab/kumar_lab/research_data/Github/vr_reconstruction/caoa_data` on the DGX (the user chose this folder), checksum-verified. ScanNet and ShapeNet were not sent because the lab's `3D Datasets` copies are MD5-identical. `configs/paths.dgx.yaml` holds the DGX paths; `docs/HANDOFF_DGX.md` is the handoff note for the DGX agent. The user said a separate agent on the DGX does the setup and runs.

