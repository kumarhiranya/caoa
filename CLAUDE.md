# CAOA reproduction: context for Claude

Clean re-implementation of *CAOA: Completion-Assisted Object-CAD Alignment* (arXiv 2606.18429).
The legacy research code (`vr_reconstruction`, with its ODGNet fork) is the reference. It isn't in this repo.
The full copy is on the user's workstation (`~/research/Github/vr_reconstruction`), and a trimmed copy (code, configs,
logs, the needed checkpoints and annotations) is at `<DATA_ROOT>/legacy` on the DGX. See "Data on the DGX".

Read these before working:
- `docs/PLAN.md`: the living plan. Phases P0–P12 with status, validation gates, the legacy evaluation
  inventory (E1–E11), reproduction targets, the legacy duplicate choices (D1–D13), paper-vs-code mismatches,
  and the switch, issue and disk registers.
- `docs/results/*.md`: per-phase results. `docs/results/legacy_curves/`: legacy training curves to compare against.
- `docs/claude_context/*.md`: Claude's memory notes from the workstation sessions (index: `MEMORY.md`).
  Absolute paths in them refer to the workstation.

## Standing rules from the user (always follow)

1. **Recreate first, fix later.** Reproduce the legacy behaviour as it was first, and preserve every
   evaluation method the legacy CAOA code used. "Faithful" means functional parity: wrapping or presenting
   results differently is fine.
2. **The codebase wins over the paper.** Record paper mismatches in PLAN.md, but don't chase them
   (there may have been post-processing or corrections).
3. **Scan2CAD fidelity.** The official Scan2CAD benchmark is replicated exactly. If Scan2CAD itself has
   an issue, tell the user and ask before changing anything.
4. **Ask about legacy duplicates.** The legacy code has many near-duplicate functions. If there is any
   doubt about which one CAOA used, ask the user. Never decide alone ("it's important we are on the same page").
5. **Review fixes with the user before applying them.** Fixes go into this repo, behind config switches.
   The legacy repo stays untouched.
6. **Scope:** CAOA's evaluations only. CIS2VR-era evaluations are out.
7. **Disk:** give a size estimate and get the user's OK before extracting or copying datasets or building large caches.
8. **Process:** every phase ends with a validation gate. Write a results note in `docs/results/`, update
   PLAN.md, and commit after the phase passes (WIP commits are OK). Don't commit `cache/` or `outputs/`.

## Status (2026-10-21)

- **Done:** P0–P3b (data layer, legacy and official metrics, OCAM on spconv, legacy evaluation parity),
  P6 (UpTrans port; re-scored legacy checkpoints within 0.07%). P4/P5: code done and caches built.
- **The project moved to the DGX on 2026-10-21.** The workstation GPU is shared with other processes.
  The workstation keeps the source data; the data needed here comes from the transfer bundle (below).
- **P8 (OCAM + completion): ready, not yet run.** Configs `configs/pose/legacy_comp_*.yaml`; the pose completion caches
  are in the bundle. The workstation runs were stopped at about 5% to move here. Run them on the DGX (runbook below).
- **P7 (CAPCM, S2C-C retraining): ready, not yet run.** Runbook below.
- P9–P12 are planned. Their data is the bundle's `later/` tier.
- If anything is still committed from the workstation, `git pull --rebase` before pushing.

## Data on the DGX

All data lives under one `<DATA_ROOT>`, sent from the workstation as a transfer bundle
(`scripts/make_transfer_bundle.py`, staged at `/home/data/caoa_dgx_bundle/` there; its `MANIFEST.md` lists every item):

| Tier | Size | Contents | Needed for |
|---|---|---|---|
| `now/` | 7.8 GB | caoa caches (pairs, completion r = 0 and r = 1 m, pose completions); a trimmed legacy repo (`legacy/`: code, configs, ODGNet experiment logs, 3 ODGNet `ckpt-best`, annotations, SEM embeddings and splits, legacy target tables, `pose_mnk` logs); ShapeNet split files | P7, P8, the test suite |
| `later/` | 29.6 GB | ScanNet (4 files per scene), the 3,049 Scan2CAD CADs, SEM checkpoints, SoftGroup val predictions | P9–P12, cache rebuilds |

```bash
# on the workstation
rsync -aL --info=progress2 /home/data/caoa_dgx_bundle/now/   <dgx>:<DATA_ROOT>/
rsync -aL --info=progress2 /home/data/caoa_dgx_bundle/later/ <dgx>:<DATA_ROOT>/
# on the DGX, in this repo
sed "s|<DATA_ROOT>|/abs/path/to/DATA_ROOT|" configs/paths.dgx.template.yaml > configs/paths.local.yaml
uv sync --extra train && uv run pytest -q
```

With only `now/` in place, the tests give 51 passed and 5 skipped (SN-Indoor, CAD meshes, ScanNet). With `later/` too, only
the SN-Indoor test skips. Not bundled: SN-Indoor partials (16 GB, only for SN-Indoor or concat retraining),
SoftGroup train predictions, ScanNet `.sens` and 2D data.

## P8 on the DGX: OCAM + completion (legacy inputs)

Details and targets are in PLAN.md § P8. There are six runs, one GPU each:

```bash
for c in s2cc_only_emb1024 s2cc_inst_emb1024 s2cc_only_emb512 s2cc_only_emb256 concat_only_emb1024 concat_inst_emb1024; do
  CUDA_VISIBLE_DEVICES=<gpu> nohup uv run python scripts/train_pose.py configs/pose/legacy_comp_$c.yaml > outputs/p8_$c.log 2>&1 &
done
```

- The S2C-C runs must match the logged legacy `pose_mnk` runs (best / last): `s2cc_inst_emb1024` 81.3 / 80.7,
  `s2cc_only_emb1024` 87.8 / 87.3, `s2cc_only_emb512` 85.0 / 84.8, `s2cc_only_emb256` 81.3 / 80.9 (legacy metric, 1,864 test set).
  The logs (`m_train_log.json`, `s2c_total` per 1k steps) are in `<DATA_ROOT>/legacy/checkpoints/pose_mnk/`.
- The concat runs are the paper's final setup; compare them with Table 1 (no saved legacy run).
- Speed: GPU-bound, because completions fill many more sparse voxels than partial scans. The workstation ran
  at ~1.3 it/s alone, so 15k or 25k steps take a few hours on a free GPU.
- Each run writes `outputs/<name>/results.json`, with the legacy report per grouping, `calc_pose_accuracy_s2c`,
  and the official Scan2CAD benchmark plus matched-only in both conventions.
- Afterwards: write `docs/results/P8.md` (per run: best and last vs target, official numbers), update PLAN.md, and commit.
  The next P8 step is the leak-free `world`-frame completion. It's a fix, so discuss it with the user first.

## P7 on the DGX: retrain S2C-C (CAPCM)

**Scope (user decision):** retrain the **S2C-C model only**, to check that `scripts/train_completion.py`
reproduces legacy training. The other Table 2 rows are covered by the legacy checkpoints re-scored in P6.
Don't expand the scope (SN-Indoor, concat, context) without asking. The SN-Indoor and concat
configs aren't written yet.

Legacy run being reproduced: `ODGNet/experiments/UpTrans/S2CCompletion/s2ccompletion_upTransb1`.
Our config: `configs/completion/legacy_s2cc.yaml`. Same recipe: UpTrans, AdamW 5e-4 / wd 5e-4, LambdaLR
(20, 0.8, min 0.01), BN-momentum scheduler, batch 32, epochs 0–400 inclusive, validation every 10 epochs,
ckpt-best by CDL1, then `test()` on ckpt-best.

### Data and run

P7 reads only the r = 0 completion cache (`<DATA_ROOT>/caoa_cache/completion_v1_spatial0cm/`: train 6,220 /
val 1,744 samples) and `semantic_embeddings.npy`, both in the bundle's `now/` tier.

```bash
# smoke test (~10 min): 2 epochs, validation at epoch 1, then test() on ckpt-best; delete its outputs afterwards
uv run python scripts/train_completion.py configs/completion/legacy_s2cc.yaml --max-epoch 1 --val-freq 1
rm -rf outputs/legacy_s2cc

# the full run, on one GPU
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_completion.py configs/completion/legacy_s2cc.yaml
```

- **GPU memory:** about 40 GB at batch 32 (legacy logged 39.99 GB). Use an 80 GB GPU, since a 40 GB one is borderline.
  On the workstation, it ran out of memory next to other jobs.
- **Time:** 0.89 s per step on an RTX A6000 (194 steps per epoch, 401 epochs), about 19 h plus validation.
  Expect less on A100/H100.
- **No `--resume` yet.** If DGX jobs have a time limit under ~20 h, add resume first, after checking with the user.
  It means saving the optimizer state and best metrics in `ckpt-last`, following ODGNet's
  `builder.resume_model` / `resume_optimizer`. The legacy run itself was resumed several times (4 logs).
- **Outputs** (`outputs/legacy_s2cc/`, git-ignored): `config.yaml`, `train_log.json`, `ckpt-best.pth`,
  `ckpt-last.pth`, `ckpt-epoch-*.pth` (last 10, as legacy), `test_results.json`. Checkpoints are 46 MB each,
  weights only; legacy's 138 MB also held the optimizer state.

### Gate and targets

- **Validation curve:** compare `train_log.json` with `docs/results/legacy_curves/s2ccompletion_upTransb1_val.csv`.
  These are legacy `validate()` numbers, class-averaged, on the val split (= the test split). The legacy best
  is epoch 180 with F 0.0535 / CDL1 43.60 / CDL2 16.91.
- **Test:** `test_results.json` (`test()` aggregation on ckpt-best) vs legacy F 0.0602 / CDL1 46.802 / CDL2 25.692
  (Table 2 "S2C-Completion"). Re-scoring the legacy checkpoint with this code gives 0.0601 / 46.835 / 25.533 (P6).
- **Tolerance:** agree it with the user, since run-to-run training variance is unknown. Report best epoch, best
  validation CDL1 and the test numbers next to the targets.
- **Known difference:** the train split lacks `scene0212_00` (its ScanNet archive is empty on every copy). That's
  8 pose pairs, fewer after the completion class filter.
- **Afterwards:** write `docs/results/P7.md`, update the P7 row and section in PLAN.md, then commit and push
  (results notes only, no checkpoints).

## Conventions

- uv project, Python 3.12, package `caoa`. Scripts in `scripts/`, configs in `configs/`, outputs in `outputs/<name>/`.
- Data paths come from `configs/paths.yaml`, overridden by the git-ignored `configs/paths.local.yaml` or `$CAOA_PATHS`.
- Parity tests compare against verbatim legacy and upstream code (`tests/*_ref.py`, and legacy files read from
  `legacy_repo`). Data-dependent tests skip when their data is missing.
- Commits end with the `Co-Authored-By` line given by the session.
