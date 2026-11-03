# CAOA reproduction: context for Claude

Clean re-implementation of *CAOA: Completion-Assisted Object-CAD Alignment* (arXiv 2606.18429).
The legacy research code (`vr_reconstruction`, with its ODGNet fork) is the reference. It isn't in this repo.
The full copy is on the user's workstation (`~/research/Github/vr_reconstruction`), and a trimmed copy (code, configs,
logs, the needed checkpoints and annotations) is at `caoa_data/legacy` on the DGX. See "Data on the DGX".

Read these before working:
- `docs/HANDOFF_DGX.md`: **start here on the DGX.** Where the data is, what was verified, what isn't there,
  and the DGX itself (Slurm limits, storage, software).
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

## Status (2026-11-03)

- **Done:** P0–P3b (data layer, legacy and official metrics, OCAM on spconv, legacy evaluation parity),
  P6 (UpTrans port; re-scored legacy checkpoints within 0.07%). P4/P5: code done and caches built.
- **The project moved to the DGX on 2026-10-21.** The workstation GPU is shared with other processes.
  The workstation keeps the source data. The data was sent to the DGX on 2026-10-24 and verified (below).
- **P7 done** (2026-10-27, `docs/results/P7.md`): S2C-C retrained, test CDL1 45.00 vs legacy 46.80; accepted by the user.
- **P8 legacy reproduction done** (2026-10-27, `docs/results/P8.md`): all six runs; the S2C-C runs match the `pose_mnk`
  logs (87.8 / 81.9 / 85.6 / 82.7 vs 87.8 / 81.3 / 85.0 / 81.3). Next in P8: the leak-free `world`-frame completion
  (done 2026-10-28, below). The runbooks below are kept for reference.
- **P8 leak fixed** (2026-10-28, `docs/results/P8.md`): completion-only OCAM read the GT rotation from the completion's point
  layout; leak-free (`completion_frame: world`) 82.6–83.1 vs 86.75 leaky; inst+comp barely affected.
- **P9 port done** (`docs/results/P9.md`): PTv3 vendored in `caoa/models/ptv3`. The legacy SEM `embeddings.npy` is degenerate
  (exported with the wrong inputs); regenerated as `cad_bbox` (user). P10a/b done (below); extra CAPCM runs
  (concat, S2C-C + context) done on the DGX by user request. P10c–P12: below and PLAN.md.
- **Evaluation (user, 2026-10-29):** S2C-C annotations only; objects we don't evaluate on never count as failures. The
  official all-GT protocol (`benchmark/*`) is no longer reported; the addition to the legacy metric is `s2c_criteria/*`
  (Scan2CAD's per-object criteria on the evaluated S2C-C objects). CAD retrieval (lost legacy module; 3D-Match or similar)
  is framework completion only; evaluations use the GT CAD.
- **P7 extension done** (concat 44.42 vs 45.37; context 22.18 vs 22.47). **P10a/b done** (`docs/results/P10.md`): Siamese
  with separate trunks is the best leak-free OCAM; P10c (+ Chamfer loss) running on DGX job 134281 (dgx05).
- If anything is still committed from the workstation, `git pull --rebase` before pushing.

## Data on the DGX

Sent on 2026-10-24 from the workstation's transfer bundle (`scripts/make_transfer_bundle.py`) and checksum-verified.
The runs happen inside the Slurm job's pyxis container, where the lab share is mounted at `/workspace/lab`:
`sed "s|/network/rit/lab/kumar_lab|/workspace/lab|" configs/paths.dgx.yaml > configs/paths.local.yaml` sets every path. Details, contents and what's missing are in
`docs/HANDOFF_DGX.md`.

| What | Where on the DGX |
|---|---|
| `caoa_data/`: caoa caches (pairs, completion r = 0 and r = 1 m, pose completions) and a trimmed legacy repo (`legacy/`: code, configs, ODGNet logs, 3 ODGNet `ckpt-best`, annotations, SEM embeddings, splits and checkpoints, target tables, `pose_mnk` logs, SoftGroup val predictions); 15.2 GB | `/network/rit/lab/kumar_lab/research_data/3D Datasets/caoa_data` |
| ScanNet scans; ShapeNetCore v2 and the SN-Indoor partials (the lab's copies, identical to the workstation's) | `/network/rit/lab/kumar_lab/research_data/3D Datasets/{ScanNet_v2/dataset/scans, ShapeNetV2, ShapeNetV2-augmented}` |

Not on the DGX: ODGNet checkpoints other than the three `ckpt-best` (only P6's re-scoring used them),
SoftGroup train predictions, ScanNet `.sens` and 2D data. Ask the user if a task needs them.

## P8 on the DGX: OCAM + completion (legacy inputs)

Details and targets are in PLAN.md § P8. There are six runs, one GPU each. The commands below are for a machine
with local GPUs: on the DGX, submit each one as a Slurm job with `--gres=gpu:1` (see `docs/HANDOFF_DGX.md`).

```bash
for c in s2cc_only_emb1024 s2cc_inst_emb1024 s2cc_only_emb512 s2cc_only_emb256 concat_only_emb1024 concat_inst_emb1024; do
  CUDA_VISIBLE_DEVICES=<gpu> nohup uv run python scripts/train_pose.py configs/pose/legacy_comp_$c.yaml > outputs/p8_$c.log 2>&1 &
done
```

- The S2C-C runs must match the logged legacy `pose_mnk` runs (best / last): `s2cc_inst_emb1024` 81.3 / 80.7,
  `s2cc_only_emb1024` 87.8 / 87.3, `s2cc_only_emb512` 85.0 / 84.8, `s2cc_only_emb256` 81.3 / 80.9 (legacy metric, 1,864 test set).
  The logs (`m_train_log.json`, `s2c_total` per 1k steps) are in `caoa_data/legacy/checkpoints/pose_mnk/`.
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

P7 reads only the r = 0 completion cache (`caoa_data/caoa_cache/completion_v1_spatial0cm/`: train 6,220 /
val 1,744 samples) and `semantic_embeddings.npy`, both on the DGX.

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
- **No `--resume`, and none needed:** the DGX QoS allows 7 days per job. If it's ever needed, save the optimizer
  state and best metrics in `ckpt-last`, following ODGNet's `builder.resume_model` / `resume_optimizer`
  (check with the user first). The legacy run itself was resumed several times (4 logs).
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
