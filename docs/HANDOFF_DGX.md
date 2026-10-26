# Handoff: continuing on the DGX (2026-10-24)

For the Claude agent picking up this project on the UAlbany DGX. Read `CLAUDE.md` first (standing rules, P7/P8
runbooks, targets); this note covers what is where on the DGX and what was checked before the handoff.
Workstation-side context is in `docs/claude_context/` (paths there refer to the workstation).

## Where things are

| What | Path on the DGX |
|---|---|
| This repo (cloned 2026-10-21 at `d8f8b76`; `git pull` first) | `~/dgx_lab/research_data/Github/caoa` (= `/network/rit/dgx/dgx_kumar_lab/research_data/Github/caoa`) |
| Data package: `caoa_cache/` + `legacy/` (11,201 files, 15.2 GB) | `/network/rit/lab/kumar_lab/research_data/3D Datasets/caoa_data` |
| ScanNet scans (the 4 files per scene we use) | `/network/rit/lab/kumar_lab/research_data/3D Datasets/ScanNet_v2/dataset/scans` |
| ShapeNetCore v2 (CADs, split files) and the SN-Indoor partials | `/network/rit/lab/kumar_lab/research_data/3D Datasets/ShapeNetV2` and `.../ShapeNetV2-augmented` |

These are the lab share's host paths (login node). The training runs happen inside the Slurm job's pyxis
container, where the lab share is mounted only at `/workspace/lab` (and `~/dgx_lab` at `/workspace/dgx_lab` as
well as its `/network/...` path). Set the paths there with
`sed "s|/network/rit/lab/kumar_lab|/workspace/lab|" configs/paths.dgx.yaml > configs/paths.local.yaml`;
outside the container, `cp configs/paths.dgx.yaml configs/paths.local.yaml`.

- `caoa_data` was first sent into the lab's old `vr_reconstruction` checkout; the user moved it next to the lab
  datasets on 2026-10-24. The legacy code we compare against is `caoa_data/legacy` (`paths.legacy_repo`), not
  that checkout.
- ScanNet and ShapeNet were not sent: the lab already had them. For every file the package would have carried
  (6,036 ScanNet files, 3,049 CAD models, 3 ShapeNet split files, 4 Scan2CAD files) the DGX copy has the same MD5.
  All 194,208 files of the workstation's ShapeNet mirror (CADs used by S2C and SN-Indoor, and the 171,992 SN-Indoor
  partials) exist on the DGX with the same sizes (sizes only, not checksums).
- `scene0212_00` is missing on the DGX too (its ScanNet archive is empty everywhere), so the known train-split
  difference (6,663 = 6,671 − 8 pose pairs) stays.

## Contents of `caoa_data`

| Path | Contents | Used by |
|---|---|---|
| `caoa_cache/pairs_v2` | pose pairs, 1,505 scenes | P8, tests, P10 |
| `caoa_cache/completion_v1_spatial0cm`, `..._spatial100cm` | completion samples, context r = 0 and r = 1 m | P7 (r = 0), tests |
| `caoa_cache/pose_completions_v1/{odg_s2ccompletion_upTransb1, odg_concat_snaug_s2cc_upTransb_comp}` | completed pose inputs, S2C-C and concat models | P8 |
| `legacy/` | legacy code and configs (all `.py/.yaml/.yml/.cu/.cpp/.h/.txt/.sh/.md` under 5 MB, incl. ODGNet, Pointcept/PTv3, SoftGroup, Scan2CAD code) | parity tests, P9–P11 ports |
| `legacy/ODGNet/experiments/**/{config.yaml,*.log}` | ODGNet run configs and training logs | P7 curves |
| `legacy/ODGNet/experiments/.../ckpt-best.pth` ×3 | S2C-C (`UpTrans/S2CCompletion/s2ccompletion_upTransb1`), concat (`UpTrans_comp/Concat_SN_S2CC/concat_snaug_s2cc_upTransb_comp`), S2C-C + context (`UpTrans_comp/S2CCompletion/s2ccompletion_upTransb1_comp`) | P7 target, P8, context tests |
| `legacy/scan2cad_annotator/` | `scan2cad_v2_completion_annotations.json`, `s2c_cad_splits.pkl` | everything; P9 splits |
| `legacy/Scan2CAD/dataset/` | `full_annotations.json`, `cad_appearances.json` (official benchmark) | evaluation |
| `legacy/data/ShapeNetSemantic/classification_ptv3/semantic_embeddings.npy`, `legacy/data/s2c_symmetry/` | SEM embeddings (classification, symmetry) | P7, P10, P9 gate |
| `legacy/checkpoints/symmetry_ptv3/`, `legacy/checkpoints/classification_ptv3/` | SEM (PTv3) checkpoints | P9 |
| `legacy/checkpoints/pose_mnk/*/` | the saved legacy pose runs' logs and configs (no weights) | P8 targets |
| `legacy/results/*.csv` | legacy target tables (`comp.csv`, `compinst.csv`, `sg_mnk_comp.csv`, `perclass_results.csv`, `ODG_*_S2CC.csv`, ...) | P6/P8/P11 targets |
| `legacy/data/objects/softgroup_preds/sg_raw/{val/, *_val.npy, *.json}` | SoftGroup val predictions and legacy val caches | P11 |

## Checked before the handoff

- After the transfer, a checksum pass (`rsync --checksum --dry-run`) found no file differing from the workstation.
- Every path in `configs/paths.dgx.yaml` exists on the DGX.
- On the workstation, a replica of this exact layout (built from the same bundle, plus the ScanNet/ShapeNet files
  that the DGX copies match) ran the test suite, a P8 smoke run (`legacy_comp_s2cc_only_emb1024`, one epoch +
  full evaluation) and a P7 smoke run (2 epochs at batch 8, validation, `test()` on ckpt-best), all under
  `strace`. All three passed:
  - Tests: 55 passed, 1 skipped. The skip is the SN-Indoor test, because the replica left out the partials. The
    DGX has them, so it should run there. That's 56 tests in all; the workstation has 63, and the other 7 are
    `test_legacy_checkpoints_load_strictly` cases for checkpoints that aren't on the DGX.
  - P8 smoke: 6,663 train / 1,864 val pairs with the S2C-C completions, 105 steps, then the full evaluation
    (legacy report, `calc_pose_accuracy_s2c`, official benchmark on 312 scenes, matched-only).
  - P7 smoke: 6,220 train / 1,744 val samples. Validation at epoch 1 gave CDL1 59.48, and `test()` gave
    F 0.0473 / CDL1 56.45 / CDL2 28.79.
  - The trace showed no reads from outside the data package, the repo, the Python environment and system files.
    The only other accesses were existence checks on the repo's `outputs/` and `cache/` folders.

## Not on the DGX

Left on the workstation (`~/research/Github/vr_reconstruction`) or the user's external drive. Ask the user if a
task needs any of it; sending it needs the workstation on the GlobalProtect VPN.
- Other ODGNet checkpoints: the baselines (PCN, ShapeNet-34/55), `ckpt-last` of concat and SN-Indoor, and all other
  runs. Only `scripts/rescore_completion.py` (P6, done) uses them, and `test_legacy_checkpoints_load_strictly`
  runs only for the three checkpoints above.
- SoftGroup train predictions (18 GB) and `sg_processed` (67 GB); ScanNet `.sens` / 2D data; `pose_mnk` weights.
- CIS2VR-era material (ModelNet, `object_db`, `real_world_data`, ...): out of scope.

## The DGX (checked 2026-10-24)

- **Login:** `dgx-head01.its.albany.edu`, user `hk418542`. The login node has no GPU; work on GPUs goes through
  Slurm. Compute nodes can't be reached with ssh (the node's GPU model wasn't checked yet: run
  `srun --gres=gpu:1 nvidia-smi` once).
- **Slurm:** partition `dgx` (default), 24 nodes × 8 GPUs, 256 CPUs, ~2 TB RAM each. QoS `freetier`: at most
  7 days per job, 8 GPUs, 8 jobs and 256 CPUs per user at once. The 7-day limit is far above P7's ~19 h, so
  `train_completion.py` needs no resume. Runbooks in `CLAUDE.md` show `CUDA_VISIBLE_DEVICES=... nohup`; on
  the DGX, submit each run as its own job with `--gres=gpu:1`.
- **Other jobs on the account:** the user runs unrelated jobs under the same user (on 2026-10-24: `khg-code`
  with 2 GPUs and `khg-ollama` with 3). They count toward the 8-GPU cap. Check `squeue -u $USER`, agree the
  GPU plan with the user (P7 needs 1 GPU, P8 six), and never touch those jobs.
- **Storage:**

  | Location | Size / free | Use |
  |---|---|---|
  | home `/network/rit/home/hk418542` | 13 GB / 5.4 GB | nothing large: keep the uv cache and Python installs out |
  | `~/lab` = `/network/rit/lab/kumar_lab` | 9.3 TB / 5.8 TB | data, caches, uv cache |
  | `~/dgx_lab` = `/network/rit/dgx/dgx_kumar_lab` | 1 TB / 51 GB (96% full) | the repo (`.venv` ~6.5 GB, outputs); nothing big |

- **Software:** no `uv` yet; system `python3`; `module avail` has CUDA 12.1/12.2/12.8 and gcc. The environment is
  torch 2.9.1+cu126 and spconv-cu126 (`uv sync --extra train`), whose wheels carry their own CUDA runtime; they
  need a recent enough NVIDIA driver on the compute nodes (CUDA 12.6). On the workstation, the uv cache was
  6.4 GB and `.venv` 6.5 GB. Suggested: install uv in `~/.local/bin`, and set `UV_CACHE_DIR` and
  `UV_PYTHON_INSTALL_DIR` under `~/lab` (with the cache and `.venv` on different filesystems, uv copies
  instead of hard-linking; `UV_LINK_MODE=copy` silences the warning).

## First steps

1. `git pull`; `cp configs/paths.dgx.yaml configs/paths.local.yaml`.
2. Set up uv and `uv sync --extra train` as above; run `uv run pytest -q` in a GPU job. Expected: see "Checked
   before the handoff".
3. Agree the GPU plan with the user, then run P8 and P7 per `CLAUDE.md`.
