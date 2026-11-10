# CAOA: context for Claude

Implementation of *CAOA: Completion-Assisted Object-CAD Alignment* (arXiv 2606.18429). The README describes the
pipeline, setup, training and evaluation; `docs/RESULTS.md` has the results.

## Standing rules from the user

1. **Scan2CAD fidelity.** The official Scan2CAD benchmark code (`caoa/eval/scan2cad_benchmark.py`) is replicated
   exactly. If Scan2CAD itself has an issue, tell the user and ask before changing anything.
2. **Evaluation:** S2C-C annotations only. Objects we don't evaluate on never count as failures (predicted instances:
   only matched S2C-C objects are scored).
3. **Results** describe what this code produces; the headline metric is the S2C box criteria, then the CAOA metric.
4. **Disk:** give a size estimate and get the user's OK before copying datasets or building large caches.
5. **Process:** commit after each finished step. Don't commit `cache/` or `outputs/`. Review fixes and design changes
   with the user before applying them.

## Running on the UAlbany DGX

- Runs happen inside the Slurm job's pyxis container, where the lab share is mounted at `/workspace/lab`:
  `sed "s|/network/rit/lab/kumar_lab|/workspace/lab|" configs/paths.dgx.yaml > configs/paths.local.yaml`.
- Data: `caoa_data/` on the lab share (`research_data/3D Datasets/caoa_data`): `data/` (S2C-C and Scan2CAD files,
  semantic embeddings, instance predictions), `caoa_cache/`, `release_v1/`, `sphericalmask/` (SphericalMask's own
  setup, docs/instance_predictions.md). ScanNet and ShapeNetV2 are the lab's copies next to it.
- spconv: one GPU per process (`CUDA_VISIBLE_DEVICES=<g>`).
- Long runs: queue scripts in `outputs/*.sh`, started with `setsid nohup`, logging start / exit lines to a queue log.
  Wait on log lines, not on `pgrep -f <pattern>` (it matches its own command line).
- Speeds (A100): OCAM ~3 h (single trunk) to ~5 h (Siamese) for 25k steps with both GPUs busy; CAPCM ~16 h; the full
  pipeline on one instance source ~7 min.

## Conventions

- uv project, Python 3.12, package `caoa`. Scripts in `scripts/`, configs in `configs/`, outputs in `outputs/<name>/`.
- Data paths come from `configs/paths.yaml`, overridden by the git-ignored `configs/paths.local.yaml` or `$CAOA_PATHS`.
- Run names: `capcm_<data>`, `sem`, `ocam_<variant>[_seed<N>]`, `retrieval`, `pipeline_<instances>__<ocam>[_common]`.
- Commits end with the `Co-Authored-By` line given by the session.
