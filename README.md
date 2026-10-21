# caoa

Clean re-implementation of *CAOA: Completion-Assisted Object-CAD Alignment*
(arXiv 2606.18429). The legacy code (`vr_reconstruction`) is the reference: we
first reproduce what it did, evaluations included, and only then fix things
behind config switches. The phased plan, findings and decisions are in
[`docs/PLAN.md`](docs/PLAN.md); per-phase results are in `docs/results/`.
[`CLAUDE.md`](CLAUDE.md) has the standing rules, the current status, and the runbook for
the next step (P7 on the DGX).

## Setup

```bash
uv sync --extra train          # Python 3.12, torch 2.9.1+cu126, spconv-cu126
cp configs/paths.yaml configs/paths.local.yaml   # then edit the data locations
# (or, on a machine that received the transfer bundle: configs/paths.dgx.template.yaml, see CLAUDE.md)
uv run pytest -q               # parity tests against verbatim legacy/upstream code
```

## Usage

```bash
uv run python scripts/build_pairs.py                                   # object–CAD pair cache
uv run python scripts/train_pose.py configs/pose/legacy_nocomp.yaml    # legacy OCAM pilot
uv run python scripts/build_completion_cache.py --radius 0             # S2C-C completion samples (also --radius 1.0)
uv run python scripts/train_completion.py configs/completion/legacy_s2cc.yaml   # CAPCM (P7)
uv run python scripts/build_pose_completions.py --name ... --ckpt ... --config ...   # completions for the pose pairs (P8)
uv run python scripts/train_pose.py configs/pose/legacy_comp_s2cc_only_emb1024.yaml  # OCAM + completion (P8)
```

`scripts/train_pose.py` writes `outputs/<name>/results.json` with every reported
number: the legacy evaluation (per class grouping), the older
`calc_pose_accuracy_s2c`, and, labelled as additions, the official Scan2CAD
protocol in both submission conventions and matched-only accuracy.

## Layout

- `caoa/data`: ScanNet scenes, Scan2CAD annotations, object–CAD pairs and class rules, the pose dataset.
- `caoa/eval`: legacy metrics and reports, the official Scan2CAD benchmark, groupings.
- `caoa/models`: the OCAM network on spconv (MinkowskiEngine semantics); the ODGNet UpTrans port (CAPCM) with pure-PyTorch point ops and Chamfer.
- `tests/*_ref.py`: verbatim copies of the legacy and upstream code used as references.
