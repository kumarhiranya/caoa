# caoa-repro

Clean re-implementation of *CAOA: Completion-Assisted Object-CAD Alignment*
(arXiv 2606.18429). The legacy code (`vr_reconstruction`) is the reference: we
first reproduce what it did, evaluations included, and only then fix things
behind config switches. The phased plan, findings and decisions are in
[`docs/PLAN.md`](docs/PLAN.md); per-phase results are in `docs/results/`.

## Setup

```bash
uv sync --extra train          # Python 3.12, torch 2.9.1+cu126, spconv-cu126
cp configs/paths.yaml configs/paths.local.yaml   # then edit the data locations
uv run pytest -q               # parity tests against verbatim legacy/upstream code
```

## Usage

```bash
uv run python scripts/build_pairs.py                                   # object–CAD pair cache
uv run python scripts/train_pose.py configs/pose/legacy_nocomp.yaml    # legacy OCAM pilot
```

`scripts/train_pose.py` writes `outputs/<name>/results.json` with every reported
number: the legacy evaluation (per class grouping), the older
`calc_pose_accuracy_s2c`, and, labelled as additions, the official Scan2CAD
protocol in both submission conventions and matched-only accuracy.

## Layout

- `caoa/data`: ScanNet scenes, Scan2CAD annotations, object–CAD pairs and class rules, the pose dataset.
- `caoa/eval`: legacy metrics and reports, the official Scan2CAD benchmark, groupings.
- `caoa/models`: the OCAM network on spconv, with MinkowskiEngine semantics.
- `tests/*_ref.py`: verbatim copies of the legacy and upstream code used as references.
