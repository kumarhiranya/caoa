"""Turn per-object pose predictions into every number we report.

Predictions come out of the network in the test frame of `ObjectPoseDataset`
(object scaled to unit size); `to_world` maps them back to metres.

Legacy evaluation (reproduced first, see docs/PLAN.md):
* `legacy/<grouping>`: `eval_pose_s2c_v2`'s numbers (E2/E3) per class grouping;
* `legacy_valid_classes`: the older `calc_pose_accuracy_s2c` (E5).
Additions, always labelled as such:
* `benchmark/<convention>`: the official Scan2CAD protocol over the given scenes
  (all their GT objects, misses counted), in both submission conventions;
* `matched/<convention>`: Scan2CAD per-object formulas on the matched pairs only;
* `proxy`: Scan2CAD formulas in the normalised frame.
"""

import copy

import numpy as np

from caoa.data.pairs import Pair
from caoa.eval import groupings, legacy_report, metrics, scan2cad_benchmark
from caoa.geometry import nearest_rotation


def split_pose(p: np.ndarray):
    return p[:, :3], p[:, 3:12].reshape(-1, 3, 3), p[:, 12:15]


def to_world(pred: np.ndarray, norm_scale: np.ndarray, norm_offset: np.ndarray):
    """Box centre (m), rotation (projected onto SO(3)), extents (m) in the Scan2CAD world frame."""
    t, R, s = split_pose(pred.astype(np.float64))
    centers = (t + norm_offset) / norm_scale[:, None]
    # Extents must be positive to define a CAD scale; a non-positive prediction is simply wrong.
    extents = np.maximum(s, 1e-6) / norm_scale[:, None]
    return centers, np.stack([nearest_rotation(r) for r in R]), extents


def alignments_for(pairs: list[Pair], s2c_annotations: dict, centers, rotations, extents, convention: str):
    out = []
    for p, c, R, e in zip(pairs, centers, rotations, extents):
        anno = s2c_annotations[p.scene_id]
        model = anno["aligned_models"][p.model_index]
        out.append(scan2cad_benchmark.alignment_from_box(anno["trs"], p.catid_cad, p.id_cad, model["bbox"], model["center"], c, R, e, convention))
    return out


def evaluate_legacy_only(pairs, pred, target) -> dict:
    """E2 with the legacy GT-instance grouping; what the legacy training loop validated with."""
    return legacy_report.eval_pose_report(pred, target, [p.sym for p in pairs], groupings.class_labels(pairs, "scannet_class"))


def evaluate(pairs, pred, target, norm_scale, norm_offset, s2c_annotations, full_annotations, cad_appearances, scenes) -> dict:
    """pred / target: (N, 15) in the normalised test frame; `scenes`: benchmark scenes (all their GT counts)."""
    syms = [p.sym for p in pairs]
    results = {
        "legacy": {g: legacy_report.eval_pose_report(pred, target, syms, groupings.class_labels(pairs, g)) for g in groupings.GROUPINGS},
        "legacy_valid_classes": legacy_report.valid_classes_report(pred, target, syms, groupings.class_labels(pairs, "scannet_class")),
        "proxy": metrics.summarize(
            metrics.alignment_errors(*split_pose(pred), *split_pose(target), syms, metrics.MetricConfig()), [p.sem_name for p in pairs]
        ),
    }

    centers, rotations, extents = to_world(pred, norm_scale, norm_offset)
    scene_set = set(scenes)
    full = [a for a in full_annotations if a["id_scan"] in scene_set]
    for convention in scan2cad_benchmark.CONVENTIONS:
        alignments = alignments_for(pairs, s2c_annotations, centers, rotations, extents, convention)

        submission = {s: [] for s in scenes}
        for p, a in sorted(zip(pairs, alignments), key=lambda x: (x[0].scene_id, x[0].model_index)):
            if p.scene_id in scene_set:
                submission[p.scene_id].append(a)
        bench = scan2cad_benchmark.evaluate(submission, copy.deepcopy(full), cad_appearances)
        results[f"benchmark/{convention}"] = {
            "instance_mean": bench.instance_mean,
            "class_mean": bench.class_mean,
            "per_class": bench.class_accuracy(),
            "support": {k: total for k, (_, total) in bench.per_class.items()},
        }

        correct = []
        for p, a in zip(pairs, alignments):
            anno = s2c_annotations[p.scene_id]
            model = anno["aligned_models"][p.model_index]
            correct.append(scan2cad_benchmark.is_correct(a, *scan2cad_benchmark.ground_truth_pose(anno["trs"], model), model["sym"]))
        correct = np.array(correct)
        cats = np.array([scan2cad_benchmark.category_name(p.catid_cad) for p in pairs])
        per_class = {c: float(correct[cats == c].mean()) for c in sorted(set(cats))}
        results[f"matched/{convention}"] = {
            "instance_mean": float(correct.mean()),
            "class_mean": float(np.mean(list(per_class.values()))),
            "per_class": per_class,
        }
    return results
