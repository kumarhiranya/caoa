"""Turn per-object pose predictions into every number we report.

Predictions come out of the network in the test frame of `ObjectPoseDataset` (object scaled to unit size); `to_world`
maps them back to metres. Only the evaluated S2C-C objects are scored; objects outside that set never count.

* `caoa/<grouping>`: the CAOA metric (`caoa.eval.caoa_metric`; the paper's metric) per class grouping
  (`caoa.eval.groupings`; "table1" is the paper's Table 1 grouping);
* `s2c_criteria/<convention>`: Scan2CAD's per-object criteria (20 cm / 20 deg / 20 %, the benchmark's own error
  formulas) on the CAD placed by the predicted box, each object against its own S2C-C ground truth, in both submission
  conventions (`scan2cad_benchmark.CONVENTIONS`);
* `s2c_box`: the same criteria on the predicted box instead of the CAD (`caoa.eval.box_criteria`; CAD-independent).
"""

import numpy as np

from caoa.data.pairs import Pair
from caoa.eval import box_criteria, caoa_metric, groupings, scan2cad_benchmark
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


def evaluate_caoa_only(pairs, pred, target) -> dict:
    """The CAOA metric with the pairs' own classes; what training validates with."""
    return caoa_metric.eval_pose_report(pred, target, [p.sym for p in pairs], groupings.class_labels(pairs, "scannet_class"))


def evaluate(pairs, pred, target, norm_scale, norm_offset, s2c_annotations) -> dict:
    """pred / target: (N, 15) in the normalised test frame; `s2c_annotations`: the S2C-C annotations, by scene."""
    syms = [p.sym for p in pairs]
    results = {"caoa": {g: caoa_metric.eval_pose_report(pred, target, syms, groupings.class_labels(pairs, g)) for g in groupings.GROUPINGS}}

    centers, rotations, extents = to_world(pred, norm_scale, norm_offset)
    cats = np.array([scan2cad_benchmark.category_name(p.catid_cad) for p in pairs])
    for convention in scan2cad_benchmark.CONVENTIONS:
        correct = []
        for p, a in zip(pairs, alignments_for(pairs, s2c_annotations, centers, rotations, extents, convention)):
            anno = s2c_annotations[p.scene_id]
            model = anno["aligned_models"][p.model_index]
            correct.append(scan2cad_benchmark.is_correct(a, *scan2cad_benchmark.ground_truth_pose(anno["trs"], model), model["sym"]))
        correct = np.array(correct)
        per_class = {c: float(correct[cats == c].mean()) for c in sorted(set(cats))}
        results[f"s2c_criteria/{convention}"] = {
            "instance_mean": float(correct.mean()),
            "class_mean": float(np.mean(list(per_class.values()))),
            "per_class": per_class,
            "support": {c: int((cats == c).sum()) for c in per_class},
        }
    results["s2c_box"] = box_criteria.evaluate_boxes(pairs, centers, rotations, extents, s2c_annotations)
    return results
