"""The CAOA metric's report on a set of objects (the paper's alignment evaluation).

`eval_pose_report`: MAE of the pose vector (t | 3x3 rotation | s) and its blocks, the accuracies under the CAOA metric
(`caoa.eval.metrics`, `MetricConfig.caoa`), mean rotation and scale error, and per-class tables (`per_class_accuracy`,
with macro and micro averages) for a class grouping (`caoa.eval.groupings`).

Inputs are (N, 15) pose arrays (t, row-major 3x3 rotation, s) in the pose dataset's normalised frame, plus per-object
symmetry tags and class labels.
"""

from collections import defaultdict

import numpy as np

from caoa.eval import metrics

def per_class_accuracy(correct, class_names) -> tuple[dict, dict]:
    """Per-class accuracy plus "macro_avg" and "micro_avg", and per-class counts."""
    correct_counts, total_counts = defaultdict(int), defaultdict(int)
    for ok, name in zip(correct, class_names):
        total_counts[name] += 1
        correct_counts[name] += bool(ok)
    acc = {name: correct_counts[name] / total_counts[name] for name in total_counts}
    acc["macro_avg"] = float(np.mean(list(acc.values())))
    acc["micro_avg"] = sum(correct_counts.values()) / sum(total_counts.values())
    return acc, dict(total_counts)


def _split(p):
    p = np.asarray(p, dtype=np.float64)
    return p[:, :3], p[:, 3:12].reshape(-1, 3, 3), p[:, 12:15]


def eval_pose_report(pred, target, syms, class_labels) -> dict:
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    diff = np.abs(pred - target)
    errors = metrics.alignment_errors(*_split(pred), *_split(target), syms, metrics.MetricConfig.caoa())

    per_class = {}
    for key, flags in (("total", errors["ok"]), ("t", errors["ok_t"]), ("q", errors["ok_r"]), ("s", errors["ok_s"])):
        per_class[key], support = per_class_accuracy(flags, class_labels)
    per_class["support"] = support

    return {
        "mae": {
            "pose": float(diff.mean()),
            "t": float(diff[:, :3].mean()),
            "q": float(diff[:, 3:12].mean()),
            "s": float(diff[:, 12:].mean()),
        },
        "avg": {
            "t_acc": float(errors["ok_t"].mean()),
            "q_acc": float(errors["ok_r"].mean()),
            "s_acc": float(errors["ok_s"].mean()),
            "total_acc": float(errors["ok"].mean()),
            "rotation_error_deg": float(errors["err_r"].mean()),
            "mean_scale_error_pct": float(errors["err_s"].mean()),
        },
        "per_class": per_class,
    }
