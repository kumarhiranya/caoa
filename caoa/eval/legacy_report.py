"""Legacy GT-instance pose evaluation (E2, E3, E5 in docs/PLAN.md), with functional parity.

* `eval_pose_report`: what `utils/training_utils.py:eval_pose_s2c_v2` computes after
  inference: MAE of the pose vector (t | 3x3 rotation | s) and its blocks, the legacy
  accuracies (`calc_pose_accuracy_s2c_nocls`), mean rotation and scale error, and
  the per-class tables of `per_class_accuracy` (with macro and micro averages).
* `valid_classes_report`: the older `utils/eval_utils.py:calc_pose_accuracy_s2c`,
  restricted to the 8 VALID_CLASSES, with its per-term rounding and its own
  (official-style) scale error.

Inputs are (N, 15) pose arrays (t, row-major 3x3 rotation, s) in the legacy
normalised test frame, plus per-sample symmetry tags and class labels.
"""

from collections import defaultdict

import numpy as np

from caoa.eval import metrics

VALID_CLASSES = ("bathtub", "bed", "bookshelf", "cabinet", "chair", "desk", "sofa", "table")


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
    errors = metrics.alignment_errors(*_split(pred), *_split(target), syms, metrics.MetricConfig.legacy())

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


def valid_classes_report(pred, target, syms, labels, translation_thresh=0.2, rot_thresh=20.0, scale_thresh=20.0) -> dict:
    t_p, R_p, s_p = _split(pred)
    t_g, R_g, s_g = _split(target)
    labels = np.asarray(labels)
    syms = np.asarray(syms)
    n_total = len(labels)
    present = [c for c in VALID_CLASSES if (labels == c).any()]

    per_class = {}
    cls_avg = dict.fromkeys(("t_acc", "q_acc", "s_acc", "total_acc"), 0.0)
    avg = dict.fromkeys(("t_acc", "q_acc", "s_acc", "total_acc"), 0.0)
    for c in present:
        m = labels == c
        err_s = 100.0 * np.abs(np.mean(s_p[m] / s_g[m] - 1.0, axis=1))
        acc_s = err_s < scale_thresh
        err_r = np.array([metrics.rotation_error_deg(a, b, s, "legacy") for a, b, s in zip(R_p[m], R_g[m], syms[m])])
        acc_r = err_r <= rot_thresh
        acc_t = np.linalg.norm(t_g[m] - t_p[m], axis=1) < translation_thresh
        acc = {"q_acc": acc_r.mean(), "t_acc": acc_t.mean(), "s_acc": acc_s.mean(), "total_acc": (acc_r & acc_t & acc_s).mean()}
        support = int(m.sum())
        per_class[c] = {k: float(v) for k, v in acc.items()} | {"support": support}
        for k, v in acc.items():
            cls_avg[k] += round(v / len(present), 4)
            avg[k] += round(v * support / n_total, 4)
    return {"per_class": per_class, "cls_avg": cls_avg, "avg": avg}
