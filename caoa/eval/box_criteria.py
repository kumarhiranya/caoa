"""S2C box criteria: Scan2CAD's per-object criteria applied to the oriented box instead of the CAD.

A CAD-independent variant of `s2c_criteria`, reported next to it (the existing criteria are unchanged and stay the
reference against the GT CAD). The predicted box (centre, rotation, full extents) and the GT box are both mapped to
scan coordinates as the benchmark does (inverse scan `trs`), then compared with the benchmark's own formulas and
thresholds (`scan2cad_benchmark`):
* translation: distance between the box centres <= 20 cm;
* rotation: quaternion angle, minimised over the GT's rotational symmetries about the up axis, <= 20 deg;
* scale: |mean(extent_pred / extent_gt) - 1| <= 20 %.

Relation to `s2c_criteria/minus_center` with the GT CAD: rotation and scale are the same tests (the CAD scale ratio is the
box-extent ratio), but the benchmark's translation is that of T @ R @ S @ T(-center), a point that sits 2 R S center
from the box centre; rotation and scale errors move it whenever the CAD's bbox centre is off its origin. So the two
agree exactly only for CADs centred on their origin and otherwise differ by a few
points either way. With a retrieved CAD the box variant measures the alignment alone: the CAD's native proportions
and origin, which Scan2CAD's criteria compare, no longer matter.
"""

import numpy as np

from caoa.data.scan2cad import trs_matrix
from caoa.eval import scan2cad_benchmark as s2c
from caoa.geometry import compose, decompose, matrix_to_quat


def box_in_scan(scan_trs: dict, center, rotation, extents):
    """(t, q (w, x, y, z), s) of the box T @ R @ S in scan coordinates."""
    t, R, s = decompose(np.linalg.inv(trs_matrix(scan_trs)) @ compose(np.asarray(center, dtype=np.float64), np.asarray(rotation), np.asarray(extents, dtype=np.float64)))
    return t, matrix_to_quat(R), s


def box_errors(pred_box, gt_box, sym: str, scan_trs: dict) -> tuple[float, float, float]:
    """Translation (m), rotation (deg), scale (%) errors; boxes are (center, rotation, extents) in the Scan2CAD world."""
    t_p, q_p, s_p = box_in_scan(scan_trs, *pred_box)
    t_g, q_g, s_g = box_in_scan(scan_trs, *gt_box)
    return (float(np.linalg.norm(t_p - t_g)), s2c.rotation_error_deg(q_p, q_g, sym), 100.0 * abs(float(np.mean(s_p / s_g)) - 1.0))


def evaluate_boxes(pairs, centers, rotations, extents, s2c_annotations) -> dict:
    """Instance and class mean (benchmark categories) over `pairs`, plus the per-component pass rates."""
    ok = []
    for p, c, R, e in zip(pairs, centers, rotations, extents):
        et, er, es = box_errors((c, R, e), (p.box_center, p.box_rotation, p.box_extents), p.sym, s2c_annotations[p.scene_id]["trs"])
        ok.append((et <= s2c.THRESHOLD_TRANSLATION, er <= s2c.THRESHOLD_ROTATION, es <= s2c.THRESHOLD_SCALE))
    ok = np.array(ok, dtype=bool)
    correct = ok.all(1)
    cats = np.array([s2c.category_name(p.catid_cad) for p in pairs])
    per_class = {c: float(correct[cats == c].mean()) for c in sorted(set(cats))}
    return {
        "instance_mean": float(correct.mean()),
        "class_mean": float(np.mean(list(per_class.values()))),
        "acc_t": float(ok[:, 0].mean()), "acc_r": float(ok[:, 1].mean()), "acc_s": float(ok[:, 2].mean()),
        "per_class": per_class,
        "support": {c: int((cats == c).sum()) for c in per_class},
    }
