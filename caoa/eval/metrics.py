"""Per-object alignment errors and pass/fail flags (translation / rotation / scale; 20 cm-or-units / 20 deg / 20 %).

Each object is scored against its own target. Two scoring rules, selected per component:

* "scan2cad": the per-object formulas of the official Scan2CAD benchmark (shared with `scan2cad_benchmark`):
  quaternion angle minimised over the symmetry-equivalent ground truths, scale error |mean(s_pred / s_gt) - 1|,
  every comparison inclusive (<=).
* "caoa": the CAOA metric, the one the paper's alignment numbers use (`MetricConfig.caoa`):
    - the rotation error is |xyz Euler angles| of the relative rotation, computed from (w, x, y, z) quaternions that
      are read in scipy's (x, y, z, w) order; reduced with max() for asymmetric objects and with min() over all
      components of all symmetry-equivalent targets for symmetric ones;
    - scale error mean(|s_pred / s_gt - 1|); translation and scale use strict <.

Translation units are those of the frame the poses are expressed in; the CAOA metric is computed in the pose
dataset's normalised frame (each object scaled to unit size), so its 0.2 is 20 % of the object's size.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from caoa.eval import scan2cad_benchmark
from caoa.geometry import matrix_to_quat, nearest_rotation, yaw_matrix

SYMMETRY_ORDER = scan2cad_benchmark.SYMMETRY_ORDER


@dataclass(frozen=True)
class MetricConfig:
    rotation: str = "scan2cad"  # "scan2cad" | "caoa"
    scale: str = "scan2cad"  # "scan2cad" | "caoa"
    strict_thresholds: bool = False  # the CAOA metric compares translation and scale with <, not <=
    translation_threshold: float = scan2cad_benchmark.THRESHOLD_TRANSLATION
    rotation_threshold_deg: float = scan2cad_benchmark.THRESHOLD_ROTATION
    scale_threshold_pct: float = scan2cad_benchmark.THRESHOLD_SCALE

    @classmethod
    def caoa(cls, **overrides):
        return cls(rotation="caoa", scale="caoa", strict_thresholds=True, **overrides)


def _caoa_euler_diff(R_pred: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    # The CAOA metric's convention: w-first quaternions read as x, y, z, w.
    pred = Rotation.from_quat(matrix_to_quat(R_pred)).as_matrix()
    target = Rotation.from_quat(matrix_to_quat(R_target)).as_matrix()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        return np.abs(Rotation.from_matrix(target @ pred.T).as_euler("xyz", degrees=True))


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray, sym: str, mode: str = "scan2cad") -> float:
    """R_pred may be any 3x3 matrix (raw network output); it is projected onto SO(3) first."""
    R_pred = nearest_rotation(R_pred)
    if mode == "scan2cad":
        return scan2cad_benchmark.rotation_error_deg(matrix_to_quat(R_pred), matrix_to_quat(R_gt), sym)
    if mode == "caoa":
        m = SYMMETRY_ORDER.get(sym, 1)
        diffs = [_caoa_euler_diff(R_pred, R_gt @ yaw_matrix(2.0 * np.pi * i / m)) for i in range(m)]
        return float(np.min(diffs)) if sym in SYMMETRY_ORDER else float(diffs[0].max())
    raise ValueError(f"unknown rotation mode {mode!r}")


def scale_error_pct(s_pred: np.ndarray, s_gt: np.ndarray, mode: str = "scan2cad") -> np.ndarray:
    if mode == "scan2cad":
        return 100.0 * np.abs(np.mean(s_pred / s_gt, axis=-1) - 1.0)
    if mode == "caoa":
        return 100.0 * np.mean(np.abs(s_pred / (s_gt + 1e-4) - 1.0), axis=-1)
    raise ValueError(f"unknown scale mode {mode!r}")


def alignment_errors(t_pred, R_pred, s_pred, t_gt, R_gt, s_gt, syms, cfg: MetricConfig = MetricConfig()) -> dict:
    """Per-object errors and pass/fail flags. Arrays are (N, 3), (N, 3, 3), (N, 3)."""
    err_t = np.linalg.norm(np.asarray(t_pred) - np.asarray(t_gt), axis=1)
    err_r = np.array([rotation_error_deg(Rp, Rg, s, cfg.rotation) for Rp, Rg, s in zip(R_pred, R_gt, syms)])
    err_s = scale_error_pct(np.asarray(s_pred), np.asarray(s_gt), cfg.scale)

    within = np.less if cfg.strict_thresholds else np.less_equal
    ok_t, ok_s = within(err_t, cfg.translation_threshold), within(err_s, cfg.scale_threshold_pct)
    ok_r = err_r <= cfg.rotation_threshold_deg
    return {"err_t": err_t, "err_r": err_r, "err_s": err_s, "ok_t": ok_t, "ok_r": ok_r, "ok_s": ok_s, "ok": ok_t & ok_r & ok_s}


def summarize(errors: dict, classes) -> dict:
    """Instance-average and class-average accuracy, plus per-class accuracy and support."""
    classes = np.asarray(classes)
    per_class = {c: float(errors["ok"][classes == c].mean()) for c in sorted(set(classes.tolist()))}
    return {
        "acc": float(errors["ok"].mean()),
        "class_avg_acc": float(np.mean(list(per_class.values()))),
        "acc_t": float(errors["ok_t"].mean()),
        "acc_r": float(errors["ok_r"].mean()),
        "acc_s": float(errors["ok_s"].mean()),
        "per_class": per_class,
        "support": {c: int((classes == c).sum()) for c in per_class},
    }
