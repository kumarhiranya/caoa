"""The official Scan2CAD alignment benchmark, reproduced as-is.

Port of `Routines/Script/EvaluateBenchmark.py` from skanti/Scan2CAD, with CSV/JSON file handling
replaced by in-memory inputs. Known quirks are kept on purpose; see
the module-level notes below and `tests/test_scan2cad_benchmark.py`.

Protocol:
* A submission is, per scene, a list of alignments (ShapeNet catid, CAD id,
  t, q (w, x, y, z), s) giving the CAD model's pose T @ R @ S in raw ScanNet
  scan coordinates (the convention of Scan2CAD's own aligner and its
  Alignment2Mesh viewer).
* Alignments of a CAD model beyond the number of times it appears in the
  scene (`cad_appearances.json`) are ignored, as are CAD models absent from it.
* Each alignment, in submission order, is compared with the not-yet-matched
  ground-truth models of the same category, in annotation order; the first
  one within 20 cm / 20 deg / 20 % counts as correct and is consumed.
* Accuracy = correct / all ground-truth models of the evaluated scenes, so
  missed objects count as failures. The class mean is over the top-8
  categories plus "other".

Quirks kept for fidelity:
* The ground-truth pose is T @ R @ S @ T(-center), not T @ R @ S, so a perfect
  plain-convention submission is off by |R S center| in translation (perfect
  poses score 77.2% on val). Results are reported in both conventions; see
  `alignment_from_box`.
* Scale error is |mean(s_pred / s_gt) - 1|; per-axis errors can cancel.
* Infinite rotational symmetry is sampled every 10 degrees (36 steps).
* Predicted quaternions are not normalised.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from caoa.data.scan2cad import trs_matrix
from caoa.geometry import compose, decompose, matrix_to_quat

THRESHOLD_TRANSLATION = 0.2  # metres
THRESHOLD_ROTATION = 20.0  # degrees
THRESHOLD_SCALE = 20.0  # percent

TOP8_CATEGORIES = {
    "03211117": "display",
    "04379243": "table",
    "02808440": "bathtub",
    "02747177": "trashbin",
    "04256520": "sofa",
    "03001627": "chair",
    "02933112": "cabinet",
    "02871439": "bookshelf",
}
SYMMETRY_ORDER = {"__SYM_ROTATE_UP_2": 2, "__SYM_ROTATE_UP_4": 4, "__SYM_ROTATE_UP_INF": 36}


def category_name(catid: str) -> str:
    return TOP8_CATEGORIES.get(catid, "other")


@dataclass
class Alignment:
    catid_cad: str
    id_cad: str
    t: np.ndarray  # (3,) scan coordinates, metres
    q: np.ndarray  # (4,) w, x, y, z
    s: np.ndarray  # (3,)


@dataclass
class BenchmarkResult:
    instance_mean: float
    class_mean: float
    per_class: dict[str, tuple[int, int]]  # name -> (n_good, n_total)
    per_scan: dict[str, tuple[int, int]]

    def class_accuracy(self) -> dict[str, float]:
        return {k: good / total for k, (good, total) in self.per_class.items()}


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _yaw_quat(angle):
    return np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])


def _rotation_diff_deg(q, q_ref) -> float:
    dot = abs(float(np.dot(q_ref, q)))
    if dot > 1.0:  # the original's arccos raises here (np.seterr(all="raise")) and it returns 0
        return 0.0
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def rotation_error_deg(q, q_gt, sym: str) -> float:
    m = SYMMETRY_ORDER.get(sym, 1)
    return min(_rotation_diff_deg(q, _quat_mul(q_gt, _yaw_quat(2.0 * np.pi * i / m))) for i in range(m))


def ground_truth_pose(scan_trs: dict, model: dict):
    """(t, q, s) the benchmark compares against, in scan coordinates."""
    M = np.linalg.inv(trs_matrix(scan_trs)) @ trs_matrix(model["trs"]) @ compose(t=-np.asarray(model["center"], dtype=np.float64))
    t, R, s = decompose(M)
    return t, matrix_to_quat(R), s


def is_correct(alignment: Alignment, t_gt, q_gt, s_gt, sym: str) -> bool:
    error_translation = np.linalg.norm(np.asarray(alignment.t) - t_gt)
    error_scale = 100.0 * abs(np.mean(np.asarray(alignment.s) / s_gt) - 1.0)
    error_rotation = rotation_error_deg(np.asarray(alignment.q, dtype=np.float64), q_gt, sym)
    return error_translation <= THRESHOLD_TRANSLATION and error_rotation <= THRESHOLD_ROTATION and error_scale <= THRESHOLD_SCALE


def evaluate(
    predictions: Mapping[str, Sequence[Alignment]],
    annotations: Iterable[dict],
    cad_appearances: Mapping[str, Mapping[str, int]],
) -> BenchmarkResult:
    """Score a submission. Scenes are those in `predictions` (an empty list still counts the scene's objects)."""
    per_scan = defaultdict(lambda: [0, 0])
    per_class = defaultdict(lambda: [0, 0])
    groundtruth: dict[tuple[str, str], list[dict]] = {}
    sym_of: dict[tuple[str, str], str] = {}
    scan_trs: dict[str, dict] = {}

    for anno in annotations:
        id_scan = anno["id_scan"]
        if id_scan not in predictions:
            continue
        scan_trs[id_scan] = anno["trs"]
        for model in anno["aligned_models"]:
            groundtruth.setdefault((id_scan, model["catid_cad"]), []).append(model)
            sym_of[(model["catid_cad"], model["id_cad"])] = model["sym"]
            per_class[category_name(model["catid_cad"])][1] += 1
            per_scan[id_scan][1] += 1

    for id_scan, alignments in predictions.items():
        if id_scan not in scan_trs:
            continue
        allowed = cad_appearances[id_scan]
        used = defaultdict(int)
        for a in alignments:
            cadkey = f"{a.catid_cad}_{a.id_cad}"
            if used[cadkey] >= allowed.get(cadkey, 0):
                continue
            used[cadkey] += 1

            sym = sym_of[(a.catid_cad, a.id_cad)]
            candidates = groundtruth[(id_scan, a.catid_cad)]
            for idx, model_gt in enumerate(candidates):
                if is_correct(a, *ground_truth_pose(scan_trs[id_scan], model_gt), sym):
                    per_scan[id_scan][0] += 1
                    per_class[category_name(a.catid_cad)][0] += 1
                    del candidates[idx]
                    break

    n_good = sum(good for good, _ in per_scan.values())
    n_total = sum(total for _, total in per_scan.values())
    return BenchmarkResult(
        instance_mean=n_good / n_total,
        class_mean=float(np.mean([good / total for good, total in per_class.values()])),
        per_class={k: tuple(v) for k, v in per_class.items()},
        per_scan={k: tuple(v) for k, v in per_scan.items()},
    )


CONVENTIONS = ("plain", "minus_center")


def alignment_from_box(
    scan_trs: dict, catid_cad: str, id_cad: str, cad_bbox, cad_center, box_center, box_rotation, box_extents, convention: str = "plain"
) -> Alignment:
    """Submission entry for a CAD model placed on a predicted oriented box.

    The box is in Scan2CAD world coordinates (as in `caoa.data.pairs`);
    `cad_bbox` (half extents) and `cad_center` are the CAD model's own box in
    CAD coordinates. Both conventions are reported, labelled:
      "plain": the CAD pose T @ R @ S, the format Scan2CAD's tools produce
               and read (penalised by the benchmark's -center reference);
      "minus_center": T @ R @ S @ T(-center), the benchmark's own GT
               definition, under which a perfect pose scores 100%.
    """
    c = np.asarray(cad_center, dtype=np.float64)
    s = np.asarray(box_extents) / (2.0 * np.asarray(cad_bbox, dtype=np.float64))
    R = np.asarray(box_rotation)
    M = compose(np.asarray(box_center) - R @ (s * c), R, s)
    if convention == "minus_center":
        M = M @ compose(t=-c)
    elif convention != "plain":
        raise ValueError(f"unknown convention {convention!r}, expected one of {CONVENTIONS}")
    t, R_scan, s_scan = decompose(np.linalg.inv(trs_matrix(scan_trs)) @ M)
    return Alignment(catid_cad, id_cad, t, matrix_to_quat(R_scan), s_scan)
