import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from caoa.data import scan2cad
from caoa.data.labels import scannet_split
from caoa.eval import scan2cad_benchmark as bench
from caoa.geometry import compose, decompose, matrix_to_quat, yaw_matrix
from caoa.paths import load_paths
from tests import scan2cad_ref

PATHS = load_paths()
pytestmark = pytest.mark.skipif(
    not (PATHS.scan2cad_full_annotations.exists() and PATHS.scan2cad_cad_appearances.exists()),
    reason="Scan2CAD benchmark files not found",
)


@pytest.fixture(scope="module")
def benchmark_inputs():
    with open(PATHS.scan2cad_full_annotations) as f:
        annotations = json.load(f)
    with open(PATHS.scan2cad_cad_appearances) as f:
        appearances = json.load(f)
    return annotations, appearances


def perturbed_alignment(rng, scan_trs, model, catid=None, id_cad=None):
    """A submission entry near the GT model's pose, in one of the two translation conventions."""
    M = np.linalg.inv(scan2cad.trs_matrix(scan_trs)) @ scan2cad.trs_matrix(model["trs"])
    if rng.random() < 0.3:  # the benchmark's own (-center) reference point
        M = M @ compose(t=-np.asarray(model["center"]))
    t, R, s = decompose(M)
    t = t + rng.normal(0, rng.choice([0.0, 0.05, 0.15]), 3)
    # Yaw about the CAD model's own up axis (+y), plus a little tilt.
    R = R @ yaw_matrix(np.radians(rng.choice([0, 10, 25, 90, 180]))) @ Rotation.from_rotvec(rng.normal(0, 0.05, 3)).as_matrix()
    s = s * (rng.uniform(0.8, 1.25, 3) if rng.random() < 0.5 else rng.uniform(0.8, 1.25))
    return bench.Alignment(catid or model["catid_cad"], id_cad or model["id_cad"], t, matrix_to_quat(R), s)


def synthetic_submission(rng, annotations, scenes):
    by_scene = {a["id_scan"]: a for a in annotations}
    all_models = [m for a in annotations for m in a["aligned_models"]]
    predictions = {}
    for sid in scenes:
        anno = by_scene[sid]
        preds = [perturbed_alignment(rng, anno["trs"], m) for m in anno["aligned_models"] if rng.random() > 0.15]
        if preds:
            preds.append(copy.deepcopy(preds[0]))  # may exceed the CAD's appearance count
        stranger = all_models[rng.integers(len(all_models))]  # CAD usually not in this scene's pool
        preds.append(perturbed_alignment(rng, anno["trs"], anno["aligned_models"][0], stranger["catid_cad"], stranger["id_cad"]))
        rng.shuffle(preds)
        predictions[sid] = preds
    return predictions


def as_csv_rows(predictions):
    return {sid: [[a.catid_cad, a.id_cad, *a.t, *a.q, *a.s] for a in preds] for sid, preds in predictions.items()}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_upstream_script(benchmark_inputs, seed):
    annotations, appearances = benchmark_inputs
    rng = np.random.default_rng(seed)
    val_scenes = [a["id_scan"] for a in annotations if scannet_split(a["id_scan"]) == "val"]
    scenes = rng.choice(val_scenes, 80, replace=False).tolist()
    predictions = synthetic_submission(rng, annotations, scenes)
    predictions[scenes[-1]] = []  # a scene with an empty submission still counts its objects

    ours = bench.evaluate(predictions, copy.deepcopy(annotations), appearances)
    with np.errstate(all="raise"):
        ref_instance, ref_class, ref_per_class, ref_per_scan = scan2cad_ref.evaluate(as_csv_rows(predictions), appearances, copy.deepcopy(annotations))

    assert ours.per_scan == ref_per_scan
    assert ours.per_class == ref_per_class
    assert ours.instance_mean == pytest.approx(ref_instance, abs=1e-12)
    assert ours.class_mean == pytest.approx(ref_class, abs=1e-12)
    assert 0.05 < ours.instance_mean < 0.95  # the synthetic submission exercises both outcomes


@pytest.mark.parametrize("convention", bench.CONVENTIONS)
def test_alignment_from_box_recovers_cad_pose(benchmark_inputs, convention):
    annotations, _ = benchmark_inputs
    for anno in annotations[:100]:
        for model in anno["aligned_models"]:
            center, R, extents = scan2cad.oriented_box(model)
            a = bench.alignment_from_box(anno["trs"], model["catid_cad"], model["id_cad"], model["bbox"], model["center"], center, R, extents, convention)
            M = np.linalg.inv(scan2cad.trs_matrix(anno["trs"])) @ scan2cad.trs_matrix(model["trs"])
            if convention == "minus_center":
                M = M @ compose(t=-np.asarray(model["center"]))
            np.testing.assert_allclose(compose(a.t, Rotation.from_quat(np.roll(a.q, -1)).as_matrix(), a.s), M, atol=1e-9)


def test_perfect_minus_center_submission_scores_100(benchmark_inputs):
    annotations, appearances = benchmark_inputs
    val = [a for a in annotations if scannet_split(a["id_scan"]) == "val"][:60]
    predictions = {
        a["id_scan"]: [
            bench.alignment_from_box(a["trs"], m["catid_cad"], m["id_cad"], m["bbox"], m["center"], *scan2cad.oriented_box(m), "minus_center")
            for m in a["aligned_models"]
        ]
        for a in val
    }
    assert bench.evaluate(predictions, copy.deepcopy(val), appearances).instance_mean == 1.0
