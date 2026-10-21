"""Functional parity of the legacy GT-instance evaluation (E2, E3, E5), class rules and groupings."""

import itertools
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from caoa.data.pairs import CLASS_RULES, cache_dir, class_rule_name, load_pairs
from caoa.eval import groupings, legacy_report
from caoa.geometry import matrix_to_quat, nearest_rotation
from caoa.paths import RESOURCES, load_paths
from tests import legacy_ref

SYMS = ["__SYM_NONE", "__SYM_ROTATE_UP_2", "__SYM_ROTATE_UP_4", "__SYM_ROTATE_UP_INF"]
CLASSES = ["chair", "table", "bed", "sofa", "toilet", "sink", "bathtub", "desk", "counter"]


def synthetic(seed, n=300, noise_deg=25):
    rng = np.random.default_rng(seed)
    t_gt = rng.uniform(-0.5, 0.5, (n, 3))
    R_gt = Rotation.random(n, random_state=rng).as_matrix()
    s_gt = rng.uniform(0.2, 1.0, (n, 3))
    t_p = t_gt + rng.normal(0, 0.12, (n, 3))
    R_p = Rotation.from_rotvec(np.radians(noise_deg) * rng.normal(size=(n, 3))).as_matrix() @ R_gt + rng.normal(0, 0.05, (n, 3, 3))
    s_p = s_gt * rng.uniform(0.7, 1.3, (n, 3))
    pose = lambda t, R, s: np.concatenate([t, R.reshape(-1, 9), s], axis=1)  # noqa: E731
    syms = rng.choice(SYMS, n).tolist()
    labels = rng.choice(CLASSES, n).tolist()
    return pose(t_p, R_p, s_p), pose(t_gt, R_gt, s_gt), syms, labels


@pytest.mark.parametrize("seed", [0, 1])
def test_eval_pose_report_matches_eval_pose_s2c_v2(seed):
    pred, target, syms, labels = synthetic(seed)
    ours = legacy_report.eval_pose_report(pred, target, syms, labels)
    ref = legacy_ref.eval_pose_s2c_v2_summary(target, pred, syms, labels)

    for k in ("pose", "t", "q", "s"):
        assert ours["mae"][k] == pytest.approx(ref[f"{k}_mae"], abs=1e-12)
    for k in ("t_acc", "q_acc", "s_acc", "total_acc"):
        assert ours["avg"][k] == pytest.approx(ref["avg"][k], abs=1e-12)
    assert ours["avg"]["rotation_error_deg"] == pytest.approx(ref["avg"]["rotation_error_deg"], abs=1e-6)
    assert ours["avg"]["mean_scale_error_pct"] == pytest.approx(ref["avg"]["_mean_scale_error"], abs=1e-9)
    for ours_key, ref_key in (("total", "per_class_total"), ("t", "per_class_t"), ("q", "per_class_q"), ("s", "per_class_s")):
        assert ours["per_class"][ours_key] == pytest.approx(ref["avg"][ref_key], abs=1e-12)
    assert ours["per_class"]["support"] == ref["avg"]["per_class_support"]


@pytest.mark.parametrize("seed", [0, 1])
def test_valid_classes_report_matches_calc_pose_accuracy_s2c(seed):
    pred, target, syms, labels = synthetic(seed)

    # The legacy function takes quaternion poses (t | q | s); feed it the rotations our port scores.
    def quat_pose(p, project):
        R = p[:, 3:12].reshape(-1, 3, 3)
        q = np.stack([matrix_to_quat(nearest_rotation(r) if project else r) for r in R])
        return np.concatenate([p[:, :3], q, p[:, 12:]], axis=1)

    ours = legacy_report.valid_classes_report(pred, target, syms, labels)
    ref_per_class, ref_cls_avg, ref_avg = legacy_ref.calc_pose_accuracy_s2c(quat_pose(target, False), quat_pose(pred, True), syms, labels, ret=True)

    assert set(ours["per_class"]) == set(ref_per_class)
    for c, r in ref_per_class.items():
        for k in ("q_acc", "t_acc", "s_acc", "total_acc", "support"):
            assert ours["per_class"][c][k] == pytest.approx(r[k], abs=1e-12)
    for k in ("t_acc", "q_acc", "s_acc", "total_acc"):
        assert ours["cls_avg"][k] == pytest.approx(ref_cls_avg[k], abs=1e-9)
        assert ours["avg"][k] == pytest.approx(ref_avg[k], abs=1e-9)


def test_class_rules_match_legacy_code():
    taxonomy, _ = legacy_ref.load_taxonomy(RESOURCES / "shapenet_taxonomy.json")
    valid_names = legacy_ref.get_cadid_valid_names(taxonomy, legacy_ref.VALID_CLASSES)
    scannet_classes = sorted(legacy_ref.SCANNETV2_NAMES - {"wall", "floor"})
    for sc, catid in itertools.product(scannet_classes, sorted(taxonomy)):
        assert class_rule_name(sc, catid, "v2") == legacy_ref.legacy_sem_name(sc, catid, valid_names, ("unannotated",))
        assert class_rule_name(sc, catid, "relpose_cache") == legacy_ref.legacy_sem_name(sc, catid, valid_names, ("unannotated", "otherfurniture"))
        assert class_rule_name(sc, catid, "all_matched") is not None


def test_cad_taxonomy_names_match_legacy_sg_naming():
    synset_names, _ = legacy_ref.LoadShapenetTaxonomy(RESOURCES / "shapenet_taxonomy.json")
    checked = 0
    for catid, names in synset_names.items():
        if len(names & legacy_ref.SCANNETV2_NAMES) > 1:
            continue  # legacy result would depend on set order
        assert groupings.cad_taxonomy_name(catid) == legacy_ref.sg_class_name(synset_names, catid)
        checked += 1
    assert checked > 300
    # Names that appear in the legacy results/sg_mnk_comp.csv
    expected = {"02747177": "ash bin", "03337140": "file", "03928116": "forte-piano", "04554684": "automatic washer",
                "03207941": "dish washer", "02808440": "bathtub", "04256520": "sofa", "03211117": "display"}
    assert {c: groupings.cad_taxonomy_name(c) for c in expected} == expected
    assert {c: groupings.table1_name(c) for c in ("02808440", "02818832", "02747177", "04379243")} == {
        "02808440": "bath", "02818832": "other", "02747177": "other", "04379243": "table"}


INST_CSV_SUPPORT = dict(table=260, bookshelf=110, cabinet=200, chair=931, sofa=73, desk=75, bed=67, toilet=54,
                        bathtub=28, refridgerator=31, sink=34, counter=1)


@pytest.mark.data
def test_relpose_cache_rule_reproduces_the_legacy_test_set():
    """With all 312 val scenes, the legacy test set is 1,864 samples with inst.csv's per-class supports."""
    paths = load_paths()
    val_scenes = (RESOURCES / "scannetv2_val.txt").read_text().split()
    cached = {p.stem for p in cache_dir(paths.cache).glob("scene*.pkl")}
    missing = set(val_scenes) - cached
    if missing:
        pytest.skip(f"{len(missing)} val scenes not in the pair cache")
    pairs = load_pairs(paths.cache, "val", class_rule="relpose_cache")
    counts = {}
    for p in pairs:
        counts[p.sem_name] = counts.get(p.sem_name, 0) + 1
    assert counts == INST_CSV_SUPPORT
    assert len(pairs) == 1864
    assert set(CLASS_RULES) == {"v2", "relpose_cache", "all_matched"}


@pytest.mark.data
def test_relpose_cache_rule_reproduces_the_legacy_train_set():
    """Legacy train set: 6,671 pairs. scene0212_00 (8 matched models) has an empty archive on every copy."""
    paths = load_paths()
    train_scenes = set((RESOURCES / "scannetv2_train.txt").read_text().split())
    cached = {p.stem for p in cache_dir(paths.cache).glob("scene*.pkl")}
    annotated = set(json.load(open(paths.s2c_annotations)))
    missing = (train_scenes & annotated) - cached
    if missing - {"scene0212_00"}:
        pytest.skip(f"{len(missing)} annotated train scenes not in the pair cache")
    assert len(load_pairs(paths.cache, "train", class_rule="relpose_cache")) == 6671 - 8 * ("scene0212_00" in missing)
