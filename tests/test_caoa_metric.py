"""The CAOA metric's report, the class names of the pairs and the Table 1 grouping."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from caoa.data.pairs import SEM_RELABELLED, cache_dir, class_name, load_pairs
from caoa.eval import caoa_metric, groupings
from caoa.paths import RESOURCES, load_paths

SYMS = ["__SYM_NONE", "__SYM_ROTATE_UP_2", "__SYM_ROTATE_UP_4", "__SYM_ROTATE_UP_INF"]
CLASSES = ["chair", "table", "bed", "sofa", "toilet", "sink", "bathtub", "desk", "counter"]


def synthetic(seed, n=300, noise_deg=25):
    rng = np.random.default_rng(seed)
    t_gt = rng.uniform(-0.5, 0.5, (n, 3))
    R_gt = Rotation.random(n, random_state=rng).as_matrix()
    s_gt = rng.uniform(0.2, 1.0, (n, 3))
    t_p = t_gt + rng.normal(0, 0.12, (n, 3))
    R_p = Rotation.from_rotvec(np.radians(noise_deg) * rng.normal(size=(n, 3))).as_matrix() @ R_gt
    s_p = s_gt * rng.uniform(0.7, 1.3, (n, 3))
    pose = lambda t, R, s: np.concatenate([t, R.reshape(-1, 9), s], axis=1)  # noqa: E731
    return pose(t_p, R_p, s_p), pose(t_gt, R_gt, s_gt), rng.choice(SYMS, n).tolist(), rng.choice(CLASSES, n).tolist()


def test_perfect_predictions_score_one():
    _, target, syms, labels = synthetic(0)
    rep = caoa_metric.eval_pose_report(target, target, syms, labels)
    assert rep["avg"]["total_acc"] == 1.0 and rep["mae"]["pose"] == 0.0
    assert rep["per_class"]["total"]["micro_avg"] == rep["per_class"]["total"]["macro_avg"] == 1.0


@pytest.mark.parametrize("seed", [0, 1])
def test_report_is_consistent(seed):
    pred, target, syms, labels = synthetic(seed)
    rep = caoa_metric.eval_pose_report(pred, target, syms, labels)
    total = rep["per_class"]["total"]
    support = rep["per_class"]["support"]
    assert total["micro_avg"] == pytest.approx(rep["avg"]["total_acc"])
    assert total["micro_avg"] == pytest.approx(sum(total[c] * support[c] for c in support) / len(labels))
    assert total["macro_avg"] == pytest.approx(np.mean([total[c] for c in support]))
    assert rep["avg"]["total_acc"] <= min(rep["avg"][k] for k in ("t_acc", "q_acc", "s_acc"))
    assert 0 < rep["avg"]["total_acc"] < 1


def test_per_class_accuracy():
    acc, support = caoa_metric.per_class_accuracy([1, 0, 1, 1], ["a", "a", "b", "b"])
    assert acc == {"a": 0.5, "b": 1.0, "macro_avg": 0.75, "micro_avg": 0.75}
    assert support == {"a": 2, "b": 2}


def test_class_names():
    chair, table = "03001627", "04379243"
    assert class_name("chair", table) == "chair"
    assert class_name("unannotated", chair) == "chair"
    assert class_name("otherfurniture", chair) == "chair"
    assert class_name("otherfurniture", chair, SEM_RELABELLED) == "otherfurniture"
    assert class_name("unannotated", "02747177") is None  # ash bin: not an eval class -> dropped


def test_cad_taxonomy_and_table1_names():
    expected = {"02747177": "ash bin", "03337140": "file", "03928116": "forte-piano", "04554684": "automatic washer",
                "03207941": "dish washer", "02808440": "bathtub", "04256520": "sofa", "03211117": "display"}
    assert {c: groupings.cad_taxonomy_name(c) for c in expected} == expected
    assert {c: groupings.table1_name(c) for c in ("02808440", "02818832", "02747177", "04379243")} == {
        "02808440": "bath", "02818832": "other", "02747177": "other", "04379243": "table"}


VAL_SUPPORT = dict(table=260, bookshelf=110, cabinet=200, chair=931, sofa=73, desk=75, bed=67, toilet=54,
                   bathtub=28, refridgerator=31, sink=34, counter=1)


@pytest.mark.data
def test_val_set_is_the_1864_objects():
    paths = load_paths()
    val_scenes = (RESOURCES / "scannetv2_val.txt").read_text().split()
    cached = {p.stem for p in cache_dir(paths.cache).glob("scene*.pkl")}
    if set(val_scenes) - cached:
        pytest.skip("val scenes missing from the pair cache")
    pairs = load_pairs(paths.cache, "val")
    counts = {}
    for p in pairs:
        counts[p.sem_name] = counts.get(p.sem_name, 0) + 1
    assert counts == VAL_SUPPORT and len(pairs) == 1864


@pytest.mark.data
def test_train_set_size():
    """6,663 train pairs (scene0212_00, 8 matched models, has an empty ScanNet archive)."""
    paths = load_paths()
    train_scenes = set((RESOURCES / "scannetv2_train.txt").read_text().split())
    cached = {p.stem for p in cache_dir(paths.cache).glob("scene*.pkl")}
    annotated = set(json.load(open(paths.s2c_annotations)))
    if (train_scenes & annotated) - cached - {"scene0212_00"}:
        pytest.skip("annotated train scenes missing from the pair cache")
    assert len(load_pairs(paths.cache, "train")) == 6663
