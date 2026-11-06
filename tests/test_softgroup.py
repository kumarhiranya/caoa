import numpy as np

from caoa.data.softgroup import Instance, match_instances, rle_decode


def rle_encode(mask: np.ndarray) -> dict:
    """SoftGroup's `rle_encode` (softgroup/util/utils.py), for the round trip."""
    pixels = np.concatenate([[0], mask.astype(np.uint8), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {"length": len(mask), "counts": " ".join(str(x) for x in runs)}


def test_rle_round_trip():
    rng = np.random.default_rng(0)
    for p in (0.01, 0.3, 0.9):
        mask = rng.random(5000) < p
        mask[0] = mask[-1] = True  # runs touching both ends
        rle = rle_encode(mask)
        assert np.array_equal(rle_decode(rle), mask)
        # SphericalMask's `rle_encode_gpu_batch`: the same runs as an integer array
        as_array = {"length": rle["length"], "counts": np.asarray(rle["counts"].split(), dtype=np.int64)}
        assert np.array_equal(rle_decode(as_array), mask)


def _inst(ids):
    return Instance("s", 0, "chair", "chair", 1.0, np.asarray(ids), None)


def test_matching_is_one_to_one_highest_iou_first():
    n = 100
    gt = [np.zeros(n, bool) for _ in range(2)]
    gt[0][:40] = True
    gt[1][50:90] = True
    preds = [_inst(range(0, 40)), _inst(range(0, 30)), _inst(range(55, 95)), _inst(range(90, 100))]
    matches, iou = match_instances(preds, gt, iou_threshold=0.5)
    assert matches[0] == (0, 1.0)  # the exact mask wins object 0; pred 1 (IoU 0.75) stays unmatched
    assert 1 not in matches
    assert matches[2][0] == 1 and abs(matches[2][1] - 35 / 45) < 1e-12
    assert 3 not in matches  # no overlap
    assert iou.shape == (4, 2)


def test_matching_threshold():
    gt = [np.zeros(10, bool)]
    gt[0][:4] = True
    matches, _ = match_instances([_inst(range(0, 8))], gt, iou_threshold=0.5)  # IoU 0.5: inclusive
    assert matches == {0: (0, 0.5)}
    matches, _ = match_instances([_inst(range(0, 9))], gt, iou_threshold=0.5)
    assert matches == {}
