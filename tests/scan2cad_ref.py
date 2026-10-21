"""Upstream Scan2CAD EvaluateBenchmark.py (skanti/Scan2CAD, Routines/Script), for parity tests.

Changes from the original: inputs are passed in memory instead of read from
CSV/JSON files (`predictions` maps scene id -> list of CSV rows), prints are
removed, and the per-class counts are returned. The scoring logic and SE3
helpers are copied verbatim; run it under np.errstate(all="raise") as the
original sets np.seterr(all="raise").
"""

import collections

import numpy as np
import quaternion


def compose_mat4(t, q, s, center=None):
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M


def decompose_mat4(M):
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz

    q = quaternion.from_rotation_matrix(R[0:3, 0:3])

    t = M[0:3, 3]
    return t, q, s


def get_top8_classes_scannet():
    top = collections.defaultdict(lambda: "other")
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "trashbin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    return top


def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:  # noqa: E722 (verbatim)
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def evaluate(predictions, appearances_cad, annotations):
    benchmark_per_scan = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    benchmark_per_class = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    catid2catname = get_top8_classes_scannet()

    groundtruth = {}
    cad2info = {}
    idscan2trs = {}

    testscenes = list(predictions.keys())

    testscenes_gt = []
    for r in annotations:
        id_scan = r["id_scan"]
        if id_scan not in testscenes:
            continue
        testscenes_gt.append(id_scan)

        idscan2trs[id_scan] = r["trs"]

        for model in r["aligned_models"]:
            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]
            catname_cad = catid2catname[catid_cad]
            model["n_total"] = len(r["aligned_models"])
            groundtruth.setdefault((id_scan, catid_cad), []).append(model)
            cad2info[(catid_cad, id_cad)] = {"sym": model["sym"], "catname": catname_cad}

            benchmark_per_class[catname_cad]["n_total"] += 1
            benchmark_per_scan[id_scan]["n_total"] += 1

    counter = 0
    for id_scan, alignments in predictions.items():
        if id_scan not in testscenes_gt:
            continue
        benchmark_per_scan[id_scan]["seen"] = 1

        appearance_counter = {}

        for alignment in alignments:
            catid_cad = alignment[0]
            id_cad = alignment[1]
            cadkey = catid_cad + "_" + id_cad
            if cadkey in appearances_cad[id_scan]:
                n_appearances_allowed = appearances_cad[id_scan][cadkey]
            else:
                n_appearances_allowed = 0

            appearance_counter.setdefault(cadkey, 0)
            if appearance_counter[cadkey] >= n_appearances_allowed:
                continue
            appearance_counter[cadkey] += 1

            catname_cad = cad2info[(catid_cad, id_cad)]["catname"]
            sym = cad2info[(catid_cad, id_cad)]["sym"]
            t = np.asarray(alignment[2:5], dtype=np.float64)
            q0 = np.asarray(alignment[5:9], dtype=np.float64)
            q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
            s = np.asarray(alignment[9:12], dtype=np.float64)

            key = (id_scan, catid_cad)
            for idx, model_gt in enumerate(groundtruth[key]):
                is_same_class = model_gt["catid_cad"] == catid_cad
                if is_same_class:
                    Mscan = compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])
                    Mcad = compose_mat4(model_gt["trs"]["translation"], model_gt["trs"]["rotation"], model_gt["trs"]["scale"], -np.array(model_gt["center"]))

                    t_gt, q_gt, s_gt = decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))

                    error_translation = np.linalg.norm(t - t_gt, ord=2)
                    error_scale = 100.0 * np.abs(np.mean(s / s_gt) - 1)

                    if sym == "__SYM_ROTATE_UP_2":
                        m = 2
                        tmp = [calc_rotation_diff(q, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0])) for i in range(m)]
                        error_rotation = np.min(tmp)
                    elif sym == "__SYM_ROTATE_UP_4":
                        m = 4
                        tmp = [calc_rotation_diff(q, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0])) for i in range(m)]
                        error_rotation = np.min(tmp)
                    elif sym == "__SYM_ROTATE_UP_INF":
                        m = 36
                        tmp = [calc_rotation_diff(q, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0])) for i in range(m)]
                        error_rotation = np.min(tmp)
                    else:
                        error_rotation = calc_rotation_diff(q, q_gt)

                    threshold_translation = 0.2
                    threshold_rotation = 20
                    threshold_scale = 20

                    is_valid_transformation = error_translation <= threshold_translation and error_rotation <= threshold_rotation and error_scale <= threshold_scale

                    counter += 1
                    if is_valid_transformation:
                        benchmark_per_scan[id_scan]["n_good"] += 1
                        benchmark_per_class[catname_cad]["n_good"] += 1
                        del groundtruth[key][idx]
                        break

    total_accuracy = {"n_good": 0, "n_total": 0, "n_scans": 0}
    for k, v in benchmark_per_scan.items():
        if "seen" in v:
            total_accuracy["n_good"] += v["n_good"]
            total_accuracy["n_total"] += v["n_total"]
            total_accuracy["n_scans"] += 1
    instance_mean_accuracy = float(total_accuracy["n_good"]) / total_accuracy["n_total"]

    accuracy_per_class = {}
    for k, v in benchmark_per_class.items():
        accuracy_per_class[k] = float(v["n_good"]) / v["n_total"]
    class_mean_accuracy = np.mean([v for k, v in accuracy_per_class.items()])
    per_class = {k: (v["n_good"], v["n_total"]) for k, v in benchmark_per_class.items()}
    per_scan = {k: (v["n_good"], v["n_total"]) for k, v in benchmark_per_scan.items() if "seen" in v}
    return instance_mean_accuracy, class_mean_accuracy, per_class, per_scan
