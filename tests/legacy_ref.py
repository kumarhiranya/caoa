"""Verbatim copies of legacy functions, used as references in parity tests.

Sources (vr_reconstruction repo):
  utils/eval_utils.py:          calc_rotation_diff, calc_rotation_accuracy, calc_pose_accuracy_s2c_nocls
  datasets/custom_datasets_pose.py: calc_Mbbox, calc_Mcad
  utils/dataset_utils.py:       decompose_mat4
  SoftGroup prepare_data_inst.py / scannet_util.py: prepare_inst, get_raw2scannetv2_label_map

Only prints, plotting and file output were removed; the logic is unchanged.
"""

import json

import numpy as np
import plyfile
import quaternion
from scipy.spatial.transform import Rotation as R


def calc_rotation_diff(quat1, quat2, output_format="deg"):
    if isinstance(quat1, np.quaternion):
        quat1 = quaternion.as_float_array(quat1)
    if isinstance(quat2, np.quaternion):
        quat2 = quaternion.as_float_array(quat2)
    rot_matrix1 = R.from_quat(quat1).as_matrix()
    rot_matrix2 = R.from_quat(quat2).as_matrix()
    rot_diff_matrix = np.dot(rot_matrix2, rot_matrix1.T)
    if output_format == "deg":
        rot_diff_deg = R.from_matrix(rot_diff_matrix).as_euler("xyz", degrees=True)
    else:
        rot_diff_deg = R.from_matrix(rot_diff_matrix).as_euler("xyz", degrees=False)
    return rot_diff_deg


def calc_rotation_accuracy(gt_quat, pred_quat, symmetries, threshold=20, return_rotation_error=False):
    error_rotation_arr = []
    for quat_p, quat_gt, sym in zip(pred_quat, gt_quat, symmetries):
        if len(quat_p) == 9:
            quat_p = quaternion.from_rotation_matrix(quat_p.reshape((3, 3)))
            quat_gt = quaternion.from_rotation_matrix(quat_gt.reshape((3, 3)))
            quat_gt = [quat_gt.w, quat_gt.x, quat_gt.y, quat_gt.z]
        if sym == "__SYM_ROTATE_UP_2":
            m = 2
            tmp = [abs(calc_rotation_diff(quat_p, np.quaternion(*quat_gt) * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0]))) for i in range(m)]
            error_rotation = np.min(tmp)
        elif sym == "__SYM_ROTATE_UP_4":
            m = 4
            tmp = [abs(calc_rotation_diff(quat_p, np.quaternion(*quat_gt) * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0]))) for i in range(m)]
            error_rotation = np.min(tmp)
        elif sym == "__SYM_ROTATE_UP_INF":
            m = 36
            tmp = [abs(calc_rotation_diff(quat_p, np.quaternion(*quat_gt) * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0]))) for i in range(m)]
            error_rotation = np.min(tmp)
        else:
            error_rotation = abs(calc_rotation_diff(quat_p, quat_gt))
        error_rotation_arr.append(error_rotation.max())
    error_rotation_arr = np.array(error_rotation_arr)
    accuracy_list = error_rotation_arr <= threshold
    if return_rotation_error:
        return accuracy_list, error_rotation_arr
    return accuracy_list


def calc_pose_accuracy_s2c_nocls(gt_pose, pred_pose, symmetries, scale_thresh=20, rot_thresh=20, translation_thresh=0.2):
    error_scale = np.mean(np.abs((pred_pose[:, -3:] / (gt_pose[:, -3:] + 1e-4)) - 1), axis=1) * 100
    acc_scale = error_scale < scale_thresh
    acc_rot, err_rot_deg = calc_rotation_accuracy(gt_quat=gt_pose[:, 3:-3], pred_quat=pred_pose[:, 3:-3], symmetries=symmetries, threshold=rot_thresh, return_rotation_error=True)
    error_translation = np.linalg.norm(gt_pose[:, :3] - pred_pose[:, :3], ord=2, axis=1)
    acc_translation = error_translation < translation_thresh
    total_acc = acc_rot & acc_translation & acc_scale
    return {
        "acc_t": acc_translation, "acc_r": acc_rot, "acc_s": acc_scale, "total": total_acc,
        "err_r": err_rot_deg, "err_s": error_scale, "err_t": error_translation,
    }


def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M


def calc_Mcad(model):
    t = np.asarray(model["trs"]["translation"], dtype=np.float64)
    q = np.asarray(model["trs"]["rotation"], dtype=np.float64)
    s = np.asarray(model["trs"]["scale"], dtype=np.float64)
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    Rm = np.eye(4)
    Rm[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)
    return T.dot(Rm).dot(S)


def decompose_mat4(M):
    Rm = M[0:3, 0:3]
    sx = np.linalg.norm(Rm[0:3, 0])
    sy = np.linalg.norm(Rm[0:3, 1])
    sz = np.linalg.norm(Rm[0:3, 2])
    s = np.array([sx, sy, sz])
    Rm[:, 0] /= sx
    Rm[:, 1] /= sy
    Rm[:, 2] /= sz
    rot_mat = Rm[0:3, 0:3]
    q = quaternion.from_rotation_matrix(rot_mat)
    t = M[0:3, 3]
    return t, [q.w, q.x, q.y, q.z], s


g_label_names = [
    "unannotated", "wall", "floor", "chair", "table", "desk", "bed", "bookshelf", "sofa", "sink",
    "bathtub", "toilet", "curtain", "counter", "door", "window", "shower curtain", "refridgerator",
    "picture", "cabinet", "otherfurniture",
]


def get_raw2scannetv2_label_map(tsv_path):
    lines = [line.rstrip() for line in open(tsv_path)]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(g_label_names)
        elements = lines[i].split("\t")
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = "unannotated"
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet


remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


def prepare_inst(fn, g_raw2scannetv2, split="val"):
    """SoftGroup `prepare_data_inst.f`, returning instead of saving. `fn` is the *_vh_clean_2.ply path."""
    fn2 = fn[:-3] + "labels.ply"
    fn3 = fn[:-15] + "_vh_clean_2.0.010000.segs.json"
    fn4 = fn[:-15] + ".aggregation.json"

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    raw_points = points[:, :3]
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    f2 = plyfile.PlyData().read(fn2)
    sem_labels = remapper[np.array(f2.elements[0]["label"])]

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d["segIndices"]
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d["segGroups"]:
            if g_raw2scannetv2[x["label"]] != "wall" and g_raw2scannetv2[x["label"]] != "floor":
                instance_segids.append(x["segments"])
                labels.append(x["label"])
    # The original compared the relative path "val/scene0217_00_vh_clean_2.ply".
    if fn.endswith("scene0217_00_vh_clean_2.ply") and split == "val" and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]:
        instance_segids = instance_segids[: int(len(instance_segids) / 2)]

    instance_labels = np.ones(sem_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
    return raw_points, colors, sem_labels, instance_labels


# --- utils/eval_utils.py, verbatim except prints -----------------------------------------------

VALID_CLASSES = {'bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'desk', 'sofa', 'table'}


def calc_pose_accuracy_s2c(gt_pose, pred_pose, symmetries, labels, scale_thresh=20, rot_thresh=20, translation_thresh=.2, ret=False, verbose=True):
    perclass_pose = {c: {} for c in VALID_CLASSES}
    total_obj_num = gt_pose.shape[0]
    cls_avg_acc = {'t_acc': 0, 'q_acc': 0, 's_acc': 0, 'total_acc': 0, }
    avg_acc = {'t_acc': 0, 'q_acc': 0, 's_acc': 0, 'total_acc': 0, }

    for c in perclass_pose.keys():
        mask = [x == c for x in labels]
        perclass_pose[c]['gt'] = gt_pose[mask]
        perclass_pose[c]['pred'] = pred_pose[mask]
        perclass_pose[c]['sym'] = [string for string, m in zip(symmetries, mask) if m]
        perclass_pose[c]['support'] = perclass_pose[c]['gt'].shape[0]

    perclass_results = {c: {} for c in perclass_pose.keys() if perclass_pose[c]['support'] > 0}
    for c in perclass_results.keys():
        gt_pose_ = perclass_pose[c]['gt']
        pred_pose_ = perclass_pose[c]['pred']
        symmetries_ = perclass_pose[c]['sym']

        if perclass_pose[c]['support'] > 0:
            error_scale = np.array([100.0 * np.abs(np.mean(s_pred / s_bbox - 1)) for s_pred, s_bbox in zip(pred_pose_[:, 7:], gt_pose_[:, 7:])])
            acc_scale = error_scale < scale_thresh

            acc_rot = calc_rotation_accuracy(gt_quat=gt_pose_[:, 3:7], pred_quat=pred_pose_[:, 3:7], symmetries=symmetries_, threshold=rot_thresh)

            error_translation = np.linalg.norm(gt_pose_[:, :3] - pred_pose_[:, :3], ord=2, axis=1)
            acc_translation = error_translation < translation_thresh
            total_acc = acc_rot & acc_translation & acc_scale

            perclass_results[c]['q_acc'] = acc_rot.mean()
            perclass_results[c]['t_acc'] = acc_translation.mean()
            perclass_results[c]['s_acc'] = acc_scale.mean()
            perclass_results[c]['total_acc'] = total_acc.mean()
            perclass_results[c]['support'] = perclass_pose[c]['support']

            cls_avg_acc['q_acc'] += round(acc_rot.mean() / len(perclass_results.keys()), 4)
            cls_avg_acc['t_acc'] += round(acc_translation.mean() / len(perclass_results.keys()), 4)
            cls_avg_acc['s_acc'] += round(acc_scale.mean() / len(perclass_results.keys()), 4)
            cls_avg_acc['total_acc'] += round(total_acc.mean() / len(perclass_results.keys()), 4)
            avg_acc['q_acc'] += round(acc_rot.mean() * perclass_pose[c]['support'] / total_obj_num, 4)
            avg_acc['t_acc'] += round(acc_translation.mean() * perclass_pose[c]['support'] / total_obj_num, 4)
            avg_acc['s_acc'] += round(acc_scale.mean() * perclass_pose[c]['support'] / total_obj_num, 4)
            avg_acc['total_acc'] += round(total_acc.mean() * perclass_pose[c]['support'] / total_obj_num, 4)

    if ret:
        return perclass_results, cls_avg_acc, avg_acc


def per_class_accuracy(correct_predictions, class_names):
    from collections import defaultdict

    if len(correct_predictions) != len(class_names):
        raise ValueError("Length of correct_predictions and class_names must be the same.")
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    for is_correct, class_name in zip(correct_predictions, class_names):
        total_counts[class_name] += 1
        if is_correct:
            correct_counts[class_name] += 1
    per_class_acc = {class_name: correct_counts[class_name] / total_counts[class_name]
                     for class_name in total_counts}
    macro_avg_accuracy = np.mean(list(per_class_acc.values()))
    per_class_acc['macro_avg'] = macro_avg_accuracy
    total_correct = sum(correct_counts.values())
    total_predictions = sum(total_counts.values())
    micro_avg_accuracy = total_correct / total_predictions
    per_class_acc['micro_avg'] = micro_avg_accuracy
    return per_class_acc, dict(total_counts)


def calc_pose_accuracy_s2c_nocls_full(gt_pose, pred_pose, symmetries, scale_thresh=20, rot_thresh=20, translation_thresh=.2, per_class=False, class_labels=None):
    """`calc_pose_accuracy_s2c_nocls` with its per-class branch, returning avg_acc as the original does."""
    error_scale = np.mean(np.abs((pred_pose[:, -3:] / (gt_pose[:, -3:] + 1e-4)) - 1), axis=1) * 100
    acc_scale = error_scale < scale_thresh
    acc_rot, err_rot_deg = calc_rotation_accuracy(gt_quat=gt_pose[:, 3:-3], pred_quat=pred_pose[:, 3:-3], symmetries=symmetries, threshold=rot_thresh, return_rotation_error=True)
    error_translation = np.linalg.norm(gt_pose[:, :3] - pred_pose[:, :3], ord=2, axis=1)
    acc_translation = error_translation < translation_thresh
    total_acc = acc_rot & acc_translation & acc_scale

    avg_acc = {'t_acc': 0, 'q_acc': 0, 's_acc': 0, 'total_acc': 0, }
    avg_acc['q_acc'] = acc_rot.mean()
    avg_acc['t_acc'] = acc_translation.mean()
    avg_acc['s_acc'] = acc_scale.mean()
    avg_acc['total_acc'] = total_acc.mean()
    avg_acc['rotation_error_deg'] = err_rot_deg.mean()
    if per_class:
        avg_acc['per_class_total'], counts = per_class_accuracy(total_acc, class_labels)
        avg_acc['per_class_t'], _ = per_class_accuracy(acc_translation, class_labels)
        avg_acc['per_class_q'], _ = per_class_accuracy(acc_rot, class_labels)
        avg_acc['per_class_s'], _ = per_class_accuracy(acc_scale, class_labels)
        avg_acc['per_class_support'] = counts
    avg_acc['_mean_scale_error'] = error_scale.mean()  # printed by the original ("Mean scale % error")
    return avg_acc


def eval_pose_s2c_v2_summary(gt_pose, pred_pose, symmetries, class_labels, per_class_results=True):
    """The post-inference part of utils/training_utils.py `eval_pose_s2c_v2` (MAE + accuracy)."""
    from sklearn import metrics

    pose_mae = metrics.mean_absolute_error(gt_pose, pred_pose)
    t_mae = metrics.mean_absolute_error(gt_pose[:, :3], pred_pose[:, :3])
    q_mae = metrics.mean_absolute_error(gt_pose[:, 3:-3], pred_pose[:, 3:-3])
    s_mae = metrics.mean_absolute_error(gt_pose[:, -3:], pred_pose[:, -3:])
    avg_acc = calc_pose_accuracy_s2c_nocls_full(gt_pose, pred_pose, symmetries, per_class=per_class_results, class_labels=class_labels)
    return {"pose_mae": pose_mae, "t_mae": t_mae, "q_mae": q_mae, "s_mae": s_mae, "avg": avg_acc}


# --- utils/general_utils.py: SHAPENET_SYNSET_NAME / shapenet2scannet_name, and the class naming in
# --- SGPredsS2CCompletionPoseVal (custom_datasets_pose.py:6561-6563) ---------------------------

SCANNETV2_NAMES = {
    'unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink',
    'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator',
    'picture', 'cabinet', 'otherfurniture'
}


def LoadShapenetTaxonomy(taxonomy_path):
    with open(taxonomy_path, 'r') as f:
        taxonomy_json = json.load(f)
    taxonomy, inv_taxonomy = {}, {}
    for synset in taxonomy_json:
        tempId = synset['synsetId']
        taxonomy[tempId] = set(synset['name'].split(','))
        for n in taxonomy[tempId]:
            inv_taxonomy[n] = tempId
    return taxonomy, inv_taxonomy


def shapenet2scannet_name(s2c_names):
    valid_s2c_name = None
    for n in s2c_names:
        if n in SCANNETV2_NAMES:
            valid_s2c_name = n
            return valid_s2c_name
    return None


def sg_class_name(SHAPENET_SYNSET_NAME, catid_cad):
    scannet_name = shapenet2scannet_name(SHAPENET_SYNSET_NAME[catid_cad])
    if scannet_name is None:
        scannet_name = sorted(list(SHAPENET_SYNSET_NAME[catid_cad]))[0]
    return scannet_name


# --- datasets/custom_datasets_pose.py: class handling in load_s2c_data_pose_v2 -----------------
# V2 (`S2CCompletionPoseV2`, :4970) and RelPose (`S2CCompletionRelPose`, :5690 in fe50118) differ only in
# the list of names that get replaced by the CAD's valid class.

def legacy_sem_name(inst_sem_name, catid_cad, catid_valid_names, relabel=('unannotated',)):
    sem_name = inst_sem_name
    if sem_name in list(relabel):
        if catid_cad in catid_valid_names.keys():
            sem_name = catid_valid_names[catid_cad]
        else:
            sem_name = None
    return sem_name


# S2CCompletionPoseV2.load_taxonomy / get_cadid_valid_names (custom_datasets_pose.py:937, :1469)
def load_taxonomy(taxonomy_path):
    with open(taxonomy_path, 'r') as f:
        taxonomy_json = json.load(f)
    taxonomy, inv_taxonomy = {}, {}
    for synset in taxonomy_json:
        tempId = synset['synsetId']
        taxonomy[tempId] = synset
        taxonomy[tempId]['name'] = taxonomy[tempId]['name'].split(',')
        for n in taxonomy[tempId]['name']:
            inv_taxonomy[n] = tempId
    return taxonomy, inv_taxonomy


def get_cadid_valid_names(shapenet_id2names, valid_names):
    catid_to_name = {}
    for catid, info in shapenet_id2names.items():
        names = info['name']
        valid_s2c_name = None
        for n in names:
            if n in valid_names:
                valid_s2c_name = n
                break
        if valid_s2c_name is not None:
            catid_to_name[catid] = valid_s2c_name
    return catid_to_name
