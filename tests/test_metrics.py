import numpy as np
import pytest
import quaternion
from scipy.spatial.transform import Rotation

from caoa.eval.metrics import MetricConfig, alignment_errors, rotation_error_deg
from caoa.geometry import yaw_matrix
from tests import legacy_ref

SYMS = ["__SYM_NONE", "__SYM_ROTATE_UP_2", "__SYM_ROTATE_UP_4", "__SYM_ROTATE_UP_INF"]


def random_poses(rng, n, noise_deg):
    t_gt = rng.uniform(-0.5, 0.5, (n, 3))
    R_gt = Rotation.random(n, random_state=rng).as_matrix()
    s_gt = rng.uniform(0.2, 1.0, (n, 3))
    t_pred = t_gt + rng.normal(0, 0.12, (n, 3))
    R_noise = Rotation.from_rotvec(np.radians(noise_deg) * rng.normal(size=(n, 3))).as_matrix()
    # Unconstrained 9-value outputs, like the network's: perturb off SO(3).
    R_pred = R_noise @ R_gt + rng.normal(0, 0.05, (n, 3, 3))
    s_pred = s_gt * rng.uniform(0.7, 1.3, (n, 3))
    syms = rng.choice(SYMS, n).tolist()
    return t_pred, R_pred, s_pred, t_gt, R_gt, s_gt, syms


@pytest.mark.parametrize("noise_deg", [5, 30, 120])
def test_legacy_mode_matches_legacy_code(noise_deg):
    rng = np.random.default_rng(noise_deg)
    t_pred, R_pred, s_pred, t_gt, R_gt, s_gt, syms = random_poses(rng, 400, noise_deg)

    ours = alignment_errors(t_pred, R_pred, s_pred, t_gt, R_gt, s_gt, syms, MetricConfig.legacy())

    # Legacy pose vectors: t (3) | flattened rotation matrix (9) | s (3)
    gt = np.concatenate([t_gt, R_gt.reshape(-1, 9), s_gt], axis=1)
    pred = np.concatenate([t_pred, R_pred.reshape(-1, 9), s_pred], axis=1)
    ref = legacy_ref.calc_pose_accuracy_s2c_nocls(gt, pred, syms)

    np.testing.assert_allclose(ours["err_r"], ref["err_r"], atol=1e-6)
    np.testing.assert_allclose(ours["err_s"], ref["err_s"], atol=1e-9)
    np.testing.assert_array_equal(ours["ok"], ref["total"])


def test_scan2cad_rotation_is_geodesic_with_symmetry():
    R_gt = Rotation.random(random_state=0).as_matrix()
    # 90 degrees of yaw in the object frame: wrong for 2-fold symmetry, exact for 4-fold.
    R_pred = R_gt @ yaw_matrix(np.pi / 2)
    assert rotation_error_deg(R_pred, R_gt, "__SYM_NONE") == pytest.approx(90.0)
    assert rotation_error_deg(R_pred, R_gt, "__SYM_ROTATE_UP_2") == pytest.approx(90.0)
    assert rotation_error_deg(R_pred, R_gt, "__SYM_ROTATE_UP_4") == pytest.approx(0.0, abs=1e-5)


def test_legacy_rotation_is_lenient_for_symmetric_objects():
    """Documents the legacy bug: a 90-degree yaw error passes for a 2-fold symmetric object."""
    R_gt = np.eye(3)
    R_pred = yaw_matrix(np.pi / 2)
    assert rotation_error_deg(R_pred, R_gt, "__SYM_ROTATE_UP_2", "legacy") < 20.0
    assert rotation_error_deg(R_pred, R_gt, "__SYM_ROTATE_UP_2", "scan2cad") == pytest.approx(90.0)


def test_nearest_rotation_matches_numpy_quaternion():
    rng = np.random.default_rng(1)
    from caoa.geometry import matrix_to_quat, nearest_rotation

    for _ in range(50):
        M = Rotation.random(random_state=rng).as_matrix() + rng.normal(0, 0.1, (3, 3))
        q_ref = quaternion.as_float_array(quaternion.from_rotation_matrix(M))
        q_ours = matrix_to_quat(nearest_rotation(M))
        assert abs(abs(np.dot(q_ref, q_ours)) - 1.0) < 1e-9
