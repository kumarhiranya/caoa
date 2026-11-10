"""Rigid/similarity transforms.

Quaternions are (w, x, y, z), as stored by Scan2CAD. Rotations are passed
around as 3x3 matrices; quaternions only appear at the annotation boundary.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def matrix_to_quat(R) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def nearest_rotation(M) -> np.ndarray:
    """Closest proper rotation to a 3x3 matrix in Frobenius norm (SVD projection).

    Same result as numpy-quaternion's Bar-Itzhack conversion; used on the OCAM's unconstrained 9-value rotation outputs.
    """
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def yaw_matrix(angle_rad: float) -> np.ndarray:
    """Rotation about +y, the up axis of the Scan2CAD / ShapeNet frame."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def compose(t=None, R=None, s=None) -> np.ndarray:
    """4x4 matrix T @ R @ S."""
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if s is not None:
        M[:3, :3] = M[:3, :3] @ np.diag(np.broadcast_to(np.asarray(s, dtype=np.float64), 3))
    if t is not None:
        M[:3, 3] = t
    return M


def decompose(M):
    """Inverse of `compose` for matrices without shear: returns (t, R, s)."""
    A = np.asarray(M, dtype=np.float64)[:3, :3]
    s = np.linalg.norm(A, axis=0)
    return M[:3, 3].copy(), A / s, s


def transform_points(points, M) -> np.ndarray:
    return points @ M[:3, :3].T + M[:3, 3]
