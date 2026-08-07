"""3D Gaussian primitive utilities (paper Sec. 3.2).

Each Gaussian is parameterised by a centroid ``mu^g``, an opacity ``alpha``,
spherical harmonics ``SH`` and a covariance ``Sigma = R S S^T R^T`` where ``R``
is a rotation matrix and ``S`` a diagonal scale matrix.
"""

import numpy as np
from scipy.spatial.transform import Rotation

SH_DIM = 48  # 3 f_dc + 45 f_rest


def preprocess_3dgs_attributes(raw_features, include_sh=False):
    """Apply the 3DGS activations to the raw PLY parameters.

    ``scale`` is stored logarithmically and ``opacity`` before a sigmoid, so
    both are activated here. The rotation quaternion is passed through
    untouched.

    Note:
        The scale is deliberately left in metres -- no per-scene min/max
        normalisation is applied. Eq. (4)/(5) rely on ``||S'||_2`` being
        comparable in absolute terms, and the pruning threshold of Eq. (5) is a
        per-scene percentile, so re-normalising here would be redundant.

    Args:
        raw_features: ``(M, 8)`` (or ``(M, 56)`` with SH) array from
            :func:`g2p_prep.io.load_3dgs_ply`.
        include_sh: also activate and return the spherical-harmonics block.

    Returns:
        ``(M, 8)`` array laid out as ``scale(3) | opacity(1) | quaternion(4)``,
        with ``SH(48)`` appended when ``include_sh``.
    """
    idx = 0
    scale = raw_features[:, idx:idx + 3]; idx += 3
    opacity = raw_features[:, idx:idx + 1]; idx += 1
    rotation = raw_features[:, idx:idx + 4]; idx += 4

    sh_dc = sh_rest = None
    if include_sh and raw_features.shape[1] >= idx + SH_DIM:
        sh_dc = raw_features[:, idx:idx + 3]; idx += 3
        sh_rest = raw_features[:, idx:idx + (SH_DIM - 3)]

    # scale: undo the log parameterisation, guard against non-finite values.
    scale = np.exp(scale)
    scale = np.nan_to_num(scale, nan=1e-6, posinf=1e-6, neginf=1e-6)
    scale = np.maximum(scale, 1e-6)

    # opacity: sigmoid activation into [0, 1].
    opacity = 1.0 / (1.0 + np.exp(-opacity))
    opacity = np.nan_to_num(opacity, nan=0.0, posinf=1.0, neginf=0.0)
    opacity = np.clip(opacity, 0.0, 1.0)

    features = [scale, opacity, rotation]
    if include_sh and sh_dc is not None and sh_rest is not None:
        features.append(np.clip((sh_dc + 1.0) * 0.5, 0.0, 1.0))
        features.append(np.nan_to_num(sh_rest, nan=0.0))

    return np.hstack(features)


def build_covariance(scale, rotation_quat):
    """Build ``Sigma = R S S^T R^T`` for a single Gaussian.

    Warning:
        3DGS PLY files store the rotation as ``(w, x, y, z)`` while
        ``scipy.spatial.transform.Rotation.from_quat`` expects ``(x, y, z, w)``.
        This function passes the quaternion through unchanged, which means the
        covariance is built from a permuted quaternion. All reported numbers
        were produced with this behaviour, so it is preserved verbatim.
    """
    S = np.diag(scale)
    R = Rotation.from_quat(rotation_quat).as_matrix()
    return R @ S @ S.T @ R.T


def quaternion_to_direction(quaternions):
    """Rotate the reference vector ``[0, 0, 1]`` by each ``(w, x, y, z)`` quaternion.

    Args:
        quaternions: ``(N, 4)`` array.

    Returns:
        ``(N, 3)`` unit direction vectors.
    """
    norm = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / np.where(norm == 0, 1, norm)

    w, x, y, z = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )

    directions = np.zeros((len(quaternions), 3))
    directions[:, 0] = 2 * (x * z + w * y)
    directions[:, 1] = 2 * (y * z - w * x)
    directions[:, 2] = 1 - 2 * (x * x + y * y)

    dir_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / np.where(dir_norm == 0, 1, dir_norm)


def weighted_direction(neighbor_quaternions, weights):
    """Weighted mean of neighbour rotation directions, re-normalised to unit length."""
    directions = quaternion_to_direction(neighbor_quaternions)
    direction = np.sum(directions * weights[:, np.newaxis], axis=0)

    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        return direction / norm
    # Fully cancelling neighbours: fall back to the reference direction.
    return np.array([0, 0, 1], dtype=np.float32)
