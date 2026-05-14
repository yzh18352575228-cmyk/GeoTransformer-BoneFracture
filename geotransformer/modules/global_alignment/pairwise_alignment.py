"""Pairwise rigid alignment via Horn's closed-form SVD method."""

import numpy as np


def pairwise_alignment(pointsS, pointsT, weight, method='horn87'):
    """Compute optimal rotation + translation between weighted point sets.

    Args:
        pointsS: (N, 3) source points
        pointsT: (N, 3) target points
        weight: (N, N) weight matrix (e.g., correspondence confidences)
        method: 'horn87' only for now

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    if method == 'horn87':
        return horn_87(pointsS, pointsT, weight)
    else:
        raise NotImplementedError(f'{method} not implemented')


def horn_87(pointsS, pointsT, weight):
    """Horn's absolute orientation method (1987).

    Closed-form weighted least-squares rigid alignment using quaternion SVD.

    Args:
        pointsS: (N, 3) source points
        pointsT: (N, 3) target points
        weight: (N, N) weight/diagonal matrix

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    pointsS = pointsS.T  # (3, N)
    pointsT = pointsT.T  # (3, N)

    centerS = pointsS.mean(axis=1).reshape(-1, 1)
    centerT = pointsT.mean(axis=1).reshape(-1, 1)
    pointsS_centered = pointsS - centerS
    pointsT_centered = pointsT - centerT

    M = pointsS_centered @ weight @ pointsT_centered.T

    # Build the 4x4 symmetric matrix for quaternion eigenvalue
    N = np.array([
        [
            M[0, 0] + M[1, 1] + M[2, 2],
            M[1, 2] - M[2, 1],
            M[2, 0] - M[0, 2],
            M[0, 1] - M[1, 0],
        ],
        [
            M[1, 2] - M[2, 1],
            M[0, 0] - M[1, 1] - M[2, 2],
            M[0, 1] + M[1, 0],
            M[0, 2] + M[2, 0],
        ],
        [
            M[2, 0] - M[0, 2],
            M[0, 1] + M[1, 0],
            M[1, 1] - M[0, 0] - M[2, 2],
            M[1, 2] + M[2, 1],
        ],
        [
            M[0, 1] - M[1, 0],
            M[2, 0] + M[0, 2],
            M[1, 2] + M[2, 1],
            M[2, 2] - M[0, 0] - M[1, 1],
        ],
    ])

    v, u = np.linalg.eigh(N)
    q = u[:, v.argmax()]  # quaternion corresponding to max eigenvalue

    # Build rotation from quaternion
    R = np.array([
        [
            q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2,
            2 * (q[1] * q[2] - q[0] * q[3]),
            2 * (q[1] * q[3] + q[0] * q[2]),
        ],
        [
            2 * (q[2] * q[1] + q[0] * q[3]),
            q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2,
            2 * (q[2] * q[3] - q[0] * q[1]),
        ],
        [
            2 * (q[3] * q[1] - q[0] * q[2]),
            2 * (q[3] * q[2] + q[0] * q[1]),
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2,
        ],
    ])

    # Recover original points for translation
    pointsS_orig = pointsS_centered + centerS
    pointsT_orig = pointsT_centered + centerT

    w_sum = np.sum(weight, axis=-1).reshape((-1, 1))
    t = (weight @ pointsT_orig.T).T - (w_sum * (R @ pointsS_orig).T).T
    t = np.sum(t, axis=-1) / np.sum(weight)

    return R.astype(np.float32), t.astype(np.float32)
