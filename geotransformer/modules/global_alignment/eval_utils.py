"""Evaluation utilities adapted from Jigsaw.

Uses NumPy/scipy (no PyTorch/CUDA dependency) for computing standard
multi-fragment assembly metrics: Part Accuracy, Chamfer Distance,
Translation RMSE/MAE, Rotation RMSE/MAE.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def chamfer_distance_np(pc1, pc2):
    """Compute Chamfer distance between two point clouds.

    Args:
        pc1: (N, 3) numpy array
        pc2: (M, 3) numpy array

    Returns:
        dist1: (N,) distance from each point in pc1 to nearest in pc2
        dist2: (M,) distance from each point in pc2 to nearest in pc1
    """
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    dist1, _ = tree2.query(pc1, k=1)
    dist2, _ = tree1.query(pc2, k=1)
    return dist1, dist2


def apply_transform_np(points, rot, trans):
    """Apply rotation and translation to points.

    Args:
        points: (N, 3) numpy array
        rot: (3, 3) rotation matrix
        trans: (3,) translation vector

    Returns:
        (N, 3) transformed points
    """
    return (rot @ points.T).T + trans


def calc_part_acc(points_list, pred_rots, pred_trans, gt_rots, gt_trans, threshold=0.01):
    """Compute Part Accuracy (Jigsaw metric).

    A part is considered correctly assembled if Chamfer distance
    to ground truth is below threshold.

    Args:
        points_list: list of (N_i, 3) arrays, one per fragment
        pred_rots: list of (3, 3) predicted rotation matrices
        pred_trans: list of (3,) predicted translation vectors
        gt_rots: list of (3, 3) ground truth rotation matrices
        gt_trans: list of (3,) ground truth translation vectors
        threshold: Chamfer distance threshold (default 0.01)

    Returns:
        part_acc: float, fraction of correctly assembled parts
        cd_mean: float, mean Chamfer distance per part
    """
    n_parts = len(points_list)
    acc_count = 0
    cd_values = []

    for i in range(n_parts):
        pts = points_list[i]
        pred_pts = apply_transform_np(pts, pred_rots[i], pred_trans[i])
        gt_pts = apply_transform_np(pts, gt_rots[i], gt_trans[i])

        dist1, dist2 = chamfer_distance_np(pred_pts, gt_pts)
        cd = np.mean(dist1) + np.mean(dist2)
        cd_values.append(cd)
        if cd < threshold:
            acc_count += 1

    part_acc = acc_count / n_parts if n_parts > 0 else 0.0
    cd_mean = np.mean(cd_values) if cd_values else 0.0
    return part_acc, cd_mean


def trans_metrics(pred_trans, gt_trans, metric='rmse'):
    """Translation error metrics (Jigsaw-compatible: per-part then average).

    Computes per-fragment L2 error, then averages across fragments.
    Fragment 0 (reference) is excluded since its error is always zero.

    Args:
        pred_trans: list of (3,) arrays
        gt_trans: list of (3,) arrays
        metric: 'mse', 'rmse', or 'mae'

    Returns:
        float, per-fragment average metric value
    """
    errors = []
    for i in range(len(pred_trans)):
        e = np.linalg.norm(np.array(pred_trans[i]) - np.array(gt_trans[i]))
        errors.append(e)
    errors = np.array(errors)
    # Exclude piece_0 (index 0) — reference always has zero error
    errors = errors[1:] if len(errors) > 1 else errors

    if len(errors) == 0:
        return 0.0
    if metric == 'mse':
        return float(np.mean(errors ** 2))
    elif metric == 'rmse':
        return float(np.sqrt(np.mean(errors ** 2)))
    elif metric == 'mae':
        return float(np.mean(errors))
    else:
        raise ValueError(f'Unknown metric: {metric}')


def rot_metrics(pred_rots, gt_rots, metric='rmse'):
    """Rotation error metrics in degrees (Jigsaw-compatible: per-part geodesic).

    Uses geodesic angle: arccos((trace(R_pred^T @ R_gt) - 1) / 2).
    Fragment 0 (reference) is excluded.

    Args:
        pred_rots: list of (3, 3) arrays
        gt_rots: list of (3, 3) arrays
        metric: 'mse', 'rmse', or 'mae'

    Returns:
        float, per-fragment average metric value in degrees
    """
    errors = []
    for pr, gr in zip(pred_rots, gt_rots):
        # Geodesic distance on SO(3)
        R_diff = pr.T @ gr
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        angle_rad = np.arccos((trace - 1.0) / 2.0)
        angle_deg = np.degrees(angle_rad)
        errors.append(angle_deg)
    errors = np.array(errors)
    # Exclude piece_0 (index 0) — reference always has zero error
    errors = errors[1:] if len(errors) > 1 else errors

    if len(errors) == 0:
        return 0.0
    if metric == 'mse':
        return float(np.mean(errors ** 2))
    elif metric == 'rmse':
        return float(np.sqrt(np.mean(errors ** 2)))
    elif metric == 'mae':
        return float(np.mean(errors))
    else:
        raise ValueError(f'Unknown metric: {metric}')
