"""Shonan averaging for rotation on SO(3). Requires GTSAM library.

Only loaded when gtsam is available; otherwise spanning tree is used.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

import gtsam


def estimate_poses_given_rot(factors, rotations, uncertainty, d=3):
    """Estimate translations given rotations via linear least-squares."""
    I_d = np.eye(d)

    def R_mat(j):
        return rotations.atRot3(j) if d == 3 else rotations.atRot2(j)

    def pose(rot, t):
        return gtsam.Pose3(rot, t) if d == 3 else gtsam.Pose2(rot, t)

    graph = gtsam.GaussianFactorGraph()
    model = gtsam.noiseModel.Unit.Create(d)

    # Anchor t_0
    graph.add(0, I_d, np.zeros((d,)), model)

    # t_j - t_i = R_i * t_ij for all edges
    for idx in range(len(factors)):
        factor = factors[idx]
        keys = factor.keys()
        i, j, Tij = keys[0], keys[1], factor.measured()
        if i == j:
            continue
        model = gtsam.noiseModel.Diagonal.Variances(
            uncertainty[idx] * 1e-2 * np.ones(d)
        )
        measured = R_mat(i).rotate(Tij.translation())
        graph.add(j, I_d, i, -I_d, measured, model)

    translations = graph.optimize()
    result = gtsam.Values()
    for j in range(rotations.size()):
        tj = translations.at(j)
        result.insert(j, pose(R_mat(j), tj))
    return result


def shonan_averaging(v_num, edges, transformations, uncertainty, verbose=False):
    """Shonan averaging for global rotation+translation estimation.

    Args:
        v_num: number of vertices
        edges: [m, 2], directed edges
        transformations: [m, 4, 4], relative transforms
        uncertainty: [m], per-edge uncertainty

    Returns:
        global_pose_results: [v_num, 4, 4]
        success: 1 on success, 0 on failure
    """
    edge_num = edges.shape[0]
    factors = []
    new_uncertainty = []

    for i in range(edge_num):
        if edges[i, 1] == edges[i, 0]:
            continue
        odomModel = gtsam.noiseModel.Diagonal.Variances(
            uncertainty[i] * np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
        )
        factor = gtsam.BetweenFactorPose3(
            int(edges[i, 0]),
            int(edges[i, 1]),
            gtsam.Pose3(transformations[i, :, :]),
            odomModel,
        )
        factors.append(factor)
        new_uncertainty.append(uncertainty[i])

    shonan = gtsam.ShonanAveraging3(gtsam.BetweenFactorPose3s(factors))
    initial = shonan.initializeRandomly()

    try:
        rotations, _ = shonan.run(initial, 3, 10)
        poses = estimate_poses_given_rot(
            factors, rotations, np.array(new_uncertainty), d=3
        )
    except Exception:
        if verbose:
            print("Shonan did not converge")
        return np.stack([np.eye(4)] * v_num), 0

    global_pose_results = []
    for i in range(poses.size()):
        global_pose_results.append(poses.atPose3(i).matrix())
    return np.stack(global_pose_results), 1
