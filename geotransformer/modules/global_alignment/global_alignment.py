"""Global alignment of multiple fragments via pose graph optimization.

Uses spanning tree alignment (always works, no extra dependency).
Optionally tries GTSAM Shonan averaging if available.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .pose_graph_utils import connect_graph
from .spanning_tree_alignment import spanning_tree_alignment

try:
    import gtsam
    from .shonan_averaging import shonan_averaging as _shonan_impl
    _has_gtsam = True
except ImportError:
    _has_gtsam = False


def global_alignment(v_num, edges, transformations, uncertainty, method='auto', verbose=False):
    """Solve global pose graph for multi-fragment assembly.

    Args:
        v_num: number of fragments
        edges: [m, 2], set of directed edges (i, j) where T_ij maps
               points from fragment j to fragment i
        transformations: [m, 4, 4], relative transforms T_ij = inv(T_i) @ T_j
        uncertainty: [m], edge uncertainty (lower = more reliable)
        method: 'auto' (try GTSAM, fallback), 'spanning_tree', or 'gtsam'
        verbose: print debug info

    Returns:
        global_poses: [v_num, 4, 4], per-fragment absolute rigid transforms
    """
    if len(edges) == 0:
        return np.stack([np.eye(4)] * v_num, axis=0) if v_num > 0 else np.zeros((0, 4, 4))

    # Make graph connected via auxiliary hub vertex
    auxiliary_edges = connect_graph(v_num, edges)
    all_edges = np.concatenate([np.array(edges), auxiliary_edges], axis=0).astype(np.int32)

    # Add random auxiliary transformations
    aux_transforms = []
    for _ in range(auxiliary_edges.shape[0]):
        T = np.eye(4)
        T[:3, :3] = R.random().as_matrix()
        T[:3, 3] = np.random.rand(3)
        aux_transforms.append(T)
    aux_transforms = np.stack(aux_transforms) if aux_transforms else np.zeros((0, 4, 4))

    all_transforms = np.concatenate([np.array(transformations), aux_transforms], axis=0)
    aux_uncertainty = np.ones(auxiliary_edges.shape[0])
    all_uncertainty = np.concatenate([np.array(uncertainty), aux_uncertainty])

    total_vertices = v_num + 1  # +1 for hub vertex

    use_gtsam = method == 'gtsam' or (method == 'auto' and _has_gtsam)

    if use_gtsam:
        try:
            global_pose_results, success = _shonan_impl(
                total_vertices, all_edges, all_transforms, all_uncertainty, verbose=verbose
            )
            if success:
                # Normalize to first fragment
                for i in range(v_num):
                    global_pose_results[v_num - i - 1, :, :] = (
                        np.linalg.inv(global_pose_results[0, :, :])
                        @ global_pose_results[v_num - i - 1, :, :]
                    )
                return global_pose_results[:v_num, :, :]
        except Exception:
            if verbose:
                print('GTSAM Shonan failed, falling back to spanning tree')

    # Spanning tree fallback
    global_pose_results, _ = spanning_tree_alignment(
        total_vertices, all_edges, all_transforms, all_uncertainty
    )
    for i in range(v_num):
        global_pose_results[v_num - i - 1, :, :] = (
            np.linalg.inv(global_pose_results[0, :, :])
            @ global_pose_results[v_num - i - 1, :, :]
        )
    return global_pose_results[:v_num, :, :]
