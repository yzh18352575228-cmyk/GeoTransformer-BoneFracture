import numpy as np

from .pose_graph_utils import minimum_spanning_tree


def spanning_tree_alignment(v_num, edges, transformations, uncertainty):
    """Align fragments using minimum spanning tree.

    Args:
        v_num: number of vertices (fragments)
        edges: [m, 2], directed edges (i, j)
        transformations: [m, 4, 4], T_ij = inv(T_i) @ T_j
        uncertainty: [m], edge uncertainty (lower = more reliable)

    Returns:
        global_transformation: [v_num, 4, 4]
        success: 1
    """
    mst_order, mst_pred = minimum_spanning_tree(v_num, edges, uncertainty)
    global_transformation = np.zeros((v_num, 4, 4))
    global_transformation[0, :, :] = np.eye(4)

    # Build hash map of transformations between all pairs
    hash_map = np.zeros((v_num, v_num, 4, 4))
    for i in range(edges.shape[0]):
        src = int(edges[i, 0])
        tgt = int(edges[i, 1])
        hash_map[src, tgt, :, :] = transformations[i, :, :]
        hash_map[tgt, src, :, :] = np.linalg.inv(transformations[i, :, :])

    for i in range(1, v_num):
        y = mst_order[i]
        x = mst_pred[y]
        global_transformation[y, :, :] = (
            global_transformation[x, :, :] @ hash_map[x, y, :, :]
        )

    return global_transformation, 1
