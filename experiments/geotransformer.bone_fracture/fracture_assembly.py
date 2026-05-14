"""Multi-fragment bone fracture assembly.

Wraps GeoTransformer pairwise predictions into a complete assembly pipeline,
outputs Jigsaw-compatible metrics: Part Accuracy, Chamfer Distance,
Translation RMSE/MAE, Rotation RMSE/MAE.

Pipeline:
  1. For each small piece: GeoTransformer predicts T_i (piece_i -> piece_0)
  2. Build pose graph from all pairwise results
  3. Global alignment (spanning tree / Shonan averaging)
  4. Compute Jigsaw-style assembly metrics
"""

import numpy as np
from scipy.spatial.transform import Rotation

from geotransformer.modules.global_alignment import (
    global_alignment,
    calc_part_acc,
    trans_metrics,
    rot_metrics,
)
from geotransformer.utils.pointcloud import (
    apply_transform,
    inverse_transform,
    get_rotation_translation_from_transform,
)


def assemble_fragments_pairwise(model, ref_points, src_points_list, neighbor_limits, cfg, device='cuda'):
    """Run pairwise GeoTransformer for each source fragment against the reference.

    Args:
        model: trained GeoTransformer model (on target device)
        ref_points: (N, 3) reference fragment points
        src_points_list: list of (M_i, 3) source fragment points
        neighbor_limits: precomputed neighbor limits
        cfg: EasyDict config
        device: 'cuda' or 'cpu'

    Returns:
        pairwise_results: list of dicts with keys:
            - estimated_transform: (4, 4) predicted transform
            - corr_scores: correspondence scores
            - ref_corr_points, src_corr_points
    """
    import torch
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda

    model.eval()
    pairwise_results = []

    for i, src_points in enumerate(src_points_list):
        data_dict = {
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'transform': np.eye(4, dtype=np.float32),  # dummy; needed by model forward
        }

        collated = registration_collate_fn_stack_mode(
            [data_dict],
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            neighbor_limits,
        )

        collated = to_cuda(collated)

        with torch.no_grad():
            output_dict = model(collated)

        output_dict = release_cuda(output_dict)
        collated = release_cuda(collated)

        pairwise_results.append({
            'estimated_transform': output_dict['estimated_transform'],
            'ref_corr_points': output_dict.get('ref_corr_points', None),
            'src_corr_points': output_dict.get('src_corr_points', None),
            'corr_scores': output_dict.get('corr_scores', None),
            'num_correspondences': output_dict['corr_scores'].shape[0] if 'corr_scores' in output_dict else 0,
        })

    return pairwise_results


def build_pose_graph_from_pairwise(pairwise_results, n_fragments):
    """Build pose graph from pairwise GeoTransformer results.

    fragment_0 is the reference (identity pose). Each pairwise result
    T_i maps piece_i -> piece_0, so we have edges (0, i) with T_0i = T_i.

    Args:
        pairwise_results: list of dicts, estimated_transform maps src->ref
        n_fragments: total number of fragments (including reference)

    Returns:
        edges: (m, 2) array
        transformations: (m, 4, 4) array
        uncertainty: (m,) array
        per_fragment_poses: (n_fragments, 4, 4) naive per-fragment transforms
    """
    edges = []
    transformations = []
    uncertainty = []

    # Start with reference identity
    per_fragment_poses = [np.eye(4)]

    for i, result in enumerate(pairwise_results):
        src_idx = i + 1  # fragment index
        ref_idx = 0      # always piece_0 as reference

        T_src_to_ref = result['estimated_transform']  # maps src -> ref

        # Edge: (ref_idx, src_idx), transform = inv(T_0) @ T_src = I @ T_src = T_src
        # Meaning: to map a point from src_j to global, T_j; then inv(T_0) to ref_i
        # T_0j = inv(T_0) @ T_j = I @ T_src_to_ref = T_src_to_ref
        edges.append([ref_idx, src_idx])
        transformations.append(T_src_to_ref)
        uncertainty.append(1.0 / max(result['num_correspondences'], 1))

        per_fragment_poses.append(T_src_to_ref)

    edges = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)
    transformations = np.array(transformations) if transformations else np.zeros((0, 4, 4))
    uncertainty = np.array(uncertainty) if uncertainty else np.zeros(0)
    per_fragment_poses = np.stack(per_fragment_poses, axis=0)

    return edges, transformations, uncertainty, per_fragment_poses


def run_global_alignment(edges, transformations, uncertainty, n_fragments):
    """Run global pose graph optimization.

    Args:
        edges: (m, 2) array
        transformations: (m, 4, 4) array
        uncertainty: (m,) array
        n_fragments: total number of fragments

    Returns:
        global_poses: (n_fragments, 4, 4) optimized per-fragment transforms
    """
    if n_fragments <= 1:
        return np.eye(4).reshape(1, 4, 4)

    global_poses = global_alignment(
        n_fragments, edges, transformations, uncertainty, method='auto'
    )
    return global_poses


def compute_assembly_metrics(points_list, pred_rots, pred_trans, gt_rots, gt_trans):
    """Compute Jigsaw-compatible assembly metrics.

    Args:
        points_list: list of (N_i, 3) arrays, original piece points
        pred_rots: list of (3, 3) predicted rotation matrices
        pred_trans: list of (3,) predicted translation vectors
        gt_rots: list of (3, 3) ground truth rotation matrices
        gt_trans: list of (3,) ground truth translation vectors

    Returns:
        metrics: dict with keys:
            part_acc, chamfer_distance, trans_rmse, trans_mae, rot_rmse, rot_mae
    """
    part_acc, cd = calc_part_acc(points_list, pred_rots, pred_trans, gt_rots, gt_trans)
    t_rmse = trans_metrics(pred_trans, gt_trans, 'rmse')
    t_mae = trans_metrics(pred_trans, gt_trans, 'mae')
    r_rmse = rot_metrics(pred_rots, gt_rots, 'rmse')
    r_mae = rot_metrics(pred_rots, gt_rots, 'mae')

    return {
        'part_acc': part_acc,
        'chamfer_distance': cd,
        'trans_rmse': t_rmse,
        'trans_mae': t_mae,
        'rot_rmse': r_rmse,
        'rot_mae': r_mae,
    }


def full_assembly_pipeline(model, ref_points, src_points_list, gt_displacements, cfg,
                           neighbor_limits, device='cuda'):
    """Complete multi-fragment assembly pipeline.

    Args:
        model: GeoTransformer model
        ref_points: (N, 3) reference fragment (piece_0)
        src_points_list: list of (M_i, 3) source fragments
        gt_displacements: list of (4, 4) ground truth displacement transforms
                           applied to each source during training
        cfg: EasyDict config
        neighbor_limits: precomputed neighbor limits
        device: 'cuda' or 'cpu'

    Returns:
        assembly_result: dict with:
            - global_poses: (P, 4, 4) optimized per-fragment transforms
            - pairwise_poses: (P, 4, 4) naive pairwise transforms
            - pairwise_results: raw GeoTransformer outputs
            - metrics: Jigsaw-compatible metrics dict
            - points_list: all fragment point clouds
            - pred_rots, pred_trans: per-fragment predictions
            - gt_rots, gt_trans: per-fragment ground truth
    """
    n_fragments = len(src_points_list) + 1  # +1 for reference

    # Step 1: Pairwise GeoTransformer
    pairwise_results = assemble_fragments_pairwise(
        model, ref_points, src_points_list, neighbor_limits, cfg, device
    )

    # Step 2: Build pose graph
    edges, transformations, uncertainty, pairwise_poses = build_pose_graph_from_pairwise(
        pairwise_results, n_fragments
    )

    # Step 3: Global alignment
    global_poses = run_global_alignment(edges, transformations, uncertainty, n_fragments)

    # Step 4: Prepare ground truth (reference = identity, src_i = inverse displacement)
    gt_rots = [np.eye(3)]
    gt_trans = [np.zeros(3)]
    for T_disp in gt_displacements:
        # GT transform to bring displaced src back to assembled position
        T_gt = inverse_transform(T_disp)
        R, t = get_rotation_translation_from_transform(T_gt)
        gt_rots.append(R)
        gt_trans.append(t)

    # Step 5: Extract predicted rotations and translations
    pred_rots = [R for R in global_poses[:, :3, :3]]
    pred_trans = [t for t in global_poses[:, :3, 3]]

    # Step 6: All point clouds (reference in original position)
    points_list = [ref_points] + src_points_list

    # Step 7: Compute metrics
    metrics = compute_assembly_metrics(points_list, pred_rots, pred_trans, gt_rots, gt_trans)

    return {
        'global_poses': global_poses,
        'pairwise_poses': pairwise_poses,
        'pairwise_results': pairwise_results,
        'metrics': metrics,
        'points_list': points_list,
        'pred_rots': pred_rots,
        'pred_trans': pred_trans,
        'gt_rots': gt_rots,
        'gt_trans': gt_trans,
        'n_fragments': n_fragments,
    }
