"""Multi-fragment bone fracture assembly demo with visualization.

Usage:
    # Multi-fragment assembly (all pieces of one bone variant):
    python demo.py --bone_dir=my_data_processed/<bone>/<variant>/ \\
                   --weights=path/to/checkpoint.pth.tar [--save_ply]

    # Single pair (debug):
    python demo.py --src_file=piece_1.npy --ref_file=piece_0.npy \\
                   --weights=path/to/checkpoint.pth.tar

Output:
    - Console: per-pair and assembly metrics
    - PLY files (--save_ply): colored point clouds for MeshLab/CloudCompare
"""

import argparse
import os
import os.path as osp
import glob
import numpy as np
import torch

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.pointcloud import apply_transform, inverse_transform

from config import make_cfg
from model import create_model
from dataset import BoneFracturePairDataset


def make_parser():
    parser = argparse.ArgumentParser()
    # Multi-fragment mode
    parser.add_argument('--bone_dir', default=None, help='dir with piece_*.npy files')
    # Single pair mode
    parser.add_argument('--src_file', default=None, help='source point cloud .npy')
    parser.add_argument('--ref_file', default=None, help='reference point cloud .npy')
    # Shared
    parser.add_argument('--weights', required=True, help='model checkpoint .pth.tar')
    parser.add_argument('--save_ply', action='store_true', help='save PLY files')
    parser.add_argument('--displace', action='store_true', default=True,
                        help='apply random displacement (default: True)')
    return parser


def get_neighbor_limits(cfg):
    train_dataset = BoneFracturePairDataset(
        cfg.data.dataset_root, 'train', point_limit=5000,
        use_augmentation=False,
        rotation_magnitude=cfg.test.rotation_magnitude,
        translation_magnitude=cfg.test.translation_magnitude,
    )
    from geotransformer.utils.data import calibrate_neighbors_stack_mode
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset, registration_collate_fn_stack_mode,
        cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius,
    )
    return neighbor_limits


def run_single_pair(model, ref_points, src_points, neighbor_limits, cfg):
    """Run GeoTransformer on a single pair."""
    data_dict = {
        'ref_points': ref_points.astype(np.float32),
        'src_points': src_points.astype(np.float32),
        'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
        'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
        'transform': np.eye(4, dtype=np.float32),
    }
    collated = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius, neighbor_limits,
    )
    collated = to_cuda(collated)

    with torch.no_grad():
        output_dict = model(collated)

    output_dict = release_cuda(output_dict)
    return output_dict


def save_ply_files(file_prefix, point_clouds, labels, colors):
    """Save point clouds as PLY files for external visualization."""
    try:
        import open3d as o3d

        # Save each piece individually
        for i, (pts, label) in enumerate(zip(point_clouds, labels)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color(colors[i % len(colors)])
            path = '{}_piece{}_{}.ply'.format(file_prefix, i, label)
            o3d.io.write_point_cloud(path, pcd)
            print('  Saved: {}'.format(path))

        # Save merged
        merged = o3d.geometry.PointCloud()
        all_pts = []
        all_cols = []
        for i, pts in enumerate(point_clouds):
            all_pts.append(pts)
            col = np.array(colors[i % len(colors)])
            all_cols.append(np.tile(col, (pts.shape[0], 1)))
        merged.points = o3d.utility.Vector3dVector(np.concatenate(all_pts, axis=0))
        merged.colors = o3d.utility.Vector3dVector(np.concatenate(all_cols, axis=0))
        path = '{}_merged.ply'.format(file_prefix)
        o3d.io.write_point_cloud(path, merged)
        print('  Saved: {}'.format(path))
    except Exception as e:
        print('  PLY save failed (Open3D may not be available): {}'.format(e))


def run_multi_fragment(args, cfg, model, neighbor_limits):
    """Complete multi-fragment assembly pipeline."""
    piece_files = sorted(glob.glob(osp.join(args.bone_dir, 'piece_*.npy')))
    if len(piece_files) < 2:
        print('ERROR: Need at least 2 pieces, found {}'.format(len(piece_files)))
        return

    print('=' * 60)
    print('Bone Fracture Assembly Demo')
    print('=' * 60)
    print('Directory: {}'.format(args.bone_dir))
    print('Pieces found: {}'.format(len(piece_files)))

    # === Load all pieces in assembled position ===
    all_original = []
    for pf in piece_files:
        pts = np.load(pf).astype(np.float32)
        all_original.append(pts)
        print('  {}: {} points, bbox [{:.3f},{:.3f}]×[{:.3f},{:.3f}]×[{:.3f},{:.3f}]'.format(
            osp.basename(pf), pts.shape[0],
            pts[:, 0].min(), pts[:, 0].max(),
            pts[:, 1].min(), pts[:, 1].max(),
            pts[:, 2].min(), pts[:, 2].max(),
        ))

    ref_original = all_original[0]  # piece_0 = reference
    src_originals = all_original[1:]

    # === Apply random displacements ===
    from geotransformer.utils.pointcloud import random_sample_transform
    rng = np.random.RandomState(42)

    gt_displacements = []
    displaced_srcs = []
    for src_orig in src_originals:
        T = random_sample_transform(
            rotation_magnitude=cfg.test.rotation_magnitude,
            translation_magnitude=cfg.test.translation_magnitude,
        )
        displaced = apply_transform(src_orig, T)
        gt_displacements.append(T)
        displaced_srcs.append(displaced)

    # === Run GeoTransformer on each displaced pair ===
    print('\n--- Pairwise GeoTransformer Inference ---')
    pred_transforms = []
    for i, (src_disp, src_orig) in enumerate(zip(displaced_srcs, src_originals)):
        output = run_single_pair(model, ref_original, src_disp, neighbor_limits, cfg)
        T_pred = output['estimated_transform']
        n_corr = output['corr_scores'].shape[0] if 'corr_scores' in output else 0
        pred_transforms.append(T_pred)

        # Compute per-pair error
        T_gt = inverse_transform(gt_displacements[i])
        from geotransformer.utils.registration import compute_registration_error
        rre, rte = compute_registration_error(T_gt, T_pred)
        print('  Pair (piece_0 <- piece_{}): {:4d} corr, RRE={:6.2f}°, RTE={:6.2f}mm'.format(
            i + 1, n_corr, rre, rte * 1000))

    # === Compute assembled positions ===
    print('\n--- Assembly Results ---')

    # Model assembly
    assembled_pred = [ref_original]  # piece_0 stays
    for src_disp, T_pred in zip(displaced_srcs, pred_transforms):
        assembled_pred.append(apply_transform(src_disp, T_pred))

    # GT assembly (undo displacement)
    assembled_gt = [ref_original]
    for src_disp, T_disp in zip(displaced_srcs, gt_displacements):
        T_gt = inverse_transform(T_disp)
        assembled_gt.append(apply_transform(src_disp, T_gt))

    # === Compute metrics ===
    from geotransformer.modules.global_alignment.eval_utils import (
        calc_part_acc, trans_metrics, rot_metrics, chamfer_distance_np,
    )

    pred_rots = [np.eye(3)] + [T[:3, :3] for T in pred_transforms]
    pred_trans = [np.zeros(3)] + [T[:3, 3] for T in pred_transforms]
    gt_rots = [np.eye(3)] + [inverse_transform(T)[:3, :3] for T in gt_displacements]
    gt_trans = [np.zeros(3)] + [inverse_transform(T)[:3, 3] for T in gt_displacements]

    part_acc, cd = calc_part_acc(
        [ref_original] + displaced_srcs, pred_rots, pred_trans, gt_rots, gt_trans
    )
    t_rmse = trans_metrics(pred_trans, gt_trans, 'rmse')
    t_mae = trans_metrics(pred_trans, gt_trans, 'mae')
    r_rmse = rot_metrics(pred_rots, gt_rots, 'rmse')
    r_mae = rot_metrics(pred_rots, gt_rots, 'mae')

    print('  Part Accuracy:   {:.4f}'.format(part_acc))
    print('  Chamfer Dist:    {:.6f}'.format(cd))
    print('  Trans RMSE:      {:.2f} mm'.format(t_rmse * 1000))
    print('  Trans MAE:       {:.2f} mm'.format(t_mae * 1000))
    print('  Rot RMSE:        {:.4f} deg'.format(r_rmse))
    print('  Rot MAE:         {:.4f} deg'.format(r_mae))

    # Check assembly quality vs displacement
    disp_rre = []
    for T_disp in gt_displacements:
        from scipy.spatial.transform import Rotation
        R_disp = T_disp[:3, :3]
        r = Rotation.from_matrix(R_disp)
        disp_rre.append(np.linalg.norm(r.as_rotvec()) * 180 / np.pi)
    print('\n  Displacement magnitudes: {:.1f}° rotation, {:.1f}mm translation'.format(
        np.mean(disp_rre), np.mean([np.linalg.norm(T[:3, 3]) * 1000 for T in gt_displacements])
    ))

    # === Save PLY visualization ===
    if args.save_ply:
        print('\n--- Saving PLY Files ---')
        colors = [
            [1.0, 0.8, 0.0],    # gold (ref/piece_0)
            [0.0, 0.6, 1.0],    # blue
            [1.0, 0.2, 0.2],    # red
            [0.2, 1.0, 0.2],    # green
            [1.0, 0.5, 0.0],    # orange
            [0.5, 0.0, 1.0],    # purple
        ]

        base = osp.splitext(piece_files[0])[0].replace('piece_0', '')
        labels = ['ref'] + ['piece_{}'.format(i + 1) for i in range(len(src_originals))]

        # 1. Initial (displaced) state
        print('\n[Initial displaced state]')
        initial_pts = [ref_original] + displaced_srcs
        save_ply_files(base + 'initial', initial_pts, labels, colors)

        # 2. Model assembly
        print('\n[Model assembly]')
        save_ply_files(base + 'assembled', assembled_pred, labels, colors)

        # 3. Ground truth assembly
        print('\n[Ground truth assembly]')
        save_ply_files(base + 'gt', assembled_gt, labels, colors)

    print('\n' + '=' * 60)
    print('Done. To visualize, open the PLY files in MeshLab or CloudCompare.')
    print('Compare: *_initial_merged.ply vs *_assembled_merged.ply vs *_gt_merged.ply')
    print('=' * 60)


def run_single_pair_demo(args, cfg, model, neighbor_limits):
    """Single pair demo (for debugging)."""
    ref_points = np.load(args.ref_file).astype(np.float32)
    src_points = np.load(args.src_file).astype(np.float32)

    # Apply displacement
    from geotransformer.utils.pointcloud import random_sample_transform
    T_disp = random_sample_transform(
        rotation_magnitude=cfg.test.rotation_magnitude,
        translation_magnitude=cfg.test.translation_magnitude,
    )
    src_displaced = apply_transform(src_points, T_disp)

    print('Ref:  {} points'.format(ref_points.shape[0]))
    print('Src:  {} points'.format(src_points.shape[0]))
    print('Displacement: {:.1f}° rotation, {:.1f}mm translation'.format(
        np.linalg.norm(
            __import__('scipy').spatial.transform.Rotation.from_matrix(T_disp[:3, :3]).as_rotvec()
        ) * 180 / np.pi,
        np.linalg.norm(T_disp[:3, 3]) * 1000,
    ))

    output = run_single_pair(model, ref_points, src_displaced, neighbor_limits, cfg)
    T_pred = output['estimated_transform']
    T_gt = inverse_transform(T_disp)

    from geotransformer.utils.registration import compute_registration_error
    rre, rte = compute_registration_error(T_gt, T_pred)
    print('RRE: {:.4f} deg'.format(rre))
    print('RTE: {:.4f} m ({:.2f} mm)'.format(rte, rte * 1000))
    print('Num correspondences: {}'.format(
        output['corr_scores'].shape[0] if 'corr_scores' in output else 'N/A'
    ))

    if args.save_ply:
        src_assembled = apply_transform(src_displaced, T_pred)
        src_gt = apply_transform(src_displaced, T_gt)
        colors = [[1.0, 0.8, 0.0], [0.0, 0.6, 1.0]]
        save_ply_files('pair_initial', [ref_points, src_displaced],
                       ['ref', 'src_displaced'], colors)
        save_ply_files('pair_assembled', [ref_points, src_assembled],
                       ['ref', 'src_assembled'], colors)
        save_ply_files('pair_gt', [ref_points, src_gt],
                       ['ref', 'src_gt'], colors)


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # Load model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict['model'])
    model.eval()
    print('Model loaded from {} (epoch {})'.format(
        args.weights, state_dict.get('epoch', 'unknown')))

    neighbor_limits = get_neighbor_limits(cfg)
    print('Neighbor limits: {}'.format(neighbor_limits.tolist()))

    if args.bone_dir is not None:
        run_multi_fragment(args, cfg, model, neighbor_limits)
    elif args.src_file and args.ref_file:
        run_single_pair_demo(args, cfg, model, neighbor_limits)
    else:
        print('ERROR: Provide --bone_dir or --src_file + --ref_file')


if __name__ == '__main__':
    main()
