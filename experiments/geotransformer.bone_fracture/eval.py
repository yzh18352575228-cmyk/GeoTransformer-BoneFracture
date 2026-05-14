"""Evaluate bone fracture assembly with Jigsaw-compatible metrics.

FIXED: src fragments are now ACTUALLY displaced before inference.
The model must recover the inverse transform to bring them back to assembly.

Metrics: Part Accuracy, Chamfer Distance, Translation RMSE/MAE, Rotation RMSE/MAE.

Usage:
    python eval.py --test_epoch=120 --subset=test --verbose
"""

import argparse
import os.path as osp
import pickle
import sys
import json
import time

import numpy as np
import torch

from geotransformer.engine import Logger
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.pointcloud import apply_transform, inverse_transform
from geotransformer.modules.global_alignment.eval_utils import (
    calc_part_acc, trans_metrics, rot_metrics, chamfer_distance_np,
)

from config import make_cfg
from model import create_model
from dataset import BoneFracturePairDataset


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default=None, help='path to model checkpoint')
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch from snapshot dir')
    parser.add_argument('--subset', default='test', choices=['val', 'test'],
                        help='data subset to evaluate (default: test)')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--save_ply', action='store_true', help='save assembly PLY files')
    return parser


def load_model(cfg, args):
    model = create_model(cfg).cuda()
    snapshot = args.snapshot
    if snapshot is None and args.test_epoch is not None:
        snapshot = osp.join(cfg.snapshot_dir, 'epoch-{}.pth.tar'.format(args.test_epoch))
    if snapshot is None:
        raise ValueError('Either --snapshot or --test_epoch must be provided')
    print('Loading checkpoint: {}'.format(snapshot))
    state_dict = torch.load(snapshot)
    model.load_state_dict(state_dict['model'])
    print('Loaded epoch: {}'.format(state_dict.get('epoch', 'unknown')))
    model.eval()
    return model


def get_neighbor_limits(cfg):
    from geotransformer.utils.data import calibrate_neighbors_stack_mode
    train_dataset = BoneFracturePairDataset(
        cfg.data.dataset_root, 'train',
        point_limit=5000,
        use_augmentation=False,
        rotation_magnitude=cfg.test.rotation_magnitude,
        translation_magnitude=cfg.test.translation_magnitude,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    return neighbor_limits


def group_metadata_by_bone(metadata_list):
    groups = {}
    for item in metadata_list:
        key = (item['bone_id'], item['variant'])
        if key not in groups:
            groups[key] = {}
        groups[key][item['src_piece']] = item
    return groups


def run_pairwise_inference(model, ref_points, src_points, neighbor_limits, cfg):
    """Run GeoTransformer on a single (ref, src) pair. Returns estimated transform."""
    import torch
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda

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
    return output_dict['estimated_transform'], output_dict.get('corr_scores', None)


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = make_cfg()

    log_file = osp.join(cfg.log_dir, 'eval-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    logger = Logger(log_file=log_file)
    logger.info('Command: ' + ' '.join(sys.argv))
    logger.info('NOTE: eval now applies displacement to src BEFORE inference (FIXED)')

    model = load_model(cfg, args)
    neighbor_limits = get_neighbor_limits(cfg)
    logger.info('Neighbor limits: {}'.format(neighbor_limits.tolist()))

    with open(osp.join(cfg.data.dataset_root, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    eval_metadata = metadata[args.subset]
    groups = group_metadata_by_bone(eval_metadata)
    logger.info('Evaluation groups ({}, bone/variant): {}'.format(args.subset, len(groups)))

    meter = SummaryBoard()
    for metric_name in ['part_acc', 'chamfer_distance', 'trans_rmse', 'trans_mae', 'rot_rmse', 'rot_mae']:
        meter.register_meter(metric_name)

    all_results = []
    ply_dir = osp.join(cfg.output_dir, 'assembly_ply') if args.save_ply else None
    if ply_dir:
        import os
        os.makedirs(ply_dir, exist_ok=True)

    for (bone_id, variant), pieces in groups.items():
        logger.info('Processing: {} / {}'.format(bone_id, variant))

        # === Load piece_0 (reference) ===
        ref_file = None
        for piece_key, item in pieces.items():
            ref_file = item['ref_file']
            break
        ref_points = np.load(osp.join(cfg.data.dataset_root, ref_file)).astype(np.float32)

        # === Load source pieces, generate displacements, and APPLY them ===
        displaced_src_list = []      # displaced pieces → fed to model
        original_src_list = []       # original assembled → GT comparison
        gt_displacements = []        # applied transforms

        piece_keys = sorted(pieces.keys())
        from geotransformer.utils.pointcloud import random_sample_transform
        rng = np.random.RandomState(cfg.seed)

        for pk in piece_keys:
            src_file = pieces[pk]['src_file']
            src_original = np.load(osp.join(cfg.data.dataset_root, src_file)).astype(np.float32)

            # Generate random displacement
            euler = rng.rand(3) * np.pi * cfg.test.rotation_magnitude / 180.0
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_euler('zyx', euler).as_matrix()
            translation = rng.uniform(-cfg.test.translation_magnitude, cfg.test.translation_magnitude, 3)
            T_displace = np.eye(4)
            T_displace[:3, :3] = rotation
            T_displace[:3, 3] = translation

            # === FIX: Actually displace the source fragment ===
            displaced_src = apply_transform(src_original, T_displace)

            displaced_src_list.append(displaced_src.astype(np.float32))
            original_src_list.append(src_original)
            gt_displacements.append(T_displace)

        if len(displaced_src_list) == 0:
            logger.info('  No source pieces, skipping')
            continue

        # === Run GeoTransformer on each displaced pair ===
        pred_transforms = []
        pairwise_corr_counts = []
        for displaced_src in displaced_src_list:
            T_pred, corr_scores = run_pairwise_inference(
                model, ref_points, displaced_src, neighbor_limits, cfg
            )
            pred_transforms.append(T_pred)
            pairwise_corr_counts.append(len(corr_scores) if corr_scores is not None else 0)

        # === Compute assembled positions ===
        # piece_0 stays at ref_points (reference frame)
        # piece_i: apply T_pred to displaced_src → model's assembled position
        assembled_ref = ref_points
        assembled_pred = [assembled_ref]  # piece_0 first
        assembled_gt = [assembled_ref]    # piece_0 first

        for i, (displaced_src, T_pred, T_disp, original_src) in enumerate(
            zip(displaced_src_list, pred_transforms, gt_displacements, original_src_list)
        ):
            # Model's assembly: apply predicted transform to displaced fragment
            model_assembly = apply_transform(displaced_src, T_pred)

            # GT assembly: apply inverse displacement (undoes the displacement)
            T_gt = inverse_transform(T_disp)
            gt_assembly = apply_transform(displaced_src, T_gt)  # should ≈ original_src

            assembled_pred.append(model_assembly)
            assembled_gt.append(gt_assembly)

        # === Save PLY for visualization ===
        if ply_dir:
            try:
                import open3d as o3d
                # Model's assembly
                colors = [[1, 0.8, 0], [0, 0.6, 1], [1, 0.2, 0.2], [0.2, 1, 0.2],
                          [1, 0.5, 0], [0.5, 0, 1]]
                pcds_pred, pcds_gt = [], []
                for i, pts in enumerate(assembled_pred):
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.paint_uniform_color(colors[i % len(colors)])
                    pcds_pred.append(pcd)
                o3d.io.write_point_cloud(
                    osp.join(ply_dir, '{}_{}_pred.ply'.format(bone_id[:8], variant)),
                    pcds_pred[0] if len(pcds_pred) == 1 else
                    o3d.geometry.PointCloud(sum([list(p.points) for p in pcds_pred], []))
                )
                for i, pts in enumerate(assembled_gt):
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.paint_uniform_color(colors[i % len(colors)])
                    pcds_gt.append(pcd)
                # Save individual pieces
                for i, pts in enumerate(assembled_pred):
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.paint_uniform_color(colors[i % len(colors)])
                    o3d.io.write_point_cloud(
                        osp.join(ply_dir, '{}_{}_pred_piece{}.ply'.format(bone_id[:8], variant, i)),
                        pcd
                    )
            except Exception as e:
                logger.info('  PLY save failed: {}'.format(e))

        # === Compute Jigsaw metrics ===
        # Pred transforms: from displaced → assembled
        pred_rots = [np.eye(3)]  # piece_0 identity
        pred_trans = [np.zeros(3)]
        gt_rots = [np.eye(3)]
        gt_trans = [np.zeros(3)]

        for T_pred, T_disp in zip(pred_transforms, gt_displacements):
            R_pred, t_pred = T_pred[:3, :3], T_pred[:3, 3]
            pred_rots.append(R_pred)
            pred_trans.append(t_pred)

            T_gt = inverse_transform(T_disp)
            R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
            gt_rots.append(R_gt)
            gt_trans.append(t_gt)

        # For Part Accuracy: compare model assembly vs GT assembly
        # Both should put the fragment in assembled position
        part_acc, cd = calc_part_acc(
            [ref_points] + displaced_src_list,  # base points (displaced for non-ref)
            pred_rots, pred_trans,
            gt_rots, gt_trans,
        )

        t_rmse = trans_metrics(pred_trans, gt_trans, 'rmse')
        t_mae = trans_metrics(pred_trans, gt_trans, 'mae')
        r_rmse = rot_metrics(pred_rots, gt_rots, 'rmse')
        r_mae = rot_metrics(pred_rots, gt_rots, 'mae')

        metrics = {
            'part_acc': part_acc,
            'chamfer_distance': cd,
            'trans_rmse': t_rmse,
            'trans_mae': t_mae,
            'rot_rmse': r_rmse,
            'rot_mae': r_mae,
        }

        for k, v in metrics.items():
            meter.update(k, v)

        if args.verbose:
            logger.info('  Fragments: {} ({} src)'.format(len(displaced_src_list) + 1, len(displaced_src_list)))
            logger.info('  Corr counts: {}'.format(pairwise_corr_counts))
            logger.info('  Part Acc: {:.4f}'.format(part_acc))
            logger.info('  CD:       {:.6f}'.format(cd))
            logger.info('  Trans RMSE: {:.4f} m ({:.2f} mm)'.format(t_rmse, t_rmse * 1000))
            logger.info('  Trans MAE:  {:.4f} m ({:.2f} mm)'.format(t_mae, t_mae * 1000))
            logger.info('  Rot RMSE: {:.4f} deg'.format(r_rmse))
            logger.info('  Rot MAE:  {:.4f} deg'.format(r_mae))

        all_results.append({
            'bone_id': bone_id,
            'variant': variant,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'n_fragments': len(displaced_src_list) + 1,
        })

    # Print aggregate
    logger.critical('=== Bone Fracture Assembly Results (Jigsaw Metrics) ===')
    logger.critical('  Num assemblies:  {}'.format(len(all_results)))
    logger.critical('  Part Accuracy:   {:.4f}'.format(meter.mean('part_acc')))
    logger.critical('  Chamfer Dist:    {:.6f}'.format(meter.mean('chamfer_distance')))
    logger.critical('  Trans RMSE (m):  {:.4f}'.format(meter.mean('trans_rmse')))
    logger.critical('  Trans RMSE (mm): {:.2f}'.format(meter.mean('trans_rmse') * 1000))
    logger.critical('  Trans MAE  (m):  {:.4f}'.format(meter.mean('trans_mae')))
    logger.critical('  Trans MAE  (mm): {:.2f}'.format(meter.mean('trans_mae') * 1000))
    logger.critical('  Rot RMSE  (deg): {:.4f}'.format(meter.mean('rot_rmse')))
    logger.critical('  Rot MAE   (deg): {:.4f}'.format(meter.mean('rot_mae')))

    summary = {
        'num_assemblies': len(all_results),
        'metrics': {
            name: float(meter.mean(name))
            for name in ['part_acc', 'chamfer_distance', 'trans_rmse', 'trans_mae', 'rot_rmse', 'rot_mae']
        },
        'trans_rmse_mm': float(meter.mean('trans_rmse') * 1000),
        'trans_mae_mm': float(meter.mean('trans_mae') * 1000),
        'per_assembly': all_results,
    }
    summary_path = osp.join(cfg.output_dir, 'assembly_metrics.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info('Summary saved to {}'.format(summary_path))


if __name__ == '__main__':
    main()
