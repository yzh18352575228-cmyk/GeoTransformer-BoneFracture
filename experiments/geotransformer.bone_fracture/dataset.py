import os.path as osp
import pickle
import numpy as np
import torch

from geotransformer.utils.pointcloud import (
    random_sample_transform,
    apply_transform,
    inverse_transform,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


class BoneFracturePairDataset(torch.utils.data.Dataset):
    """Dataset for bone fracture fragment registration.

    Pieces are in their assembled (correct) positions. During training, random
    rigid displacements are applied to source fragments online, and the inverse
    transform serves as the ground truth.

    Args:
        dataset_root: path to my_data_processed/
        subset: 'train' or 'val'
        point_limit: max points per cloud (random subsample if exceeded)
        use_augmentation: enable noise + rotation augmentation
        augmentation_noise: noise magnitude for augmentation (meters)
        augmentation_rotation: rotation magnitude factor for augmentation
        rotation_magnitude: max fracture displacement rotation (degrees)
        translation_magnitude: max fracture displacement translation (meters)
    """

    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1.0,
        rotation_magnitude=30.0,
        translation_magnitude=0.02,
    ):
        super(BoneFracturePairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude

        # Load metadata
        with open(osp.join(self.dataset_root, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            self.metadata_list = metadata[subset]

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, rel_path):
        points = np.load(osp.join(self.dataset_root, rel_path))
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[:self.point_limit]
            points = points[indices]
        return points

    def _augment_point_cloud(self, ref_points, src_points, transform):
        """Apply random rotation to one cloud + noise to both (3DMatch-style)."""
        from geotransformer.utils.pointcloud import (
            random_sample_rotation,
            get_rotation_translation_from_transform,
        )
        aug_rotation = random_sample_rotation(self.aug_rotation)
        rotation, translation = get_rotation_translation_from_transform(transform)

        if np.random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        new_transform = get_transform_from_rotation_translation(rotation, translation)
        return ref_points, src_points, new_transform

    def __getitem__(self, index):
        metadata = self.metadata_list[index]

        # Load pieces in their assembled positions
        ref_points = self._load_point_cloud(metadata['ref_file'])
        src_points = self._load_point_cloud(metadata['src_file'])

        # Generate random fracture displacement for src
        T_displace = random_sample_transform(
            rotation_magnitude=self.rotation_magnitude,
            translation_magnitude=self.translation_magnitude,
        )
        src_points = apply_transform(src_points, T_displace)

        # Ground truth: maps displaced src back to ref (assembled position)
        transform = inverse_transform(T_displace)

        # Optional standard augmentation
        if self.use_augmentation:
            ref_points, src_points, transform = self._augment_point_cloud(
                ref_points, src_points, transform
            )

        data_dict = {
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'transform': transform.astype(np.float32),
        }
        return data_dict


def train_valid_data_loader(cfg, distributed):
    train_dataset = BoneFracturePairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        rotation_magnitude=cfg.train.rotation_magnitude,
        translation_magnitude=cfg.train.translation_magnitude,
    )

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = BoneFracturePairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        rotation_magnitude=cfg.test.rotation_magnitude,
        translation_magnitude=cfg.test.translation_magnitude,
    )

    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg):
    train_dataset = BoneFracturePairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        rotation_magnitude=cfg.train.rotation_magnitude,
        translation_magnitude=cfg.train.translation_magnitude,
    )

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = BoneFracturePairDataset(
        cfg.data.dataset_root,
        'test',                        # held-out test bones
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        rotation_magnitude=cfg.test.rotation_magnitude,
        translation_magnitude=cfg.test.translation_magnitude,
    )

    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits
