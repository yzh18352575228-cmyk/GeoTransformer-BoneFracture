import os
import os.path as osp
import pickle
import glob
import random
import numpy as np


def parse_obj_vertices(obj_path):
    """Extract vertex positions from an OBJ file."""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float32)


def preprocess_my_data(data_root, output_root, seed=7351):
    """
    Convert OBJ files to .npy and generate metadata with bone-level split.

    Directory structure in data_root:
        <bone_id>/
            fractured_XX/   piece_0.obj, piece_1.obj, ...

    Split: 38 bones train / 6 bones val / 6 bones test (shuffled by seed).
    Same bone NEVER appears in more than one split.
    """
    if not osp.exists(output_root):
        os.makedirs(output_root)

    random.seed(seed)

    bone_dirs = sorted([
        d for d in os.listdir(data_root)
        if osp.isdir(osp.join(data_root, d))
    ])
    random.shuffle(bone_dirs)

    # Split: ~76% train, ~12% val, ~12% test
    n_train = int(len(bone_dirs) * 0.76)
    n_val = int(len(bone_dirs) * 0.12)
    train_bones = set(bone_dirs[:n_train])
    val_bones = set(bone_dirs[n_train:n_train + n_val])
    test_bones = set(bone_dirs[n_train + n_val:])

    print(f'Found {len(bone_dirs)} bones')
    print(f'Train: {len(train_bones)} bones')
    print(f'Val:   {len(val_bones)} bones')
    print(f'Test:  {len(test_bones)} bones')

    # Also log which bones go where
    bone_assignment = {}
    for b in train_bones:
        bone_assignment[b] = 'train'
    for b in val_bones:
        bone_assignment[b] = 'val'
    for b in test_bones:
        bone_assignment[b] = 'test'

    train_metadata = []
    val_metadata = []
    test_metadata = []

    total_pairs = 0

    for bone_id in sorted(bone_dirs):
        bone_path = osp.join(data_root, bone_id)
        variant_dirs = sorted([
            d for d in os.listdir(bone_path)
            if osp.isdir(osp.join(bone_path, d))
        ])

        split = bone_assignment[bone_id]

        for variant in variant_dirs:
            variant_path = osp.join(bone_path, variant)
            piece_files = sorted(glob.glob(osp.join(variant_path, 'piece_*.obj')))

            if len(piece_files) < 2:
                continue

            # Create output directory
            out_dir = osp.join(output_root, bone_id, variant)
            os.makedirs(out_dir, exist_ok=True)

            # Convert each piece to .npy
            piece_paths = {}
            for pf in piece_files:
                piece_name = osp.splitext(osp.basename(pf))[0]
                out_path = osp.join(out_dir, f'{piece_name}.npy')
                vertices = parse_obj_vertices(pf)
                np.save(out_path, vertices)
                piece_paths[piece_name] = f'{bone_id}/{variant}/{piece_name}.npy'

            # Generate pairwise metadata: (piece_0, piece_i) for i > 0
            ref_key = 'piece_0'
            if ref_key not in piece_paths:
                continue

            if split == 'train':
                metadata_list = train_metadata
            elif split == 'val':
                metadata_list = val_metadata
            else:
                metadata_list = test_metadata

            for piece_id in sorted(piece_paths.keys()):
                if piece_id == ref_key:
                    continue
                metadata_list.append({
                    'bone_id': bone_id,
                    'variant': variant,
                    'ref_file': piece_paths[ref_key],
                    'src_file': piece_paths[piece_id],
                    'ref_piece': ref_key,
                    'src_piece': piece_id,
                })
                total_pairs += 1

    print(f'Converted {total_pairs} total .npy files')
    print(f'Train pairs: {len(train_metadata)}')
    print(f'Val pairs:   {len(val_metadata)}')
    print(f'Test pairs:  {len(test_metadata)}')

    # Save metadata
    metadata = {
        'train': train_metadata,
        'val': val_metadata,
        'test': test_metadata,
    }
    metadata_path = osp.join(output_root, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f'Metadata saved to {metadata_path}')

    return metadata


def main():
    project_root = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))
    data_root = osp.join(project_root, 'my_data')
    output_root = osp.join(project_root, 'my_data_processed')

    print(f'Data root:   {data_root}')
    print(f'Output root: {output_root}')
    print()

    metadata = preprocess_my_data(data_root, output_root)

    # Quick verification
    print('\n=== Verification ===')
    sample = metadata['train'][0]
    print(f'  Train sample: {sample["bone_id"]}/{sample["variant"]} ({sample["ref_piece"]}->{sample["src_piece"]})')
    sample = metadata['val'][0]
    print(f'  Val sample:   {sample["bone_id"]}/{sample["variant"]} ({sample["ref_piece"]}->{sample["src_piece"]})')
    sample = metadata['test'][0]
    print(f'  Test sample:  {sample["bone_id"]}/{sample["variant"]} ({sample["ref_piece"]}->{sample["src_piece"]})')

    # Verify no bone overlap
    train_bones = set(m['bone_id'] for m in metadata['train'])
    val_bones = set(m['bone_id'] for m in metadata['val'])
    test_bones = set(m['bone_id'] for m in metadata['test'])
    assert train_bones.isdisjoint(val_bones), 'Overlap train/val!'
    assert train_bones.isdisjoint(test_bones), 'Overlap train/test!'
    assert val_bones.isdisjoint(test_bones), 'Overlap val/test!'
    print('  Bone split verified: no overlap between train/val/test')


if __name__ == '__main__':
    main()
