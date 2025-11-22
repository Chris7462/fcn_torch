"""
Create dataloaders from cached features
Similar to create_camvid_dataloaders.py but for pre-computed features
"""

import json
from torch.utils.data import DataLoader
from feature_dataset import FeatureDataset


def create_feature_dataloaders(
        feature_dir,
        dataset_info_path,
        batch_size=16,
        num_workers=4
        ):
    """
    Create train, validation, and test dataloaders from cached features

    Args:
        feature_dir: Directory containing cached feature files
        dataset_info_path: Path to dataset_info.json (for class info and weights)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        dict with 'train', 'val', 'test' dataloaders and dataset metadata
    """

    # Load dataset info
    with open(dataset_info_path, 'r') as f:
        info = json.load(f)

    print(f"Creating feature dataloaders from: {feature_dir}")

    # Create datasets
    train_dataset = FeatureDataset(feature_dir, 'train')
    val_dataset = FeatureDataset(feature_dir, 'val')
    test_dataset = FeatureDataset(feature_dir, 'test')

    # Create dataloaders
    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
            )

    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
            )

    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
            )

    print(f"\nFeature DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    print(f"  Batch size: {batch_size}")

    return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'class_weights': info['class_weights'],
            'num_classes': info['num_classes'],
            'class_names': info['class_names'],
            'ignore_index': info['ignore_index']
            }


# Example usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test feature dataloaders')
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='Directory containing cached features')
    parser.add_argument('--dataset-info', type=str, required=True,
                        help='Path to dataset_info.json')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    args = parser.parse_args()

    # Create dataloaders
    dataloaders = create_feature_dataloaders(
            feature_dir=args.feature_dir,
            dataset_info_path=args.dataset_info,
            batch_size=args.batch_size,
            num_workers=4
            )

    train_loader = dataloaders['train']
    class_weights = dataloaders['class_weights']
    num_classes = dataloaders['num_classes']

    print(f"\nDataset info:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class weights: {class_weights[:3]}... (showing first 3)")

    # Test loading one batch
    print("\nTesting train loader...")
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Keys: {list(batch.keys())}")
        print(f"  Mask shape: {batch['mask'].shape}")

        # Check backbone type
        if 'x1' in batch:
            print(f"  ResNet101 features:")
            print(f"    x1 shape: {batch['x1'].shape}")
            print(f"    x2 shape: {batch['x2'].shape}")
            print(f"    x3 shape: {batch['x3'].shape}")
            print(f"    x4 shape: {batch['x4'].shape}")
        else:
            print(f"  VGG16 features:")
            print(f"    x3 shape: {batch['x3'].shape}")
            print(f"    x4 shape: {batch['x4'].shape}")
            print(f"    x5 shape: {batch['x5'].shape}")

        print(f"  Filenames (first 3): {batch['filename'][:3]}")
        break

    print("\n✓ Feature dataloaders working correctly!")
