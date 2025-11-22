"""
Dataset that loads pre-computed features instead of raw images
Supports both VGG16 and ResNet101 feature formats
"""

import os
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """
    PyTorch Dataset for loading pre-computed backbone features

    Args:
        feature_dir: Directory containing cached feature files
        split_name: 'train', 'val', or 'test'
    """

    def __init__(self, feature_dir, split_name):
        self.feature_dir = feature_dir
        self.split_name = split_name

        # Find all feature files for this split
        self.feature_files = sorted([
            f for f in os.listdir(feature_dir) 
            if f.startswith(f"{split_name}_") and f.endswith('.pt')
            ])

        if len(self.feature_files) == 0:
            raise ValueError(f"No feature files found for split '{split_name}' in {feature_dir}")

        print(f"  {split_name}: {len(self.feature_files)} cached features")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        """
        Load pre-computed features

        Returns:
            dict with keys:
                - For VGG16: 'x3', 'x4', 'x5', 'mask', 'filename'
                - For ResNet101: 'x1', 'x2', 'x3', 'x4', 'mask', 'filename'
        """
        feature_path = os.path.join(self.feature_dir, self.feature_files[idx])
        data = torch.load(feature_path, weights_only=False)
        return data


# Example usage and testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test FeatureDataset')
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='Directory containing cached features')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Split to test (default: train)')
    args = parser.parse_args()

    # Create dataset
    print(f"Loading features from: {args.feature_dir}")
    dataset = FeatureDataset(args.feature_dir, args.split)

    print(f"\nDataset size: {len(dataset)}")

    # Load first sample
    if len(dataset) > 0:
        print("\nLoading first sample...")
        sample = dataset[0]

        print(f"Keys: {list(sample.keys())}")
        print(f"Filename: {sample['filename']}")
        print(f"Mask shape: {sample['mask'].shape}")

        # Check which backbone (VGG16 has x3,x4,x5 | ResNet has x1,x2,x3,x4)
        if 'x1' in sample:
            print("\nResNet101 features detected:")
            print(f"  x1 shape: {sample['x1'].shape}")  # (256, H/4, W/4)
            print(f"  x2 shape: {sample['x2'].shape}")  # (512, H/8, W/8)
            print(f"  x3 shape: {sample['x3'].shape}")  # (1024, H/16, W/16)
            print(f"  x4 shape: {sample['x4'].shape}")  # (2048, H/32, W/32)
        else:
            print("\nVGG16 features detected:")
            print(f"  x3 shape: {sample['x3'].shape}")  # (256, H/8, W/8)
            print(f"  x4 shape: {sample['x4'].shape}")  # (512, H/16, W/16)
            print(f"  x5 shape: {sample['x5'].shape}")  # (512, H/32, W/32)

        print("\n✓ Feature dataset working correctly!")
