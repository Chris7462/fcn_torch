"""
CamVid Dataset for PyTorch
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CamVidDataset(Dataset):
    """
    CamVid Dataset for Semantic Segmentation

    Args:
        split_file: Path to train.txt, val.txt, or test.txt
        raw_image_dir: Directory containing raw images
        label_dir: Directory containing label images (_L.png)
        dataset_info_path: Path to dataset_info.json
        transform: Albumentations transform pipeline
    """

    def __init__(
        self,
        split_file,
        raw_image_dir,
        label_image_dir,
        dataset_info_path,
        transform=None
    ):
        self.raw_image_dir = raw_image_dir
        self.label_image_dir = label_image_dir
        self.transform = transform

        # Load file list
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]

        # Load dataset info
        with open(dataset_info_path, 'r') as f:
            info = json.load(f)

        self.num_classes = info['num_classes']
        self.class_names = info['class_names']
        self.ignore_index = info['ignore_index']

        # Build color to class mapping
        self.color_to_class = {}
        for color_str, class_idx in info['color_to_class'].items():
            # Parse "(r, g, b)" string back to tuple
            r, g, b = eval(color_str)
            self.color_to_class[(r, g, b)] = class_idx

        print(f"Loaded {len(self.file_list)} images for this split")

    def __len__(self):
        return len(self.file_list)

    def rgb_to_mask(self, mask_rgb):
        """Convert RGB mask to class indices"""
        h, w = mask_rgb.shape[:2]
        mask = np.full((h, w), self.ignore_index, dtype=np.int64)

        # Vectorized approach for speed
        for (r, g, b), class_idx in self.color_to_class.items():
            matches = (mask_rgb[:, :, 0] == r) & \
                      (mask_rgb[:, :, 1] == g) & \
                      (mask_rgb[:, :, 2] == b)
            mask[matches] = class_idx

        return mask

    def __getitem__(self, idx):
        # Get filename
        filename = self.file_list[idx]

        # Load raw image
        img_path = os.path.join(self.raw_image_dir, filename)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load label image
        label_filename = filename[:-4] + '_L.png'
        label_path = os.path.join(self.label_image_dir, label_filename)
        label_rgb = np.array(Image.open(label_path).convert('RGB'))

        # Convert RGB mask to class indices
        mask = self.rgb_to_mask(label_rgb)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Ensure mask is Long type for PyTorch
        mask = mask.long()

        return {
            'image': image,
            'mask': mask,
            'filename': filename
        }


# Example usage
if __name__ == '__main__':
    from create_camvid_dataloaders import get_training_transform

    # Test the dataset
    dataset_info_path = './CamVid/splits/dataset_info.json'

    # Load mean and std from dataset info
    with open(dataset_info_path, 'r') as f:
        info = json.load(f)

    # Use computed stats if available, otherwise use ImageNet defaults
    mean = info.get('mean', [0.485, 0.456, 0.406])
    std = info.get('std', [0.229, 0.224, 0.225])

    # Create dataset
    train_dataset = CamVidDataset(
        split_file='./CamVid/splits/train.txt',
        raw_image_dir='./CamVid/701_StillsRaw_full',
        label_image_dir='./CamVid/LabeledApproved_full',
        dataset_info_path=dataset_info_path,
        transform=get_training_transform(target_size=(960, 720), mean=mean, std=std)
    )

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Class names: {train_dataset.class_names[:5]}...")  # Show first 5

    # Test loading one sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\nSample loaded:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Filename: {sample['filename']}")
        print(f"  Unique mask values: {torch.unique(sample['mask']).numpy()}")
