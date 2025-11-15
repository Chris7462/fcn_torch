"""
Cityscapes Dataset for PyTorch
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CityscapesDataset(Dataset):
    """
    Cityscapes Dataset for Semantic Segmentation

    Args:
        split_file: Path to train.txt, val.txt, or test.txt
        leftimg_dir: Directory containing leftImg8bit images
        gtfine_dir: Directory containing gtFine label images
        dataset_info_path: Path to dataset_info.json
        transform: Albumentations transform pipeline
    """

    def __init__(
        self,
        split_file,
        leftimg_dir,
        gtfine_dir,
        dataset_info_path,
        transform=None
    ):
        self.leftimg_dir = leftimg_dir
        self.gtfine_dir = gtfine_dir
        self.transform = transform

        # Load file list (format: city/basename)
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]

        # Load dataset info
        with open(dataset_info_path, 'r') as f:
            info = json.load(f)

        self.num_classes = info['num_classes']
        self.class_names = info['class_names']
        self.ignore_index = info['ignore_index']
        self.label_id_to_train_id = {int(k): v for k, v in info['label_id_to_train_id'].items()}

        print(f"Loaded {len(self.file_list)} images for this split")

    def __len__(self):
        return len(self.file_list)

    def convert_label_id_to_train_id(self, label_id_mask):
        """Convert labelId mask to trainId mask"""
        train_id_mask = np.full(label_id_mask.shape, self.ignore_index, dtype=np.int64)

        for label_id, train_id in self.label_id_to_train_id.items():
            train_id_mask[label_id_mask == label_id] = train_id

        return train_id_mask

    def __getitem__(self, idx):
        # Get city/basename (e.g., 'aachen/aachen_000000_000019')
        file_path = self.file_list[idx]
        city, basename = file_path.split('/')

        # Load raw image
        img_filename = f'{basename}_leftImg8bit.png'
        img_path = os.path.join(self.leftimg_dir, city, img_filename)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load label image (labelIds)
        label_filename = f'{basename}_gtFine_labelIds.png'
        label_path = os.path.join(self.gtfine_dir, city, label_filename)
        label_id_mask = np.array(Image.open(label_path))

        # Convert labelId to trainId
        mask = self.convert_label_id_to_train_id(label_id_mask)

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
            'filename': file_path
        }


# Example usage
if __name__ == '__main__':
    from create_cityscapes_dataloaders import get_training_transform

    # Test the dataset
    dataset_info_path = './Cityscapes/splits/dataset_info.json'

    # Load mean and std from dataset info
    with open(dataset_info_path, 'r') as f:
        info = json.load(f)

    mean = info.get('mean', [0.485, 0.456, 0.406])
    std = info.get('std', [0.229, 0.224, 0.225])

    # Create dataset
    train_dataset = CityscapesDataset(
        split_file='./Cityscapes/splits/train.txt',
        leftimg_dir='./Cityscapes/leftImg8bit/train',
        gtfine_dir='./Cityscapes/gtFine/train',
        dataset_info_path=dataset_info_path,
        transform=get_training_transform(target_size=(2048, 1024), mean=mean, std=std)
    )

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Class names: {train_dataset.class_names[:5]}...")

    # Test loading one sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\nSample loaded:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Filename: {sample['filename']}")
        print(f"  Unique mask values: {torch.unique(sample['mask']).numpy()}")
