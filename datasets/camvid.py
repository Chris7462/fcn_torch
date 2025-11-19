"""
CamVid Dataset for Semantic Segmentation
11 classes grouped from original 32 classes
"""

import os
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset
from .registry import DATASETS
from .process import get_training_transform, get_val_transform


@DATASETS.register_module
class CamVid(BaseDataset):
    """
    CamVid Dataset

    Args:
        img_dir: Directory containing raw images (701_StillsRaw_full)
        label_dir: Directory containing label images (LabeledApproved_full)
        split_file: Path to split file (train.txt, val.txt, test.txt)
        dataset_info_path: Path to dataset_info.json
        cfg: Global config object
    """

    def __init__(self, img_dir, label_dir, split_file, dataset_info_path, cfg):
        super().__init__(img_dir, label_dir, split_file, dataset_info_path, cfg)

    def get_transform(self):
        """Get CamVid-specific transform pipeline"""
        if self.is_training:
            return get_training_transform(
                img_height=self.cfg.img_height,
                img_width=self.cfg.img_width,
                mean=self.mean,
                std=self.std
            )
        else:
            return get_val_transform(
                img_height=self.cfg.img_height,
                img_width=self.cfg.img_width,
                mean=self.mean,
                std=self.std
            )

    def __getitem__(self, idx):
        """
        Load and return one sample

        Returns:
            dict with keys:
                - 'image': Tensor [3, H, W]
                - 'mask': Tensor [H, W] with class indices
                - 'filename': str
        """
        # Get filename
        filename = self.file_list[idx]

        # Load raw image
        img_path = os.path.join(self.img_dir, filename)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load label image
        label_filename = filename[:-4] + '_L.png'
        label_path = os.path.join(self.label_dir, label_filename)
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
