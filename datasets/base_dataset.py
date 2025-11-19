"""
Base Dataset Class for Semantic Segmentation
"""

import os
import json
import numpy as np
from torch.utils.data import Dataset
from .registry import DATASETS


@DATASETS.register_module
class BaseDataset(Dataset):
    """
    Base class for all segmentation datasets

    Args:
        img_dir: Directory containing raw images
        label_dir: Directory containing label images
        split_file: Path to split file (train.txt, val.txt, test.txt)
        dataset_info_path: Path to dataset_info.json
        cfg: Global config object
    """

    def __init__(self, img_dir, label_dir, split_file, dataset_info_path, cfg):
        self.cfg = cfg
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.split_file = split_file
        self.dataset_info_path = dataset_info_path

        # Determine if training
        self.is_training = 'train' in os.path.basename(split_file)

        # Load dataset info
        self.load_dataset_info()

        # Load file list
        self.file_list = []
        self.init()

        # Get transforms
        self.transform = self.get_transform()

    def load_dataset_info(self):
        """Load dataset metadata from JSON"""
        with open(self.dataset_info_path, 'r') as f:
            info = json.load(f)

        self.num_classes = info['num_classes']
        self.class_names = info['class_names']
        self.ignore_index = info['ignore_index']
        self.mean = info.get('mean', [0.485, 0.456, 0.406])
        self.std = info.get('std', [0.229, 0.224, 0.225])

        # Build color to class mapping
        self.color_to_class = {}
        for color_str, class_idx in info['color_to_class'].items():
            # Parse "(r, g, b)" string back to tuple
            r, g, b = eval(color_str)
            self.color_to_class[(r, g, b)] = class_idx

    def init(self):
        """Load file list - can be overridden by subclasses"""
        with open(self.split_file, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]

    def get_transform(self):
        """Get transform pipeline - to be implemented by subclasses"""
        raise NotImplementedError

    def rgb_to_mask(self, mask_rgb):
        """Convert RGB mask to class indices"""
        h, w = mask_rgb.shape[:2]
        mask = np.full((h, w), self.ignore_index, dtype=np.int64)

        # Vectorized conversion
        for (r, g, b), class_idx in self.color_to_class.items():
            matches = (mask_rgb[:, :, 0] == r) & \
                      (mask_rgb[:, :, 1] == g) & \
                      (mask_rgb[:, :, 2] == b)
            mask[matches] = class_idx

        return mask

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """To be implemented by subclasses"""
        raise NotImplementedError
