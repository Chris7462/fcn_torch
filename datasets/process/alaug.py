"""
Albumentations Transform Pipelines for Semantic Segmentation
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_transform(img_height, img_width, mean, std):
    """
    Get training transforms with data augmentation

    Args:
        img_height: Target height after crop
        img_width: Target width after crop
        mean: RGB mean for normalization [R, G, B]
        std: RGB std for normalization [R, G, B]

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(height=360, width=480),
        A.CenterCrop(height=img_height, width=img_width),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_val_transform(img_height, img_width, mean, std):
    """
    Get validation/test transforms (no augmentation)

    Args:
        img_height: Target height after crop
        img_width: Target width after crop
        mean: RGB mean for normalization [R, G, B]
        std: RGB std for normalization [R, G, B]

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(height=360, width=480),
        A.CenterCrop(height=img_height, width=img_width),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
