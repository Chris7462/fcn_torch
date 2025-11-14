"""
CamVid DataLoader Creation
Creates train, validation, and test dataloaders
"""

import json
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from camvid_dataset import CamVidDataset


def get_training_transform(target_size, mean, std):
    """
    Get training transforms with data augmentation

    Args:
        target_size: (width, height) tuple
        mean: RGB mean for normalization (list of 3 floats)
        std: RGB std for normalization (list of 3 floats)
    """
    return A.Compose([
        A.Resize(height=360, width=480),
        A.CenterCrop(height=target_size[1], width=target_size[0]),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_validation_transform(target_size, mean, std):
    """
    Get validation/test transforms (no augmentation)

    Args:
        target_size: (width, height) tuple
        mean: RGB mean for normalization (list of 3 floats)
        std: RGB std for normalization (list of 3 floats)
    """
    return A.Compose([
        A.Resize(height=360, width=480),
        A.CenterCrop(height=target_size[1], width=target_size[0]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def create_dataloaders(
    raw_image_dir,
    label_dir,
    splits_dir,
    dataset_info_path,
    batch_size=8,
    num_workers=4,
    target_size=(480, 352)
):
    """
    Create train, validation, and test dataloaders

    Args:
        raw_image_dir: Directory with raw images
        label_dir: Directory with label images
        splits_dir: Directory containing train.txt, val.txt, test.txt
        dataset_info_path: Path to dataset_info.json
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        target_size: (width, height) tuple for center cropping (default: 480x352, divisible by 32)

    Returns:
        dict with 'train', 'val', 'test' dataloaders and 'class_weights'
    """

    # Load dataset info to get mean/std
    with open(dataset_info_path, 'r') as f:
        info = json.load(f)

    # Fall back to ImageNet stats if not provided
    mean = info.get('mean', [0.485, 0.456, 0.406])
    std = info.get('std', [0.229, 0.224, 0.225])

    print(f"Using normalization - Mean: {mean}, Std: {std}")

    # Create transforms
    train_transform = get_training_transform(target_size=target_size, mean=mean, std=std)
    val_transform = get_validation_transform(target_size=target_size, mean=mean, std=std)

    # Create datasets
    train_dataset = CamVidDataset(
        split_file=f'{splits_dir}/train.txt',
        raw_image_dir=raw_image_dir,
        label_image_dir=label_dir,
        dataset_info_path=dataset_info_path,
        transform=train_transform
    )

    val_dataset = CamVidDataset(
        split_file=f'{splits_dir}/val.txt',
        raw_image_dir=raw_image_dir,
        label_image_dir=label_dir,
        dataset_info_path=dataset_info_path,
        transform=val_transform
    )

    test_dataset = CamVidDataset(
        split_file=f'{splits_dir}/test.txt',
        raw_image_dir=raw_image_dir,
        label_image_dir=label_dir,
        dataset_info_path=dataset_info_path,
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
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

    # Load pre-computed class weights from dataset_info.json
    class_weights = info['class_weights']

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} images)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} images)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} images)")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {target_size[0]}x{target_size[1]}")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_weights': class_weights,
        'num_classes': info['num_classes'],
        'class_names': info['class_names'],
        'ignore_index': info['ignore_index']
    }


# Example usage
if __name__ == '__main__':
    # Configuration
    RAW_IMAGE_DIR = '/data/CamVid/701_StillsRaw_full'
    LABEL_DIR = '/data/CamVid/LabeledApproved_full'
    SPLITS_DIR = '/data/CamVid/splits'
    DATASET_INFO_PATH = '/data/CamVid/splits/dataset_info.json'

    # Create dataloaders
    dataloaders = create_dataloaders(
        raw_image_dir=RAW_IMAGE_DIR,
        label_dir=LABEL_DIR,
        splits_dir=SPLITS_DIR,
        dataset_info_path=DATASET_INFO_PATH,
        batch_size=8,
        num_workers=4,
        target_size=(480, 352),
        use_computed_stats=True
    )

    # Access dataloaders
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    class_weights = dataloaders['class_weights']

    print(f"\nClass weights shape: {len(class_weights)}")
    print(f"Number of classes: {dataloaders['num_classes']}")

    # Example: iterate through one batch
    print("\nTesting train loader...")
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        masks = batch['mask']
        filenames = batch['filename']

        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")  # [B, 3, H, W]
        print(f"  Masks shape: {masks.shape}")    # [B, H, W]
        print(f"  First filename: {filenames[0]}")
        break
