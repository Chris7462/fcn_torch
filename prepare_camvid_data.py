"""
CamVid Dataset Preparation Script
- Scans raw and labeled folders
- Verifies file pairs
- Creates train/val/test splits
- Computes dataset statistics
- Computes class weights
- Groups 32 classes into 11 classes following MATLAB's SegNet methodology
"""

import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

# Configuration
RAW_IMAGE_DIR = "./CamVid/701_StillsRaw_full"
LABEL_IMAGE_DIR = "./CamVid/LabeledApproved_full"
LABEL_COLORS_FILE = "./CamVid/label_colors.txt"
OUTPUT_DIR = "./CamVid/splits"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Target image size (width, height) for training
TARGET_SIZE = (480, 352)  # Resize to 480x360, then centercrop to 480x352. Divisible by 32


def get_11_class_mapping():
    """
    Returns the 11-class grouping following MATLAB's SegNet methodology.
    Maps RGB colors to the new 11 classes.
    
    Returns:
        rgb_to_new_class: dict mapping (R,G,B) tuples to new class index (0-10)
        new_class_names: list of 11 class names
    """
    # Define the 11 classes and their RGB colors from original 32 classes
    class_grouping = {
        'Sky': [
            (128, 128, 128),  # Sky
        ],
        'Building': [
            (0, 128, 64),     # Bridge
            (128, 0, 0),      # Building
            (64, 192, 0),     # Wall
            (64, 0, 64),      # Tunnel
            (192, 0, 128),    # Archway
        ],
        'Pole': [
            (192, 192, 128),  # Column_Pole
            (0, 0, 64),       # TrafficCone
        ],
        'Road': [
            (128, 64, 128),   # Road
            (128, 0, 192),    # LaneMkgsDriv
            (192, 0, 64),     # LaneMkgsNonDriv
        ],
        'Pavement': [
            (0, 0, 192),      # Sidewalk
            (64, 192, 128),   # ParkingBlock
            (128, 128, 192),  # RoadShoulder
        ],
        'Tree': [
            (128, 128, 0),    # Tree
            (192, 192, 0),    # VegetationMisc
        ],
        'SignSymbol': [
            (192, 128, 128),  # SignSymbol
            (128, 128, 64),   # Misc_Text
            (0, 64, 64),      # TrafficLight
        ],
        'Fence': [
            (64, 64, 128),    # Fence
        ],
        'Car': [
            (64, 0, 128),     # Car
            (64, 128, 192),   # SUVPickupTruck
            (192, 128, 192),  # Truck_Bus
            (192, 64, 128),   # Train
            (128, 64, 64),    # OtherMoving
        ],
        'Pedestrian': [
            (64, 64, 0),      # Pedestrian
            (192, 128, 64),   # Child
            (64, 0, 192),     # CartLuggagePram
            (64, 128, 64),    # Animal
        ],
        'Bicyclist': [
            (0, 128, 192),    # Bicyclist
            (192, 0, 192),    # MotorcycleScooter
        ],
    }
    
    # Build the mapping
    rgb_to_new_class = {}
    new_class_names = list(class_grouping.keys())
    
    for new_class_idx, (class_name, rgb_list) in enumerate(class_grouping.items()):
        for rgb in rgb_list:
            rgb_to_new_class[rgb] = new_class_idx
    
    return rgb_to_new_class, new_class_names


def load_color_mapping(label_colors_file):
    """Load RGB to class name mapping and apply 11-class grouping"""
    rgb_to_new_class, new_class_names = get_11_class_mapping()
    
    # Verify we have all colors from label_colors.txt mapped
    # (This is just for validation - we use the hardcoded mapping above)
    print(f"   Grouping 32 original classes into {len(new_class_names)} classes")
    
    return rgb_to_new_class, new_class_names


def find_image_pairs(raw_dir, label_dir):
    """Find matching pairs of raw images and labeled masks"""
    raw_files = set()
    label_files = set()

    # Get all raw images
    for f in os.listdir(raw_dir):
        if f.endswith('.png'):
            raw_files.add(f)

    # Get all label images
    for f in os.listdir(label_dir):
        if f.endswith('_L.png'):
            # Get the base name without _L suffix
            base_name = f[:-6] + '.png'  # Remove '_L.png', add '.png'
            label_files.add(base_name)

    # Find pairs (files that exist in both)
    paired_files = sorted(raw_files.intersection(label_files))

    return paired_files


def create_splits(file_list, train_ratio, val_ratio, test_ratio, seed=42):
    """Split files into train/val/test sets with validation"""
    ratio_sum = train_ratio + val_ratio + test_ratio
    assert abs(ratio_sum - 1.0) < 1e-6, f'Ratios must sum to 1.0, got {ratio_sum}'

    # Shuffle
    random.seed(seed)
    files = file_list.copy()
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def rgb_to_class_index(mask_rgb, color_to_class, ignore_index=255):
    """Convert RGB mask to class indices"""
    h, w = mask_rgb.shape[:2]
    mask_indexed = np.full((h, w), ignore_index, dtype=np.uint8)

    # Create a mapping for all pixels
    for (r, g, b), class_idx in color_to_class.items():
        # Find pixels matching this color
        matches = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
        mask_indexed[matches] = class_idx

    return mask_indexed


def compute_class_distribution(file_list, label_image_dir, color_to_class, num_classes):
    """Compute pixel count for each class"""
    class_counts = np.zeros(num_classes, dtype=np.int64)

    print("Computing class distribution...")
    for filename in tqdm(file_list):
        label_filename = filename[:-4] + '_L.png'
        label_path = os.path.join(label_image_dir, label_filename)

        if os.path.exists(label_path):
            mask = np.array(Image.open(label_path))
            mask_indexed = rgb_to_class_index(mask, color_to_class)

            # Count pixels for each class (ignore 255)
            for class_idx in range(num_classes):
                class_counts[class_idx] += np.sum(mask_indexed == class_idx)

    return class_counts


def compute_class_weights(class_counts):
    """
    Compute class weights for handling class imbalance
    Uses inverse frequency weighting

    Args:
        class_counts: numpy array of pixel counts for each class

    Returns:
        list: Class weights
    """
    class_counts = np.array(class_counts)

    # Set weight to 0 for classes with no samples
    class_weights = np.zeros(len(class_counts))

    # Only compute weights for classes that actually exist
    mask = class_counts > 0

    if mask.sum() > 0:
        # Inverse frequency for existing classes
        total_pixels = class_counts[mask].sum()
        class_weights[mask] = total_pixels / (mask.sum() * class_counts[mask])

    # Normalize
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    return class_weights.tolist()


def compute_image_statistics(file_list, raw_dir, target_size=(480, 352)):
    """Compute mean and std of images at the target training size using resize then center crop"""
    print(f"Computing image statistics mean and std at size of {target_size[0]}x{target_size[1]}...")

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for filename in tqdm(file_list):
        img_path = os.path.join(raw_dir, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path)

            # Resize to 480x360
            img = img.resize((480, 360), Image.BILINEAR)

            # Center crop to target size
            width, height = img.size
            left = (width - target_size[0]) // 2
            top = (height - target_size[1]) // 2
            right = left + target_size[0]
            bottom = top + target_size[1]
            img = img.crop((left, top, right, bottom))

            img = np.array(img).astype(np.float64) / 255.0

            pixel_sum += img.sum(axis=(0, 1))
            pixel_squared_sum += (img ** 2).sum(axis=(0, 1))
            pixel_count += img.shape[0] * img.shape[1]

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_squared_sum / pixel_count - mean ** 2)

    return mean, std


def save_split_files(output_dir, train_files, val_files, test_files):
    """Save train/val/test file lists"""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_files))

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_files))

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_files))

    print(f"Split files saved to {output_dir}")


def save_dataset_info(output_dir, color_to_class, class_names, mean, std, class_counts, class_weights, target_size):
    """Save dataset metadata"""
    import json

    info = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'ignore_index': 255,
        'color_to_class': {str(k): v for k, v in color_to_class.items()},
        'mean': mean.tolist(),
        'std': std.tolist(),
        'class_counts': class_counts.tolist(),
        'class_weights': class_weights,
        'image_size': list(target_size)  # [width, height]
    }

    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Dataset info saved to {output_dir}/dataset_info.json")


def main():
    print("=" * 60)
    print("CamVid Dataset Preparation (11 Classes)")
    print("=" * 60)
    print(f"Images will be resized to 480x360 then center-cropped to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    # Load color mapping
    print("\n1. Loading color mapping and applying 11-class grouping...")
    color_to_class, class_names = load_color_mapping(LABEL_COLORS_FILE)
    print(f"   Found {len(class_names)} classes: {', '.join(class_names)}")

    # Find image pairs
    print("\n2. Finding image-label pairs...")
    paired_files = find_image_pairs(RAW_IMAGE_DIR, LABEL_IMAGE_DIR)
    print(f"   Found {len(paired_files)} valid pairs")

    # Create splits
    print("\n3. Creating train/val/test splits...")
    train_files, val_files, test_files = create_splits(
        paired_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    print(f"   Train: {len(train_files)} images")
    print(f"   Val:   {len(val_files)} images")
    print(f"   Test:  {len(test_files)} images")

    # Save split files first (needed for dataloader)
    print("\n4. Saving split files...")
    save_split_files(OUTPUT_DIR, train_files, val_files, test_files)

    # Compute statistics on training set at target size
    print("\n5. Computing dataset statistics...")
    mean, std = compute_image_statistics(train_files, RAW_IMAGE_DIR, TARGET_SIZE)
    print(f"   Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"   Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    # Compute class distribution
    print("\n6. Computing class distribution...")
    class_counts = compute_class_distribution(
        train_files, LABEL_IMAGE_DIR, color_to_class, len(class_names)
    )

    # Compute class weights
    print("\n7. Computing class weights...")
    class_weights = compute_class_weights(class_counts)

    # Show class distribution
    print("\n   Class distribution (training set):")
    total_pixels = class_counts.sum()
    for idx, (name, count, weight) in enumerate(zip(class_names, class_counts, class_weights)):
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"   {idx:2d}. {name:20s}: {count:10d} ({percentage:5.2f}%) weight: {weight:.4f}")

    # Save everything
    print("\n8. Saving dataset info...")
    save_dataset_info(OUTPUT_DIR, color_to_class, class_names, mean, std, class_counts, class_weights, TARGET_SIZE)

    print("\n" + "=" * 60)
    print("Preparation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
