"""
CamVid Dataset Preparation Script
- Scans raw and labeled folders
- Verifies file pairs
- Creates train/val/test splits
- Computes dataset statistics
- Computes class weights
"""

import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

# Configuration
RAW_IMAGE_DIR = "./CamVid/701_StillsRaw_full"  # For testing with uploaded files
LABEL_DIR = "./CamVid/LabeledApproved_full"
LABEL_COLORS_FILE = "./CamVid/label_colors.txt"
OUTPUT_DIR = "./CamVid/splits"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def load_color_mapping(label_colors_file):
    """Load RGB to class name mapping from label_colors.txt"""
    color_to_class = {}
    class_names = []

    with open(label_colors_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # skip invalid lines
            r, g, b = map(int, parts[:3])
            class_name = ' '.join(parts[3:])
            color_to_class[(r, g, b)] = len(class_names)
            class_names.append(class_name)

    return color_to_class, class_names


def find_image_pairs(raw_dir, label_dir):
    """Find matching pairs of raw images and labeled masks"""
    raw_files = set()
    label_files = set()

    # Get all raw images
    for f in os.listdir(raw_dir):
        if f.endswith('.png') and not f.endswith('_L.png'):
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
    """Split files into train/val/test sets"""
    random.seed(seed)
    np.random.seed(seed)

    # Shuffle
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


def compute_class_distribution(file_list, label_dir, color_to_class, num_classes):
    """Compute pixel count for each class"""
    class_counts = np.zeros(num_classes, dtype=np.int64)

    print("Computing class distribution...")
    for filename in tqdm(file_list):
        label_filename = filename[:-4] + '_L.png'
        label_path = os.path.join(label_dir, label_filename)

        if os.path.exists(label_path):
            mask = np.array(Image.open(label_path))
            mask_indexed = rgb_to_class_index(mask, color_to_class)

            # Count pixels for each class (ignore 255)
            for class_idx in range(num_classes):
                class_counts[class_idx] += np.sum(mask_indexed == class_idx)

    return class_counts


def compute_image_statistics(file_list, raw_dir):
    """Compute mean and std of training images"""
    print("Computing image statistics (mean, std)...")

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for filename in tqdm(file_list):
        img_path = os.path.join(raw_dir, filename)
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path)).astype(np.float64) / 255.0
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


def save_dataset_info(output_dir, color_to_class, class_names, mean, std, class_counts):
    """Save dataset metadata"""
    import json

    info = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'color_to_class': {str(k): v for k, v in color_to_class.items()},
        'mean': mean.tolist(),
        'std': std.tolist(),
        'class_counts': class_counts.tolist(),
        'image_size': [480, 360],  # width x height
        'ignore_index': 255
    }

    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Dataset info saved to {output_dir}/dataset_info.json")


def main():
    print("=" * 60)
    print("CamVid Dataset Preparation")
    print("=" * 60)

    # Load color mapping
    print("\n1. Loading color mapping...")
    color_to_class, class_names = load_color_mapping(LABEL_COLORS_FILE)
    print(f"   Found {len(class_names)} classes")

    # Find image pairs
    print("\n2. Finding image-label pairs...")
    paired_files = find_image_pairs(RAW_IMAGE_DIR, LABEL_DIR)
    print(f"   Found {len(paired_files)} valid pairs")

    # Create splits
    print("\n3. Creating train/val/test splits...")
    train_files, val_files, test_files = create_splits(
        paired_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    print(f"   Train: {len(train_files)} images")
    print(f"   Val:   {len(val_files)} images")
    print(f"   Test:  {len(test_files)} images")

    # Compute statistics on training set
    print("\n4. Computing dataset statistics...")
    mean, std = compute_image_statistics(train_files, RAW_IMAGE_DIR)
    print(f"   Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"   Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    # Compute class distribution
    print("\n5. Computing class distribution...")
    class_counts = compute_class_distribution(
        train_files, LABEL_DIR, color_to_class, len(class_names)
    )

    # Show class distribution
    print("\n   Class distribution (training set):")
    total_pixels = class_counts.sum()
    for idx, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"   {idx:2d}. {name:20s}: {count:10d} ({percentage:5.2f}%)")

    # Save everything
    print("\n6. Saving files...")
    save_split_files(OUTPUT_DIR, train_files, val_files, test_files)
    save_dataset_info(OUTPUT_DIR, color_to_class, class_names, mean, std, class_counts)

    print("\n" + "=" * 60)
    print("Preparation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
