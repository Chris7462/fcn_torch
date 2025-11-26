"""
Cityscapes Dataset Preparation Script
- Scans leftImg8bit and gtFine folders
- Uses official 34→19 class mapping
- Splits train into 90% train / 10% test
- Uses original val as val
- Computes dataset statistics and class weights
"""

import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import json

# Configuration
LEFTIMG_DIR = "./Cityscapes/leftImg8bit"
GTFINE_DIR = "./Cityscapes/gtFine"
OUTPUT_DIR = "./Cityscapes/splits"

# Split ratios
TRAIN_RATIO = 0.9
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Target image size (width, height) - keep original for now
#TARGET_SIZE = (2048, 1024)
TARGET_SIZE = (1024, 512)


# Official Cityscapes 34 → 19 class mapping
CITYSCAPES_CLASSES = {
    'unlabeled'            : (-1, (  0,  0,  0)),
    'ego vehicle'          : (-1, (  0,  0,  0)),
    'rectification border' : (-1, (  0,  0,  0)),
    'out of roi'           : (-1, (  0,  0,  0)),
    'static'               : (-1, (  0,  0,  0)),
    'dynamic'              : (-1, (111, 74,  0)),
    'ground'               : (-1, ( 81,  0, 81)),
    'road'                 : ( 0, (128, 64,128)),
    'sidewalk'             : ( 1, (244, 35,232)),
    'parking'              : (-1, (250,170,160)),
    'rail track'           : (-1, (230,150,140)),
    'building'             : ( 2, ( 70, 70, 70)),
    'wall'                 : ( 3, (102,102,156)),
    'fence'                : ( 4, (190,153,153)),
    'guard rail'           : (-1, (180,165,180)),
    'bridge'               : (-1, (150,100,100)),
    'tunnel'               : (-1, (150,120, 90)),
    'pole'                 : ( 5, (153,153,153)),
    'polegroup'            : (-1, (153,153,153)),
    'traffic light'        : ( 6, (250,170, 30)),
    'traffic sign'         : ( 7, (220,220,  0)),
    'vegetation'           : ( 8, (107,142, 35)),
    'terrain'              : ( 9, (152,251,152)),
    'sky'                  : (10, ( 70,130,180)),
    'person'               : (11, (220, 20, 60)),
    'rider'                : (12, (255,  0,  0)),
    'car'                  : (13, (  0,  0,142)),
    'truck'                : (14, (  0,  0, 70)),
    'bus'                  : (15, (  0, 60,100)),
    'caravan'              : (-1, (  0,  0, 90)),
    'trailer'              : (-1, (  0,  0,110)),
    'train'                : (16, (  0, 80,100)),
    'motorcycle'           : (17, (  0,  0,230)),
    'bicycle'              : (18, (119, 11, 32)),
    'license plate'        : (-1, (  0,  0,142)),
}

TRAINING_CLASSES = [
    'road',           # 0
    'sidewalk',       # 1
    'building',       # 2
    'wall',           # 3
    'fence',          # 4
    'pole',           # 5
    'traffic light',  # 6
    'traffic sign',   # 7
    'vegetation',     # 8
    'terrain',        # 9
    'sky',            # 10
    'person',         # 11
    'rider',          # 12
    'car',            # 13
    'truck',          # 14
    'bus',            # 15
    'train',          # 16
    'motorcycle',     # 17
    'bicycle',        # 18
]

# Mapping from labelId (0-33) to trainId (0-18 or 255 for ignore)
LABEL_ID_TO_TRAIN_ID = {
    0: 255,   # unlabeled -> ignore
    1: 255,   # ego vehicle -> ignore
    2: 255,   # rectification border -> ignore
    3: 255,   # out of roi -> ignore
    4: 255,   # static -> ignore
    5: 255,   # dynamic -> ignore
    6: 255,   # ground -> ignore
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking -> ignore
    10: 255,  # rail track -> ignore
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail -> ignore
    15: 255,  # bridge -> ignore
    16: 255,  # tunnel -> ignore
    17: 5,    # pole
    18: 255,  # polegroup -> ignore
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan -> ignore
    30: 255,  # trailer -> ignore
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    -1: 255,  # license plate -> ignore
}


def get_color_mapping():
    """Get trainId to RGB color mapping"""
    color_mapping = {}
    for name, (train_id, color) in CITYSCAPES_CLASSES.items():
        if train_id >= 0:  # Only valid training classes
            color_mapping[train_id] = color
    return color_mapping


def find_image_pairs(leftimg_dir, gtfine_dir, split):
    """
    Find matching pairs of images and labels for a given split

    Args:
        leftimg_dir: Base directory for leftImg8bit
        gtfine_dir: Base directory for gtFine
        split: 'train' or 'val'

    Returns:
        List of (city, basename) tuples
    """
    pairs = []
    split_img_dir = os.path.join(leftimg_dir, split)
    split_label_dir = os.path.join(gtfine_dir, split)

    # Iterate through cities
    for city in sorted(os.listdir(split_img_dir)):
        city_img_dir = os.path.join(split_img_dir, city)
        city_label_dir = os.path.join(split_label_dir, city)

        if not os.path.isdir(city_img_dir):
            continue

        # Get all images in this city
        for img_file in sorted(os.listdir(city_img_dir)):
            if img_file.endswith('_leftImg8bit.png'):
                # Extract basename (e.g., 'aachen_000000_000019')
                basename = img_file.replace('_leftImg8bit.png', '')

                # Check if corresponding label exists
                label_file = f'{basename}_gtFine_labelIds.png'
                label_path = os.path.join(city_label_dir, label_file)

                if os.path.exists(label_path):
                    pairs.append((city, basename))

    return pairs


def create_splits(train_pairs, val_pairs, train_ratio, seed=42):
    """
    Split train into train/test, keep val as is

    Args:
        train_pairs: List of (city, basename) tuples for original train
        val_pairs: List of (city, basename) tuples for val
        train_ratio: Ratio for new train split (rest goes to test)
        seed: Random seed

    Returns:
        new_train_pairs, val_pairs, test_pairs
    """
    # Shuffle train pairs
    random.seed(seed)
    train_pairs_shuffled = train_pairs.copy()
    random.shuffle(train_pairs_shuffled)

    # Split train into new_train and test
    n_total = len(train_pairs_shuffled)
    n_train = int(n_total * train_ratio)

    new_train_pairs = train_pairs_shuffled[:n_train]
    test_pairs = train_pairs_shuffled[n_train:]

    return new_train_pairs, val_pairs, test_pairs


def label_id_to_train_id(label_id_mask):
    """Convert labelId mask to trainId mask"""
    train_id_mask = np.full(label_id_mask.shape, 255, dtype=np.uint8)

    for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
        train_id_mask[label_id_mask == label_id] = train_id

    return train_id_mask


def compute_class_distribution(pairs, gtfine_dir, num_classes):
    """Compute pixel count for each class"""
    class_counts = np.zeros(num_classes, dtype=np.int64)

    print("Computing class distribution...")
    for city, basename in tqdm(pairs):
        label_file = f'{basename}_gtFine_labelIds.png'
        label_path = os.path.join(gtfine_dir, city, label_file)

        if os.path.exists(label_path):
            label_id_mask = np.array(Image.open(label_path))
            train_id_mask = label_id_to_train_id(label_id_mask)

            # Count pixels for each class (ignore 255)
            for class_idx in range(num_classes):
                class_counts[class_idx] += np.sum(train_id_mask == class_idx)

    return class_counts


def compute_class_weights(class_counts):
    """
    Compute class weights for handling class imbalance
    Uses Median Frequency Balancing: weight[c] = median_freq / freq[c]

    Args:
        class_counts: numpy array of pixel counts for each class

    Returns:
        list: Class weights
    """
    class_counts = np.array(class_counts)
    class_weights = np.zeros(len(class_counts))

    mask = class_counts > 0
    if mask.sum() > 0:
        # Calculate frequencies
        total_pixels = class_counts.sum()
        freq = class_counts[mask] / total_pixels

        # Median Frequency Balancing
        median_freq = np.median(freq)
        class_weights[mask] = median_freq / freq

    return class_weights.tolist()


def compute_image_statistics(pairs, leftimg_dir, target_size=(2048, 1024)):
    """Compute mean and std of images"""
    print(f"Computing image statistics mean and std at size of {target_size[0]}x{target_size[1]}...")

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for city, basename in tqdm(pairs):
        img_file = f'{basename}_leftImg8bit.png'
        img_path = os.path.join(leftimg_dir, city, img_file)

        if os.path.exists(img_path):
            img = Image.open(img_path)

            # Resize if needed (for now, keep original)
            if img.size != (target_size[0], target_size[1]):
                img = img.resize((target_size[0], target_size[1]), Image.BILINEAR)

            img = np.array(img).astype(np.float64) / 255.0

            pixel_sum += img.sum(axis=(0, 1))
            pixel_squared_sum += (img ** 2).sum(axis=(0, 1))
            pixel_count += img.shape[0] * img.shape[1]

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_squared_sum / pixel_count - mean ** 2)

    return mean, std


def save_split_files(output_dir, train_pairs, val_pairs, test_pairs):
    """Save train/val/test file lists"""
    os.makedirs(output_dir, exist_ok=True)

    # Format: city/basename (e.g., aachen/aachen_000000_000019)
    def format_pair(pair):
        return f"{pair[0]}/{pair[1]}"

    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join([format_pair(p) for p in train_pairs]))

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join([format_pair(p) for p in val_pairs]))

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join([format_pair(p) for p in test_pairs]))

    print(f"Split files saved to {output_dir}")


def save_dataset_info(output_dir, label_id_to_train_id, class_names, mean, std,
                     class_counts, class_weights, target_size, color_mapping):
    """Save dataset metadata"""
    info = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'ignore_index': 255,
        'label_id_to_train_id': label_id_to_train_id,
        'color_mapping': {str(k): v for k, v in color_mapping.items()},
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
    print("Cityscapes Dataset Preparation (19 Classes)")
    print("=" * 60)
    print(f"Images will be kept at original size {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    # Get color mapping
    print("\n1. Loading official Cityscapes 34 -> 19 class mapping...")
    color_mapping = get_color_mapping()
    print(f"   Found {len(TRAINING_CLASSES)} classes: {', '.join(TRAINING_CLASSES)}")

    # Find image pairs
    print("\n2. Finding image-label pairs...")
    train_pairs = find_image_pairs(LEFTIMG_DIR, GTFINE_DIR, 'train')
    val_pairs = find_image_pairs(LEFTIMG_DIR, GTFINE_DIR, 'val')
    print(f"   Found {len(train_pairs)} training pairs")
    print(f"   Found {len(val_pairs)} validation pairs")

    # Create splits
    print("\n3. Creating train/val/test splits...")
    print(f"   Splitting train: {TRAIN_RATIO*100:.0f}% train, {TEST_RATIO*100:.0f}% test")
    new_train_pairs, val_pairs, test_pairs = create_splits(
        train_pairs, val_pairs, TRAIN_RATIO, RANDOM_SEED
    )
    print(f"   Train: {len(new_train_pairs)} images")
    print(f"   Val:   {len(val_pairs)} images")
    print(f"   Test:  {len(test_pairs)} images")

    # Save split files first
    print("\n4. Saving split files...")
    save_split_files(OUTPUT_DIR, new_train_pairs, val_pairs, test_pairs)

    # Compute statistics on training set
    print("\n5. Computing dataset statistics...")
    # For statistics, use train split images
    train_leftimg_dir = os.path.join(LEFTIMG_DIR, 'train')
    mean, std = compute_image_statistics(new_train_pairs, train_leftimg_dir, TARGET_SIZE)
    print(f"   Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"   Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    # Compute class distribution
    print("\n6. Computing class distribution...")
    train_gtfine_dir = os.path.join(GTFINE_DIR, 'train')
    class_counts = compute_class_distribution(
        new_train_pairs, train_gtfine_dir, len(TRAINING_CLASSES)
    )

    # Compute class weights using Median Frequency Balancing
    print("\n7. Computing class weights (Median Frequency Balancing)...")
    class_weights = compute_class_weights(class_counts)

    # Show class distribution
    print("\n   Class distribution (training set):")
    total_pixels = class_counts.sum()
    for idx, (name, count, weight) in enumerate(zip(TRAINING_CLASSES, class_counts, class_weights)):
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"   {idx:2d}. {name:20s}: {count:10d} ({percentage:5.2f}%) weight: {weight:.4f}")

    # Save everything
    print("\n8. Saving dataset info...")
    save_dataset_info(OUTPUT_DIR, LABEL_ID_TO_TRAIN_ID, TRAINING_CLASSES, mean, std,
                     class_counts, class_weights, TARGET_SIZE, color_mapping)

    print("\n" + "=" * 60)
    print("Preparation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
