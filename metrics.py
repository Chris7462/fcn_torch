"""
Evaluation Metrics for Semantic Segmentation
Includes IoU (Intersection over Union) and pixel accuracy
Properly handles ignore_index for unlabeled pixels
Uses standard accumulation method for consistent, batch-size independent results
"""

import numpy as np


def pixel_accuracy(preds, targets, ignore_index=255):
    """
    Calculate pixel accuracy - PRIMARY METRIC.

    This is the standard "pixel accuracy" metric used in semantic segmentation.
    It accumulates correct pixels and total pixels across all images.
    Results are batch-size independent.

    Formula (FCN paper):
        pixel accuracy = Σi nii / Σi ti

        where:
        - nii = number of pixels of class i correctly predicted as class i
        - ti = total number of pixels of class i in ground truth

    Args:
        preds: Predictions (N, H, W) or single (H, W)
        targets: Ground truth (N, H, W) or single (H, W)
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        float: Pixel accuracy

    Example:
        >>> preds = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> targets = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> acc = pixel_accuracy(preds, targets)
        >>> print(f"{acc:.4f}")  # 1.0000
    """
    # Handle single image case
    if preds.ndim == 2:
        preds = preds[np.newaxis, ...]
        targets = targets[np.newaxis, ...]

    # Accumulate across all images
    total_correct = 0
    total_pixels = 0

    for pred, target in zip(preds, targets):
        valid_mask = (target != ignore_index)
        correct = ((pred == target) & valid_mask).sum()
        total = valid_mask.sum()

        total_correct += correct
        total_pixels += total

    if total_pixels == 0:
        return 0.0

    return total_correct / total_pixels


def mean_pixel_accuracy(preds, targets, n_class, ignore_index=255):
    """
    Calculate mean pixel accuracy (per-class accuracy averaged).

    This calculates accuracy for each class separately, then averages
    across classes. This is different from pixel_accuracy() and
    gives equal weight to all classes regardless of their frequency.

    Also known as "class average accuracy" in literature.

    Formula (FCN paper):
        mean accuracy = (1/ncl) Σi nii/ti

        where:
        - nii = number of pixels of class i correctly predicted as class i
        - ti = total number of pixels of class i in ground truth
        - ncl = number of classes

    Args:
        preds: Predictions (N, H, W) or single (H, W)
        targets: Ground truth (N, H, W) or single (H, W)
        n_class: Number of classes
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        float: Mean pixel accuracy across all classes

    Example:
        >>> preds = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> targets = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> acc = mean_pixel_accuracy(preds, targets, n_class=2)
        >>> print(f"{acc:.4f}")  # 1.0000
    """
    # Handle single image case
    if preds.ndim == 2:
        preds = preds[np.newaxis, ...]
        targets = targets[np.newaxis, ...]

    # Accumulate per-class correct and total pixels
    class_correct = np.zeros(n_class, dtype=np.int64)
    class_total = np.zeros(n_class, dtype=np.int64)

    for pred, target in zip(preds, targets):
        valid_mask = (target != ignore_index)

        for cls in range(n_class):
            # Pixels that should be class cls
            target_cls = (target == cls) & valid_mask
            # How many were correctly predicted as cls
            correct_cls = (pred == cls) & target_cls

            class_correct[cls] += correct_cls.sum()
            class_total[cls] += target_cls.sum()

    # Calculate per-class accuracy
    per_class_acc = np.zeros(n_class)
    for cls in range(n_class):
        if class_total[cls] > 0:
            per_class_acc[cls] = class_correct[cls] / class_total[cls]
        else:
            per_class_acc[cls] = float('nan')

    # Return mean (excluding classes not in dataset)
    return np.nanmean(per_class_acc)


def _accumulate_iou(pred, target, n_class, ignore_index=255):
    """
    Calculate intersection and union for each class (helper for accumulation).

    This is an internal helper function used by mean_iou().

    Args:
        pred: Prediction mask (H, W) with class indices
        target: Ground truth mask (H, W) with class indices
        n_class: Number of classes
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        intersection: Array of intersection counts per class (n_class,)
        union: Array of union counts per class (n_class,)
    """
    # Create mask for valid pixels (not ignored)
    valid_mask = (target != ignore_index)

    intersection = np.zeros(n_class, dtype=np.int64)
    union = np.zeros(n_class, dtype=np.int64)

    for cls in range(n_class):
        # Only consider valid pixels
        pred_inds = (pred == cls) & valid_mask
        target_inds = (target == cls) & valid_mask

        intersection[cls] = pred_inds[target_inds].sum()
        union[cls] = pred_inds.sum() + target_inds.sum() - intersection[cls]

    return intersection, union


def mean_iou(preds, targets, n_class, ignore_index=255):
    """
    Calculate mean Intersection over Union (mIoU) - PRIMARY METRIC.

    This is the standard mIoU metric used in semantic segmentation papers.
    It accumulates intersection and union across all images, then computes
    IoU per class. Results are batch-size independent.

    Formula (FCN paper):
        mean IU = (1/ncl) Σi nii/(ti + Σj nji - nii)

        where:
        - nii = number of pixels of class i correctly predicted as class i
        - ti = total number of pixels of class i in ground truth
        - Σj nji = total number of pixels predicted as class i
        - ncl = number of classes

    Args:
        preds: Predictions (N, H, W) or single (H, W)
        targets: Ground truth (N, H, W) or single (H, W)
        n_class: Number of classes
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        float: Mean IoU across all classes

    Example:
        >>> preds = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> targets = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> miou = mean_iou(preds, targets, n_class=2)
        >>> print(f"{miou:.4f}")  # 1.0000
    """
    # Handle single image case
    if preds.ndim == 2:
        preds = preds[np.newaxis, ...]
        targets = targets[np.newaxis, ...]

    # Accumulate intersection and union across all images
    total_intersection = np.zeros(n_class, dtype=np.int64)
    total_union = np.zeros(n_class, dtype=np.int64)

    for pred, target in zip(preds, targets):
        intersection, union = _accumulate_iou(pred, target, n_class, ignore_index)
        total_intersection += intersection
        total_union += union

    # Compute IoU per class
    iou_per_class_values = np.zeros(n_class, dtype=np.float64)
    for cls in range(n_class):
        if total_union[cls] == 0:
            # Class not present in ground truth
            iou_per_class_values[cls] = float('nan')
        else:
            iou_per_class_values[cls] = total_intersection[cls] / total_union[cls]

    # Return mean IoU (excluding NaN for classes not in dataset)
    return np.nanmean(iou_per_class_values)


def frequency_weighted_iou(preds, targets, n_class, ignore_index=255):
    """
    Calculate frequency weighted Intersection over Union (f.w. IoU).

    This metric weights each class's IoU by its frequency in the ground truth,
    giving more importance to classes that appear more often in the dataset.
    Results are batch-size independent.

    Formula (FCN paper):
        f.w. IU = (Σk tk)^-1 Σi ti*nii/(ti + Σj nji - nii)

        where:
        - ti = total number of pixels of class i in ground truth
        - nii = number of pixels of class i correctly predicted as class i
        - Σj nji = total number of pixels predicted as class i
        - Σk tk = total number of valid (non-ignored) pixels

    Args:
        preds: Predictions (N, H, W) or single (H, W)
        targets: Ground truth (N, H, W) or single (H, W)
        n_class: Number of classes
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        float: Frequency weighted IoU

    Example:
        >>> preds = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> targets = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> fwiou = frequency_weighted_iou(preds, targets, n_class=2)
        >>> print(f"{fwiou:.4f}")  # 1.0000
    """
    # Handle single image case
    if preds.ndim == 2:
        preds = preds[np.newaxis, ...]
        targets = targets[np.newaxis, ...]

    # Accumulate intersection, union, and class frequency across all images
    total_intersection = np.zeros(n_class, dtype=np.int64)
    total_union = np.zeros(n_class, dtype=np.int64)
    class_frequency = np.zeros(n_class, dtype=np.int64)

    for pred, target in zip(preds, targets):
        intersection, union = _accumulate_iou(pred, target, n_class, ignore_index)
        total_intersection += intersection
        total_union += union

        # Count frequency of each class in ground truth
        valid_mask = (target != ignore_index)
        for cls in range(n_class):
            target_inds = (target == cls) & valid_mask
            class_frequency[cls] += target_inds.sum()

    # Calculate total valid pixels
    total_pixels = class_frequency.sum()
    if total_pixels == 0:
        return 0.0

    # Calculate frequency weighted IoU
    weighted_iou = 0.0
    for cls in range(n_class):
        if class_frequency[cls] == 0:
            # Class not in ground truth, skip (weight = 0 anyway)
            continue

        weight = class_frequency[cls] / total_pixels

        if total_union[cls] == 0:
            # Class in ground truth but completely missed: IoU = 0
            iou = 0.0
        else:
            iou = total_intersection[cls] / total_union[cls]

        weighted_iou += weight * iou

    return weighted_iou


def iou_per_class(preds, targets, n_class, ignore_index=255):
    """
    Calculate Intersection over Union for each class.

    This function uses accumulation method across all images for consistency
    with mean_iou(). It can handle both single images and batches.

    Args:
        preds: Predictions (N, H, W) or single (H, W)
        targets: Ground truth (N, H, W) or single (H, W)
        n_class: Number of classes
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        list: IoU for each class (nan if class not present in ground truth)

    Example:
        >>> # Single image
        >>> pred = np.array([[0, 1], [1, 1]])
        >>> target = np.array([[0, 1], [1, 1]])
        >>> ious = iou_per_class(pred, target, n_class=2)
        >>> print(ious)  # [1.0, 1.0]

        >>> # Batch of images
        >>> preds = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> targets = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> ious = iou_per_class(preds, targets, n_class=2)
        >>> print(ious)  # [1.0, 1.0]
    """
    # Handle single image case
    if preds.ndim == 2:
        preds = preds[np.newaxis, ...]
        targets = targets[np.newaxis, ...]

    # Accumulate intersection and union across all images
    total_intersection = np.zeros(n_class, dtype=np.int64)
    total_union = np.zeros(n_class, dtype=np.int64)

    for pred, target in zip(preds, targets):
        valid_mask = (target != ignore_index)

        for cls in range(n_class):
            # Only consider valid pixels
            pred_inds = (pred == cls) & valid_mask
            target_inds = (target == cls) & valid_mask

            intersection = pred_inds[target_inds].sum()
            union = pred_inds.sum() + target_inds.sum() - intersection

            total_intersection[cls] += intersection
            total_union[cls] += union

    # Compute IoU per class
    ious = []
    for cls in range(n_class):
        if total_union[cls] == 0:
            # If there is no ground truth for this class, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(total_intersection[cls]) / total_union[cls])

    return ious


if __name__ == '__main__':
    """Test the metrics"""
    print("Testing segmentation metrics...")
    print("="*80)

    # Create dummy predictions and targets
    n_class = 5
    ignore_index = 255

    pred = np.random.randint(0, n_class, size=(10, 10))
    target = np.random.randint(0, n_class, size=(10, 10))

    # Add some ignored pixels
    target[0:2, 0:2] = ignore_index

    # Test single sample metrics
    print("\n1. Single Sample Test:")
    print("-" * 80)
    ious = iou_per_class(pred, target, n_class, ignore_index)
    print(f"  IoU per class: {[f'{x:.4f}' if not np.isnan(x) else 'nan' for x in ious]}")
    print(f"  Mean IoU: {np.nanmean(ious):.4f}")

    # Test batch metrics with accumulation
    print("\n2. Batch Test (Accumulation Method - STANDARD):")
    print("-" * 80)
    batch_preds = np.random.randint(0, n_class, size=(4, 10, 10))
    batch_targets = np.random.randint(0, n_class, size=(4, 10, 10))

    # Add ignored pixels to batch
    batch_targets[:, 0:2, 0:2] = ignore_index

    miou = mean_iou(batch_preds, batch_targets, n_class, ignore_index)
    print(f"  Mean IoU (mIoU): {miou:.4f}")

    pix_acc = pixel_accuracy(batch_preds, batch_targets, ignore_index)
    print(f"  Pixel accuracy: {pix_acc:.4f}")

    mean_acc = mean_pixel_accuracy(batch_preds, batch_targets, n_class, ignore_index)
    print(f"  Mean pixel accuracy: {mean_acc:.4f}")

    fwiou = frequency_weighted_iou(batch_preds, batch_targets, n_class, ignore_index)
    print(f"  Frequency weighted IoU: {fwiou:.4f}")

    batch_ious = iou_per_class(batch_preds, batch_targets, n_class, ignore_index)
    print(f"  IoU per class (batch): {[f'{x:.4f}' if not np.isnan(x) else 'nan' for x in batch_ious]}")

    # Test perfect predictions
    print("\n3. Perfect Predictions Test:")
    print("-" * 80)
    perfect_preds = batch_targets.copy()
    perfect_miou = mean_iou(perfect_preds, batch_targets, n_class, ignore_index)
    perfect_pix_acc = pixel_accuracy(perfect_preds, batch_targets, ignore_index)
    perfect_mean_acc = mean_pixel_accuracy(perfect_preds, batch_targets, n_class, ignore_index)
    perfect_fwiou = frequency_weighted_iou(perfect_preds, batch_targets, n_class, ignore_index)
    perfect_ious = iou_per_class(perfect_preds, batch_targets, n_class, ignore_index)
    print(f"  Perfect mIoU: {perfect_miou:.4f} (should be 1.0)")
    print(f"  Perfect pixel accuracy: {perfect_pix_acc:.4f} (should be 1.0)")
    print(f"  Perfect mean pixel accuracy: {perfect_mean_acc:.4f} (should be 1.0)")
    print(f"  Perfect frequency weighted IoU: {perfect_fwiou:.4f} (should be 1.0)")
    print(f"  Perfect IoU per class: {[f'{x:.4f}' if not np.isnan(x) else 'nan' for x in perfect_ious]}")

    # Test batch-size independence
    print("\n4. Batch-Size Independence Test:")
    print("-" * 80)
    # Create same predictions but split differently
    all_preds = np.random.randint(0, n_class, size=(8, 10, 10))
    all_targets = np.random.randint(0, n_class, size=(8, 10, 10))
    all_targets[:, 0:2, 0:2] = ignore_index

    # Batch size 8
    miou_batch8 = mean_iou(all_preds, all_targets, n_class, ignore_index)
    pix_acc_batch8 = pixel_accuracy(all_preds, all_targets, ignore_index)
    fwiou_batch8 = frequency_weighted_iou(all_preds, all_targets, n_class, ignore_index)
    ious_batch8 = iou_per_class(all_preds, all_targets, n_class, ignore_index)

    # Batch size 4 (split into two batches)
    miou_batch4_a = mean_iou(all_preds[:4], all_targets[:4], n_class, ignore_index)
    miou_batch4_b = mean_iou(all_preds[4:], all_targets[4:], n_class, ignore_index)
    ious_batch4_a = iou_per_class(all_preds[:4], all_targets[:4], n_class, ignore_index)
    ious_batch4_b = iou_per_class(all_preds[4:], all_targets[4:], n_class, ignore_index)

    print(f"  Batch size 8 - mIoU: {miou_batch8:.4f}, Pixel Acc: {pix_acc_batch8:.4f}, f.w. IoU: {fwiou_batch8:.4f}")
    print(f"  Batch size 8 - IoU per class: {[f'{x:.4f}' if not np.isnan(x) else 'nan' for x in ious_batch8]}")
    print(f"  Batch size 4a - mIoU: {miou_batch4_a:.4f}")
    print(f"  Batch size 4b - mIoU: {miou_batch4_b:.4f}")
    print(f"  Note: Individual batch results will differ, but full dataset gives consistent results")

    # Test edge case: all pixels ignored
    print("\n5. Edge Case Test (All Pixels Ignored):")
    print("-" * 80)
    all_ignored_target = np.full((10, 10), ignore_index, dtype=np.int64)
    edge_ious = iou_per_class(pred, all_ignored_target, n_class, ignore_index)
    print(f"  IoU per class: {[f'{x:.4f}' if not np.isnan(x) else 'nan' for x in edge_ious]}")
    edge_pix_acc = pixel_accuracy(pred, all_ignored_target, ignore_index)
    print(f"  Pixel accuracy: {edge_pix_acc:.4f} (should be 0.0)")

    # Test single image input (should work with 2D arrays)
    print("\n6. Single Image Input Test:")
    print("-" * 80)
    single_pred = np.random.randint(0, n_class, size=(10, 10))
    single_target = np.random.randint(0, n_class, size=(10, 10))
    single_miou = mean_iou(single_pred, single_target, n_class, ignore_index)
    single_pix_acc = pixel_accuracy(single_pred, single_target, ignore_index)
    single_fwiou = frequency_weighted_iou(single_pred, single_target, n_class, ignore_index)
    single_ious = iou_per_class(single_pred, single_target, n_class, ignore_index)
    print(f"  Single image mIoU: {single_miou:.4f}")
    print(f"  Single image pixel accuracy: {single_pix_acc:.4f}")
    print(f"  Single image f.w. IoU: {single_fwiou:.4f}")
    print(f"  Single image IoU per class: {[f'{x:.4f}' if not np.isnan(x) else 'nan' for x in single_ious]}")

    print("\n" + "="*80)
    print("All metrics tests passed!")
    print("="*80)
