"""
Evaluation Metrics for Semantic Segmentation
Includes IoU (Intersection over Union) and pixel accuracy
Properly handles ignore_index for unlabeled pixels
Uses standard accumulation method for consistent, batch-size independent results
"""

import numpy as np


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


def iou_per_class(pred, target, n_class, ignore_index=255):
    """
    Calculate Intersection over Union for each class (single image).

    Args:
        pred: Prediction mask (H, W) with class indices
        target: Ground truth mask (H, W) with class indices
        n_class: Number of classes
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        list: IoU for each class (nan if class not present in ground truth)

    Example:
        >>> pred = np.array([[0, 1], [1, 1]])
        >>> target = np.array([[0, 1], [1, 1]])
        >>> ious = iou_per_class(pred, target, n_class=2)
        >>> print(ious)  # [1.0, 1.0]
    """
    ious = []

    # Create mask for valid pixels (not ignored)
    valid_mask = (target != ignore_index)

    for cls in range(n_class):
        # Only consider valid pixels
        pred_inds = (pred == cls) & valid_mask
        target_inds = (target == cls) & valid_mask

        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection

        if union == 0:
            # If there is no ground truth for this class, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / max(union, 1))

    return ious


def mean_iou(preds, targets, n_class, ignore_index=255):
    """
    Calculate mean Intersection over Union (mIoU) - PRIMARY METRIC.

    This is the standard mIoU metric used in semantic segmentation papers.
    It accumulates intersection and union across all images, then computes
    IoU per class. Results are batch-size independent.

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


def pixel_accuracy(pred, target, ignore_index=255):
    """
    Calculate pixel accuracy for a single image.

    Args:
        pred: Prediction mask (H, W) with class indices
        target: Ground truth mask (H, W) with class indices
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        float: Pixel accuracy

    Example:
        >>> pred = np.array([[0, 1], [1, 1]])
        >>> target = np.array([[0, 1], [1, 1]])
        >>> acc = pixel_accuracy(pred, target)
        >>> print(f"{acc:.4f}")  # 1.0000
    """
    # Only evaluate on valid pixels (not ignored)
    valid_mask = (target != ignore_index)

    correct = ((pred == target) & valid_mask).sum()
    total = valid_mask.sum()

    if total == 0:
        return 0.0

    return correct / total


def global_pixel_accuracy(preds, targets, ignore_index=255):
    """
    Calculate global pixel accuracy - PRIMARY METRIC.

    This is the standard "pixel accuracy" metric used in semantic segmentation.
    It accumulates correct pixels and total pixels across all images.
    Results are batch-size independent.

    Args:
        preds: Predictions (N, H, W) or single (H, W)
        targets: Ground truth (N, H, W) or single (H, W)
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        float: Global pixel accuracy

    Example:
        >>> preds = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> targets = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 1]]])
        >>> acc = global_pixel_accuracy(preds, targets)
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


def class_pixel_accuracy(preds, targets, n_class, ignore_index=255):
    """
    Calculate mean pixel accuracy (per-class accuracy averaged).

    This calculates accuracy for each class separately, then averages
    across classes. This is different from global_pixel_accuracy() and
    gives equal weight to all classes regardless of their frequency.

    Also known as "mean accuracy" or "class average accuracy" in literature.

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
        >>> acc = class_pixel_accuracy(preds, targets, n_class=2)
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
