"""
Evaluation Metrics for Semantic Segmentation
Adapted from the original FCN implementation
Updated to properly handle ignore_index
"""

import numpy as np


def iou(pred, target, n_class, ignore_index=255):
    """
    Calculate class intersections over unions

    Args:
        pred: prediction mask (H, W) with class indices
        target: ground truth mask (H, W) with class indices
        n_class: number of classes
        ignore_index: index to ignore in evaluation (default: 255)

    Returns:
        list: IoU for each class (nan if class not present in ground truth)
    """
    ious = []

    # Create mask for valid pixels (not ignored)
    valid_mask = target != ignore_index

    for cls in range(n_class):
        # Only consider valid pixels
        pred_inds = (pred == cls) & valid_mask
        target_inds = (target == cls) & valid_mask

        intersection = (pred_inds & target_inds).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection

        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))

    return ious


def pixel_acc(pred, target, ignore_index=255):
    """
    Calculate pixel accuracy

    Args:
        pred: prediction mask (H, W) with class indices
        target: ground truth mask (H, W) with class indices
        ignore_index: index to ignore in evaluation (default: 255)

    Returns:
        float: pixel accuracy
    """
    # Create mask for valid pixels (not ignored)
    valid_mask = target != ignore_index

    # Only count correct predictions on valid pixels
    correct = ((pred == target) & valid_mask).sum()
    total = valid_mask.sum()

    if total == 0:
        return 0.0

    return correct / total


def batch_iou(preds, targets, n_class, ignore_index=255):
    """
    Calculate IoU for a batch of predictions

    Args:
        preds: batch of predictions (N, H, W)
        targets: batch of ground truth (N, H, W)
        n_class: number of classes
        ignore_index: index to ignore in evaluation (default: 255)

    Returns:
        numpy array: mean IoU per class across the batch (n_class,)
    """
    total_ious = []
    for pred, target in zip(preds, targets):
        total_ious.append(iou(pred, target, n_class, ignore_index))

    # n_class * batch_size
    total_ious = np.array(total_ious).T
    # Average across batch for each class
    ious = np.nanmean(total_ious, axis=1)
    return ious


def batch_pixel_acc(preds, targets, ignore_index=255):
    """
    Calculate pixel accuracy for a batch

    Args:
        preds: batch of predictions (N, H, W)
        targets: batch of ground truth (N, H, W)
        ignore_index: index to ignore in evaluation (default: 255)

    Returns:
        float: mean pixel accuracy across the batch
    """
    pixel_accs = []
    for pred, target in zip(preds, targets):
        pixel_accs.append(pixel_acc(pred, target, ignore_index))
    return np.array(pixel_accs).mean()
