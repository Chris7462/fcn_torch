"""
Evaluation Metrics for Semantic Segmentation
Adapted from the original FCN implementation
"""

import numpy as np


def iou(pred, target, n_class):
    """
    Calculate class intersections over unions

    Args:
        pred: prediction mask (H, W) with class indices
        target: ground truth mask (H, W) with class indices
        n_class: number of classes

    Returns:
        list: IoU for each class (nan if class not present in ground truth)
    """
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    """
    Calculate pixel accuracy

    Args:
        pred: prediction mask (H, W) with class indices
        target: ground truth mask (H, W) with class indices

    Returns:
        float: pixel accuracy
    """
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


def batch_iou(preds, targets, n_class):
    """
    Calculate IoU for a batch of predictions

    Args:
        preds: batch of predictions (N, H, W)
        targets: batch of ground truth (N, H, W)
        n_class: number of classes

    Returns:
        numpy array: mean IoU per class across the batch (n_class,)
    """
    total_ious = []
    for pred, target in zip(preds, targets):
        total_ious.append(iou(pred, target, n_class))

    # n_class * batch_size
    total_ious = np.array(total_ious).T
    # Average across batch for each class
    ious = np.nanmean(total_ious, axis=1)
    return ious


def batch_pixel_acc(preds, targets):
    """
    Calculate pixel accuracy for a batch

    Args:
        preds: batch of predictions (N, H, W)
        targets: batch of ground truth (N, H, W)

    Returns:
        float: mean pixel accuracy across the batch
    """
    pixel_accs = []
    for pred, target in zip(preds, targets):
        pixel_accs.append(pixel_acc(pred, target))
    return np.array(pixel_accs).mean()
