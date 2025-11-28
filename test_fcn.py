"""
Test FCN Model on CamVid Test Dataset
Evaluates a trained model and generates predictions
Updated to use all 4 FCN metrics and per-class IoU
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from create_camvid_dataloaders import create_camvid_dataloaders
from fcn import create_fcn_model
from metrics import pixel_accuracy, mean_pixel_accuracy, mean_iou, frequency_weighted_iou, iou_per_class


# Model settings
BACKBONE = 'resnet101'  # 'vgg16', 'resnet50' (TODO), 'resnet101', 'efficientnet' (TODO)


def get_color_mapping():
    """
    Returns the RGB color for each of the 11 classes
    Same mapping used in prepare_camvid_data.py
    """
    class_colors = {
        0: (128, 128, 128),   # Sky
        1: (128, 0, 0),       # Building
        2: (192, 192, 128),   # Pole
        3: (128, 64, 128),    # Road
        4: (0, 0, 192),       # Pavement
        5: (128, 128, 0),     # Tree
        6: (192, 128, 128),   # SignSymbol
        7: (64, 64, 128),     # Fence
        8: (64, 0, 128),      # Car
        9: (64, 64, 0),       # Pedestrian
        10: (0, 128, 192),    # Bicyclist
    }
    return class_colors


def index_to_rgb(mask_indexed, color_mapping, ignore_index=255):
    """
    Convert class indices to RGB color-coded image

    Args:
        mask_indexed: (H, W) array with class indices
        color_mapping: dict mapping class index to (R, G, B)
        ignore_index: index to map to black (0, 0, 0)

    Returns:
        RGB image (H, W, 3)
    """
    h, w = mask_indexed.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in color_mapping.items():
        rgb_mask[mask_indexed == class_idx] = color

    # Void pixels remain black (0, 0, 0)
    rgb_mask[mask_indexed == ignore_index] = (0, 0, 0)

    return rgb_mask


def test_model(model, test_loader, criterion, device, n_class, ignore_index,
               output_dir, class_names, color_mapping, mean, std):
    """
    Test the model and save predictions

    Args:
        model: Trained FCN model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run on
        n_class: Number of classes
        ignore_index: Index to ignore in loss
        output_dir: Directory to save outputs
        class_names: List of class names
        color_mapping: Dict mapping class index to RGB color
        mean: Mean used for normalization (list of 3 floats)
        std: Std used for normalization (list of 3 floats)

    Returns:
        Dictionary with test metrics
    """
    model.eval()

    # Create output directories
    pred_dir = os.path.join(output_dir, 'predictions')
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    running_loss = 0.0

    # Accumulate predictions and targets for metrics
    all_preds = []
    all_targets = []

    print("\nRunning inference on test set...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Get predictions
            preds = outputs.argmax(dim=1).cpu().numpy()  # (N, H, W)
            targets = masks.cpu().numpy()  # (N, H, W)
            images_np = images.cpu().numpy()  # (N, 3, H, W)

            # Accumulate for metrics calculation
            all_preds.append(preds)
            all_targets.append(targets)

            # Save predictions for each image in batch
            for i in range(len(filenames)):
                filename = filenames[i]
                pred_mask = preds[i]  # (H, W)
                target_mask = targets[i]  # (H, W)
                img = images_np[i]  # (3, H, W)

                # Convert prediction to RGB
                pred_rgb = index_to_rgb(pred_mask, color_mapping, ignore_index)

                # Save color-coded prediction
                pred_filename = filename[:-4] + '_pred.png'
                pred_path = os.path.join(pred_dir, pred_filename)
                Image.fromarray(pred_rgb).save(pred_path)

                # Create side-by-side visualization
                # Denormalize image: img = (img * std) + mean
                img = img.transpose(1, 2, 0)  # (H, W, 3)

                # Reverse normalization
                mean_np = np.array(mean).reshape(1, 1, 3)
                std_np = np.array(std).reshape(1, 1, 3)
                img = (img * std_np) + mean_np

                # Convert to uint8 [0, 255]
                img = (img * 255).astype(np.uint8)
                img = np.clip(img, 0, 255)

                # Convert target to RGB
                target_rgb = index_to_rgb(target_mask, color_mapping, ignore_index)

                # Create side-by-side figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img)
                axes[0].set_title('Input Image')
                axes[0].axis('off')

                axes[1].imshow(target_rgb)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred_rgb)
                axes[2].set_title('Prediction')
                axes[2].axis('off')

                plt.tight_layout()

                # Save visualization
                vis_filename = filename[:-4] + '_comparison.png'
                vis_path = os.path.join(vis_dir, vis_filename)
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()

    # Calculate final metrics on accumulated predictions
    avg_loss = running_loss / len(test_loader)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mean_iou_val = mean_iou(all_preds, all_targets, n_class, ignore_index)
    pixel_acc = pixel_accuracy(all_preds, all_targets, ignore_index)
    mean_acc = mean_pixel_accuracy(all_preds, all_targets, n_class, ignore_index)
    fwiou = frequency_weighted_iou(all_preds, all_targets, n_class, ignore_index)
    # Compute per-class IoU using the extended function from metrics
    per_class_iou = iou_per_class(all_preds, all_targets, n_class, ignore_index)
    # Print results
    print("\n" + "="*80)
    print("Test Results")
    print("="*80)
    print(f"Test Loss:                    {avg_loss:.4f}")
    print(f"Mean IoU (mIoU):              {mean_iou_val:.4f}")
    print(f"Pixel Accuracy:               {pixel_acc:.4f}")
    print(f"Mean Pixel Accuracy:          {mean_acc:.4f}")
    print(f"Frequency Weighted IoU:       {fwiou:.4f}")
    print("="*80)

    # Print per-class IoU
    print(f"\nPer-Class IoU:")
    for cls_id, (cls_name, cls_iou) in enumerate(zip(class_names, per_class_iou)):
        if np.isnan(cls_iou):
            print(f"  {cls_id:2d}. {cls_name:20s}: N/A (not in dataset)")
        else:
            print(f"  {cls_id:2d}. {cls_name:20s}: {cls_iou:.4f}")

    print("\n" + "="*80)

    # Interpretation guidance
    if pixel_acc > mean_acc + 0.05:
        print("\nNote: Pixel Accuracy >> Mean Pixel Accuracy")
        print("  This suggests the model performs well on frequent classes")
        print("  but may struggle with rare classes.")
        print()

    print(f"\nPredictions saved to: {pred_dir}")
    print(f"Visualizations saved to: {vis_dir}")

    return {
        'test_loss': avg_loss,
        'test_miou': mean_iou_val,
        'test_pixel_acc': pixel_acc,
        'test_mean_acc': mean_acc,
        'test_fwiou': fwiou,
        'per_class_iou': {name: float(iou) if not np.isnan(iou) else None 
                          for name, iou in zip(class_names, per_class_iou)}
    }


def main():
    parser = argparse.ArgumentParser(description='Test FCN model on CamVid test dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for testing (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs (default: ./outputs)')
    args = parser.parse_args()

    # Paths (same as train_fcn.py)
    RAW_IMAGE_DIR = './CamVid/701_StillsRaw_full'
    LABEL_DIR = './CamVid/LabeledApproved_full'
    SPLITS_DIR = './CamVid/splits'
    DATASET_INFO_PATH = './CamVid/splits/dataset_info.json'
    TARGET_SIZE = (480, 352)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Load mean and std from dataset info
    import json
    with open(DATASET_INFO_PATH, 'r') as f:
        dataset_info = json.load(f)
    mean = dataset_info.get('mean', [0.485, 0.456, 0.406])
    std = dataset_info.get('std', [0.229, 0.224, 0.225])

    # Create dataloaders
    print("\nCreating test dataloader...")
    dataloaders = create_camvid_dataloaders(
        raw_image_dir=RAW_IMAGE_DIR,
        label_dir=LABEL_DIR,
        splits_dir=SPLITS_DIR,
        dataset_info_path=DATASET_INFO_PATH,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=TARGET_SIZE
    )

    test_loader = dataloaders['test']
    class_weights = torch.tensor(dataloaders['class_weights'], dtype=torch.float32).to(device)
    num_classes = dataloaders['num_classes']
    class_names = dataloaders['class_names']
    ignore_index = dataloaders['ignore_index']

    print(f"Number of classes: {num_classes}")
    print(f"Test set size: {len(test_loader.dataset)} images")

    # Create model
    print(f"\nCreating FCN model with {BACKBONE} backbone...")
    model = create_fcn_model(n_class=num_classes, backbone=BACKBONE, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Model loaded successfully!")

    # Setup loss
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    # Get color mapping
    color_mapping = get_color_mapping()

    # Create output directory with experiment name
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
    output_dir = os.path.join(args.output_dir, f'{checkpoint_name}_test')
    os.makedirs(output_dir, exist_ok=True)

    # Run test
    results = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        n_class=num_classes,
        ignore_index=ignore_index,
        output_dir=output_dir,
        class_names=class_names,
        color_mapping=color_mapping,
        mean=mean,
        std=std
    )

    # Save results to JSON
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
