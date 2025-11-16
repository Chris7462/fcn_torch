"""
Test FCN Model on CamVid Test Dataset
Evaluates a trained model and generates predictions
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
from metrics import batch_iou, batch_pixel_acc


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
    all_ious = []
    all_pixel_accs = []

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

            # Calculate metrics
            batch_ious = batch_iou(preds, targets, n_class)
            batch_pix_acc = batch_pixel_acc(preds, targets)

            all_ious.append(batch_ious)
            all_pixel_accs.append(batch_pix_acc)

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

    # Calculate final metrics
    avg_loss = running_loss / len(test_loader)

    # Calculate mean IoU per class
    all_ious = np.array(all_ious)  # (num_batches, n_class)
    mean_ious = np.nanmean(all_ious, axis=0)  # (n_class,)
    mean_iou = np.nanmean(mean_ious)

    mean_pixel_acc = np.mean(all_pixel_accs)

    # Print results
    print("\n" + "="*80)
    print("Test Results")
    print("="*80)
    print(f"Test Loss:       {avg_loss:.4f}")
    print(f"Test mIoU:       {mean_iou:.4f}")
    print(f"Test Pixel Acc:  {mean_pixel_acc:.4f}")
    print("\nPer-class IoU:")
    for idx, (name, iou) in enumerate(zip(class_names, mean_ious)):
        print(f"  {idx:2d}. {name:15s}: {iou:.4f}")
    print("="*80)

    print(f"\n✓ Predictions saved to: {pred_dir}")
    print(f"✓ Visualizations saved to: {vis_dir}")

    return {
        'test_loss': avg_loss,
        'test_miou': mean_iou,
        'test_pixel_acc': mean_pixel_acc,
        'per_class_iou': mean_ious.tolist()
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
    print("\nCreating model...")
    model = create_fcn_model(n_class=num_classes, backbone='vgg16', pretrained=False)
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


if __name__ == '__main__':
    main()
