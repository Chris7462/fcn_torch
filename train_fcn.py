"""
FCN Training Script for CamVid Dataset
Following original FCN paper training settings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

from create_dataloaders import create_dataloaders
from fcn import create_fcn_model
from metrics import batch_iou, batch_pixel_acc


# ================== Configuration ==================
# Paths
RAW_IMAGE_DIR = './CamVid/701_StillsRaw_full'
LABEL_DIR = './CamVid/LabeledApproved_full'
SPLITS_DIR = './CamVid/splits'
DATASET_INFO_PATH = './CamVid/splits/dataset_info.json'

# Model settings
BACKBONE = 'vgg16'  # 'vgg16', 'resnet50' (TODO), 'efficientnet' (TODO)
N_CLASS = 11

# Training settings (following original FCN paper)
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
STEP_SIZE = 50
GAMMA = 0.5

# Data settings
TARGET_SIZE = (480, 352)  # (width, height) - resize to 480x360 then centercrop to 480x352
NUM_WORKERS = 4

# Output directories
MODEL_DIR = './models'
PLOT_DIR = './plots'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Experiment name
EXPERIMENT_NAME = f"FCNs-{BACKBONE}_batch{BATCH_SIZE}_epoch{EPOCHS}_SGD_lr{LR}_mom{MOMENTUM}_wd{WEIGHT_DECAY}"
print(f"Experiment: {EXPERIMENT_NAME}")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_pixel_accs = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        with torch.no_grad():
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            batch_pix_acc = batch_pixel_acc(preds, targets)
            all_pixel_accs.append(batch_pix_acc)

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_pix_acc:.4f}'})

    avg_loss = running_loss / len(train_loader)
    avg_pixel_acc = np.mean(all_pixel_accs)

    return avg_loss, avg_pixel_acc


def validate(model, val_loader, criterion, device, n_class, ignore_index, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_ious = []
    all_pixel_accs = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Get predictions
            preds = outputs.argmax(dim=1).cpu().numpy()  # (N, H, W)
            targets = masks.cpu().numpy()  # (N, H, W)

            # Calculate metrics (ignore void class)
            batch_ious = batch_iou(preds, targets, n_class)
            batch_pix_acc = batch_pixel_acc(preds, targets)

            all_ious.append(batch_ious)
            all_pixel_accs.append(batch_pix_acc)

    avg_loss = running_loss / len(val_loader)

    # Calculate mean IoU (ignore nan values for classes not present)
    all_ious = np.array(all_ious)  # (num_batches, n_class)
    mean_ious = np.nanmean(all_ious, axis=0)  # (n_class,)
    mean_iou = np.nanmean(mean_ious)

    mean_pixel_acc = np.mean(all_pixel_accs)

    return avg_loss, mean_iou, mean_pixel_acc, mean_ious


def plot_training_history(history, save_dir, experiment_name):
    """
    Plot training and validation loss and metrics curves.

    Args:
        history: Dictionary with keys 'train_loss', 'train_pixel_acc', 'val_loss',
                'val_pixel_acc', 'val_miou'
        save_dir: Directory to save the plot
        experiment_name: Name for the saved file
    """
    import matplotlib
    matplotlib.use('Agg')
    plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    num_epochs = len(history['train_loss'])
    epochs_range = np.arange(1, num_epochs + 1)

    # Plot 1: Loss
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss',
                 marker='o', markersize=3, linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'], label='Val Loss',
                 marker='s', markersize=3, linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch #', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Metrics (Pixel Accuracy and mIoU)
    axes[1].plot(epochs_range, history['train_pixel_acc'], label='Train Pixel Acc',
                 marker='o', markersize=3, linewidth=2)
    axes[1].plot(epochs_range, history['val_pixel_acc'], label='Val Pixel Acc',
                 marker='s', markersize=3, linewidth=2)
    axes[1].plot(epochs_range, history['val_miou'], label='Val mIoU',
                 marker='^', markersize=3, linewidth=2, linestyle='--')
    axes[1].set_title('Training and Validation Metrics', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch #', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f'{experiment_name}_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to: {save_path}")
    plt.close()


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        raw_image_dir=RAW_IMAGE_DIR,
        label_dir=LABEL_DIR,
        splits_dir=SPLITS_DIR,
        dataset_info_path=DATASET_INFO_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        target_size=TARGET_SIZE
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    class_weights = torch.tensor(dataloaders['class_weights'], dtype=torch.float32).to(device)
    num_classes = dataloaders['num_classes']
    ignore_index = dataloaders['ignore_index']

    print(f"Number of classes: {num_classes}")
    print(f"Ignore index: {ignore_index}")

    # Create model
    print(f"\nCreating FCN model with {BACKBONE} backbone...")
    model = create_fcn_model(n_class=num_classes, backbone=BACKBONE, pretrained=True)
    model = model.to(device)

    # Setup loss and optimizer (following original FCN paper)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    print(f"\nOptimizer: SGD")
    print(f"  Learning rate: {LR}")
    print(f"  Momentum: {MOMENTUM}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  LR Scheduler: StepLR (step_size={STEP_SIZE}, gamma={GAMMA})")

    # Training history
    history = {
        'train_loss': [],
        'train_pixel_acc': [],
        'val_loss': [],
        'val_miou': [],
        'val_pixel_acc': []
    }

    best_miou = 0.0

    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    # Training loop
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss, train_pixel_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_miou, val_pixel_acc, class_ious = validate(
            model, val_loader, criterion, device, num_classes, ignore_index, epoch
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        history['val_pixel_acc'].append(val_pixel_acc)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Pixel Acc: {train_pixel_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val mIoU: {val_miou:.4f}")
        print(f"  Val Pixel Acc: {val_pixel_acc:.4f}")

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            model_path = os.path.join(MODEL_DIR, f'{EXPERIMENT_NAME}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'history': history
            }, model_path)
            print(f"  ✓ Best model saved! (mIoU: {best_miou:.4f})")

        # Plot training history after each epoch
        plot_training_history(history, PLOT_DIR, EXPERIMENT_NAME)

        print("-" * 80)

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation mIoU: {best_miou:.4f}")
    print("="*80)

    # Save final model
    final_model_path = os.path.join(MODEL_DIR, f'{EXPERIMENT_NAME}_final.pth')
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_miou': best_miou,
        'history': history
    }, final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")

    # Save history as numpy
    history_path = os.path.join(PLOT_DIR, f'{EXPERIMENT_NAME}_history.npz')
    np.savez(history_path, **history)
    print(f"✓ Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
