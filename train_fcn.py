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
N_CLASS = 32

# Training settings (following original FCN paper)
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
STEP_SIZE = 50
GAMMA = 0.5

# Data settings
TARGET_SIZE = (960, 704)  # (width, height) - use (480, 360) for faster training
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

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / len(train_loader)
    return avg_loss


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
    """Plot and save training history"""
    epochs = range(1, len(history['train_loss']) + 1)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Validation Loss
    axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Validation mIoU
    axes[1, 0].plot(epochs, history['val_miou'], 'g-', label='Val mIoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mIoU')
    axes[1, 0].set_title('Validation Mean IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Validation Pixel Accuracy
    axes[1, 1].plot(epochs, history['val_pixel_acc'], 'm-', label='Val Pixel Acc')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Pixel Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f'{experiment_name}_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
        target_size=TARGET_SIZE,
        use_computed_stats=True
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
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss, val_miou, val_pixel_acc, class_ious = validate(
            model, val_loader, criterion, device, num_classes, ignore_index, epoch
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        history['val_pixel_acc'].append(val_pixel_acc)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}")
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

    # Plot and save training history
    plot_training_history(history, PLOT_DIR, EXPERIMENT_NAME)

    # Save history as numpy
    history_path = os.path.join(PLOT_DIR, f'{EXPERIMENT_NAME}_history.npz')
    np.savez(history_path, **history)
    print(f"✓ Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
