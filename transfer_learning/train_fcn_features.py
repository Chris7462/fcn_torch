"""
FCN Training Script using Pre-computed Features
Trains only the FCN head with cached backbone features
Much faster than training with full images
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
import argparse

from create_feature_dataloaders import create_feature_dataloaders
from fcn import create_fcn_model
from metrics import batch_iou, batch_pixel_acc


# ================== Configuration ==================
# Training settings
BATCH_SIZE = 32  # Can use larger batch size since we're not running backbone
EPOCHS = 200
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# ReduceLROnPlateau settings
LR_PATIENCE = 20
LR_FACTOR = 0.5
LR_MIN = 1e-6

# Data settings
NUM_WORKERS = 4

# Output directories
MODEL_DIR = './models'
PLOT_DIR = './plots'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, backbone):
    """Train for one epoch using pre-computed features"""
    model.train()
    running_loss = 0.0
    all_pixel_accs = []

    current_lr = optimizer.param_groups[0]['lr']
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train] LR={current_lr:.6f}")

    for batch_idx, batch in enumerate(pbar):
        masks = batch['mask'].to(device)

        # Load features and move to device
        if backbone == 'resnet101':
            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            x3 = batch['x3'].to(device)
            x4 = batch['x4'].to(device)
        else:  # vgg16
            x3 = batch['x3'].to(device)
            x4 = batch['x4'].to(device)
            x5 = batch['x5'].to(device)

        # Forward pass (using pre-computed features)
        optimizer.zero_grad()

        if backbone == 'resnet101':
            outputs = model.forward_features(x1, x2, x3, x4)
        else:  # vgg16
            outputs = model.forward_features(x3, x4, x5)

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


def validate(model, val_loader, criterion, device, n_class, ignore_index, epoch, total_epochs, backbone):
    """Validate the model using pre-computed features"""
    model.eval()
    running_loss = 0.0
    all_ious = []
    all_pixel_accs = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]  ")
        for batch_idx, batch in enumerate(pbar):
            masks = batch['mask'].to(device)

            # Load features and move to device
            if backbone == 'resnet101':
                x1 = batch['x1'].to(device)
                x2 = batch['x2'].to(device)
                x3 = batch['x3'].to(device)
                x4 = batch['x4'].to(device)
            else:  # vgg16
                x3 = batch['x3'].to(device)
                x4 = batch['x4'].to(device)
                x5 = batch['x5'].to(device)

            # Forward pass (using pre-computed features)
            if backbone == 'resnet101':
                outputs = model.forward_features(x1, x2, x3, x4)
            else:  # vgg16
                outputs = model.forward_features(x3, x4, x5)

            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Get predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()

            # Calculate metrics
            batch_ious = batch_iou(preds, targets, n_class)
            batch_pix_acc = batch_pixel_acc(preds, targets)

            all_ious.append(batch_ious)
            all_pixel_accs.append(batch_pix_acc)

    avg_loss = running_loss / len(val_loader)

    # Calculate mean IoU
    all_ious = np.array(all_ious)
    mean_ious = np.nanmean(all_ious, axis=0)
    mean_iou = np.nanmean(mean_ious)

    mean_pixel_acc = np.mean(all_pixel_accs)

    return avg_loss, mean_iou, mean_pixel_acc, mean_ious


def plot_training_history(history, save_dir, experiment_name):
    """Plot training and validation curves"""
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

    # Plot 2: Metrics
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

    save_path = os.path.join(save_dir, f'{experiment_name}_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train FCN model using pre-computed features')
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='Directory containing cached features')
    parser.add_argument('--dataset-info', type=str, required=True,
                        help='Path to dataset_info.json')
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet101'],
                        help='Backbone architecture (default: resnet101)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--lr', type=float, default=LR,
                        help=f'Learning rate (default: {LR})')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--override-lr', type=float, default=None,
                        help='Override learning rate when resuming training')
    args = parser.parse_args()

    # Extract dataset name from feature_dir
    dataset_name = os.path.basename(args.feature_dir).split('_')[0]

    # Experiment name
    EXPERIMENT_NAME = f"FCNs-{args.backbone}_{dataset_name}_features_batch{args.batch_size}_epoch{args.epochs}_SGD_lr{args.lr}_ReduceLROnPlateau"
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Feature directory: {args.feature_dir}")
    print(f"Backbone: {args.backbone}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create feature dataloaders
    print("\nCreating feature dataloaders...")
    dataloaders = create_feature_dataloaders(
            feature_dir=args.feature_dir,
            dataset_info_path=args.dataset_info,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS
            )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    class_weights = torch.tensor(dataloaders['class_weights'], dtype=torch.float32).to(device)
    num_classes = dataloaders['num_classes']
    ignore_index = dataloaders['ignore_index']

    print(f"Number of classes: {num_classes}")
    print(f"Ignore index: {ignore_index}")

    # Create model (we only need the FCN head, not the backbone)
    print(f"\nCreating FCN head for {args.backbone} backbone...")
    model = create_fcn_model(n_class=num_classes, backbone=args.backbone, pretrained=False)

    # Remove backbone since we don't need it
    model.pretrained_net = None
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Use ReduceLROnPlateau scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=LR_MIN
            )

    print(f"\nOptimizer: SGD")
    print(f"  Learning rate: {args.lr}")
    print(f"  Momentum: {MOMENTUM}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  LR Scheduler: ReduceLROnPlateau")
    print(f"    Patience: {LR_PATIENCE} epochs")
    print(f"    Factor: {LR_FACTOR}")
    print(f"    Min LR: {LR_MIN}")

    # Training state
    start_epoch = 0
    best_miou = 0.0
    history = {
            'train_loss': [],
            'train_pixel_acc': [],
            'val_loss': [],
            'val_miou': [],
            'val_pixel_acc': []
            }

    # Load checkpoint if resuming
    if args.resume:
        print(f"\n[INFO] Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        best_miou = checkpoint['best_miou']
        history = checkpoint['history']

        print(f"[INFO] Resuming from epoch {start_epoch}")
        print(f"[INFO] Best mIoU so far: {best_miou:.4f}")

        if args.override_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.override_lr
            print(f'[INFO] Overriding learning rate to {args.override_lr}')

    print("\n" + "="*80)
    print("Starting Training (Feature Extraction Mode)")
    print("="*80)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_pixel_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args.epochs, args.backbone
                )

        # Validate
        val_loss, val_miou, val_pixel_acc, class_ious = validate(
                model, val_loader, criterion, device, num_classes, ignore_index, epoch, args.epochs, args.backbone
                )

        # Update learning rate based on validation mIoU
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_miou)
        current_lr = optimizer.param_groups[0]['lr']

        # Print if LR changed
        if current_lr != old_lr:
            print(f"\n{'='*80}")
            print(f"⚠️  Learning Rate Reduced: {old_lr:.6f} → {current_lr:.6f}")
            print(f"{'='*80}")

        # Update history
        history['train_loss'].append(train_loss)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        history['val_pixel_acc'].append(val_pixel_acc)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Pixel Acc: {train_pixel_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val mIoU: {val_miou:.4f}")
        print(f"  Val Pixel Acc: {val_pixel_acc:.4f}")

        # Check if this is the best model
        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou

        # Check if periodic checkpoint (every 10 epochs)
        is_periodic = (epoch + 1) % 10 == 0

        # Prepare checkpoint
        checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'history': history
                }

        # Always save last checkpoint
        last_model_path = os.path.join(MODEL_DIR, f'{EXPERIMENT_NAME}_last.pth')
        torch.save(checkpoint, last_model_path)

        # Save best checkpoint
        if is_best:
            best_model_path = os.path.join(MODEL_DIR, f'{EXPERIMENT_NAME}_best.pth')
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ Best model saved! (mIoU: {best_miou:.4f})")

        # Save periodic checkpoint
        if is_periodic:
            periodic_path = os.path.join(MODEL_DIR, f'{EXPERIMENT_NAME}_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
            print(f"  ✓ Periodic checkpoint saved (epoch {epoch+1})")

        # Plot training history
        plot_training_history(history, PLOT_DIR, EXPERIMENT_NAME)

        print("-" * 80)

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation mIoU: {best_miou:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
