"""
FCN Training Script using Registry System
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import argparse
from tqdm import tqdm

from utils.config import Config
from datasets import build_dataloader
from fcn import create_fcn_model
from metrics import batch_iou, batch_pixel_acc


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_pixel_accs = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
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


def validate(model, val_loader, criterion, device, n_class, ignore_index, epoch, total_epochs):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_ious = []
    all_pixel_accs = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]  ")
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
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    Load checkpoint to resume training

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to map checkpoint to

    Returns:
        start_epoch: Epoch to resume from
        best_miou: Best mIoU achieved so far
        history: Training history dict
    """
    print(f"\n[INFO] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    print('[INFO] Model state loaded')

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('[INFO] Optimizer state loaded')

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('[INFO] Scheduler state loaded')

    start_epoch = checkpoint['epoch']
    best_miou = checkpoint['best_miou']
    history = checkpoint['history']

    print(f"[INFO] Resuming from epoch {start_epoch}")
    print(f"[INFO] Best mIoU so far: {best_miou:.4f}")

    return start_epoch, best_miou, history


def main(args=None):
    # Load config
    cfg = Config.fromfile(args.config)
    print(f"Loaded config from: {args.config}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build dataloaders using registry
    print("\nBuilding dataloaders...")
    train_loader = build_dataloader(cfg.dataset.train, cfg, is_train=True)
    val_loader = build_dataloader(cfg.dataset.val, cfg, is_train=False)

    print(f"Train: {len(train_loader.dataset)} images")
    print(f"Val:   {len(val_loader.dataset)} images")

    # Load class weights from dataset_info.json
    with open(cfg.dataset.train.dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    class_weights = torch.tensor(dataset_info['class_weights'], dtype=torch.float32).to(device)

    print(f"Number of classes: {cfg.num_classes}")
    print(f"Ignore index: {cfg.ignore_label}")

    # Create model
    print(f"\nCreating model...")
    model = create_fcn_model(n_class=cfg.num_classes, backbone='vgg16', pretrained=cfg.backbone.pretrained)
    model = model.to(device)

    # Setup loss and optimizer (following original FCN paper)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=cfg.ignore_label)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay
    )
    scheduler_fn = lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler.step_size,
        gamma=cfg.scheduler.gamma
    )

    print(f"\nOptimizer: SGD")
    print(f"  Learning rate: {cfg.optimizer.lr}")
    print(f"  Momentum: {cfg.optimizer.momentum}")
    print(f"  Weight decay: {cfg.optimizer.weight_decay}")
    print(f"  LR Scheduler: StepLR (step_size={cfg.scheduler.step_size}, gamma={cfg.scheduler.gamma})")

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
        start_epoch, best_miou, history = load_checkpoint(
            args.resume, model, optimizer, scheduler_fn, device
        )

        # Override learning rate if specified
        if args.override_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.override_lr
            print(f'[INFO] Overriding learning rate to {args.override_lr}')

    # Experiment name for plots
    experiment_name = f"FCN_{os.path.basename(args.config).replace('.py', '')}"

    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_pixel_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg.epochs
        )

        # Validate
        val_loss, val_miou, val_pixel_acc, class_ious = validate(
            model, val_loader, criterion, device, cfg.num_classes, cfg.ignore_label, epoch, cfg.epochs
        )

        # Update learning rate
        scheduler_fn.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        history['val_pixel_acc'].append(val_pixel_acc)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{cfg.epochs} Summary:")
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

        # Check if periodic checkpoint
        is_periodic = (epoch + 1) % cfg.save_ep == 0

        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch + 1,  # Next epoch to resume from
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_fn.state_dict() if scheduler_fn else None,
            'best_miou': best_miou,
            'history': history
        }

        # Always save last checkpoint
        last_model_path = os.path.join(cfg.model_dir, f'{experiment_name}_last.pth')
        torch.save(checkpoint, last_model_path)

        # Save best checkpoint
        if is_best:
            best_model_path = os.path.join(cfg.model_dir, f'{experiment_name}_best.pth')
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ Best model saved! (mIoU: {best_miou:.4f})")

        # Save periodic checkpoint (every 10 epochs)
        if is_periodic:
            periodic_path = os.path.join(cfg.model_dir, f'{experiment_name}_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
            print(f"  ✓ Periodic checkpoint saved (epoch {epoch+1})")

        # Plot training history after each epoch
        plot_training_history(history, cfg.plot_dir, experiment_name)

        print("-" * 80)

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation mIoU: {best_miou:.4f}")
    print("="*80)

    # Save final model
    final_model_path = os.path.join(cfg.model_dir, f'{experiment_name}_final.pth')
    torch.save({
        'epoch': cfg.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler_fn.state_dict(),
        'best_miou': best_miou,
        'history': history
    }, final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train FCN model for semantic segmentation')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to resume training from')
    parser.add_argument('--override-lr', type=float, default=None,
                    help='Override learning rate when resuming training')
    args = parser.parse_args()

    main(args)
