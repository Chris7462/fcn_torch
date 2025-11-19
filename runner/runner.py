"""
Main Runner Class for Training, Validation, and Testing
Based on RESA's runner.py (simplified for FCN)
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models import build_net
from datasets import build_dataloader
from utils.metrics import batch_iou, batch_pixel_acc
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .recorder import build_recorder
from .net_utils import save_model, load_network


class Runner(object):
    """
    Runner for training, validation, and testing

    Args:
        cfg: Global config object
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build recorder (logger + metrics tracking)
        self.recorder = build_recorder(self.cfg)
        self.recorder.logger.info(f'Using device: {self.device}')
        if torch.cuda.is_available():
            self.recorder.logger.info(f'GPU: {torch.cuda.get_device_name(0)}')

        # Build model
        self.net = build_net(self.cfg)
        self.net = self.net.to(self.device)
        self.recorder.logger.info('Network: \n' + str(self.net))

        # Print model info
        if hasattr(self.net, 'get_model_info'):
            model_info = self.net.get_model_info()
            self.recorder.logger.info(f'Model info: {model_info}')

        # Resume from checkpoint if specified
        self.resume()

        # Build optimizer and scheduler
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)

        self.recorder.logger.info(f'Optimizer: {self.cfg.optimizer.type}')
        self.recorder.logger.info(f'  Learning rate: {self.cfg.optimizer.lr}')
        self.recorder.logger.info(f'  Weight decay: {self.cfg.optimizer.weight_decay}')
        if hasattr(self.cfg.optimizer, 'momentum'):
            self.recorder.logger.info(f'  Momentum: {self.cfg.optimizer.momentum}')

        if self.scheduler:
            self.recorder.logger.info(f'Scheduler: {self.cfg.scheduler.type}')

    def resume(self):
        """Resume from checkpoint"""
        if not self.cfg.load_from:
            return

        self.recorder.logger.info(f'Loading checkpoint from: {self.cfg.load_from}')
        checkpoint = torch.load(self.cfg.load_from, map_location=self.device, weights_only=False)

        # Load model weights
        self.net.load_state_dict(checkpoint['net'])
        self.recorder.logger.info('Model weights loaded')

        # Load recorder state if resuming training (not just for testing)
        if 'recorder' in checkpoint and not self.cfg.test:
            self.recorder.load_state_dict(checkpoint['recorder'])
            self.recorder.logger.info(f'Resuming from epoch {self.recorder.epoch}')
            self.recorder.logger.info(f'Best mIoU so far: {self.recorder.best_miou:.4f}')

    def train_epoch(self, epoch, train_loader):
        """Train for one epoch"""
        self.net.train()
        running_loss = 0.0
        all_pixel_accs = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.net(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            self.recorder.step += 1

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

    def validate(self, val_loader):
        """Validate the model"""
        self.net.eval()
        running_loss = 0.0
        all_ious = []
        all_pixel_accs = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation")
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Forward pass
                outputs = self.net(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()

                # Get predictions
                preds = outputs.argmax(dim=1).cpu().numpy()
                targets = masks.cpu().numpy()

                # Calculate metrics
                batch_ious = batch_iou(preds, targets, self.cfg.num_classes)
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

    def test(self, test_loader):
        """
        Test the model and save predictions

        Args:
            test_loader: Test data loader
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        self.net.eval()

        # Create output directories
        output_dir = self.recorder.work_dir
        pred_dir = os.path.join(output_dir, 'predictions')
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        # Load dataset info for color mapping
        import json
        with open(self.cfg.dataset.test.dataset_info_path, 'r') as f:
            dataset_info = json.load(f)

        # Build color mapping (class_idx -> RGB)
        color_mapping = {}
        for color_str, class_idx in dataset_info['color_to_class'].items():
            r, g, b = eval(color_str)
            color_mapping[class_idx] = (r, g, b)

        mean = np.array(dataset_info['mean']).reshape(1, 1, 3)
        std = np.array(dataset_info['std']).reshape(1, 1, 3)

        self.recorder.logger.info("Running inference on test set...")

        running_loss = 0.0
        all_ious = []
        all_pixel_accs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                filenames = batch['filename']

                # Forward pass
                outputs = self.net(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()

                # Get predictions
                preds = outputs.argmax(dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                images_np = images.cpu().numpy()

                # Calculate metrics
                batch_ious = batch_iou(preds, targets, self.cfg.num_classes)
                batch_pix_acc = batch_pixel_acc(preds, targets)

                all_ious.append(batch_ious)
                all_pixel_accs.append(batch_pix_acc)

                # Save predictions for each image in batch
                for i in range(len(filenames)):
                    filename = filenames[i]
                    pred_mask = preds[i]
                    target_mask = targets[i]
                    img = images_np[i]

                    # Convert prediction to RGB
                    pred_rgb = self.index_to_rgb(pred_mask, color_mapping, self.cfg.ignore_label)

                    # Save color-coded prediction
                    pred_filename = filename[:-4] + '_pred.png'
                    pred_path = os.path.join(pred_dir, pred_filename)
                    Image.fromarray(pred_rgb).save(pred_path)

                    # Create visualization
                    img = img.transpose(1, 2, 0)
                    img = (img * std) + mean
                    img = (img * 255).astype(np.uint8)
                    img = np.clip(img, 0, 255)

                    target_rgb = self.index_to_rgb(target_mask, color_mapping, self.cfg.ignore_label)

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

                    vis_filename = filename[:-4] + '_comparison.png'
                    vis_path = os.path.join(vis_dir, vis_filename)
                    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                    plt.close()

        # Calculate final metrics
        avg_loss = running_loss / len(test_loader)
        all_ious = np.array(all_ious)
        mean_ious = np.nanmean(all_ious, axis=0)
        mean_iou = np.nanmean(mean_ious)
        mean_pixel_acc = np.mean(all_pixel_accs)

        # Print results
        self.recorder.logger.info("\n" + "="*80)
        self.recorder.logger.info("Test Results")
        self.recorder.logger.info("="*80)
        self.recorder.logger.info(f"Test Loss:       {avg_loss:.4f}")
        self.recorder.logger.info(f"Test mIoU:       {mean_iou:.4f}")
        self.recorder.logger.info(f"Test Pixel Acc:  {mean_pixel_acc:.4f}")
        self.recorder.logger.info("\nPer-class IoU:")
        for idx, (name, iou) in enumerate(zip(dataset_info['class_names'], mean_ious)):
            self.recorder.logger.info(f"  {idx:2d}. {name:15s}: {iou:.4f}")
        self.recorder.logger.info("="*80)

        self.recorder.logger.info(f"\n✓ Predictions saved to: {pred_dir}")
        self.recorder.logger.info(f"✓ Visualizations saved to: {vis_dir}")

    def index_to_rgb(self, mask_indexed, color_mapping, ignore_index=255):
        """Convert class indices to RGB color-coded image"""
        h, w = mask_indexed.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, color in color_mapping.items():
            rgb_mask[mask_indexed == class_idx] = color

        # Void pixels remain black
        rgb_mask[mask_indexed == ignore_index] = (0, 0, 0)

        return rgb_mask

    def train(self):
        """Main training loop"""
        self.recorder.logger.info('Starting training...')

        # Build dataloaders
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)
        val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)

        self.recorder.logger.info(f'Train: {len(train_loader.dataset)} images')
        self.recorder.logger.info(f'Val:   {len(val_loader.dataset)} images')

        # Setup loss function with class weights
        import json
        with open(self.cfg.dataset.train.dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        class_weights = torch.tensor(dataset_info['class_weights'], dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.cfg.ignore_label)

        self.recorder.logger.info(f'Number of classes: {self.cfg.num_classes}')
        self.recorder.logger.info(f'Ignore index: {self.cfg.ignore_label}')

        # Create plots directory
        import matplotlib
        matplotlib.use('Agg')
        plot_dir = os.path.join(self.recorder.work_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        self.recorder.logger.info("\n" + "="*80)
        self.recorder.logger.info("Starting Training")
        self.recorder.logger.info("="*80)

        start_epoch = self.recorder.epoch

        for epoch in range(start_epoch, self.cfg.epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_pixel_acc = self.train_epoch(epoch, train_loader)

            # Validate
            val_loss, val_miou, val_pixel_acc, class_ious = self.validate(val_loader)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update metrics
            self.recorder.update_metrics(
                train_loss=train_loss,
                train_pixel_acc=train_pixel_acc,
                val_loss=val_loss,
                val_miou=val_miou,
                val_pixel_acc=val_pixel_acc
            )

            epoch_time = time.time() - epoch_start

            # Log epoch summary
            self.recorder.record_epoch(
                epoch, current_lr, train_loss, train_pixel_acc,
                val_loss, val_miou, val_pixel_acc, epoch_time
            )

            # Check if this is the best model
            is_best = val_miou > self.recorder.best_miou
            if is_best:
                self.recorder.best_miou = val_miou
                self.recorder.logger.info(f"✓ Best model saved! (mIoU: {self.recorder.best_miou:.4f})")

            # Save checkpoint
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1 or is_best:
                save_model(self.net, self.optimizer, self.scheduler, self.recorder, is_best)

            # Plot training history
            if (epoch + 1) % self.cfg.eval_ep == 0:
                self.plot_training_history(plot_dir)

            self.recorder.logger.info("-" * 80)

        self.recorder.logger.info("\n" + "="*80)
        self.recorder.logger.info("Training Complete!")
        self.recorder.logger.info(f"Best Validation mIoU: {self.recorder.best_miou:.4f}")
        self.recorder.logger.info("="*80)

    def plot_training_history(self, save_dir):
        """Plot training history"""
        import matplotlib.pyplot as plt

        history = {
            'train_loss': self.recorder.train_loss,
            'train_pixel_acc': self.recorder.train_pixel_acc,
            'val_loss': self.recorder.val_loss,
            'val_miou': self.recorder.val_miou,
            'val_pixel_acc': self.recorder.val_pixel_acc
        }

        if len(history['train_loss']) == 0:
            return

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

        save_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
