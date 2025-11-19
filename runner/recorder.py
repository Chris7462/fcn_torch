"""
Recorder for Logging and Metrics Tracking
"""

import os
import datetime
from .logger import get_logger


class Recorder:
    """
    Records training progress including losses, metrics, and learning rates

    Args:
        cfg: Global config object
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = self.get_work_dir()
        cfg.work_dir = self.work_dir
        self.log_path = os.path.join(self.work_dir, 'log.txt')

        self.logger = get_logger('fcn', self.log_path)
        self.logger.info('Config: \n' + cfg.text)

        # Training state
        self.epoch = 0
        self.step = 0
        self.lr = 0.0

        # Metrics tracking
        self.train_loss = []
        self.train_pixel_acc = []
        self.val_loss = []
        self.val_miou = []
        self.val_pixel_acc = []
        self.best_miou = 0.0

    def get_work_dir(self):
        """
        Create work directory with timestamp
        Format: work_dirs/{dataset_name}/{timestamp}
        """
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = self.cfg.dataset.train.type  # e.g., 'CamVid'
        work_dir = os.path.join(self.cfg.work_dirs, dataset_name, now)

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        return work_dir

    def update_metrics(self, train_loss=None, train_pixel_acc=None,
                      val_loss=None, val_miou=None, val_pixel_acc=None):
        """
        Update metrics for current epoch

        Args:
            train_loss: Training loss
            train_pixel_acc: Training pixel accuracy
            val_loss: Validation loss
            val_miou: Validation mean IoU
            val_pixel_acc: Validation pixel accuracy
        """
        if train_loss is not None:
            self.train_loss.append(train_loss)
        if train_pixel_acc is not None:
            self.train_pixel_acc.append(train_pixel_acc)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_miou is not None:
            self.val_miou.append(val_miou)
        if val_pixel_acc is not None:
            self.val_pixel_acc.append(val_pixel_acc)

    def record_epoch(self, epoch, lr, train_loss, train_pixel_acc,
                    val_loss, val_miou, val_pixel_acc, epoch_time):
        """
        Log epoch summary

        Args:
            epoch: Current epoch number
            lr: Current learning rate
            train_loss: Training loss
            train_pixel_acc: Training pixel accuracy
            val_loss: Validation loss
            val_miou: Validation mean IoU
            val_pixel_acc: Validation pixel accuracy
            epoch_time: Time taken for epoch (seconds)
        """
        self.epoch = epoch
        self.lr = lr

        log_str = (f"\nEpoch {epoch+1}/{self.cfg.epochs} Summary:\n"
                  f"  Time: {epoch_time:.2f}s\n"
                  f"  LR: {lr:.6f}\n"
                  f"  Train Loss: {train_loss:.4f}\n"
                  f"  Train Pixel Acc: {train_pixel_acc:.4f}\n"
                  f"  Val Loss: {val_loss:.4f}\n"
                  f"  Val mIoU: {val_miou:.4f}\n"
                  f"  Val Pixel Acc: {val_pixel_acc:.4f}")

        self.logger.info(log_str)

    def state_dict(self):
        """Return recorder state for checkpoint"""
        return {
            'epoch': self.epoch,
            'step': self.step,
            'best_miou': self.best_miou,
            'train_loss': self.train_loss,
            'train_pixel_acc': self.train_pixel_acc,
            'val_loss': self.val_loss,
            'val_miou': self.val_miou,
            'val_pixel_acc': self.val_pixel_acc,
        }

    def load_state_dict(self, state_dict):
        """Load recorder state from checkpoint"""
        self.epoch = state_dict['epoch']
        self.step = state_dict['step']
        self.best_miou = state_dict['best_miou']
        self.train_loss = state_dict['train_loss']
        self.train_pixel_acc = state_dict['train_pixel_acc']
        self.val_loss = state_dict['val_loss']
        self.val_miou = state_dict['val_miou']
        self.val_pixel_acc = state_dict['val_pixel_acc']


def build_recorder(cfg):
    """Build recorder from config"""
    return Recorder(cfg)
