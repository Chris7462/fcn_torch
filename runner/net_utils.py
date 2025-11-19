"""
Network Utilities for Model Saving and Loading
"""

import torch
import os


def save_model(net, optim, scheduler, recorder, is_best=False):
    """
    Save model checkpoint

    Args:
        net: Network model
        optim: Optimizer
        scheduler: Learning rate scheduler
        recorder: Recorder instance
        is_best: Whether this is the best model
    """
    model_dir = os.path.join(recorder.work_dir, 'ckpt')
    os.makedirs(model_dir, exist_ok=True)

    epoch = recorder.epoch
    ckpt_name = 'best' if is_best else f'epoch_{epoch}'

    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, f'{ckpt_name}.pth'))

    # Always save 'last.pth' as well
    if not is_best:
        torch.save({
            'net': net.state_dict(),
            'optim': optim.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'recorder': recorder.state_dict(),
            'epoch': epoch
        }, os.path.join(model_dir, 'last.pth'))


def load_network(net, model_dir, logger=None):
    """
    Load network weights from checkpoint

    Args:
        net: Network model
        model_dir: Path to checkpoint file
        logger: Logger instance for logging
    """
    if logger:
        logger.info(f'Loading checkpoint from: {model_dir}')

    pretrained_model = torch.load(model_dir, map_location='cpu')
    net.load_state_dict(pretrained_model['net'], strict=True)

    if logger:
        logger.info('Model weights loaded successfully')
