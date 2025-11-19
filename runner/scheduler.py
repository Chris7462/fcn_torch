"""
Learning Rate Scheduler Builder for FCN
"""

import torch


_scheduler_factory = {
    'LambdaLR': torch.optim.lr_scheduler.LambdaLR,
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
    'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
}


def build_scheduler(cfg, optimizer):
    """
    Build learning rate scheduler from config

    Args:
        cfg: Global config object
        optimizer: Optimizer instance

    Returns:
        Scheduler instance
    """
    if not hasattr(cfg, 'scheduler'):
        return None

    scheduler_type = cfg.scheduler.type

    if scheduler_type not in _scheduler_factory:
        raise ValueError(f"Scheduler type '{scheduler_type}' not supported. "
                        f"Available: {list(_scheduler_factory.keys())}")

    # Copy scheduler config and remove 'type' key
    cfg_cp = cfg.scheduler.copy()
    cfg_cp.pop('type')

    # Convert to dict if needed
    if hasattr(cfg_cp, 'to_dict'):
        cfg_cp = cfg_cp.to_dict()

    scheduler = _scheduler_factory[scheduler_type](optimizer, **cfg_cp)

    return scheduler
