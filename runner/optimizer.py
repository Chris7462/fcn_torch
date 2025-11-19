"""
Optimizer Builder for FCN
"""

import torch


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def build_optimizer(cfg, net):
    """
    Build optimizer from config

    Args:
        cfg: Global config object
        net: Network model

    Returns:
        Optimizer instance
    """
    params = []
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer_type = cfg.optimizer.type.lower()

    if optimizer_type not in _optimizer_factory:
        raise ValueError(f"Optimizer type '{cfg.optimizer.type}' not supported. "
                        f"Available: {list(_optimizer_factory.keys())}")

    if optimizer_type == 'adam':
        optimizer = _optimizer_factory[optimizer_type](params, lr, weight_decay=weight_decay)
    else:  # sgd
        optimizer = _optimizer_factory[optimizer_type](
            params, lr, weight_decay=weight_decay, momentum=cfg.optimizer.momentum)

    return optimizer
