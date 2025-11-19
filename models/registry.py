"""
Model Registry for FCN
"""

from utils import Registry, build_from_cfg


NET = Registry('net')


def build_net(cfg):
    """
    Build network from config

    Args:
        cfg: Global config object

    Returns:
        Network model instance
    """
    return build_from_cfg(cfg.net, NET, default_args=dict(cfg=cfg))
