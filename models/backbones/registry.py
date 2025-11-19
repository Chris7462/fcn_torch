"""
Backbone Registry for FCN
"""

from utils import Registry, build_from_cfg


BACKBONES = Registry('backbones')


def build_backbone(cfg):
    """
    Build backbone from config

    Args:
        cfg: Backbone config dict

    Returns:
        Backbone model instance
    """
    return build_from_cfg(cfg, BACKBONES)
