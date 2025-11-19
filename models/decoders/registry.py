"""
Decoder Registry for FCN
"""

from utils import Registry, build_from_cfg


DECODERS = Registry('decoders')


def build_decoder(cfg, default_args=None):
    """
    Build decoder from config

    Args:
        cfg: Decoder config dict
        default_args: Default arguments to pass to decoder

    Returns:
        Decoder model instance
    """
    return build_from_cfg(cfg, DECODERS, default_args=default_args)
