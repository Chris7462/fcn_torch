"""
FCN Model with Registry System
Simplified FCN-32s style (TorchVision approach)
"""

import torch
import torch.nn as nn

from .registry import NET
from .backbones import build_backbone
from .decoders import build_decoder


@NET.register_module
class FCNs(nn.Module):
    """
    Fully Convolutional Network for Semantic Segmentation

    Args:
        cfg: Global config object
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Build backbone
        self.backbone = build_backbone(cfg.backbone)

        # Validate backbone has out_channels
        assert hasattr(self.backbone, 'out_channels') and self.backbone.out_channels is not None, \
            "Backbone must define out_channels attribute"

        # Build decoder
        self.decoder = build_decoder(
            cfg.decoder,
            default_args=dict(
                in_channels=self.backbone.out_channels,
                num_classes=cfg.num_classes,
                img_height=cfg.img_height,
                img_width=cfg.img_width
            )
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Segmentation output [B, num_classes, H, W]
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Decode features to segmentation
        output = self.decoder(features['out'])

        return output

    def get_model_info(self):
        """Get model information"""
        info = {
            'model': 'FCNs',
            'num_classes': self.cfg.num_classes,
            'img_size': (self.cfg.img_height, self.cfg.img_width),
        }

        # Add backbone info if available
        if hasattr(self.backbone, 'get_model_info'):
            info['backbone'] = self.backbone.get_model_info()
        else:
            info['backbone'] = {
                'type': self.backbone.__class__.__name__,
                'out_channels': self.backbone.out_channels
            }

        return info
