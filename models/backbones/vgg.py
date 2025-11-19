"""
VGG16 Backbone for FCN
Uses torchvision's pretrained VGG16
"""

import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

from .registry import BACKBONES


@BACKBONES.register_module
class VGG16(nn.Module):
    """
    VGG16 backbone for FCN

    Args:
        pretrained: Whether to use pretrained weights
        out_channels: Output channels (default: 512 for VGG16)
    """

    def __init__(self, pretrained=True, out_channels=None):
        super().__init__()

        # Load VGG16
        if pretrained:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        else:
            vgg = vgg16(weights=None)

        # Extract feature layers (before classifier)
        self.features = vgg.features

        # Set output channels
        self.out_channels = out_channels if out_channels is not None else 512

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Dict with 'out' key containing features [B, 512, H/32, W/32]
        """
        x = self.features(x)
        return {'out': x}

    def get_model_info(self):
        """Get model information"""
        return {
            'backbone': 'VGG16',
            'out_channels': self.out_channels,
            'pretrained': True
        }
