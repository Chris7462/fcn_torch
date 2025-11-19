"""
FCN Decoder Head (TorchVision Style)
Simplified decoder without skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import DECODERS


@DECODERS.register_module
class FCNHead(nn.Module):
    """
    Simple FCN decoder head (FCN-32s style)
    Based on TorchVision's implementation

    Args:
        in_channels: Number of input channels from backbone
        num_classes: Number of output classes
        img_height: Target output height
        img_width: Target output width
    """

    def __init__(self, in_channels, num_classes, img_height, img_width):
        super().__init__()

        inter_channels = in_channels // 4

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        )

        self.img_height = img_height
        self.img_width = img_width

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input features [B, in_channels, H, W]

        Returns:
            Output segmentation [B, num_classes, img_height, img_width]
        """
        x = self.conv(x)
        x = F.interpolate(x, size=(self.img_height, self.img_width),
                         mode='bilinear', align_corners=False)
        return x
