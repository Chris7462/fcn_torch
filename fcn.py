# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


class FCNs(nn.Module):
    """
    FCN-8s with skip connections from pool3, pool4, and pool5
    Following the original FCN paper
    For VGG16 backbone
    """
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.bn1(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.bn2(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.bn3(self.deconv3(score)))  # size=(N, 256, x.H/4, x.W/4)
        score = self.relu(self.bn4(self.deconv4(score)))  # size=(N, 128, x.H/2, x.W/2)
        score = self.relu(self.bn5(self.deconv5(score)))  # size=(N, 64, x.H/1, x.W/1)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCNsResNet(nn.Module):
    """
    FCN with ResNet101 backbone
    Uses 4 skip connections from layer1, layer2, layer3, layer4
    """
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)

        # Upsampling path
        # layer4 (2048) -> layer3 (1024)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(1024)

        # layer3 (1024) -> layer2 (512)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(512)

        # layer2 (512) -> layer1 (256)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(256)

        # layer1 (256) -> H/2 (128)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(128)

        # H/2 (128) -> H (64)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(64)

        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x4 = output['x4']  # layer4: (N, 2048, x.H/32, x.W/32)
        x3 = output['x3']  # layer3: (N, 1024, x.H/16, x.W/16)
        x2 = output['x2']  # layer2: (N, 512, x.H/8, x.W/8)
        x1 = output['x1']  # layer1: (N, 256, x.H/4, x.W/4)

        score = self.relu(self.bn1(self.deconv1(x4)))     # size=(N, 1024, x.H/16, x.W/16)
        score = score + x3                                # element-wise add, size=(N, 1024, x.H/16, x.W/16)
        score = self.relu(self.bn2(self.deconv2(score)))  # size=(N, 512, x.H/8, x.W/8)
        score = score + x2                                # element-wise add, size=(N, 512, x.H/8, x.W/8)
        score = self.relu(self.bn3(self.deconv3(score)))  # size=(N, 256, x.H/4, x.W/4)
        score = score + x1                                # element-wise add, size=(N, 256, x.H/4, x.W/4)
        score = self.relu(self.bn4(self.deconv4(score)))  # size=(N, 128, x.H/2, x.W/2)
        score = self.relu(self.bn5(self.deconv5(score)))  # size=(N, 64, x.H/1, x.W/1)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


def create_fcn_model(n_class, backbone='vgg16', pretrained=True, freeze_backbone=False):
    """
    Factory function to create FCN models with different backbones

    Args:
        n_class: Number of output classes
        backbone: Backbone architecture ('vgg16', 'resnet101', 'efficientnet')
        pretrained: Whether to use pretrained weights
        freeze_backbone: If True, freeze backbone weights (only train FCN head)

    Returns:
        FCNs model
    """
    if backbone == 'vgg16':
        # Load standard VGG16
        from torchvision.models import VGG16_Weights
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg16 = models.vgg16(weights=weights)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in vgg16.features.parameters():
                param.requires_grad = False

        # Extract features from pool3, pool4, pool5
        # VGG16 features module uses string indices: '0', '1', '2', ...
        return_layers = {
            '16': 'x3',  # after 3rd maxpool (pool3) - index 16 in features
            '23': 'x4',  # after 4th maxpool (pool4) - index 23 in features
            '30': 'x5',  # after 5th maxpool (pool5) - index 30 in features
        }
        backbone_features = IntermediateLayerGetter(vgg16.features, return_layers=return_layers)

        fcn_model = FCNs(pretrained_net=backbone_features, n_class=n_class)
        return fcn_model

    elif backbone == 'resnet101':
        # Load standard ResNet101
        from torchvision.models import ResNet101_Weights
        weights = ResNet101_Weights.DEFAULT if pretrained else None  # Uses IMAGENET1K_V2
        resnet101 = models.resnet101(weights=weights)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in resnet101.parameters():
                param.requires_grad = False

        # Extract features from layer1, layer2, layer3, layer4
        # These correspond to conv2_x, conv3_x, conv4_x, conv5_x
        return_layers = {
            'layer1': 'x1',  # conv2_x: (N, 256, H/4, W/4)
            'layer2': 'x2',  # conv3_x: (N, 512, H/8, W/8)
            'layer3': 'x3',  # conv4_x: (N, 1024, H/16, W/16)
            'layer4': 'x4',  # conv5_x: (N, 2048, H/32, W/32)
        }
        backbone_features = IntermediateLayerGetter(resnet101, return_layers=return_layers)

        fcn_model = FCNsResNet(pretrained_net=backbone_features, n_class=n_class)
        return fcn_model

    elif backbone == 'resnet50':
        # TODO: Implement ResNet50 backbone (similar to ResNet101)
        raise NotImplementedError("ResNet50 backbone not yet implemented")

    elif backbone == 'efficientnet':
        # TODO: Implement EfficientNet backbone
        raise NotImplementedError("EfficientNet backbone not yet implemented")

    else:
        raise ValueError(f"Unknown backbone: {backbone}")
