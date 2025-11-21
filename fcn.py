from __future__ import print_function

import torch.nn as nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


class FCNs(nn.Module):
    """
    FCN-8s with skip connections from pool3, pool4, and pool5
    Following the original FCN paper
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


def create_fcn_model(n_class, backbone='vgg16', pretrained=True):
    """
    Factory function to create FCN models with different backbones

    Args:
        n_class: Number of output classes
        backbone: Backbone architecture ('vgg16', 'resnet50', 'efficientnet')
        pretrained: Whether to use pretrained weights

    Returns:
        FCNs model
    """
    if backbone == 'vgg16':
        # Load standard VGG16
        from torchvision.models import VGG16_Weights
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg16 = models.vgg16(weights=weights)

        # Extract features from pool3, pool4, pool5
        return_layers = {
            '16': 'x3',  # after 3rd maxpool (pool3)
            '23': 'x4',  # after 4th maxpool (pool4)
            '30': 'x5',  # after 5th maxpool (pool5)
        }
        backbone = IntermediateLayerGetter(vgg16.features, return_layers=return_layers)

        fcn_model = FCNs(pretrained_net=backbone, n_class=n_class)
        return fcn_model

    elif backbone == 'resnet50':
        # TODO: Implement ResNet50 backbone
        raise NotImplementedError("ResNet50 backbone not yet implemented")

    elif backbone == 'efficientnet':
        # TODO: Implement EfficientNet backbone
        raise NotImplementedError("EfficientNet backbone not yet implemented")

    else:
        raise ValueError(f"Unknown backbone: {backbone}")
