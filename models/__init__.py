from .registry import NET, build_net
from .fcn import FCNs
from .backbones import BACKBONES, build_backbone, VGG16
from .decoders import DECODERS, build_decoder, FCNHead


__all__ = [
    'NET', 'build_net',
    'FCNs',
    'BACKBONES', 'build_backbone', 'VGG16',
    'DECODERS', 'build_decoder', 'FCNHead'
]
