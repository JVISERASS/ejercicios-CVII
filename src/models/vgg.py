"""
VGG - Arquitectura VGG con bloques convolucionales uniformes.
"""

import torch
from torch import nn


class VGGBlock(nn.Module):
    """Bloque básico de VGG: Conv -> ReLU -> Conv -> ReLU -> MaxPool."""
    
    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        super().__init__()
        
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, 
                         out_channels, 
                         kernel_size=3, 
                         padding=1),
                nn.ReLU(inplace=True)
            ])
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG(nn.Module):
    """
    Implementación de VGG.
    Configuraciones:
    - VGG11: [64, 128, 256, 256, 512, 512, 512, 512] con [1, 1, 2, 2, 2] convs
    - VGG16: [64, 128, 256, 256, 512, 512, 512, 512] con [2, 2, 3, 3, 3] convs
    """
    
    def __init__(self, config: list, num_classes: int = 1000) -> None:
        super().__init__()
        
        # Build feature extractor
        self.features = self._make_features(config)
        
        # Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def _make_features(self, config: list) -> nn.Sequential:
        """Construye las capas de características según la configuración."""
        layers = []
        in_channels = 3
        
        for cfg in config:
            if cfg == 'M':  # MaxPool
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:  # Conv layer
                layers.extend([
                    nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ])
                in_channels = cfg
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Configuraciones predefinidas
VGG_CONFIGS = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg11(num_classes: int = 1000) -> VGG:
    """VGG-11 model."""
    return VGG(VGG_CONFIGS['vgg11'], num_classes)


def vgg16(num_classes: int = 1000) -> VGG:
    """VGG-16 model."""
    return VGG(VGG_CONFIGS['vgg16'], num_classes)


def vgg19(num_classes: int = 1000) -> VGG:
    """VGG-19 model."""
    return VGG(VGG_CONFIGS['vgg19'], num_classes)