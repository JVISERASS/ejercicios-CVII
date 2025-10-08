"""
AlexNet - Componentes y conceptos clave de la arquitectura AlexNet.
"""

import torch
from torch import nn
import torch.nn.functional as F


class AlexNetComponents:
    """Implementación de componentes específicos de AlexNet."""
    
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        """
        ReLU activation function - innovación clave de AlexNet.
        """
        return F.relu(x)
    
    @staticmethod
    def dropout(x: torch.Tensor, rate: float = 0.5, training: bool = True) -> torch.Tensor:
        """
        Dropout regularization - primera vez usado en CNN grandes.
        """
        return F.dropout(x, p=rate, training=training)
    
    @staticmethod
    def data_augmentation(x: torch.Tensor) -> torch.Tensor:
        """
        Simulación de data augmentation usado en AlexNet.
        """
        # Ejemplo básico - en práctica se haría con transforms
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[-1])
        
        # Random crop (simulado con padding)
        if torch.rand(1) > 0.5:
            pad = torch.randint(0, 10, (1,)).item()
            x = F.pad(x, (pad, pad, pad, pad))
            x = x[:, :, pad:-pad, pad:-pad] if pad > 0 else x
        
        return x


class AlexNet(nn.Module):
    """
    Implementación simplificada de AlexNet.
    """
    
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        
        # Features (convolutional layers)
        self.features = nn.Sequential(
            # Conv1: 96 filters, 11x11, stride 4
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 256 filters, 5x5
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 384 filters, 3x3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 384 filters, 3x3
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 256 filters, 3x3
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x