"""
Classic CNN architectures.

This module contains implementations of important CNN architectures:
- AlexNet: First successful deep CNN  
- VGG: Deep networks with small filters
- ResNet: Residual networks with skip connections
- U-Net: Encoder-decoder for segmentation
"""

from .alexnet import AlexNet, AlexNetComponents
from .vgg import VGG, vgg11, vgg16, vgg19, VGGBlock
from .resnet import ResNet, BasicBlock, Bottleneck, resnet18, resnet34, resnet50, resnet101
from .unet import UNet, unet_small, unet_standard, DoubleConv, Down, Up, OutConv

# Create aliases for backwards compatibility with tests
ResidualBlock = BasicBlock
ResNet18 = resnet18
VGG16 = vgg16

# Simple models for testing
import torch
from torch import nn

class SimpleConvNet(nn.Module):
    """Simple convolutional network for testing purposes."""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AttentionBlock(nn.Module):
    """Simple attention block for testing."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights

class TransformerEncoder(nn.Module):
    """Simple transformer encoder block for testing."""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feedforward
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        
        return x

__all__ = [
    # AlexNet
    'AlexNet',
    'AlexNetComponents',
    
    # VGG
    'VGG',
    'VGGBlock', 
    'vgg11',
    'vgg16',
    'vgg19',
    'VGG16',  # alias
    
    # ResNet
    'ResNet',
    'BasicBlock',
    'Bottleneck',
    'resnet18',
    'resnet34', 
    'resnet50',
    'resnet101',
    'ResidualBlock',  # alias for BasicBlock
    'ResNet18',       # alias for resnet18
    
    # U-Net
    'UNet',
    'unet_small',
    'unet_standard',
    'DoubleConv',
    'Down',
    'Up',
    'OutConv',
    
    # Simple models for testing
    'SimpleConvNet',
    'AttentionBlock',
    'TransformerEncoder',
]