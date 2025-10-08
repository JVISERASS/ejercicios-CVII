"""
Style transfer loss functions.

This module contains loss functions used in neural style transfer:
- Content loss: Preserves image content
- Style loss: Captures artistic style via Gram matrices
- Total variation loss: Encourages spatial smoothness
"""

from .content_loss import ContentLoss
from .style_loss import StyleLoss
from .total_variation_loss import TotalVariationLoss, TotalVariationLossL2

# Additional classes for testing
import torch
from torch import nn
import torch.nn.functional as F

class GramMatrix(nn.Module):
    """Computes Gram matrix for style representation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix.
        
        Args:
            input: Feature tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Gram matrix of shape (batch_size, channels, channels)
        """
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(channels * height * width)

class FeatureExtractor(nn.Module):
    """Feature extractor for perceptual losses."""
    
    def __init__(self, model_name: str = 'vgg19', layers: list = None):
        super().__init__()
        if layers is None:
            layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
        
        self.layers = layers
        # Simplified feature extractor for testing
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # conv1_1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),  # conv2_1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),  # conv3_1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),  # conv4_1
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """Extract features at specified layers."""
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i // 2 < len(self.layers):  # Every 2 layers (conv + relu)
                layer_name = self.layers[i // 2] if i // 2 < len(self.layers) else f'layer_{i//2}'
                features[layer_name] = x
        return features

class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained network features."""
    
    def __init__(self, feature_extractor: nn.Module = None, layers: list = None):
        super().__init__()
        if feature_extractor is None:
            feature_extractor = FeatureExtractor(layers=layers)
        
        self.feature_extractor = feature_extractor
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between input and target.
        
        Args:
            input: Input tensor
            target: Target tensor
            
        Returns:
            Perceptual loss
        """
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        
        loss = 0.0
        for layer in input_features:
            loss += F.mse_loss(input_features[layer], target_features[layer])
        
        return loss

__all__ = [
    'ContentLoss',
    'StyleLoss', 
    'TotalVariationLoss',
    'TotalVariationLossL2',
    'GramMatrix',
    'FeatureExtractor',
    'PerceptualLoss',
]