"""
Style Loss - Pérdida de estilo basada en matrices de Gram.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLoss(nn.Module):
    """
    Style loss for style transfer.
    Measures the MSE between the Gram matrices of the target and input feature maps.
    """
    
    def __init__(self, target_feature: torch.Tensor) -> None:
        super().__init__()
        self.target_gram = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix for style representation.
        
        Args:
            input: Feature maps [batch, channels, height, width]
            
        Returns:
            Gram matrix [batch, channels, channels]
        """
        b, c, h, w = input.size()
        
        # Reshape feature maps to [batch, channels, height*width]
        features = input.view(b, c, h * w)
        
        # Compute Gram matrix: G = F * F^T
        G = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by number of elements
        return G / (c * h * w)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute style loss between input and target Gram matrices.
        
        Args:
            input: Input feature maps from generated image
            
        Returns:
            Style loss (MSE between Gram matrices)
        """
        G = self.gram_matrix(input)
        return F.mse_loss(G, self.target_gram)