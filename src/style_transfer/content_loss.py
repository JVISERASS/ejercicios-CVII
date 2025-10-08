"""
Content Loss - Pérdida de contenido para style transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """
    Content loss for style transfer.
    Measures the MSE between the target and input feature maps.
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute content loss between input and target features.
        
        Args:
            input: Input feature maps from generated image
            target: Target feature maps from content image
            
        Returns:
            Content loss (MSE between input and target)
        """
        # Detach target to avoid computing gradients for it
        target = target.detach()
        return F.mse_loss(input, target)