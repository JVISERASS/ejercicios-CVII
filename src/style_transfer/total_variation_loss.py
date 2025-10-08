"""
Total Variation Loss - Pérdida de variación total para suavizado espacial.
"""

import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss for style transfer.
    Encourages spatial smoothness in the generated image by penalizing 
    high frequency noise and artifacts.
    """
    
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss.
        
        Args:
            input: Input image tensor [batch, channels, height, width]
            
        Returns:
            Total variation loss encouraging spatial smoothness
        """
        batch_size, channels, height, width = input.size()
        
        # Horizontal variation (differences between adjacent pixels horizontally)
        h_tv = torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:]))
        
        # Vertical variation (differences between adjacent pixels vertically)  
        w_tv = torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        
        # Total variation loss
        tv_loss = (h_tv + w_tv) / (batch_size * channels * height * width)
        
        return self.weight * tv_loss


class TotalVariationLossL2(nn.Module):
    """
    L2 version of Total Variation loss (smoother penalty).
    """
    
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 total variation loss.
        
        Args:
            input: Input image tensor [batch, channels, height, width]
            
        Returns:
            L2 total variation loss
        """
        batch_size, channels, height, width = input.size()
        
        # L2 horizontal variation
        h_tv = torch.sum((input[:, :, :, :-1] - input[:, :, :, 1:]) ** 2)
        
        # L2 vertical variation
        w_tv = torch.sum((input[:, :, :-1, :] - input[:, :, 1:, :]) ** 2)
        
        # L2 total variation loss
        tv_loss = (h_tv + w_tv) / (batch_size * channels * height * width)
        
        return self.weight * tv_loss