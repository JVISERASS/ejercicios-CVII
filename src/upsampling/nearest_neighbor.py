"""
Nearest Neighbor Interpolation - Interpolación por vecino más cercano.
"""

import torch


def nearest_neighbor(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Nearest neighbor interpolation.
    
    Args:
        x: Tensor [N, C, H, W]
        size: (H_out, W_out) - target size
        
    Returns:
        Upsampled tensor using nearest neighbor interpolation
    """
    N, C, H, W = x.shape
    h_out, w_out = size
    
    # Map output coordinates directly to input coordinates
    # For each output position, find which input pixel to use
    idx_y = torch.arange(h_out, device=x.device) * H // h_out
    idx_x = torch.arange(w_out, device=x.device) * W // w_out
    
    # Use broadcasting for vectorized indexing
    # idx_y[:, None] creates column vector, idx_x creates row vector
    # Broadcasting handles the rest automatically
    output = x[:, :, idx_y[:, None], idx_x[None, :]]
    
    return output



