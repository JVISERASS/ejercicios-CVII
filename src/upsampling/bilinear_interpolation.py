"""
Bilinear Interpolation - Interpolación bilineal para upsampling.
"""

import torch


def bilinear_interpolation(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Bilinear interpolation.
    
    Args:
        x: Tensor [N, C, H, W]
        size: (H_out, W_out) - target size
        
    Returns:
        Upsampled tensor using bilinear interpolation
    """
    N, C, H, W = x.shape
    H_out, W_out = size
    
    # Create coordinate grids
    grid_y = torch.linspace(0, H - 1, H_out, device=x.device)
    grid_x = torch.linspace(0, W - 1, W_out, device=x.device)
    
    # Floor and ceiling coordinates
    y0 = torch.floor(grid_y).long()
    x0 = torch.floor(grid_x).long()
    y1 = torch.clamp(y0 + 1, max=H - 1)
    x1 = torch.clamp(x0 + 1, max=W - 1)
    
    # Interpolation weights
    wy = (grid_y - y0.float()).view(-1, 1)
    wx = (grid_x - x0.float()).view(1, -1)

    # Get corner values
    v00 = x[:, :, y0.unsqueeze(1), x0]
    v01 = x[:, :, y0.unsqueeze(1), x1]
    v10 = x[:, :, y1.unsqueeze(1), x0]
    v11 = x[:, :, y1.unsqueeze(1), x1]

    # Bilinear interpolation
    out = (
        (1 - wy) * (1 - wx) * v00 +
        (1 - wy) * wx * v01 +
        wy * (1 - wx) * v10 +
        wy * wx * v11
    )
    
    return out