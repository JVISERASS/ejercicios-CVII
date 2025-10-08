"""
Bed of Nails - Upsampling con vecino más cercano insertando ceros.
"""

import torch


def bed_of_nails(x: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Bed of nails upsampling (nearest neighbor with zeros).
    
    Args:
        x: Tensor [N, C, H, W]
        stride: Upsampling factor
        
    Returns:
        Upsampled tensor with zeros between original values
    """
    N, C, H, W = x.shape
    output = torch.zeros(N, C, H * stride, W * stride, device=x.device, dtype=x.dtype)
    output[:, :, ::stride, ::stride] = x
    return output